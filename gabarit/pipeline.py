"""Frame-by-frame processing chain shared by the ROS 2 node, the web app and
the evaluation scripts:

PointCloud2 -> cloud -> self-calibration + track path -> ego-motion ->
envelope candidates -> clustering / rejection -> temporal confirmation ->
decision (path clear / obstacle ahead, distance).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from gabarit import limit_blas_threads
from gabarit.cloud import Cloud, from_raw, working_set
from gabarit.config import Config
from gabarit.detect import FrameGeometry, detect, track_coordinates
from gabarit.egomotion import EgoMotion, propagate_track
from gabarit.rosbag import RawCloud
from gabarit.track import TrackEstimator, TrackModel
from gabarit.tracker import Tracker

# point classes for visualisation
CLS_OTHER, CLS_GROUND, CLS_ENVELOPE, CLS_CANDIDATE, CLS_OBSTACLE = 0, 1, 2, 3, 4


@dataclass
class FrameResult:
    data: dict[str, Any]
    cloud: Cloud
    lev: np.ndarray
    geo: FrameGeometry
    track: TrackModel
    labels: np.ndarray       # per-point class for visualisation


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        limit_blas_threads()
        self.cfg = cfg or Config()
        self.track = TrackEstimator(self.cfg)
        self.ego = EgoMotion(self.cfg.tracking.max_speed)
        self.tracker = Tracker(self.cfg)
        self.prior: TrackModel | None = None
        self.frame_index = 0
        self.last_stamp: float | None = None
        self.turnout_hist: list = []

    def reset(self) -> None:
        self.track.reset()
        self.ego.reset()
        self.tracker.reset()
        self.prior = None
        self.frame_index = 0
        self.last_stamp = None
        self.turnout_hist = []

    def _blend_path(self, track: TrackModel, prior: TrackModel) -> None:
        """Temporal low-pass of the far path (the near part is fixed by the rails).

        The far path is estimated from sparse wall returns and jitters between
        frames; a standing train must see an identical corridor."""
        moving = abs(self.ego.v) > 1.0
        w_far = 0.5 if moving else 0.15
        w = np.clip((track.s - 40.0) / 40.0, 0.0, 1.0)
        w_new = 1.0 - w * (1.0 - w_far)
        track.c = _blend(track.c, prior.c, w_new)
        track.z = _blend(track.z, prior.z, w_new)
        track.theta = np.gradient(track.c, track.s)
        self.track.prev = track

    def process_raw(self, raw: RawCloud, stamp: float | None = None, **kw) -> FrameResult:
        t0 = time.perf_counter()
        cloud = from_raw(raw, self.cfg.sensor.forward_axis, self.cfg.sensor.min_range)
        cloud = cloud.subset(working_set(cloud, self.cfg.detection.max_range + 5.0))
        decode_ms = (time.perf_counter() - t0) * 1000
        res = self.process_cloud(cloud, stamp if stamp is not None else raw.stamp, **kw)
        res.data["timing_ms"]["decode"] = round(decode_ms, 2)
        res.data["timing_ms"]["total"] = round(res.data["timing_ms"]["total"] + decode_ms, 2)
        return res

    def process_cloud(self, cloud: Cloud, stamp: float, frame_index: int | None = None) -> FrameResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        # prior path: previous frame moved by the predicted displacement
        prior = None
        if self.track.prev is not None and self.last_stamp is not None:
            dt = stamp - self.last_stamp
            dt = dt if 0.02 < dt < 1.0 else 0.1
            prior = propagate_track(self.track.prev, self.ego.v * dt)
        lev, track = self.track.estimate(cloud.xyz, prior)
        if prior is not None:
            self._blend_path(track, prior)
        t1 = time.perf_counter()
        delta, speed = self.ego.update(lev, stamp)
        t2 = time.perf_counter()
        geo = track_coordinates(lev, track, cfg)
        cands = detect(lev, cloud.ring, geo, track, cfg, cloud.split)
        t3 = time.perf_counter()
        tracks = self.tracker.update(cands, delta)
        t4 = time.perf_counter()
        self.last_stamp = stamp
        # turnout ahead: reported when seen in 3 of the last 5 frames
        for h in self.turnout_hist:
            if h is not None:
                h[0] -= delta
        self.turnout_hist.append([self.track.turnout_at] if self.track.turnout_at is not None else None)
        del self.turnout_hist[:-5]
        seen = [h[0] for h in self.turnout_hist if h is not None]
        turnout = float(np.median(seen)) if len(seen) >= 3 else None
        idx = self.frame_index if frame_index is None else frame_index
        self.frame_index = idx + 1

        # confirmed obstacles are held through short dropouts (hysteresis)
        confirmed = sorted((t for t in tracks if t.confirmed), key=lambda t: t.cand.distance)
        pending = [t for t in tracks if not t.confirmed and t.misses == 0]
        obstacle = bool(confirmed)
        nearest = confirmed[0].cand.distance if confirmed else None
        nearest_s = confirmed[0].cand.s_near if confirmed else None
        observed = float(min(track.observed_until, cfg.detection.max_range))
        free = nearest_s if obstacle else observed
        if obstacle:
            status = "OBSTACLE"
        elif pending:
            status = "CHECK"
        elif not track.rail.any() and not track.lat_obs.any():
            status = "DEGRADED"
        else:
            status = "CLEAR"
        v = max(speed, 0.0)
        stop = v * cfg.reaction_time + v * v / (2 * cfg.braking_deceleration)
        ttc = (nearest_s / v) if (obstacle and v > 0.5) else None

        labels = np.zeros(len(cloud), np.uint8)
        ground = geo.in_range & (np.abs(geo.d) < geo.half_width + 0.3) & (geo.h < cfg.envelope.min_height)
        labels[ground] = CLS_GROUND
        labels[geo.corridor] = CLS_ENVELOPE
        for c in cands:
            if c.point_index is not None:
                labels[c.point_index] = CLS_CANDIDATE if c.rejected is None else CLS_ENVELOPE
        for t in confirmed:
            # a track held through a dropout keeps the indices of an older frame
            if t.misses == 0 and t.cand.point_index is not None:
                labels[t.cand.point_index] = CLS_OBSTACLE

        data = {
            "frame": idx,
            "stamp": stamp,
            "status": status,
            "obstacle": obstacle,
            "nearest_distance_m": None if nearest is None else round(float(nearest), 2),
            "nearest_track_distance_m": None if nearest_s is None else round(float(nearest_s), 2),
            "free_distance_m": round(float(free), 1),
            "observed_until_m": round(observed, 1),
            "speed_mps": round(float(speed), 2),
            "speed_kmh": round(float(speed) * 3.6, 1),
            "displacement_m": round(float(delta), 3),
            "ego_quality": round(self.ego.quality, 3),
            "stopping_distance_m": round(float(stop), 1),
            "time_to_obstacle_s": None if ttc is None else round(float(ttc), 1),
            "obstacles": [t.public() for t in confirmed],
            "pending": [t.public() for t in pending],
            "candidates": [c.public() for c in cands[:40]],
            "turnout_ahead_m": None if turnout is None else round(turnout, 1),
            "calibration": track.calibration,
            "envelope": {"width_m": cfg.envelope.width, "height_m": cfg.envelope.height,
                         "min_height_m": cfg.envelope.min_height},
            "points": {"raw": cloud.raw_count, "valid": len(cloud), "envelope": int(geo.corridor.sum())},
            "timing_ms": {
                "track": round((t1 - t0) * 1000, 2),
                "egomotion": round((t2 - t1) * 1000, 2),
                "detect": round((t3 - t2) * 1000, 2),
                "tracking": round((t4 - t3) * 1000, 2),
                "total": round((t4 - t0) * 1000, 2),
            },
        }
        return FrameResult(data, cloud, lev, geo, track, labels)


def _blend(new: np.ndarray, old: np.ndarray, w_new: np.ndarray) -> np.ndarray:
    ok = np.isfinite(old)
    out = new.copy()
    out[ok] = w_new[ok] * new[ok] + (1 - w_new[ok]) * old[ok]
    return out


def stopping_distance(speed: float, cfg: Config) -> float:
    return speed * cfg.reaction_time + speed * speed / (2 * cfg.braking_deceleration)


def kmh(v: float) -> float:
    return v * 3.6 if math.isfinite(v) else 0.0
