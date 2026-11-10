"""Frame-by-frame processing chain shared by the ROS 2 node, the web app and
the evaluation scripts:

PointCloud2 -> cloud -> self-calibration + track path -> ego-motion ->
envelope candidates -> clustering / rejection -> temporal confirmation ->
decision (path clear / obstacle ahead, distance).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from gabarit import limit_blas_threads
from gabarit.cloud import Cloud, from_raw
from gabarit.config import Config
from gabarit.component_model import ComponentModelAssistant
from gabarit.detect import Candidate, FrameGeometry, detect, track_coordinates
from gabarit.egomotion import EgoMotion, propagate_track
from gabarit.rosbag import RawCloud
from gabarit.path_fusion import FarPathFusion
from gabarit.track import TrackEstimator, TrackModel
from gabarit.tracker import Track, Tracker

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
        # Optional path predictions get independent association/history. A
        # mistaken neural axis can add an alarm, but cannot remove or delay a
        # geometric detection or change the authoritative path's next prior.
        self.learned_tracker = Tracker(self.cfg)
        # Public merged output for evaluators; neither adapter may mutate it.
        self.output_tracks: list[Track] = []
        self.prior: TrackModel | None = None
        self.frame_index = 0
        self.processed_frames = 0
        self.last_stamp: float | None = None
        self.source_frame_id: str | None = None
        self.turnout_hist: list = []
        self.component_model = ComponentModelAssistant() if self.cfg.models.component_policy != "off" else None
        self.path_advisor = None
        self.last_inference_stamp: float | None = None
        self.path_fusion = FarPathFusion(self.cfg.models)
        if self.cfg.models.learned_path:
            from pathlib import Path
            from gabarit.learned_path import FarBevAdvisor
            weights = Path(self.cfg.models.far_path_weights)
            if not weights.is_absolute():
                weights = Path(__file__).resolve().parents[1] / weights
            self.path_advisor = FarBevAdvisor(weights, device=self.cfg.models.device)

    def reset(self) -> None:
        self.track.reset()
        self.ego.reset()
        self.tracker.reset()
        self.learned_tracker.reset()
        self.output_tracks = []
        self.prior = None
        self.frame_index = 0
        self.processed_frames = 0
        self.last_stamp = None
        self.source_frame_id = None
        self.turnout_hist = []
        if self.component_model is not None:
            self.component_model.reset()
        if self.path_advisor is not None:
            self.path_advisor.reset()
        self.path_fusion.reset()
        self.last_inference_stamp = None

    def _blend_path(self, track: TrackModel, prior: TrackModel) -> None:
        """Temporal low-pass of the far path (the near part is fixed by the rails).

        The far path is estimated from sparse wall returns and jitters between
        frames; a standing train must see an identical corridor."""
        moving = abs(self.ego.v) > 1.0
        w_far = 0.5 if moving else 0.15
        w = np.clip((track.s - 40.0) / 40.0, 0.0, 1.0)
        w_new = 1.0 - w * (1.0 - w_far)
        for value_name, sigma_name in (("c", "sig_c"), ("z", "sig_z")):
            new, old = getattr(track, value_name), getattr(prior, value_name)
            new_sigma, old_sigma = getattr(track, sigma_name), getattr(prior, sigma_name)
            if value_name == "z" and prior.sig_z_path is not None:
                # Path drift and local floor measurement noise are distinct:
                # only the former accumulates through the temporal mixture.
                usable = np.isfinite(prior.sig_z_path) & (prior.sig_z_path >= 0.0)
                old_sigma = np.where(usable, prior.sig_z_path, old_sigma)
            valid = (np.isfinite(new) & np.isfinite(old) & np.isfinite(new_sigma) &
                     np.isfinite(old_sigma) & (new_sigma >= 0.0) & (old_sigma >= 0.0))
            blended, sigma = new.copy(), new_sigma.copy()
            weight = w_new[valid]
            delta = new[valid] - old[valid]
            blended[valid] = weight * new[valid] + (1.0 - weight) * old[valid]
            variance = (weight * new_sigma[valid] ** 2 +
                        (1.0 - weight) * old_sigma[valid] ** 2 +
                        weight * (1.0 - weight) * delta ** 2)
            sigma[valid] = np.sqrt(np.maximum(variance, 0.0))
            setattr(track, value_name, blended)
            if value_name == "z":
                # sig_z controls the physical clearance above a local floor.
                # Reusing mixed path drift there cropped real confirmed bodies.
                track.sig_z_path = sigma
            else:
                track.sig_c = sigma
        track.theta = np.gradient(track.c, track.s)
        self.track.prev = track

    def process_raw(self, raw: RawCloud, stamp: float | None = None, *,
                    preserve_source_index: bool = False, **kw) -> FrameResult:
        t0 = time.perf_counter()
        source_reset = self.source_frame_id is not None and raw.frame_id != self.source_frame_id
        if source_reset:
            self.reset()
        self.source_frame_id = raw.frame_id
        sensor = self.cfg.sensor
        cloud = from_raw(raw, sensor.forward_axis, sensor.min_range, up_axis=sensor.up_axis,
                         scan_pattern=sensor.scan_pattern, dual_returns=sensor.dual_returns,
                         elevation_resolution_deg=sensor.elevation_resolution_deg,
                         working_range=self.cfg.detection.max_range + 5.0,
                         preserve_source_index=preserve_source_index)
        decode_ms = (time.perf_counter() - t0) * 1000
        res = self.process_cloud(cloud, stamp if stamp is not None else raw.stamp, **kw)
        if source_reset:
            res.data["state_reset"] = "frame_id_change"
        res.data["timing_ms"]["decode"] = round(decode_ms, 2)
        res.data["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 2)
        res.data["realtime"]["deadline_met"] = res.data["timing_ms"]["total"] <= 100.0
        return res

    def process_cloud(self, cloud: Cloud, stamp: float, frame_index: int | None = None) -> FrameResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        if not math.isfinite(stamp):
            raise ValueError("frame timestamp must be finite")
        # Replaying another bag, looping or seeking must not inherit obstacle
        # tracks and calibration from an unrelated point in recording time.
        reset_reason = None
        if self.source_frame_id is not None and cloud.frame_id != self.source_frame_id:
            self.reset()
            reset_reason = "frame_id_change"
        self.source_frame_id = cloud.frame_id
        if self.last_stamp is not None and (stamp <= self.last_stamp or stamp - self.last_stamp >= 1.0):
            reset_reason = ("time_rewind" if stamp < self.last_stamp else
                            "duplicate_stamp" if stamp == self.last_stamp else "time_gap")
            source_frame_id = self.source_frame_id
            self.reset()
            self.source_frame_id = source_frame_id
        # prior path: previous frame moved by the predicted displacement
        prior = None
        previous_path = self.track.prev
        if self.track.prev is not None and self.last_stamp is not None:
            dt = stamp - self.last_stamp
            dt = dt if 0.02 < dt < 1.0 else 0.1
            prior = propagate_track(self.track.prev, self.ego.v * dt)
        lev, track = self.track.estimate(cloud.xyz, prior)
        if prior is not None:
            self._blend_path(track, prior)
        t_geometry = time.perf_counter()
        delta, speed = self.ego.update(lev, stamp)
        t_ego = time.perf_counter()
        neural_ms = 0.0
        learned_track = None
        learned_status = {"state": "disabled"}
        if self.path_advisor is not None:
            self.path_fusion.advance(delta)
            inferred_this_frame = self.processed_frames % cfg.models.path_interval == 0
            if inferred_this_frame:
                t_neural = time.perf_counter()
                observation = self.path_advisor.observe(lev, cloud.intensity, track)
                self.last_inference_stamp = stamp
                if observation is not None:
                    independent = replace(track, c=track.c.copy(), z=track.z.copy(), theta=track.theta.copy())
                    self.path_fusion.apply(observation, independent, lev)
                else:
                    self.path_fusion.reset()
                neural_ms = (time.perf_counter() - t_neural) * 1000.0
            learned_track = self.path_fusion.refine(track, lev)
            learned_status = {**self.path_advisor.status(), "fusion": dict(self.path_fusion.last_report)}
            learned_status.update(inferred_this_frame=inferred_this_frame,
                                  last_inference_stamp=self.last_inference_stamp)
            learned_status["refined_path"] = learned_track.public(step=2) if learned_track is not None else None
        t2 = time.perf_counter()
        geo = track_coordinates(lev, track, cfg)
        filter_stats: dict = {}
        cands = detect(lev, cloud.ring, geo, track, cfg, cloud.split, filter_stats=filter_stats)
        _sensor_distances(cands, cloud.xyz)
        allow_single = self.ego.quality >= 0.8 and abs(delta) <= 0.01
        if not allow_single:
            for candidate in cands:
                if candidate.single_beam and candidate.rejected is None:
                    candidate.rejected = "single_beam_unverified_standstill"
        component_status = {"policy": "off", "evaluated": 0, "promoted": 0}
        component_ms = 0.0
        if self.component_model is not None:
            t_component = time.perf_counter()
            fresh = track.current_observed_until if track.current_observed_until is not None else track.observed_until
            component_status = self.component_model.apply(
                cands, cloud.xyz, stamp=stamp, observed_until_m=max(0.0, float(fresh)),
                path_reliable=bool(track.rail.any() and track.calibration.get("valid", False)),
                policy=cfg.models.component_policy, min_score=cfg.models.component_min_score,
                min_hits=cfg.models.component_min_hits, half_width_m=cfg.envelope.width / 2.0,
                forward_delta_m=delta,
            )
            component_ms = (time.perf_counter() - t_component) * 1000.0
        t3 = time.perf_counter()
        tracking_prior = propagate_track(previous_path,delta) if previous_path is not None else None
        tracks = self.tracker.update(cands, delta, allow_single=allow_single,
                                     path_prior=tracking_prior,path_current=track,
                                     lev=lev,ring=cloud.ring,geo=geo)
        t4 = time.perf_counter()
        supplemental = []
        learned_detection_ms = learned_tracking_ms = 0.0
        if self.path_advisor is not None:
            started = time.perf_counter()
            if learned_track is not None:
                learned_geo = track_coordinates(lev, learned_track, cfg)
                changed = ((np.abs(learned_geo.d - geo.d) > 1e-6) |
                           (np.abs(learned_geo.h - geo.h) > 1e-6))
                learned_candidates = detect(lev, cloud.ring, learned_geo, learned_track, cfg, cloud.split)
                for candidate in learned_candidates:
                    projected = _project_supplemental(candidate, geo, changed, track, cloud.ring, cands)
                    if projected is not None and (allow_single or not projected.single_beam):
                        supplemental.append(projected)
            learned_detection_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            _sensor_distances(supplemental, cloud.xyz)
            extra_tracks = self.learned_tracker.update(
                supplemental, delta, allow_single=allow_single,
                path_prior=tracking_prior, path_current=track, lev=lev, ring=cloud.ring,
            )
            # External IDs occupy a disjoint namespace. Internal tracking IDs
            # stay positive so ordinary Tracker association remains unchanged.
            extra_tracks = [replace(item, id=-item.id) for item in extra_tracks]
            learned_tracking_ms = (time.perf_counter() - started) * 1000.0
            tracks = _merge_tracks(tracks, extra_tracks)
            learned_status["supplemental_candidates"] = len(supplemental)
            learned_status["supplemental_confirmed"] = sum(item.confirmed and item.id < 0 for item in tracks)
        # Local body renewal creates a new candidate inside Tracker. Measure it
        # from the sensor origin too, after all candidate-producing branches.
        _sensor_distances([item.cand for item in tracks if item.misses == 0], cloud.xyz)
        self.output_tracks = list(tracks)
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
        self.processed_frames += 1

        # confirmed obstacles are held through short dropouts (hysteresis)
        confirmed = sorted((t for t in tracks if t.confirmed), key=lambda t: t.cand.distance)
        pending = [t for t in tracks if not t.confirmed and t.misses == 0]
        obstacle = bool(confirmed)
        nearest = confirmed[0].cand.distance if confirmed else None
        nearest_s = confirmed[0].cand.s_near if confirmed else None
        envelope_until = float(min(track.observed_until, cfg.detection.max_range))
        fresh_until = track.current_observed_until if track.current_observed_until is not None else track.observed_until
        observed = float(min(fresh_until, cfg.detection.max_range))
        free = min(nearest_s, observed) if obstacle else observed
        if obstacle:
            status = "OBSTACLE"
        elif pending:
            status = "CHECK"
        elif not track.rail.any() and not track.lat_obs.any():
            status = "DEGRADED"
        else:
            status = "CLEAR"
        v = max(speed, 0.0)
        ttc = (nearest_s / v) if (obstacle and v > 0.5) else None

        labels = np.zeros(len(cloud), np.uint8)
        ground = geo.in_range & (np.abs(geo.d) < geo.half_width + 0.3) & (geo.h < cfg.envelope.min_height)
        labels[ground] = CLS_GROUND
        labels[geo.corridor] = CLS_ENVELOPE
        for c in cands:
            if c.point_index is not None:
                labels[c.point_index] = CLS_CANDIDATE if c.rejected is None else CLS_ENVELOPE
        for c in supplemental:
            if c.point_index is not None and c.rejected is None:
                labels[c.point_index] = np.maximum(labels[c.point_index], CLS_CANDIDATE)
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
            "current_observed_until_m": round(observed, 1),
            "envelope_until_m": round(envelope_until, 1),
            "envelope_held": track.coast_frames > 0,
            "envelope_coasted_frames": track.coast_frames,
            "analysis_range_m": min(cfg.detection.max_range, cfg.track.horizon),
            "state_reset": reset_reason,
            "sensor": {"forward_axis": cfg.sensor.forward_axis, "up_axis": cfg.sensor.up_axis,
                       "scan_pattern": cfg.sensor.scan_pattern, "frame_id": cloud.frame_id},
            "speed_mps": round(float(speed), 2),
            "speed_kmh": round(float(speed) * 3.6, 1),
            "displacement_m": round(float(delta), 3),
            "ego_quality": round(self.ego.quality, 3),
            "time_to_obstacle_s": None if ttc is None else round(float(ttc), 1),
            "obstacles": [t.public() for t in confirmed],
            "pending": [t.public() for t in pending],
            "candidates": [c.public() for c in (cands + supplemental)[:40]],
            "turnout_ahead_m": None if turnout is None else round(turnout, 1),
            "calibration": track.calibration,
            "envelope": {"width_m": cfg.envelope.width, "height_m": cfg.envelope.height,
                         "min_height_m": cfg.envelope.min_height},
            "points": {"raw": cloud.raw_count, "valid": len(cloud), "envelope": int(geo.corridor.sum()),
                       "filtered": int(filter_stats.get("removed", 0)),
                       "filter_examined": int(filter_stats.get("examined", 0))},
            "path_guard": {"active": self.track.noisy_rail_frames >= 2,
                           "isolated_near_returns": self.track.rail_noise_points},
            "path_lateral_limit_m": cfg.tracking.far_lateral_std,
            "models": {"components": component_status, "far_path": learned_status},
            "timing_ms": {
                "decode": 0.0,
                "filter": round(float(filter_stats.get("ms", 0.0)), 2),
                "track": round((t_geometry - t0) * 1000, 2),
                "egomotion": round((t_ego - t_geometry) * 1000, 2),
                "neural_path": round(neural_ms, 2),
                "learned_detection": round(learned_detection_ms, 2),
                "learned_tracking": round(learned_tracking_ms, 2),
                "component_model": round(component_ms, 2),
                "detect": round(max(0.0, (t3 - t2) * 1000 - filter_stats.get("ms", 0.0) - component_ms), 2),
                "tracking": round((t4 - t3) * 1000, 2),
                "total": round((t4 - t0) * 1000, 2),
            },
            "realtime": {"target_hz": 10.0, "budget_ms": 100.0, "deadline_met": False},
        }
        data["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 2)
        data["realtime"]["deadline_met"] = data["timing_ms"]["total"] <= 100.0
        return FrameResult(data, cloud, lev, geo, track, labels)


def _sensor_distances(candidates: list[Candidate], sensor_xyz: np.ndarray) -> None:
    """Leveling rotates the cloud AND moves its origin onto the rail plane.

    Candidate range is the nearest actual sensor return, independent of that
    calibration translation. Held boxes have no current indices and retain
    the tracker's explicitly predicted range.
    """
    for candidate in candidates:
        indices = candidate.point_index
        if indices is not None and len(indices):
            points = sensor_xyz[indices]
            candidate.distance = float(np.sqrt(np.square(points.astype(np.float64)).sum(axis=1).min()))


def _same_returns(a: Candidate, b: Candidate) -> bool:
    if a.point_index is None or b.point_index is None:
        return False
    count = np.intersect1d(a.point_index, b.point_index, assume_unique=True).size
    return count >= max(3, .5 * min(len(a.point_index), len(b.point_index)))


def _project_supplemental(candidate: Candidate, geo: FrameGeometry, changed: np.ndarray,
                          track: TrackModel, ring: np.ndarray, baseline: list[Candidate]) -> Candidate | None:
    """Express real extra returns in the unchanged geometric path frame."""
    indices = candidate.point_index
    if (candidate.rejected not in (None, "low_score") or candidate.rail_local or indices is None
            or not len(indices) or not changed[indices].any()):
        return None
    # Model assistance may recover an edge/path-score rejection, but cannot
    # override explicit ground, fixtures, dust or single-return safeguards.
    recoverable = {None, "low_score", "edge_far", "far_weak"}
    if any(other.rejected not in recoverable and _same_returns(candidate, other) for other in baseline):
        return None
    fresh = track.current_observed_until if track.current_observed_until is not None else track.observed_until
    s, d, h = geo.s[indices], geo.d[indices], geo.h[indices]
    if not np.isfinite(np.column_stack((s, d, h))).all() or float(s.max()) > min(fresh, track.observed_until):
        return None
    return replace(candidate, id=-abs(candidate.id), s_near=float(s.min()), s_far=float(s.max()),
                   d_min=float(d.min()), d_max=float(d.max()), h_min=float(h.min()), h_max=float(h.max()),
                   lateral=float(np.median(d)), n_points=len(indices), n_rings=len(np.unique(ring[indices])),
                   reasons=candidate.reasons + ["attention_please_supplemental_path"],
                   point_index=indices.copy())


def _same_body(a: Track, b: Track) -> bool:
    if a.misses == 0 and b.misses == 0 and _same_returns(a.cand, b.cand):
        return True
    def bounds(c):
        return np.array([[c.s_near, c.s_far], [c.d_min, c.d_max], [c.h_min, c.h_max]])
    aa, bb = bounds(a.cand), bounds(b.cand)
    if not np.isfinite(np.r_[aa, bb]).all():
        return False
    overlap = np.minimum(aa[:, 1], bb[:, 1]) - np.maximum(aa[:, 0], bb[:, 0])
    smaller = np.minimum(np.diff(aa, axis=1).ravel(), np.diff(bb, axis=1).ravel())
    fraction = np.array([.5, .5, .5])
    if a.misses or b.misses:
        # A held box can describe the previous visible part of the same body;
        # the current geometric tracker may have renewed its opposite side.
        # Keep longitudinal/vertical agreement, relax only lateral cropping.
        fraction[1] = .25
    return bool(np.all(overlap + np.array([.15, .08, .12]) >= fraction * smaller))


def _merge_tracks(baseline: list[Track], supplemental: list[Track]) -> list[Track]:
    """Deduplicate the displayed union without touching either tracker state.

    Every baseline confirmed track survives. An extra confirmed body may
    replace only a duplicate pending *display entry*, never its history.
    """
    confirmed = [item for item in baseline if item.confirmed]
    for item in supplemental:
        if item.confirmed and not any(_same_body(item, other) for other in confirmed):
            confirmed.append(item)
    pending = [item for item in baseline if not item.confirmed and item.misses == 0
               and not any(_same_body(item, other) for other in confirmed)]
    for item in supplemental:
        if (not item.confirmed and item.misses == 0 and
                not any(_same_body(item, other) for other in confirmed + pending)):
            pending.append(item)
    return confirmed + pending


def _blend(new: np.ndarray, old: np.ndarray, w_new: np.ndarray) -> np.ndarray:
    ok = np.isfinite(old)
    out = new.copy()
    out[ok] = w_new[ok] * new[ok] + (1 - w_new[ok]) * old[ok]
    return out


def kmh(v: float) -> float:
    return v * 3.6 if math.isfinite(v) else 0.0
