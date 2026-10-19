"""Playback session of the web interface: bag source, pipeline, synthetic
obstacles and binary frame packing."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gabarit.config import Config
from gabarit.pipeline import FrameResult, Pipeline
from gabarit.rosbag import BagInfo, BagReader
from gabarit.synth import SHAPES, SyntheticObstacle, inject

MAGIC = b"GBR1"
MAX_POINTS = 90_000


def pack_frame(res: FrameResult, meta: dict[str, Any]) -> bytes:
    """Binary frame: magic, json length, json, float32 xyz, uint8 class, uint8 intensity."""
    lev = res.lev
    labels = res.labels
    f = lev[:, 1]
    keep = (f > -2) & (f < meta.get("analysis_range_m", 200.0) + 5.0) & (lev[:, 2] > -1.5) & (lev[:, 2] < 7.0)
    c_at = np.interp(f, res.track.s, res.track.c)
    keep &= np.abs(lev[:, 0] - c_at) < 18
    # thin the dense near field, keep everything relevant for the decision
    n = np.arange(len(lev))
    thin = np.where(f < 15, 6, np.where(f < 30, 3, np.where(f < 60, 2, 1)))
    keep &= (n % thin == 0) | (labels >= 2)
    idx = np.flatnonzero(keep)
    if len(idx) > MAX_POINTS:
        important = idx[labels[idx] >= 2]
        rest = idx[labels[idx] < 2]
        take = max(MAX_POINTS - len(important), 0)
        rest = rest[np.linspace(0, len(rest) - 1, take).astype(int)] if take and len(rest) else rest[:0]
        idx = np.concatenate([rest, important])
    xyz = np.ascontiguousarray(lev[idx], dtype=np.float32)
    cls = np.ascontiguousarray(labels[idx], dtype=np.uint8)
    inten = np.clip(res.cloud.intensity[idx], 0, 255).astype(np.uint8)
    tr = res.track
    step = max(1, len(tr.s) // 120)
    meta = dict(meta)
    meta["track"] = {
        "s": np.round(tr.s[::step], 2).tolist(),
        "c": np.round(tr.c[::step], 3).tolist(),
        "z": np.round(tr.z[::step], 3).tolist(),
        "wl": [None if not math.isfinite(v) else round(float(v), 2) for v in tr.wall_left[::step]],
        "wr": [None if not math.isfinite(v) else round(float(v), 2) for v in tr.wall_right[::step]],
        "observed_until": round(tr.observed_until, 1),
        "current_observed_until": round(tr.current_observed_until if tr.current_observed_until is not None
                                        else tr.observed_until, 1),
        "coasted_frames": tr.coast_frames,
        "rail_until": round(tr.rail_until, 1),
    }
    body = json.dumps(meta, ensure_ascii=False, default=_json_default).encode("utf-8")
    pad = (-len(body)) % 4
    header = MAGIC + struct.pack("<I", len(body) + pad) + body + b" " * pad
    return header + struct.pack("<I", len(idx)) + xyz.tobytes() + cls.tobytes() + inten.tobytes()


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@dataclass
class WorldObstacle:
    ob: SyntheticObstacle
    world: float   # position along the travelled distance (odometer frame)


class BagSession:
    """Sequential processing of one bag with random access (seek = warm-up)."""

    WARMUP = 12

    def __init__(self, info: BagInfo, cfg: Config):
        self.info = info
        self.cfg = cfg
        self.reader = BagReader(info)
        self.pipe = Pipeline(cfg)
        self.next_index = 0
        self.odometer = 0.0
        self.obstacles: list[WorldObstacle] = []
        self.lock = threading.Lock()
        self.next_obstacle_id = 1
        self.last_meta: dict | None = None

    @property
    def frames(self) -> int:
        return len(self.reader)

    def close(self) -> None:
        # Cancelling an asyncio player does not cancel its worker thread.
        # A bag switch must wait for the old frame before closing its SQLite DB.
        with self.lock:
            self.reader.close()

    def set_config(self, cfg: Config) -> None:
        with self.lock:
            self.cfg = cfg
            idx = max(self.next_index - 1, 0)
            self.pipe = Pipeline(cfg)
            self._warm(idx)

    def _warm(self, index: int) -> None:
        self.pipe.reset()
        start = max(0, index - self.WARMUP)
        for i in range(start, index):
            ts, raw = self.reader.read(i)
            raw = self._inject(raw)
            self.pipe.process_raw(raw, ts * 1e-9, frame_index=i)
            # Reprocessing history calibrates the detector; it is not another
            # playback movement. Only step() advances the virtual scene. Test
            # objects remain at their displayed relative range after a seek
            # or a configuration change.
        self.next_index = index

    def seek(self, index: int) -> None:
        with self.lock:
            index = int(min(max(index, 0), self.frames - 1))
            self._warm(index)

    def add_obstacle(self, shape: str, s: float, d: float, yaw_deg: float = 0.0) -> dict:
        with self.lock:
            ob = SyntheticObstacle(shape=shape if shape in SHAPES else "person", s=float(s), d=float(d),
                                   id=self.next_obstacle_id, yaw_deg=float(yaw_deg))
            self.next_obstacle_id += 1
            self.obstacles.append(WorldObstacle(ob, self.odometer + ob.s))
            return ob.public()

    def clear_obstacles(self) -> None:
        with self.lock:
            self.obstacles.clear()

    def _advance(self, displacement: float) -> None:
        self.odometer += displacement
        for w in self.obstacles:
            w.ob.s = w.world - self.odometer
        self.obstacles = [w for w in self.obstacles if w.ob.s > 1.0]

    def _inject(self, raw):
        if not self.obstacles or self.pipe.track.prev is None or not self.pipe.track.calib.valid:
            return raw
        injected, _ = inject(raw, [w.ob for w in self.obstacles], self.pipe.track.prev, self.pipe.track.calib,
                             self.cfg.sensor.forward_axis, up_axis=self.cfg.sensor.up_axis,
                             scan_pattern=self.cfg.sensor.scan_pattern)
        return injected

    def step(self) -> tuple[bytes, dict] | None:
        """Process the next frame; returns (packed frame, metadata)."""
        with self.lock:
            if self.next_index >= self.frames:
                return None
            i = self.next_index
            ts, raw = self.reader.read(i)
            t0 = time.perf_counter()
            raw = self._inject(raw)
            inject_ms = (time.perf_counter() - t0) * 1000
            res = self.pipe.process_raw(raw, ts * 1e-9, frame_index=i)
            self._advance(res.data["displacement_m"])
            self.next_index = i + 1
            meta = dict(res.data)
            meta.update({
                "type": "frame",
                "bag": self.info.id,
                "frames": self.frames,
                "bag_time_s": round((ts - self.info.start_ns) / 1e9, 3),
                "synthetic": [w.ob.public() for w in self.obstacles],
                "inject_ms": round(inject_ms, 2),
            })
            self.last_meta = meta
            return pack_frame(res, meta), meta


def scan_bag(info: BagInfo, cfg: Config, progress=None, stop_flag=None) -> list[dict]:
    """Fast pass over the whole bag for the timeline (no point payload)."""
    reader = BagReader(info)
    pipe = Pipeline(cfg)
    out = []
    try:
        for i, ts, raw in reader.iterate():
            if stop_flag is not None and stop_flag():
                break
            d = pipe.process_raw(raw, ts * 1e-9, frame_index=i).data
            item = {
                "i": i,
                "status": d["status"],
                "dist": d["nearest_distance_m"],
                "free": d["free_distance_m"],
                "v": d["speed_kmh"],
                "ids": [o["track_id"] for o in d["obstacles"]],
                "ms": d["timing_ms"]["total"],
            }
            out.append(item)
            if progress is not None and (i % 10 == 0 or i == len(reader) - 1):
                progress(out, len(reader))
    finally:
        reader.close()
    return out


def load_cached_scan(cache: Path, info: BagInfo, cfg_hash: str) -> list[dict] | None:
    p = cache / f"{info.name}-{cfg_hash}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def save_cached_scan(cache: Path, info: BagInfo, cfg_hash: str, items: list[dict]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{info.name}-{cfg_hash}.json").write_text(json.dumps(items), encoding="utf-8")
