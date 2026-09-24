"""Temporal confirmation of obstacles.

Tracks live in track coordinates. Between frames a static object moves towards
the train by the ego displacement, so the prediction is ``s - delta``. An
obstacle is confirmed after ``confirm_hits`` associations within the last
``confirm_window`` frames and is kept while it is seen (hysteresis on drop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gabarit.config import Config
from gabarit.detect import Candidate


@dataclass
class Track:
    id: int
    s: float
    d: float
    cand: Candidate
    history: list = field(default_factory=list)   # 1 = hit, 0 = miss (latest last)
    hits: int = 0
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    first_seen_s: float = 0.0
    confirmed_at_s: float | None = None
    confirmed_frame: int | None = None
    first_frame: int = 0
    score: float = 0.0
    v_rel: float = 0.0   # own motion along the track (m/frame), positive = away
    lat_hist: list = field(default_factory=list)

    def public(self) -> dict:
        c = self.cand.public()
        c.update({
            "track_id": self.id,
            "confirmed": self.confirmed,
            "age_frames": self.age,
            "hits": self.hits,
            "confidence": round(self.confidence, 3),
            "first_seen_m": round(self.first_seen_s, 1),
            "confirmed_at_m": None if self.confirmed_at_s is None else round(self.confirmed_at_s, 1),
            "missed": self.misses > 0,
        })
        return c

    @property
    def confidence(self) -> float:
        recent = self.history[-6:]
        rate = sum(recent) / max(len(recent), 1)
        return float(min(1.0, 0.6 * rate + 0.4 * min(self.score, 1.0)))


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame = 0

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    def update(self, cands: list[Candidate], delta: float) -> list[Track]:
        tc = self.cfg.tracking
        self.frame += 1
        good = [c for c in cands if c.rejected is None]
        # predict
        for t in self.tracks:
            t.s -= delta + t.v_rel
        # greedy association by cost
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = tc.gate + 0.01 * max(t.s, 0.0)
            for ci, c in enumerate(good):
                ds = c.s_near - t.s
                dd = c.lateral - t.d
                if abs(ds) < gate and abs(dd) < tc.lateral_gate + 0.004 * max(t.s, 0.0):
                    pairs.append((abs(ds) / gate + abs(dd) / tc.lateral_gate, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t = self.tracks[ti]
            c = good[ci]
            # objects may move on their own (a walking person): follow the measurement
            innov = c.s_near - t.s
            t.v_rel = float(np.clip(0.6 * t.v_rel - 0.4 * innov, -0.4, 0.4))
            t.s = c.s_near
            t.d = c.lateral
            t.lat_hist.append(c.lateral)
            del t.lat_hist[:-10]
            t.cand = c
            t.score = 0.6 * t.score + 0.4 * c.score
            t.history.append(1)
            t.hits += 1
            t.misses = 0
        for ti, t in enumerate(self.tracks):
            t.age += 1
            if ti not in used_t:
                t.history.append(0)
                t.misses += 1
            del t.history[:-12]
            if t.cand.low:
                win, need = tc.low_confirm_window, tc.low_confirm_hits
            elif t.s > tc.far_distance:
                win, need = tc.far_confirm_window, tc.far_confirm_hits
            else:
                win, need = tc.confirm_window, tc.confirm_hits
            recent = t.history[-win:]
            stable = t.s <= tc.far_distance or len(t.lat_hist) < 3 or float(np.std(t.lat_hist[-need:])) < tc.far_lateral_std
            if not t.confirmed and sum(recent) >= need and stable:
                t.confirmed = True
                t.confirmed_at_s = t.s
                t.confirmed_frame = self.frame
        # drop stale tracks
        keep = []
        for t in self.tracks:
            limit = tc.drop_misses if t.confirmed else (3 if t.cand.low or t.s > tc.far_distance else 2)
            if t.misses <= limit and t.s > -2.0:
                keep.append(t)
        self.tracks = keep
        # new tracks
        for ci, c in enumerate(good):
            if ci in used_c:
                continue
            t = Track(self.next_id, c.s_near, c.lateral, c, [1], 1, 0, 0, False, c.s_near,
                      first_frame=self.frame, score=c.score, lat_hist=[c.lateral])
            self.next_id += 1
            if tc.confirm_hits <= 1:
                t.confirmed = True
                t.confirmed_at_s = t.s
                t.confirmed_frame = self.frame
            self.tracks.append(t)
        return self.tracks
