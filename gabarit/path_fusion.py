"""Conservative integration of an independently inferred far path.

The neural prediction may refine an independent path copy, but cannot turn
unobserved space into a measured clearance envelope or overwrite an observed
rail. Pipeline uses this copy only for additive, independently tracked evidence.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from gabarit.config import ModelsConfig
from gabarit.track import TrackModel


def _sample(s, values, target):
    return np.interp(target, s, values, left=np.nan, right=np.nan)


def _runs(mask: np.ndarray, minimum: int = 3) -> np.ndarray:
    """Keep runs, rather than isolated confident model pixels."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    out = np.zeros_like(mask)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= minimum:
            out[start:end] = True
    return out


def _supported(target, c, z, xyz):
    """Require fresh returns around stations; old model confidence is insufficient."""
    finite = np.isfinite(xyz).all(axis=1)
    points = xyz[finite]
    if not len(points):
        return np.zeros(target.shape, dtype=bool)
    pc = np.interp(points[:, 1], target, c)
    pz = np.interp(points[:, 1], target, z)
    support = points[(np.abs(points[:, 0] - pc) <= 8.0) &
                     (points[:, 2] - pz >= -2.0) & (points[:, 2] - pz <= 7.0), 1]
    support = np.sort(support)
    count = (np.searchsorted(support, target + 3.0, side="right") -
             np.searchsorted(support, target - 3.0, side="left"))
    return count >= 3


class FarPathFusion:
    def __init__(self, cfg: ModelsConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.previous = None
        self._correction = None
        self.last_report = {"state": "waiting", "fused_stations": 0}

    def advance(self, forward_delta_m: float) -> None:
        if self.previous is not None:
            self.previous[0] -= forward_delta_m
        if self._correction is not None:
            self._correction["s"] -= forward_delta_m
            self._correction["age"] += 1

    def refine(self, track: TrackModel, xyz: np.ndarray) -> TrackModel | None:
        """Apply a bounded cached correction to an independent current path copy.

        Authoritative geometry and its temporal prior are never mutated. The
        cache lasts at most one inference interval and cannot claim new range,
        overwrite current rails or survive absent current cloud support.
        """
        cached = self._correction
        if cached is None:
            return None
        age = cached["age"]
        self.last_report["cache_age_frames"] = age
        if age > self.cfg.path_interval:
            self._correction = None
            self.last_report["active_stations"] = 0
            return None
        target = track.s
        dc = _sample(cached["s"], cached["dc"], target)
        dz = _sample(cached["s"], cached["dz"], target)
        use = _sample(cached["s"], cached["valid"].astype(float), target) >= 0.999
        fresh = track.current_observed_until if track.current_observed_until is not None else track.observed_until
        w = self.cfg.path_blend * np.clip((target - self.cfg.path_min_distance) / 20.0, 0.0, 1.0)
        use &= ((target >= max(self.cfg.path_min_distance, track.rail_until + 3.0)) &
                (target <= min(track.observed_until, fresh)) & ~track.rail &
                np.isfinite(dc) & np.isfinite(dz) &
                (np.abs(dc) <= w * self.cfg.path_max_lateral_change + 1e-9) &
                (np.abs(dz) <= w * self.cfg.path_max_vertical_change + 1e-9))
        # Forward transport alone cannot compensate yaw or a changing sensor
        # calibration. Expire stations whose geometric reference has moved.
        use &= ((np.abs(track.c - _sample(cached["s"], cached["base_c"], target)) <= 0.20) &
                (np.abs(track.z - _sample(cached["s"], cached["base_z"], target)) <= 0.08))
        c, z = track.c.copy(), track.z.copy()
        c[use] += dc[use]
        z[use] += dz[use]
        use &= _supported(target, c, z, xyz)
        use = _runs(use)
        self.last_report["active_stations"] = int(use.sum())
        if not use.any():
            return None
        c, z = track.c.copy(), track.z.copy()
        c[use] += dc[use]
        z[use] += dz[use]
        return replace(track, c=c, z=z, theta=np.gradient(c, target))

    def apply(self, observation, track: TrackModel, xyz: np.ndarray) -> dict:
        """Confirm observations and update a caller-owned independent path copy."""
        cfg = self.cfg
        # A new rejected observation invalidates the last accepted correction.
        self._correction = None
        try:
            s = np.asarray(observation.s, dtype=float)
            arrays = [np.asarray(getattr(observation, name), dtype=float)
                      for name in ("c", "z", "sig_c", "sig_z", "valid")]
        except (ValueError, TypeError, AttributeError):
            self.reset()
            self.last_report = {"state": "invalid_observation", "fused_stations": 0}
            return self.last_report
        if (s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all() or
                not np.all(np.diff(s) > 0)):
            self.reset()
            self.last_report = {"state": "invalid_observation", "fused_stations": 0}
            return self.last_report
        if any(a.shape != s.shape for a in arrays):
            self.reset()
            self.last_report = {"state": "invalid_observation", "fused_stations": 0}
            return self.last_report
        c, z, sc, sz, valid = arrays
        valid = (np.isfinite(valid) & (valid > 0.5) & np.isfinite(c) & np.isfinite(z) &
                 np.isfinite(sc) & np.isfinite(sz) & (sc >= 0) & (sz >= 0) &
                 (sc <= cfg.path_max_sigma) & (sz <= cfg.path_max_sigma))
        # Interpolating a confidence mask must not bridge invalid stations.
        target = track.s
        good = _sample(s, valid.astype(float), target) >= 0.999
        nc, nz = _sample(s, c, target), _sample(s, z, target)
        fresh = track.current_observed_until if track.current_observed_until is not None else track.observed_until
        good &= ((target >= max(cfg.path_min_distance, track.rail_until + 3.0)) &
                 (target <= min(track.observed_until, fresh)) & ~track.rail &
                 np.isfinite(track.c) & np.isfinite(track.z) &
                 (np.abs(nc - track.c) <= cfg.path_max_lateral_change) &
                 (np.abs(nz - track.z) <= cfg.path_max_vertical_change))

        # A confident empty BEV pixel is not evidence. Require current returns
        # around each accepted station, using distance windows of +/- 3 m.
        good &= _supported(target, nc, nz, xyz)
        good = _runs(good)

        hits = good.astype(int)
        if self.previous is not None:
            ps, pc, pz, ph = self.previous
            consistent = ((np.abs(nc - _sample(ps, pc, target)) <= 0.30) &
                          (np.abs(nz - _sample(ps, pz, target)) <= 0.15))
            old_hits = np.nan_to_num(_sample(ps, ph, target), nan=0.0).astype(int)
            hits += np.where(good & consistent, old_hits, 0)
        self.previous = [target.copy(), nc.copy(), nz.copy(), np.minimum(hits, cfg.path_confirmations)]
        use = _runs(good & (hits >= cfg.path_confirmations))
        if use.any():
            # The near end fades in smoothly. Keep geometric uncertainty and
            # measurement flags: neural sigma is not calibrated sensor error.
            w = cfg.path_blend * np.clip((target - cfg.path_min_distance) / 20.0, 0.0, 1.0)
            dc, dz = np.zeros_like(target), np.zeros_like(target)
            dc[use] = w[use] * (nc[use] - track.c[use])
            dz[use] = w[use] * (nz[use] - track.z[use])
            self._correction = {"s": target.copy(), "dc": dc, "dz": dz,
                                "base_c": track.c.copy(), "base_z": track.z.copy(),
                                "valid": use.copy(), "age": 0}
            track.c[use] += dc[use]
            track.z[use] += dz[use]
            track.theta = np.gradient(track.c, target)
        self.last_report = {
            "state": "fused" if use.any() else "warming_up" if good.any() else "no_agreement",
            "eligible_stations": int(good.sum()), "fused_stations": int(use.sum()),
            "fused_until_m": float(target[use][-1]) if use.any() else None,
        }
        return self.last_report
