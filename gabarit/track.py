"""Sensor self-calibration and estimation of the track path ahead of the train.

Near the train the running rails are detected directly (two height peaks with
the track gauge between them). They give the pose of the sensor relative to
the track and the centre line up to ~40-60 m. Further away the rails are not
resolved, so the path is continued by following the tunnel structure that is
parallel to the track: both walls (or the platform edge / column row), the
floor and the ceiling. Each structure is followed by its own Kalman filter
along the distance; their offsets to the track are learned where the rails are
visible. The centre line itself is a third Kalman filter (position, heading,
curvature) that fuses rails, walls and the path of the previous frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from gabarit.config import Config
from gabarit.filtering import support_mask

LAT_BIN = 0.04
LAT_MIN, LAT_MAX = -8.0, 8.0
N_LAT = int(round((LAT_MAX - LAT_MIN) / LAT_BIN))


# --------------------------------------------------------------------------
# Calibration


@dataclass
class Calibration:
    height: float = 1.1        # rail head below the sensor, m
    pitch: float = 0.0         # slope dz/df of the rail plane in the sensor frame
    roll: float = 0.0          # slope dz/dx of the rail plane in the sensor frame
    valid: bool = False
    frames: int = 0

    @classmethod
    def from_config(cls, cfg: Config) -> "Calibration":
        s = cfg.sensor
        return cls(s.default_height, math.tan(math.radians(s.default_pitch_deg)),
                   math.tan(math.radians(s.default_roll_deg)), False, 0)

    def level(self, xyz: np.ndarray) -> np.ndarray:
        """Rotate/translate sensor points so that the rail plane is z = 0."""
        a = math.atan(self.pitch)
        b = math.atan(self.roll)
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        x, f, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        f1 = f * ca + z * sa
        z1 = -f * sa + z * ca
        x2 = x * cb + z1 * sb
        z2 = -x * sb + z1 * cb
        out = np.empty_like(xyz)
        out[:, 0] = x2
        out[:, 1] = f1
        out[:, 2] = z2 + self.height
        return out

    def update(self, height: float, pitch: float, roll: float, alpha: float) -> None:
        if not self.valid:
            self.height, self.pitch, self.roll = height, pitch, roll
            self.valid = True
        else:
            self.height += alpha * (height - self.height)
            self.pitch += alpha * (pitch - self.pitch)
            self.roll += alpha * (roll - self.roll)
        self.frames += 1

    def public(self) -> dict:
        return {
            "height_m": round(self.height, 3),
            "pitch_deg": round(math.degrees(math.atan(self.pitch)), 3),
            "roll_deg": round(math.degrees(math.atan(self.roll)), 3),
            "valid": self.valid,
        }


# --------------------------------------------------------------------------
# Small helpers


class _KF:
    """Constant-curvature Kalman filter along the distance: [value, slope, curvature].

    Written with plain floats: it runs ~600 times per frame and NumPy call
    overhead would dominate for 3x3 matrices.
    """

    __slots__ = ("x", "d", "k", "P", "q", "miss", "alive")

    def __init__(self, x0, sig, q):
        self.x, self.d, self.k = float(x0[0]), float(x0[1]), float(x0[2])
        self.P = [[float(sig[0]) ** 2, 0.0, 0.0], [0.0, float(sig[1]) ** 2, 0.0], [0.0, 0.0, float(sig[2]) ** 2]]
        self.q = q
        self.miss = 0.0
        self.alive = True

    @property
    def X(self):
        return (self.x, self.d, self.k)

    def predict(self, ds: float) -> None:
        h = 0.5 * ds * ds
        self.x += ds * self.d + h * self.k
        self.d += ds * self.k
        P0, P1, P2 = self.P
        r0 = [P0[j] + ds * P1[j] + h * P2[j] for j in range(3)]
        r1 = [P1[j] + ds * P2[j] for j in range(3)]
        r2 = list(P2)
        Pn = [[r[0] + ds * r[1] + h * r[2], r[1] + ds * r[2], r[2]] for r in (r0, r1, r2)]
        Pn[0][0] += 1e-5 * ds
        Pn[1][1] += 1e-7 * ds
        Pn[2][2] += self.q * self.q * ds
        self.P = Pn

    def update(self, z: float, var: float) -> None:
        P = self.P
        S = P[0][0] + var
        K = (P[0][0] / S, P[1][0] / S, P[2][0] / S)
        r = z - self.x
        self.x += K[0] * r
        self.d += K[1] * r
        self.k += K[2] * r
        row = P[0]
        self.P = [[P[i][j] - K[i] * row[j] for j in range(3)] for i in range(3)]

    @property
    def sig(self) -> float:
        return math.sqrt(max(self.P[0][0], 0.0))


def _slices(cfg: Config) -> np.ndarray:
    t = cfg.track
    out = []
    s = t.start
    while s < t.horizon:
        ds = t.near_step if s < t.near_far_switch else t.far_step
        out.append((s, min(s + ds, t.horizon)))
        s += ds
    return np.array(out)


def _pct(a, q: float) -> float:
    """Percentile (0..100) by partition, much cheaper than np.percentile for small arrays."""
    a = np.asarray(a, dtype=float)
    k = int(round((len(a) - 1) * q / 100.0))
    return float(np.partition(a, k)[k])


def _med(a) -> float:
    if isinstance(a, list):
        b = sorted(a)
        return float(b[len(b) // 2]) if b else float("nan")
    return _pct(a, 50.0)


@lru_cache(maxsize=128)
def _window_indices(n: int, half: int) -> np.ndarray:
    # Repeated rail windows have the same short lengths. Reusing these indices
    # avoids thousands of np.pad / sliding_window_view dispatches per second.
    return np.arange(n)[:, None] + np.arange(2 * half + 1)[None, :]


def _window_percentile(a: np.ndarray, half: int, q: float) -> np.ndarray:
    """Percentile in a sliding window ignoring NaN (fast, sort based)."""
    pad = np.full(len(a) + 2 * half, np.nan)
    pad[half:half + len(a)] = a
    win = pad[_window_indices(len(a), half)]
    finite = np.isfinite(win)
    cnt = finite.sum(axis=1)
    srt = np.sort(np.where(finite, win, np.inf), axis=1)
    idx = np.floor(np.maximum(cnt - 1, 0) * q).astype(int)
    out = srt[np.arange(len(a)), idx]
    out[cnt == 0] = np.nan
    return out


def _window_max(a: np.ndarray, half: int) -> np.ndarray:
    pad = np.full(len(a) + 2 * half, -np.inf)
    pad[half:half + len(a)] = np.where(np.isfinite(a), a, -np.inf)
    win = pad[_window_indices(len(a), half)]
    return win.max(axis=1)


def _short_line(x: list, y: list) -> tuple[float, float]:
    """Least-squares line for a few rail centres, without a LAPACK call."""
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    xx = sum((v - xm) ** 2 for v in x)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / xx if xx > 0.0 else 0.0
    return slope, ym - slope * xm


def _rail_pair(xs, prof, center, gauge, tol, search, min_prom, center_bounds=None):
    """Find the best pair of rail peaks in one lateral max-height profile."""
    finite = np.isfinite(prof)
    if finite.sum() < 6:
        return None
    lo = np.searchsorted(xs, center - gauge / 2 - search - 0.35)
    hi = np.searchsorted(xs, center + gauge / 2 + search + 0.35)
    lo, hi = max(lo, 0), min(hi, len(xs))
    seg = prof[lo:hi]
    if np.isfinite(seg).sum() < 6:
        return None
    base = _window_percentile(seg, 8, 0.3)
    locmax = _window_max(seg, 2)
    prom = seg - base
    peak = np.isfinite(seg) & (seg >= locmax) & (prom > min_prom)
    idx = np.flatnonzero(peak)
    if len(idx) < 2:
        return None
    px = xs[lo + idx]
    pz = seg[idx]
    pp = prom[idx]
    best = None
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            g = px[j] - px[i]
            if abs(g - gauge) > tol:
                continue
            c = 0.5 * (px[i] + px[j])
            if center_bounds is not None and not center_bounds[0] <= c <= center_bounds[1]:
                continue
            if abs(c - center) > search or abs(pz[i] - pz[j]) > 0.12:
                continue
            score = pp[i] + pp[j] - 2 * abs(g - gauge) - abs(c - center)
            if best is None or score > best[0]:
                best = (score, px[i], pz[i], px[j], pz[j])
    return best


def _all_pairs(xs, prof, lo_x, hi_x, gauge, tol, min_prom):
    """Centres of every pair of rail-like peaks spaced by the gauge (turnouts)."""
    lo = max(int(np.searchsorted(xs, lo_x)), 0)
    hi = min(int(np.searchsorted(xs, hi_x)), len(xs))
    seg = prof[lo:hi]
    if np.isfinite(seg).sum() < 6:
        return []
    base = _window_percentile(seg, 8, 0.3)
    locmax = _window_max(seg, 2)
    idx = np.flatnonzero(np.isfinite(seg) & (seg >= locmax) & (seg - base > min_prom))
    px = xs[lo + idx]
    return [0.5 * (a + b) for i, a in enumerate(px) for b in px[i + 1:] if abs(b - a - gauge) < tol]


def _detect_turnout(pairs_by_s: list) -> float | None:
    """A turnout shows a second pair of rails whose centre moves steadily away
    from the running track (0.03-0.15 m per metre). Returns the distance where
    it starts, or None. Isolated extra pairs (trough edges, contact rail) do not
    form such a straight diverging line and are ignored."""
    pts = [(s, d) for s, ds in pairs_by_s for d in ds if 0.12 < abs(d) < 1.6]
    for sign in (+1, -1):
        cand = sorted((s, abs(d)) for s, d in pts if np.sign(d) == sign)
        if len(cand) < 5:
            continue
        ss = np.array([c[0] for c in cand])
        dd = np.array([c[1] for c in cand])
        for i in range(len(cand)):
            win = (ss >= ss[i]) & (ss <= ss[i] + 16.0)
            if len(np.unique(ss[win])) < 5:
                continue
            x, y = ss[win], dd[win]
            # RANSAC-free robust line: keep points close to the median line
            k, b = np.polyfit(x, y, 1)
            keep = np.abs(y - (k * x + b)) < 0.12
            if keep.sum() < 5 or len(np.unique(x[keep])) < 5:
                continue
            k, b = np.polyfit(x[keep], y[keep], 1)
            span = (k * x[keep].max() + b) - (k * x[keep].min() + b)
            resid = np.std(y[keep] - (k * x[keep] + b))
            if 0.03 < k < 0.15 and span > 0.5 and resid < 0.08:
                return float(x[keep].min())
    return None


def _group_percentile(values: np.ndarray, groups: np.ndarray, ngroups: int, q: float,
                      lo: float = -4.0, hi: float = 4.0, step: float = 0.02) -> np.ndarray:
    """Approximate per-group percentile via a histogram (resolution ``step``)."""
    out = np.full(ngroups, np.nan)
    if not len(values):
        return out
    nb = int(round((hi - lo) / step))
    b = np.clip(((values - lo) / step).astype(np.int64), 0, nb - 1)
    hist = np.bincount(groups * nb + b, minlength=ngroups * nb).reshape(ngroups, nb)
    cum = np.cumsum(hist, axis=1)
    tot = cum[:, -1]
    ok = tot > 0
    target = np.maximum(1, np.ceil(tot * q))
    idx = (cum < target[:, None]).sum(axis=1)
    out[ok] = lo + (idx[ok] + 0.5) * step
    return out


def _plausible(s: np.ndarray, c: np.ndarray) -> bool:
    """Metro curves have radius > ~150 m: reject exploded solutions."""
    if not np.all(np.isfinite(c)):
        return False
    curv = np.abs(np.gradient(np.gradient(c, s), s))
    near = s < 120
    return bool(np.all(curv[near] < 1 / 120.0) and np.all(np.abs(c[near]) < 40.0))


def _smooth_profile(x: np.ndarray, meas: list, prior: np.ndarray | None, prior_sig: np.ndarray | None,
                    fallback: np.ndarray, lam: float, huber: float, side_steps: bool):
    """Robust penalised least squares: third-difference (change of curvature)
    penalty, Huber weights, optional correction of per-side offset steps.

    Returns the profile, its approximate standard deviation and a mask of
    slices supported by accepted measurements.
    """
    n = len(x)
    if not meas:
        sig = np.full(n, 0.5)
        return fallback.copy(), sig, np.zeros(n, bool)
    # fit on a uniform 1 m grid, report at the slice centres
    g0 = float(np.floor(x[0]))
    grid = np.arange(g0, float(np.ceil(x[-1])) + 1.0, 1.0)
    ng = len(grid)
    m = np.array(meas, float)
    kk = m[:, 0].astype(int)
    k = np.clip(np.round(x[kk] - g0).astype(int), 0, ng - 1)
    val = m[:, 1].copy()
    w0 = 1.0 / m[:, 2]
    src = m[:, 3].astype(int)
    D = np.diff(np.eye(ng), 3, axis=0)
    P = lam * (D.T @ D)
    pw = np.zeros(ng)
    pv = np.zeros(ng)
    if prior is not None:
        pg = np.interp(grid, x, prior)
        sg = np.interp(grid, x, prior_sig)
        ok = np.isfinite(pg)
        pw[ok] = 0.15 / np.maximum(sg[ok], 0.05) ** 2
        pv[ok] = pg[ok]
        # the previous path only fills gaps between current measurements
        near = np.zeros(ng, bool)
        for off in range(-3, 4):
            near[np.clip(k + off, 0, ng - 1)] = True
        pw[near] = 0.0
    w = w0.copy()
    for it in range(4):
        A = P.copy()
        A[np.diag_indices(ng)] += pw + 1e-6
        b = pw * pv
        np.add.at(A, (k, k), w)
        np.add.at(b, k, w * val)
        sol = np.linalg.solve(A, b)
        r = (val - sol[k]) * np.sqrt(w0)
        w = w0 * np.where(np.abs(r) < huber, 1.0, huber / np.maximum(np.abs(r), 1e-9))
        if side_steps and it == 1:
            # a wall with a step (niche, change of tunnel section) shows a
            # consistent residual while the other side agrees: re-anchor it
            for side in (+1, -1):
                sel = np.flatnonzero(src == side)
                if len(sel) < 6 or not (src == -side).any():
                    continue
                res_side = val[sel] - sol[k[sel]]
                med = np.array([np.median(res_side[max(0, j - 3): j + 4]) for j in range(len(sel))])
                val[sel] -= np.where(np.abs(med) > 0.15, med, 0.0)
    try:
        cov_diag = np.diag(np.linalg.inv(A))
    except np.linalg.LinAlgError:  # pragma: no cover
        cov_diag = np.full(ng, 0.25)
    sig = np.interp(x, grid, np.sqrt(np.maximum(cov_diag, 1e-6)))
    sol = np.interp(x, grid, sol)
    good = w > 0.5 * w0
    supported = np.zeros(n, bool)
    supported[kk[good]] = True
    return sol, sig, supported


# --------------------------------------------------------------------------
# Track model


@dataclass
class TrackModel:
    s: np.ndarray             # slice centres (distance ahead), m
    c: np.ndarray             # lateral position of the track centre line
    theta: np.ndarray         # heading dc/ds
    z: np.ndarray             # height of the rail-head plane
    sig_c: np.ndarray
    sig_z: np.ndarray
    rail: np.ndarray          # bool: rails detected in the slice
    lat_obs: np.ndarray       # bool: some lateral measurement in the slice
    ver_obs: np.ndarray
    floor: np.ndarray         # measured floor height in the slice (nan if none)
    wall_left: np.ndarray     # tracked inner wall positions (nan if lost)
    wall_right: np.ndarray
    rails: np.ndarray         # (k, 5) raw rail detections: s, xl, zl, xr, zr
    observed_until: float     # path verified by measurements up to this distance
    rail_until: float
    calibration: dict = field(default_factory=dict)
    # A short, explicitly marked continuation through missing wall / rail
    # returns. Measurement flags still refer to this frame only.
    coast_frames: int = 0
    current_observed_until: float | None = None

    def at(self, f: np.ndarray):
        """Interpolated centre, rail height and uncertainty at distances ``f``."""
        c = np.interp(f, self.s, self.c)
        z = np.interp(f, self.s, self.z)
        sc = np.interp(f, self.s, self.sig_c)
        return c, z, sc

    def wall_offset(self, s: float, side: int, extend: float = 15.0) -> float | None:
        """Lateral position of the tracked wall relative to the centre line at
        distance ``s`` (left > 0, right < 0); the wall is continued along its
        last 10 m for up to ``extend`` metres past its last observation."""
        w = self.wall_left if side > 0 else self.wall_right
        ok = np.isfinite(w)
        if not ok.any():
            return None
        ws, wv = self.s[ok], w[ok]
        if s < ws[0]:
            return None
        if s <= ws[-1]:
            x = float(np.interp(s, ws, wv))
        elif s - ws[-1] <= extend:
            tail = ws > ws[-1] - 10.0
            slope = float(np.polyfit(ws[tail], wv[tail], 1)[0]) if tail.sum() >= 3 else 0.0
            x = float(wv[-1] + slope * (s - ws[-1]))
        else:
            return None
        return x - float(np.interp(s, self.s, self.c))

    def public(self, step: int = 1) -> dict:
        sl = slice(None, None, step)
        return {
            "s": np.round(self.s[sl], 2).tolist(),
            "c": np.round(self.c[sl], 3).tolist(),
            "z": np.round(self.z[sl], 3).tolist(),
            "sig_c": np.round(self.sig_c[sl], 3).tolist(),
            "observed_until": round(self.observed_until, 1),
            "rail_until": round(self.rail_until, 1),
            "coasted_frames": self.coast_frames,
            "current_observed_until": round(self.current_observed_until if self.current_observed_until is not None
                                            else self.observed_until, 1),
            "rails": np.round(self.rails, 3).tolist(),
        }


class TrackEstimator:
    """Stateful estimator: keeps calibration and the previous path between frames."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.calib = Calibration.from_config(cfg)
        self.edges = _slices(cfg)
        self.centres = self.edges.mean(axis=1)
        self.prev: TrackModel | None = None
        self.turnout_at: float | None = None
        self.offsets = {"L": None, "R": None, "F": None, "C": None}
        self.noisy_rail_frames = 0
        self.rail_noise_points = 0

    def reset(self) -> None:
        self.calib = Calibration.from_config(self.cfg)
        self.prev = None
        self.turnout_at = None
        self.offsets = {"L": None, "R": None, "F": None, "C": None}
        self.noisy_rail_frames = 0
        self.rail_noise_points = 0

    # -- calibration -------------------------------------------------------

    def _rails_near(self, lev: np.ndarray, c_guess: np.ndarray, until: float,
                    prior: TrackModel | None = None) -> np.ndarray:
        """Detect rail pairs slice by slice in the levelled cloud (vectorised profiles)."""
        t = self.cfg.track
        self.turnout_at = None
        edges = self.edges[self.edges[:, 1] <= until]
        if not len(edges):
            return np.zeros((0, 5))
        f = lev[:, 1]
        sel = (f >= edges[0, 0]) & (f < edges[-1, 1])
        pts = lev[sel]
        sidx = np.searchsorted(edges[:, 1], pts[:, 1], side="right")
        cg = c_guess[sidx]
        near = np.abs(pts[:, 0] - cg) < 2.6
        pts, sidx = pts[near], sidx[near]
        if len(pts) < 50:
            return np.zeros((0, 5))
        nsl = len(edges)
        zlow = _group_percentile(pts[:, 2], sidx, nsl, 0.03)
        band = pts[:, 2] < zlow[sidx] + 0.6
        pts, sidx = pts[band], sidx[band]
        col = np.floor((pts[:, 0] - LAT_MIN) / LAT_BIN).astype(np.int64)
        ok = (col >= 0) & (col < N_LAT)
        pts, sidx, col = pts[ok], sidx[ok], col[ok]
        prof = np.full(nsl * N_LAT, -np.inf, np.float32)
        np.maximum.at(prof, sidx * N_LAT + col, pts[:, 2])
        prof = prof.reshape(nsl, N_LAT).astype(np.float64)
        prof[~np.isfinite(prof)] = np.nan
        xs = LAT_MIN + (np.arange(N_LAT) + 0.5) * LAT_BIN
        out = []
        # The sensor rides on the track: the chain starts near the sensor axis
        # and is extended slice by slice from the rails already found (the
        # guess of the previous frame is only used to seed the first slice).
        seed = float(np.clip(c_guess[0], -0.6, 0.6))
        found_s: list = []
        found_c: list = []
        pairs: list = []
        for i in range(nsl):
            sm = 0.5 * (edges[i, 0] + edges[i, 1])
            if not found_s:
                pred, search = seed, 0.6
            else:
                k = min(len(found_s), 6)
                if k >= 3 and found_s[-1] - found_s[-k] > 3.0:
                    a, b = _short_line(found_s[-k:], found_c[-k:])
                    pred = a * sm + b
                else:
                    pred = found_c[-1]
                search = 0.3 if sm - found_s[-1] < 8.0 else 0.45
            bounds = None
            if self.noisy_rail_frames >= 2 and prior is not None and sm <= prior.rail_until+2.:
                sigma = float(np.interp(sm,prior.s,prior.sig_c))
                if sigma < .15:
                    margin = max(.25,3.*sigma)
                    bounds = (c_guess[i]-margin,c_guess[i]+margin)
            r = _rail_pair(xs, prof[i], pred, t.gauge, t.gauge_tolerance, search,
                           t.rail_min_prominence,bounds)
            if r is not None:
                out.append((sm, r[1], r[2], r[3], r[4]))
                found_s.append(sm)
                found_c.append(0.5 * (r[1] + r[3]))
            if 8.0 <= sm <= 60.0 and found_s:
                ref = found_c[-1]
                pairs.append((sm, [c - ref for c in _all_pairs(xs, prof[i], ref - 2.6, ref + 2.6, t.gauge,
                                                                t.gauge_tolerance, t.rail_min_prominence)]))
        self.turnout_at = _detect_turnout(pairs)
        return np.array(out).reshape(-1, 5)

    def _calibrate(self, rails: np.ndarray) -> None:
        """Update the sensor pose from rails detected in the levelled frame."""
        cfg = self.cfg
        if not cfg.sensor.auto_calibration:
            self.calib.valid = True
            return
        rails = rails[rails[:, 0] <= 32.0]
        if len(rails) < 6:
            return
        s = rails[:, 0]
        zc = 0.5 * (rails[:, 2] + rails[:, 4])
        gauge = rails[:, 3] - rails[:, 1]
        roll = np.median((rails[:, 4] - rails[:, 2]) / gauge)
        w = np.ones_like(s)
        for _ in range(3):
            pz = np.polyfit(s, zc, 1, w=w)
            r = np.abs(zc - np.polyval(pz, s))
            w = 1.0 / (1.0 + (r / 0.03) ** 2)
        slope, z0 = pz
        # residual pose in the levelled frame -> compose with the current pose
        c = self.calib
        new_pitch = math.tan(math.atan(c.pitch) + math.atan(slope))
        new_roll = math.tan(math.atan(c.roll) + math.atan(roll))
        new_height = c.height - z0
        c.update(new_height, new_pitch, new_roll, 1.0 if not c.valid else cfg.sensor.calibration_alpha)

    def _guess_centres(self, n: int) -> np.ndarray:
        if self.prev is not None:
            return np.interp(self.centres[:n], self.prev.s, self.prev.c)
        return np.full(n, self.cfg.sensor.default_lateral)

    @staticmethod
    def _fuse_walls(meas: list, ctr: "_KF", prior: float | None, bad: dict, woff: dict, lin: float | None = None,
                    drift_scale: float = 1.0):
        """Gate the wall-implied centre positions of one slice.

        The path bends smoothly, whereas a wall can jump (niche, platform end,
        change of the tunnel section). A side whose implied centre jumps away
        from the prediction is rejected; if that persists while the other side
        still agrees, its offset is re-anchored instead of bending the path.
        A smooth drift of the walls away from the path is the opposite case: the
        curve of the path is wrong, and the walls bend it (``drift_scale``
        inflates their variance close to the rails, which fix the path there).
        """
        if not meas:
            return []
        pred = ctr.X[0]
        ref = prior if prior is not None and abs(prior - pred) < 0.3 else pred
        gate = max(0.25, 3.0 * ctr.sig)
        ok = []
        rej = []
        for z, var, side, y in meas:
            (ok if abs(z - ref) < gate else rej).append((z, var, side, y))
        for z, var, side, y in ok:
            bad[side] = 0
        drift = []
        for z, var, side, y in rej:
            bad[side] += 1
            if bad[side] >= 3 and ok:
                # the other wall confirms the path: this side jumped
                woff[side] = [side * (y - pred)]
                bad[side] = 0
            elif bad[side] >= 3 and len(meas) == 1:
                if abs(z - ref) < 0.6:
                    # the only wall drifts away smoothly and persistently: the
                    # curve of the path is wrong, not the wall (a niche or a
                    # platform end is a jump); the wall bends the path
                    drift.append((z, 4.0 * drift_scale * var, side, y))
                else:
                    woff[side] = [side * (y - pred)]
                    bad[side] = 0
        if not ok and len(rej) == 2:
            (z1, v1, s1, y1), (z2, v2, s2, y2) = rej
            if abs(z1 - z2) < 0.35 and min(bad.values()) >= 2:
                # both walls agree with each other and not with the path: a
                # curve sharper / gentler than predicted
                drift.append((0.5 * (z1 + z2), 2.0 * drift_scale * max(v1, v2), 2, 0.0))
            elif min(bad.values()) >= 5:
                for z, var, side, y in rej:
                    woff[side] = [side * (y - pred)]
                    bad[side] = 0
        if drift:
            return [(z, var, side) for z, var, side, y in drift]
        if len(ok) == 2 and abs(ok[0][0] - ok[1][0]) < 0.35:
            return [(0.5 * (ok[0][0] + ok[1][0]), 0.5 * max(ok[0][1], ok[1][1]), 2)]
        if len(ok) == 2:
            # both walls seen but they disagree: one side widens (switch
            # chamber, niche, platform end). In a real curve both walls bend
            # together, so the side implying the least bending relative to the
            # path measured on the rails is kept.
            key = lin if lin is not None else ref
            best = min(ok, key=lambda m: abs(m[0] - key))
            return [(best[0], best[1], best[2])]
        return [(z, var, side) for z, var, side, y in ok]

    # -- main --------------------------------------------------------------

    def estimate(self, xyz: np.ndarray, prior: TrackModel | None = None) -> tuple[np.ndarray, TrackModel]:
        """Returns levelled points and the track model for one frame.

        ``prior`` is the previous path already transformed into the current
        frame (see :func:`gabarit.egomotion.propagate_track`).
        """
        cfg = self.cfg
        t = cfg.track
        edges = self.edges
        centres = self.centres
        n = len(edges)
        guess = np.interp(centres, prior.s, np.nan_to_num(prior.c)) if prior is not None else self._guess_centres(n)
        if not self.calib.valid:
            # first frame: calibrate before estimating anything else
            self._calibrate(self._rails_near(self.calib.level(xyz), guess, 32.0))
        lev = self.calib.level(xyz)
        # Scattered unsupported returns in the previously clear near corridor
        # indicate disturbed measurements. Require persistence: a single sparse
        # fixture must not change rail selection. Keep real rail returns and
        # constrain their branch only against a sufficiently certain prior.
        self.rail_noise_points = 0
        if prior is not None and cfg.filter.enabled:
            f = lev[:,1]
            d = lev[:,0]-np.interp(f,prior.s,prior.c)
            h = lev[:,2]-np.interp(f,prior.s,prior.z)
            region = ((f >= t.start) & (f < min(60.,prior.observed_until))
                      & (np.abs(d) < cfg.envelope.width/2+.2) & (h > .6) & (h < 2.5))
            self.rail_noise_points = int(np.count_nonzero(~support_mask(lev[region],cfg.filter)))
        if self.rail_noise_points >= 16:
            self.noisy_rail_frames = min(5,self.noisy_rail_frames+1)
        else:
            self.noisy_rail_frames = max(0,self.noisy_rail_frames-1)
        rails = self._rails_near(lev, guess, t.rail_search_max,prior)
        rail_by_slice = {}
        for r in rails:
            rail_by_slice[int(np.searchsorted(edges[:, 1], r[0], side="right"))] = r

        # initial state from the first rails (or the prior / defaults)
        if len(rails) >= 3:
            k = min(len(rails), 12)
            pc = np.polyfit(rails[:k, 0], 0.5 * (rails[:k, 1] + rails[:k, 3]), 1)
            c0 = np.polyval(pc, edges[0, 0])
            th0 = pc[0]
        elif prior is not None:
            c0, th0 = prior.c[0], prior.theta[0]
        else:
            c0, th0 = cfg.sensor.default_lateral, 0.0
        k0 = 0.0
        if prior is not None:
            k0 = float(np.clip(np.nan_to_num(np.gradient(prior.theta, prior.s)[:10].mean()), -0.005, 0.005))
        ctr = _KF([c0, th0, k0], [0.03, 0.004, 0.0005], t.curvature_noise)
        ver = _KF([0.0, 0.0, 0.0], [0.03, 0.002, 0.0002], t.curvature_noise * 0.3)

        f_all = lev[:, 1]
        pre = (f_all >= edges[0, 0]) & (f_all < edges[-1, 1]) & (lev[:, 2] > -3.0) & (lev[:, 2] < 7.0)
        pts = lev[pre]
        dx = np.abs(pts[:, 0] - np.interp(pts[:, 1], centres, guess))
        keep = dx < 6.5
        # the near field is still dense: thin it (percentiles are unaffected)
        dense = pts[:, 1] < 25.0
        keep &= ~dense | (np.arange(len(pts)) % 2 == 0)
        pts = pts[keep]
        sidx = np.searchsorted(edges[:, 1], pts[:, 1], side="right").astype(np.int32)
        order = np.argsort(sidx, kind="stable")
        pts = pts[order]
        bounds = np.searchsorted(sidx[order], np.arange(n + 1))
        bounds = np.column_stack((bounds[:-1], bounds[1:]))

        prior_c = prior_z = prior_sc = prior_sz = None
        if prior is not None:
            prior_c = np.interp(centres, prior.s, prior.c, right=np.nan)
            prior_z = np.interp(centres, prior.s, prior.z, right=np.nan)
            prior_sc = np.interp(centres, prior.s, prior.sig_c)
            prior_sz = np.interp(centres, prior.s, prior.sig_z)
            # only the part of the previous path that was actually observed
            beyond = centres > prior.observed_until + 5.0
            prior_c[beyond] = np.nan
            prior_z[beyond] = np.nan

        rec_lat: list = []   # (slice, value, variance, source: 0 rails, +1/-1 walls)
        rec_ver: list = []   # (slice, value, variance, source: 0 rails, 2 floor, 3 ceiling)
        walls: dict[int, _KF | None] = {+1: None, -1: None}
        bad_side = {+1: 0, -1: 0}
        anchor = None   # (s, centre, heading) at the last slice with rails
        woff: dict[int, list] = {+1: [], -1: []}
        flo: _KF | None = None
        cei: _KF | None = None
        foff: list = []
        coff: list = []
        for key, lst in (("L", woff[+1]), ("R", woff[-1]), ("F", foff), ("C", coff)):
            if self.offsets[key] is not None:
                lst.append(self.offsets[key])
        wb0, wb1 = t.wall_band
        res = np.zeros((n, 12))
        s_prev = edges[0, 0]
        last_lat = edges[0, 0]
        last_rail = 0.0
        for i in range(n):
            sm = centres[i]
            ds = sm - s_prev
            s_prev = sm
            ctr.predict(ds)
            ver.predict(ds)
            for w in (walls[+1], walls[-1], flo, cei):
                if w is not None and w.alive:
                    w.predict(ds)
            a, b = bounds[i]
            q = pts[a:b]
            c = ctr.X[0]
            zg = ver.X[0]
            rail = rail_by_slice.get(i)
            if rail is not None:
                rc = 0.5 * (rail[1] + rail[3])
                rz = 0.5 * (rail[2] + rail[4])
                if abs(rc - c) > max(0.35, 4 * ctr.sig) and i > 3:
                    rail = None
            lat_meas = []
            ver_meas = []
            floor_here = np.nan
            if len(q) >= 3:
                x = q[:, 0]
                h = q[:, 2] - zg
                band = (h > wb0) & (h < wb1)
                side_d = x - c
                for side in (+1, -1):
                    w = walls[side]
                    if w is None or not w.alive:
                        # (re)acquire the nearest structure on this side
                        sel = band & (side * side_d > 1.3) & (side * side_d < (5.0 if rail is not None else 3.5))
                        if sel.sum() >= 3 and (rail is not None or sm < 120):
                            y = _pct(x[sel], 15 if side > 0 else 85)
                            walls[side] = _KF([y, ctr.X[1], ctr.X[2]], [0.05, 0.003, 0.0003], t.curvature_noise)
                            if rail is not None:
                                woff[side] = [side * (y - rc)]
                            elif not woff[side]:
                                woff[side] = [side * (y - c)]
                            else:
                                woff[side] = [side * (y - c)]
                        continue
                    win = max(t.wall_window, 3 * w.sig)
                    sel = band & (np.abs(x - w.X[0]) < win) & (side * side_d > t.wall_min_offset)
                    if sel.sum() >= 2:
                        y = float(_pct(x[sel], 15 if side > 0 else 85))
                        w.update(y, 0.06 ** 2)
                        w.miss = 0.0
                        if rail is not None:
                            woff[side].append(side * (y - rc))
                            del woff[side][:-25]
                        elif woff[side]:
                            lat_meas.append((y - side * float(_med(woff[side])), 0.09 ** 2, side, y))
                    else:
                        w.miss += ds
                        if w.miss > t.wall_lost_after:
                            w.alive = False
                # floor under the path
                cen = np.abs(side_d) < 1.6
                if flo is None or not flo.alive:
                    sel = cen & (h > -0.9) & (h < 0.4)
                    if sel.sum() >= 4 and (rail is not None or flo is None and sm < 30):
                        v = float(_pct(q[sel, 2], 25))
                        flo = _KF([v, ver.X[1], 0.0], [0.05, 0.002, 0.0002], t.curvature_noise * 0.3)
                        if rail is not None:
                            foff[:] = [v - rz]
                else:
                    win = max(0.25, 3 * flo.sig)
                    sel = cen & (np.abs(q[:, 2] - flo.X[0]) < win)
                    if sel.sum() >= 3:
                        v = float(_pct(q[sel, 2], 25))
                        floor_here = v
                        flo.update(v, 0.05 ** 2)
                        flo.miss = 0.0
                        if rail is not None:
                            foff.append(v - rz)
                            del foff[:-25]
                        elif foff:
                            ver_meas.append((v - float(_med(foff)), 0.06 ** 2))
                            rec_ver.append((i, v - float(_med(foff)), 0.06 ** 2, 2))
                    else:
                        flo.miss += ds
                        if flo.miss > 20:
                            flo.alive = False
                # ceiling above the path
                cen2 = np.abs(side_d) < 1.2
                if cei is None or not cei.alive:
                    sel = cen2 & (h > 2.4) & (h < 6.5)
                    if sel.sum() >= 4 and sm < 60:
                        v = float(_med(q[sel, 2]))
                        cei = _KF([v, ver.X[1], 0.0], [0.08, 0.002, 0.0002], t.curvature_noise * 0.3)
                        coff[:] = [v - (rz if rail is not None else zg)]
                else:
                    win = max(0.3, 3 * cei.sig)
                    sel = cen2 & (np.abs(q[:, 2] - cei.X[0]) < win)
                    if sel.sum() >= 3:
                        v = float(_med(q[sel, 2]))
                        cei.update(v, 0.06 ** 2)
                        cei.miss = 0.0
                        if rail is not None:
                            coff.append(v - rz)
                            del coff[:-25]
                        elif coff:
                            ver_meas.append((v - float(_med(coff)), 0.10 ** 2))
                            rec_ver.append((i, v - float(_med(coff)), 0.10 ** 2, 3))
                    else:
                        cei.miss += ds
                        if cei.miss > 20:
                            cei.alive = False
            used_lat = False
            pc_i = prior_c[i] if prior_c is not None and np.isfinite(prior_c[i]) else None
            if rail is not None:
                ctr.update(rc, 0.03 ** 2)
                ver.update(rz, 0.02 ** 2)
                used_lat = True
                last_rail = sm
                anchor = (sm, ctr.X[0], ctr.X[1])
                rec_lat.append((i, rc, (0.03 + 0.002 * max(sm - 20.0, 0.0)) ** 2, 0))
                rec_ver.append((i, rz, 0.02 ** 2, 0))
            else:
                lin = anchor[1] + anchor[2] * (sm - anchor[0]) if anchor is not None else None
                if lin is not None and len(lat_meas) == 2 and abs(lat_meas[0][0] - lat_meas[1][0]) >= 0.35:
                    # disagreeing sides: keep the one closest to the straight
                    # continuation of the rails (may be outside the normal gate)
                    best = min(lat_meas, key=lambda m: abs(m[0] - lin))
                    other = [m for m in lat_meas if m is not best][0]
                    bad_side[other[2]] += 1
                    if abs(best[0] - ctr.X[0]) < max(1.0, 3 * ctr.sig):
                        lat_meas = [best]
                gap = sm - anchor[0] if anchor is not None else 99.0
                scale = float(np.interp(gap, (20.0, 40.0), (4.0, 1.0)))
                for zc, var, side in self._fuse_walls(lat_meas, ctr, pc_i, bad_side, woff, lin, scale):
                    ctr.update(zc, var)
                    rec_lat.append((i, zc, var, side))
                    used_lat = True
                gv = max(0.3, 3 * ver.sig)
                for zv, var in ver_meas:
                    if abs(zv - ver.X[0]) < gv:
                        ver.update(zv, var)
            used_ver = rail is not None or bool(ver_meas)
            if pc_i is not None and not used_lat:
                # the previous path only fills slices without own measurements
                var = (prior_sc[i] ** 2 + 0.03 ** 2) / max(t.temporal_weight, 1e-3)
                if abs(pc_i - ctr.X[0]) < max(0.8, 3 * math.sqrt(ctr.P[0][0] + var)):
                    ctr.update(pc_i, var)
            if prior_z is not None and np.isfinite(prior_z[i]) and rail is None and not ver_meas:
                varz = (prior_sz[i] ** 2 + 0.02 ** 2) / max(t.temporal_weight, 1e-3)
                if abs(prior_z[i] - ver.X[0]) < 0.5:
                    ver.update(prior_z[i], varz)
            if used_lat:
                last_lat = sm
            res[i] = (sm, ctr.X[0], ctr.X[1], ver.X[0], ctr.sig, ver.sig, rail is not None, used_lat, used_ver,
                      floor_here,
                      walls[+1].X[0] if walls[+1] is not None and walls[+1].alive else np.nan,
                      walls[-1].X[0] if walls[-1] is not None and walls[-1].alive else np.nan)

        for key, lst in (("L", woff[+1]), ("R", woff[-1]), ("F", foff), ("C", coff)):
            self.offsets[key] = float(_med(lst)) if lst else None
        # global robust smoothing of the centre line and of the vertical profile
        c_fit, sc_fit, obs_c = _smooth_profile(centres, rec_lat, prior_c, prior_sc, res[:, 1],
                                               lam=t.smooth_lateral, huber=2.0, side_steps=False)
        z_fit, sz_fit, obs_z = _smooth_profile(centres, rec_ver, prior_z, prior_sz, res[:, 3],
                                               lam=t.smooth_vertical, huber=2.0, side_steps=False)
        if prior is not None and not _plausible(centres, c_fit):
            self.prev = None
            return self.estimate(xyz, None)
        observed = centres[obs_c]
        last_lat = float(observed.max()) if len(observed) else float(edges[0, 0])
        current_observed_until = last_lat
        coast_frames = 0
        # A sparse frame must not collapse an already measured corridor to the
        # first slice. Retain the motion-compensated previous tail for a bounded
        # number of frames, increasing uncertainty and retaining fresh flags.
        # An unobserved startup path is never promoted into a measured path.
        if (prior is not None and prior.observed_until > last_lat + 2.0
                and prior.coast_frames < t.hold_missing_frames):
            old_c = np.interp(centres, prior.s, prior.c, right=np.nan)
            old_z = np.interp(centres, prior.s, prior.z, right=np.nan)
            retained = (centres > last_lat) & (centres <= prior.observed_until) & np.isfinite(old_c)
            if retained.any():
                weight = np.clip((centres - last_lat) / 8.0, 0.0, 1.0) if len(observed) else np.ones(n)
                c_fit[retained] += weight[retained] * (old_c[retained] - c_fit[retained])
                if not rec_ver:
                    ok_z = retained & np.isfinite(old_z)
                    z_fit[ok_z] = old_z[ok_z]
                sc_fit[retained] = np.maximum(sc_fit[retained], np.interp(centres, prior.s, prior.sig_c)[retained] + 0.03)
                sz_fit[retained] = np.maximum(sz_fit[retained], np.interp(centres, prior.s, prior.sig_z)[retained] + 0.02)
                last_lat = float(centres[retained].max())
                coast_frames = prior.coast_frames + 1
        model = TrackModel(
            s=res[:, 0], c=c_fit, theta=np.gradient(c_fit, centres), z=z_fit, sig_c=sc_fit, sig_z=sz_fit,
            rail=res[:, 6] > 0, lat_obs=obs_c, ver_obs=obs_z, floor=res[:, 9],
            wall_left=res[:, 10], wall_right=res[:, 11], rails=rails,
            observed_until=last_lat, rail_until=float(last_rail), calibration=self.calib.public(),
            coast_frames=coast_frames, current_observed_until=current_observed_until,
        )
        self.prev = model
        # pose update for the next frame (EMA, uses this frame's rails)
        self._calibrate(rails)
        return lev, model
