"""Train ego-motion from the lidar alone.

The tunnel is almost invariant along the track, but wall fixtures, cable
brackets, lamps and tubing joints make it textured along the distance. The
forward displacement between two frames is found by correlating bird's-eye
occupancy images of these structures (1-DOF registration). A constant-velocity
Kalman filter smooths the result and resolves the ambiguity of periodic
structures. The estimate is used to predict obstacle tracks, to carry the path
from frame to frame and to show speed / time to reach an obstacle.
"""

from __future__ import annotations

import math

import numpy as np

from gabarit.track import TrackModel

F_MIN, F_MAX, F_BIN = 4.0, 48.0, 0.05
X_MIN, X_MAX, X_BIN = -6.0, 6.0, 0.25
NF = int(round((F_MAX - F_MIN) / F_BIN))
NX = int(round((X_MAX - X_MIN) / X_BIN))


def occupancy(lev: np.ndarray) -> np.ndarray:
    x, f, z = lev[:, 0], lev[:, 1], lev[:, 2]
    sel = (f >= F_MIN) & (f < F_MAX) & (x >= X_MIN) & (x < X_MAX) & (z > 0.3) & (z < 4.5)
    fi = ((f[sel] - F_MIN) / F_BIN).astype(np.int64)
    xi = ((x[sel] - X_MIN) / X_BIN).astype(np.int64)
    img = np.bincount(fi * NX + xi, minlength=NF * NX).reshape(NF, NX).astype(np.float32)
    img = np.log1p(img)
    # light smoothing along the distance
    k = np.array([0.25, 0.5, 0.25], np.float32)
    img[1:-1] = k[0] * img[:-2] + k[1] * img[1:-1] + k[2] * img[2:]
    # high-pass along the track: walls are invariant to the shift and would
    # make the correlation flat; only the texture (brackets, joints) is kept
    w = 41  # 2 m
    c = np.cumsum(np.vstack([np.zeros((1, NX), np.float32), img]), axis=0)
    lo = np.clip(np.arange(NF) - w // 2, 0, NF)
    hi = np.clip(np.arange(NF) + w // 2 + 1, 0, NF)
    mean = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return img - mean


def _score(prev: np.ndarray, cur: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Overlap-normalised correlation for all shifts, without rescanning images.

    FFT supplies cross products; prefix sums supply the mean and energy of
    each overlapping region. Global centring first avoids cancellation for
    nearly uniform surfaces and does not change a centred correlation.
    """
    a = np.asarray(prev, dtype=np.float64)
    b = np.asarray(cur, dtype=np.float64)
    a, b = a-a.mean(), b-b.mean()
    nfft = 1 << (2*NF-1).bit_length()
    cross = np.fft.irfft((np.fft.rfft(a, n=nfft, axis=0) *
                         np.conj(np.fft.rfft(b, n=nfft, axis=0))).sum(axis=1), n=nfft)
    def prefix(image):
        return (np.r_[0., np.cumsum(image.sum(axis=1))],
                np.r_[0., np.cumsum(np.square(image).sum(axis=1))])
    sa, qa = prefix(a)
    sb, qb = prefix(b)
    valid = NF-np.abs(shifts) >= NF//3
    sh = shifts[valid]
    count = (NF-np.abs(sh))*NX
    a0, a1 = np.maximum(sh, 0), NF+np.minimum(sh, 0)
    b0, b1 = np.maximum(-sh, 0), NF-np.maximum(sh, 0)
    suma, sumb = sa[a1]-sa[a0], sb[b1]-sb[b0]
    va = np.maximum(qa[a1]-qa[a0]-suma*suma/count, 0.)
    vb = np.maximum(qb[b1]-qb[b0]-sumb*sumb/count, 0.)
    out = np.full(len(shifts), -np.inf)
    out[valid] = (cross[sh % nfft]-suma*sumb/count) / (np.sqrt(va*vb)+1e-6)
    return out


class EgoMotion:
    def __init__(self, max_speed: float = 25.0):
        self.max_speed = max_speed
        self.prev_img: np.ndarray | None = None
        self.prev_stamp: float | None = None
        self.v = 0.0
        self.P = 25.0
        self.initialised = False
        self.last_shift = 0.0
        self.quality = 0.0
        self.disagree = 0

    def reset(self) -> None:
        self.__init__(self.max_speed)

    def update(self, lev: np.ndarray, stamp: float) -> tuple[float, float]:
        """Returns (displacement since previous frame [m], speed [m/s])."""
        img = occupancy(lev)
        if self.prev_img is None or self.prev_stamp is None:
            self.prev_img, self.prev_stamp = img, stamp
            return 0.0, self.v
        dt = stamp - self.prev_stamp
        if not (0.02 < dt < 1.0):
            dt = 0.1
        # prediction
        self.P += (1.5 * dt) ** 2
        pred = self.v * dt
        lo, hi = -0.3, self.max_speed * dt + 0.2
        shifts = np.arange(int(math.floor(lo / F_BIN)), int(math.ceil(hi / F_BIN)) + 1)
        sc = _score(self.prev_img, img, shifts)
        # mild preference for the predicted motion resolves periodic structures
        if self.initialised:
            sig = max(0.15, 3 * math.sqrt(self.P) * dt)
            pen = 0.3 * np.minimum(((shifts * F_BIN - pred) / sig) ** 2, 9.0)
        else:
            pen = 0.0
        k = int(np.argmax(sc - pen))
        # the motion prior must not lock the filter on a wrong speed: when the
        # data alone clearly prefer another shift for several frames running,
        # the uncertainty is reset and the filter re-acquires the motion
        k_raw = int(np.argmax(sc))
        if self.initialised and sc[k_raw] > 0.35 and sc[k_raw] - sc[k] > 0.1:
            self.disagree += 1
            if self.disagree >= 3:
                self.P = 25.0
                k = k_raw
                self.disagree = 0
        else:
            self.disagree = 0
        best = sc[k]
        sh = float(shifts[k])
        if 0 < k < len(sc) - 1 and np.isfinite(sc[k - 1]) and np.isfinite(sc[k + 1]):
            den = sc[k - 1] - 2 * sc[k] + sc[k + 1]
            if den < 0:
                sh += 0.5 * (sc[k - 1] - sc[k + 1]) / den
        disp = sh * F_BIN
        self.quality = float(best)
        if best > 0.2:
            meas_v = disp / dt
            R = (0.06 / dt) ** 2 / max(best, 0.2)
            K = self.P / (self.P + R)
            self.v += K * (meas_v - self.v)
            self.P *= (1 - K)
            self.initialised = True
        self.v = float(np.clip(self.v, -2.0, self.max_speed))
        if abs(self.v) < 0.15:
            self.v = 0.0 if self.v * self.v < 0.02 else self.v
        self.prev_img, self.prev_stamp = img, stamp
        self.last_shift = self.v * dt
        return self.last_shift, self.v


def propagate_track(model: TrackModel, delta: float, sigma_growth: float = 0.03) -> TrackModel:
    """Express the previous frame's path in the current frame after moving ``delta``."""
    s, c, z = model.s, model.c, model.z
    if abs(delta) < 1e-3:
        out = TrackModel(**{**model.__dict__})
        out.sig_c = model.sig_c + sigma_growth * 0.3
        out.sig_z = model.sig_z + sigma_growth * 0.3
        if model.sig_z_path is not None:
            out.sig_z_path = model.sig_z_path + sigma_growth * 0.3
        return out
    th = np.gradient(c, s)
    ox = float(np.interp(delta, s, c) - np.interp(0.0, s, c, left=c[0]))
    alpha = float(np.interp(delta, s, th) - th[0])
    xr = c - ox
    fr = s - delta
    ca, sa = math.cos(alpha), math.sin(alpha)
    x_new = xr * ca - fr * sa
    f_new = xr * sa + fr * ca
    zs = np.gradient(z, s)
    beta = float(np.interp(delta, s, zs) - zs[0])
    z_new = (z - float(np.interp(delta, s, z))) - beta * fr
    ok = f_new > 0
    grid = model.s
    out = TrackModel(**{**model.__dict__})
    out.c = np.interp(grid, f_new[ok], x_new[ok], left=np.nan, right=np.nan)
    out.z = np.interp(grid, f_new[ok], z_new[ok], left=np.nan, right=np.nan)
    out.theta = np.gradient(np.nan_to_num(out.c), grid)
    out.sig_c = np.interp(grid, f_new[ok], model.sig_c[ok]) + sigma_growth
    out.sig_z = np.interp(grid, f_new[ok], model.sig_z[ok]) + sigma_growth
    if model.sig_z_path is not None:
        out.sig_z_path = np.interp(grid, f_new[ok], model.sig_z_path[ok]) + sigma_growth
    # A retained horizon travels towards the train as well. Leaving the old
    # distances here would extend an unobserved corridor on every dropout.
    out.observed_until = max(float(grid[0]), model.observed_until - delta)
    out.rail_until = max(0.0, model.rail_until - delta)
    if model.current_observed_until is not None:
        out.current_observed_until = max(float(grid[0]), model.current_observed_until - delta)
    # fill the start (behind the old horizon start) by the first valid values
    for arr in (out.c, out.z):
        valid = np.isfinite(arr)
        if valid.any():
            first = np.argmax(valid)
            arr[:first] = arr[first]
    return out
