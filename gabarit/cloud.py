"""Conversion of PointCloud2 records into working arrays.

Working frame of the sensor ("S"): x — lateral (left positive), f — forward,
z — up. The organized structure (ring / column) is kept for ray-casting of
synthetic obstacles and for per-beam statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gabarit.rosbag import RawCloud

_AXES = {
    "-y": ((0, 1.0), (1, -1.0)),   # lateral = +x, forward = -y
    "+y": ((0, -1.0), (1, 1.0)),
    "+x": ((1, -1.0), (0, 1.0)),   # REP-103: lateral = +y (left), forward = +x
    "-x": ((1, 1.0), (0, -1.0)),
}


@dataclass
class Cloud:
    stamp: float
    frame_id: str
    xyz: np.ndarray        # (N, 3) float32: lateral, forward, up (sensor frame)
    intensity: np.ndarray  # (N,) float32
    ring: np.ndarray       # (N,) int16, -1 if unknown
    column: np.ndarray     # (N,) int32 index of the firing column, -1 if unknown
    raw_count: int
    rings: int
    columns: int
    # first return of a firing whose second return lies >= 1.5 m farther: the
    # beam went partly through something (glass, film, a thin edge)
    split: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "Cloud":
        return Cloud(self.stamp, self.frame_id, self.xyz[mask], self.intensity[mask], self.ring[mask],
                     self.column[mask], self.raw_count, self.rings, self.columns,
                     None if self.split is None else self.split[mask])


def working_set(cloud: Cloud, max_range: float = 205.0) -> np.ndarray:
    """Points worth processing: the region ahead, with the redundant density
    of the near field removed.

    The sensor works in dual-return mode: neighbouring columns are the two
    returns of the same firing and are identical on solid surfaces. Up to 40 m
    one return is kept, closer than 12 m every other firing as well - hundreds
    of points per square metre remain there.
    """
    x, f, z = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    keep = (f > -1.0) & (f < max_range) & (np.abs(x) < 25.0) & (z > -4.0) & (z < 9.0)
    if cloud.columns:
        col = cloud.column
        keep &= (f >= 40.0) | (col % 2 == 0)
        keep &= (f >= 12.0) | (col % 4 == 0)
    else:
        # unorganized cloud: firing columns from the azimuth (0.1 deg step);
        # closer than 12 m every 4th one is kept (<= 8 cm apart there)
        near = keep & (f < 12.0)
        az = np.arctan2(x[near], f[near])
        col = np.floor(az * (1800.0 / np.pi)).astype(np.int64)
        keep[np.flatnonzero(near)[col % 4 != 0]] = False
    return keep


def estimate_rings(x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray,
                   bin_deg: float = 0.01, min_sep_deg: float = 0.07) -> np.ndarray:
    """Beam index of every point from its elevation angle (-1 if unknown).

    A mechanical lidar fires each beam at a fixed elevation, so elevations of
    one frame form narrow peaks (Pandar128: 0.125 deg apart in the centre,
    0.5-1 deg at the edges). Peaks of the elevation histogram are the beams;
    every point goes to the nearest one, boundaries halfway between peaks.
    """
    out = np.full(len(x), -1, np.int16)
    horiz = np.sqrt(x * x + y * y)
    use = valid & (horiz > 2.0)
    if use.sum() < 1000:
        return out
    # tan(elevation) instead of the angle: monotonic and much cheaper
    with np.errstate(divide="ignore", invalid="ignore"):
        el = z / horiz
    step = np.float32(np.tan(np.radians(bin_deg)))
    eu = el[use]
    lo, hi = float(eu.min()) - step, float(eu.max()) + step
    nb = int(np.ceil((hi - lo) / step)) + 1
    hist = np.bincount(((eu - lo) / step).astype(np.int64), minlength=nb).astype(np.float64)
    sm = np.convolve(hist, np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, mode="same")
    # local maxima within +-min_sep, ignoring empty noise
    w = max(1, int(round(min_sep_deg / bin_deg / 2)))
    pad = np.pad(sm, w, constant_values=0.0)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)
    floor = max(5.0, 0.02 * float(np.percentile(sm[sm > 0], 90)))
    peaks = np.flatnonzero((sm >= win) & (sm > floor))
    if len(peaks) < 2:
        return out
    # plateaus give neighbouring equal maxima: keep one per min_sep
    keep = np.concatenate(([True], np.diff(peaks) * bin_deg >= min_sep_deg))
    centers = lo + (peaks[keep] + 0.5) * step
    edges = ((centers[1:] + centers[:-1]) / 2).astype(np.float32)
    idx = np.flatnonzero(valid)
    out[idx] = np.searchsorted(edges, el[idx]).astype(np.int16)
    return out


def dual_return_split(r2: np.ndarray, valid: np.ndarray, rings: int, columns: int, gap: float = 1.5) -> np.ndarray:
    """Points that are the nearer of the two returns of one firing with the
    other return at least ``gap`` metres farther.

    In the dual-return mode neighbouring columns (2k, 2k+1) are the two
    returns of the same firing and coincide on solid surfaces. A transparent
    or semi-transparent obstacle (glass, film) reflects a weak part of the
    pulse and lets the rest through to the tunnel behind it: the returns split.
    """
    n = len(r2)
    out = np.zeros(n, bool)
    if columns < 2 or n != rings * columns:
        return out
    r = np.sqrt(np.where(valid, r2, np.inf)).reshape(columns, rings)
    pairs = columns // 2
    a, b = r[0:2 * pairs:2], r[1:2 * pairs:2]
    ok = np.isfinite(a) & np.isfinite(b)
    diff = np.where(ok, b - np.where(ok, a, 0.0), 0.0) if ok.any() else np.zeros_like(a)
    near_a = ok & (diff >= gap)
    near_b = ok & (-diff >= gap)
    m = np.zeros((columns, rings), bool)
    m[0:2 * pairs:2] = near_a
    m[1:2 * pairs:2] = near_b
    return m.reshape(-1)


def from_raw(raw: RawCloud, forward_axis: str = "-y", min_range: float = 1.0) -> Cloud:
    arr = raw.points
    names = arr.dtype.names or ()
    for key in ("x", "y", "z"):
        if key not in names:
            raise ValueError(f"point cloud has no field {key}")
    coords = [np.asarray(arr[k], dtype=np.float32) for k in ("x", "y", "z")]
    (li, ls), (fi, fs) = _AXES[forward_axis]
    lat = coords[li] * ls
    fwd = coords[fi] * fs
    up = coords[2]
    n = len(arr)
    r2 = lat * lat + fwd * fwd + up * up
    valid = np.isfinite(r2) & (r2 > min_range * min_range)
    if "ring" in names:
        ring_all = np.asarray(arr["ring"], dtype=np.int16)
        rings = int(ring_all.max()) + 1 if n else 0
        organized = rings > 0 and n % rings == 0 and np.array_equal(ring_all[:rings], np.arange(rings))
    else:
        # unorganized cloud without the ring field (e.g. x, y, z, intensity only):
        # the beam index is recovered from the elevation angle
        ring_all = estimate_rings(coords[0], coords[1], coords[2], valid)
        rings = int(ring_all.max()) + 1 if n and ring_all.max() >= 0 else 0
        organized = False
    columns = n // rings if organized else 0
    col_all = (np.arange(n, dtype=np.int32) // rings) if organized else np.full(n, -1, np.int32)
    split = dual_return_split(r2, valid, rings, columns) if organized else np.zeros(n, bool)
    xyz = np.column_stack((lat[valid], fwd[valid], up[valid])).astype(np.float32, copy=False)
    if "intensity" in names:
        inten = np.asarray(arr["intensity"], dtype=np.float32)[valid]
    else:
        inten = np.zeros(len(xyz), np.float32)
    return Cloud(raw.stamp, raw.frame_id, xyz, inten, ring_all[valid], col_all[valid], n, rings, columns,
                 split[valid])
