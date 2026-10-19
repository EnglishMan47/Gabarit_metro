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
    "+x": ((1, 1.0), (0, 1.0)),   # REP-103: lateral = +y (left), forward = +x
    "-x": ((1, -1.0), (0, -1.0)),
}


def axis_transform(forward_axis: str = "-y", up_axis: str = "+z") -> np.ndarray:
    """Raw xyz -> working (left, forward, up) matrix.

    For row vectors, ``working = raw_xyz @ matrix.T``; the inverse is
    ``raw_xyz = working @ matrix``. Left is up cross forward, so a native
    Falcon Prime (+z forward, +x up) maps to (-y, +z, +x).
    """
    def vector(axis: str) -> np.ndarray:
        if axis not in {s + a for s in "+-" for a in "xyz"}:
            raise ValueError(f"invalid signed axis {axis!r}; expected +x, -x, +y, -y, +z, -z")
        v = np.zeros(3, np.float32)
        v["xyz".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
        return v
    forward, up = vector(forward_axis), vector(up_axis)
    if np.dot(forward, up) != 0:
        raise ValueError("forward_axis and up_axis must be perpendicular")
    return np.stack((np.cross(up, forward), forward, up))


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
    scan_pattern: str = "mechanical"
    dual_returns: bool = True
    source_index: np.ndarray | None = None  # optional raw-return identity for independent evaluation

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "Cloud":
        return Cloud(self.stamp, self.frame_id, self.xyz[mask], self.intensity[mask], self.ring[mask],
                     self.column[mask], self.raw_count, self.rings, self.columns,
                     None if self.split is None else self.split[mask],
                     self.scan_pattern, self.dual_returns,
                     None if self.source_index is None else self.source_index[mask])


def working_set(cloud: Cloud, max_range: float = 205.0) -> np.ndarray:
    """Points worth processing: the region ahead, with the redundant density
    of the near field removed.

    The sensor works in dual-return mode: neighbouring columns are the two
    returns of the same firing and are identical on solid surfaces. Up to 40 m
    one return is kept, closer than 12 m every other firing as well - hundreds
    of points per square metre remain there.
    """
    return _working_mask(cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2], cloud.column,
                         cloud.columns, cloud.scan_pattern, cloud.dual_returns, max_range)


def _working_mask(x, f, z, col, columns, scan_pattern, dual_returns, max_range):
    """Shared by the public mask and conversion, before allocating xyz."""
    keep = (f > -1.0) & (f < max_range) & (np.abs(x) < 25.0) & (z > -4.0) & (z < 9.0)
    if scan_pattern == "mirror":
        # Mirror scan rows and returns are not mechanical adjacent columns.
        # Preserve this sparse, nonuniform ROI sampling, including near objects.
        return keep
    if columns and dual_returns:
        keep &= (f >= 40.0) | (col % 2 == 0)
        keep &= (f >= 12.0) | (col % 4 == 0)
    elif columns:
        keep &= (f >= 12.0) | (col % 2 == 0)
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


def from_raw(raw: RawCloud, forward_axis: str = "-y", min_range: float = 1.0, *,
             up_axis: str = "+z", scan_pattern: str = "mechanical", dual_returns: bool = True,
             elevation_resolution_deg: float = 0.125, working_range: float | None = None,
             preserve_source_index: bool = False) -> Cloud:
    """Convert raw xyz using an explicit signed forward/up pair.

    The default preserves the supplied Pandar scan layout. Mirror scanners
    instead use measured elevation bands as geometric support, without
    inferring fixed laser rings or pairing neighbouring points as returns.
    """
    if scan_pattern not in ("mechanical", "mirror"):
        raise ValueError("scan_pattern must be mechanical or mirror")
    if not np.isfinite(min_range) or min_range < 0:
        raise ValueError("min_range must be finite and nonnegative")
    if not np.isfinite(elevation_resolution_deg) or not 0 < elevation_resolution_deg <= 10:
        raise ValueError("elevation_resolution_deg must be within (0, 10]")
    arr = raw.points
    names = arr.dtype.names or ()
    for key in ("x", "y", "z"):
        if key not in names:
            raise ValueError(f"point cloud has no field {key}")
    def scalar(key: str, dtype: type) -> np.ndarray:
        values = np.asarray(arr[key], dtype=dtype)
        if values.shape != (len(arr),):
            raise ValueError(f"point field {key} must have count=1")
        return values
    coords = [scalar(k, np.float32) for k in ("x", "y", "z")]
    transform = axis_transform(forward_axis, up_axis)
    # Every row is one signed axis: avoid a full matrix multiplication/copy.
    li, fi, ui = np.abs(transform).argmax(axis=1)
    ls, fs, us = transform[np.arange(3), (li, fi, ui)]
    lat = coords[li] if ls > 0 else -coords[li]
    fwd = coords[fi] if fs > 0 else -coords[fi]
    up = coords[ui] if us > 0 else -coords[ui]
    n = len(arr)
    r2 = lat * lat + fwd * fwd + up * up
    valid = np.isfinite(r2) & (r2 > min_range * min_range)
    ring_key = next((key for key in ("ring", "channel") if key in names), None)
    if scan_pattern == "mirror":
        ring_all = np.full(n, -1, np.int16)
        if valid.any():
            elevation = np.degrees(np.arctan2(up[valid], np.hypot(lat[valid], fwd[valid])))
            bands = np.floor(elevation / elevation_resolution_deg).astype(np.int32)
            # Typical mirror scans have <= 2250 angular bands. A simple
            # offset avoids sorting every return on every frame; compact only
            # unusually fine user bins that would overflow int16.
            indices = bands - bands.min()
            if indices.max() > np.iinfo(np.int16).max:
                _, indices = np.unique(bands, return_inverse=True)
            if indices.max() > np.iinfo(np.int16).max:
                raise ValueError("too many elevation bands; increase elevation_resolution_deg")
            ring_all[valid] = indices.astype(np.int16)
        rings = int(ring_all.max()) + 1 if n else 0
        organized = False
    elif ring_key is not None:
        values = np.asarray(arr[ring_key])
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"point field {ring_key} must contain scalar numeric indices")
        bad = bool(n and (values.min() < -1 or values.max() > 32766))
        if not np.issubdtype(values.dtype, np.integer):
            bad = bad or bool(np.any(~np.isfinite(values)) or np.any(values != np.floor(values)))
        if bad:
            raise ValueError(f"point field {ring_key} must contain integer indices in [-1, 32766]")
        ring_all = values.astype(np.int16)
        rings = int(ring_all.max()) + 1 if n else 0
        organized = (rings > 0 and n % rings == 0
                     and np.array_equal(ring_all.reshape(-1, rings),
                                        np.broadcast_to(np.arange(rings, dtype=np.int16), (n // rings, rings))))
    else:
        # unorganized cloud without the ring field (e.g. x, y, z, intensity only):
        # the beam index is recovered from the elevation angle
        ring_all = estimate_rings(lat, fwd, up, valid)
        rings = int(ring_all.max()) + 1 if n and ring_all.max() >= 0 else 0
        organized = False
    columns = n // rings if organized else 0
    col_all = (np.arange(n, dtype=np.int32) // rings) if organized else np.full(n, -1, np.int32)
    split = dual_return_split(r2, valid, rings, columns) if organized and dual_returns else np.zeros(n, bool)
    if working_range is not None:
        # Same mask as cloud.subset(working_set(cloud)), with no intermediate
        # copies of every valid point in dense 360-degree scans.
        valid &= _working_mask(lat, fwd, up, col_all, columns, scan_pattern, dual_returns, working_range)
    xyz = np.column_stack((lat[valid], fwd[valid], up[valid])).astype(np.float32, copy=False)
    intensity_key = next((key for key in ("intensity", "reflectivity", "reflectance") if key in names), None)
    if intensity_key is not None:
        inten = scalar(intensity_key, np.float32)[valid]
    else:
        inten = np.zeros(len(xyz), np.float32)
    return Cloud(raw.stamp, raw.frame_id, xyz, inten, ring_all[valid], col_all[valid], n, rings, columns,
                 split[valid], scan_pattern, bool(dual_returns and scan_pattern == "mechanical"),
                 np.flatnonzero(valid) if preserve_source_index else None)
