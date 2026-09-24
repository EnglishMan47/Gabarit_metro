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

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "Cloud":
        return Cloud(self.stamp, self.frame_id, self.xyz[mask], self.intensity[mask], self.ring[mask],
                     self.column[mask], self.raw_count, self.rings, self.columns)


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
    return keep


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
    if "ring" in names:
        ring_all = np.asarray(arr["ring"], dtype=np.int16)
        rings = int(ring_all.max()) + 1 if n else 0
    else:
        ring_all = np.full(n, -1, np.int16)
        rings = 0
    organized = rings > 0 and n % rings == 0 and np.array_equal(ring_all[:rings], np.arange(rings))
    columns = n // rings if organized else 0
    col_all = (np.arange(n, dtype=np.int32) // rings) if organized else np.full(n, -1, np.int32)
    r2 = lat * lat + fwd * fwd + up * up
    valid = np.isfinite(r2) & (r2 > min_range * min_range)
    xyz = np.column_stack((lat[valid], fwd[valid], up[valid])).astype(np.float32, copy=False)
    if "intensity" in names:
        inten = np.asarray(arr["intensity"], dtype=np.float32)[valid]
    else:
        inten = np.zeros(len(xyz), np.float32)
    return Cloud(raw.stamp, raw.frame_id, xyz, inten, ring_all[valid], col_all[valid], n, rings, columns)
