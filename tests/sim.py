"""Synthetic straight tunnel scanned by a Pandar128-like lidar (for tests)."""

from __future__ import annotations

import math
import struct

import numpy as np

from gabarit.rosbag import RawCloud
from gabarit.synth import _ray_box

# Pandar128-like elevations: dense 0.125 deg in the centre, coarser at the edges
_EL = np.concatenate([np.linspace(15.0, 2.25, 26), np.linspace(2.0, -6.0, 65), np.linspace(-6.5, -25.0, 37)])
RINGS = 128
DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "timestamp"],
                  "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
                  "offsets": [0, 4, 8, 12, 16, 18], "itemsize": 26})


def tunnel_scan(height: float = 1.1, gauge: float = 1.6, half_width: float = 2.4, ceiling: float = 4.2,
                cols: int = 900, fov_deg: float = 50.0, max_range: float = 200.0, yaw_deg: float = 0.0,
                boxes: list | None = None, stamp: float = 0.0) -> RawCloud:
    """Organised scan (column-major, ring fastest) in the provided bag convention
    (x lateral, forward = -y, z up). Rails head at z = -height.

    ``boxes``: extra obstacles as (lateral centre, forward start, width, length, bottom, top)
    in sensor coordinates (bottom/top relative to the rail head).
    """
    el = np.radians(_EL)
    az = np.radians(np.linspace(-fov_deg, fov_deg, cols))
    E, A = np.meshgrid(el, az)  # (cols, rings)
    dx = np.cos(E) * np.sin(A)
    df = np.cos(E) * np.cos(A)
    dz = np.sin(E)
    d = np.column_stack((dx.ravel(), df.ravel(), dz.ravel()))
    t = np.full(len(d), np.inf)
    inten = np.full(len(d), 10.0)
    bed = -height - 0.18
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(d[:, 2] < 0, (bed - 0.0) / d[:, 2], np.inf)
        tw = np.where(d[:, 0] > 0, half_width / d[:, 0], np.where(d[:, 0] < 0, -half_width / d[:, 0], np.inf))
        tc = np.where(d[:, 2] > 0, (ceiling - height) / d[:, 2], np.inf)
    for cand in (tf, tw, tc):
        cand = np.where(cand > 0, cand, np.inf)
        t = np.minimum(t, cand)
    for side in (-1, 1):
        # rail: 0.07 wide, from the bed up to the head, along the whole range
        o_local = np.array([-(side * gauge / 2), 0.0 - 0.0, 0.0 - (-height - 0.18)])
        tr = _ray_box(np.array([o_local[0], 0.0, o_local[2]]), d, (0.035, 1e4), 0.18)
        better = tr < t
        t = np.where(better, tr, t)
        inten = np.where(better, 60.0, inten)
    for (xc, f0, w, l, b0, b1) in boxes or []:
        ob = np.array([-xc, -(f0 + l / 2), -(-height + b0)])
        tb = _ray_box(ob, d, (w / 2, l / 2), b1 - b0)
        better = tb < t
        t = np.where(better, tb, t)
        inten = np.where(better, 40.0, inten)
    t = np.where(t < max_range, t, 0.0)
    pts = d * t[:, None]
    if yaw_deg:
        r = math.radians(yaw_deg)   # sensor turned left: the track appears turned right
        x, f = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = x * math.cos(r) - f * math.sin(r)
        pts[:, 1] = x * math.sin(r) + f * math.cos(r)
    arr = np.zeros(len(d), DTYPE)
    arr["x"] = pts[:, 0]
    arr["y"] = -pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["intensity"] = np.where(t > 0, inten, 0)
    arr["ring"] = np.tile(np.arange(RINGS), cols)
    arr["timestamp"] = stamp
    return RawCloud(stamp, "sim_lidar", 1, len(arr), arr)


def encode_pointcloud2(raw: RawCloud) -> bytes:
    """Serialize a RawCloud into a CDR PointCloud2 message (for decoder tests)."""
    out = bytearray(b"\x00\x01\x00\x00")

    def align(n: int) -> None:
        while (len(out) - 4) % n:
            out.append(0)

    def u32(v: int) -> None:
        align(4)
        out.extend(struct.pack("<I", v))

    def string(s: str) -> None:
        b = s.encode() + b"\x00"
        u32(len(b))
        out.extend(b)

    sec = int(raw.stamp)
    u32(sec)
    u32(int((raw.stamp - sec) * 1e9))
    string(raw.frame_id)
    u32(raw.height)
    u32(raw.width)
    names = raw.points.dtype.names
    u32(len(names))
    types = {"<f4": 7, "<u2": 4, "<f8": 8}
    for n in names:
        string(n)
        u32(raw.points.dtype.fields[n][1])
        out.append(types[raw.points.dtype.fields[n][0].str])
        u32(1)
    out.append(0)
    u32(raw.points.dtype.itemsize)
    u32(raw.points.dtype.itemsize * raw.width)
    data = raw.points.tobytes()
    u32(len(data))
    out.extend(data)
    out.append(1)
    return bytes(out)
