"""Physically consistent synthetic obstacles for testing.

An object is placed on the estimated track (distance ``s`` along the path,
lateral offset ``d``, standing on the ballast) and ray-cast into the organised
lidar scan: every beam whose ray hits the object before its real return gets
the object's range. Occlusion of the tunnel behind the object, the density
fall-off with distance and partial visibility are therefore reproduced, unlike
simply pasting a point cluster.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from gabarit.cloud import axis_transform
from gabarit.rosbag import RawCloud
from gabarit.track import Calibration, TrackModel

SHAPES = {
    # name: (kind, width, length, height, base above rail head)
    "person": ("cylinder", 0.5, 0.35, 1.75, -0.15),
    "person_sitting": ("cylinder", 0.6, 0.65, 1.1, -0.15),
    "person_lying": ("cylinder", 0.55, 1.75, 0.45, -0.15),
    "child": ("cylinder", 0.35, 0.3, 1.2, -0.15),
    "box_30cm": ("box", 0.3, 0.3, 0.1, 0.0),        # minimal object from the Q&A, lying on a rail
    "box_50cm": ("box", 0.5, 0.5, 0.5, -0.15),
    "barrel": ("cylinder", 0.6, 0.6, 0.9, -0.15),
    "trolley": ("box", 1.4, 1.0, 1.0, 0.0),
    "cable": ("cylinder", 0.05, 0.05, 1.6, 1.3),     # hanging piece of cable
    "glass": ("glass", 1.2, 0.02, 1.8, 0.0),         # glass pane across the track: mostly transparent
}
GLASS_REFLECTION = 0.25   # share of beams that return a weak echo from the pane


@dataclass
class SyntheticObstacle:
    shape: str = "person"
    s: float = 100.0          # distance along the track from the sensor, m
    d: float = 0.0            # lateral offset from the track centre, m (left +)
    width: float | None = None
    length: float | None = None
    height: float | None = None
    base: float | None = None
    intensity: float = 40.0
    id: int = 0
    yaw_deg: float = 0.0       # extra rotation relative to the local track heading

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.s, self.d, self.yaw_deg, self.intensity)):
            raise ValueError("synthetic position, orientation and intensity must be finite")
        for value in (self.width, self.length, self.height):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("synthetic dimensions must be finite and positive")
        if self.base is not None and not math.isfinite(self.base):
            raise ValueError("synthetic base must be finite")

    def dims(self) -> tuple[str, float, float, float, float]:
        kind, w, l, h, b = SHAPES.get(self.shape, SHAPES["box_50cm"])
        return (kind, self.width or w, self.length or l, self.height or h,
                self.base if self.base is not None else b)

    def public(self) -> dict:
        kind, w, l, h, b = self.dims()
        out = asdict(self)
        out.update({"kind": kind, "width": w, "length": l, "height": h, "base": b})
        return out


def _beam_directions(xyz: np.ndarray, rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth/elevation of every beam of an organised scan (also for no-return beams)."""
    n = len(xyz)
    cols = n // rings
    x = xyz[:, 0].reshape(cols, rings).T
    f = xyz[:, 1].reshape(cols, rings).T
    z = xyz[:, 2].reshape(cols, rings).T
    r = np.sqrt(x * x + f * f + z * z)
    valid = r > 0.5
    az = np.arctan2(x, f)
    el = np.arcsin(np.divide(z, r, out=np.zeros_like(z), where=valid))
    col = np.arange(cols)
    az_full = np.full_like(az, np.nan)
    el_ring = np.full(rings, np.nan)
    for k in range(rings):
        v = valid[k]
        if v.sum() >= 2:
            az_full[k] = np.interp(col, col[v], np.unwrap(az[k, v]))
            el_ring[k] = np.median(el[k, v])
    return az_full, el_ring


def inject(raw: RawCloud, obstacles: list[SyntheticObstacle], track: TrackModel, calib: Calibration,
           forward_axis: str = "-y", rng: np.random.Generator | None = None, *,
           up_axis: str = "+z", scan_pattern: str = "mechanical") -> tuple[RawCloud, int]:
    """Return a copy of ``raw`` with the obstacles ray-cast into it and the number of hit beams."""
    if not obstacles:
        return raw, 0
    arr = raw.points
    names = arr.dtype.names or ()
    n = len(arr)
    if not n:
        return raw, 0
    transform = axis_transform(forward_axis, up_axis)
    coords = [np.asarray(arr[k], np.float64) for k in ("x", "y", "z")]
    xyz = np.column_stack(coords) @ transform.T
    rings = int(arr["ring"].max()) + 1 if "ring" in names else 0
    organized = (scan_pattern == "mechanical" and rings > 0 and not n % rings
                 and np.array_equal(arr["ring"].reshape(-1, rings),
                                    np.broadcast_to(np.arange(rings), (n // rings, rings))))
    if not organized:
        return _inject_unorganized(raw, xyz, obstacles, track, calib, transform, rng)
    az, el = _beam_directions(xyz, rings)
    cols = n // rings
    rng = rng or np.random.default_rng(0)
    rng_cur = np.sqrt((xyz ** 2).sum(axis=1)).reshape(cols, rings).T
    rng_cur = np.where(rng_cur > 0.5, rng_cur, np.inf)
    # levelling rotation (sensor -> levelled frame), origin at (0, 0, height)
    a = math.atan(calib.pitch)
    b = math.atan(calib.roll)
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    hit_range = np.full((rings, cols), np.inf)
    hit_int = np.zeros((rings, cols))
    for ob in obstacles:
        kind, w, l, h, base = ob.dims()
        c = float(np.interp(ob.s, track.s, track.c))
        zr = float(np.interp(ob.s, track.s, track.z))
        th = float(np.interp(ob.s, track.s, np.gradient(track.c, track.s)))
        cx, cf, z0 = c + ob.d, ob.s, zr + base
        # angular window (levelled frame ~ sensor frame for these small angles)
        rad = 0.5 * max(w, l) + 0.3
        dist = math.hypot(cx, cf)
        if dist < 1.0:
            continue
        az0 = math.atan2(cx, cf)
        daz = math.atan2(rad, dist) + 0.01
        el_lo = math.atan2(z0 - calib.height - 0.5, dist) - 0.02
        el_hi = math.atan2(z0 + h - calib.height + 0.5, dist) + 0.02
        ring_sel = np.flatnonzero((el >= el_lo - abs(a) - abs(b)) & (el <= el_hi + abs(a) + abs(b)))
        for k in ring_sel:
            cs = np.flatnonzero(np.abs(az[k] - az0) < daz + abs(b))
            if not len(cs):
                continue
            # ray directions in the sensor frame
            e = el[k]
            dx = np.cos(e) * np.sin(az[k, cs])
            df = np.cos(e) * np.cos(az[k, cs])
            dz = np.full(len(cs), math.sin(e))
            # to the levelled frame (same rotation as Calibration.level)
            f1 = df * ca + dz * sa
            z1 = -df * sa + dz * ca
            x2 = dx * cb + z1 * sb
            z2 = -dx * sb + z1 * cb
            ox, of, oz = 0.0, 0.0, calib.height
            # object local frame: rotate by the track heading
            heading = math.atan(th) + math.radians(ob.yaw_deg)
            ct, st = math.cos(heading), math.sin(heading)
            rx = (x2 * ct - f1 * st)
            rf = (x2 * st + f1 * ct)
            px = (ox - cx) * ct - (of - cf) * st
            pf = (ox - cx) * st + (of - cf) * ct
            pz = oz - z0
            if kind in ("box", "glass"):
                t = _ray_box(np.array([px, pf, pz]), np.column_stack((rx, rf, z2)), (w / 2, l / 2), h)
                if kind == "glass":
                    # most of the pulse passes through: only some beams echo from the
                    # pane, as the first return of the firing (even column); the second
                    # return (odd column) keeps the tunnel behind the glass
                    echo = (rng.random(len(cs)) < GLASS_REFLECTION) & (cs % 2 == 0)
                    t = np.where(echo, t, np.inf)
            else:
                t = _ray_cylinder(np.array([px, pf, pz]), np.column_stack((rx, rf, z2)), w / 2, l / 2, h)
            better = t < hit_range[k, cs]
            hit_range[k, cs[better]] = t[better]
            hit_int[k, cs[better]] = 4.0 if kind == "glass" else ob.intensity
    use = np.isfinite(hit_range) & (hit_range < rng_cur)
    if not use.any():
        return raw, 0
    kk, cc = np.nonzero(use)
    idx = cc * rings + kk
    r = np.clip(hit_range[kk, cc] + rng.normal(0, 0.015, len(kk)), 0.0, rng_cur[kk, cc])
    e = el[kk]
    azs = az[kk, cc]
    lat = r * np.cos(e) * np.sin(azs)
    fwd = r * np.cos(e) * np.cos(azs)
    up = r * np.sin(e)
    out = arr.copy()
    raw_hits = np.column_stack((lat, fwd, up)) @ transform
    for key, v in zip(("x", "y", "z"), raw_hits.T):
        out[key][idx] = v.astype(out[key].dtype)
    if "intensity" in names:
        out["intensity"][idx] = hit_int[kk, cc].astype(out["intensity"].dtype)
    return RawCloud(raw.stamp, raw.frame_id, raw.height, raw.width, out), int(use.sum())


def _inject_unorganized(raw, xyz, obstacles, track, calib, transform, rng):
    """Ray-cast along measured directions for mirror/no-ring point clouds.

    Missing beams cannot be reconstructed from an unorganized return list.
    Existing rays are shortened only when an object occludes their return.
    """
    rng = rng or np.random.default_rng(0)
    lengths = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (lengths > 0.5)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return raw, 0
    directions = xyz[idx] / lengths[idx, None]
    # Calibration.level includes translation; subtract it for direction vectors.
    level_directions = calib.level(directions)
    level_directions[:, 2] -= calib.height
    hits = lengths[idx].copy()
    intensities = np.zeros(len(idx))
    changed = np.zeros(len(idx), bool)
    for ob in obstacles:
        kind, w, length, h, base = ob.dims()
        center = float(np.interp(ob.s, track.s, track.c)) + ob.d
        bottom = float(np.interp(ob.s, track.s, track.z)) + base
        heading = math.atan(float(np.interp(ob.s, track.s, np.gradient(track.c, track.s)))) + math.radians(ob.yaw_deg)
        ct, st = math.cos(heading), math.sin(heading)
        local = level_directions.copy()
        local[:, 0] = level_directions[:, 0] * ct - level_directions[:, 1] * st
        local[:, 1] = level_directions[:, 0] * st + level_directions[:, 1] * ct
        origin = np.array([-center * ct + ob.s * st, -center * st - ob.s * ct, calib.height - bottom])
        if kind in ("box", "glass"):
            distance = _ray_box(origin, local, (w / 2, length / 2), h)
            if kind == "glass":
                distance = np.where(rng.random(len(idx)) < GLASS_REFLECTION, distance, np.inf)
        else:
            distance = _ray_cylinder(origin, local, w / 2, length / 2, h)
        nearer = distance < hits
        hits[nearer] = distance[nearer]
        intensities[nearer] = 4.0 if kind == "glass" else ob.intensity
        changed |= nearer
    if not changed.any():
        return raw, 0
    selected = idx[changed]
    distances = np.clip(hits[changed] + rng.normal(0, 0.015, changed.sum()), 0.0, lengths[selected])
    raw_hits = (directions[changed] * distances[:, None]) @ transform
    out = raw.points.copy()
    for key, values in zip("xyz", raw_hits.T):
        out[key][selected] = values
    for key in ("intensity", "reflectivity", "reflectance"):
        if key in (out.dtype.names or ()):
            out[key][selected] = intensities[changed]
            break
    return RawCloud(raw.stamp, raw.frame_id, raw.height, raw.width, out), int(changed.sum())


def inject_dust(raw: RawCloud, fraction: float = 0.02, near: float = 3.0, far: float = 70.0,
                rng: np.random.Generator | None = None, *, forward_axis: str = "-y",
                up_axis: str = "+z") -> tuple[RawCloud, int]:
    """Replace a share of forward returns with sparse weak aerosol echoes.

    This reproducible stress scenario is not a calibrated physical aerosol
    simulator. It preserves fields/organization and cannot synthesize unseen rays.
    """
    if not (all(math.isfinite(v) for v in (fraction, near, far)) and 0 <= fraction <= 1 and 0 < near < far):
        raise ValueError("dust requires fraction in [0, 1] and 0 < near < far")
    arr = raw.points
    if not len(arr) or fraction == 0:
        return raw, 0
    transform = axis_transform(forward_axis, up_axis)
    xyz = np.column_stack([arr[k] for k in "xyz"]).astype(float) @ transform.T
    ranges = np.linalg.norm(xyz, axis=1)
    rng = rng or np.random.default_rng(0)
    selected = np.flatnonzero(np.isfinite(ranges) & (ranges > near) & (xyz[:, 1] > 0)
                              & (rng.random(len(arr)) < fraction))
    if not len(selected):
        return raw, 0
    distance = rng.uniform(near, np.minimum(far, ranges[selected]))
    raw_hits = (xyz[selected] * (distance / ranges[selected])[:, None]) @ transform
    out = arr.copy()
    for key, values in zip("xyz", raw_hits.T):
        out[key][selected] = values
    for key in ("intensity", "reflectivity", "reflectance"):
        if key in (arr.dtype.names or ()):
            out[key][selected] = 1.0
            break
    return RawCloud(raw.stamp, raw.frame_id, raw.height, raw.width, out), len(selected)


def _ray_box(o: np.ndarray, dvec: np.ndarray, half: tuple[float, float], h: float) -> np.ndarray:
    lo = np.array([-half[0], -half[1], 0.0])
    hi = np.array([half[0], half[1], h])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dvec
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    ok = (tmax >= tmin) & (tmax > 0)
    return np.where(ok, np.maximum(tmin, 0.0), np.inf)


def _ray_cylinder(o: np.ndarray, dvec: np.ndarray, ra: float, rb: float, h: float) -> np.ndarray:
    """Elliptic vertical cylinder (semi-axes ra across, rb along the track)."""
    dx, df, dz = dvec[:, 0] / ra, dvec[:, 1] / rb, dvec[:, 2]
    ox, of, oz = o[0] / ra, o[1] / rb, o[2]
    A = dx * dx + df * df
    B = 2 * (ox * dx + of * df)
    C = ox * ox + of * of - 1.0
    disc = B * B - 4 * A * C
    out = np.full(len(dvec), np.inf)
    ok = (disc >= 0) & (A > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t = np.where(ok, (-B - sq) / (2 * np.where(A > 1e-12, A, 1.0)), np.inf)
    zt = oz + t * dz
    side = ok & (t > 0) & (zt >= 0) & (zt <= h)
    out[side] = t[side]
    # caps (top), for rays from above
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (h - oz) / dz
    xc = ox + tc * dx
    fc = of + tc * df
    cap = (tc > 0) & (xc * xc + fc * fc <= 1.0) & np.isfinite(tc)
    out = np.where(cap & (tc < out), tc, out)
    return out
