"""Per-frame obstacle candidates inside the clearance envelope.

Points are expressed in track coordinates: s — distance along the track,
d — lateral offset from the centre line, h — height above the rail head.
Everything inside the envelope (|d| <= W/2, h_min <= h <= H) is suspicious.
Candidates are clustered with a range-adaptive voxel grid and every cluster is
checked against the ways a *normal* tunnel can leak into the envelope:
residual ground, structures attached to walls or the ceiling that only touch
the envelope boundary, sparse noise (spray, reflections).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gabarit.config import Config
from gabarit.track import TrackModel, _window_percentile

AZ_RES = math.radians(0.083)   # horizontal step of one firing column
EL_RES = math.radians(0.125)   # vertical step of the central channels

_NEIGH = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)])


def connected_components(keys: np.ndarray) -> np.ndarray:
    """Labels of 26-connected components of integer voxel keys (N, 3)."""
    n = len(keys)
    if n == 0:
        return np.zeros(0, np.int64)
    k = keys - keys.min(axis=0) + 1
    dims = k.max(axis=0) + 2
    code = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    uniq, inv = np.unique(code, return_inverse=True)
    m = len(uniq)
    src = []
    dst = []
    for a, b, c in _NEIGH:
        nb = uniq + (a * dims[1] + b) * dims[2] + c
        pos = np.searchsorted(uniq, nb)
        pos = np.minimum(pos, m - 1)
        hit = uniq[pos] == nb
        src.append(np.flatnonzero(hit))
        dst.append(pos[hit])
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    label = np.arange(m)
    if len(src):
        for _ in range(64):
            lo = np.minimum(label[src], label[dst])
            new = label.copy()
            np.minimum.at(new, src, lo)
            np.minimum.at(new, dst, lo)
            new = new[new]  # pointer jumping
            if np.array_equal(new, label):
                break
            label = new
    _, comp = np.unique(label, return_inverse=True)
    return comp[inv]


@dataclass
class Candidate:
    id: int
    s_near: float
    s_far: float
    d_min: float
    d_max: float
    h_min: float
    h_max: float
    distance: float          # Euclidean distance from the sensor to the nearest point
    n_points: int
    n_rings: int
    lateral: float           # median d
    score: float
    reasons: list = field(default_factory=list)
    rejected: str | None = None
    point_index: np.ndarray | None = None  # indices into the frame cloud
    low: bool = False        # low object on / between the rails (needs longer confirmation)

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.s_far - self.s_near, self.d_max - self.d_min, self.h_max - self.h_min)

    def public(self) -> dict:
        return {
            "id": self.id,
            "distance_m": round(self.distance, 2),
            "s_m": round(self.s_near, 2),
            "lateral_m": round(self.lateral, 2),
            "box": {
                "s": [round(self.s_near, 2), round(self.s_far, 2)],
                "d": [round(self.d_min, 2), round(self.d_max, 2)],
                "h": [round(self.h_min, 2), round(self.h_max, 2)],
            },
            "size_m": [round(v, 2) for v in self.size],
            "points": self.n_points,
            "rings": self.n_rings,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "rejected": self.rejected,
            "low": self.low,
        }


@dataclass
class FrameGeometry:
    """Track coordinates of the cloud, reused by the pipeline for visualisation."""

    s: np.ndarray
    d: np.ndarray
    h: np.ndarray
    in_range: np.ndarray
    corridor: np.ndarray       # inside the clearance envelope (candidate points)
    half_width: float


def track_coordinates(lev: np.ndarray, track: TrackModel, cfg: Config) -> FrameGeometry:
    f = lev[:, 1]
    in_range = (f >= track.s[0] - 0.5) & (f <= min(cfg.detection.max_range, track.s[-1]))
    c, zr, _ = track.at(f)
    d = lev[:, 0] - c
    h = lev[:, 2] - zr
    env = cfg.envelope
    half = env.width / 2 + env.lateral_margin
    sig_z = np.interp(f, track.s, track.sig_z)
    hmin = env.min_height + np.minimum(2.0 * sig_z, 0.4)
    corridor = in_range & (np.abs(d) <= half) & (h >= hmin) & (h <= env.height)
    return FrameGeometry(f, d, h, in_range, corridor, half)


def _longitudinal(s: np.ndarray, d: np.ndarray, h: np.ndarray, members: np.ndarray, cand: "Candidate",
                  half: float) -> bool:
    """True if the cluster is a piece of a structure running along the track.

    The clearance envelope is free by construction, so a line of similar
    returns continuing ahead of and behind the cluster (platform edge, cable
    tray, wall seen through a small path error) is infrastructure. Obstacles
    interrupt the free envelope instead.
    """
    dm = cand.lateral
    hm = 0.5 * (cand.h_min + cand.h_max)
    tol_d = 0.2 + 0.002 * cand.s_near
    span = 12.0
    near = (np.abs(d - dm) < tol_d) & (h > cand.h_min - 0.3) & (h < cand.h_max + 0.3)
    before = near & (s < cand.s_near - 0.7) & (s > cand.s_near - span)
    after = near & (s > cand.s_far + 0.7) & (s < cand.s_far + span)
    nb = len(np.unique(np.floor(s[before])))
    na = len(np.unique(np.floor(s[after])))
    del members, hm, half
    return nb >= 2 and na >= 2 and nb + na >= 5


def _fixture_row(s: np.ndarray, d: np.ndarray, h: np.ndarray, cand: "Candidate") -> bool:
    """True if similar returns repeat along the same side line (brackets, boxes,
    signal heads mounted on the wall at a regular pitch)."""
    tol_d = 0.2 + 0.002 * cand.s_near
    near = (np.abs(d - cand.lateral) < tol_d) & (h > cand.h_min - 0.4) & (h < cand.h_max + 0.4)
    other = near & ((s < cand.s_near - 1.0) | (s > cand.s_far + 1.0)) & (np.abs(s - cand.s_near) < 30.0)
    return len(np.unique(np.floor(s[other] / 1.0))) >= 4


def expected_points(width: float, height: float, rng: float, returns: int = 2) -> float:
    cols = width / max(rng * AZ_RES, 1e-3)
    rows = height / max(rng * EL_RES, 1e-3)
    return max(cols, 1.0) * max(rows, 1.0) * returns


def _detect_envelope(lev: np.ndarray, ring: np.ndarray, geo: FrameGeometry, track: TrackModel, cfg: Config) -> list[Candidate]:
    det = cfg.detection
    env = cfg.envelope
    half = geo.half_width
    # context: corridor plus a band around it to recognise attached structures
    ctx = geo.in_range & (np.abs(geo.d) <= half + 1.2) & (geo.h >= env.min_height) & (geo.h <= env.height + 1.2)
    ctx_idx = np.flatnonzero(ctx)
    if not len(ctx_idx) or not geo.corridor[ctx_idx].any():
        return []
    s = geo.s[ctx_idx]
    d = geo.d[ctx_idx]
    h = geo.h[ctx_idx]
    v0, g = det.voxel_near, det.voxel_growth
    vox = v0 + g * np.maximum(s, 0.0)
    u = np.log(v0 + g * np.maximum(s, 0.0)) / g * 1.6 if g > 0 else s / v0
    keys = np.column_stack((np.floor(u), np.floor(d / vox), np.floor(h / vox))).astype(np.int64)
    labels = connected_components(keys)
    inside = geo.corridor[ctx_idx]
    cands: list[Candidate] = []
    lab_in = np.unique(labels[inside])
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, lab_in, "left")
    ends = np.searchsorted(sorted_labels, lab_in, "right")
    next_id = 1
    for lab, a, b in zip(lab_in, starts, ends):
        members = order[a:b]
        mem_in = members[inside[members]]
        n_in = len(mem_in)
        if n_in < 2:
            continue
        si, di, hi = s[mem_in], d[mem_in], h[mem_in]
        idx_in = ctx_idx[mem_in]
        rings = int(len(np.unique(ring[idx_in]))) if ring is not None and ring.size else n_in
        xyz = lev[idx_in]
        dist = float(np.sqrt((xyz ** 2).sum(axis=1)).min())
        cand = Candidate(
            id=next_id, s_near=float(si.min()), s_far=float(si.max()), d_min=float(di.min()),
            d_max=float(di.max()), h_min=float(hi.min()), h_max=float(hi.max()), distance=dist,
            n_points=n_in, n_rings=rings, lateral=float(np.median(di)), score=0.0,
            point_index=idx_in,
        )
        next_id += 1
        # --- whole connected structure (including outside the envelope)
        d_all, h_all = d[members], h[members]
        out_lat = np.abs(d_all) > half + det.wall_attach_margin
        frac_out = float(out_lat.mean())
        top_all = float(h_all.max())
        rng = max(cand.s_near, 1.0)
        # low objects lying on or between the rails: the rail head is the top of
        # the track structure, so anything above it is foreign; the height
        # reference is reliable only where the rails themselves are resolved
        low_zone = (cand.h_max < det.low_max_height and cand.s_near <= min(track.rail_until + 8.0, det.low_max_range)
                    and float(np.abs(di).max()) < det.low_max_lateral)
        cand.low = bool(low_zone)
        # a compact body filling a large part of the envelope cross-section:
        # vault fixtures never reach a metre down into the envelope over a
        # metre of its width, so the ceiling rules do not apply to it
        massive = (cand.d_max - cand.d_min >= 0.8 and cand.h_min <= env.height - 1.0
                   and cand.h_max - cand.h_min >= 0.8 and cand.s_far - cand.s_near <= 1.5)
        # --- rejections: normal tunnel leaking into the envelope
        if low_zone and n_in >= det.low_min_points and float(np.abs(d_all).max()) < half + 0.2:
            pass
        elif n_in < det.min_points or rings < det.min_rings:
            cand.rejected = "sparse"
        elif not low_zone and cand.h_max - cand.h_min < det.min_height_extent and cand.h_max < env.min_height + 0.25:
            cand.rejected = "ground"
        elif float(s[members].max() - s[members].min()) > 4.0 and float(h_all.max() - h_all.min()) > 0.8:
            # obstacles are compact along the track; tall structures stretched
            # along it are walls, platform faces or tunnel sections on a curve
            cand.rejected = "elongated_structure"
        elif frac_out > 0.35 and float(np.abs(d_all).max()) > half + 0.8 \
                and float(s[members].max() - s[members].min()) > 2.5:
            # an oblique surface continuing well outside the envelope: a tunnel
            # wall seen through an error of the path on a curve / widening
            cand.rejected = "oblique_surface"
        elif frac_out > 0.5 and np.abs(di).min() > half - env.edge_zone - 0.15:
            cand.rejected = "wall_attached"
        elif top_all > env.height + det.ceiling_margin and cand.h_min > env.height - 0.6:
            cand.rejected = "ceiling_fixture"
        elif top_all > env.height + 0.5 and cand.h_min < 0.8 and cand.h_max > env.height - 0.4:
            # continuous from the ballast up to the vault: a column / pillar of a
            # double-track tunnel or a station; nothing movable is 4 m tall
            cand.rejected = "column"
        elif top_all > env.height + 0.5 and cand.h_max > env.height - 0.4 and cand.s_near > 60.0 and not massive:
            # hanging from the vault far away, where the path is uncertain
            cand.rejected = "ceiling_attached"
        if cand.rejected is None and _longitudinal(s, d, h, members, cand, half):
            cand.rejected = "longitudinal_structure"
        if cand.rejected is None and abs(cand.lateral) > 0.5 and _fixture_row(s, d, h, cand):
            cand.rejected = "side_fixture_row"
        if cand.rejected is None and cand.s_near > track.rail_until + 10.0:
            # beyond the rails the path may drift towards a wall on a curve; the
            # walls themselves are tracked independently, and anything on the
            # wall line is wall equipment (the envelope edge is >= 1.2 m from a
            # tunnel wall, >= 0.5 m from a platform edge)
            for side in (+1, -1):
                w = track.wall_offset(cand.s_near, side)
                if w is not None and side * w > 0.5:
                    gap = (w - cand.d_max) if side > 0 else (cand.d_min - w)
                    if gap < det.wall_line_gap:
                        cand.rejected = "wall_line"
        if cand.rejected is None and cand.s_near > 80.0:
            # far away only person-sized evidence is trusted: a thin pole or a
            # couple of returns can be signal equipment seen through path error
            az = np.arctan2(xyz[:, 0], xyz[:, 1])
            n_cols = int(round(float(az.max() - az.min()) / AZ_RES)) + 1
            need_rings = 3 if cand.s_near < 120.0 else 4
            if rings < need_rings or n_cols < 2 or cand.h_max - cand.h_min < 0.4:
                cand.rejected = "far_weak"
            elif cand.d_max - cand.d_min < 0.15:
                cand.rejected = "pole_far"
            elif cand.h_min > 2.2:
                cand.rejected = "high_far"
        if cand.rejected is None and not low_zone and cand.s_near > 40.0 and cand.h_min > 1.0                 and cand.h_max < env.height - 0.3 and cand.size[1] < 0.8:
            # hangs in the middle of the envelope without touching the track or
            # the vault: equipment on a wall arm seen through a path error
            # (a person or a fallen object always reaches down to the track)
            cand.rejected = "floating"
        if cand.rejected is None and cand.s_near > 50.0 and cand.h_min > 2.2:
            cand.rejected = "high_far"
        if cand.rejected is None and cand.s_near > 50.0:
            # far away the path is known to ~0.1-0.3 m: an object that only
            # touches the boundary cannot be told from wall equipment
            core = half - (0.15 + 0.003 * (cand.s_near - 50.0))
            if float(np.abs(di).min()) > core:
                cand.rejected = "edge_far"
        # --- score of the remaining evidence
        exp = expected_points(max(cand.size[1], 0.3), max(cand.size[2], 0.3), rng)
        density = min(1.0, n_in / max(0.25 * exp, 3.0))
        vert = min(1.0, (cand.h_max - cand.h_min) / 0.3)
        if low_zone:
            density = min(1.0, n_in / 4.0)
            vert = 1.0
        edge = 1.0 if np.abs(di).min() < half - env.edge_zone else 0.55
        sig_c = float(np.interp(cand.s_near, track.s, track.sig_c))
        path_conf = 1.0 if sig_c < 0.15 else max(0.3, 1.0 - (sig_c - 0.15) * 2.0)
        observed = 1.0 if cand.s_near <= track.observed_until + 10.0 else 0.6
        cand.score = float(density * (0.5 + 0.5 * vert) * edge * path_conf * observed)
        reasons = [f"{n_in} точек на {rings} лучах"]
        if edge < 1.0:
            reasons.append("у границы габарита")
        if path_conf < 1.0:
            reasons.append("неточная ось пути")
        if observed < 1.0:
            reasons.append("путь за пределом наблюдения")
        cand.reasons = reasons
        if cand.rejected is None and cand.score < det.min_score:
            cand.rejected = "low_score"
        cands.append(cand)
    cands.sort(key=lambda c: (c.rejected is not None, c.s_near))
    return cands[: det.max_clusters * 2]


def detect(lev: np.ndarray, ring: np.ndarray, geo: FrameGeometry, track: TrackModel, cfg: Config) -> list[Candidate]:
    """All candidates of one frame: envelope clusters + objects lying on the rails."""
    cands = _detect_envelope(lev, ring, geo, track, cfg)
    rail = detect_rail_objects(lev, ring, geo, track, cfg)
    for rc in rail:
        clash = any(c.rejected is None and abs(c.s_near - rc.s_near) < 1.5 and abs(c.lateral - rc.lateral) < 0.6
                    for c in cands)
        if not clash:
            rc.id = len(cands) + 1
            cands.append(rc)
    cands.sort(key=lambda c: (c.rejected is not None, c.s_near))
    return cands


def detect_rail_objects(lev: np.ndarray, ring: np.ndarray, geo: FrameGeometry, track: TrackModel,
                        cfg: Config) -> list[Candidate]:
    """Small objects lying on a running rail (e.g. 300x300x100 mm).

    The rail head is the highest part of the track structure and is sampled by
    every ring crossing it. Along each rail the top height is compared with a
    robust local baseline of the same rail (+-5 m), so errors of the vertical
    profile cancel out. A short bump above the rail head is a foreign object;
    long raised runs are track equipment (switch parts, check rails).
    """
    det = cfg.detection
    gh = cfg.track.gauge / 2
    s_max = min(det.low_max_range, max(track.rail_until, 0.0) + 15.0)
    out: list[Candidate] = []
    if s_max < 8.0:
        return out
    step = 0.5
    nb = int(np.ceil(s_max / step)) + 1
    for side in (+1, -1):
        band = geo.in_range & (np.abs(geo.d - side * gh) < 0.2) & (geo.s > 4.0) & (geo.s < s_max)             & (geo.h > -0.35) & (geo.h < 0.6)
        idx = np.flatnonzero(band)
        if len(idx) < 20:
            continue
        b = (geo.s[idx] / step).astype(np.int64)
        top = np.full(nb, -np.inf)
        np.maximum.at(top, b, geo.h[idx])
        top[~np.isfinite(top)] = np.nan
        base = _window_percentile(top, 10, 0.4)
        thr = base + 0.07 + 0.0008 * np.arange(nb) * step
        above = geo.h[idx] > thr[b]
        cnt = np.bincount(b[above], minlength=nb)
        hot = (cnt >= det.low_min_points) & np.isfinite(base)
        k = 0
        while k < nb:
            if not hot[k]:
                k += 1
                continue
            j = k
            while j + 1 < nb and hot[j + 1]:
                j += 1
            run = j - k + 1
            sel = idx[above & (b >= k) & (b <= j)]
            k = j + 1
            if run > 3 or len(sel) < det.low_min_points:
                continue  # long raised structure: switch blade, check rail, joint bar
            ss, dd, hh = geo.s[sel], geo.d[sel], geo.h[sel]
            base_here = float(np.nanmedian(base[max(0, int(ss.min() / step)):int(ss.max() / step) + 1]))
            xyz = lev[sel]
            c = Candidate(
                id=0, s_near=float(ss.min()), s_far=float(ss.max()), d_min=float(dd.min()), d_max=float(dd.max()),
                h_min=float(hh.min() - base_here), h_max=float(hh.max() - base_here),
                distance=float(np.sqrt((xyz ** 2).sum(axis=1)).min()), n_points=len(sel),
                n_rings=int(len(np.unique(ring[sel]))) if ring is not None and ring.size else len(sel),
                lateral=float(np.median(dd)), score=min(1.0, len(sel) / 6.0),
                reasons=[f"выступ {hh.max() - base_here:.2f} м над головкой рельса", f"{len(sel)} точек"],
                point_index=sel, low=True,
            )
            # a single grazing ring over the rail head gives ~0.1 m bumps: require
            # a clear protrusion seen by two beams
            if c.h_max > 0.6 or c.n_rings < 2 or c.h_max < 0.13:
                continue
            out.append(c)
    return out
