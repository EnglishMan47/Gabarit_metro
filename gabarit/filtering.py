"""Conservative spatial support filter for isolated lidar returns.

Only returns without the configured number of distinct nearby samples are
removed. The cell size grows with range to preserve sparse distant surfaces.
This is an isolation filter, not a dust classifier: dense or persistent dust
can have spatial support and still needs temporal/geometric rejection.
"""

from __future__ import annotations

import numpy as np

from gabarit.config import FilterConfig

_NEIGH = tuple((a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
               for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0))


def _codes(keys: np.ndarray, padding: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Collision-free integer cell addresses for the bounded working cloud."""
    k = keys - keys.min(axis=0) + padding
    dims = k.max(axis=0) + padding + 1
    if int(dims[0]) * int(dims[1]) * int(dims[2]) > np.iinfo(np.int64).max:
        raise ValueError("point coordinates exceed the spatial filter's working region")
    return (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2], dims


def support_mask(xyz: np.ndarray, cfg: FilterConfig | None = None) -> np.ndarray:
    """Mask of supported points in the sensor working frame (left, ahead, up).

    Support is counted in the same or one of the 26 adjacent adaptive cells.
    Returns in the same 5 mm cell count as one sample, so duplicated dual
    echoes alone do not rescue an otherwise isolated return. No intensity or
    ring-field assumption is made. Complexity is O(N log N), memory O(N).
    """
    cfg = cfg or FilterConfig()
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    keep = np.isfinite(xyz).all(axis=1)
    if not cfg.enabled or not keep.any():
        return keep
    valid = np.flatnonzero(keep)
    p = xyz[valid].astype(np.float64, copy=False)
    f = np.maximum(p[:, 1], 0.0)
    v0, g = cfg.voxel_near, cfg.voxel_growth
    vox = v0 + g * f
    along = np.log1p(g * f / v0) / g if g > 0.0 else f / v0
    keys = np.column_stack((np.floor(along), np.floor(p[:, 0] / vox),
                            np.floor(p[:, 2] / vox))).astype(np.int64)
    code, dims = _codes(keys, padding=1)
    occupied, inv = np.unique(code, return_inverse=True)
    fine, _ = _codes(np.floor(p / 0.005).astype(np.int64))
    _, distinct = np.unique(fine, return_index=True)
    counts = np.bincount(inv[distinct], minlength=len(occupied))
    support = counts.copy()
    for a, b, c in _NEIGH:
        neighbor = occupied + (a * dims[1] + b) * dims[2] + c
        pos = np.minimum(np.searchsorted(occupied, neighbor), len(occupied) - 1)
        src = np.flatnonzero(occupied[pos] == neighbor)
        dst = pos[src]
        support[src] += counts[dst]
        support[dst] += counts[src]
    keep[valid] = support[inv] > cfg.min_neighbors
    return keep
