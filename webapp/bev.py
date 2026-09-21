"""Build the fixed bird's-eye-view tensor consumed by the optional ONNX model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from webapp.config import DetectorConfig
from webapp.scene_model import NormalSceneModel


BEV_CHANNELS = (
    "density",
    "max_height",
    "mean_height",
    "mean_intensity",
    "background_deviation",
    "corridor_mask",
    "temporal_stability",
)


@dataclass(frozen=True)
class BevSpec:
    """Versioned, model-facing grid that does not change with UI controls."""

    version: str = "bev-v1"
    min_x: float = -3.2
    max_x: float = 2.8
    min_forward: float = 0.0
    max_forward: float = 90.0
    resolution_x: float = 0.20
    resolution_forward: float = 0.50
    density_cap: int = 32
    intensity_scale: float = 255.0

    @property
    def width(self) -> int:
        return int(round((self.max_x - self.min_x) / self.resolution_x))

    @property
    def height(self) -> int:
        return int(round((self.max_forward - self.min_forward) / self.resolution_forward))

    def public(self) -> dict[str, Any]:
        return {**asdict(self), "channels": list(BEV_CHANNELS), "shape": [len(BEV_CHANNELS), self.height, self.width]}


DEFAULT_BEV_SPEC = BevSpec()


def _profiles(detection: dict[str, Any], points: np.ndarray, config: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    floor = np.asarray([entry["z_m"] for entry in detection["floor_profile"]], dtype=np.float32)
    centers = np.asarray([entry["center_x_m"] for entry in detection["path_profile"]], dtype=np.float32)
    bins = np.floor(points[:, 1] / config.floor_bin_size).astype(np.int32)
    bins = np.clip(bins, 0, len(floor) - 1)
    return floor[bins], centers[bins], bins


def build_bev(
    points: np.ndarray,
    intensity: np.ndarray | None,
    detection: dict[str, Any],
    config: DetectorConfig,
    scene_model: NormalSceneModel | None = None,
    *,
    spec: BevSpec = DEFAULT_BEV_SPEC,
    temporal_stability: np.ndarray | None = None,
) -> np.ndarray:
    """Return a float32 tensor with shape ``channels x forward x lateral``."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    tensor = np.zeros((len(BEV_CHANNELS), spec.height, spec.width), dtype=np.float32)
    if not len(points):
        return tensor

    floor, path_center, _ = _profiles(detection, points, config)
    heights = points[:, 2] - floor
    ix = np.floor((points[:, 0] - spec.min_x) / spec.resolution_x).astype(np.int32)
    iy = np.floor((points[:, 1] - spec.min_forward) / spec.resolution_forward).astype(np.int32)
    valid = (
        (ix >= 0) & (ix < spec.width) & (iy >= 0) & (iy < spec.height)
        & np.isfinite(heights)
    )
    if not np.any(valid):
        return tensor

    ix, iy = ix[valid], iy[valid]
    local_heights = heights[valid]
    flat = iy * spec.width + ix
    cell_count = spec.height * spec.width
    counts = np.bincount(flat, minlength=cell_count).astype(np.float32)
    height_sum = np.bincount(flat, weights=local_heights, minlength=cell_count).astype(np.float32)
    maximum = np.full(cell_count, -np.inf, dtype=np.float32)
    np.maximum.at(maximum, flat, local_heights)
    maximum[~np.isfinite(maximum)] = 0.0

    density = np.log1p(np.minimum(counts, spec.density_cap)) / np.log1p(spec.density_cap)
    mean_height = np.divide(height_sum, counts, out=np.zeros_like(height_sum), where=counts > 0)
    tensor[0] = density.reshape(spec.height, spec.width)
    tensor[1] = np.clip(maximum.reshape(spec.height, spec.width) / config.max_height_above_floor, 0.0, 1.0)
    tensor[2] = np.clip(mean_height.reshape(spec.height, spec.width) / config.max_height_above_floor, 0.0, 1.0)

    if intensity is not None and len(intensity) == len(points):
        intensity_sum = np.bincount(flat, weights=np.asarray(intensity, dtype=np.float32)[valid], minlength=cell_count)
        mean_intensity = np.divide(intensity_sum, counts, out=np.zeros_like(intensity_sum), where=counts > 0)
        tensor[3] = np.clip(mean_intensity.reshape(spec.height, spec.width) / spec.intensity_scale, 0.0, 1.0)

    if scene_model is not None:
        matched = scene_model.match(points[valid], local_heights, config).astype(np.float32)
        matched_sum = np.bincount(flat, weights=matched, minlength=cell_count)
        matched_fraction = np.divide(matched_sum, counts, out=np.zeros_like(matched_sum), where=counts > 0)
        deviation = np.where(counts > 0, 1.0 - matched_fraction, 0.0)
        tensor[4] = deviation.reshape(spec.height, spec.width)

    corridor = np.abs(points[valid, 0] - path_center[valid]) <= config.corridor_width / 2
    corridor_cells = np.bincount(flat, weights=corridor.astype(np.float32), minlength=cell_count) > 0
    tensor[5] = corridor_cells.reshape(spec.height, spec.width).astype(np.float32)

    if temporal_stability is not None:
        expected = (spec.height, spec.width)
        if temporal_stability.shape != expected:
            raise ValueError(f"temporal_stability must have shape {expected}")
        tensor[6] = np.clip(np.asarray(temporal_stability, dtype=np.float32), 0.0, 1.0)
    return np.ascontiguousarray(tensor, dtype=np.float32)
