"""Versioned configuration for the web detector."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "detector.json"


@dataclass(frozen=True)
class DetectorConfig:
    version: str = "1.0"
    corridor_center_x: float = -0.2
    corridor_width: float = 2.2
    corridor_height: float = 3.0
    min_forward: float = 3.0
    max_forward: float = 90.0
    fallback_floor_z: float = -2.3
    floor_bin_size: float = 2.0
    floor_quantile: float = 0.08
    floor_min_points: int = 80
    floor_smooth_radius: int = 2
    floor_lateral_margin: float = 0.8
    rail_min_height: float = 0.04
    rail_max_height_for_path: float = 0.32
    rail_search_inner: float = 0.40
    rail_search_outer: float = 1.20
    rail_histogram_bin: float = 0.08
    rail_min_peak_points: int = 4
    rail_min_gauge: float = 0.90
    rail_max_gauge: float = 2.20
    path_max_center_shift: float = 0.25
    path_min_coverage: float = 0.35
    min_height_above_floor: float = 0.10
    max_height_above_floor: float = 3.0
    range_bands: tuple[float, ...] = (20.0, 40.0, 65.0, 90.0)
    voxel_sizes: tuple[float, ...] = (0.20, 0.25, 0.32, 0.40)
    min_cluster_points: tuple[int, ...] = (18, 14, 9, 6)
    max_cluster_width: float = 2.8
    max_cluster_depth: float = 4.0
    max_cluster_height: float = 3.2
    rail_max_height: float = 0.22
    rail_min_depth: float = 1.2
    background_voxel_x: float = 0.20
    background_voxel_forward: float = 0.50
    background_voxel_height: float = 0.20
    background_reject_fraction: float = 0.50
    background_model_path: str = "models/normal_scene.npz"
    max_objects: int = 20
    visualization_max_points: int = 16_000
    track_match_distance: float = 1.8
    track_confirm_frames: int = 3
    track_clear_frames: int = 3
    track_max_missing_frames: int = 3

    def validate(self) -> None:
        for name, value in self.public().items():
            values = value if isinstance(value, (list, tuple)) else (value,)
            if any(isinstance(item, (int, float)) and not math.isfinite(item) for item in values):
                raise ValueError(f"{name} must be finite")
        if not self.range_bands or any(value <= 0 for value in self.voxel_sizes + self.min_cluster_points):
            raise ValueError("range bands and positive clustering parameters are required")
        if self.min_forward < 0 or self.min_forward >= self.max_forward:
            raise ValueError("min_forward must be below max_forward")
        if not 0 <= self.floor_quantile <= 1 or not 0 <= self.background_reject_fraction <= 1:
            raise ValueError("quantile and background fraction must be in 0..1")
        if min(self.track_confirm_frames, self.track_clear_frames, self.max_objects, self.visualization_max_points) < 1:
            raise ValueError("frame counts and output limits must be positive")
        if min(self.background_voxel_x, self.background_voxel_forward, self.background_voxel_height, self.rail_histogram_bin) <= 0:
            raise ValueError("grid sizes must be positive")
        if not 1.0 <= self.corridor_width <= 5.0:
            raise ValueError("corridor_width must be between 1 and 5 metres")
        if not 5.0 <= self.max_forward <= 200.0:
            raise ValueError("max_forward must be between 5 and 200 metres")
        if not 0 < self.floor_bin_size <= 10:
            raise ValueError("floor_bin_size must be positive")
        if len(self.range_bands) != len(self.voxel_sizes):
            raise ValueError("range_bands and voxel_sizes must have equal lengths")
        if len(self.range_bands) != len(self.min_cluster_points):
            raise ValueError("range_bands and min_cluster_points must have equal lengths")
        if tuple(sorted(self.range_bands)) != self.range_bands:
            raise ValueError("range_bands must be sorted")

    def public(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **values: Any) -> "DetectorConfig":
        allowed = {field.name for field in fields(self)}
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"unknown detector settings: {sorted(unknown)}")
        result = replace(self, **values)
        result.validate()
        return result


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> DetectorConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in ("range_bands", "voxel_sizes", "min_cluster_points"):
        if name in payload:
            payload[name] = tuple(payload[name])
    config = DetectorConfig(**payload)
    config.validate()
    return config
