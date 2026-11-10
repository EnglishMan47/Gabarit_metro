"""Configuration of the detector. Every tunable value lives here.

Values can be overridden from a YAML/JSON file (``config/gabarit.yaml``) or
programmatically with :meth:`Config.with_overrides`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SensorConfig:
    # Axis of the raw cloud that points along the direction of travel.
    # "-y" matches the provided bags; "+x" is the ROS REP-103 convention.
    forward_axis: str = "-y"
    # Automatic calibration of the sensor pose from the running rails.
    auto_calibration: bool = True
    # Fallback pose used until rails are seen (rail head below the sensor, m).
    default_height: float = 1.1
    default_pitch_deg: float = 0.0
    default_roll_deg: float = 0.0
    default_yaw_deg: float = 0.0
    default_lateral: float = 0.0
    # Smoothing of the per-frame calibration (0..1, weight of the new frame).
    calibration_alpha: float = 0.15
    min_range: float = 1.0
    # Raw axis pointing up. Falcon Prime native coordinates are forward +z,
    # up +x; a ROS driver may already have converted them to REP-103.
    up_axis: str = "+z"
    scan_pattern: str = "mechanical"   # mechanical or mirror
    # Pandar bags use adjacent columns for the two returns of one firing.
    # Disable for single return and for formats without this exact layout.
    dual_returns: bool = True
    azimuth_resolution_deg: float = 0.083
    elevation_resolution_deg: float = 0.125
    # A nonuniform mirror scan can be coarser outside its ROI. Use the
    # coarse physical spacing for density, retaining fine bands for beams.
    density_azimuth_resolution_deg: float = 0.0
    density_elevation_resolution_deg: float = 0.0

    def __post_init__(self) -> None:
        axes = {sign + axis for sign in ("+", "-") for axis in "xyz"}
        if self.forward_axis not in axes or self.up_axis not in axes:
            raise ValueError("sensor forward_axis and up_axis must be signed axes: +x, -x, +y, -y, +z, -z")
        if self.forward_axis[1] == self.up_axis[1]:
            raise ValueError("sensor forward_axis and up_axis must be perpendicular")
        if self.scan_pattern not in ("mechanical", "mirror"):
            raise ValueError("sensor scan_pattern must be mechanical or mirror")
        for name in ("azimuth_resolution_deg", "elevation_resolution_deg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0 or value > 10:
                raise ValueError(f"sensor {name} must be finite and within (0, 10] degrees")
        for name in ('density_azimuth_resolution_deg','density_elevation_resolution_deg'):
            value = getattr(self,name)
            if not math.isfinite(value) or value < 0 or value > 10:
                raise ValueError(f"sensor {name} must be finite and within [0, 10] degrees")
        if not math.isfinite(self.min_range) or self.min_range < 0:
            raise ValueError("sensor min_range must be finite and nonnegative")


@dataclass(frozen=True)
class TrackConfig:
    gauge: float = 1.60           # distance between rail-head centres, m
    gauge_tolerance: float = 0.10
    rail_min_prominence: float = 0.06
    rail_search_max: float = 60.0  # rails are tracked up to this distance
    start: float = 3.0             # corridor begins here (m ahead of the sensor)
    horizon: float = 200.0         # the path is estimated up to this distance
    near_step: float = 1.0
    far_step: float = 2.0
    near_far_switch: float = 40.0
    wall_band: tuple = (0.3, 2.5)  # heights used to follow the walls
    wall_min_offset: float = 1.15  # wall points must be this far from the path centre
    wall_window: float = 0.35
    wall_lost_after: float = 12.0  # m without wall points
    curvature_noise: float = 1.5e-4  # process noise of path curvature, 1/m^2 per m
    temporal_weight: float = 0.6   # weight of the previous frame path prior
    smooth_lateral: float = 3e6    # penalty on the change of curvature of the path
    smooth_vertical: float = 3e7
    hold_missing_frames: int = 4   # retain the last observed path on short cloud dropouts

    def __post_init__(self) -> None:
        if self.hold_missing_frames < 0:
            raise ValueError("track hold_missing_frames must be nonnegative")


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = True
    voxel_near: float = 0.08
    voxel_growth: float = 0.003
    min_neighbors: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.voxel_near) or self.voxel_near <= 0:
            raise ValueError("filter voxel_near must be finite and positive")
        if not math.isfinite(self.voxel_growth) or self.voxel_growth < 0:
            raise ValueError("filter voxel_growth must be finite and nonnegative")
        if self.min_neighbors < 1:
            raise ValueError("filter min_neighbors must be at least 1")


@dataclass(frozen=True)
class EnvelopeConfig:
    # Clearance envelope (gabarit) around the track centre line.
    width: float = 2.1
    height: float = 3.0
    min_height: float = 0.12       # above rail head
    lateral_margin: float = 0.0    # extra safety margin on each side
    edge_zone: float = 0.25        # inner width of the uncertain boundary zone


@dataclass(frozen=True)
class DetectionConfig:
    max_range: float = 200.0
    voxel_near: float = 0.25
    voxel_growth: float = 0.004    # voxel size increase per metre of range
    min_points: int = 4
    min_rings: int = 2
    min_height_extent: float = 0.10
    min_score: float = 0.5
    ground_margin: float = 0.12    # clusters lower than this above local floor are ground
    ceiling_margin: float = 0.35
    wall_attach_margin: float = 0.35
    wall_line_gap: float = 0.5     # beyond the rails: closer to a tracked wall = wall equipment
    transparent_max_range: float = 100.0   # split dual echoes are trusted as glass up to this range
    max_clusters: int = 32
    # low objects on the rails (e.g. 300x300x100 mm): stricter temporal check
    low_max_height: float = 0.45
    low_max_range: float = 80.0
    low_max_lateral: float = 0.95
    low_min_points: int = 3
    low_min_protrusion: float = 0.08  # resolve a 0.10 m object above the local rail head
    low_min_width: float = 0.12  # sub-0.13 m bumps need width beyond the narrow rail-head echo
    low_dense_range: float = 35.0  # short-range body sampling; retain stricter far rail rejection
    low_single_beam_range: float = 50.0  # broad single-beam body evidence; only verified standstill, confirmed slowly
    hanging_max_range: float = 60.0  # thin vertical body crossing the upper envelope boundary


@dataclass(frozen=True)
class TrackingConfig:
    gate: float = 2.0              # association distance along the track, m
    lateral_gate: float = 0.8
    confirm_hits: int = 3          # hits within the window to confirm an obstacle
    confirm_window: int = 4
    drop_misses: int = 4
    low_confirm_hits: int = 6      # low objects: hits within the longer window
    low_confirm_window: int = 8
    low_body_confirm_hits: int = 3  # compact low body with transverse/multi-beam support
    low_body_confirm_window: int = 4
    low_single_confirm_hits: int = 8  # single-beam bodies require a standing train and longer persistence
    low_single_confirm_window: int = 16
    far_distance: float = 60.0     # beyond: the path is less certain, confirm longer
    far_confirm_hits: int = 6
    far_confirm_window: int = 8
    far_lateral_std: float = 0.3
    max_speed: float = 25.0        # m/s (90 km/h)


@dataclass(frozen=True)
class ModelsConfig:
    # The compact forest is NumPy-only; neural inference is an optional extra.
    component_policy: str = "score"
    component_min_score: float = 0.90
    component_min_hits: int = 3
    learned_path: bool = False
    far_path_weights: str = ".models/attention_please/far_bev.pt"
    device: str = "auto"
    path_interval: int = 10
    path_max_sigma: float = 0.35
    path_min_distance: float = 40.0
    path_max_lateral_change: float = 0.60
    path_max_vertical_change: float = 0.20
    path_blend: float = 0.25
    path_confirmations: int = 2

    def __post_init__(self) -> None:
        for name in ("component_min_hits", "path_interval", "path_confirmations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.component_policy not in ("off", "score", "assist"):
            raise ValueError("component_policy must be off, score or assist")
        if not 0.0 <= self.component_min_score <= 1.0:
            raise ValueError("component_min_score must be in [0, 1]")
        if self.component_min_hits < 2 or self.path_interval < 1 or self.path_confirmations < 2:
            raise ValueError("model confirmation needs at least two hits; path_interval must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("model device must be auto, cpu or cuda")
        if not 0.0 < self.path_blend <= 1.0:
            raise ValueError("path_blend must be in (0, 1]")
        for name in ("path_max_sigma", "path_max_lateral_change", "path_max_vertical_change"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.path_min_distance) or self.path_min_distance < 0.0:
            raise ValueError("path_min_distance must be finite and nonnegative")


@dataclass(frozen=True)
class Config:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any] | None) -> "Config":
        return _merge(self, overrides or {})

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as err:  # pragma: no cover
                raise RuntimeError("PyYAML is required for YAML configs") from err
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cfg.with_overrides(data)


def _merge(obj: Any, overrides: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(obj)}
    changes = {}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"section {key} expects a mapping")
            changes[key] = _merge(current, value)
        elif isinstance(current, tuple):
            changes[key] = tuple(value)
        elif isinstance(current, bool):
            changes[key] = bool(value)
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            if isinstance(obj, ModelsConfig) and isinstance(current, int) and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            changes[key] = type(current)(value)
        else:
            changes[key] = value
    return replace(obj, **changes)
