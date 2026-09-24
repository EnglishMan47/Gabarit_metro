"""Configuration of the detector. Every tunable value lives here.

Values can be overridden from a YAML/JSON file (``config/gabarit.yaml``) or
programmatically with :meth:`Config.with_overrides`.
"""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class EnvelopeConfig:
    # Clearance envelope (gabarit) around the track centre line.
    width: float = 2.2
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
    max_clusters: int = 32
    # low objects on the rails (e.g. 300x300x100 mm): stricter temporal check
    low_max_height: float = 0.45
    low_max_range: float = 80.0
    low_max_lateral: float = 0.95
    low_min_points: int = 3


@dataclass(frozen=True)
class TrackingConfig:
    gate: float = 2.0              # association distance along the track, m
    lateral_gate: float = 0.8
    confirm_hits: int = 3          # hits within the window to confirm an obstacle
    confirm_window: int = 4
    drop_misses: int = 4
    low_confirm_hits: int = 6      # low objects: hits within the longer window
    low_confirm_window: int = 8
    far_distance: float = 60.0     # beyond: the path is less certain, confirm longer
    far_confirm_hits: int = 6
    far_confirm_window: int = 8
    far_lateral_std: float = 0.3
    max_speed: float = 25.0        # m/s (90 km/h)


@dataclass(frozen=True)
class Config:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    # Deceleration used only for the informative stopping-distance estimate.
    braking_deceleration: float = 1.1
    reaction_time: float = 1.5

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
            changes[key] = type(current)(value)
        else:
            changes[key] = value
    return replace(obj, **changes)
