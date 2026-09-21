"""Geometric obstacle detector for lidar point clouds."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from webapp.config import DetectorConfig
from webapp.scene_model import NormalSceneModel


def normalize_xyz(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert PointCloud2 fields to lateral/forward/vertical coordinates."""
    names = raw.dtype.names or ()
    missing = {name for name in ("x", "y", "z") if name not in names}
    if missing:
        raise ValueError(f"point cloud is missing fields: {sorted(missing)}")
    points = np.column_stack((raw["x"], -raw["y"], raw["z"])).astype(np.float32)
    valid = np.all(np.isfinite(points), axis=1) & np.any(points != 0, axis=1)
    intensity = None
    if "intensity" in names:
        intensity = np.asarray(raw["intensity"], dtype=np.float32)[valid]
    return points[valid], intensity


def estimate_floor(points: np.ndarray, config: DetectorConfig) -> dict[str, Any]:
    """Estimate a smooth local floor height in forward-distance bins."""
    bin_count = max(1, int(np.ceil(config.max_forward / config.floor_bin_size)))
    floor = np.full(bin_count, np.nan, dtype=np.float32)
    counts = np.zeros(bin_count, dtype=np.int32)
    half_width = config.corridor_width / 2 + config.floor_lateral_margin
    usable = (
        (np.abs(points[:, 0] - config.corridor_center_x) <= half_width)
        & (points[:, 1] >= config.min_forward)
        & (points[:, 1] <= config.max_forward)
        & (points[:, 2] >= config.fallback_floor_z - 1.2)
        & (points[:, 2] <= config.fallback_floor_z + 1.0)
    )
    selected = points[usable]
    if len(selected):
        indices = np.floor(selected[:, 1] / config.floor_bin_size).astype(np.int32)
        indices = np.clip(indices, 0, bin_count - 1)
        for index in np.unique(indices):
            values = selected[indices == index, 2]
            counts[index] = len(values)
            if len(values) >= config.floor_min_points:
                floor[index] = np.quantile(values, config.floor_quantile)
    known = np.flatnonzero(np.isfinite(floor))
    coverage = float(len(known) / bin_count)
    if len(known):
        floor = np.interp(np.arange(bin_count), known, floor[known]).astype(np.float32)
    else:
        floor.fill(config.fallback_floor_z)
    if config.floor_smooth_radius > 0:
        source = floor.copy()
        for index in range(bin_count):
            left = max(0, index - config.floor_smooth_radius)
            right = min(bin_count, index + config.floor_smooth_radius + 1)
            floor[index] = np.median(source[left:right])
    centers = (np.arange(bin_count, dtype=np.float32) + 0.5) * config.floor_bin_size
    return {"centers": centers, "z": floor, "counts": counts, "coverage": coverage}


def floor_for_points(points: np.ndarray, floor_profile: dict[str, Any], config: DetectorConfig) -> np.ndarray:
    indices = np.floor(points[:, 1] / config.floor_bin_size).astype(np.int32)
    indices = np.clip(indices, 0, len(floor_profile["z"]) - 1)
    return np.asarray(floor_profile["z"], dtype=np.float32)[indices]


def estimate_path(points: np.ndarray, heights: np.ndarray, config: DetectorConfig) -> dict[str, Any]:
    """Estimate rail midpoint per distance bin, with a fixed-center fallback."""
    bin_count = max(1, int(np.ceil(config.max_forward / config.floor_bin_size)))
    centers = np.full(bin_count, config.corridor_center_x, dtype=np.float32)
    left_rail = np.full(bin_count, np.nan, dtype=np.float32)
    right_rail = np.full(bin_count, np.nan, dtype=np.float32)
    confident = np.zeros(bin_count, dtype=bool)
    usable = (
        (points[:, 1] >= config.min_forward)
        & (points[:, 1] <= config.max_forward)
        & (heights >= config.rail_min_height)
        & (heights <= config.rail_max_height_for_path)
        & (points[:, 0] >= config.corridor_center_x - config.rail_search_outer)
        & (points[:, 0] <= config.corridor_center_x + config.rail_search_outer)
    )
    rail_points = points[usable]
    longitudinal = np.floor(rail_points[:, 1] / config.floor_bin_size).astype(np.int32)
    for index in range(bin_count):
        segment_x = rail_points[longitudinal == index, 0]
        rails: list[float] = []
        for low, high in (
            (
                config.corridor_center_x - config.rail_search_outer,
                config.corridor_center_x - config.rail_search_inner,
            ),
            (
                config.corridor_center_x + config.rail_search_inner,
                config.corridor_center_x + config.rail_search_outer,
            ),
        ):
            values = segment_x[(segment_x >= low) & (segment_x <= high)]
            if not len(values):
                rails.append(float("nan"))
                continue
            edges = np.arange(low, high + config.rail_histogram_bin, config.rail_histogram_bin)
            counts, edges = np.histogram(values, bins=edges)
            peak = int(np.argmax(counts))
            rails.append(float((edges[peak] + edges[peak + 1]) / 2) if counts[peak] >= config.rail_min_peak_points else float("nan"))
        left, right = rails
        gauge = right - left
        if np.isfinite(gauge) and config.rail_min_gauge <= gauge <= config.rail_max_gauge:
            estimated = (left + right) / 2
            shift = np.clip(
                estimated - config.corridor_center_x,
                -config.path_max_center_shift,
                config.path_max_center_shift,
            )
            centers[index] = config.corridor_center_x + shift
            left_rail[index], right_rail[index] = left, right
            confident[index] = True
    coverage = float(np.mean(confident))
    source = centers.copy()
    for index in range(bin_count):
        left = max(0, index - config.floor_smooth_radius)
        right = min(bin_count, index + config.floor_smooth_radius + 1)
        local = source[left:right][confident[left:right]]
        if len(local):
            centers[index] = np.median(local)
    if coverage < config.path_min_coverage:
        centers.fill(config.corridor_center_x)
    distances = (np.arange(bin_count, dtype=np.float32) + 0.5) * config.floor_bin_size
    return {
        "distances": distances,
        "centers": centers,
        "left_rail": left_rail,
        "right_rail": right_rail,
        "confident": confident,
        "coverage": coverage,
    }


def path_center_for_points(points: np.ndarray, path_profile: dict[str, Any], config: DetectorConfig) -> np.ndarray:
    indices = np.floor(points[:, 1] / config.floor_bin_size).astype(np.int32)
    indices = np.clip(indices, 0, len(path_profile["centers"]) - 1)
    return np.asarray(path_profile["centers"], dtype=np.float32)[indices]


def _component_labels(keys: np.ndarray, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(int(value) for value in key): index for index, key in enumerate(keys)}
    labels = np.full(len(keys), -1, dtype=np.int32)
    component = 0
    for seed in range(len(keys)):
        if labels[seed] >= 0:
            continue
        labels[seed] = component
        stack = [seed]
        while stack:
            current = stack.pop()
            key = keys[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        neighbour = lookup.get(
                            (int(key[0] + dx), int(key[1] + dy), int(key[2] + dz))
                        )
                        if neighbour is not None and labels[neighbour] < 0:
                            labels[neighbour] = component
                            stack.append(neighbour)
        component += 1
    point_labels = labels[inverse]
    counts = np.bincount(point_labels, minlength=component)
    return point_labels, counts


def _cluster_band(
    points: np.ndarray,
    heights: np.ndarray,
    config: DetectorConfig,
    voxel_size: float,
    minimum_points: int,
    scene_model: NormalSceneModel | None,
    diagnostics: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not len(points):
        return objects, rejected
    keys = np.floor(points / voxel_size).astype(np.int32)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    point_labels, counts = _component_labels(unique_keys, inverse)
    eligible = np.flatnonzero(counts >= minimum_points)
    eligible = eligible[np.argsort(counts[eligible])[::-1]]
    for label in eligible:
        cluster_mask = point_labels == label
        cluster = points[cluster_mask]
        cluster_heights = heights[cluster_mask]
        low = np.min(cluster, axis=0)
        high = np.max(cluster, axis=0)
        size = high - low
        height_low = float(np.min(cluster_heights))
        height_high = float(np.max(cluster_heights))
        reason = None
        background_fraction = 0.0
        if scene_model is not None:
            matches = scene_model.match(cluster, cluster_heights, config)
            background_fraction = float(np.mean(matches)) if len(matches) else 0.0
            if background_fraction >= config.background_reject_fraction:
                reason = "background_occupancy"
        if reason is None and height_high <= config.rail_max_height and size[1] >= config.rail_min_depth:
            reason = "rail_like"
        if reason is None and (
            size[0] > config.max_cluster_width
            or size[1] > config.max_cluster_depth
            or size[2] > config.max_cluster_height
        ):
            reason = "infrastructure_shape"
        if reason is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
            rejected.append(
                {"reason": reason, "point_count": int(len(cluster)), "forward_m": float(np.median(cluster[:, 1]))}
            )
            continue
        center = np.median(cluster, axis=0)
        reasons = ["compact_cluster"]
        if height_low <= 0.20:
            reasons.append("above_floor")
        if scene_model is not None and background_fraction < config.background_reject_fraction:
            reasons.append("background_deviation")
        objects.append(
            {
                "track_id": None,
                "point_count": int(len(cluster)),
                "distance_surface_m": float(np.min(np.linalg.norm(cluster, axis=1))),
                "distance_center_m": float(np.linalg.norm(center)),
                "center_m": {"x": float(center[0]), "forward": float(center[1]), "z": float(center[2])},
                "min_m": {"x": float(low[0]), "forward": float(low[1]), "z": float(low[2])},
                "max_m": {"x": float(high[0]), "forward": float(high[1]), "z": float(high[2])},
                "size_visible_m": {"x": float(size[0]), "forward": float(size[1]), "z": float(size[2])},
                "height_above_floor_m": {"min": height_low, "max": height_high},
                "background_fraction": background_fraction,
                "inside_corridor": True,
                "reasons": reasons,
            }
        )
    return objects, rejected


def detect_obstacles(
    points: np.ndarray,
    config: DetectorConfig,
    scene_model: NormalSceneModel | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    floor_profile = estimate_floor(points, config)
    local_floor = floor_for_points(points, floor_profile, config)
    heights = points[:, 2] - local_floor
    path_profile = estimate_path(points, heights, config)
    local_center = path_center_for_points(points, path_profile, config)
    half_width = config.corridor_width / 2
    corridor_mask = (
        (np.abs(points[:, 0] - local_center) <= half_width)
        & (points[:, 1] >= config.min_forward)
        & (points[:, 1] <= config.max_forward)
        & (heights >= config.min_height_above_floor)
        & (heights <= config.max_height_above_floor)
    )
    candidates = points[corridor_mask]
    candidate_heights = heights[corridor_mask]
    objects: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = {}
    lower = config.min_forward
    for band_index, (upper, voxel_size, minimum_points) in enumerate(zip(
        config.range_bands, config.voxel_sizes, config.min_cluster_points, strict=True
    )):
        if band_index == len(config.range_bands) - 1:
            upper = max(upper, config.max_forward + 0.001)
        band_mask = (candidates[:, 1] >= lower) & (candidates[:, 1] < upper)
        band_objects, band_rejected = _cluster_band(
            candidates[band_mask], candidate_heights[band_mask], config, float(voxel_size),
            int(minimum_points), scene_model, diagnostics,
        )
        objects.extend(band_objects)
        rejected.extend(band_rejected)
        lower = upper
    objects.sort(key=lambda item: item["distance_surface_m"])
    objects = objects[: config.max_objects]
    for index, item in enumerate(objects, start=1):
        item["track_id"] = index
    elapsed_ms = (perf_counter() - started) * 1000
    floor_entries = [
        {"forward_m": float(forward), "z_m": float(z), "point_count": int(count)}
        for forward, z, count in zip(
            floor_profile["centers"], floor_profile["z"], floor_profile["counts"], strict=True
        )
    ]
    path_entries = [
        {
            "forward_m": float(distance),
            "center_x_m": float(center),
            "left_rail_x_m": float(left) if np.isfinite(left) else None,
            "right_rail_x_m": float(right) if np.isfinite(right) else None,
            "rail_confident": bool(confident),
        }
        for distance, center, left, right, confident in zip(
            path_profile["distances"], path_profile["centers"], path_profile["left_rail"],
            path_profile["right_rail"], path_profile["confident"], strict=True
        )
    ]
    return {
        "raw_detected": bool(objects),
        "obstacle_detected": bool(objects),
        "event_detected": bool(objects),
        "nearest_distance_m": objects[0]["distance_surface_m"] if objects else None,
        "candidate_point_count": int(len(candidates)),
        "processing_ms": elapsed_ms,
        "corridor": {
            "center_x_m": config.corridor_center_x,
            "width_m": config.corridor_width,
            "height_m": config.corridor_height,
            "fallback_floor_z_m": config.fallback_floor_z,
            "floor_z_m": float(np.median(floor_profile["z"])),
            "min_forward_m": config.min_forward,
            "max_forward_m": config.max_forward,
        },
        "floor_profile": floor_entries,
        "floor_coverage": floor_profile["coverage"],
        "path_profile": path_entries,
        "path_coverage": path_profile["coverage"],
        "objects": objects,
        "rejected_preview": rejected[:20],
        "diagnostics": {
            "rejected": diagnostics,
            "scene_model_loaded": scene_model is not None,
            "path_fallback_used": path_profile["coverage"] < config.path_min_coverage,
        },
        "config": config.public(),
    }


def visualization_sample(
    points: np.ndarray,
    intensity: np.ndarray | None,
    *,
    max_points: int = 16_000,
    lateral_limit: float = 8.0,
    forward_limit: float = 100.0,
    min_z: float = -3.0,
    max_z: float = 3.0,
) -> list[list[float]]:
    visible = (
        (np.abs(points[:, 0]) <= lateral_limit)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= forward_limit)
        & (points[:, 2] >= min_z)
        & (points[:, 2] <= max_z)
    )
    selected = points[visible]
    selected_intensity = intensity[visible] if intensity is not None else None
    if len(selected) > max_points:
        indices = np.linspace(0, len(selected) - 1, max_points, dtype=np.int64)
        selected = selected[indices]
        if selected_intensity is not None:
            selected_intensity = selected_intensity[indices]
    if selected_intensity is None:
        selected_intensity = np.zeros(len(selected), dtype=np.float32)
    combined = np.column_stack((selected, selected_intensity))
    # Convert to Python floats before rounding: float32.tolist() otherwise
    # expands three-decimal values to long binary approximation strings.
    return np.round(combined.astype(np.float64), 3).tolist()
