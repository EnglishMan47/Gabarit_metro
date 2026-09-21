"""Normal-scene occupancy model used to reject recurring infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from webapp.config import DetectorConfig


@dataclass(frozen=True)
class NormalSceneModel:
    codes: np.ndarray
    frame_counts: np.ndarray
    sampled_frames: int
    min_frame_count: int
    voxel_x: float
    voxel_forward: float
    voxel_height: float
    corridor_center_x: float = -0.2
    corridor_width: float = 2.2
    max_forward: float = 90.0

    @classmethod
    def load(cls, path: Path) -> "NormalSceneModel":
        data = np.load(path)
        return cls(
            codes=np.asarray(data["codes"], dtype=np.int64),
            frame_counts=np.asarray(data["frame_counts"], dtype=np.int32),
            sampled_frames=int(data["sampled_frames"]),
            min_frame_count=int(data["min_frame_count"]),
            voxel_x=round(float(data["voxel_x"]), 8),
            voxel_forward=round(float(data["voxel_forward"]), 8),
            voxel_height=round(float(data["voxel_height"]), 8),
            corridor_center_x=float(data["corridor_center_x"]) if "corridor_center_x" in data else -0.2,
            corridor_width=float(data["corridor_width"]) if "corridor_width" in data else 2.2,
            max_forward=float(data["max_forward"]) if "max_forward" in data else 90.0,
        )

    def match(self, points: np.ndarray, heights: np.ndarray, config: DetectorConfig) -> np.ndarray:
        if len(self.codes) == 0:
            return np.zeros(len(points), dtype=bool)
        # Codes must always use the training grid, including when the UI changes
        # corridor width or range. Otherwise unrelated cells silently collide.
        grid = config.with_overrides(
            corridor_center_x=self.corridor_center_x, corridor_width=self.corridor_width,
            max_forward=self.max_forward, background_voxel_x=self.voxel_x,
            background_voxel_forward=self.voxel_forward, background_voxel_height=self.voxel_height,
        )
        inside = (
            (np.abs(points[:, 0] - self.corridor_center_x) <= self.corridor_width / 2)
            & (points[:, 1] >= 0) & (points[:, 1] <= self.max_forward)
        )
        codes = encode_background_voxels(points, heights, grid)
        positions = np.searchsorted(self.codes, codes)
        positions = np.minimum(positions, len(self.codes) - 1)
        return inside & (self.codes[positions] == codes)


def encode_background_voxels(
    points: np.ndarray, heights: np.ndarray, config: DetectorConfig
) -> np.ndarray:
    min_x = config.corridor_center_x - config.corridor_width / 2
    ix = np.floor((points[:, 0] - min_x) / config.background_voxel_x).astype(np.int64)
    iy = np.floor(points[:, 1] / config.background_voxel_forward).astype(np.int64)
    iz = np.floor(heights / config.background_voxel_height).astype(np.int64)
    nx = int(np.ceil(config.corridor_width / config.background_voxel_x)) + 4
    nf = int(np.ceil(config.max_forward / config.background_voxel_forward)) + 4
    return ix + nx * (iy + nf * iz)


def save_scene_model(
    path: Path,
    codes: np.ndarray,
    frame_counts: np.ndarray,
    sampled_frames: int,
    min_frame_count: int,
    config: DetectorConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(codes)
    np.savez_compressed(
        path,
        codes=np.asarray(codes, dtype=np.int64)[order],
        frame_counts=np.asarray(frame_counts, dtype=np.int32)[order],
        sampled_frames=np.asarray(sampled_frames, dtype=np.int32),
        min_frame_count=np.asarray(min_frame_count, dtype=np.int32),
        voxel_x=np.asarray(config.background_voxel_x, dtype=np.float32),
        voxel_forward=np.asarray(config.background_voxel_forward, dtype=np.float32),
        voxel_height=np.asarray(config.background_voxel_height, dtype=np.float32),
        corridor_center_x=np.asarray(config.corridor_center_x, dtype=np.float64),
        corridor_width=np.asarray(config.corridor_width, dtype=np.float64),
        max_forward=np.asarray(config.max_forward, dtype=np.float64),
    )
