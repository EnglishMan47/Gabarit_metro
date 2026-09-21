from __future__ import annotations

import unittest

import numpy as np

from webapp.config import DetectorConfig
from webapp.detector import detect_obstacles, estimate_floor


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectorConfig(
            corridor_center_x=0.0,
            corridor_width=2.2,
            fallback_floor_z=-2.0,
            floor_min_points=20,
            min_cluster_points=(8, 8, 8, 8),
        )

    def test_detects_compact_cluster_inside_corridor(self) -> None:
        rng = np.random.default_rng(4)
        floor = np.column_stack(
            (
                rng.uniform(-1.0, 1.0, 400),
                rng.uniform(1.0, 40.0, 400),
                rng.normal(-2.0, 0.015, 400),
            )
        )
        obstacle = rng.normal((0.25, 20.0, -1.1), (0.12, 0.12, 0.35), (120, 3))
        result = detect_obstacles(np.vstack((floor, obstacle)).astype(np.float32), self.config)
        self.assertTrue(result["obstacle_detected"])
        self.assertGreaterEqual(len(result["objects"]), 1)
        self.assertAlmostEqual(result["nearest_distance_m"], 20.0, delta=1.0)

    def test_ignores_cluster_outside_corridor(self) -> None:
        rng = np.random.default_rng(5)
        outside = rng.normal((3.0, 12.0, -1.0), (0.1, 0.1, 0.3), (150, 3)).astype(np.float32)
        result = detect_obstacles(outside, self.config)
        self.assertFalse(result["obstacle_detected"])
        self.assertEqual(result["objects"], [])

    def test_ignores_floor_returns(self) -> None:
        rng = np.random.default_rng(6)
        floor = np.column_stack(
            (
                rng.uniform(-1.0, 1.0, 600),
                rng.uniform(1.0, 80.0, 600),
                rng.normal(-2.0, 0.02, 600),
            )
        ).astype(np.float32)
        result = detect_obstacles(floor, self.config)
        self.assertFalse(result["obstacle_detected"])

    def test_estimates_sloped_floor_locally(self) -> None:
        rng = np.random.default_rng(7)
        forward = rng.uniform(3.0, 80.0, 12_000)
        expected = -2.2 + forward * 0.004
        floor = np.column_stack((rng.uniform(-1.4, 1.4, len(forward)), forward, expected + rng.normal(0, 0.01, len(forward))))
        profile = estimate_floor(floor.astype(np.float32), self.config)
        measured = np.asarray(profile["z"])
        target = -2.2 + np.asarray(profile["centers"]) * 0.004
        self.assertGreater(profile["coverage"], 0.8)
        self.assertLess(float(np.median(np.abs(measured[2:-4] - target[2:-4]))), 0.06)


if __name__ == "__main__":
    unittest.main()
