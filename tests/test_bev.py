from __future__ import annotations

import unittest

import numpy as np

from webapp.bev import BEV_CHANNELS, DEFAULT_BEV_SPEC, build_bev
from webapp.config import DetectorConfig
from webapp.detector import detect_obstacles


class BevTests(unittest.TestCase):
    def test_builds_finite_fixed_shape_tensor(self) -> None:
        config = DetectorConfig()
        rng = np.random.default_rng(42)
        floor = np.column_stack((
            rng.uniform(-1.5, 1.0, 800),
            rng.uniform(3.0, 80.0, 800),
            rng.normal(-2.3, 0.02, 800),
        )).astype(np.float32)
        obstacle = rng.normal([0.0, 30.0, -1.2], [0.12, 0.18, 0.28], (100, 3)).astype(np.float32)
        points = np.vstack((floor, obstacle))
        intensity = np.linspace(0, 255, len(points), dtype=np.float32)
        detection = detect_obstacles(points, config)
        tensor = build_bev(points, intensity, detection, config)
        self.assertEqual(tensor.shape, (len(BEV_CHANNELS), DEFAULT_BEV_SPEC.height, DEFAULT_BEV_SPEC.width))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(tensor)))
        self.assertGreater(float(tensor[0].max()), 0.0)
        self.assertGreater(int(np.count_nonzero(tensor[5])), 0)

    def test_rejects_wrong_temporal_shape(self) -> None:
        config = DetectorConfig()
        points = np.array([[0.0, 10.0, -2.0]], dtype=np.float32)
        detection = detect_obstacles(points, config)
        with self.assertRaises(ValueError):
            build_bev(points, None, detection, config, temporal_stability=np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
