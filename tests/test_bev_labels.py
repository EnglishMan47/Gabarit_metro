from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from webapp.bev import BevSpec
from webapp.bev_labels import load_frame_boxes, rasterize_boxes


class BevLabelTests(unittest.TestCase):
    def test_loads_only_relevant_in_gauge_annotations(self) -> None:
        payload = {
            "labels": {
                "primary_person": {
                    "frames": [
                        {"frame": 1, "inside_gauge": False, "min_m": {}, "max_m": {}},
                        {"frame": 2, "inside_gauge": True, "min_m": {"x": 0, "forward": 1}, "max_m": {"x": 1, "forward": 2}},
                    ]
                },
                "secondary_person": {
                    "frames": [{"frame": 2, "min_m": {"x": 0}, "max_m": {"x": 1}}]
                },
                "object": {
                    "frames": [{"frame": 3, "min_m": {"x": 0, "forward": 1}, "max_m": {"x": 1, "forward": 2}}]
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            lookup = load_frame_boxes(path)
        self.assertEqual(sorted(lookup), [2, 3])
        self.assertEqual(lookup[2][0]["name"], "primary_person")
        self.assertEqual(lookup[3][0]["name"], "object")

    def test_rasterization_clips_boxes_and_ignores_outside_box(self) -> None:
        spec = BevSpec(min_x=0, max_x=2, min_forward=0, max_forward=2, resolution_x=1, resolution_forward=1)
        mask = rasterize_boxes(
            [
                {"min_m": {"x": 0.1, "forward": 0.1}, "max_m": {"x": 0.9, "forward": 0.9}},
                {"min_m": {"x": 5.0, "forward": 5.0}, "max_m": {"x": 6.0, "forward": 6.0}},
            ],
            spec,
        )
        np.testing.assert_array_equal(mask, [[1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
