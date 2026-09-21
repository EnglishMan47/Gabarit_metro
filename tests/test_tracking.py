from __future__ import annotations

import unittest

from webapp.config import DetectorConfig
from webapp.tracking import TrackManager


def candidate(forward: float = 20.0) -> dict:
    return {
        "center_m": {"x": 0.1, "forward": forward, "z": -1.0},
        "distance_surface_m": forward - 0.3,
    }


class TrackingTests(unittest.TestCase):
    def test_confirmation_requires_consecutive_observations(self) -> None:
        tracker = TrackManager(DetectorConfig())
        tracker.update(0, [candidate()])
        tracker.update(1, [])
        tracker.update(2, [candidate()])
        self.assertFalse(tracker.update(3, [candidate()])["event_detected"])
        self.assertTrue(tracker.update(4, [candidate()])["event_detected"])
        self.assertEqual(tracker.finalize()[0]["confirmed_frame"], 4)

    def test_displayed_distance_uses_latest_observation(self) -> None:
        tracker = TrackManager(DetectorConfig(track_confirm_frames=1))
        tracker.update(0, [candidate(20)])
        result = tracker.update(1, [candidate(21)])
        self.assertAlmostEqual(result["event_nearest_distance_m"], 20.7)
        self.assertAlmostEqual(tracker.finalize()[0]["nearest_distance_m"], 19.7)

    def test_confirms_on_second_frame_and_clears_with_hysteresis(self) -> None:
        tracker = TrackManager(DetectorConfig(track_confirm_frames=2, track_clear_frames=2))
        first = tracker.update(0, [candidate()])
        second = tracker.update(1, [candidate(20.1)])
        one_missing = tracker.update(2, [])
        two_missing = tracker.update(3, [])
        self.assertFalse(first["event_detected"])
        self.assertTrue(second["event_detected"])
        self.assertTrue(one_missing["event_detected"])
        self.assertFalse(two_missing["event_detected"])

    def test_expected_step_keeps_track_in_sampled_evaluation(self) -> None:
        tracker = TrackManager(DetectorConfig(track_confirm_frames=2), expected_step=10)
        tracker.update(0, [candidate()])
        result = tracker.update(10, [candidate(20.2)])
        self.assertTrue(result["event_detected"])
        self.assertEqual(result["objects"][0]["track_id"], 1)


if __name__ == "__main__":
    unittest.main()
