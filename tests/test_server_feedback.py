from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webapp.server import Application


class FeedbackTests(unittest.TestCase):
    def test_feedback_is_listed_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = Application(root / "bags", root / "feedback.jsonl")
            saved = app.save_feedback({
                "bag": "sample.db3",
                "frame_index": 12,
                "decision": "confirmed",
                "nearest_distance_m": 18.4,
            })
            self.assertEqual(app.feedback_records(), [saved])
            json_body, json_type, _ = app.feedback_export("json")
            csv_body, csv_type, _ = app.feedback_export("csv")
            self.assertEqual(json.loads(json_body), [saved])
            self.assertIn("application/json", json_type)
            self.assertIn("text/csv", csv_type)
            self.assertIn("sample.db3", csv_body.decode("utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
