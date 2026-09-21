#!/usr/bin/env python3
"""Run a small deterministic threshold search on separated sampled frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_detector import closest_match
from webapp.bag_store import BagStore
from webapp.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from webapp.detector import detect_obstacles
from webapp.scene_model import NormalSceneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/for_hackathon"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--annotations", type=Path, default=Path("annotations/doubleT_obstacle.json"))
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/tuning.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_config(args.config)
    model = NormalSceneModel.load(PROJECT_ROOT / base.background_model_path)
    store = BagStore(args.data_root)
    labels = json.loads(args.annotations.read_text(encoding="utf-8"))["labels"]
    primary = {int(item["frame"]): item for item in labels["primary_person"]["frames"]}
    fixed = {int(item["frame"]): item for item in labels["object_on_rails"]["frames"]}
    results = []
    for threshold in (0.45, 0.50, 0.55):
        config = base.with_overrides(background_reject_fraction=threshold)
        expected = hits = negative_frames = negative_alarms = 0
        for bag in store.list_bags():
            for frame in store.iter_frames(bag["id"], step=args.step):
                detected = detect_obstacles(frame["points"], config, model)
                if bag["name"] == "doubleT_obstacle":
                    for lookup, tolerance in ((primary, 2.0), (fixed, 1.5)):
                        label = lookup.get(frame["index"])
                        if label and (lookup is not primary or label.get("inside_gauge")):
                            expected += 1
                            matched, _ = closest_match(detected["objects"], label, tolerance)
                            hits += int(matched)
                else:
                    negative_frames += 1
                    negative_alarms += int(detected["raw_detected"])
        results.append({
            "background_reject_fraction": threshold,
            "positive_recall": hits / expected if expected else None,
            "positive_hits": hits,
            "positive_expected": expected,
            "negative_raw_alarm_rate": negative_alarms / negative_frames if negative_frames else None,
            "negative_raw_alarm_frames": negative_alarms,
            "negative_frames": negative_frames,
        })
    ranked = sorted(results, key=lambda item: (-item["positive_recall"], item["negative_raw_alarm_rate"], -item["background_reject_fraction"]))
    report = {
        "method": "fixed-grid-v1",
        "step": args.step,
        "search_space": {"background_reject_fraction": [0.45, 0.50, 0.55]},
        "selection_order": ["positive_recall_desc", "negative_raw_alarm_rate_asc"],
        "selected": ranked[0],
        "results": results,
        "note": "Adjacent sampled frames are used for engineering tuning only; final metrics use step=1.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
