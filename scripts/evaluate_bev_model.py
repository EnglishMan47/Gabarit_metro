#!/usr/bin/env python3
"""Evaluate an exported BEV model without ROS and write transparent split metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/bev-dataset"))
    parser.add_argument("--model", type=Path, default=Path("models/bev_segmentation.onnx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-evaluation.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_arrays(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = probability >= threshold
    actual = truth >= 0.5
    true_positive = int(np.count_nonzero(predicted & actual))
    false_positive = int(np.count_nonzero(predicted & ~actual))
    false_negative = int(np.count_nonzero(~predicted & actual))
    true_negative = int(np.count_nonzero(~predicted & ~actual))
    positive_frames = np.any(actual, axis=(1, 2))
    detected_frames = np.any(predicted & actual, axis=(1, 2))
    return {
        "samples": int(len(actual)),
        "positive_samples": int(np.count_nonzero(positive_frames)),
        "true_positive_cells": true_positive,
        "false_positive_cells": false_positive,
        "false_negative_cells": false_negative,
        "true_negative_cells": true_negative,
        "precision": safe_ratio(true_positive, true_positive + false_positive),
        "recall": safe_ratio(true_positive, true_positive + false_negative),
        "iou": safe_ratio(true_positive, true_positive + false_positive + false_negative),
        "false_positive_rate": safe_ratio(false_positive, false_positive + true_negative),
        "positive_frame_recall": safe_ratio(
            int(np.count_nonzero(detected_frames & positive_frames)), int(np.count_nonzero(positive_frames))
        ),
        "probability_min": float(np.min(probability)),
        "probability_mean": float(np.mean(probability)),
        "probability_max": float(np.max(probability)),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0 or args.batch_size < 1:
        raise SystemExit("threshold must be between 0 and 1 and batch-size must be positive")
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    bags: dict[str, object] = {}
    for shard in manifest["shards"]:
        with np.load(args.dataset / shard["path"]) as data:
            features = np.asarray(data["bev"], dtype=np.float32)
            truth = np.asarray(data["mask"], dtype=np.float32)
        outputs = []
        for start in range(0, len(features), args.batch_size):
            batch = np.ascontiguousarray(features[start : start + args.batch_size])
            outputs.append(np.asarray(session.run(None, {input_name: batch})[0], dtype=np.float32)[:, 0])
        probability = np.concatenate(outputs)
        metrics = evaluate_arrays(probability, truth, args.threshold)
        metrics["split"] = shard["split"]
        bags[shard["bag"]] = metrics
        splits.setdefault(shard["split"], []).append((probability, truth))

    split_metrics = {
        split: evaluate_arrays(
            np.concatenate([item[0] for item in arrays]),
            np.concatenate([item[1] for item in arrays]),
            args.threshold,
        )
        for split, arrays in splits.items()
    }
    result = {
        "format": "metro-lidar-bev-evaluation-v1",
        "model": str(args.model),
        "threshold": args.threshold,
        "development_only": True,
        "limitation": "the only positive run belongs to train; validation and test measure false positives only",
        "splits": split_metrics,
        "bags": bags,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
