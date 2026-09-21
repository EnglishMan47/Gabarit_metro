#!/usr/bin/env python3
"""Build versioned BEV training shards from complete lidar runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.bag_store import BagStore
from webapp.bev import DEFAULT_BEV_SPEC, build_bev
from webapp.bev_labels import load_frame_boxes, rasterize_boxes
from webapp.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from webapp.detector import detect_obstacles
from webapp.scene_model import NormalSceneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/for_hackathon"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--annotations", type=Path, default=Path("annotations/doubleT_obstacle.json"))
    parser.add_argument("--positive-bag", default="doubleT_obstacle")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bev-dataset"))
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--limit-per-bag", type=int)
    return parser.parse_args()


def assign_splits(bags: list[dict[str, object]], positive_bag: str) -> dict[str, str]:
    negatives = sorted(str(item["name"]) for item in bags if item["name"] != positive_bag)
    split: dict[str, str] = {positive_bag: "train"}
    for index, name in enumerate(negatives):
        split[name] = "train" if index < max(1, len(negatives) - 2) else "validation" if index == len(negatives) - 2 else "test"
    return split


def main() -> int:
    args = parse_args()
    if args.step < 1 or (args.limit_per_bag is not None and args.limit_per_bag < 1):
        raise SystemExit("step and limit-per-bag must be positive")
    if not args.annotations.is_file():
        raise SystemExit(f"annotation file not found: {args.annotations}")
    config = load_config(args.config)
    model_path = PROJECT_ROOT / config.background_model_path
    scene_model = NormalSceneModel.load(model_path) if model_path.is_file() else None
    store = BagStore(args.data_root)
    bags = store.list_bags()
    if not bags:
        raise SystemExit("no lidar bags found")
    if args.positive_bag not in {item["name"] for item in bags}:
        raise SystemExit(f"positive bag not found: {args.positive_bag}")
    frame_boxes = load_frame_boxes(args.annotations)
    splits = assign_splits(bags, args.positive_bag)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "metro-lidar-bev-dataset-v1",
        "bev": DEFAULT_BEV_SPEC.public(),
        "config_version": config.version,
        "step": args.step,
        "annotation_source": str(args.annotations),
        "annotation_status": "provisional; positive run is development data, not independent validation",
        "shards": [],
    }
    for bag in bags:
        tensors: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        frames: list[int] = []
        for frame in store.iter_frames(bag["id"], step=args.step):
            detection = detect_obstacles(frame["points"], config, scene_model)
            tensors.append(build_bev(frame["points"], frame["intensity"], detection, config, scene_model))
            boxes = frame_boxes.get(frame["index"], []) if bag["name"] == args.positive_bag else []
            masks.append(rasterize_boxes(boxes))
            frames.append(frame["index"])
            if args.limit_per_bag is not None and len(frames) >= args.limit_per_bag:
                break
        shard_path = args.output_dir / f"{bag['name']}.npz"
        np.savez_compressed(
            shard_path,
            bev=np.asarray(tensors, dtype=np.float16),
            mask=np.asarray(masks, dtype=np.uint8),
            frame=np.asarray(frames, dtype=np.int32),
        )
        positive_samples = int(sum(bool(np.any(mask)) for mask in masks))
        entry = {
            "bag": bag["name"],
            "bag_id": bag["id"],
            "split": splits[bag["name"]],
            "samples": len(frames),
            "positive_samples": positive_samples,
            "path": shard_path.name,
        }
        manifest["shards"].append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
