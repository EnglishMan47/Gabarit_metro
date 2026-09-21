#!/usr/bin/env python3
"""Build a recurring-occupancy model from obstacle-free lidar bags."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.bag_store import BagStore
from webapp.config import DEFAULT_CONFIG_PATH, load_config
from webapp.detector import estimate_floor, floor_for_points
from webapp.scene_model import encode_background_voxels, save_scene_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/for_hackathon"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("models/normal_scene.npz"))
    parser.add_argument("--step", type=int, default=5, help="sample every Nth frame")
    parser.add_argument("--min-rate", type=float, default=0.02)
    parser.add_argument("--exclude-bag", action="append", default=[], help="hold out this bag from model fitting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step < 1 or not 0 <= args.min_rate <= 1:
        raise SystemExit("step must be positive and min-rate must be in 0..1")
    config = load_config(args.config)
    store = BagStore(args.data_root)
    bags = [item for item in store.list_bags() if "obstacle" not in item["name"].lower() and item["name"] not in args.exclude_bag]
    if not bags:
        raise SystemExit("no obstacle-free bags found")
    counter: Counter[int] = Counter()
    sampled_frames = 0
    source_frames: dict[str, int] = {}
    for bag in bags:
        bag_samples = 0
        for frame in store.iter_frames(bag["id"], step=args.step):
            points = frame["points"]
            floor = estimate_floor(points, config)
            heights = points[:, 2] - floor_for_points(points, floor, config)
            half = config.corridor_width / 2
            mask = (
                (points[:, 0] >= config.corridor_center_x - half)
                & (points[:, 0] <= config.corridor_center_x + half)
                & (points[:, 1] >= config.min_forward)
                & (points[:, 1] <= config.max_forward)
                & (heights >= config.min_height_above_floor)
                & (heights <= config.max_height_above_floor)
            )
            codes = np.unique(encode_background_voxels(points[mask], heights[mask], config))
            counter.update(int(code) for code in codes)
            sampled_frames += 1
            bag_samples += 1
            if sampled_frames % 50 == 0:
                print(f"sampled {sampled_frames} frames; occupied voxels={len(counter)}")
        source_frames[bag["name"]] = bag_samples
    min_frame_count = max(3, int(np.ceil(sampled_frames * args.min_rate)))
    retained = [(code, count) for code, count in counter.items() if count >= min_frame_count]
    codes = np.asarray([item[0] for item in retained], dtype=np.int64)
    counts = np.asarray([item[1] for item in retained], dtype=np.int32)
    save_scene_model(args.output, codes, counts, sampled_frames, min_frame_count, config)
    metadata = {
        "sampled_frames": sampled_frames,
        "step": args.step,
        "min_rate": args.min_rate,
        "min_frame_count": min_frame_count,
        "retained_voxels": int(len(codes)),
        "sources": source_frames,
        "config_version": config.version,
        "excluded_bags": args.exclude_bag,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
