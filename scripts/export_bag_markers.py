#!/usr/bin/env python3
"""Export organizer MarkerArray messages without feeding labels to the detector.

Output is one JSON object per recorded MarkerArray. The offline decoder is
verified against ROS serialization; no detector or ROS runtime is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.rosbag import discover_bags  # noqa: E402
from gabarit.annotation_sources import marker_records  # noqa: E402


def marker_dict(marker) -> dict:
    return {
        "id": marker.id, "ns": marker.ns, "type": marker.type, "action": marker.action,
        "frame_id": marker.header.frame_id,
        "stamp": marker.header.stamp.sec + marker.header.stamp.nanosec * 1e-9,
        "position": [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
        "quaternion": [marker.pose.orientation.x, marker.pose.orientation.y,
                       marker.pose.orientation.z, marker.pose.orientation.w],
        "scale": [marker.scale.x, marker.scale.y, marker.scale.z],
        "color": [marker.color.r, marker.color.g, marker.color.b, marker.color.a],
        "text": marker.text,
        "points": [[point.x, point.y, point.z] for point in marker.points],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="bag directory or root with SQLite bags")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    bags = discover_bags(args.data)
    if not bags:
        parser.error("No PointCloud2 SQLite bags found")
    args.out.mkdir(parents=True, exist_ok=True)
    for bag in bags:
        # Preserve relative identities: bags with the same basename cannot
        # overwrite one another's annotations.
        output = args.out / (bag.id + ".markers.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        records = marker_records(bag)
        with output.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{bag.id}: {len(records)} MarkerArray messages -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
