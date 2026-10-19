#!/usr/bin/env python3
"""Export organizer MarkerArray messages without feeding labels to the detector.

Run in a ROS 2 environment (e.g. gabarit shell). The normal offline detector
does not need ROS. Output is one JSON object per recorded MarkerArray.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gabarit.rosbag import discover_bags  # noqa: E402


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
    try:
        from rclpy.serialization import deserialize_message
        from visualization_msgs.msg import MarkerArray
    except ImportError:
        parser.error("ROS 2 is required for MarkerArray export. Run inside `gabarit shell`.")
    bags = discover_bags(args.data)
    if not bags:
        parser.error("No PointCloud2 SQLite bags found")
    args.out.mkdir(parents=True, exist_ok=True)
    for bag in bags:
        # Preserve relative identities: bags with the same basename cannot
        # overwrite one another's annotations.
        output = args.out / (bag.id + ".markers.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for part in bag.storage_paths or (bag.path,):
            with sqlite3.connect(part.resolve().as_uri() + "?mode=ro", uri=True) as con:
                topics = con.execute("SELECT id,name FROM topics WHERE type=?",
                                     ("visualization_msgs/msg/MarkerArray",)).fetchall()
                for topic_id, topic in topics:
                    for row_id, stamp, data in con.execute(
                            "SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (topic_id,)):
                        message = deserialize_message(data, MarkerArray)
                        records.append((stamp, str(part), row_id, {
                            "timestamp_ns": stamp, "topic": topic,
                            "markers": [marker_dict(marker) for marker in message.markers],
                        }))
        records.sort(key=lambda item: item[:3])
        with output.open("w", encoding="utf-8") as stream:
            for *_, record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{bag.id}: {len(records)} MarkerArray messages -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
