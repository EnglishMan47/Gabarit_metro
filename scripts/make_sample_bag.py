"""Write a small ROS 2 bag (sqlite3) with a synthetic tunnel for smoke runs.

The clouds use the 26-byte point layout announced by the organizers
(x, y, z, intensity: float32; ring: uint16; timestamp: float64). The first
frames are an empty tunnel, then an object stands on the track at --distance m
and a second one stands beside the envelope (it must be ignored).

    python scripts/make_sample_bag.py artifacts/sample_bag
    python -m gabarit run artifacts/sample_bag
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sim import encode_pointcloud2, tunnel_scan  # noqa: E402

TOPIC = "/sensing/lidar/hesai128/pointcloud"
QOS = ("- history: 1\n  depth: 10\n  reliability: 1\n  durability: 2\n  deadline:\n    sec: 9223372036\n"
       "    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n"
       "  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n"
       "  avoid_ros_namespace_conventions: false")
T0 = 1_790_000_000_000_000_000   # ns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="bag directory to create")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--clear-frames", type=int, default=8, help="empty tunnel before the object appears")
    ap.add_argument("--distance", type=float, default=45.0, help="object on the track, m")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / f"{args.out.name}_0.db3"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL,
                            serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL,
                              timestamp INTEGER NOT NULL, data BLOB NOT NULL);
        CREATE INDEX timestamp_idx ON messages (timestamp ASC);
    """)
    con.execute("INSERT INTO topics VALUES (1, ?, 'sensor_msgs/msg/PointCloud2', 'cdr', ?)", (TOPIC, QOS))
    # (lateral centre, forward start, width, length, bottom, top) relative to the rail head
    on_track = (0.0, args.distance, 0.5, 0.4, 0.0, 1.7)      # person-sized, on the track axis
    beside = (1.9, 30.0, 0.4, 0.4, 0.0, 1.5)                 # between the envelope and the wall
    for k in range(args.frames):
        ts = T0 + k * 100_000_000                            # 10 Hz
        boxes = [beside] + ([on_track] if k >= args.clear_frames else [])
        raw = tunnel_scan(boxes=boxes, stamp=ts * 1e-9)
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)",
                    (ts, encode_pointcloud2(raw)))
    con.commit()
    con.close()

    dur = (args.frames - 1) * 100_000_000
    (args.out / "metadata.yaml").write_text(f"""rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  duration:
    nanoseconds: {dur}
  starting_time:
    nanoseconds_since_epoch: {T0}
  message_count: {args.frames}
  topics_with_message_count:
    - topic_metadata:
        name: {TOPIC}
        type: sensor_msgs/msg/PointCloud2
        serialization_format: cdr
        offered_qos_profiles: {QOS!r}
      message_count: {args.frames}
  compression_format: ""
  compression_mode: ""
  relative_file_paths:
    - {db.name}
  files:
    - path: {db.name}
      starting_time:
        nanoseconds_since_epoch: {T0}
      duration:
        nanoseconds: {dur}
      message_count: {args.frames}
""".replace("'", '"'), encoding="utf-8")
    print(f"{db}: {args.frames} frames, object on the track at {args.distance} m from frame {args.clear_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
