"""Read PointCloud2 frames directly from ROS 2 SQLite3 bag files."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from time import perf_counter

from scripts.inspect_pointcloud_bag import decode_pointcloud2, sqlite_uri
from webapp.detector import normalize_xyz


@dataclass(frozen=True)
class BagInfo:
    id: str
    name: str
    path: Path
    frame_count: int
    duration_s: float
    topic: str
    message_type: str
    topic_id: int

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "topic": self.topic,
            "message_type": self.message_type,
        }


class BagStore:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self._bags: dict[str, BagInfo] = {}
        self.refresh()

    def refresh(self) -> None:
        bags: dict[str, BagInfo] = {}
        if not self.data_root.exists():
            self._bags = bags
            return
        for path in sorted(self.data_root.rglob("*.db3")):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(self.data_root).as_posix()
                connection = sqlite3.connect(sqlite_uri(resolved), uri=True)
                try:
                    topic_row = connection.execute(
                        "SELECT name, type, id FROM topics WHERE type = 'sensor_msgs/msg/PointCloud2' ORDER BY id LIMIT 1"
                    ).fetchone()
                    if topic_row is None:
                        continue
                    count, first, last = connection.execute(
                        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id = ?",
                        (topic_row[2],),
                    ).fetchone()
                finally:
                    connection.close()
                if not count or topic_row is None:
                    continue
                bags[relative] = BagInfo(
                    id=relative,
                    name=path.parent.name,
                    path=resolved,
                    frame_count=int(count),
                    duration_s=(int(last) - int(first)) / 1_000_000_000,
                    topic=str(topic_row[0]),
                    message_type=str(topic_row[1]),
                    topic_id=int(topic_row[2]),
                )
            except (OSError, sqlite3.Error, ValueError):
                continue
        self._bags = bags

    def list_bags(self) -> list[dict[str, Any]]:
        return [bag.public() for bag in self._bags.values()]

    def get(self, bag_id: str) -> BagInfo:
        try:
            return self._bags[bag_id]
        except KeyError as error:
            raise KeyError(f"unknown bag: {bag_id}") from error

    def read_frame(self, bag_id: str, index: int) -> dict[str, Any]:
        bag = self.get(bag_id)
        if index < 0 or index >= bag.frame_count:
            raise IndexError(f"frame {index} is outside 0..{bag.frame_count - 1}")
        connection = sqlite3.connect(sqlite_uri(bag.path), uri=True)
        try:
            row = connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY id LIMIT 1 OFFSET ?", (bag.topic_id, index)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise IndexError(f"frame {index} not found")
        timestamp, payload = row
        decode_started = perf_counter()
        cloud = decode_pointcloud2(payload)
        points, intensity = normalize_xyz(cloud.numpy_points())
        return {
            "bag": bag,
            "index": index,
            "bag_timestamp_ns": int(timestamp),
            "header_timestamp_ns": cloud.sec * 1_000_000_000 + cloud.nanosec,
            "frame_id": cloud.frame_id,
            "raw_point_count": cloud.point_count,
            "valid_point_count": int(len(points)),
            "points": points,
            "intensity": intensity,
            "decode_ms": (perf_counter() - decode_started) * 1000,
        }

    def iter_frames(
        self, bag_id: str, *, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[dict[str, Any]]:
        bag = self.get(bag_id)
        if step <= 0:
            raise ValueError("step must be positive")
        final = bag.frame_count if stop is None else min(stop, bag.frame_count)
        if start < 0 or start >= final:
            return
        connection = sqlite3.connect(sqlite_uri(bag.path), uri=True)
        try:
            rows = connection.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY id", (bag.topic_id,))
            for index, (timestamp, payload) in enumerate(rows):
                if index >= final:
                    break
                if index < start or index >= final or (index - start) % step:
                    continue
                decode_started = perf_counter()
                cloud = decode_pointcloud2(payload)
                points, intensity = normalize_xyz(cloud.numpy_points())
                yield {
                    "bag": bag,
                    "index": index,
                    "bag_timestamp_ns": int(timestamp),
                    "header_timestamp_ns": cloud.sec * 1_000_000_000 + cloud.nanosec,
                    "frame_id": cloud.frame_id,
                    "raw_point_count": cloud.point_count,
                    "valid_point_count": int(len(points)),
                    "points": points,
                    "intensity": intensity,
                    "decode_ms": (perf_counter() - decode_started) * 1000,
                }
        finally:
            connection.close()
