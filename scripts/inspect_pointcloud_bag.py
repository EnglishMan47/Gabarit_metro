#!/usr/bin/env python3
"""Read and summarize ROS 2 PointCloud2 messages stored in SQLite3 bags.

The script intentionally avoids ROS imports so that the supplied bags can be
checked on Windows and consumed directly by the local web application.
It supports CDR-serialized sensor_msgs/msg/PointCloud2 messages and writes a
JSON report suitable for comparing the two topic/frame variants in the data.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POINT_FIELD_DTYPES: dict[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


class CdrError(ValueError):
    """Raised when a serialized message is malformed or unsupported."""


class CdrReader:
    def __init__(self, payload: bytes):
        if len(payload) < 4:
            raise CdrError("CDR payload is shorter than its encapsulation header")
        representation = payload[:2]
        if representation == b"\x00\x01":
            self.endian = "<"
        elif representation == b"\x00\x00":
            self.endian = ">"
        else:
            raise CdrError(f"unsupported CDR representation: {representation.hex()}")
        self.payload = payload
        self.view = memoryview(payload)
        self.offset = 4

    def align(self, size: int) -> None:
        self.offset += (-self.offset) % size

    def unpack(self, code: str, alignment: int) -> int | float:
        self.align(alignment)
        size = struct.calcsize(code)
        if self.offset + size > len(self.payload):
            raise CdrError("unexpected end of CDR payload")
        value = struct.unpack_from(self.endian + code, self.payload, self.offset)[0]
        self.offset += size
        return value

    def u8(self) -> int:
        return int(self.unpack("B", 1))

    def u32(self) -> int:
        return int(self.unpack("I", 4))

    def i32(self) -> int:
        return int(self.unpack("i", 4))

    def string(self) -> str:
        length = self.u32()
        if length == 0 or self.offset + length > len(self.payload):
            raise CdrError(f"invalid CDR string length: {length}")
        raw = self.payload[self.offset : self.offset + length]
        self.offset += length
        if raw[-1] != 0:
            raise CdrError("CDR string is not NUL-terminated")
        return raw[:-1].decode("utf-8")

    def byte_sequence(self) -> memoryview:
        length = self.u32()
        end = self.offset + length
        if end > len(self.payload):
            raise CdrError(f"byte sequence exceeds payload: {length} bytes")
        result = self.view[self.offset:end]
        self.offset = end
        return result


@dataclass(frozen=True)
class PointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class PointCloud2:
    sec: int
    nanosec: int
    frame_id: str
    height: int
    width: int
    fields: tuple[PointField, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: memoryview
    is_dense: bool

    @property
    def point_count(self) -> int:
        return self.height * self.width

    def numpy_points(self) -> np.ndarray:
        byte_order = ">" if self.is_bigendian else "<"
        names: list[str] = []
        formats: list[str] = []
        offsets: list[int] = []
        for field in self.fields:
            if field.datatype not in POINT_FIELD_DTYPES:
                raise CdrError(f"unsupported PointField datatype: {field.datatype}")
            if field.count != 1:
                raise CdrError(f"array PointField is unsupported: {field.name}[{field.count}]")
            names.append(field.name)
            formats.append(byte_order + POINT_FIELD_DTYPES[field.datatype])
            offsets.append(field.offset)
        dtype = np.dtype(
            {"names": names, "formats": formats, "offsets": offsets, "itemsize": self.point_step}
        )
        expected = self.point_count * self.point_step
        if len(self.data) < expected:
            raise CdrError(f"point data is truncated: {len(self.data)} < {expected}")
        return np.frombuffer(self.data[:expected], dtype=dtype, count=self.point_count)


def decode_pointcloud2(payload: bytes) -> PointCloud2:
    reader = CdrReader(payload)
    sec = reader.i32()
    nanosec = reader.u32()
    frame_id = reader.string()
    height = reader.u32()
    width = reader.u32()
    field_count = reader.u32()
    fields = []
    for _ in range(field_count):
        fields.append(
            PointField(
                name=reader.string(),
                offset=reader.u32(),
                datatype=reader.u8(),
                count=reader.u32(),
            )
        )
    is_bigendian = bool(reader.u8())
    point_step = reader.u32()
    row_step = reader.u32()
    data = reader.byte_sequence()
    is_dense = bool(reader.u8())
    return PointCloud2(
        sec=sec,
        nanosec=nanosec,
        frame_id=frame_id,
        height=height,
        width=width,
        fields=tuple(fields),
        is_bigendian=is_bigendian,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_dense=is_dense,
    )


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite": 0, "min": None, "p01": None, "median": None, "p99": None, "max": None}
    quantiles = np.quantile(finite, [0.01, 0.5, 0.99])
    return {
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": float(np.max(finite)),
    }


def summarize_cloud(cloud: PointCloud2, max_points: int) -> dict[str, Any]:
    points = cloud.numpy_points()
    missing = {name for name in ("x", "y", "z") if name not in points.dtype.names}
    if missing:
        raise CdrError(f"missing coordinate fields: {sorted(missing)}")
    stride = max(1, math.ceil(points.size / max_points))
    sample = points[::stride]
    x = sample["x"].astype(np.float64, copy=False)
    y = sample["y"].astype(np.float64, copy=False)
    z = sample["z"].astype(np.float64, copy=False)
    finite_xyz = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    zero_xyz = (x == 0) & (y == 0) & (z == 0)
    nonzero_xyz = finite_xyz & ~zero_xyz
    radius = np.sqrt(x[nonzero_xyz] ** 2 + y[nonzero_xyz] ** 2 + z[nonzero_xyz] ** 2)
    return {
        "header_time": cloud.sec + cloud.nanosec / 1_000_000_000,
        "frame_id": cloud.frame_id,
        "height": cloud.height,
        "width": cloud.width,
        "point_count": cloud.point_count,
        "sample_stride": stride,
        "sample_count": int(sample.size),
        "point_step": cloud.point_step,
        "row_step": cloud.row_step,
        "is_bigendian": cloud.is_bigendian,
        "is_dense": cloud.is_dense,
        "fields": [field.__dict__ for field in cloud.fields],
        "finite_xyz": int(np.count_nonzero(finite_xyz)),
        "nonzero_xyz": int(np.count_nonzero(nonzero_xyz)),
        "zero_xyz": int(np.count_nonzero(zero_xyz)),
        "x": finite_summary(x[nonzero_xyz]),
        "y": finite_summary(y[nonzero_xyz]),
        "z": finite_summary(z[nonzero_xyz]),
        "radius": finite_summary(radius),
    }


def sqlite_uri(path: Path) -> str:
    return "file:" + path.resolve().as_posix() + "?mode=ro"


def sample_offsets(message_count: int, frame_samples: int) -> list[int]:
    if message_count <= 0:
        return []
    count = min(message_count, max(1, frame_samples))
    return sorted({round(i * (message_count - 1) / max(1, count - 1)) for i in range(count)})


def inspect_database(db_path: Path, frame_samples: int, max_points: int) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_uri(db_path), uri=True)
    try:
        topics = connection.execute(
            "SELECT id, name, type, serialization_format FROM topics ORDER BY id"
        ).fetchall()
        point_topics = [topic for topic in topics if topic[2] == "sensor_msgs/msg/PointCloud2"]
        if not point_topics:
            raise CdrError("bag has no sensor_msgs/msg/PointCloud2 topic")
        topic_ids = [topic[0] for topic in point_topics]
        placeholders = ",".join("?" for _ in topic_ids)
        message_count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM messages WHERE topic_id IN ({placeholders})", topic_ids
            ).fetchone()[0]
        )
        frames = []
        for offset in sample_offsets(message_count, frame_samples):
            # Selecting a BLOB while sorting by timestamp can make SQLite build a
            # multi-gigabyte temporary table. Resolve the small integer primary
            # key first, then fetch exactly one payload.
            message_id = connection.execute(
                f"SELECT id FROM messages WHERE topic_id IN ({placeholders}) "
                "ORDER BY id LIMIT 1 OFFSET ?",
                (*topic_ids, offset),
            ).fetchone()[0]
            timestamp, payload = connection.execute(
                "SELECT timestamp, data FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            summary = summarize_cloud(decode_pointcloud2(payload), max_points=max_points)
            summary["message_index"] = offset
            summary["bag_timestamp_ns"] = int(timestamp)
            frames.append(summary)
        return {
            "bag": db_path.parent.name,
            "database": str(db_path),
            "message_count": message_count,
            "topics": [
                {"id": topic[0], "name": topic[1], "type": topic[2], "serialization": topic[3]}
                for topic in topics
            ],
            "sampled_frames": frames,
        }
    finally:
        connection.close()


def find_databases(paths: Iterable[Path]) -> list[Path]:
    databases: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".db3":
            databases.add(path.resolve())
        elif path.is_dir():
            databases.update(candidate.resolve() for candidate in path.rglob("*.db3"))
    return sorted(databases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help=".db3 file or directory containing bags")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--frames", type=int, default=3, help="number of evenly spaced frames per bag")
    parser.add_argument(
        "--max-points", type=int, default=200_000, help="maximum sampled points per summarized frame"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.max_points < 1:
        raise SystemExit("--frames and --max-points must be positive")
    databases = find_databases(args.inputs)
    if not databases:
        raise SystemExit("no .db3 files found")
    report = {
        "format": "pointcloud2-bag-inspection-v1",
        "database_count": len(databases),
        "bags": [inspect_database(path, args.frames, args.max_points) for path in databases],
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Wrote {args.output} ({len(databases)} bags)")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
