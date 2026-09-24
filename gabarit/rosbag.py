"""ROS 2 bag (sqlite3 storage) reader and CDR PointCloud2 decoder without ROS.

The same decoder is used by the ROS 2 node only when ``sensor_msgs`` is not
available, so behaviour is identical in both modes.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"
_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


class CdrError(ValueError):
    pass


@dataclass
class RawCloud:
    """Decoded PointCloud2: structured array keeps the original field layout."""

    stamp: float
    frame_id: str
    height: int
    width: int
    points: np.ndarray  # structured array, one record per point

    @property
    def count(self) -> int:
        return self.height * self.width


def decode_pointcloud2(payload: bytes | memoryview) -> RawCloud:
    buf = bytes(payload) if isinstance(payload, memoryview) else payload
    if len(buf) < 8:
        raise CdrError("payload too short")
    if buf[:2] == b"\x00\x01":
        e = "<"
    elif buf[:2] == b"\x00\x00":
        e = ">"
    else:
        raise CdrError(f"unsupported CDR encapsulation {buf[:2].hex()}")
    o = 4

    def align(n: int) -> None:
        nonlocal o
        o += (-(o - 4)) % n

    def u32() -> int:
        nonlocal o
        align(4)
        if o + 4 > len(buf):
            raise CdrError("truncated message")
        v = struct.unpack_from(e + "I", buf, o)[0]
        o += 4
        return v

    def string() -> str:
        nonlocal o
        n = u32()
        if o + n > len(buf):
            raise CdrError("truncated string")
        v = buf[o:o + max(n - 1, 0)].decode("utf-8", "replace")
        o += n
        return v

    def u8() -> int:
        nonlocal o
        v = buf[o]
        o += 1
        return v

    sec = u32()
    nsec = u32()
    frame_id = string()
    height = u32()
    width = u32()
    fields = []
    for _ in range(u32()):
        name = string()
        offset = u32()
        datatype = u8()
        count = u32()
        if datatype not in _DTYPES or count != 1:
            raise CdrError(f"unsupported point field {name}")
        fields.append((name, offset, datatype))
    big = bool(u8())
    step = u32()
    u32()  # row_step
    size = u32()
    order = ">" if big else "<"
    dtype = np.dtype({
        "names": [f[0] for f in fields],
        "formats": [order + _DTYPES[f[2]] for f in fields],
        "offsets": [f[1] for f in fields],
        "itemsize": step,
    })
    n = height * width
    if size < n * step or o + n * step > len(buf):
        raise CdrError("point data truncated")
    arr = np.frombuffer(buf, dtype=dtype, count=n, offset=o)
    return RawCloud(sec + nsec * 1e-9, frame_id, height, width, arr)


@dataclass(frozen=True)
class BagInfo:
    id: str
    name: str
    path: Path
    topic: str
    topic_id: int
    frame_count: int
    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "topic": self.topic,
            "frames": self.frame_count,
            "duration_s": round(self.duration_s, 2),
        }


def _connect(path: Path) -> sqlite3.Connection:
    uri = "file:" + path.resolve().as_posix() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return con


def open_bag(path: str | Path, root: Path | None = None, topic: str | None = None) -> BagInfo:
    """Open a bag given a directory (with *.db3) or a .db3 file."""
    path = Path(path)
    db = path if path.suffix == ".db3" else next(iter(sorted(path.glob("*.db3"))), None)
    if db is None or not db.exists():
        raise FileNotFoundError(f"no .db3 file in {path}")
    con = _connect(db)
    try:
        rows = con.execute("SELECT id, name, type FROM topics").fetchall()
        cands = [r for r in rows if r[2] == POINTCLOUD2 and (topic is None or r[1] == topic)]
        if not cands:
            raise ValueError(f"no PointCloud2 topic in {db}")
        tid, tname, _ = cands[0]
        cnt, t0, t1 = con.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id=?", (tid,)
        ).fetchone()
    finally:
        con.close()
    bag_dir = db.parent
    ident = bag_dir.relative_to(root).as_posix() if root and bag_dir.is_relative_to(root) else bag_dir.name
    return BagInfo(ident, bag_dir.name, db, tname, int(tid), int(cnt or 0), int(t0 or 0), int(t1 or 0))


def discover_bags(root: str | Path) -> list[BagInfo]:
    root = Path(root).resolve()
    out = []
    if not root.exists():
        return out
    seen = set()
    for db in sorted(root.rglob("*.db3")):
        if db.parent in seen:
            continue
        seen.add(db.parent)
        try:
            info = open_bag(db, root)
        except (OSError, ValueError, sqlite3.Error):
            continue
        if info.frame_count:
            out.append(info)
    return out


class BagReader:
    """Random and sequential access to PointCloud2 messages of one bag."""

    def __init__(self, info: BagInfo):
        self.info = info
        self._con = _connect(info.path)
        self._ids = [r[0] for r in self._con.execute(
            "SELECT id FROM messages WHERE topic_id=? ORDER BY timestamp, id", (info.topic_id,))]

    def __len__(self) -> int:
        return len(self._ids)

    def read(self, index: int) -> tuple[int, RawCloud]:
        if not 0 <= index < len(self._ids):
            raise IndexError(index)
        ts, data = self._con.execute(
            "SELECT timestamp, data FROM messages WHERE id=?", (self._ids[index],)).fetchone()
        return int(ts), decode_pointcloud2(data)

    def iterate(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, int, RawCloud]]:
        stop = len(self._ids) if stop is None else min(stop, len(self._ids))
        for i in range(start, stop, step):
            ts, cloud = self.read(i)
            yield i, ts, cloud

    def close(self) -> None:
        self._con.close()
