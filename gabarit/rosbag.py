"""ROS 2 bag (sqlite3 storage) reader and CDR PointCloud2 decoder without ROS.

The same decoder is used by the ROS 2 node only when ``sensor_msgs`` is not
available, so behaviour is identical in both modes.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

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


def pointcloud2_array(data: bytes | memoryview, fields: Iterable[tuple[str, int, int, int]],
                      height: int, width: int, point_step: int, row_step: int,
                      is_bigendian: bool = False) -> np.ndarray:
    """Decode PointCloud2 records, including row padding and array-valued fields.

    ``fields`` contains (name, offset, ROS datatype, count) tuples. The result
    has one structured record per point; padding bytes never become points.
    Shared by the offline CDR reader and the live ROS adapter.
    """
    if min(height, width, point_step, row_step) < 0:
        raise CdrError("negative PointCloud2 dimensions or strides")
    if height * width and point_step == 0:
        raise CdrError("point_step must be positive for a non-empty cloud")
    if row_step < width * point_step:
        raise CdrError("row_step is shorter than a row of points")
    if memoryview(data).nbytes < height * row_step:
        raise CdrError("point data truncated (including row padding)")
    names, formats, offsets = [], [], []
    order = ">" if is_bigendian else "<"
    for name, offset, datatype, count in fields:
        if datatype not in _DTYPES or count < 1:
            raise CdrError(f"unsupported point field {name}: datatype={datatype}, count={count}")
        if name in names:
            raise CdrError(f"duplicate point field {name}")
        base = np.dtype(order + _DTYPES[datatype])
        if offset < 0 or offset + base.itemsize * count > point_step:
            raise CdrError(f"point field {name} exceeds point_step")
        names.append(name)
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(offset)
    try:
        dtype = np.dtype({"names": names, "formats": formats,
                          "offsets": offsets, "itemsize": point_step})
        if height * width == 0:
            return np.empty(0, dtype=dtype)
        arr = np.ndarray((height, width), dtype=dtype, buffer=data,
                         strides=(row_step, point_step))
        return arr.reshape(-1)
    except (TypeError, ValueError) as err:
        raise CdrError(f"invalid PointCloud2 layout: {err}") from err


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
        if o >= len(buf):
            raise CdrError("truncated message")
        v = buf[o]
        o += 1
        return v

    # builtin_interfaces/Time.sec is signed, unlike the following uint32.
    sec = u32()
    if sec >= 2**31:
        sec -= 2**32
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
        fields.append((name, offset, datatype, count))
    big = bool(u8())
    step = u32()
    row_step = u32()
    size = u32()
    if o + size > len(buf):
        raise CdrError("point data truncated")
    arr = pointcloud2_array(memoryview(buf)[o:o + size], fields, height, width,
                           step, row_step, big)
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
    storage_paths: tuple[Path, ...] = ()

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def public(self) -> dict:
        title, group = bag_title(self.name)
        return {
            "id": self.id,
            "name": self.name,
            "title": title,
            "group": group,
            "topic": self.topic,
            "frames": self.frame_count,
            "duration_s": round(self.duration_s, 2),
            "storage_files": len(self.storage_paths) if self.storage_paths else 1,
        }


# readable titles of the recordings of the task (others are built from the name)
_TITLES = {
    "doubleT_obstacle": "Человек и предмет на рельсе · двухпутный тоннель, стоянка",
    "doubleT_platform": "Подход к платформе · двухпутный тоннель, торможение",
    "roundT_doubleT": "Кривая, переход в двухпутный тоннель · 55–69 км/ч",
    "roundT_pressureGate_roundT": "Гермозатвор на кривой · круглый тоннель",
    "roundT_squareT_pressureGate_squareT": "Смена сечения и гермозатвор · кривая",
    "squareT_platform_squareT_switch": "Платформа, стоянка, стрелочный перевод",
    "cloud_with_fake_obj": "Испытательный проезд · 10 препятствий разного размера через ~100 м",
    "falcon_test_second_dark": "Falcon Prime · новая реальная запись · профиль Falcon ROS",
    "new_data": "Новая реальная запись · Pandar · 221 часть",
    "synthetic_platform_cloud_only": "Новая синтетика · платформа",
    "synthetic_switch_cloud_only": "Новая синтетика · стрелка",
    "synthetic_tunnels_cloud_only": "Новая синтетика · тоннели",
}
_WORDS = {
    "roundT": "круглый тоннель", "squareT": "прямоугольный тоннель", "doubleT": "двухпутный тоннель",
    "platform": "платформа", "pressureGate": "гермозатвор", "switch": "стрелка", "obstacle": "препятствие",
}


def bag_title(name: str) -> tuple[str, str]:
    """Russian title and group (with obstacles / regular ride / synthetic) of a recording."""
    low = name.lower()
    if "fake" in low or "synth" in low:
        group = "Синтетические препятствия"
    elif "obstacle" in low:
        group = "С препятствиями"
    elif name in {"doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
                  "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch"}:
        group = "Штатные проезды (без препятствий)"
    else:
        group = "Записи без разметки"
    if name in _TITLES:
        return _TITLES[name], group
    words = [_WORDS.get(w, w) for w in name.split("_") if w]
    title = ", ".join(dict.fromkeys(words)) if words else name
    return title[:1].upper() + title[1:], group


def _connect(path: Path) -> sqlite3.Connection:
    uri = "file:" + path.resolve().as_posix() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return con


def _bag_storage_paths(path: Path) -> tuple[Path, ...]:
    """Resolve all SQLite chunks of a directory in metadata order.

    An explicitly selected .db3 remains a single-file selection; directory
    discovery always uses the whole recording, not the first lexical chunk.
    """
    if path.suffix == ".db3":
        return (path,) if path.is_file() else ()
    metadata = path / "metadata.yaml"
    if metadata.exists():
        import yaml
        content = yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}
        names = content.get("rosbag2_bagfile_information", {}).get("relative_file_paths", [])
        if names:
            result = []
            for name in names:
                candidate = (path / name).resolve()
                if not candidate.is_relative_to(path.resolve()):
                    raise ValueError("bag metadata references a path outside its directory")
                if candidate.suffix != ".db3":
                    raise ValueError("offline reader requires SQLite .db3 storage; use ROS for MCAP")
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                if candidate in result:
                    raise ValueError("duplicate storage file in bag metadata")
                result.append(candidate)
            return tuple(result)
    return tuple(sorted(path.glob("*.db3")))


def open_bag(path: str | Path, root: Path | None = None, topic: str | None = None) -> BagInfo:
    """Open a bag given a directory (with *.db3) or a .db3 file."""
    path = Path(path).resolve()
    paths = _bag_storage_paths(path)
    if not paths:
        raise FileNotFoundError(f"no .db3 file in {path}")
    selected = None
    count, starts, ends = 0, [], []
    for db in paths:
        con = _connect(db)
        try:
            rows = con.execute("SELECT id, name, type FROM topics ORDER BY id").fetchall()
            cands = [r for r in rows if r[2] == POINTCLOUD2 and (topic is None or r[1] == topic)]
            if selected is None and cands:
                selected = cands[0]
            if selected is None:
                continue
            row = next((r for r in rows if r[1] == selected[1] and r[2] == POINTCLOUD2), None)
            if row is None:
                continue
            cnt, t0, t1 = con.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id=?", (row[0],)
            ).fetchone()
            count += cnt
            if t0 is not None:
                starts.append(t0); ends.append(t1)
        finally:
            con.close()
    if selected is None:
        raise ValueError(f"no PointCloud2 topic in {path}")
    tid, tname, _ = selected
    bag_dir = path.parent if path.suffix == ".db3" else path
    ident = bag_dir.relative_to(root).as_posix() if root and bag_dir.is_relative_to(root) else bag_dir.name
    return BagInfo(ident, bag_dir.name, paths[0], tname, int(tid), int(count),
                   int(min(starts)) if starts else 0, int(max(ends)) if ends else 0, paths)


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
            info = open_bag(db.parent, root)
        except (OSError, ValueError, sqlite3.Error):
            continue
        if info.frame_count:
            out.append(info)
    return out


class BagReader:
    """Random and sequential access to PointCloud2 messages of one bag."""

    def __init__(self, info: BagInfo):
        self.info = info
        self._cons = []
        self._ids = []
        try:
            for source_index, path in enumerate(info.storage_paths or (info.path,)):
                con = _connect(path)
                self._cons.append(con)
                rows = con.execute(
                    "SELECT m.id, m.timestamp FROM messages m JOIN topics t ON t.id=m.topic_id "
                    "WHERE t.name=? AND t.type=? ORDER BY m.timestamp,m.id", (info.topic, POINTCLOUD2))
                self._ids.extend((int(ts), source_index, int(message_id)) for message_id, ts in rows)
            self._ids.sort()
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._ids)

    def read(self, index: int) -> tuple[int, RawCloud]:
        if not 0 <= index < len(self._ids):
            raise IndexError(index)
        ts, source_index, message_id = self._ids[index]
        (data,) = self._cons[source_index].execute(
            "SELECT data FROM messages WHERE id=?", (message_id,)).fetchone()
        return int(ts), decode_pointcloud2(data)

    def iterate(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, int, RawCloud]]:
        stop = len(self._ids) if stop is None else min(stop, len(self._ids))
        for i in range(start, stop, step):
            ts, cloud = self.read(i)
            yield i, ts, cloud

    def close(self) -> None:
        for con in self._cons:
            con.close()
