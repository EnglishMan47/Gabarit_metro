"""Independent ROS MarkerArray annotations, decoded without ROS or a detector."""
from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

import numpy as np


class MarkerCdrError(ValueError):
    pass


class _Reader:
    def __init__(self, payload):
        self.data, self.offset = memoryview(payload), 4
        if len(payload) < 8 or payload[:2] not in (b'\x00\x01', b'\x00\x00'):
            raise MarkerCdrError('unsupported MarkerArray CDR encapsulation')
        self.endian = '<' if payload[:2] == b'\x00\x01' else '>'

    def values(self, fmt, alignment):
        self.offset += (-(self.offset-4)) % alignment
        size = struct.calcsize(self.endian+fmt)
        if self.offset+size > len(self.data):
            raise MarkerCdrError('truncated MarkerArray')
        values = struct.unpack_from(self.endian+fmt, self.data, self.offset)
        self.offset += size
        return values

    def count(self):
        value = self.values('I', 4)[0]
        if value > 1_000_000:
            raise MarkerCdrError('unreasonable MarkerArray sequence length')
        return value

    def string(self):
        count = self.count()
        if count < 1 or self.offset+count > len(self.data):
            raise MarkerCdrError('invalid MarkerArray string')
        value = bytes(self.data[self.offset:self.offset+count])
        self.offset += count
        if value[-1] != 0:
            raise MarkerCdrError('unterminated MarkerArray string')
        return value[:-1].decode('utf-8')

    def octets(self):
        count = self.count()
        if self.offset+count > len(self.data):
            raise MarkerCdrError('truncated embedded marker data')
        self.offset += count
        return count


def decode_marker_array(payload: bytes) -> list[dict]:
    reader = _Reader(payload)
    markers = []
    for _ in range(reader.count()):
        sec, nanosec = reader.values('iI', 4)
        frame_id, namespace = reader.string(), reader.string()
        ident, kind, action = reader.values('iii', 4)
        pose = reader.values('7d', 8)
        scale, color = reader.values('3d', 8), reader.values('4f', 4)
        life_sec, life_ns = reader.values('iI', 4)
        locked = bool(reader.values('B', 1)[0])
        points = [list(reader.values('3d', 8)) for _ in range(reader.count())]
        colors = [list(reader.values('4f', 4)) for _ in range(reader.count())]
        texture_resource = reader.string()
        reader.values('iI', 4)  # CompressedImage.header.stamp
        texture_frame = reader.string()
        texture_format = reader.string()
        texture_bytes = reader.octets()
        uv_count = reader.count()
        for _ in range(uv_count):
            reader.values('2f', 4)
        text, mesh = reader.string(), reader.string()
        mesh_filename, mesh_bytes = reader.string(), reader.octets()
        embedded = bool(reader.values('B', 1)[0])
        if not all(math.isfinite(v) for v in (*pose, *scale, *color)):
            raise MarkerCdrError('nonfinite marker pose/scale/color')
        markers.append(dict(id=ident, ns=namespace, type=kind, action=action,
            frame_id=frame_id, stamp=sec+nanosec*1e-9, position=list(pose[:3]), quaternion=list(pose[3:]),
            scale=list(scale), color=list(color), lifetime_s=life_sec+life_ns*1e-9,
            frame_locked=locked, points=points, colors=colors, text=text,
            mesh_resource=mesh, mesh_use_embedded_materials=embedded,
            texture_resource=texture_resource, texture_frame=texture_frame,
            texture_format=texture_format, texture_bytes=texture_bytes,
            uv_count=uv_count, mesh_filename=mesh_filename, mesh_bytes=mesh_bytes))
    return markers


def marker_records(info) -> list[dict]:
    records = []
    for part in info.storage_paths or (info.path,):
        with sqlite3.connect(Path(part).resolve().as_uri()+'?mode=ro', uri=True) as con:
            topics = con.execute('SELECT id,name FROM topics WHERE type=?',
                                 ('visualization_msgs/msg/MarkerArray',)).fetchall()
            for topic_id, topic in topics:
                for ident, stamp, payload in con.execute('SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topic_id,)):
                    records.append(dict(timestamp_ns=stamp, topic=topic, source_part=str(part),
                                        source_message_id=ident, markers=decode_marker_array(payload)))
    return sorted(records, key=lambda r:(r['timestamp_ns'], r['source_part'], r['source_message_id']))


def rotation(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-9:
        raise ValueError('marker needs a finite nonzero quaternion')
    x,y,z,w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def marker_bounds(marker) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oriented local bounds and rotation, without any estimated track path."""
    if marker['action'] != 0:
        raise ValueError('deleted markers have no active annotation geometry')
    if marker['type'] == 1:  # CUBE
        half = np.asarray(marker['scale'], float)/2
        lo, hi = -half, half
    elif marker['type'] in (4,5,6,7,8) and marker['points']:
        points = np.asarray(marker['points'], dtype=float)
        lo, hi = points.min(axis=0), points.max(axis=0)
    else:
        raise ValueError('marker is not a bounded box/point annotation')
    if not np.isfinite(np.r_[lo,hi]).all() or np.any(hi < lo):
        raise ValueError('invalid marker bounds')
    return lo, hi, rotation(marker['quaternion'])


def point_membership(raw_xyz: np.ndarray, marker: dict, margin_m: float = .02) -> np.ndarray:
    lo, hi, rot = marker_bounds(marker)
    points = (np.asarray(raw_xyz)-marker['position']) @ rot
    return np.isfinite(points).all(axis=1) & np.all((points >= lo-margin_m) & (points <= hi+margin_m), axis=1)
