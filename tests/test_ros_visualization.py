"""Exercise ROS adapter geometry with message stubs, without an installed ROS."""
from __future__ import annotations

import ast
import array
from dataclasses import replace
from enum import IntEnum
import math
import struct
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gabarit.cloud import axis_transform
from gabarit.config import Config
from gabarit.track import Calibration
from gabarit.rosbag import decode_pointcloud2
from tests.sim import encode_pointcloud2, tunnel_scan


class Message:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Marker(Message):
    ADD, DELETE, DELETEALL = 0, 2, 3
    CUBE, LINE_STRIP, LINE_LIST, TEXT_VIEW_FACING = 1, 4, 5, 9
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pose = NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=0))
        self.scale = NS(x=0, y=0, z=0)
        self.color = NS(r=0, g=0, b=0, a=0)
        self.points = []


class MarkerArray(Message):
    def __init__(self, **kwargs):
        self.markers = kwargs.get("markers", [])


def adapter_class():
    """Compile only the pure geometry methods of the real ROS adapter."""
    path = Path(__file__).parents[1] / "ros2/gabarit_ros/gabarit_ros/node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "GabaritNode")
    names = {"_markers", "_to_sensor", "_cloud", "_xyz_cloud", "_input_reliability", "_subscribe"}
    body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in names]
    selected = ast.ClassDef(name="Adapter", bases=[], keywords=[], body=body, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, selected], type_ignores=[]))
    point_field = type("PointField", (Message,), {"FLOAT32": 7})
    namespace = {"np": np, "math": math, "array": array, "Marker": Marker, "MarkerArray": MarkerArray,
                 "Point": Message, "PointCloud2": Message, "PointField": point_field,
                 "QoSProfile": Message, "ReliabilityPolicy": Reliability,
                 "HistoryPolicy": NS(KEEP_LAST=1)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Adapter"]


class Reliability(IntEnum):
    RELIABLE = 1
    BEST_EFFORT = 2


def adapter(forward="-y", up="+z", pitch=0.0, roll=0.0):
    node = adapter_class()()
    cfg = Config()
    node.cfg = replace(cfg, sensor=replace(cfg.sensor, forward_axis=forward, up_axis=up))
    node.sensor_axes = axis_transform(forward, up)
    node.pipe = NS(track=NS(calib=Calibration(height=1.1, pitch=pitch, roll=roll)))
    node._last_marker_counts = {}
    return node


def result(obstacles=None):
    track = NS(observed_until=20.0, s=np.array([0.0, 25.0]), c=np.array([0.0, 0.0]), z=np.array([0.0, 0.0]))
    return NS(track=track, data={"obstacle": bool(obstacles), "obstacles": obstacles or [], "pending": []})


@pytest.mark.parametrize("forward,up", [("-y", "+z"), ("+x", "+z"), ("+z", "+x"), ("-x", "+z")])
def test_raw_markers_roundtrip_calibration_and_sensor_axes(forward, up):
    node = adapter(forward, up, pitch=0.08, roll=-0.03)
    expected = np.array([[0.6, 30.0, 2.0]], dtype=np.float32)
    point = node._to_sensor(*expected[0], None)
    sensor = np.array([[point.x, point.y, point.z]])
    levelled = node.pipe.track.calib.level(sensor @ node.sensor_axes.T)
    np.testing.assert_allclose(levelled, expected, atol=1e-5)


def test_marker_array_keeps_envelope_and_deletes_only_expired_boxes():
    node = adapter()
    obstacle = {"box": {"s": [10.0, 11.0], "d": [-0.4, 0.4], "h": [0.1, 1.7]},
                "distance_m": 10.0, "confirmed": True}
    first = node._markers(Message(frame_id="gabarit_view"), result([obstacle]), view=True)
    assert not any(marker.action == Marker.DELETEALL for marker in first.markers)
    box = next(marker for marker in first.markers if marker.ns == "obstacles" and marker.id == 100)
    assert len(box.points) == 24  # twelve edges; correct even with Falcon axes
    assert {point.x for point in box.points} == {10.0, 11.0}
    assert {point.y for point in box.points} == {-0.4, 0.4}
    second = node._markers(Message(frame_id="gabarit_view"), result(), view=True)
    assert {(marker.ns, marker.id) for marker in second.markers if marker.action == Marker.DELETE} == {
        ("obstacles", 100), ("obstacles", 200)}
    assert {(marker.ns, marker.action) for marker in second.markers[:2]} == {
        ("envelope", Marker.ADD), ("path", Marker.ADD)}


def test_view_cloud_layout_includes_intensity_and_clears_empty_cloud():
    node = adapter()
    pts = np.array([[20.0, 0.5, 0.2], [21.0, -0.5, 0.3]])
    cloud = node._xyz_cloud(Message(frame_id="gabarit_view"), pts, [10, 20])
    assert cloud.point_step == 16 and cloud.row_step == 32
    assert isinstance(cloud.data, array.array) and cloud.data.typecode == "B"
    assert [field.name for field in cloud.fields] == ["x", "y", "z", "intensity"]
    np.testing.assert_allclose(np.frombuffer(cloud.data, dtype=np.float32).reshape(2, 4)[:, :3], pts)
    empty = node._cloud(Message(frame_id="falcon"), np.empty((0, 3)))
    assert empty.width == 0 and empty.row_step == 0 and len(empty.data) == 0


def test_falcon_rviz_envelope_extends_to_observed_range_beyond_300m():
    node = adapter('+x', '+z')
    node.cfg = Config.load(Path(__file__).parents[1] / 'config/falcon_prime_ros.yaml')
    res = result()
    res.track.observed_until = 320.
    res.track.s = np.array([0., 350.])
    markers = node._markers(Message(frame_id='gabarit_view'), res, view=True)
    envelope = next(m for m in markers.markers if m.ns == 'envelope')
    assert max(p.x for p in envelope.points) > 300.


def test_input_qos_adapts_when_recordings_offer_different_reliability():
    node = adapter()
    params = {"input_qos": "auto", "reliable": True}
    node.get_parameter = lambda key: NS(value=params[key])
    endpoints = []
    node.get_publishers_info_by_topic = lambda name: endpoints
    node.subs, node._subscription_reliability = {}, {}
    created, destroyed = [], []
    def create(typ, name, callback, qos, **kwargs):
        assert kwargs == {"raw": True}
        value = NS(qos=qos)
        created.append(value)
        return value
    node.create_subscription = create
    node.destroy_subscription = destroyed.append
    node.get_logger = lambda: NS(info=lambda message: None)
    node._subscribe("/cloud")
    assert created[-1].qos.reliability == Reliability.BEST_EFFORT
    endpoints.append(NS(qos_profile=NS(reliability=Reliability.RELIABLE)))
    node._subscribe("/cloud")
    assert created[-1].qos.reliability == Reliability.RELIABLE
    endpoints[:] = [NS(qos_profile=NS(reliability=Reliability.BEST_EFFORT))]
    node._subscribe("/cloud")
    assert created[-1].qos.reliability == Reliability.BEST_EFFORT
    node._subscribe("/cloud")
    assert len(created) == 3 and len(destroyed) == 2
    params["input_qos"] = "reliable"
    node._subscribe("/cloud")
    assert created[-1].qos.reliability == Reliability.RELIABLE


def test_raw_ros_subscription_preserves_header_nanoseconds_and_cloud_fields():
    path = Path(__file__).parents[1] / "ros2/gabarit_ros/gabarit_ros/node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "input_to_raw")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"decode_pointcloud2": decode_pointcloud2, "struct": struct, "Header": Message, "Time": Message}
    exec(compile(module, str(path), "exec"), namespace)
    original = tunnel_scan(cols=12, stamp=1789643489.1234567)
    payload = encode_pointcloud2(original)
    raw, header = namespace["input_to_raw"](payload)
    sec, nsec = struct.unpack_from("<iI", payload, 4)
    assert (header.stamp.sec, header.stamp.nanosec) == (sec, nsec)
    assert header.frame_id == original.frame_id == raw.frame_id
    assert raw.points.dtype == original.points.dtype
    assert np.array_equal(raw.points, original.points)


@pytest.mark.parametrize("count,intensity", [(0, False), (0, True), (5, False), (5, True)])
def test_serialized_viewer_cloud_roundtrip(count, intensity):
    path = Path(__file__).parents[1] / "ros2/gabarit_ros/gabarit_ros/node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "xyz_cloud_cdr")
    namespace = {"np": np, "struct": struct}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
    pts = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    values = np.arange(count, dtype=np.float32) if intensity else None
    header = NS(frame_id="gabarit_view", stamp=NS(sec=-1, nanosec=123456789))
    payload = namespace["xyz_cloud_cdr"](header, pts, values)
    raw = decode_pointcloud2(payload)
    assert raw.frame_id == "gabarit_view" and raw.stamp == pytest.approx(-0.876543211)
    assert raw.width == count and raw.height == 1
    for column, name in enumerate("xyz"):
        assert np.array_equal(raw.points[name], pts[:, column])
    if intensity:
        assert np.array_equal(raw.points["intensity"], values)


def test_rviz_preview_keeps_every_candidate_and_obstacle_point():
    path = Path(__file__).parents[1] / "ros2/gabarit_ros/gabarit_ros/node.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "view_indices")
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
    labels = np.zeros(100000, np.uint8)
    labels[123:850] = 3
    labels[80900:82000] = 4
    idx = namespace["view_indices"](labels)
    assert len(idx) == len(np.unique(idx)) == 30000
    assert set(np.flatnonzero(labels >= 3)) <= set(idx)
    assert 0 in idx and len(labels)-1 in idx
    assert namespace["view_indices"](np.zeros(0, np.uint8)).size == 0
