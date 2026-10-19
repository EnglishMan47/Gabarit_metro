"""Coordinate conventions and PointCloud2 layouts across Pandar and Falcon."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from gabarit.cloud import axis_transform, from_raw, working_set
from gabarit.config import Config, SensorConfig
from gabarit.rosbag import CdrError, RawCloud, decode_pointcloud2, pointcloud2_array
from tests.sim import tunnel_scan


def _raw(xyz, ring=None, intensity="reflectivity"):
    dtype = [(key, "<f4") for key in "xyz"] + [(intensity, "<f4")]
    if ring is not None:
        dtype.append(("channel", "<u2"))
    arr = np.zeros(len(xyz), dtype=dtype)
    for k, key in enumerate("xyz"):
        arr[key] = np.asarray(xyz)[:, k]
    arr[intensity] = np.arange(len(arr), dtype=np.float32) + 7
    if ring is not None:
        arr["channel"] = ring
    return RawCloud(0.1, "test", 1, len(arr), arr)


def test_native_falcon_and_rep103_map_to_same_physical_points():
    # Physical positions: 2 m left, 20 m ahead, 1 m above; 3 right, 8 ahead, 1 below.
    native = _raw([[1, -2, 20], [-1, 3, 8]])
    rep103 = _raw([[20, 2, 1], [8, -3, -1]], intensity="reflectance")
    kwargs = dict(scan_pattern="mirror", dual_returns=False, elevation_resolution_deg=0.08)
    a = from_raw(native, "+z", up_axis="+x", **kwargs)
    b = from_raw(rep103, "+x", up_axis="+z", **kwargs)
    np.testing.assert_allclose(a.xyz, [[2, 20, 1], [-3, 8, -1]])
    np.testing.assert_allclose(a.xyz, b.xyz)
    np.testing.assert_allclose(a.intensity, [7, 8])
    np.testing.assert_allclose(a.intensity, b.intensity)


@pytest.mark.parametrize("forward,up,expected", [
    ("-y", "+z", [2, -3, 4]),
    ("+y", "+z", [-2, 3, 4]),
    ("+x", "+z", [3, 2, 4]),
    ("-x", "+z", [-3, -2, 4]),
    ("+z", "+x", [-3, 4, 2]),
    ("-z", "-x", [-3, -4, -2]),
    ("+x", "-z", [-3, 2, -4]),
])
def test_signed_axis_transform_and_inverse(forward, up, expected):
    matrix = axis_transform(forward, up)
    xyz = np.array([[2, 3, 4]], np.float32)
    mapped = xyz @ matrix.T
    np.testing.assert_allclose(mapped, [expected])
    np.testing.assert_allclose(mapped @ matrix, xyz)


def test_pandar_default_keeps_coordinates_and_full_organization():
    raw = tunnel_scan(cols=4)
    cloud = from_raw(raw)
    valid = (raw.points["x"] ** 2 + raw.points["y"] ** 2 + raw.points["z"] ** 2) > 1
    np.testing.assert_allclose(cloud.xyz[:, 0], raw.points["x"][valid])
    np.testing.assert_allclose(cloud.xyz[:, 1], -raw.points["y"][valid])
    np.testing.assert_allclose(cloud.xyz[:, 2], raw.points["z"][valid])
    assert cloud.columns == 4 and cloud.rings == 128
    raw.points["ring"][-1] = 0
    assert from_raw(raw).columns == 0  # One valid first column cannot prove organization.


def test_mirror_scanner_never_pairs_returns_or_discards_near_columns():
    raw = _raw([[0, 0, 4], [0.1, 0, 4], [0, 0, 9], [0.1, 0, 9]], ring=[0, 1, 0, 1])
    cloud = from_raw(raw, "+z", up_axis="+x", scan_pattern="mirror", dual_returns=True,
                     elevation_resolution_deg=0.08)
    assert cloud.columns == 0 and np.all(cloud.column == -1)
    assert not cloud.split.any() and not cloud.dual_returns
    assert np.all(working_set(cloud))
    assert len(np.unique(cloud.ring)) >= 2
    sub = cloud.subset(np.array([True, False, True, False]))
    assert sub.scan_pattern == "mirror" and not sub.dual_returns


def test_single_return_mechanical_scan_does_not_create_transparency():
    raw = _raw([[0, -4, 0], [0, -4, 0.1], [0, -9, 0], [0, -9, 0.1]], ring=[0, 1, 0, 1])
    assert from_raw(raw).split.any()
    assert not from_raw(raw, dual_returns=False).split.any()


def test_mirror_geometry_works_without_channel_or_intensity():
    raw = _raw([[0, 0, 20], [0.2, 0, 20]])
    raw.points = raw.points[["x", "y", "z"]]
    cloud = from_raw(raw, "+z", up_axis="+x", scan_pattern="mirror")
    assert len(np.unique(cloud.ring)) == 2 and cloud.rings >= 2
    assert np.all(cloud.intensity == 0)


@pytest.mark.parametrize("overrides", [
    {"sensor": {"forward_axis": "z"}},
    {"sensor": {"forward_axis": "+z", "up_axis": "-z"}},
    {"sensor": {"scan_pattern": "guess"}},
    {"sensor": {"elevation_resolution_deg": 0}},
    {"sensor": {"min_range": -1}},
    {"filter": {"voxel_near": 0}},
    {"filter": {"voxel_growth": -0.1}},
    {"filter": {"min_neighbors": 0}},
    {"track": {"hold_missing_frames": -1}},
])
def test_invalid_sensor_filter_configuration_fails_early(overrides):
    with pytest.raises(ValueError):
        Config().with_overrides(overrides)


def test_falcon_profiles_are_loadable_and_keep_horizon():
    root = Path(__file__).resolve().parents[1]
    native = Config.load(root / "config/falcon_prime.yaml")
    ros = Config.load(root / "config/falcon_prime_ros.yaml")
    assert (native.sensor.forward_axis, native.sensor.up_axis) == ("+z", "+x")
    assert (ros.sensor.forward_axis, ros.sensor.up_axis) == ("+x", "+z")
    assert native.sensor.scan_pattern == ros.sensor.scan_pattern == "mirror"
    assert not native.sensor.dual_returns
    assert native.track.horizon == ros.track.horizon == 350
    assert native.detection.max_range == ros.detection.max_range == 350


def test_sensor_config_legacy_positional_arguments_remain_compatible():
    cfg = SensorConfig("-y", False, 1.4)
    assert cfg.forward_axis == "-y"
    assert not cfg.auto_calibration and cfg.default_height == 1.4
    assert cfg.up_axis == "+z"


@pytest.mark.parametrize('name',['density_azimuth_resolution_deg','density_elevation_resolution_deg'])
@pytest.mark.parametrize('value',[-.1,float('nan'),float('inf'),10.1])
def test_invalid_physical_density_spacing_is_rejected(name,value):
    with pytest.raises(ValueError,match=name):
        Config().with_overrides({'sensor':{name:value}})


@pytest.mark.parametrize("pattern,dual,organized", [
    ("mechanical", True, True), ("mechanical", False, True),
    ("mechanical", True, False), ("mirror", False, False)])
def test_conversion_with_working_range_is_exactly_the_public_subset(pattern, dual, organized):
    raw = tunnel_scan(cols=300)
    if not organized:
        raw = RawCloud(raw.stamp, raw.frame_id, 1, len(raw.points), raw.points[::-1].copy())
    args = dict(scan_pattern=pattern, dual_returns=dual)
    full = from_raw(raw, **args)
    expected = full.subset(working_set(full, 105.0))
    actual = from_raw(raw, **args, working_range=105.0)
    for key in ("xyz", "intensity", "ring", "column", "split"):
        assert np.array_equal(getattr(actual, key), getattr(expected, key)), key
    assert (actual.raw_count, actual.rings, actual.columns) == (full.raw_count, full.rings, full.columns)


def _padded_cloud_data(big=False):
    order = ">" if big else "<"
    fields = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("extra", 12, 4, 2)]
    data = bytearray(2 * 40)  # Two 16-byte points followed by 8 padding bytes per row.
    for i in range(4):
        offset = (i // 2) * 40 + (i % 2) * 16
        struct.pack_into(order + "fffHH", data, offset, i + 1, i + 11, i + 21, i + 31, i + 41)
    data[32:40] = data[72:80] = b"PADDING!"
    return fields, data


def _cdr(fields, data, *, big=False, row_step=40, stamp=-1):
    out = bytearray(b"\x00\x01\x00\x00")
    def u32(value):
        out.extend(b"\x00" * ((-(len(out) - 4)) % 4))
        out.extend(struct.pack("<I", value))
    def string(value):
        encoded = value.encode() + b"\x00"
        u32(len(encoded))
        out.extend(encoded)
    u32(stamp & 0xFFFFFFFF)
    u32(100_000_000)
    string("falcon")
    u32(2)
    u32(2)
    u32(len(fields))
    for name, offset, datatype, count in fields:
        string(name)
        u32(offset)
        out.append(datatype)
        u32(count)
    out.append(int(big))
    u32(16)
    u32(row_step)
    u32(len(data))
    out.extend(data)
    out.append(1)
    return bytes(out)


@pytest.mark.parametrize("big", [False, True])
def test_cdr_padding_multicount_and_point_endian_are_preserved(big):
    fields, data = _padded_cloud_data(big)
    raw = decode_pointcloud2(_cdr(fields, data, big=big))
    assert (raw.height, raw.width, raw.count) == (2, 2, 4)
    assert raw.stamp == pytest.approx(-0.9)
    np.testing.assert_allclose(raw.points["x"], [1, 2, 3, 4])
    np.testing.assert_allclose(raw.points["y"], [11, 12, 13, 14])
    np.testing.assert_array_equal(raw.points["extra"], [[31, 41], [32, 42], [33, 43], [34, 44]])


def test_truncated_or_invalid_point_layout_fails_with_cdr_error():
    fields, data = _padded_cloud_data()
    with pytest.raises(CdrError, match="row_step"):
        decode_pointcloud2(_cdr(fields, data, row_step=30))
    with pytest.raises(CdrError, match="truncated"):
        decode_pointcloud2(_cdr(fields, data[:-1]))
    with pytest.raises(CdrError, match="exceeds point_step"):
        pointcloud2_array(data, [("x", 15, 7, 1)], 2, 2, 16, 40)
    with pytest.raises(CdrError, match="duplicate"):
        pointcloud2_array(data, [("x", 0, 7, 1), ("x", 4, 7, 1)], 2, 2, 16, 40)
    for n in (0, 8, 13, 30):
        with pytest.raises(CdrError):
            decode_pointcloud2(_cdr(fields, data)[:n])


def test_empty_pointcloud_layout_is_supported():
    arr = pointcloud2_array(b"", [], 0, 0, 0, 0)
    assert len(arr) == 0


def test_array_coordinate_field_is_rejected_with_clear_error():
    arr = np.zeros(2, dtype=[("x", "f4", (3,)), ("y", "f4"), ("z", "f4")])
    with pytest.raises(ValueError, match="x must have count=1"):
        from_raw(RawCloud(0, "test", 1, 2, arr))
