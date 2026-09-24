"""Unit tests of the detector on synthetic tunnels (no dataset needed)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gabarit  # noqa: E402,F401
from gabarit.cloud import from_raw  # noqa: E402
from gabarit.config import Config  # noqa: E402
from gabarit.detect import connected_components  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import decode_pointcloud2  # noqa: E402
from gabarit.synth import SyntheticObstacle, inject  # noqa: E402
from gabarit.track import TrackEstimator  # noqa: E402
from tests.sim import encode_pointcloud2, tunnel_scan  # noqa: E402


def run(frames, cfg=None):
    pipe = Pipeline(cfg or Config())
    res = None
    for k, raw in enumerate(frames):
        res = pipe.process_raw(raw, 0.1 * k, frame_index=k)
    return pipe, res


def test_cdr_roundtrip():
    raw = tunnel_scan(cols=60)
    dec = decode_pointcloud2(encode_pointcloud2(raw))
    assert dec.width == raw.width and dec.frame_id == "sim_lidar"
    assert np.allclose(dec.points["x"], raw.points["x"])
    assert np.array_equal(dec.points["ring"], raw.points["ring"])


def test_config_overrides_and_unknown_keys():
    cfg = Config().with_overrides({"envelope": {"width": 2.1}, "tracking": {"confirm_hits": 4}})
    assert cfg.envelope.width == 2.1 and cfg.tracking.confirm_hits == 4
    with pytest.raises(KeyError):
        Config().with_overrides({"envelope": {"widht": 2.0}})


def test_cloud_axes_and_organisation():
    raw = tunnel_scan(cols=120)
    cloud = from_raw(raw)
    assert cloud.rings == 128 and cloud.columns == 120
    assert np.all(cloud.xyz[:, 1] > 0)          # forward is positive
    assert cloud.ring.max() == 127


def test_connected_components():
    keys = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 6, 6], [10, 0, 0]])
    lab = connected_components(keys)
    assert lab[0] == lab[1] and lab[2] == lab[3] and len(set(lab.tolist())) == 3


@pytest.mark.parametrize("height", [1.07, 1.48])
def test_self_calibration_finds_rails(height):
    est = TrackEstimator(Config())
    raw = tunnel_scan(height=height)
    _, tr = est.estimate(from_raw(raw).xyz)
    assert abs(est.calib.height - height) < 0.08
    assert tr.rail.sum() > 10
    assert abs(np.interp(20, tr.s, tr.c)) < 0.1


def test_empty_tunnel_is_clear():
    frames = [tunnel_scan(stamp=0.1 * k) for k in range(6)]
    _, res = run(frames)
    assert res.data["status"] == "CLEAR"
    assert not res.data["obstacle"]
    assert res.data["observed_until_m"] > 60


def test_box_in_envelope_is_detected_with_distance():
    box = [(0.0, 40.0, 0.6, 0.5, -0.1, 1.2)]
    frames = [tunnel_scan(boxes=box, stamp=0.1 * k) for k in range(6)]
    _, res = run(frames)
    d = res.data
    assert d["obstacle"], d["candidates"]
    assert abs(d["nearest_track_distance_m"] - 40.0) < 1.0


def test_object_outside_envelope_is_ignored():
    box = [(1.9, 30.0, 0.4, 0.4, 0.0, 1.5)]   # between envelope (1.1) and wall (2.4)
    frames = [tunnel_scan(boxes=box, stamp=0.1 * k) for k in range(6)]
    _, res = run(frames)
    assert not res.data["obstacle"]


def test_ray_cast_injection_is_detected_and_occludes():
    frames = [tunnel_scan(stamp=0.1 * k) for k in range(8)]
    pipe = Pipeline(Config())
    ob = SyntheticObstacle(shape="person", s=60.0, d=0.0)
    res = None
    hits = 0
    for k, raw in enumerate(frames):
        if pipe.track.prev is not None:
            raw, hits = inject(raw, [ob], pipe.track.prev, pipe.track.calib)
        res = pipe.process_raw(raw, 0.1 * k, frame_index=k)
    assert hits > 20
    assert res.data["obstacle"]
    assert abs(res.data["nearest_track_distance_m"] - 60.0) < 1.5


def test_single_frame_noise_is_not_confirmed():
    frames = [tunnel_scan(stamp=0.1 * k) for k in range(5)]
    frames[3] = tunnel_scan(boxes=[(0.0, 30.0, 0.5, 0.5, 0.0, 1.0)], stamp=0.3)
    _, res = run(frames)
    assert not res.data["obstacle"]


def test_yawed_sensor_path_follows_track():
    est = TrackEstimator(Config())
    raw = tunnel_scan(yaw_deg=1.5)
    _, tr = est.estimate(from_raw(raw).xyz)
    # the tunnel axis is rotated by -1.5 deg in the sensor frame
    expected = -math.tan(math.radians(1.5)) * 30.0
    assert abs(np.interp(30, tr.s, tr.c) - expected) < 0.25
