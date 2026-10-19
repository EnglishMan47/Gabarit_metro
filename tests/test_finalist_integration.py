"""Regressions across the finalist sensor, playback and synthetic-data flows."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gabarit.__main__ as cli
import gabarit.pipeline as pipeline_module
import gabarit.rosbag as rosbag_module
import web.session as session_module
from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.rosbag import RawCloud
from gabarit.synth import SyntheticObstacle, inject, inject_dust
from gabarit.track import Calibration, TrackModel
from tests.sim import tunnel_scan
from web.server import sensor_profiles


def _falcon_from_pandar(raw):
    """Independent manual-axis mapping: Falcon xyz = (Pandar z, -x, -y)."""
    arr = np.zeros(len(raw.points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectivity", "f4")])
    arr["x"] = raw.points["z"]
    arr["y"] = -raw.points["x"]
    arr["z"] = -raw.points["y"]
    arr["reflectivity"] = raw.points["intensity"]
    return RawCloud(raw.stamp, "falcon_native", 1, len(arr), arr)


def _blank_like(raw, *, stamp=None, frame_id=None):
    return RawCloud(raw.stamp if stamp is None else stamp, frame_id or raw.frame_id,
                    1, 0, np.empty(0, raw.points.dtype))


def test_native_falcon_pipeline_detects_the_same_obstacle_as_pandar():
    pandar = Pipeline(Config())
    root = Path(__file__).resolve().parents[1]
    falcon = Pipeline(Config.load(root / "config/falcon_prime.yaml"))
    for index in range(6):
        raw = tunnel_scan(cols=900, boxes=[(0, 40, 0.6, 0.5, -0.1, 1.2)], stamp=index * 0.1)
        native = _falcon_from_pandar(raw)
        a = pandar.process_raw(raw)
        b = falcon.process_raw(native)
    assert a.data["obstacle"] and b.data["obstacle"]
    assert a.data["nearest_track_distance_m"] == pytest.approx(40.0, abs=1.0)
    assert b.data["nearest_track_distance_m"] == pytest.approx(a.data["nearest_track_distance_m"], abs=0.5)
    assert b.data["sensor"]["forward_axis"] == "+z" and b.data["sensor"]["up_axis"] == "+x"
    assert b.cloud.scan_pattern == "mirror" and b.cloud.columns == 0 and not b.cloud.split.any()
    assert b.data["current_observed_until_m"] > 60
    assert b.data["calibration"]["height_m"] == pytest.approx(a.data["calibration"]["height_m"], abs=0.05)
    assert abs(np.interp(30, b.track.s, b.track.c) - np.interp(30, a.track.s, a.track.c)) < 0.1


@pytest.mark.parametrize("kind,stamp,frame_id,reason", [
    ("rewind", 0.5, None, "time_rewind"),
    ("gap_boundary", 2.0, None, "time_gap"),
    ("gap", 3.0, None, "time_gap"),
    ("frame_change", 1.1, "another_lidar", "frame_id_change"),
])
def test_replay_discontinuities_do_not_carry_confirmed_obstacles(kind, stamp, frame_id, reason):
    cfg = Config().with_overrides({"tracking": {"confirm_hits": 1}})
    pipe = Pipeline(cfg)
    raw = tunnel_scan(cols=300, boxes=[(0, 40, 0.6, 0.5, -0.1, 1.2)], stamp=1.0)
    first = pipe.process_raw(raw)
    assert first.data["obstacle"] and pipe.tracker.tracks
    reset = pipe.process_raw(_blank_like(raw, stamp=stamp, frame_id=frame_id))
    assert reset.data["state_reset"] == reason
    assert not reset.data["obstacle"] and reset.data["obstacles"] == []
    assert not reset.data["envelope_held"]
    assert reset.data["observed_until_m"] == cfg.track.start
    assert not pipe.track.calib.valid


def test_repeated_same_timestamp_cannot_confirm_copies_of_a_single_detection():
    pipe = Pipeline(Config())
    raw = tunnel_scan(cols=300, boxes=[(0, 40, 0.6, 0.5, -0.1, 1.2)], stamp=1.0)
    for repetition in range(5):
        result = pipe.process_raw(raw)
        assert not result.data["obstacle"]
        assert result.data["pending"], "The test needs real accepted candidates, not an empty detector output."
        assert all(track["hits"] == 1 for track in result.data["pending"])
        assert result.data["state_reset"] == ("duplicate_stamp" if repetition else None)


def test_held_pipeline_envelope_is_not_reported_as_fresh_free_distance():
    cfg = Config()
    pipe = Pipeline(cfg)
    raw = tunnel_scan(cols=400, stamp=0.0)
    measured = pipe.process_raw(raw)
    assert measured.data["current_observed_until_m"] > 60
    for index in range(1, cfg.track.hold_missing_frames + 1):
        held = pipe.process_raw(_blank_like(raw, stamp=index * 0.1))
        data = held.data
        assert data["envelope_held"] and data["envelope_coasted_frames"] == index
        assert data["envelope_until_m"] > data["current_observed_until_m"] + 30
        assert data["observed_until_m"] == data["current_observed_until_m"] == cfg.track.start
        assert data["free_distance_m"] == cfg.track.start
        assert data["status"] == "DEGRADED"
    expired = pipe.process_raw(_blank_like(raw, stamp=(cfg.track.hold_missing_frames + 1) * 0.1))
    assert not expired.data["envelope_held"]
    assert expired.data["envelope_until_m"] == cfg.track.start


def _straight_track():
    s = np.arange(3.0, 151.0)
    n = len(s)
    zeros = np.zeros(n)
    flags = np.ones(n, bool)
    return TrackModel(s, zeros.copy(), zeros.copy(), zeros.copy(), np.full(n, 0.05), np.full(n, 0.02),
                      flags.copy(), flags.copy(), flags.copy(), zeros.copy(),
                      np.full(n, 2.4), np.full(n, -2.4), np.empty((0, 5)), 140, 60)


def _native_cloud(working_xyz):
    xyz = np.asarray(working_xyz, np.float32)
    arr = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectivity", "f4")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 2], -xyz[:, 0], xyz[:, 1]
    arr["reflectivity"] = 12
    return RawCloud(1.0, "falcon", 1, len(arr), arr)


def _xyz(raw):
    return np.column_stack([raw.points[key] for key in "xyz"])


@pytest.mark.parametrize("yaw,expected_forward", [(0, 28), (45, 30 - np.sqrt(2)), (90, 29)])
def test_unorganized_falcon_injection_uses_actual_rays_yaw_and_nearest_occlusion(yaw, expected_forward):
    # Duplicate centre directions at 100 m and 10 m: only the far return may
    # be replaced. Zero/NaN points carry no known ray and must stay untouched.
    raw = _native_cloud([[0, 100, 0], [0, 100, 0], [0, 10, 0], [30, 100, 0],
                         [0, 0, 0], [0, np.nan, 0]])
    original = raw.points.copy()
    near = SyntheticObstacle(shape="box_50cm", s=30, width=2, length=4, height=3,
                             base=0.0, yaw_deg=yaw, intensity=70)
    far = SyntheticObstacle(shape="box_50cm", s=50, width=2, length=4, height=3,
                            base=0.0, intensity=30)
    kwargs = dict(forward_axis="+z", up_axis="+x", scan_pattern="mirror")
    a, hits = inject(raw, [far, near], _straight_track(), Calibration(), rng=np.random.default_rng(4), **kwargs)
    b, reversed_hits = inject(raw, [near, far], _straight_track(), Calibration(), rng=np.random.default_rng(4), **kwargs)
    assert hits == reversed_hits == 2
    np.testing.assert_allclose(_xyz(a), _xyz(b), equal_nan=True)
    np.testing.assert_array_equal(a.points["reflectivity"][:2], [70, 70])
    np.testing.assert_allclose(a.points["z"][:2], expected_forward, atol=0.06)
    np.testing.assert_allclose(_xyz(a)[2:], _xyz(raw)[2:], equal_nan=True)
    np.testing.assert_allclose(_xyz(raw), np.column_stack([original[key] for key in "xyz"]), equal_nan=True)
    assert a.frame_id == raw.frame_id and a.width == raw.width and a.points.dtype == raw.points.dtype
    assert np.all(np.linalg.norm(_xyz(a)[:2], axis=1) < np.linalg.norm(_xyz(raw)[:2], axis=1))


def test_dust_is_seeded_fractional_and_never_extends_or_turns_a_ray():
    rng = np.random.default_rng(5)
    points = np.column_stack((rng.uniform(-2, 2, 4000), rng.uniform(20, 100, 4000), rng.uniform(-1, 2, 4000)))
    points = np.vstack((points, [[0, 2, 0], [0, -20, 0], [0, 0, 0]]))
    raw = _native_cloud(points)
    original = _xyz(raw).copy()
    kwargs = dict(fraction=0.2, near=3, far=70, forward_axis="+z", up_axis="+x")
    a, count = inject_dust(raw, rng=np.random.default_rng(17), **kwargs)
    b, again = inject_dust(raw, rng=np.random.default_rng(17), **kwargs)
    np.testing.assert_array_equal(a.points, b.points)
    assert count == again and 0.18 < count / 4000 < 0.22
    changed = np.linalg.norm(_xyz(a) - original, axis=1) > 1e-6
    assert changed.sum() == count and not changed[-3:].any()
    ranges = np.linalg.norm(_xyz(a), axis=1)
    assert np.all(ranges <= np.linalg.norm(original, axis=1) + 1e-5)
    assert np.all(ranges[changed] >= 3) and np.all(ranges[changed] <= 70)
    before = original[changed] / np.linalg.norm(original[changed], axis=1)[:, None]
    after = _xyz(a)[changed] / ranges[changed, None]
    np.testing.assert_allclose(after, before, atol=1e-6)
    assert np.all(a.points["reflectivity"][changed] == 1)
    np.testing.assert_array_equal(_xyz(raw), original)
    same, zero = inject_dust(raw, fraction=0)
    assert same is raw and zero == 0


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_dust_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        inject_dust(_native_cloud([[0, 20, 0]]), fraction=fraction)


@pytest.mark.parametrize("rate,processing", [(1, 0.04), (0.5, 0.04), (2, 0.04), (1, 0.15)])
def test_cli_realtime_anchors_to_bag_time_without_adding_processing_delay(monkeypatch, rate, processing):
    clock = SimpleNamespace(value=100.0, starts=[], sleeps=[])
    def sleep(delay):
        assert delay >= 0
        clock.sleeps.append(delay)
        clock.value += delay
    class Reader:
        info = SimpleNamespace(name="fixture", topic="/points", start_ns=30_000_000_000)
        closed = False
        def __init__(self, info):
            self.info = info
        def __len__(self):
            return 3
        def iterate(self):
            for index in range(3):
                yield index, self.info.start_ns + index * 100_000_000, None
        def close(self):
            self.closed = True
    reader = Reader(Reader.info)
    class FakePipeline:
        def __init__(self, cfg):
            pass
        def process_raw(self, raw, stamp, **kwargs):
            clock.starts.append(clock.value)
            clock.value += processing
            return SimpleNamespace(data={"nearest_distance_m": None, "status": "CLEAR", "obstacle": False,
                                         "free_distance_m": 100, "speed_kmh": 0, "timing_ms": {"total": processing * 1000}})
    monkeypatch.setattr(cli, "time", SimpleNamespace(perf_counter=lambda: clock.value, sleep=sleep))
    monkeypatch.setattr(rosbag_module, "BagReader", lambda info: reader)
    monkeypatch.setattr(rosbag_module, "open_bag", lambda path: reader.info)
    monkeypatch.setattr(pipeline_module, "Pipeline", FakePipeline)
    assert cli.main(["run", "unused", "--realtime", "--rate", str(rate)]) == 0
    interval = max(0.1 / rate, processing)
    np.testing.assert_allclose(clock.starts, 100 + np.arange(3) * interval)
    assert reader.closed


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_rate_before_opening_a_bag(monkeypatch, rate):
    def unexpected_open(path):
        raise AssertionError("Invalid playback rate must fail before opening data.")
    monkeypatch.setattr(rosbag_module, "open_bag", unexpected_open)
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "unused", "--realtime", "--rate", rate])
    assert error.value.code == 2


def test_web_sensor_profiles_preserve_launch_config_and_explicit_falcon_axes():
    cfg = Config().with_overrides({"sensor": {"forward_axis": "-x", "up_axis": "+y",
                                               "scan_pattern": "mirror", "default_height": 1.7}})
    profiles = sensor_profiles(cfg)
    assert profiles["current"]["sensor"] == asdict(cfg.sensor)
    assert profiles["pandar"]["sensor"]["forward_axis"] == "-y"
    assert profiles["pandar"]["sensor"]["up_axis"] == "+z"
    for profile, forward, up in (("falcon_prime", "+z", "+x"), ("falcon_prime_ros", "+x", "+z")):
        sensor = profiles[profile]["sensor"]
        assert (sensor["forward_axis"], sensor["up_axis"]) == (forward, up)
        assert sensor["scan_pattern"] == "mirror" and not sensor["dual_returns"]
        assert profiles[profile]["max_range_m"] == 350.


def test_session_warmup_and_config_reprocessing_do_not_move_virtual_obstacles(monkeypatch):
    processed = []
    class Reader:
        def __init__(self, info):
            pass
        def __len__(self):
            return 20
        def read(self, index):
            return index * 100_000_000, object()
    class FakePipeline:
        def __init__(self, cfg):
            self.track = SimpleNamespace(prev=None, calib=SimpleNamespace(valid=False))
        def reset(self):
            pass
        def process_raw(self, raw, stamp, frame_index=None):
            processed.append(frame_index)
            return SimpleNamespace(data={"displacement_m": 2.0})
    monkeypatch.setattr(session_module, "BagReader", Reader)
    monkeypatch.setattr(session_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(session_module, "pack_frame", lambda res, meta: b"frame")
    session = session_module.BagSession(SimpleNamespace(id="fixture", start_ns=0), Config())
    session.add_obstacle("person", 30, 0.2)
    session.step()
    session.step()
    assert session.odometer == 4 and session.obstacles[0].ob.s == 26
    with session.lock:
        session._warm(8)
        session._warm(8)
    assert processed.count(0) >= 3, "Warm-up must actually reprocess historical frames in this regression."
    assert session.odometer == 4 and session.obstacles[0].ob.s == 26
    session.set_config(Config().with_overrides({"envelope": {"width": 2.4}}))
    assert session.odometer == 4 and session.obstacles[0].ob.s == 26
    assert session.obstacles[0].world == 30
    session.seek(12)
    assert session.odometer == 4 and session.obstacles[0].ob.s == 26


def test_session_close_waits_for_the_inflight_step_to_finish(monkeypatch):
    # Events synchronize code positions. No assertion depends on scheduler
    # speed or a brief sleep. Timeouts only bound an otherwise deadlocked test.
    read_entered, release_read, close_reached = threading.Event(), threading.Event(), threading.Event()
    state = SimpleNamespace(close_thread=None)
    trace = []
    class Reader:
        closed = False
        def __init__(self, info):
            pass
        def __len__(self):
            return 1
        def read(self, index):
            assert not self.closed
            trace.append("read_start")
            read_entered.set()
            assert release_read.wait(10), "Test did not release the blocked read."
            assert not self.closed, "Bag SQLite connection was closed while its worker was reading."
            trace.append("read_end")
            return 0, object()
        def close(self):
            self.closed = True
            trace.append("close")
            close_reached.set()
    class FakePipeline:
        def __init__(self, cfg):
            pass
        def process_raw(self, raw, stamp, **kwargs):
            trace.append("process")
            return SimpleNamespace(data={"displacement_m": 0.0})
    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
        def __enter__(self):
            if threading.get_ident() == state.close_thread:
                close_reached.set()
            self.lock.acquire()
        def __exit__(self, *args):
            self.lock.release()
    def pack(result, meta):
        trace.append("pack")
        return b"frame"
    monkeypatch.setattr(session_module, "BagReader", Reader)
    monkeypatch.setattr(session_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(session_module, "pack_frame", pack)
    session = session_module.BagSession(SimpleNamespace(id="fixture", start_ns=0), Config())
    session.lock = ObservedLock()
    def close():
        state.close_thread = threading.get_ident()
        session.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        step_future = pool.submit(session.step)
        try:
            assert read_entered.wait(10)
            close_future = pool.submit(close)
            assert close_reached.wait(10)
            assert not session.reader.closed
            assert not step_future.done() and not close_future.done()
        finally:
            release_read.set()
        result = step_future.result(timeout=10)
        close_future.result(timeout=10)
    assert result[0] == b"frame" and session.reader.closed
    assert trace == ["read_start", "read_end", "process", "pack", "close"]
