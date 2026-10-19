"""Container-control regressions runnable on hosts without Docker or ROS."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


def load_controller():
    path = Path(__file__).parents[1] / "docker" / "bag_control.py"
    spec = importlib.util.spec_from_file_location("gabarit_bag_control_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def control(tmp_path, monkeypatch):
    monkeypatch.setenv("GABARIT_PLAYER_STATE", str(tmp_path / "state"))
    return load_controller()


def test_bag_resolution_rejects_ambiguous_name(control, tmp_path):
    for parent in ("old", "new"):
        bag = tmp_path / parent / "recording"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}")
        (bag / "part_0.db3").touch()
    with pytest.raises(ValueError, match="ambiguous"):
        control.resolve_bag("recording", tmp_path)
    assert control.resolve_bag("new/recording", tmp_path) == tmp_path / "new" / "recording"


def test_split_bag_is_listed_once(control, tmp_path):
    bag = tmp_path / "organized"
    bag.mkdir()
    (bag / "metadata.yaml").touch()
    (bag / "part_0.db3").touch()
    (bag / "part_1.db3").touch()
    standalone = tmp_path / "standalone.mcap"
    standalone.touch()
    metadata_only = tmp_path / "metadata_backup" / "organized"
    metadata_only.mkdir(parents=True)
    (metadata_only / "metadata.yaml").touch()
    assert control.list_bags(tmp_path) == [bag, standalone]
    assert control.resolve_bag("organized", tmp_path) == bag


def test_selected_bag_cache_reuses_complete_copy_and_tracks_source_version(control, tmp_path, monkeypatch):
    monkeypatch.setenv("GABARIT_CACHE_DIR", str(tmp_path / "linux_cache"))
    bag = tmp_path / "source"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  relative_file_paths: [part_0.db3, part_1.db3]\n")
    for i in range(2):
        (bag / f"part_{i}.db3").write_bytes(bytes([i]) * 100)
    cached = control.cache_bag(bag)
    assert cached != bag and (cached / "part_1.db3").read_bytes() == bytes([1]) * 100
    assert control.cache_bag(bag) == cached
    assert (bag / "part_0.db3").read_bytes() == bytes([0]) * 100
    (bag / "part_0.db3").write_bytes(b"new source version")
    changed = control.cache_bag(bag)
    assert changed != cached and (changed / "part_0.db3").read_bytes() == b"new source version"
    assert (cached / "part_0.db3").read_bytes() == bytes([0]) * 100


def test_cache_rejects_metadata_escape_and_insufficient_space(control, tmp_path, monkeypatch):
    monkeypatch.setenv("GABARIT_CACHE_DIR", str(tmp_path / "linux_cache"))
    bag = tmp_path / "source"
    bag.mkdir()
    (tmp_path / "outside.db3").touch()
    metadata = bag / "metadata.yaml"
    metadata.write_text("rosbag2_bagfile_information:\n  relative_file_paths: [../outside.db3]\n")
    with pytest.raises(ValueError, match="outside"):
        control.cache_bag(bag)
    metadata.write_text("rosbag2_bagfile_information:\n  relative_file_paths: [inside.db3]\n")
    (bag / "inside.db3").write_bytes(b"original")
    monkeypatch.setenv("GABARIT_CACHE_MAX_GB", "0.000000001")
    with pytest.raises(RuntimeError, match="budget"):
        control.cache_bag(bag)
    monkeypatch.setenv("GABARIT_CACHE_MAX_GB", "16")
    monkeypatch.setattr(control.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="insufficient"):
        control.cache_bag(bag)
    assert (bag / "inside.db3").read_bytes() == b"original"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_rates_are_rejected(control, value):
    with pytest.raises(argparse.ArgumentTypeError):
        control.positive_rate(value)


def test_stale_pid_does_not_signal_reused_process(control, monkeypatch):
    control.write_state({"pid": 42, "start_token": "old", "bag": "previous"})
    monkeypatch.setattr(control, "process_token", lambda pid: "different-start-time")
    sent = []
    monkeypatch.setattr(control.os, "killpg", lambda *args: sent.append(args), raising=False)
    result = control.stop_player()
    assert sent == []
    assert result["running"] is False


def test_proc_token_handles_spaces_and_parentheses(control, monkeypatch):
    stat = "42 (name with (parentheses)) S " + " ".join(str(i) for i in range(4, 53))
    monkeypatch.setattr(control.Path, "read_text", lambda *args, **kwargs: stat)
    monkeypatch.setattr(control.os, "getpgid", lambda pid: pid, raising=False)
    assert control.process_token(42) == "22"
    monkeypatch.setattr(control.os, "getpgid", lambda pid: 999, raising=False)
    assert control.process_token(42) is None


def mock_start(control, monkeypatch):
    events = []
    monkeypatch.setattr(control, "stop_player", lambda: events.append("stop"))
    monkeypatch.setattr(control, "process_token", lambda pid: "token")
    monkeypatch.setattr(control, "is_running", lambda data: True)
    def popen(argv, **kwargs):
        events.append(("spawn", argv, kwargs))
        return SimpleNamespace(pid=4321)
    monkeypatch.setattr(control.subprocess, "Popen", popen)
    def call(name, *args, **kwargs):
        events.append(name)
        return SimpleNamespace(paused=True, success=True)
    monkeypatch.setattr(control, "ros_call", call)
    return events


def test_switch_is_paused_until_detector_reset(control, tmp_path, monkeypatch):
    events = mock_start(control, monkeypatch)
    bag = tmp_path / "bag with spaces;$(not-a-shell)"
    result = control.start_player(bag, 0.25, False, ["--topics", "/point_cloud"])
    assert events[0] == "stop"
    assert events[1][0] == "spawn"
    argv, kwargs = events[1][1:]
    assert argv[:4] == ["ros2", "bag", "play", str(bag)]
    assert "--start-paused" in argv
    assert "--disable-keyboard-controls" in argv
    assert kwargs["start_new_session"] is True
    assert not kwargs.get("shell", False)
    assert events[2:] == ["/rosbag2_player/is_paused", "/gabarit/reset", "/rosbag2_player/resume"]
    assert result["paused"] is False


def test_start_paused_does_not_resume(control, tmp_path, monkeypatch):
    events = mock_start(control, monkeypatch)
    result = control.start_player(tmp_path, 1.0, True, [])
    assert "/rosbag2_player/resume" not in events
    assert result["paused"] is True


def test_failed_reset_stops_new_player_without_resuming(control, tmp_path, monkeypatch):
    events = mock_start(control, monkeypatch)
    def fail_reset(name, *args, **kwargs):
        events.append(name)
        if name == "/gabarit/reset":
            raise RuntimeError("invalid detector config")
        return SimpleNamespace(paused=True)
    monkeypatch.setattr(control, "ros_call", fail_reset)
    with pytest.raises(RuntimeError, match="invalid detector config"):
        control.start_player(tmp_path, 1.0, False, [])
    assert events[-1] == "stop"
    assert events.count("stop") == 2
    assert "/rosbag2_player/resume" not in events


def test_missing_bag_does_not_stop_current_playback(control, tmp_path, monkeypatch):
    monkeypatch.setenv("GABARIT_DATA", str(tmp_path))
    monkeypatch.setattr(control, "player_lock", nullcontext)
    monkeypatch.setattr(control, "stop_player", lambda: pytest.fail("should not stop current bag"))
    with pytest.raises(ValueError, match="bag not found"):
        control.main(["switch", "does-not-exist"])


def test_stop_escalates_with_bounded_waits(control, monkeypatch):
    data = {"pid": 42, "start_token": "token"}
    control.write_state(data)
    alive = [True]
    clock = iter(range(1000))
    monkeypatch.setattr(control.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(control.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(control, "is_running", lambda state: alive[0])
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    sent = []
    def killpg(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGKILL:
            alive[0] = False
    monkeypatch.setattr(control.os, "killpg", killpg, raising=False)
    control.stop_player()
    assert sent == [(42, signal.SIGINT), (42, signal.SIGTERM), (42, signal.SIGKILL)]


def test_workspace_cleanup_continues_if_player_cleanup_fails(monkeypatch):
    folder = Path(__file__).parents[1] / "docker"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("gabarit_workspace_test", folder / "workspace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "player_lock", nullcontext)
    def fail_stop():
        raise RuntimeError("player cleanup failed")
    monkeypatch.setattr(module, "stop_player", fail_stop)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    alive = {11: True, 12: True}
    class Child:
        def __init__(self, pid):
            self.pid = pid
        def poll(self):
            return None if alive[self.pid] else 0
        def wait(self, timeout):
            assert not alive[self.pid]
    sent = []
    def killpg(pid, sig):
        if not alive[pid]:
            raise ProcessLookupError(pid)
        if sig == 0:
            return
        sent.append(pid)
        alive[pid] = False
    monkeypatch.setattr(module.os, "killpg", killpg, raising=False)
    workspace = module.Workspace()
    workspace.children = [("ROS launch", Child(11)), ("GUI", Child(12))]
    workspace.close()
    assert sent == [12, 11]
