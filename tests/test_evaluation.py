"""Evaluation metrics must retain unknown labels and input timing boundaries."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from gabarit.config import Config
from gabarit.egomotion import propagate_track
from gabarit.evaluation import (deadline_metrics, envelope_metrics, events_from, load_annotations,
                                path_stability, score_frame_labels, statistics, track_events)
from gabarit.rosbag import open_bag
from gabarit.track import TrackModel
from scripts.evaluate import main, run_bag
from tests.sim import encode_pointcloud2, tunnel_scan


def _ob(track_id, distance):
    return {"track_id": track_id, "distance_m": distance, "lateral_m": 0.1,
            "box": {"s": [distance - 0.1, distance + 0.5], "d": [-0.2, 0.4], "h": [0.1, 1.5]}}


def _frame(index, *, pending=None, obstacles=None, envelope=80.0, observed=None, held=False):
    return {"frame": index, "stamp": index * 0.1, "obstacle": bool(obstacles),
            "obstacles": obstacles or [], "pending": pending or [], "status": "CLEAR",
            "envelope_until_m": envelope, "observed_until_m": envelope,
            "current_observed_until_m": envelope if observed is None else observed,
            "envelope_held": held}


def test_first_candidate_and_confirmation_ranges_are_distinct_with_gaps():
    frames = [_frame(0, pending=[_ob(1, 80), _ob(2, 60)]), _frame(2),
              _frame(4, obstacles=[_ob(1, 75)]), _frame(6, obstacles=[_ob(1, 70)])]
    candidates = track_events(frames)
    assert len(candidates) == 2
    assert candidates[0]["first_candidate_distance_m"] == 80
    assert candidates[0]["first_confirmed_distance_m"] == 75
    assert candidates[0]["confirmation_delay_frames"] == 4
    assert candidates[0]["confirmation_delay_processed_frames"] == 2
    assert candidates[0]["confirmation_delay_s"] == pytest.approx(0.4)
    assert candidates[1]["first_confirmed_frame"] is None
    events = events_from(frames)
    assert len(events) == 1
    assert events[0]["first_frame"] == 4  # Original field still means first confirmed frame.
    assert events[0]["frames"] == 2
    assert events[0]["first_distance_m"] == 75 and events[0]["min_distance_m"] == 70
    assert events[0]["first_candidate_distance_m"] == 80


def test_unlabelled_alarms_do_not_become_false_positives():
    result = score_frame_labels([_frame(0, obstacles=[_ob(1, 80)])], None)
    assert not result["available"]
    assert "fp" not in result and "precision" not in result


def test_tracker_ids_reused_after_reset_are_separate_events():
    frames = [_frame(0, obstacles=[_ob(1, 80)]), _frame(1, obstacles=[_ob(1, 20)])]
    frames[1]["state_reset"] = "timestamp_gap"
    events = events_from(frames)
    assert len(events) == 2
    assert [event["sequence"] for event in events] == [0, 1]
    assert [event["first_candidate_distance_m"] for event in events] == [80, 20]


def test_supervised_metrics_only_use_processed_known_labels():
    frames = [_frame(0, obstacles=[_ob(1, 80)]), _frame(1),
              _frame(2, obstacles=[_ob(1, 79)]), _frame(3),
              _frame(4, obstacles=[_ob(2, 10)]), _frame(5)]
    labels = {0: True, 1: True, 2: False, 3: False, 4: None, 99: True}
    result = score_frame_labels(frames, labels, fn_cost=5)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["labelled_frames_processed"] == 4
    assert result["excluded_unknown_frames"] == 2
    assert result["annotation_frames_not_processed"] == 1
    assert result["precision"] == result["recall"] == result["balanced_accuracy"] == 0.5
    assert result["weighted_error_cost"] == 6
    assert result["weighted_error_cost_per_labelled_frame"] == 1.5


def test_undefined_supervised_ratios_are_null_and_not_perfect_scores():
    negative_only = score_frame_labels([_frame(0)], {0: False})
    assert negative_only["precision"] is None and negative_only["recall"] is None
    assert negative_only["balanced_accuracy"] is None
    empty = score_frame_labels([], {100: True})
    assert empty["accuracy"] is None and empty["labelled_frames_processed"] == 0
    assert empty["annotation_frames_not_processed"] == 1
    json.dumps(empty, allow_nan=False)


def test_duplicate_processed_frames_and_invalid_cost_are_rejected():
    with pytest.raises(ValueError, match="duplicate processed"):
        score_frame_labels([_frame(0), _frame(0)], {0: True})
    with pytest.raises(ValueError, match="fn_cost"):
        score_frame_labels([], {}, fn_cost=float("nan"))


@pytest.mark.parametrize("bad", [
    {"schema_version": True, "bags": {}},
    {"schema_version": 1, "bags": {"test": {"frames": [{"frame": True, "obstacle": True}]}}},
    {"schema_version": 1, "bags": {"test": {"frames": [{"frame": 0, "obstacle": "false"}]}}},
    {"schema_version": 1, "bags": {"test": {"frames": [{"frame": -1, "obstacle": True}]}}},
    {"schema_version": 1, "bags": {"test": {"frames": [{"frame": 0, "obstacle": True}, {"frame": 0, "obstacle": False}]}}},
])
def test_annotation_schema_rejects_ambiguous_labels(tmp_path, bad):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_annotations(path)


def test_statistics_and_deadline_metrics_handle_empty_and_tail_latency():
    stats = statistics([20, 40, 100, 200, None, float("nan")])
    assert stats["count"] == 4 and stats["p50"] == 70
    assert stats["p99"] > stats["p95"] > 100
    realtime = deadline_metrics([20, 40, 100, 200])
    assert realtime["within_budget_share"] == 0.75
    assert realtime["deadline_miss_frames"] == 1
    assert realtime["throughput_fps"] == pytest.approx(11.111, abs=1e-3)
    assert statistics([])["p95"] is None
    assert deadline_metrics([])["within_budget_share"] is None
    json.dumps(statistics([]), allow_nan=False)


def _track():
    s = np.arange(3.0, 101.0)
    c = 0.002 * (s - 3) ** 2
    n = len(s)
    return TrackModel(s=s, c=c, z=np.zeros(n), theta=np.gradient(c, s), sig_c=np.full(n, 0.1),
                      sig_z=np.full(n, 0.1), rail=np.ones(n, bool), lat_obs=np.ones(n, bool),
                      ver_obs=np.ones(n, bool), floor=np.zeros(n), wall_left=c + 2, wall_right=c - 2,
                      rails=np.empty((0, 5)), observed_until=90, rail_until=50)


def test_stability_compensates_motion_before_measuring_jitter():
    previous = _track()
    current = propagate_track(previous, 5.0)
    compensated = path_stability(previous, current, 5.0)
    uncompensated = path_stability(previous, current, 0.0)
    assert compensated["lateral_rmse_m"] == 0
    assert uncompensated["lateral_rmse_m"] > 0.1
    current.c = current.c + 0.2
    assert path_stability(previous, current, 5.0)["lateral_rmse_m"] == pytest.approx(0.2)
    current.coast_frames = 1
    assert path_stability(previous, current, 5.0)["held_pair"]


def test_missing_envelope_has_no_fabricated_stability_score():
    previous = _track()
    current = _track()
    current.observed_until = 3
    assert path_stability(previous, current, 0) is None
    assert path_stability(None, current, 0) is None


def test_current_observation_held_horizon_and_dropout_rates_are_separate():
    frames = [_frame(0, envelope=80), _frame(1, envelope=75, observed=3, held=True),
              _frame(2, envelope=3), _frame(3, envelope=50)]
    frames[1]["status"] = "DEGRADED"
    frames[1]["path_stability"] = {"held_pair": True, "lateral_rmse_m": 0,
                                   "vertical_rmse_m": 0, "lateral_abs_p95_m": 0}
    result = envelope_metrics(frames)
    assert result["held_share"] == result["degraded_share"] == result["no_envelope_share"] == 0.25
    assert result["current_observed_until_m"]["p50"] == 26.5
    assert result["envelope_until_m"]["p50"] == 62.5
    assert result["availability_transitions"] == 2 and result["envelope_loss_transitions"] == 1
    assert result["motion_compensated_stability"]["fresh_pairs"]["frame_pairs"] == 0
    assert result["motion_compensated_stability"]["held_pairs"]["frame_pairs"] == 1


def _bag(tmp_path, frames=0):
    folder = tmp_path / "sample_bag"
    folder.mkdir()
    path = folder / "sample_0.db3"
    con = sqlite3.connect(path)
    try:
        con.executescript("CREATE TABLE topics (id INTEGER PRIMARY KEY,name TEXT,type TEXT); "
                          "CREATE TABLE messages (id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);")
        con.execute("INSERT INTO topics VALUES (1, '/points', 'sensor_msgs/msg/PointCloud2')")
        for index in range(frames):
            raw = tunnel_scan(cols=12, stamp=index * 0.1)
            con.execute("INSERT INTO messages VALUES (?, 1, ?, ?)",
                        (index + 1, index * 100_000_000, encode_pointcloud2(raw)))
        con.commit()
    finally:
        con.close()
    return open_bag(folder, tmp_path)


def test_empty_bag_and_explicit_stop_zero_create_valid_reports(tmp_path):
    info = _bag(tmp_path)
    summary = run_bag(info, Config(), 1, None, tmp_path / "out", False)
    assert summary["frames"] == summary["alarm_frames"] == 0
    assert summary["events"] == [] and summary["timing_ms"] == {}
    assert summary["observed_until_median_m"] is None and summary["total_p95_ms"] is None
    assert summary["envelope"]["held_share"] is None
    assert summary["filter"]["removed_points"] is None
    assert (tmp_path / "out/sample_bag.jsonl").read_text() == ""
    json.dumps(summary, allow_nan=False)


def test_actual_bag_records_read_decode_and_processing_separately(tmp_path):
    info = _bag(tmp_path, frames=2)
    summary = run_bag(info, Config(), 1, 1, tmp_path / "out", False)
    assert summary["frames"] == 1
    timing = summary["timing_ms"]
    assert {"read_decode", "process_call", "end_to_end", "total"} <= timing.keys()
    assert timing["end_to_end"]["mean"] == pytest.approx(
        timing["read_decode"]["mean"] + timing["process_call"]["mean"], abs=0.001)
    assert summary["realtime"]["end_to_end"]["measured_frames"] == 1
    assert summary["filter"]["measured_frames"] == 1
    frame = json.loads((tmp_path / "out/sample_bag.jsonl").read_text())
    assert frame["frame"] == 0 and "points" in frame and "path_stability" in frame
    assert "track" not in frame
    assert frame["path_guard"]["active"] is False
    assert summary["path_guard"] == {"measured_frames": 1, "active_frames": 0}


def test_optional_path_export_preserves_frame_telemetry(tmp_path):
    info = _bag(tmp_path, frames=1)
    summary = run_bag(info, Config(), 1, None, tmp_path / "out", False, save_path=True)
    frame = json.loads((tmp_path / "out/sample_bag.jsonl").read_text())
    assert frame["track"]["s"]
    assert len(frame["track"]["s"]) == len(frame["track"]["c"])
    assert "path_guard" in frame and summary["path_guard"]["measured_frames"] == 1


def test_cli_reports_unmatched_bag_ids_and_excludes_unprocessed_labels(tmp_path):
    info = _bag(tmp_path, frames=1)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"schema_version": 1, "bags": {
        info.id: {"frames": [{"frame": 0, "obstacle": True}]},
        "unmatched_id": {"frames": [{"frame": 1, "obstacle": False}]}}}), encoding="utf-8")
    out = tmp_path / "report"
    assert main([str(tmp_path), "--stop", "0", "--annotations", str(annotation_path), "--out", str(out)]) == 0
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["annotations"]["unmatched_bag_ids"] == ["unmatched_id"]
    metrics = report["bags"][0]["supervised_metrics"]
    assert metrics["annotation_frames_not_processed"] == 1
    assert metrics["labelled_frames_processed"] == 0 and metrics["fn"] == 0


@pytest.mark.parametrize("flag,value", [("--step", "0"), ("--step", "-1"), ("--stop", "-1"),
                                        ("--progress", "-1"), ("--start", "-1"), ("--fn-cost", "nan")])
def test_cli_rejects_invalid_sampling_before_opening_data(flag, value, tmp_path):
    with pytest.raises(SystemExit) as result:
        main([str(tmp_path), flag, value])
    assert result.value.code == 2


@pytest.mark.parametrize("step,stop", [(0, None), (-1, None), (1, -1)])
def test_run_bag_rejects_invalid_ranges(step, stop):
    with pytest.raises(ValueError):
        run_bag(None, Config(), step, stop, None, False)


def test_start_preserves_source_frame_numbers_and_excludes_earlier_labels(tmp_path):
    info = _bag(tmp_path, frames=4)
    summary = run_bag(info, Config(), 1, 4, tmp_path / "out", False, start=2)
    frames = [json.loads(line) for line in (tmp_path / "out/sample_bag.jsonl").read_text().splitlines()]
    assert [frame["frame"] for frame in frames] == [2, 3]
    assert summary["sampling"] == {"start_inclusive": 2, "step": 1, "stop_exclusive": 4, "source_frames": 4}
    assert frames[0]["path_stability"] is None
