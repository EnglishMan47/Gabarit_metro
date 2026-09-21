#!/usr/bin/env python3
"""Evaluate the geometric detector on one or all supplied lidar bags."""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
import ctypes

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.bag_store import BagStore
from webapp.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from webapp.detector import detect_obstacles
from webapp.scene_model import NormalSceneModel
from webapp.tracking import TrackManager


def percentile(values: list[float], value: float) -> float | None:
    return float(np.percentile(values, value)) if values else None


def process_memory_mb() -> float | None:
    """Return OS-reported peak resident memory without a runtime dependency."""
    if os.name != "nt":
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(usage / (1024 if sys.platform != "darwin" else 1024 * 1024))
        except (ImportError, OSError):
            return None
    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return float(counters.PeakWorkingSetSize / 1024 / 1024)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/for_hackathon"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("annotations/doubleT_obstacle.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/final-summary.json"))
    parser.add_argument("--jsonl-dir", type=Path, default=Path("artifacts/evaluation/frames"))
    parser.add_argument("--bag", action="append", help="bag id or bag directory name; repeatable")
    parser.add_argument("--step", type=int, default=1)
    return parser.parse_args()


def closest_match(objects: list[dict[str, Any]], label: dict[str, Any], threshold: float) -> tuple[bool, float | None]:
    expected = np.asarray(
        [label["center_m"]["x"], label["center_m"]["forward"], label["center_m"]["z"]],
        dtype=np.float32,
    )
    if not objects:
        return False, None
    distances = []
    for item in objects:
        center = item["center_m"]
        actual = np.asarray([center["x"], center["forward"], center["z"]], dtype=np.float32)
        distances.append(float(np.linalg.norm(actual - expected)))
    best = min(distances)
    return best <= threshold, best


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        raise SystemExit("--step must be positive")
    config = load_config(args.config)
    model_path = args.model or (PROJECT_ROOT / config.background_model_path)
    scene_model = NormalSceneModel.load(model_path) if model_path.exists() else None
    store = BagStore(args.data_root)
    bags = store.list_bags()
    if args.bag:
        requested = set(args.bag)
        bags = [item for item in bags if item["id"] in requested or item["name"] in requested]
        if not bags:
            raise SystemExit(f"no bags match: {sorted(requested)}")
    annotations = None
    if args.annotations.exists():
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    label_lookup: dict[str, dict[int, dict[str, Any]]] = {}
    if annotations:
        for name, label in annotations["labels"].items():
            label_lookup[name] = {int(item["frame"]): item for item in label["frames"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "metro-lidar-evaluation-v1",
        "config": config.public(),
        "scene_model": str(model_path) if scene_model is not None else None,
        "step": args.step,
        "bags": {},
    }
    all_detection_times: list[float] = []
    all_decode_times: list[float] = []
    all_total_times: list[float] = []
    peak_memory_mb = process_memory_mb()
    for bag in bags:
        tracker = TrackManager(config, expected_step=args.step)
        detection_times: list[float] = []
        decode_times: list[float] = []
        total_times: list[float] = []
        raw_frames = 0
        stable_frames = 0
        processed = 0
        max_raw_points = 0
        max_valid_points = 0
        first_raw = None
        first_stable = None
        positive = {
            "primary_expected_inside": 0,
            "primary_hits_inside": 0,
            "primary_match_errors_m": [],
            "object_expected": 0,
            "object_hits": 0,
            "object_match_errors_m": [],
            "secondary_outside_matches": 0,
        }
        episodes: list[dict[str, Any]] = []
        current_episode: dict[str, Any] | None = None
        jsonl_path = args.jsonl_dir / f"{bag['name']}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for frame in store.iter_frames(bag["id"], step=args.step):
                total_started = perf_counter()
                result = detect_obstacles(frame["points"], config, scene_model)
                tracked = tracker.update(frame["index"], result["objects"])
                result["objects"] = tracked["objects"]
                result["event_detected"] = tracked["event_detected"]
                result["active_tracks"] = tracked["active_tracks"]
                total_ms = float(frame["decode_ms"]) + (perf_counter() - total_started) * 1000
                detection_times.append(float(result["processing_ms"]))
                decode_times.append(float(frame["decode_ms"]))
                total_times.append(total_ms)
                raw_frames += int(result["raw_detected"])
                stable_frames += int(result["event_detected"])
                processed += 1
                max_raw_points = max(max_raw_points, int(frame["raw_point_count"]))
                max_valid_points = max(max_valid_points, int(frame["valid_point_count"]))
                if result["raw_detected"] and first_raw is None:
                    first_raw = frame["index"]
                if result["event_detected"] and first_stable is None:
                    first_stable = frame["index"]
                if result["event_detected"]:
                    if current_episode is None:
                        current_episode = {
                            "first_frame": frame["index"],
                            "confirmed_frame": frame["index"],
                            "last_frame": frame["index"],
                            "frame_count": 1,
                            "nearest_distance_m": tracked["event_nearest_distance_m"],
                        }
                    else:
                        current_episode["last_frame"] = frame["index"]
                        current_episode["frame_count"] += 1
                        distance = tracked["event_nearest_distance_m"]
                        if distance is not None:
                            current_episode["nearest_distance_m"] = min(current_episode["nearest_distance_m"], distance)
                elif current_episode is not None:
                    episodes.append(current_episode)
                    current_episode = None
                if bag["name"] == "doubleT_obstacle" and label_lookup:
                    index = frame["index"]
                    primary = label_lookup.get("primary_person", {}).get(index)
                    if primary and primary.get("inside_gauge"):
                        positive["primary_expected_inside"] += 1
                        hit, error = closest_match(result["objects"], primary, 2.0)
                        positive["primary_hits_inside"] += int(hit)
                        if hit and error is not None:
                            positive["primary_match_errors_m"].append(error)
                    fixed = label_lookup.get("object_on_rails", {}).get(index)
                    if fixed:
                        positive["object_expected"] += 1
                        hit, error = closest_match(result["objects"], fixed, 1.5)
                        positive["object_hits"] += int(hit)
                        if hit and error is not None:
                            positive["object_match_errors_m"].append(error)
                    secondary = label_lookup.get("secondary_person", {}).get(index)
                    if secondary:
                        hit, _ = closest_match(result["objects"], secondary, 1.5)
                        positive["secondary_outside_matches"] += int(hit)
                row = {
                    "frame": frame["index"],
                    "timestamp_ns": frame["bag_timestamp_ns"],
                    "raw_detected": result["raw_detected"],
                    "event_detected": result["event_detected"],
                    "nearest_distance_m": result["nearest_distance_m"],
                    "processing_ms": result["processing_ms"],
                    "total_ms": total_ms,
                    "decode_ms": frame["decode_ms"],
                    "floor_coverage": result["floor_coverage"],
                    "objects": result["objects"],
                    "diagnostics": result["diagnostics"],
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        tracker.finalize()
        if current_episode is not None:
            episodes.append(current_episode)
        seconds_per_frame = bag["duration_s"] / max(1, bag["frame_count"] - 1)
        for event in episodes:
            event["duration_s"] = event["frame_count"] * args.step * seconds_per_frame
        duration_minutes = bag["duration_s"] / 60 if bag["duration_s"] else 0
        bag_report: dict[str, Any] = {
            "id": bag["id"],
            "processed_frames": processed,
            "source_frames": bag["frame_count"],
            "duration_s": bag["duration_s"],
            "raw_alarm_frames": raw_frames,
            "stable_alarm_frames": stable_frames,
            "first_raw_frame": first_raw,
            "first_stable_frame": first_stable,
            "stable_delay_frames": (
                first_stable - first_raw if first_stable is not None and first_raw is not None else None
            ),
            "max_raw_points": max_raw_points,
            "max_valid_points": max_valid_points,
            "event_count": len(episodes),
            "events_per_minute": len(episodes) / duration_minutes if duration_minutes else None,
            "events": episodes,
            "timing_ms": {
                "decode_mean": mean(decode_times) if decode_times else None,
                "decode_p50": percentile(decode_times, 50),
                "decode_p95": percentile(decode_times, 95),
                "decode_p99": percentile(decode_times, 99),
                "detector_mean": mean(detection_times) if detection_times else None,
                "detector_p50": percentile(detection_times, 50),
                "detector_p95": percentile(detection_times, 95),
                "detector_p99": percentile(detection_times, 99),
                "total_p50": percentile(total_times, 50),
                "total_p95": percentile(total_times, 95),
                "total_p99": percentile(total_times, 99),
            },
        }
        if bag["name"] == "doubleT_obstacle":
            bag_report["positive_metrics"] = {
                "primary_recall_inside": (
                    positive["primary_hits_inside"] / positive["primary_expected_inside"]
                    if positive["primary_expected_inside"] else None
                ),
                "primary_expected_inside": positive["primary_expected_inside"],
                "primary_hits_inside": positive["primary_hits_inside"],
                "primary_error_p95_m": percentile(positive["primary_match_errors_m"], 95),
                "object_recall": (
                    positive["object_hits"] / positive["object_expected"]
                    if positive["object_expected"] else None
                ),
                "object_expected": positive["object_expected"],
                "object_hits": positive["object_hits"],
                "object_error_p95_m": percentile(positive["object_match_errors_m"], 95),
                "secondary_outside_matches": positive["secondary_outside_matches"],
            }
        report["bags"][bag["name"]] = bag_report
        all_detection_times.extend(detection_times)
        all_decode_times.extend(decode_times)
        all_total_times.extend(total_times)
        current_memory = process_memory_mb()
        if current_memory is not None:
            peak_memory_mb = max(peak_memory_mb or 0, current_memory)
        print(
            f"{bag['name']}: frames={processed}, raw={raw_frames}, stable={stable_frames}, "
            f"events={len(episodes)}, detector_p95={percentile(detection_times, 95):.1f} ms"
        )
    report["overall"] = {
        "processed_frames": sum(item["processed_frames"] for item in report["bags"].values()),
        "event_count": sum(item["event_count"] for item in report["bags"].values()),
        "decode_p50_ms": percentile(all_decode_times, 50),
        "decode_p95_ms": percentile(all_decode_times, 95),
        "decode_p99_ms": percentile(all_decode_times, 99),
        "detector_p50_ms": percentile(all_detection_times, 50),
        "detector_p95_ms": percentile(all_detection_times, 95),
        "detector_p99_ms": percentile(all_detection_times, 99),
        "total_p95_ms": percentile(all_total_times, 95),
        "peak_process_memory_mb": peak_memory_mb,
        "visualization_max_points": config.visualization_max_points,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
