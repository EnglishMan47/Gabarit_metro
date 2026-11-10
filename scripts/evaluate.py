#!/usr/bin/env python3
"""Run the detector over bags and report alarms, events and timing.

Example:
    python scripts/evaluate.py data/extracted/for_hackathon --out artifacts/eval
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabarit.config import Config  # noqa: E402
from gabarit.evaluation import (Annotations, deadline_metrics, envelope_metrics, events_from,
                                load_annotations, path_stability, score_frame_labels,
                                statistics, track_events)  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, discover_bags  # noqa: E402


def run_bag(info, cfg: Config, step: int, stop: int | None, out_dir: Path | None, verbose: bool,
            progress: int = 0, annotations: Annotations | None = None, fn_cost: float = 1.0,
            start: int = 0, save_path: bool = False) -> dict:
    if step <= 0:
        raise ValueError("step must be positive")
    if stop is not None and stop < 0:
        raise ValueError("stop must be nonnegative")
    if progress < 0:
        raise ValueError("progress must be nonnegative")
    if start < 0:
        raise ValueError("start must be nonnegative")
    reader = BagReader(info)
    pipe = Pipeline(cfg)
    frames = []
    timing = []
    rejected: dict[str, int] = {}
    t_start = time.perf_counter()
    previous = None
    end = min(stop if stop is not None else len(reader), len(reader))
    try:
        for i in range(start, end, step):
            read_start = time.perf_counter()
            ts, raw = reader.read(i)
            read_end = time.perf_counter()
            res = pipe.process_raw(raw, ts * 1e-9, frame_index=i)
            process_end = time.perf_counter()
            d = res.data
            measured = {**d["timing_ms"],
                        "read_decode": round((read_end - read_start) * 1000, 4),
                        "process_call": round((process_end - read_end) * 1000, 4),
                        "end_to_end": round((process_end - read_start) * 1000, 4)}
            timing.append(measured)
            for c in d["candidates"]:
                if c["rejected"]:
                    rejected[c["rejected"]] = rejected.get(c["rejected"], 0) + 1
            slim = {k: d[k] for k in ("frame", "stamp", "status", "obstacle", "nearest_distance_m", "free_distance_m",
                                      "observed_until_m", "speed_kmh", "obstacles", "pending", "calibration")}
            for key in ("current_observed_until_m", "envelope_until_m", "envelope_held",
                        "envelope_coasted_frames", "displacement_m", "ego_quality", "points", "realtime", "state_reset", "path_guard", "models"):
                if key in d:
                    slim[key] = d[key]
            slim["timing_ms"] = measured
            slim["candidates"] = [c for c in d["candidates"] if not c["rejected"]]
            slim["path_stability"] = path_stability(None if d.get("state_reset") else previous, res.track,
                                                     d.get("displacement_m", 0.0), cfg.track.start)
            previous = res.track
            if save_path:
                slim['track'] = res.track.public()
            frames.append(slim)
            if progress and len(frames) % progress == 0:
                elapsed = time.perf_counter() - t_start
                left = len(range(i + step, end, step))
                print(f"   {info.name}: frame {i + 1}/{len(reader)}, {elapsed:.0f} s, "
                      f"~{elapsed / len(frames) * left:.0f} s left", file=sys.stderr, flush=True)
            if verbose and (d["obstacle"] or d["pending"]):
                desc = "; ".join(f"#{o['track_id']} {o['distance_m']:.1f}m d={o['lateral_m']:+.2f} "
                                 f"h={o['box']['h']} n={o['points']}" for o in d["obstacles"] + d["pending"])
                print(f"  [{i:4d}] {d['status']:8s} v={d['speed_kmh']:5.1f} "
                      f"obs={d['observed_until_m']:5.1f} {desc}")
    finally:
        reader.close()
    wall = time.perf_counter() - t_start
    events = events_from(frames)
    candidate_events = track_events(frames)
    timing_summary = {key: statistics(item.get(key) for item in timing)
                      for key in sorted({key for item in timing for key in item})}
    labels = annotations.bags.get(info.id) if annotations else None
    observed = statistics(frame["observed_until_m"] for frame in frames)
    filtered = [frame.get("points", {}).get("filtered") for frame in frames]
    filter_examined = [frame.get("points", {}).get("filter_examined") for frame in frames]
    removed_total = int(sum(value for value in filtered if value is not None)) if any(value is not None for value in filtered) else None
    examined_total = int(sum(value for value in filter_examined if value is not None)) if any(value is not None for value in filter_examined) else None
    summary = {
        "bag": info.id,
        "frames": len(frames),
        "alarm_frames": int(sum(f["obstacle"] for f in frames)),
        "check_frames": int(sum(f["status"] == "CHECK" for f in frames)),
        "events": events,
        "event_count": len(events),
        "candidate_events": candidate_events,
        "candidate_event_count": len(candidate_events),
        "first_candidate_distance_m": statistics(event["first_candidate_distance_m"] for event in candidate_events),
        "first_confirmed_distance_m": statistics(event["first_confirmed_distance_m"] for event in events),
        "rejected": rejected,
        "observed_until_median_m": observed["p50"],
        "envelope": envelope_metrics(frames, cfg.track.start),
        "timing_ms": timing_summary,
        "fps_wall": round(len(frames) / wall, 2) if wall > 0 else None,
        "wall_time_s": round(wall, 4),
        "total_p95_ms": timing_summary.get("total", {}).get("p95"),
        "realtime": {"pipeline": deadline_metrics(item.get("total") for item in timing),
                     "end_to_end": deadline_metrics(item["end_to_end"] for item in timing)},
        "filter": {"enabled": cfg.filter.enabled, "removed_points": removed_total,
                   "examined_points": examined_total,
                   "removed_share": removed_total / examined_total if examined_total and removed_total is not None else None,
                   "measured_frames": sum(value is not None for value in filtered),
                   "removed_points_per_frame": statistics(filtered)},
        "supervised_metrics": score_frame_labels(frames, labels, fn_cost),
        "sampling": {"start_inclusive": start, "step": step, "stop_exclusive": stop, "source_frames": len(reader)},
        "temporal_resets": sum(bool(frame.get("state_reset")) for frame in frames),
        "path_guard": {"measured_frames":sum('path_guard' in f for f in frames),
                       "active_frames":sum(bool(f.get('path_guard',{}).get('active')) for f in frames)},
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{info.name}.jsonl", "w", encoding="utf-8") as fh:
            for fr in frames:
                fh.write(json.dumps(fr, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--bags", nargs="*", default=None, help="subset of bag names")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="first source frame; starts a new temporal sequence")
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--progress", type=int, default=0, help="print progress to stderr every N frames")
    ap.add_argument("--annotations", type=Path, default=None, help="frame-level labels JSON; see docs/EVALUATION_FINALIST.md")
    ap.add_argument("--save-path", action="store_true", help="include fitted path geometry for manual review")
    ap.add_argument("--fn-cost", type=float, default=1.0, help="cost per false negative; false positive cost is 1")
    args = ap.parse_args(argv)
    if args.step <= 0 or (args.stop is not None and args.stop < 0) or args.progress < 0 or args.start < 0:
        ap.error("--step must be positive; --start, --stop and --progress must be nonnegative")
    if not np.isfinite(args.fn_cost) or args.fn_cost < 0:
        ap.error("--fn-cost must be finite and nonnegative")
    cfg = Config.load(args.config)
    annotations = load_annotations(args.annotations) if args.annotations else None
    bags = discover_bags(args.data)
    if args.bags:
        bags = [b for b in bags if b.name in args.bags]
    report = {"config": cfg.to_dict(), "bags": [],
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "machine": platform.machine(),
                              "processor": platform.processor(), "logical_cpus": os.cpu_count()}}
    if annotations is not None:
        selected_ids = {bag.id for bag in bags}
        report["annotations"] = {"path": str(args.annotations), "schema_version": 1,
                                 "unmatched_bag_ids": sorted(set(annotations.bags) - selected_ids),
                                 "selected_bag_ids_without_annotations": sorted(selected_ids - set(annotations.bags))}
    for info in bags:
        print(f"== {info.name} ({info.frame_count} frames)")
        s = run_bag(info, cfg, args.step, args.stop, args.out / "frames" if args.out else None, args.verbose,
                    args.progress, annotations, args.fn_cost, args.start, args.save_path)
        report["bags"].append(s)
        observed_text = "-" if s["observed_until_median_m"] is None else f"{s['observed_until_median_m']:.0f}"
        print(f"   alarms {s['alarm_frames']}/{s['frames']} frames, events {s['event_count']}, "
              f"check {s['check_frames']}, observed~{observed_text} m, "
              f"total p95 {s['total_p95_ms']} ms, rejected {s['rejected']}")
        for e in s["events"]:
            print(f"   event #{e['track_id']}: frames {e['first_frame']}-{e['last_frame']} ({e['frames']}), "
                  f"first {e['first_distance_m']} m, d={e['lateral_m']}, box={e['box']}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
