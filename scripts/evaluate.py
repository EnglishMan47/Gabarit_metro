#!/usr/bin/env python3
"""Run the detector over bags and report alarms, events and timing.

Example:
    python scripts/evaluate.py data/extracted/for_hackathon --out artifacts/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, discover_bags  # noqa: E402


def events_from(frames: list[dict]) -> list[dict]:
    """Group frames with a confirmed obstacle into events keyed by track id."""
    ev: dict[int, dict] = {}
    for fr in frames:
        for ob in fr["obstacles"]:
            tid = ob["track_id"]
            e = ev.setdefault(tid, {"track_id": tid, "first_frame": fr["frame"], "frames": 0,
                                    "first_distance_m": ob["distance_m"], "min_distance_m": ob["distance_m"],
                                    "lateral_m": ob["lateral_m"], "box": ob["box"]})
            e["last_frame"] = fr["frame"]
            e["frames"] += 1
            e["min_distance_m"] = min(e["min_distance_m"], ob["distance_m"])
    return sorted(ev.values(), key=lambda e: e["first_frame"])


def run_bag(info, cfg: Config, step: int, stop: int | None, out_dir: Path | None, verbose: bool) -> dict:
    reader = BagReader(info)
    pipe = Pipeline(cfg)
    frames = []
    timing = []
    rejected: dict[str, int] = {}
    t_start = time.perf_counter()
    for i, ts, raw in reader.iterate(0, stop, step):
        res = pipe.process_raw(raw, ts * 1e-9, frame_index=i)
        d = res.data
        timing.append(d["timing_ms"])
        for c in d["candidates"]:
            if c["rejected"]:
                rejected[c["rejected"]] = rejected.get(c["rejected"], 0) + 1
        slim = {k: d[k] for k in ("frame", "stamp", "status", "obstacle", "nearest_distance_m", "free_distance_m",
                                  "observed_until_m", "speed_kmh", "obstacles", "pending", "calibration", "timing_ms")}
        slim["candidates"] = [c for c in d["candidates"] if not c["rejected"]]
        frames.append(slim)
        if verbose and (d["obstacle"] or d["pending"]):
            desc = "; ".join(f"#{o['track_id']} {o['distance_m']:.1f}m d={o['lateral_m']:+.2f} h={o['box']['h']} n={o['points']}"
                             for o in d["obstacles"] + d["pending"])
            print(f"  [{i:4d}] {d['status']:8s} v={d['speed_kmh']:5.1f} obs={d['observed_until_m']:5.1f} {desc}")
    wall = time.perf_counter() - t_start
    reader.close()
    tot = np.array([t["total"] for t in timing])
    events = events_from(frames)
    summary = {
        "bag": info.id,
        "frames": len(frames),
        "alarm_frames": int(sum(f["obstacle"] for f in frames)),
        "check_frames": int(sum(f["status"] == "CHECK" for f in frames)),
        "events": events,
        "event_count": len(events),
        "rejected": rejected,
        "observed_until_median_m": float(np.median([f["observed_until_m"] for f in frames])),
        "timing_ms": {
            k: {"mean": round(float(np.mean([t[k] for t in timing])), 2),
                "p95": round(float(np.percentile([t[k] for t in timing], 95)), 2)}
            for k in timing[0]
        },
        "fps_wall": round(len(frames) / wall, 2),
        "total_p95_ms": round(float(np.percentile(tot, 95)), 2),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{info.name}.jsonl", "w", encoding="utf-8") as fh:
            for fr in frames:
                fh.write(json.dumps(fr, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--bags", nargs="*", default=None, help="subset of bag names")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    bags = discover_bags(args.data)
    if args.bags:
        bags = [b for b in bags if b.name in args.bags]
    report = {"config": cfg.to_dict(), "bags": []}
    for info in bags:
        print(f"== {info.name} ({info.frame_count} frames)")
        s = run_bag(info, cfg, args.step, args.stop, args.out / "frames" if args.out else None, args.verbose)
        report["bags"].append(s)
        print(f"   alarms {s['alarm_frames']}/{s['frames']} frames, events {s['event_count']}, "
              f"check {s['check_frames']}, observed~{s['observed_until_median_m']:.0f} m, "
              f"total p95 {s['total_p95_ms']} ms, rejected {s['rejected']}")
        for e in s["events"]:
            print(f"   event #{e['track_id']}: frames {e['first_frame']}-{e['last_frame']} ({e['frames']}), "
                  f"first {e['first_distance_m']} m, d={e['lateral_m']}, box={e['box']}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
