#!/usr/bin/env python3
"""Detection range on real rides with ray-cast synthetic obstacles.

For every clean bag, several start frames and several object types, an object
is placed at ``--start-distance`` ahead on the track (world-fixed, it
approaches as the train moves) and the ride is replayed. Recorded: distance at
first appearance as a candidate, at confirmation, how stable the alarm is
afterwards, and alarms far from the object (false alarms of that run).

    python scripts/synthetic_eval.py data/extracted/for_hackathon --out artifacts/synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gabarit  # noqa: E402,F401  (single-threaded BLAS)

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, discover_bags, open_bag  # noqa: E402
from gabarit.synth import SyntheticObstacle, inject  # noqa: E402


def run_case(args: tuple) -> dict:
    bag_path, shape, lateral, start, start_dist, cfg_dict, max_frames = args
    cfg = Config().with_overrides(cfg_dict)
    reader = BagReader(open_bag(bag_path))
    pipe = Pipeline(cfg)
    warm = max(0, start - 10)
    ob = SyntheticObstacle(shape=shape, s=start_dist, d=lateral)
    first_cand = first_conf = None
    conf_frames = total_after = 0
    other_alarm_frames = 0
    min_s = start_dist
    timeline = []
    t0 = time.perf_counter()
    for i in range(warm, min(len(reader), start + max_frames)):
        ts, raw = reader.read(i)
        active = i >= start
        if active and pipe.track.prev is not None:
            raw, _ = inject(raw, [ob], pipe.track.prev, pipe.track.calib, cfg.sensor.forward_axis)
        res = pipe.process_raw(raw, ts * 1e-9, frame_index=i)
        d = res.data
        if active:
            near = lambda o: abs(o["s_m"] - ob.s) < 3.0 + 0.02 * ob.s and abs(o["lateral_m"] - ob.d) < 1.2
            hit_c = any(near(c) for c in d["candidates"] if c["rejected"] is None)
            hit_o = any(near(o) for o in d["obstacles"])
            if hit_c and first_cand is None:
                first_cand = ob.s
            if hit_o and first_conf is None:
                first_conf = ob.s
            if first_conf is not None:
                total_after += 1
                conf_frames += int(hit_o)
            if any(not near(o) for o in d["obstacles"]):
                other_alarm_frames += 1
            timeline.append((i, round(ob.s, 2), int(hit_c), int(hit_o)))
            ob.s -= d["displacement_m"]
            min_s = min(min_s, ob.s)
            if ob.s < 8.0:
                break
        else:
            pass
    reader.close()
    return {
        "bag": Path(bag_path).name, "shape": shape, "lateral": lateral, "start_frame": start,
        "start_distance": start_dist, "first_candidate_m": first_cand, "confirmed_at_m": first_conf,
        "detected": first_conf is not None, "hold_ratio": (conf_frames / total_after) if total_after else None,
        "closest_m": round(min_s, 1), "other_alarm_frames": other_alarm_frames,
        "frames": len(timeline), "wall_s": round(time.perf_counter() - t0, 1),
    }


def moving_starts(bag, n: int, frames_needed: int) -> list[int]:
    """Start frames spread over the ride (the train must be moving there)."""
    total = bag.frame_count
    if total < frames_needed:
        return [0]
    return [int(x) for x in np.linspace(10, total - frames_needed, n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/synthetic"))
    ap.add_argument("--shapes", nargs="*", default=["person", "box_50cm", "child", "trolley"])
    ap.add_argument("--laterals", nargs="*", type=float, default=[0.0, 0.6])
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--start-distance", type=float, default=200.0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bags", nargs="*", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    cfg_dict = json.loads(json.dumps(cfg.to_dict()))
    bags = [b for b in discover_bags(args.data) if "obstacle" not in b.name]
    if args.bags:
        bags = [b for b in bags if b.name in args.bags]
    cases = []
    for b in bags:
        for st in moving_starts(b, args.starts, 80):
            for shape in args.shapes:
                for lat in args.laterals:
                    cases.append((str(b.path.parent), shape, lat, st, args.start_distance, cfg_dict, args.frames))
    print(f"{len(cases)} runs on {len(bags)} bags, {args.workers} workers")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_case, cases):
            results.append(r)
            print(f"  {r['bag'][:28]:28s} {r['shape']:9s} d={r['lateral']:+.1f} start={r['start_frame']:4d} "
                  f"cand@{r['first_candidate_m']} conf@{r['confirmed_at_m']} hold={r['hold_ratio']} closest={r['closest_m']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = {}
    for shape in args.shapes:
        rs = [r for r in results if r["shape"] == shape and r["closest_m"] < 60]  # object actually approached
        det = [r for r in rs if r["detected"]]
        conf = [r["confirmed_at_m"] for r in det]
        summary[shape] = {
            "runs": len(rs), "detected": len(det),
            "recall": round(len(det) / len(rs), 3) if rs else None,
            "confirm_distance_median_m": round(float(np.median(conf)), 1) if conf else None,
            "confirm_distance_p10_m": round(float(np.percentile(conf, 10)), 1) if conf else None,
            "confirm_distance_max_m": round(float(np.max(conf)), 1) if conf else None,
            "hold_ratio_mean": round(float(np.mean([r["hold_ratio"] for r in det if r["hold_ratio"] is not None])), 3) if det else None,
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
