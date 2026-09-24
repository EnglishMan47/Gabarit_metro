#!/usr/bin/env python3
"""Evaluation on the organizers' synthetic bag (cloud_with_fake_obj).

The synthetic obstacles are appended to every PointCloud2 after the original
scan, which ends with a run of empty (0, 0, 0) returns: all points after the
last empty return are ground truth. Each frame is run through the pipeline and
every ground-truth object is matched with the detector output:

  confirmed  - a confirmed obstacle track covers the object's points
  candidate  - a candidate overlaps it (with the rejection reason, if any)

Objects are linked across frames by range; the summary gives, per object, the
first range at which it was confirmed, how much of it lies inside the envelope,
and the false alarms (confirmed obstacles away from any synthetic object).

    python scripts/eval_organizers.py data/extracted/synthetic_organizers/cloud_with_fake_obj --out results/organizers
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabarit.cloud import from_raw, working_set  # noqa: E402
from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import CLS_OBSTACLE, Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, open_bag  # noqa: E402


def synthetic_mask(raw) -> np.ndarray:
    a = raw.points
    x, y, z = (np.asarray(a[k]) for k in ("x", "y", "z"))
    zero = np.flatnonzero((x == 0) & (y == 0) & (z == 0))
    m = np.zeros(len(a), bool)
    if len(zero):
        m[zero[-1] + 1:] = True
    return m


def split_objects(s: np.ndarray, gap: float = 5.0) -> list[np.ndarray]:
    order = np.argsort(s)
    cuts = np.flatnonzero(np.diff(s[order]) > gap) + 1
    return [order[g] for g in np.split(np.arange(len(order)), cuts)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/organizers"))
    ap.add_argument("--stop", type=int)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    reader = BagReader(open_bag(args.bag))
    pipe = Pipeline(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(reader) if args.stop is None else min(args.stop, len(reader))
    for i in range(n):
        ts, raw = reader.read(i)
        gt_raw = synthetic_mask(raw)
        cloud = from_raw(raw, cfg.sensor.forward_axis, cfg.sensor.min_range)
        a = raw.points
        x, y, z = (np.asarray(a[k], np.float32) for k in ("x", "y", "z"))
        r2 = x * x + y * y + z * z
        valid = np.isfinite(r2) & (r2 > cfg.sensor.min_range ** 2)
        gt = gt_raw[valid]
        keep = working_set(cloud, cfg.detection.max_range + 5.0)
        cloud = cloud.subset(keep)
        gt = gt[keep]
        res = pipe.process_cloud(cloud, ts * 1e-9, frame_index=i)
        d = res.data
        geo = res.geo
        objs = []
        gi = np.flatnonzero(gt & geo.in_range)
        if len(gi):
            for grp in split_objects(geo.s[gi]):
                idx = gi[grp]
                s, dd, hh = geo.s[idx], geo.d[idx], geo.h[idx]
                inside = geo.corridor[idx]
                rng = float(np.sqrt((res.lev[idx] ** 2).sum(axis=1)).min())
                o = {
                    "range_m": round(rng, 2), "s": [round(float(s.min()), 2), round(float(s.max()), 2)],
                    "d": [round(float(dd.min()), 2), round(float(dd.max()), 2)],
                    "h": [round(float(hh.min()), 2), round(float(hh.max()), 2)],
                    "points": int(len(idx)), "inside": int(inside.sum()),
                    "confirmed": bool((res.labels[idx] == CLS_OBSTACLE).any()),
                }
                match = [c for c in d["candidates"]
                         if c["box"]["s"][0] <= o["s"][1] + 1.0 and c["box"]["s"][1] >= o["s"][0] - 1.0
                         and c["box"]["d"][0] <= o["d"][1] + 0.3 and c["box"]["d"][1] >= o["d"][0] - 0.3]
                o["candidate"] = None if not match else (match[0]["rejected"] or "OK")
                objs.append(o)
        # confirmed obstacles not explained by any synthetic object
        false = []
        for ob in d["obstacles"]:
            s0 = ob["s_m"]
            if not any(o["s"][0] - 3.0 <= s0 <= o["s"][1] + 3.0 for o in objs):
                false.append(round(float(ob["distance_m"]), 1))
        rows.append({"frame": i, "status": d["status"], "speed_kmh": d["speed_kmh"],
                     "objects": objs, "false": false})
        if i % 100 == 0:
            print(f"frame {i}/{n}", file=sys.stderr)
    with open(args.out / "frames.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # link objects across frames: range decreases (or grows slowly when the
    # train stands / the recording goes back); a jump starts a new object
    tracks: list[dict] = []
    for r in rows:
        for o in r["objects"]:
            best = None
            for t in tracks:
                if r["frame"] - t["last_frame"] <= 30 and abs(o["s"][0] - t["last_s"]) < 12.0:
                    best = t
                    break
            if best is None:
                best = {"id": len(tracks) + 1, "obs": []}
                tracks.append(best)
            best["obs"].append((r["frame"], o))
            best["last_frame"] = r["frame"]
            best["last_s"] = o["s"][0]
    summary = []
    for t in tracks:
        obs = t["obs"]
        if len(obs) < 5:
            continue
        conf = [o["range_m"] for _, o in obs if o["confirmed"]]
        ins = [o for _, o in obs if o["inside"] > 0]
        reasons = Counter(o["candidate"] or "none" for _, o in obs if not o["confirmed"])
        summary.append({
            "id": t["id"], "frames": [obs[0][0], obs[-1][0]],
            "range_m": [obs[0][1]["range_m"], min(o["range_m"] for _, o in obs)],
            "lateral_m": [round(float(np.median([o["d"][0] for _, o in obs])), 2),
                          round(float(np.median([o["d"][1] for _, o in obs])), 2)],
            "height_m": [round(float(np.median([o["h"][0] for _, o in obs])), 2),
                         round(float(np.median([o["h"][1] for _, o in obs])), 2)],
            "frames_inside": len(ins), "frames_seen": len(obs),
            "frames_confirmed": len(conf),
            "first_confirmed_range_m": max(conf) if conf else None,
            "missed_reasons": dict(reasons.most_common(4)),
        })
    false_frames = [r["frame"] for r in rows if r["false"]]
    out = {"bag": str(args.bag), "frames": n, "objects": summary,
           "false_alarm_frames": len(false_frames),
           "false_alarm_ranges": sorted({f for r in rows for f in r["false"]})[:50]}
    (args.out / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for o in summary:
        print(o)
    print("false-alarm frames:", len(false_frames), false_frames[:40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
