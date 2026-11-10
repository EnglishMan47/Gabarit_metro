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
import hashlib
import json
import time
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, discover_bags, open_bag  # noqa: E402
from gabarit.synth import SHAPES, SyntheticObstacle, inject, inject_dust  # noqa: E402
from scripts.synthetic_reference import load_reference,reference_frame  # noqa: E402


class SyntheticMatches:
    """Associate detector evidence with injected returns, never just proximity.

    Nearby tunnel equipment is not a successful detection of a synthetic
    target. A held track remains associated only after actual point overlap.
    Ground truth is consumed after inference and never passed to the detector.
    """
    def __init__(self):
        self.track_ids: set[int] = set()

    def update(self, result, tracker, target_returns: np.ndarray, position: float) -> tuple[bool, bool, bool]:
        if result.data.get('state_reset'):
            self.track_ids.clear()
        tracking = getattr(getattr(tracker, 'cfg', None), 'tracking', None)
        max_speed = getattr(tracking, 'max_speed', 25.0)
        def overlaps(candidate):
            ids = candidate.point_index
            return ids is not None and bool(np.any(target_returns[ids]))

        candidate_hit = any(overlaps(t.cand) and t.cand.rejected is None for t in tracker.tracks
                            if t.misses == 0)
        matched = set()
        for track in tracker.tracks:
            if track.misses == 0:
                if overlaps(track.cand):
                    self.track_ids.add(track.id)
                    matched.add(track.id)
                else:
                    # An association to background invalidates the previous
                    # target identity; a later dropout must not revive it.
                    self.track_ids.discard(track.id)
            elif (track.misses > 0 and track.id in self.track_ids
                  and abs(track.s-position) < .5+.1*max_speed*track.misses):
                matched.add(track.id)
        confirmed = {t.id for t in tracker.tracks if t.confirmed}
        return candidate_hit, bool(confirmed & matched), bool(confirmed-matched)


def run_case(args: tuple) -> dict:
    bag_path, shape, lateral, start, start_dist, cfg_dict, max_frames, yaw, dust_fraction, seed, *extra = args
    cfg = Config().with_overrides(cfg_dict)
    reference_path = Path(extra[0])/(Path(bag_path).name+'.json') if extra and extra[0] else None
    reference = load_reference(str(reference_path)) if reference_path else None
    if reference is not None:
        for key in ('forward_axis','up_axis','scan_pattern'):
            if reference['config']['sensor'][key] != getattr(cfg.sensor,key):
                raise ValueError('synthetic reference sensor axes/pattern do not match evaluation config')
    geometry_digest = hashlib.sha256()
    reader = BagReader(open_bag(bag_path))
    pipe = Pipeline(cfg)
    warm = max(0, start - 10)
    negative = shape == "none"
    # Negative controls keep the same virtual approach distance and duration,
    # but never inject the anchor object into the recording.
    ob = SyntheticObstacle(shape="person" if negative else shape, s=start_dist, d=lateral, yaw_deg=yaw)
    rng = np.random.default_rng(seed)
    injected_beams = dust_beams = 0
    first_cand = first_conf = None
    conf_frames = total_after = 0
    other_alarm_frames = 0
    min_s = start_dist
    timeline = []
    matcher = SyntheticMatches()
    visible_frames = missed_visible_frames = 0
    eligible_visible_frames = eligible_missed_frames = 0
    post_confirmation_missed_eligible_frames = 0
    first_candidate_frame = first_confirmed_frame = None
    unmatched_alarms = []
    t0 = time.perf_counter()
    for i in range(warm, min(len(reader), start + max_frames)):
        ts, raw = reader.read(i)
        active = i >= start
        ref_track = ref_calib = None
        ref_delta = None
        if reference is not None:
            ref_track,ref_calib,ref_delta = reference_frame(reference,i,int(ts))
        target_raw = np.zeros(len(raw.points),bool)
        if active and not negative and (reference is not None or pipe.track.prev is not None):
            original = raw
            raw, added = inject(raw, [ob], ref_track if reference is not None else pipe.track.prev,
                                ref_calib if reference is not None else pipe.track.calib, cfg.sensor.forward_axis, rng,
                                up_axis=cfg.sensor.up_axis, scan_pattern=cfg.sensor.scan_pattern)
            for key in 'xyz':
                target_raw |= raw.points[key] != original.points[key]
            injected_beams += added
        if active and dust_fraction:
            original = raw
            raw, added = inject_dust(raw, dust_fraction, rng=rng, forward_axis=cfg.sensor.forward_axis,
                                     up_axis=cfg.sensor.up_axis)
            for key in 'xyz':
                target_raw &= raw.points[key] == original.points[key]
            dust_beams += added
        if active and reference is not None:
            for key in 'xyz':
                geometry_digest.update(raw.points[key].tobytes())
        res = pipe.process_raw(raw, ts * 1e-9, frame_index=i, preserve_source_index=True)
        d = res.data
        if active:
            target = target_raw[res.cloud.source_index]
            output = SimpleNamespace(tracks=pipe.output_tracks, cfg=cfg)
            hit_c, hit_o, other_alarm = matcher.update(res, output, target, ob.s)
            if np.any(target):
                visible_frames += 1
                missed_visible_frames += int(not hit_o)
                # Eligibility uses the injected position and configured range,
                # independently of the detector's fitted path and corridor.
                if cfg.track.start <= ob.s <= min(cfg.detection.max_range, cfg.track.horizon):
                    eligible_visible_frames += 1
                    eligible_missed_frames += int(not hit_o)
                    if first_conf is not None:
                        post_confirmation_missed_eligible_frames += int(not hit_o)
            if hit_c and not negative and first_cand is None:
                first_cand = ob.s
                first_candidate_frame = i
            if hit_o and not negative and first_conf is None:
                first_conf = ob.s
                first_confirmed_frame = i
            if first_conf is not None:
                total_after += 1
                conf_frames += int(hit_o)
            if other_alarm:
                other_alarm_frames += 1
                if len(unmatched_alarms)<64:
                    unmatched_alarms.append({'frame':i,'target_s':ob.s,'target_returns':int(target.sum()),
                        'associated_track_ids':sorted(matcher.track_ids),
                        'obstacles':[dict(t.public(), target_points=(int(target[t.cand.point_index].sum())
                                if t.cand.point_index is not None else 0))
                                     for t in pipe.output_tracks if t.confirmed]})
            timeline.append((i, round(ob.s, 2), int(hit_c), int(hit_o)))
            ob.s -= ref_delta if reference is not None else d["displacement_m"]
            min_s = min(min_s, ob.s)
            if ob.s < cfg.sensor.min_range:
                break
        else:
            pass
    reader.close()
    return {
        "bag": Path(bag_path).name, "shape": shape, "lateral": lateral, "start_frame": start,
        "yaw_deg": yaw, "dust_fraction": dust_fraction, "seed": seed, "negative_control": negative,
        "injected_beams": injected_beams, "dust_beams": dust_beams,
        "start_distance": start_dist, "first_candidate_m": first_cand, "confirmed_at_m": first_conf,
        "detected": first_conf is not None, "hold_ratio": (conf_frames / total_after) if total_after else None,
        "closest_m": round(min_s, 1), "other_alarm_frames": other_alarm_frames,
        "frames": len(timeline), "wall_s": round(time.perf_counter() - t0, 1),
        "matching": "injected_return_overlap_and_associated_track_history",
        "visible_frames": visible_frames, "missed_visible_frames": missed_visible_frames,
        "eligible_visible_frames": eligible_visible_frames, "eligible_missed_frames": eligible_missed_frames,
        "post_confirmation_missed_eligible_frames": post_confirmation_missed_eligible_frames,
        "first_candidate_frame": first_candidate_frame, "first_confirmed_frame": first_confirmed_frame,
        "evaluation_range_m": [cfg.track.start, min(cfg.detection.max_range, cfg.track.horizon)],
        "reference_trajectory": str(reference_path) if reference_path else None,
        "input_geometry_sha256": geometry_digest.hexdigest() if reference is not None else None,
        "unmatched_alarms": unmatched_alarms,
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
    ap.add_argument("--reference",type=Path,help="frozen clean-background trajectory directory; identical scenes across detector versions")
    ap.add_argument("--out", type=Path, default=Path("artifacts/synthetic"))
    ap.add_argument("--shapes", nargs="+", choices=sorted(SHAPES), default=["person", "box_50cm", "child", "trolley"])
    ap.add_argument("--yaw-degrees", nargs="+", type=float, default=[0.0], help="object orientations relative to track")
    ap.add_argument("--dust-fraction", type=float, default=0.0, help="share of returns replaced with weak aerosol echoes")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--negative-controls", action=argparse.BooleanOptionalAction, default=True,
                    help="also run each bag/start without an injected obstacle")
    ap.add_argument("--laterals", nargs="*", type=float, default=[0.0, 0.6])
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--start-distance", type=float, default=200.0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bags", nargs="*", default=None)
    args = ap.parse_args()
    if not 0 <= args.dust_fraction <= 1 or not np.isfinite(args.dust_fraction):
        ap.error("--dust-fraction must be within [0, 1]")
    if not all(np.isfinite(x) for x in args.yaw_degrees + args.laterals):
        ap.error("--yaw-degrees and --laterals must be finite")
    if args.frames < 1 or args.starts < 1 or args.workers < 1 or not np.isfinite(args.start_distance) or args.start_distance <= 0:
        ap.error("--frames, --starts, --workers and --start-distance must be positive")
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
                    for yaw in args.yaw_degrees:
                        cases.append((str(b.path.parent), shape, lat, st, args.start_distance, cfg_dict, args.frames,
                                      yaw, args.dust_fraction, args.seed + len(cases),str(args.reference) if args.reference else None))
            if args.negative_controls:
                cases.append((str(b.path.parent), "none", 0.0, st, args.start_distance, cfg_dict, args.frames,
                              0.0, args.dust_fraction, args.seed + len(cases),str(args.reference) if args.reference else None))
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
        rs = [r for r in results if r["shape"] == shape and r["closest_m"] < 60 and r["injected_beams"] > 0]
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
    negatives = [r for r in results if r["negative_control"]]
    summary["negative_controls"] = {
        "runs": len(negatives), "alarm_frames": sum(r["other_alarm_frames"] for r in negatives),
        "frames": sum(r["frames"] for r in negatives),
        "note": "No injected object; recorded bags require ground truth before calling alarms false positives.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
