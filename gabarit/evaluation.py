"""Evaluation helpers with explicit separation of measurements and ground truth.

Alarm/event counts describe detector output. FP/FN only exist when externally
provided frame labels are matched to the frames actually processed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gabarit.egomotion import propagate_track
from gabarit.track import TrackModel


def statistics(values: Iterable[float | None]) -> dict:
    """Finite-only descriptive statistics; empty inputs produce JSON nulls."""
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": len(values), "mean": round(float(np.mean(values)), 4),
            **{f"p{q}": round(float(np.percentile(values, q)), 4) for q in (50, 95, 99)},
            "max": round(max(values), 4)}


def deadline_metrics(values: Iterable[float | None], budget_ms: float = 100.0) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    count = len(values)
    met = sum(value <= budget_ms for value in values)
    mean = float(np.mean(values)) if count else None
    return {"budget_ms": budget_ms, "measured_frames": count,
            "within_budget_frames": met, "deadline_miss_frames": count - met,
            "within_budget_share": met / count if count else None,
            "deadline_miss_share": (count - met) / count if count else None,
            "throughput_fps": round(1000.0 / mean, 3) if mean and mean > 0 else None}


def track_events(frames: list[dict]) -> list[dict]:
    """Tracks observed as pending/confirmed, including tracks never confirmed.

    Distances use the first recorded Euclidean range for each stage. The
    tracker identity supplies association; this is not ground-truth matching.
    """
    events: dict[tuple[int, int], dict] = {}
    sequence = 0
    for ordinal, fr in enumerate(frames):
        if fr.get("state_reset"):
            sequence += 1
        confirmed = {ob["track_id"] for ob in fr.get("obstacles", [])}
        seen = set()
        for ob in fr.get("pending", []) + fr.get("obstacles", []):
            tid = ob["track_id"]
            if tid in seen:
                continue
            seen.add(tid)
            event = events.setdefault((sequence, tid), {
                "track_id": tid, "sequence": sequence, "first_candidate_frame": fr["frame"],
                "first_candidate_stamp": fr.get("stamp"),
                "first_candidate_distance_m": ob.get("distance_m"),
                "first_candidate_track_distance_m": (ob.get("box", {}).get("s") or [None])[0],
                "candidate_frames": 0, "first_confirmed_frame": None,
                "first_confirmed_distance_m": None, "first_confirmed_track_distance_m": None,
                "confirmation_delay_frames": None, "confirmation_delay_processed_frames": None,
                "confirmation_delay_s": None, "confirmed_frames": 0,
                "_first_processed_index": ordinal,
            })
            event["last_candidate_frame"] = fr["frame"]
            event["candidate_frames"] += 1
            if tid in confirmed:
                if event["first_confirmed_frame"] is None:
                    event["first_confirmed_frame"] = fr["frame"]
                    event["first_confirmed_distance_m"] = ob.get("distance_m")
                    event["first_confirmed_track_distance_m"] = (ob.get("box", {}).get("s") or [None])[0]
                    event["confirmation_delay_frames"] = fr["frame"] - event["first_candidate_frame"]
                    event["confirmation_delay_processed_frames"] = ordinal - event["_first_processed_index"]
                    start, now = event["first_candidate_stamp"], fr.get("stamp")
                    event["confirmation_delay_s"] = round(now - start, 6) if start is not None and now is not None else None
                event["confirmed_frames"] += 1
    for event in events.values():
        del event["_first_processed_index"]
    return sorted(events.values(), key=lambda event: (event["first_candidate_frame"], event["track_id"]))


def events_from(frames: list[dict]) -> list[dict]:
    """Backward-compatible confirmed-event fields plus first candidate metrics."""
    stages = {(event["sequence"], event["track_id"]): event for event in track_events(frames)}
    events = {}
    sequence = 0
    for fr in frames:
        if fr.get("state_reset"):
            sequence += 1
        for ob in fr.get("obstacles", []):
            tid = ob["track_id"]
            key = (sequence, tid)
            event = events.setdefault(key, {
                **stages[key], "track_id": tid, "first_frame": fr["frame"], "frames": 0,
                "first_distance_m": ob["distance_m"], "min_distance_m": ob["distance_m"],
                "lateral_m": ob.get("lateral_m"), "box": ob.get("box"),
            })
            event["last_frame"] = fr["frame"]
            event["frames"] += 1
            event["min_distance_m"] = min(event["min_distance_m"], ob["distance_m"])
    return sorted(events.values(), key=lambda event: event["first_frame"])


def path_stability(previous: TrackModel | None, current: TrackModel,
                   displacement_m: float, min_distance_m: float = 3.0) -> dict | None:
    """Motion-compensated temporal jitter on the common envelope horizon.

    This measures consistency of the estimated path, not its error relative
    to the true rails. Held estimates are explicitly tagged and summarized
    separately so that retaining a stale box cannot imply measured accuracy.
    """
    if previous is None or len(previous.s) < 2 or len(current.s) < 2:
        return None
    if not math.isfinite(displacement_m):
        return None
    common_until = min(current.observed_until, previous.observed_until - displacement_m)
    if common_until <= min_distance_m or not np.any(previous.s - displacement_m > 0):
        return None
    prior = propagate_track(previous, displacement_m)
    lateral = np.interp(current.s, prior.s, prior.c, left=np.nan, right=np.nan)
    vertical = np.interp(current.s, prior.s, prior.z, left=np.nan, right=np.nan)
    valid = ((current.s >= min_distance_m) & (current.s <= common_until)
             & np.isfinite(lateral) & np.isfinite(vertical)
             & np.isfinite(current.c) & np.isfinite(current.z))
    if not valid.any():
        return None
    dc, dz = current.c[valid] - lateral[valid], current.z[valid] - vertical[valid]
    return {"samples": int(valid.sum()), "common_until_m": round(float(common_until), 3),
            "held_pair": bool(previous.coast_frames or current.coast_frames),
            "lateral_rmse_m": round(float(np.sqrt(np.mean(dc * dc))), 6),
            "vertical_rmse_m": round(float(np.sqrt(np.mean(dz * dz))), 6),
            "lateral_abs_p95_m": round(float(np.percentile(np.abs(dc), 95)), 6),
            "lateral_abs_max_m": round(float(np.max(np.abs(dc))), 6)}


def envelope_metrics(frames: list[dict], start_m: float = 3.0) -> dict:
    count = len(frames)
    envelope = [float(fr.get("envelope_until_m", fr.get("observed_until_m", 0))) for fr in frames]
    observed = [float(fr.get("current_observed_until_m", fr.get("observed_until_m", 0))) for fr in frames]
    held = [bool(fr.get("envelope_held", False)) for fr in frames]
    present = [horizon > start_m for horizon in envelope]
    degraded = sum(fr.get("status") == "DEGRADED" for fr in frames)
    no_envelope = sum(not value for value in present)
    held_count = sum(held)
    pairs = [fr["path_stability"] for fr in frames if fr.get("path_stability")]
    fresh = [pair for pair in pairs if not pair["held_pair"]]
    retained = [pair for pair in pairs if pair["held_pair"]]
    def stability_stats(items):
        return {"frame_pairs": len(items),
                "lateral_rmse_m": statistics(pair["lateral_rmse_m"] for pair in items),
                "vertical_rmse_m": statistics(pair["vertical_rmse_m"] for pair in items),
                "lateral_abs_p95_m": statistics(pair["lateral_abs_p95_m"] for pair in items)}
    return {"current_observed_until_m": statistics(observed), "envelope_until_m": statistics(envelope),
            "held_envelope_until_m": statistics(horizon for horizon, is_held in zip(envelope, held) if is_held),
            "degraded_frames": degraded, "degraded_share": degraded / count if count else None,
            "held_frames": held_count, "held_share": held_count / count if count else None,
            "no_envelope_frames": no_envelope, "no_envelope_share": no_envelope / count if count else None,
            "availability_transitions": sum(a != b for a, b in zip(present, present[1:])),
            "envelope_loss_transitions": sum(a and not b for a, b in zip(present, present[1:])),
            "motion_compensated_stability": {"fresh_pairs": stability_stats(fresh),
                                               "including_held_pairs": stability_stats(pairs),
                                               "held_pairs": stability_stats(retained)}}


@dataclass(frozen=True)
class Annotations:
    bags: dict[str, dict[int, bool | None]]


def load_annotations(path: str | Path) -> Annotations:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data.get("schema_version") != 1 or not isinstance(data.get("bags"), dict)):
        raise ValueError("annotations require schema_version=1 and a bags mapping")
    bags = {}
    for bag_id, bag in data["bags"].items():
        if not isinstance(bag_id, str) or not bag_id or not isinstance(bag, dict) or not isinstance(bag.get("frames"), list):
            raise ValueError("each annotation bag needs a non-empty id and frames list")
        labels = {}
        for entry in bag["frames"]:
            if not isinstance(entry, dict) or "frame" not in entry or "obstacle" not in entry:
                raise ValueError(f"annotation in {bag_id} needs frame and obstacle")
            index, label = entry["frame"], entry["obstacle"]
            if type(index) is not int or index < 0:
                raise ValueError(f"annotation frame in {bag_id} must be a nonnegative integer")
            if label is not None and type(label) is not bool:
                raise ValueError(f"annotation obstacle in {bag_id} must be true, false, or null")
            if index in labels:
                raise ValueError(f"duplicate annotation frame {index} in {bag_id}")
            labels[index] = label
        bags[bag_id] = labels
    return Annotations(bags)


def score_frame_labels(frames: list[dict], labels: dict[int, bool | None] | None,
                       fn_cost: float = 1.0) -> dict:
    """Binary frame-level score on the intersection of known labels and output."""
    if not math.isfinite(fn_cost) or fn_cost < 0:
        raise ValueError("fn_cost must be finite and nonnegative")
    if labels is None:
        return {"available": False, "reason": "no_annotations_for_bag"}
    tp = tn = fp = fn = unknown = 0
    processed = set()
    for frame in frames:
        index = frame["frame"]
        if index in processed:
            raise ValueError(f"duplicate processed frame {index}")
        processed.add(index)
        truth = labels.get(index)
        if truth is None:
            unknown += 1
            continue
        predicted = bool(frame["obstacle"])
        tp += int(truth and predicted)
        tn += int(not truth and not predicted)
        fp += int(not truth and predicted)
        fn += int(truth and not predicted)
    count = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"available": True, "labelled_frames_processed": count, "excluded_unknown_frames": unknown,
            "supplied_annotation_frames": len(labels),
            "annotation_frames_not_processed": len(set(labels) - processed),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity,
            "accuracy": (tp + tn) / count if count else None,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "fn_cost": fn_cost, "weighted_error_cost": fp + fn_cost * fn,
            "weighted_error_cost_per_labelled_frame": (fp + fn_cost * fn) / count if count else None}
