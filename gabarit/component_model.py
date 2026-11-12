"""Evaluate the MIT-licensed Sakhalin Ai baseline_v3 component forest.

Upstream model and feature contract: Copyright (c) 2026 saic, MIT license;
the complete copyright and permission notice is vendor/sakhalin_ai/LICENSE.

The eight features and all 17 trees are from the pinned upstream artifacts in
``vendor/sakhalin_ai``. The output is an uncalibrated assist score, not a safety
decision or an obstacle probability. Components here are extracted differently
from upstream; the optional positive-only policy therefore adds local geometry
and temporal gates. Existing detector acceptance is never vetoed by the model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SOURCE_REVISION = "3cfd31843b911f278c08f49e53a09b1a5efa1d92"
MODEL_SHA256 = "286fbf5dc352ca8b2d6ee386092862309880ecac0549b48402f49ca137a61661"
FEATURE_NAMES = (
    "point_count", "extent_x_m", "extent_y_m", "extent_z_m", "volume_m3",
    "density_points_per_m3", "centroid_distance_m", "nearest_distance_m",
)


@dataclass(frozen=True)
class ComponentFeatures:
    point_count: int
    extent_x_m: float
    extent_y_m: float
    extent_z_m: float
    volume_m3: float
    density_points_per_m3: float
    centroid_distance_m: float
    nearest_distance_m: float

    @classmethod
    def from_points(cls, points: np.ndarray) -> "ComponentFeatures":
        """Use actual sensor-origin XYZ before leveling or rail-height removal.

        XYZ are cast to float32, as in upstream ``const float* xyz``. Extrema,
        sums, density and distances are then computed in double precision.
        Sign flips of sensor axes leave all these features unchanged.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            xyz32 = np.asarray(points, dtype=np.float32)
        if xyz32.ndim != 2 or xyz32.shape[1] != 3 or len(xyz32) == 0:
            raise ValueError("component requires a nonempty (N, 3) XYZ array")
        if not np.isfinite(xyz32).all():
            raise ValueError("component XYZ must be finite float32 values")
        xyz = xyz32.astype(np.float64)
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        volume = max(float(np.prod(extent)), 1e-6)
        centroid = xyz.sum(axis=0) / len(xyz)
        return cls(
            len(xyz), *map(float, extent), volume, len(xyz) / volume,
            float(math.hypot(*centroid)),
            float(np.sqrt(np.square(xyz).sum(axis=1)).min()),
        )

    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)


@dataclass(frozen=True)
class ComponentPrediction:
    score: float
    model_threshold: float
    tree_count: int
    features: ComponentFeatures


def _validate_node(node: dict[str, Any], depth: int = 0) -> None:
    if depth > 32 or not isinstance(node, dict):
        raise ValueError("invalid component forest node")
    for key in ("positive_weight", "negative_weight"):
        value = node.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid component forest leaf weight")
    if node["positive_weight"] + node["negative_weight"] <= 0:
        raise ValueError("empty component forest leaf")
    if "feature_index" in node:
        index = node["feature_index"]
        if type(index) is not int or not 0 <= index < len(FEATURE_NAMES):
            raise ValueError("invalid component forest feature index")
        if node.get("feature") != FEATURE_NAMES[index]:
            raise ValueError("component forest feature order mismatch")
        if not math.isfinite(node.get("threshold", math.nan)):
            raise ValueError("invalid component forest threshold")
        _validate_node(node.get("left"), depth + 1)
        _validate_node(node.get("right"), depth + 1)


@lru_cache(maxsize=1)
def _load_forest() -> tuple[tuple[dict[str, Any], ...], float]:
    raw = files("gabarit.vendor.sakhalin_ai").joinpath("baseline_v3_runtime_policy.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("vendored component model checksum mismatch")
    policy = json.loads(raw)
    model = policy["assist_score_model"]
    if model["format"] != "lidar-component-noise-forest-lite-v1":
        raise ValueError("unsupported component model format")
    if tuple(model["features"]) != FEATURE_NAMES or model["aggregation"] != "mean_tree_probability":
        raise ValueError("unsupported component model feature/aggregation contract")
    trees = tuple(model["trees"])
    if len(trees) != 17:
        raise ValueError("unexpected component forest size")
    for tree in trees:
        _validate_node(tree)
    threshold = float(policy["decision_policy"]["model_assist"]["threshold"])
    return trees, threshold


class ComponentModel:
    """Dependency-free inference of original or independently trained weights."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.name = "baseline_v3_assist_score"
        self.source_revision = SOURCE_REVISION
        self.weight_sha256 = MODEL_SHA256
        self.assist_available = True
        self.fallback_reason = None
        self._feature_float32 = False
        if model_path is None:
            self.trees, self.threshold = _load_forest()
            return
        path = Path(model_path)
        if path.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("component model exceeds size limit")
        raw = path.read_bytes()
        model = json.loads(raw)
        if (model.get("format") != "gabarit-component-forest-v1" or
                tuple(model.get("features", ())) != FEATURE_NAMES or
                model.get("aggregation") != "mean_tree_probability"):
            raise ValueError("unsupported trained component forest contract")
        trees = model.get("trees")
        if not isinstance(trees, list) or not 1 <= len(trees) <= 128:
            raise ValueError("invalid trained component forest size")
        for tree in trees:
            _validate_node(tree)
        threshold = model.get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid trained component forest threshold")
        self.trees, self.threshold = tuple(trees), float(threshold)
        self._feature_float32 = True  # sklearn's tree inference input contract
        self.weight_sha256 = hashlib.sha256(raw).hexdigest()
        self.name = "gabarit_trained_component_score"
        self.source_revision = self.weight_sha256[:16]
        if model.get("training", {}).get("calibration_zero_false_positives") is not True:
            self.assist_available = False
            self.fallback_reason = "trained model did not pass independent calibration"

    @classmethod
    def optional(cls, model_path: str | Path | None = None) -> "ComponentModel":
        """Missing optional weights never enable an unintended assist model."""
        try:
            return cls(model_path)
        except (OSError, ValueError, TypeError, RecursionError) as error:
            fallback = cls()
            fallback.assist_available = False
            fallback.fallback_reason = f"{type(error).__name__}: {error}"
            return fallback

    def predict(self, features: ComponentFeatures) -> ComponentPrediction:
        values = features.vector()
        if self._feature_float32:
            values = tuple(float(value) for value in np.asarray(values, dtype=np.float32))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("component features must be finite")
        score = 0.0
        for tree in self.trees:
            node = tree
            while "feature_index" in node:
                node = node["left"] if values[node["feature_index"]] <= node["threshold"] else node["right"]
            positive = node["positive_weight"]
            score += positive / (positive + node["negative_weight"])
        return ComponentPrediction(score / len(self.trees), self.threshold, len(self.trees), features)

    def score_points(self, points: np.ndarray) -> ComponentPrediction:
        return self.predict(ComponentFeatures.from_points(points))

    def score_candidate(self, candidate: Any, source_xyz: np.ndarray) -> ComponentPrediction:
        """Index the frame cloud; never approximate features using a track box."""
        xyz = np.asarray(source_xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("source_xyz must have shape (N, 3)")
        if candidate.point_index is None:
            raise ValueError("candidate has no source point indices")
        indices = np.asarray(candidate.point_index)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
            raise ValueError("candidate point indices must be a nonempty integer vector")
        if np.any(indices < 0) or np.any(indices >= len(xyz)):
            raise ValueError("candidate point indices outside frame cloud")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("candidate point indices must be unique")
        return self.score_points(xyz[indices])


@dataclass
class _History:
    bounds: np.ndarray
    hits: int


def _candidate_bounds(candidate: Any) -> np.ndarray:
    return np.array([
        [candidate.s_near, candidate.s_far], [candidate.d_min, candidate.d_max],
        [candidate.h_min, candidate.h_max],
    ], dtype=np.float64)


class ComponentModelAssistant:
    """Score every candidate; optionally rescue only a weak score rejection.

    ``score`` is the normal policy and changes diagnostics only. ``assist`` may
    clear ``low_score`` after multiple consecutive geometrically matched scans.
    No other rejection and no existing accepted component are changed. Ordinary
    Tracker confirmation still runs after this policy. Call reset() on a bag
    switch/seek; nonincreasing timestamps or gaps over 0.5 s also clear history.
    """

    def __init__(self, model: ComponentModel | None = None) -> None:
        self.model = model or ComponentModel()
        self._previous: list[_History] = []
        self._last_stamp: float | None = None

    def reset(self) -> None:
        self._previous = []
        self._last_stamp = None

    @staticmethod
    def _geometry_reason(candidate: Any, prediction: ComponentPrediction, *,
                         observed_until_m: float, path_reliable: bool,
                         half_width_m: float, support_accepted: bool = False) -> str:
        if candidate.rejected != "low_score" and not (support_accepted and candidate.rejected is None):
            return "existing_acceptance" if candidate.rejected is None else "protected_rejection"
        if candidate.low or candidate.transparent or candidate.single_beam:
            return "special_geometry_branch"
        if not path_reliable:
            return "unreliable_path"
        bounds = _candidate_bounds(candidate)
        if not np.isfinite(bounds).all() or np.any(bounds[:, 1] < bounds[:, 0]):
            return "invalid_geometry"
        if candidate.s_near < 2.0 or candidate.s_far > observed_until_m:
            return "outside_observed_path"
        if candidate.d_min < -half_width_m + 0.10 or candidate.d_max > half_width_m - 0.10:
            return "boundary_component"
        if candidate.h_min < 0.12 or candidate.h_max > 2.4:
            return "ground_or_high_component"
        if not 6 <= prediction.features.point_count < 1000 or candidate.n_rings < 2:
            return "insufficient_multibeam_support"
        sizes = bounds[:, 1] - bounds[:, 0]
        if sizes[0] > 1.5 or sizes[1] > 1.9 or not 0.15 <= sizes[2] <= 2.4:
            return "unsupported_component_shape"
        return "eligible"

    def _match(self, bounds: np.ndarray, used: set[int], forward_delta_m: float) -> int | None:
        best, best_distance = None, math.inf
        center = bounds.mean(axis=1)
        size = np.maximum(bounds[:, 1] - bounds[:, 0], 0.08)
        for index, previous in enumerate(self._previous):
            if index in used:
                continue
            old = previous.bounds.copy()
            old[0] -= forward_delta_m
            old_size = np.maximum(old[:, 1] - old[:, 0], 0.08)
            ratio = size / old_size
            if np.any(ratio < 0.4) or np.any(ratio > 2.5):
                continue
            # Centre agreement prevents arbitrary frame-global confirmation.
            delta = np.abs(center - old.mean(axis=1))
            gate = np.array([0.50, 0.20, 0.30])
            if np.any(delta > gate):
                continue
            # The boxes must also overlap after small sparse-sampling tolerance.
            padding = np.array([0.15, 0.08, 0.12])
            if np.any(np.minimum(bounds[:, 1], old[:, 1]) + padding < np.maximum(bounds[:, 0], old[:, 0])):
                continue
            distance = float(np.sum(delta / gate))
            if distance < best_distance:
                best, best_distance = index, distance
        return best

    def apply(self, candidates: Iterable[Any], source_xyz: np.ndarray, *, stamp: float,
              observed_until_m: float, path_reliable: bool, policy: str = "score",
              min_score: float = 0.90, min_hits: int = 3, half_width_m: float = 1.05,
              forward_delta_m: float = 0.0, support_confirmation: bool = False,
              path: Any = None, max_path_sigma: float = 0.3) -> dict[str, Any]:
        if policy not in {"off", "score", "assist"}:
            raise ValueError("component policy must be off, score or assist")
        if not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("component minimum score must be within [0, 1]")
        if type(min_hits) is not int or min_hits < 2:
            raise ValueError("component assist requires at least two consecutive scans")
        if not math.isfinite(stamp) or not math.isfinite(forward_delta_m):
            raise ValueError("component timestamp and motion delta must be finite")
        if not math.isfinite(observed_until_m) or observed_until_m < 0:
            raise ValueError("observed horizon must be finite and nonnegative")
        if not math.isfinite(half_width_m) or half_width_m <= 0.10:
            raise ValueError("component corridor half-width must exceed margin")
        threshold = max(min_score, self.model.threshold)
        if not math.isfinite(max_path_sigma) or max_path_sigma <= 0:
            raise ValueError("component maximum path sigma must be positive and finite")
        requested_policy = policy
        if policy == "assist" and not self.model.assist_available:
            policy = "score"
        result: dict[str, Any] = {
            "name": self.model.name, "source_revision": self.model.source_revision,
            "weight_sha256": self.model.weight_sha256,
            "score_kind": "uncalibrated_assist_score", "policy": policy,
            "requested_policy": requested_policy, "fallback_reason": self.model.fallback_reason,
            "model_threshold": self.model.threshold, "threshold": threshold,
            "min_hits": min_hits, "evaluated": 0, "eligible": 0,
            "promoted": 0, "supported": 0, "unavailable": 0, "candidates": [],
        }
        if policy == "off":
            self.reset()
            return result
        if self._last_stamp is not None and (stamp <= self._last_stamp or stamp - self._last_stamp > 0.5):
            self.reset()
        self._last_stamp = stamp
        used: set[int] = set()
        current: list[_History] = []
        for candidate in candidates:
            # Support belongs to this scan, not a previously reused candidate.
            if "gabarit_component_supported" in candidate.reasons:
                candidate.reasons.remove("gabarit_component_supported")
            candidate.component_model_score = None
            item: dict[str, Any] = {
                "id": int(candidate.id), "promoted": False, "hits": 0,
                "detector_rejection": candidate.rejected,
                "box": {"s": [float(candidate.s_near), float(candidate.s_far)],
                        "d": [float(candidate.d_min), float(candidate.d_max)],
                        "h": [float(candidate.h_min), float(candidate.h_max)]},
                "rings": int(candidate.n_rings),
            }
            try:
                prediction = self.model.score_candidate(candidate, source_xyz)
            except (ValueError, TypeError, OverflowError) as exc:
                item.update(reason="unavailable_features", error=str(exc))
                result["unavailable"] += 1
                result["candidates"].append(item)
                continue
            candidate.component_model_score = prediction.score
            result["evaluated"] += 1
            item.update(score=prediction.score, features=asdict(prediction.features))
            reason = self._geometry_reason(
                candidate, prediction, observed_until_m=observed_until_m,
                path_reliable=path_reliable, half_width_m=half_width_m,
                support_accepted=(support_confirmation and self.model.name == "gabarit_trained_component_score"
                                  and self.model.assist_available),
            )
            accepted_support = candidate.rejected is None and reason == "eligible"
            if accepted_support:
                if path is None:
                    reason = "missing_path_uncertainty"
                else:
                    sigma = float(np.interp(.5 * (candidate.s_near + candidate.s_far), path.s, path.sig_c))
                    if not math.isfinite(sigma) or sigma > max_path_sigma:
                        reason = "uncertain_path"
            if reason == "eligible" and prediction.score < threshold:
                reason = "below_score_threshold"
            item["reason"] = reason
            if reason == "eligible":
                result["eligible"] += 1
                bounds = _candidate_bounds(candidate)
                previous_index = self._match(bounds, used, forward_delta_m)
                hits = 1
                if previous_index is not None:
                    used.add(previous_index)
                    hits = self._previous[previous_index].hits + 1
                item["hits"] = hits
                current.append(_History(bounds, hits))
                if policy == "assist" and hits >= min_hits:
                    if accepted_support:
                        candidate.reasons.append("gabarit_component_supported")
                        item["supported"] = True
                        result["supported"] += 1
                    else:
                        candidate.rejected = None
                        tag = ("gabarit_component_assist" if self.model.name == "gabarit_trained_component_score"
                               else "sakhalin_ai_component_assist")
                        if tag not in candidate.reasons:
                            candidate.reasons.append(tag)
                        item["promoted"] = True
                        result["promoted"] += 1
            result["candidates"].append(item)
        # A missed/ineligible scan breaks confirmation. A new component cannot
        # borrow another component's score or accumulated frame count.
        self._previous = current
        return result

    def evaluate(self, candidates: Iterable[Any], source_xyz: np.ndarray, *, mode: str = "shadow",
                 **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias: shadow is the score-only policy."""
        return self.apply(candidates, source_xyz, policy="score" if mode == "shadow" else mode, **kwargs)
