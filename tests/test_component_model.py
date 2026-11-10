"""Upstream C++ parity and positive-only component policy regression tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabarit.component_model import (  # noqa: E402
    ComponentFeatures, ComponentModel, ComponentModelAssistant, FEATURE_NAMES,
    SOURCE_REVISION,
)
from gabarit.detect import Candidate  # noqa: E402

VENDOR = Path(__file__).resolve().parents[1] / "gabarit" / "vendor" / "sakhalin_ai"


def test_vendored_artifacts_are_unmodified_and_have_mit_notice():
    provenance = json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["revision"] == SOURCE_REVISION
    for name, source in provenance["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == source["sha256"]
    notice = (VENDOR / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 saic" in notice
    assert "MIT License" in notice and "Permission is hereby granted" in notice


def test_features_use_sensor_origin_points_and_upstream_float32_contract():
    points = np.array([[0.2, 20.0, -1.2], [-0.2, 21.0, 0.8], [0.1, 20.5, 0.2]], dtype=np.float64)
    features = ComponentFeatures.from_points(points)
    xyz = points.astype(np.float32).astype(np.float64)
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    assert features.point_count == 3
    assert features.vector()[1:4] == pytest.approx(extents, abs=1e-14)
    assert features.volume_m3 == pytest.approx(np.prod(extents), abs=1e-14)
    assert features.density_points_per_m3 == pytest.approx(3 / np.prod(extents))
    assert features.nearest_distance_m == pytest.approx(np.linalg.norm(xyz, axis=1).min())
    assert features.centroid_distance_m == pytest.approx(np.linalg.norm(xyz.mean(axis=0)))
    flipped = ComponentFeatures.from_points(points * [-1, -1, 1])
    assert flipped.vector() == pytest.approx(features.vector())
    # Rail-height removal is intentionally not applied: it changes origin distance.
    translated = ComponentFeatures.from_points(points + [0, 0, 1.4])
    assert translated.nearest_distance_m != features.nearest_distance_m


def test_planar_component_uses_original_volume_floor():
    features = ComponentFeatures.from_points(np.array([[0, 3, 1], [0, 4, 2]]))
    assert features.volume_m3 == 1e-6
    assert features.density_points_per_m3 == 2e6


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.ones((3, 2)), [[0, np.nan, 1]], [[0, np.inf, 1]]])
def test_invalid_source_features_are_not_invented(points):
    with pytest.raises(ValueError):
        ComponentFeatures.from_points(points)


@pytest.fixture(scope="module")
def cpp_reference(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for additional original-implementation parity check")
    temp = tmp_path_factory.mktemp("sakhalin_cpp_reference")
    source = temp / "reference.cpp"
    source.write_text(
        '#include <array>\n#include <iostream>\n#include <iomanip>\n'
        f'#include "{(VENDOR / "baseline_v3_assist_model.inc").as_posix()}"\n'
        'int main() { std::array<double, 8> f; std::cout << std::setprecision(17);\n'
        'while (std::cin >> f[0] >> f[1] >> f[2] >> f[3] >> f[4] >> f[5] >> f[6] >> f[7])\n'
        '{ std::cout << BaselineV3AssistScoreScore(f) << "\\n"; } return 0; }\n',
        encoding="utf-8",
    )
    executable = temp / ("reference.exe" if os.name == "nt" else "reference")
    command = [compiler, "-std=c++17", "-O0", str(source), "-o", str(executable)]
    if os.name == "nt" and "g++" in Path(compiler).name:
        command.append("-static")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return executable


def test_exact_score_parity_with_unmodified_upstream_cpp(cpp_reference):
    model = ComponentModel()
    assert len(model.trees) == 17
    assert model.threshold == 0.8064516129032258
    rng = np.random.default_rng(20261107)
    vectors = np.column_stack([
        rng.integers(1, 1000, 500), rng.uniform(0, 2, 500), rng.uniform(0, 3, 500),
        rng.uniform(0, 3, 500), rng.uniform(0, 5, 500), rng.uniform(0, 2000, 500),
        rng.uniform(0, 200, 500), rng.uniform(0, 200, 500),
    ]).tolist()

    def add_boundaries(node):
        if "feature_index" not in node:
            return
        for value in (np.nextafter(node["threshold"], -np.inf), node["threshold"], np.nextafter(node["threshold"], np.inf)):
            vector = [100, 0.8, 0.4, 1.0, 0.32, 312.5, 56, 55.8]
            vector[node["feature_index"]] = float(value)
            vectors.append(vector)
        add_boundaries(node["left"])
        add_boundaries(node["right"])

    for tree in model.trees:
        add_boundaries(tree)
    encoded = "\n".join(" ".join(format(value, ".17g") for value in vector) for vector in vectors) + "\n"
    result = subprocess.run([str(cpp_reference)], input=encoded, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    expected = np.array([float(line) for line in result.stdout.splitlines()])
    actual = np.array([model.predict(ComponentFeatures(*vector)).score for vector in vectors])
    assert len(expected) == len(vectors)
    assert actual == pytest.approx(expected, abs=1e-15)


def _positive_component(s=56.0, candidate_id=1, rejected="low_score"):
    # Deterministic real point cloud producing a high score with original weights.
    points = np.random.default_rng(7).uniform(-0.5, 0.5, size=(600, 3))[:100]
    points = (points * [0.8, 0.4, 1.0] + [0, s, 0]).astype(np.float32)
    candidate = Candidate(
        id=candidate_id, s_near=float(points[:, 1].min()), s_far=float(points[:, 1].max()),
        d_min=float(points[:, 0].min()), d_max=float(points[:, 0].max()),
        h_min=float(points[:, 2].min() + 1.0), h_max=float(points[:, 2].max() + 1.0),
        distance=float(np.linalg.norm(points, axis=1).min()), n_points=100, n_rings=4,
        lateral=float(np.median(points[:, 0])), score=0.1, rejected=rejected,
        point_index=np.arange(len(points)),
    )
    return candidate, points


def _apply(assistant, candidate, points, stamp, **kwargs):
    return assistant.apply(
        [candidate], points, stamp=stamp, observed_until_m=75.0,
        path_reliable=True, **kwargs,
    )


def test_candidate_features_use_actual_indices_and_no_bbox_fallback():
    candidate, points = _positive_component()
    model = ComponentModel()
    extra = np.vstack([points, [[400, 400, 400]]])
    prediction = model.score_candidate(candidate, extra)
    assert prediction.features.point_count == 100
    assert prediction.score == pytest.approx(0.9671763080111482, abs=1e-15)
    for indices in (None, np.array([len(extra)]), np.array([-1]), np.array([0.0, 1.0]), np.array([0, 0])):
        candidate.point_index = indices
        with pytest.raises(ValueError):
            model.score_candidate(candidate, extra)


def test_score_policy_has_no_effect_on_acceptance_even_after_many_hits():
    assistant = ComponentModelAssistant()
    for frame in range(6):
        candidate, points = _positive_component()
        result = _apply(assistant, candidate, points, frame * 0.1, policy="score")
        assert candidate.rejected == "low_score"
        assert candidate.reasons == []
        assert candidate.component_model_score > 0.90
        assert result["promoted"] == 0
    assert result["candidates"][0]["hits"] == 6


def test_assist_requires_consecutive_local_confirmation_and_keeps_tracker_separate():
    assistant = ComponentModelAssistant()
    for frame in range(3):
        candidate, points = _positive_component()
        result = _apply(assistant, candidate, points, frame * 0.1, policy="assist", min_hits=3)
        assert result["candidates"][0]["hits"] == frame + 1
        assert result["promoted"] == (1 if frame == 2 else 0)
        assert candidate.rejected == (None if frame == 2 else "low_score")
        # The detector's original scalar score is never overwritten.
        assert candidate.score == 0.1
    assert candidate.reasons == ["sakhalin_ai_component_assist"]
    assert not hasattr(candidate, "confirmed")


@pytest.mark.parametrize("rejected", [None, "ground", "wall_attached", "floating", "far_weak", "pole_far", "edge_far", "isolated_returns"])
def test_accepted_and_non_score_rejections_are_never_overridden(rejected):
    assistant = ComponentModelAssistant()
    for frame in range(5):
        candidate, points = _positive_component(rejected=rejected)
        result = _apply(assistant, candidate, points, frame * 0.1, policy="assist")
        assert candidate.rejected == rejected
        assert result["promoted"] == 0


@pytest.mark.parametrize("flag", ["low", "transparent", "single_beam"])
def test_small_or_special_geometry_paths_keep_their_own_decisions(flag):
    assistant = ComponentModelAssistant()
    for frame in range(4):
        candidate, points = _positive_component()
        setattr(candidate, flag, True)
        result = _apply(assistant, candidate, points, frame * 0.1, policy="assist")
        assert candidate.rejected == "low_score"
        assert result["candidates"][0]["reason"] == "special_geometry_branch"
    # A zero model score also cannot veto an accepted, sparse low object.
    candidate.rejected = None
    candidate.point_index = np.array([0, 1])
    _apply(assistant, candidate, points, 0.5, policy="assist")
    assert candidate.rejected is None


@pytest.mark.parametrize("change,reason", [
    ({"d_min": -1.02}, "boundary_component"),
    ({"h_min": 0.05}, "ground_or_high_component"),
    ({"h_max": 2.5}, "ground_or_high_component"),
    ({"n_rings": 1}, "insufficient_multibeam_support"),
    ({"s_far": 76.0}, "outside_observed_path"),
])
def test_positive_score_cannot_bypass_geometry(change, reason):
    assistant = ComponentModelAssistant()
    for frame in range(4):
        candidate, points = _positive_component()
        for name, value in change.items():
            setattr(candidate, name, value)
        result = _apply(assistant, candidate, points, frame * 0.1, policy="assist")
        assert result["candidates"][0]["reason"] == reason
        assert candidate.rejected == "low_score"


def test_history_resets_on_missed_scan_seek_gap_and_component_switch():
    assistant = ComponentModelAssistant()
    candidate, points = _positive_component()
    _apply(assistant, candidate, points, 0.0, policy="assist")
    result = _apply(assistant, candidate, points, 0.1, policy="assist")
    assert result["candidates"][0]["hits"] == 2
    # A different lateral component cannot inherit the previous score history.
    shifted, shifted_points = _positive_component(candidate_id=2)
    shifted.d_min += 0.45
    shifted.d_max += 0.45
    result = _apply(assistant, shifted, shifted_points, 0.2, policy="assist")
    assert result["candidates"][0]["hits"] == 1
    result = _apply(assistant, shifted, shifted_points, 0.1, policy="assist")
    assert result["candidates"][0]["hits"] == 1
    result = _apply(assistant, shifted, shifted_points, 1.0, policy="assist")
    assert result["candidates"][0]["hits"] == 1
    assistant.apply([], points, stamp=1.1, observed_until_m=75.0, path_reliable=True, policy="assist")
    result = _apply(assistant, shifted, shifted_points, 1.2, policy="assist")
    assert result["candidates"][0]["hits"] == 1
    assistant.reset()
    result = _apply(assistant, shifted, shifted_points, 1.3, policy="assist")
    assert result["candidates"][0]["hits"] == 1


def test_motion_compensation_and_one_to_one_history_assignment():
    assistant = ComponentModelAssistant()
    candidate, points = _positive_component()
    _apply(assistant, candidate, points, 0.0, policy="assist")
    moved, moved_points = _positive_component(s=55.4)
    result = _apply(assistant, moved, moved_points, 0.1, policy="assist", forward_delta_m=0.6)
    assert moved.component_model_score >= 0.90
    assert result["candidates"][0]["hits"] == 2
    no_motion_compensation = ComponentModelAssistant()
    candidate, points = _positive_component()
    _apply(no_motion_compensation, candidate, points, 0.0, policy="assist")
    moved, moved_points = _positive_component(s=55.4)
    result = _apply(no_motion_compensation, moved, moved_points, 0.1, policy="assist")
    assert result["candidates"][0]["hits"] == 1
    assistant.reset()
    candidate, points = _positive_component()
    _apply(assistant, candidate, points, 0.0, policy="assist")
    twin, _ = _positive_component(candidate_id=2)
    result = assistant.apply([candidate, twin], points, stamp=0.1, observed_until_m=75.0, path_reliable=True, policy="assist")
    assert sorted(item["hits"] for item in result["candidates"]) == [1, 2]


def test_unavailable_feature_diagnostics_and_off_reset():
    assistant = ComponentModelAssistant()
    candidate, points = _positive_component()
    candidate.point_index = None
    result = _apply(assistant, candidate, points, 0.0, policy="assist")
    assert result["unavailable"] == 1
    assert result["candidates"][0]["reason"] == "unavailable_features"
    assert candidate.component_model_score is None
    assert candidate.rejected == "low_score"
    result = _apply(assistant, candidate, points, 0.1, policy="off")
    assert result["evaluated"] == result["promoted"] == 0


def test_no_assist_without_observed_reliable_path():
    assistant = ComponentModelAssistant()
    for frame in range(4):
        candidate, points = _positive_component()
        result = assistant.apply([candidate], points, stamp=frame * 0.1, observed_until_m=75.0, path_reliable=False, policy="assist")
        assert candidate.rejected == "low_score"
        assert result["candidates"][0]["reason"] == "unreliable_path"


def test_public_model_feature_order_is_explicit():
    assert FEATURE_NAMES == (
        "point_count", "extent_x_m", "extent_y_m", "extent_z_m", "volume_m3",
        "density_points_per_m3", "centroid_distance_m", "nearest_distance_m",
    )
