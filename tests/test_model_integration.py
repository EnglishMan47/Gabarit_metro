"""Exercise optional models through the public raw-cloud pipeline without Torch."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gabarit.config import Config
from gabarit.cloud import from_raw
import gabarit.learned_path as learned
from gabarit.learned_path import LearnedPathObservation
from gabarit.pipeline import Pipeline, _merge_tracks
from gabarit.detect import Candidate
from gabarit.tracker import Track
from tests.sim import tunnel_scan


class FakeAdvisor:
    """A deterministic external prediction; deliberately independent of path.c."""

    def __init__(self, center=-0.2, sigma=0.1):
        self.center, self.sigma = center, sigma
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def status(self):
        return {"state": "ready", "backend": "test"}

    def observe(self, leveled, intensity, track):
        self.calls.append((leveled.copy(), intensity.copy(), track.z.copy()))
        s = track.s.copy()
        return LearnedPathObservation(
            s, np.full_like(s, self.center), track.z.copy(),
            np.full_like(s, self.sigma), np.full_like(s, 0.05),
            np.ones(s.shape, dtype=bool), float(s[-1]), track.observed_until, 0.)


def _config(**model_values):
    return Config().with_overrides({"models": {
        "component_policy": "off", "learned_path": True,
        "path_interval": 1, **model_values,
    }})


def _install(monkeypatch, advisor, **model_values):
    monkeypatch.setattr(learned, "FarBevAdvisor", lambda *args, **kwargs: advisor)
    return Pipeline(_config(**model_values))


def _baseline(cfg):
    return Pipeline(replace(cfg, models=replace(cfg.models, learned_path=False)))


def test_unavailable_neural_backend_preserves_pipeline_results():
    missing = Path(__file__).with_name("__missing_optional_model_test__.pt").resolve()
    assert not missing.exists()
    pipe = Pipeline(_config(far_path_weights=str(missing)))
    baseline = _baseline(pipe.cfg)
    raw = tunnel_scan(boxes=[(0., 40., 0.6, 0.5, -0.1, 1.2)])
    for frame in range(6):
        actual = pipe.process_raw(raw, frame * 0.1)
        expected = baseline.process_raw(raw, frame * 0.1)
        assert actual.data["status"] == expected.data["status"]
        assert actual.data["obstacles"] == expected.data["obstacles"]
        assert actual.data["free_distance_m"] == expected.data["free_distance_m"]
        np.testing.assert_array_equal(actual.labels, expected.labels)
        np.testing.assert_array_equal(actual.track.c, expected.track.c)
        np.testing.assert_array_equal(actual.track.z, expected.track.z)
    assert actual.data["obstacle"]
    assert actual.data["models"]["far_path"]["reason"] == "checkpoint_missing"


def test_direct_cloud_entrypoint_resets_sensor_and_model_state(monkeypatch):
    advisor = FakeAdvisor()
    pipe = _install(monkeypatch, advisor)
    cloud = from_raw(tunnel_scan(), "-y", 1.)
    pipe.process_cloud(cloud, 0.)
    pipe.process_cloud(cloud, .1)
    result = pipe.process_cloud(replace(cloud, frame_id="new_sensor"), .2)
    assert result.data["state_reset"] == "frame_id_change"
    assert advisor.resets == 1
    assert result.data["models"]["far_path"]["fusion"]["state"] == "warming_up"
    assert result.data["models"]["far_path"]["inferred_this_frame"] is True
    assert result.data["models"]["far_path"]["last_inference_stamp"] == .2


@pytest.mark.parametrize("center,sigma", [(1.5, 0.1), (-0.2, 1.)])
def test_untrusted_predictions_cannot_change_free_distance_or_geometry(monkeypatch, center, sigma):
    advisor = FakeAdvisor(center, sigma)
    pipe = _install(monkeypatch, advisor)
    baseline = _baseline(pipe.cfg)
    raw = tunnel_scan()
    for frame in range(5):
        actual = pipe.process_raw(raw, frame * 0.1)
        expected = baseline.process_raw(raw, frame * 0.1)
        assert actual.data["free_distance_m"] == expected.data["free_distance_m"]
        np.testing.assert_array_equal(actual.track.c, expected.track.c)
        np.testing.assert_array_equal(actual.track.lat_obs, expected.track.lat_obs)
    assert actual.data["models"]["far_path"]["fusion"]["fused_stations"] == 0


@pytest.mark.parametrize("trigger", ["rewind", "duplicate", "gap", "frame_id"])
def test_replay_discontinuity_resets_model_confirmation(monkeypatch, trigger):
    advisor = FakeAdvisor()
    pipe = _install(monkeypatch, advisor)
    raw = tunnel_scan()
    pipe.process_raw(raw, 0.)
    fused = pipe.process_raw(raw, 0.1)
    assert fused.data["models"]["far_path"]["fusion"]["state"] == "fused"
    if trigger == "frame_id":
        raw = replace(raw, frame_id="another_sensor")
    stamp = {"rewind": 0., "duplicate": 0.1, "gap": 2., "frame_id": 0.2}[trigger]
    result = pipe.process_raw(raw, stamp)
    assert advisor.resets == 1
    assert result.data["state_reset"] == {
        "rewind": "time_rewind", "duplicate": "duplicate_stamp",
        "gap": "time_gap", "frame_id": "frame_id_change",
    }[trigger]
    assert result.data["models"]["far_path"]["fusion"]["fused_stations"] == 0
    assert result.data["models"]["far_path"]["fusion"]["state"] == "warming_up"
    # The advisor must receive levelled coordinates and a point-aligned
    # intensity array, not original -y-forward coordinates or raw z=-height.
    xyz, intensity, z = advisor.calls[-1]
    assert intensity.shape == (len(xyz),)
    assert np.median(xyz[:, 1]) > 0.
    assert np.min(xyz[:, 2]) > -0.4
    assert np.max(np.abs(z)) < 0.1


def test_neural_refinement_cannot_clear_geometric_obstacle(monkeypatch):
    # This is within the permitted lateral correction, low-sigma, supported,
    # temporally stable prediction. It must not steer around a real obstacle.
    pipe = _install(monkeypatch, FakeAdvisor(center=-0.55))
    baseline = _baseline(pipe.cfg)
    raw = tunnel_scan(boxes=[(0.8, 80., 0.4, 0.8, 0.2, 1.6)])
    for frame in range(30):
        expected = baseline.process_raw(raw, frame * 0.1)
        actual = pipe.process_raw(raw, frame * 0.1)
        # An optional path cannot feed back into calibration/path estimation,
        # candidate extraction or the primary tracker's temporal history.
        np.testing.assert_array_equal(actual.track.c, expected.track.c)
        np.testing.assert_array_equal(actual.track.z, expected.track.z)
        np.testing.assert_array_equal(actual.geo.corridor, expected.geo.corridor)
        assert [(t.id, t.history, t.hits, t.misses, t.confirmed) for t in pipe.tracker.tracks] == [
            (t.id, t.history, t.hits, t.misses, t.confirmed) for t in baseline.tracker.tracks]
        assert [o for o in actual.data["obstacles"] if o["track_id"] > 0] == expected.data["obstacles"]
    assert expected.data["obstacle"] and expected.data["nearest_distance_m"] == pytest.approx(80., abs=0.5)
    assert actual.data["models"]["far_path"]["fusion"]["fused_stations"] > 0
    assert actual.data["obstacle"], "Neural path refinement removed a geometric obstacle"
    assert actual.data["nearest_distance_m"] == pytest.approx(80., abs=0.5)
    assert actual.data["models"]["far_path"]["refined_path"] is not None


def test_learned_branch_can_add_an_independently_confirmed_obstacle(monkeypatch):
    pipe = _install(monkeypatch, FakeAdvisor(center=-0.55))
    baseline = _baseline(pipe.cfg)
    # Sparse edge returns are rejected by the original path. The independently
    # inferred path provides enough body evidence for normal temporal tracking.
    raw = tunnel_scan(boxes=[(-1., 80., 0.4, 0.8, 0.2, 1.6)])
    for frame in range(14):
        expected = baseline.process_raw(raw, frame * 0.1)
        actual = pipe.process_raw(raw, frame * 0.1)
        np.testing.assert_array_equal(actual.track.c, expected.track.c)
    assert expected.data["status"] == "CLEAR"
    assert actual.data["status"] == "OBSTACLE"
    assert actual.data["observed_until_m"] == expected.data["observed_until_m"]
    assert len(actual.data["obstacles"]) == 1
    extra = actual.data["obstacles"][0]
    assert extra["track_id"] < 0
    assert any(t.confirmed and t.id == extra["track_id"] for t in pipe.output_tracks)
    assert "attention_please_supplemental_path" in extra["reasons"]
    assert extra["distance_m"] == pytest.approx(80., abs=0.5)
    # Published bounds refer to real current returns in the displayed baseline
    # frame; they must not remain in the neural path's different lateral frame.
    tracked = pipe.learned_tracker.tracks[0]
    indices = tracked.cand.point_index
    assert extra["box"]["d"][0] == pytest.approx(actual.geo.d[indices].min(), abs=0.01)
    assert extra["box"]["d"][1] == pytest.approx(actual.geo.d[indices].max(), abs=0.01)


def test_same_body_is_not_reported_twice_by_independent_trackers(monkeypatch):
    pipe = _install(monkeypatch, FakeAdvisor(center=-0.3))
    raw = tunnel_scan(boxes=[(0., 80., 0.6, 0.8, 0.2, 1.6)])
    for frame in range(14):
        actual = pipe.process_raw(raw, frame * 0.1)
    assert any(t.confirmed for t in pipe.tracker.tracks)
    assert any(t.confirmed for t in pipe.learned_tracker.tracks)
    assert len(actual.data["obstacles"]) == 1
    assert actual.data["obstacles"][0]["track_id"] > 0
    assert not actual.data["pending"]


def test_held_neural_crop_does_not_duplicate_the_fresh_geometric_body():
    # Real doubleT frame 67: the two trackers renew different sides of one
    # person. The old neural crop must not introduce a one-frame extra event.
    body = Candidate(2, 56.36, 56.69, 1.01, 1.43, .03, 1.42,
                     56.38, 81, 12, 1.21, 1.)
    crop = replace(body, s_near=56.39, s_far=56.76, d_min=.44, d_max=1.09,
                   h_min=.16, h_max=1.4, lateral=.84)
    primary = Track(2, 56.36, 1.21, body, confirmed=True)
    held = Track(-2, 56.39, .84, crop, confirmed=True, misses=1)
    assert _merge_tracks([primary], [held]) == [primary]
    # A separate range must retain its independently confirmed alarm.
    separate = replace(held, s=70., cand=replace(crop, s_near=70., s_far=70.4))
    assert _merge_tracks([primary], [separate]) == [primary, separate]


def test_cached_refinement_is_used_between_inferences_but_needs_current_returns(monkeypatch):
    advisor = FakeAdvisor(center=-0.3)
    pipe = _install(monkeypatch, advisor, path_interval=3)
    raw = tunnel_scan()
    for frame in range(4):
        actual = pipe.process_raw(raw, frame * 0.1, frame_index=100 + frame * 10)
    assert len(advisor.calls) == 2
    assert actual.data["models"]["far_path"]["refined_path"] is not None
    actual = pipe.process_raw(raw, 0.4, frame_index=140)
    assert len(advisor.calls) == 2
    assert actual.data["models"]["far_path"]["refined_path"] is not None
    assert actual.data["models"]["far_path"]["fusion"]["cache_age_frames"] == 1
    empty = replace(raw, width=0, points=raw.points[:0])
    actual = pipe.process_raw(empty, 0.5, frame_index=150)
    assert len(advisor.calls) == 2
    assert actual.data["models"]["far_path"]["refined_path"] is None
    assert actual.data["models"]["far_path"]["fusion"]["active_stations"] == 0
    pipe.reset()
    assert pipe.output_tracks == []
    assert pipe.learned_tracker.tracks == []


def test_score_only_component_model_preserves_existing_acceptance():
    cfg = Config()
    scored = Pipeline(cfg)
    disabled = Pipeline(replace(cfg, models=replace(cfg.models, component_policy="off")))
    raw = tunnel_scan(boxes=[(0., 40., 0.6, 0.5, -0.1, 1.2)])
    for frame in range(6):
        actual = scored.process_raw(raw, frame * 0.1)
        expected = disabled.process_raw(raw, frame * 0.1)
        assert actual.data["status"] == expected.data["status"]
        assert actual.data["nearest_distance_m"] == expected.data["nearest_distance_m"]
        np.testing.assert_array_equal(actual.labels, expected.labels)
        assert actual.data["models"]["components"]["promoted"] == 0
    assert actual.data["obstacle"]
    assert any(c["component_model_score"] is not None for c in actual.data["candidates"])
