from types import SimpleNamespace

import numpy as np
import pytest

from gabarit.config import Config, ModelsConfig
from gabarit.path_fusion import FarPathFusion
from gabarit.track import TrackModel


def scene(observed=140.0):
    s = np.arange(3.5, 200.0, 2.0)
    zero = np.zeros(len(s))
    rails = s < 30
    track = TrackModel(s, zero.copy(), zero.copy(), zero.copy(),
                       np.full(len(s), .1), np.full(len(s), .04),
                       rails, s <= observed, s <= observed, zero.copy(),
                       np.full(len(s), 2.4), np.full(len(s), -2.4),
                       np.zeros((0, 5)), observed, 29.5,
                       current_observed_until=observed)
    points = np.array([(x, f, z) for f in s for x, z in ((-2.4, 1.), (2.4, 1.), (0., .3))])
    return track, points


def observation(c=.2, z=.04):
    s = np.arange(11.0, 201.0, 2.0)
    return SimpleNamespace(s=s, c=np.full(len(s), c), z=np.full(len(s), z),
                           sig_c=np.full(len(s), .1), sig_z=np.full(len(s), .04),
                           valid=np.ones(len(s), bool))


def test_only_consistent_supported_far_path_is_refined_without_claiming_more_visibility():
    fusion = FarPathFusion(ModelsConfig())
    track, points = scene()
    before_sc, before_sz = track.sig_c.copy(), track.sig_z.copy()
    assert fusion.apply(observation(), track, points)["state"] == "warming_up"
    assert not track.c.any()
    report = fusion.apply(observation(), track, points)
    assert report["state"] == "fused" and report["fused_stations"] > 20
    assert np.all(track.c[track.rail] == 0)
    assert np.all(track.c[track.s > track.observed_until] == 0)
    assert np.max(track.c) == pytest.approx(.05)
    assert track.observed_until == track.current_observed_until == 140.0
    assert np.array_equal(track.sig_c, before_sc) and np.array_equal(track.sig_z, before_sz)


@pytest.mark.parametrize("change", ["disagreement", "uncertainty", "no_returns", "invalid", "gap"])
def test_bad_model_evidence_never_moves_the_path(change):
    fusion = FarPathFusion(ModelsConfig())
    track, points = scene()
    fusion.apply(observation(), track, points)
    out = observation()
    if change == "disagreement":
        out.c[:] = -.5
    elif change == "uncertainty":
        out.sig_c[:] = 2.
    elif change == "no_returns":
        points = points[:0]
    elif change == "invalid":
        out.s[3] = np.nan
    elif change == "gap":
        out.valid[::2] = False
    report = fusion.apply(out, track, points)
    assert report["fused_stations"] == 0
    assert not track.c.any() and not track.z.any()


def test_distance_advance_and_reset_do_not_reuse_unrelated_confirmation():
    fusion = FarPathFusion(ModelsConfig())
    track, points = scene()
    fusion.apply(observation(), track, points)
    fusion.advance(1.0)
    assert fusion.apply(observation(), track, points)["state"] == "fused"
    fusion.reset()
    track, points = scene()
    assert fusion.apply(observation(), track, points)["state"] == "warming_up"


def test_cached_correction_is_not_transported_through_a_changed_reference_pose():
    fusion = FarPathFusion(ModelsConfig())
    for _ in range(2):
        track, points = scene()
        fusion.apply(observation(), track, points)
    track, points = scene()
    assert fusion.refine(track, points) is not None
    track.c += .25
    assert fusion.refine(track, points) is None
    track, points = scene()
    track.z += .1
    assert fusion.refine(track, points) is None


@pytest.mark.parametrize("name,value", [("path_interval", 0), ("path_interval", 2.5),
                                       ("component_min_hits", True), ("component_min_hits", 1),
                                       ("path_confirmations", float("nan")),
                                       ("path_blend", 0), ("path_max_sigma", float("nan")),
                                       ("component_policy", "reject")])
def test_model_configuration_rejects_invalid_runtime_settings(name, value):
    with pytest.raises(ValueError):
        Config().with_overrides({"models": {name: value}})
