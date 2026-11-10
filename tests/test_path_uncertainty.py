"""Temporal path mean and uncertainty must describe the same estimate."""
import numpy as np
import pytest

from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.egomotion import propagate_track
from gabarit.detect import detect, track_coordinates
from gabarit.tracker import Tracker
from gabarit.track import TrackModel


def path(c=0., z=0., sigma_c=.1, sigma_z=.05):
    s = np.array([10., 60., 100.])
    array = lambda x: np.full(len(s), x, dtype=float)
    flags = np.ones(len(s), dtype=bool)
    return TrackModel(s, array(c), array(0.), array(z), array(sigma_c), array(sigma_z),
                      flags.copy(), flags.copy(), flags.copy(), array(0.),
                      array(2.4), array(-2.4), np.empty((0, 5)), 100., 30.)


@pytest.mark.parametrize("speed,far_weight", [(0., .15), (5., .5)])
def test_disagreeing_path_mixture_has_the_moment_matched_variance(speed, far_weight):
    pipe = Pipeline(Config().with_overrides({"models": {"component_policy": "off"}}))
    pipe.ego.v = speed
    current, prior = path(1., .2, .1, .05), path(0., 0., .2, .1)
    before_prior = prior.c.copy(), prior.sig_c.copy()
    pipe._blend_path(current, prior)
    weight = 1. - np.clip((current.s - 40.) / 40., 0., 1.) * (1. - far_weight)
    assert current.c == pytest.approx(weight)
    assert current.z == pytest.approx(weight * .2)
    assert current.sig_c ** 2 == pytest.approx(weight * .1**2 + (1.-weight) * .2**2 + weight*(1.-weight))
    assert current.sig_z_path ** 2 == pytest.approx(weight * .05**2 + (1.-weight) * .1**2 + weight*(1.-weight)*.2**2)
    assert current.sig_z == pytest.approx(.05)
    assert current.sig_c[-1] > .3
    assert np.array_equal(prior.c, before_prior[0]) and np.array_equal(prior.sig_c, before_prior[1])
    assert pipe.track.prev is current


@pytest.mark.parametrize("bad_field,bad_value", [("c", np.nan), ("sig_c", np.nan),
                                                ("sig_c", np.inf), ("sig_c", -.1)])
def test_invalid_prior_cannot_poison_a_current_measurement(bad_field, bad_value):
    pipe = Pipeline()
    current, prior = path(1.), path(0.)
    getattr(prior, bad_field)[-1] = bad_value
    pipe._blend_path(current, prior)
    assert current.c[-1] == 1.
    assert current.sig_c[-1] == .1


def test_identical_estimates_do_not_invent_disagreement_uncertainty():
    pipe = Pipeline()
    current, prior = path(1., .2), path(1., .2)
    pipe._blend_path(current, prior)
    assert current.sig_c == pytest.approx(.1)
    assert current.sig_z == pytest.approx(.05)
    assert current.sig_z_path == pytest.approx(.05)


@pytest.mark.parametrize("delta", [0., 1.])
def test_propagation_transports_total_vertical_path_uncertainty_separately(delta):
    previous = path()
    previous.sig_z_path = np.array([.2, .3, .4])
    current = propagate_track(previous, delta)
    expected = (previous.sig_z_path + .009 if not delta else
                np.interp(previous.s, previous.s - delta, previous.sig_z_path) + .03)
    assert current.sig_z_path == pytest.approx(expected)
    assert current.sig_z == pytest.approx(.05 + (.009 if not delta else .03))
    assert previous.sig_z_path == pytest.approx([.2, .3, .4])


def test_prior_total_uncertainty_is_accumulated_without_widening_floor_noise():
    pipe = Pipeline()
    current, prior = path(), path()
    prior.sig_z_path = np.full(3, .5)
    pipe._blend_path(current, prior)
    assert current.sig_z_path[-1] > .4
    assert current.sig_z == pytest.approx(.05)
    assert current.public()["sig_z_path"][-1] > .4


def test_uncertain_path_does_not_crop_away_a_resolved_barrel_body():
    # Regression distilled from raw barrel seed20262040. Global vertical path
    # uncertainty used as floor noise cropped a .8m body below the .5m resolved
    # extent, so an otherwise observable confirmed target could not renew.
    cfg = Config()
    pipe = Pipeline(cfg)
    current, prior = path(sigma_c=.03, sigma_z=.016), path(sigma_c=.03, sigma_z=.016)
    prior.sig_z_path = np.full(3, .5)
    pipe._blend_path(current, prior)
    x, h = np.meshgrid(np.linspace(-.3, .3, 6), np.linspace(.15, .95, 20))
    lev = np.column_stack((x.ravel(), np.full(x.size, 50.), h.ravel())).astype(np.float32)
    ring = np.repeat(np.arange(20), 6)
    geo = track_coordinates(lev, current, cfg)
    candidates = detect(lev, ring, geo, current, cfg)
    body = next(c for c in candidates if c.rejected is None and not c.low)
    assert body.size[2] > .7 and Tracker._resolved(body)
    tracker = Tracker(cfg)
    for _ in range(3):
        tracker.update([body], 0., lev=lev, ring=ring, geo=geo)
    assert tracker.tracks[0].confirmed
    # Current raw returns must keep the object observable through a path
    # correction, even when it lies outside the new global envelope.
    shifted = path(c=1.4, sigma_c=.03, sigma_z=.016)
    shifted.sig_z_path = current.sig_z_path.copy()
    for _ in range(cfg.tracking.drop_misses + 2):
        tracks = tracker.update([], 0., path_prior=current, path_current=shifted,
                                lev=lev, ring=ring, geo=track_coordinates(lev, shifted, cfg))
        assert tracks and tracks[0].confirmed and tracks[0].misses == 0
        current = shifted
