"""Regressions for sparse returns, short dropouts and obstacle tracking."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from gabarit.cloud import from_raw
from gabarit.config import Config, FilterConfig
from gabarit.detect import Candidate, detect, detect_rail_objects, track_coordinates
from gabarit.filtering import support_mask
from gabarit.track import TrackEstimator, TrackModel, _rail_pair, _short_line, _window_max, _window_percentile
from gabarit.tracker import Tracker
from tests.sim import tunnel_scan


def _candidate(s=20.0, d=0.0, low=False):
    return Candidate(1, s, s + 0.5, d - 0.2, d + 0.2, 0.2, 1.7,
                     float(np.hypot(s, 1.0)), 20, 4, d, 0.9,
                     point_index=np.array([0, 1]), low=low)


def _straight_track():
    s = np.arange(3.5, 200.0, 2.0)
    zero = np.zeros(len(s))
    flags = np.ones(len(s), bool)
    return TrackModel(s, zero.copy(), zero.copy(), zero.copy(),
                      np.full(len(s), 0.05), np.full(len(s), 0.02),
                      flags.copy(), flags.copy(), flags.copy(), zero.copy(),
                      np.full(len(s), 2.4), np.full(len(s), -2.4),
                      np.zeros((0, 5)), 190.0, 60.0)


def test_tracker_reports_current_box_during_dropouts_without_mutating_old_candidate():
    cfg = Config().with_overrides({"tracking": {"confirm_hits": 1}})
    tracker = Tracker(cfg)
    old = _candidate()
    tracker.update([old], 0.0)
    track = tracker.update([], 1.5)[0]
    assert track.confirmed and track.misses == 1
    assert track.s == pytest.approx(18.5)
    assert track.cand.s_near == pytest.approx(track.s)
    assert track.cand.s_far == pytest.approx(19.0)
    assert track.cand.distance == pytest.approx(np.hypot(18.5, 1.0))
    assert track.cand.point_index is None
    assert old.s_near == 20.0 and old.point_index is not None
    for _ in range(cfg.tracking.drop_misses):
        tracker.update([], 0.0)
    assert not tracker.tracks


@pytest.mark.parametrize("s,low", [(100.0, False), (20.0, True)])
def test_new_track_obeys_its_own_far_or_low_confirmation_rule(s, low):
    cfg = Config().with_overrides({"tracking": {"confirm_hits": 1,
                                                "far_confirm_hits": 3, "low_confirm_hits": 3}})
    tracker = Tracker(cfg)
    assert not tracker.update([_candidate(s, low=low)], 0.0)[0].confirmed
    assert not tracker.update([_candidate(s, low=low)], 0.0)[0].confirmed
    assert tracker.update([_candidate(s, low=low)], 0.0)[0].confirmed


def test_tracker_estimates_person_motion_with_consistent_sign():
    tracker = Tracker(Config())
    for k in range(12):
        track = tracker.update([_candidate(20.0 + 0.15 * k)], 0.0)[0]
    assert track.confirmed and track.v_rel == pytest.approx(0.15, abs=0.01)
    predicted = tracker.update([], 0.0)[0]
    assert predicted.cand.s_near == pytest.approx(20.0 + 0.15 * 12, abs=0.01)


def test_dense_weak_measurements_can_continue_but_never_start_a_confirmed_track():
    tracker = Tracker(Config())
    weak = _candidate()
    weak.score, weak.rejected = .3, 'low_score'
    for _ in range(10):
        assert not tracker.update([weak],0.)
    for _ in range(3):
        tracker.update([_candidate()],0.)
    ident = tracker.tracks[0].id
    for _ in range(10):
        track = tracker.update([weak],0.)[0]
        assert track.id == ident and track.confirmed and track.misses == 0
        assert track.cand.rejected is None


def test_sparse_weak_returns_do_not_keep_an_old_track_alive():
    tracker = Tracker(Config())
    for _ in range(3):
        tracker.update([_candidate()],0.)
    weak = _candidate()
    weak.n_points, weak.n_rings, weak.score, weak.rejected = 4,2,.3,'low_score'
    for _ in range(Config().tracking.drop_misses+1):
        tracker.update([weak],0.)
    assert not tracker.tracks


def test_nearby_rail_fragment_does_not_inherit_a_persons_confirmed_identity():
    tracker = Tracker(Config())
    for _ in range(3):
        tracker.update([_candidate(s=4.)],0.)
    person_id = tracker.tracks[0].id
    rail = _candidate(s=4.,d=-.8,low=True)
    rail.d_min, rail.d_max, rail.h_min, rail.h_max = -.92,-.75,.05,.17
    tracks = tracker.update([rail],0.)
    person = next(t for t in tracks if t.id == person_id)
    assert person.confirmed and person.misses == 1
    assert all(not t.confirmed for t in tracks if t.id != person_id)


def test_overlapping_low_fragment_does_not_collapse_a_confirmed_upright_body():
    tracker = Tracker(Config())
    for _ in range(3):
        tracker.update([_candidate()],0.)
    ident = tracker.tracks[0].id
    fragment = _candidate(low=True)
    fragment.h_min,fragment.h_max = .05,.18
    tracks = tracker.update([fragment],0.)
    old = next(t for t in tracks if t.id==ident)
    assert old.confirmed and old.misses==1 and old.cand.h_max==1.7


def test_relaxed_low_body_needs_two_independent_strict_support_frames():
    tracker = Tracker(Config())
    for _ in range(6):
        c = _low_body()
        c.strict_rail_support = False
        assert not tracker.update([c],0.)[0].confirmed
    assert not tracker.update([_low_body()],0.)[0].confirmed
    assert tracker.update([_low_body()],0.)[0].confirmed


def test_strict_rail_support_must_coincide_with_shallow_body_measurements():
    tracker = Tracker(Config())
    loose = _low_body()
    loose.strict_rail_support = False
    steep = _low_body()
    steep.h_max = .4
    for c in (loose, steep, _low_body(), loose):
        assert not tracker.update([c],0.)[0].confirmed
    assert tracker.update([_low_body()],0.)[0].confirmed


def test_dense_broad_low_body_has_its_own_repeated_cross_section_proof():
    tracker = Tracker(Config())
    for k in range(3):
        c = _low_body()
        c.n_points = 20
        c.d_min,c.d_max = .65,.95
        c.strict_rail_support = False
        track = tracker.update([c],0.)[0]
        assert track.confirmed == (k==2)


def test_dense_narrow_low_echo_does_not_replace_missing_strict_support():
    tracker = Tracker(Config())
    for _ in range(20):
        c = _low_body()
        c.n_points = 30
        c.d_min,c.d_max = .7,.9
        c.strict_rail_support = False
        assert not tracker.update([c],0.)[0].confirmed


def _resolved_body_cloud(s=20.,height=1.7,rows=20,columns=8):
    d,h = np.meshgrid(np.linspace(-.18,.18,columns),np.linspace(.2,height,rows))
    lev = np.column_stack((d.ravel(),np.full(d.size,s),h.ravel())).astype(np.float32)
    return lev,np.repeat(np.arange(rows),columns)


def _resolved_candidate(lev,rings):
    c = _candidate(s=float(lev[:,1].min()))
    c.s_far = float(lev[:,1].max())
    c.d_min,c.d_max = float(lev[:,0].min()),float(lev[:,0].max())
    c.h_min,c.h_max = float(lev[:,2].min()),float(lev[:,2].max())
    c.n_points,c.n_rings = len(lev),len(np.unique(rings))
    c.point_index = np.arange(len(lev))
    return c


@pytest.mark.parametrize('height,rows,columns',[(.75,20,8),(1.7,20,8),(1.7,4,6)])
def test_local_returns_continue_confirmed_body_but_cannot_start_tracks(height,rows,columns):
    cfg = Config()
    tracker = Tracker(cfg)
    model = _straight_track()
    lev,rings = _resolved_body_cloud(height=height,rows=rows,columns=columns)
    geo = track_coordinates(lev,model,cfg)
    assert tracker.update([],0.,lev=lev,ring=rings,geo=geo) == []
    for _ in range(3):
        tracker.update([_resolved_candidate(lev,rings)],0.,lev=lev,ring=rings,geo=geo)
    ident = tracker.tracks[0].id
    for k in range(1,9):
        lev,rings = _resolved_body_cloud(20.-k,height=height,rows=rows,columns=columns)
        tracks = tracker.update([],1.,lev=lev,ring=rings,geo=track_coordinates(lev,model,cfg))
        assert len(tracks) == 1 and tracks[0].id == ident and tracks[0].confirmed
        assert tracks[0].misses == 0 and tracks[0].cand.point_index is not None
    empty = np.empty((0,3),np.float32)
    for _ in range(cfg.tracking.drop_misses+1):
        tracker.update([],0.,lev=empty,ring=np.empty(0,int),geo=track_coordinates(empty,model,cfg))
    assert not tracker.tracks


def test_dense_long_wall_does_not_renew_a_disappeared_person():
    cfg = Config()
    tracker = Tracker(cfg)
    model = _straight_track()
    lev,rings = _resolved_body_cloud()
    for _ in range(3):
        tracker.update([_resolved_candidate(lev,rings)],0.,lev=lev,ring=rings,geo=track_coordinates(lev,model,cfg))
    # The ROI of a long surface resembles the old body. Its continuation
    # before and after the ROI must prevent the local crop from renewing it.
    d,s,h = np.meshgrid(np.linspace(-.18,.18,8),np.arange(15.,25.,.02),np.linspace(.2,1.7,20))
    wall = np.column_stack((d.ravel(),s.ravel(),h.ravel())).astype(np.float32)
    beam = np.tile(np.arange(20),d.size//20)
    for _ in range(cfg.tracking.drop_misses+1):
        tracker.update([],0.,lev=wall,ring=beam,geo=track_coordinates(wall,model,cfg))
    assert not tracker.tracks


def test_local_crop_does_not_spawn_another_id_for_the_same_raw_body():
    cfg = Config()
    tracker = Tracker(cfg)
    model = _straight_track()
    lev,rings = _resolved_body_cloud()
    geo = track_coordinates(lev,model,cfg)
    for _ in range(3):
        tracker.update([_resolved_candidate(lev,rings)],0.,lev=lev,ring=rings,geo=geo)
    ident = tracker.tracks[0].id
    for _ in range(10):
        connected = _resolved_candidate(lev,rings)
        connected.s_near -= .4
        connected.s_far += .4
        tracks = tracker.update([connected],0.,lev=lev,ring=rings,geo=geo)
        assert len(tracks) == 1 and tracks[0].id == ident


def test_overlapping_boxes_with_disjoint_returns_keep_distinct_identities():
    tracker = Tracker(Config())
    for _ in range(3):
        left,right = _candidate(d=-.1),_candidate(d=.1)
        left.point_index,right.point_index = np.arange(10),np.arange(10,20)
        tracks = tracker.update([left,right],0.)
    assert len(tracks) == 2 and all(t.confirmed for t in tracks)


def test_upright_local_body_restores_path_height_after_a_local_rail_fragment():
    cfg = Config()
    tracker = Tracker(cfg)
    model = _straight_track()
    lev,rings = _resolved_body_cloud(height=.75)
    geo = track_coordinates(lev,model,cfg)
    for _ in range(3):
        tracker.update([_resolved_candidate(lev,rings)],0.,lev=lev,ring=rings,geo=geo)
    fragment = _resolved_candidate(lev,rings)
    fragment.point_index = np.flatnonzero(lev[:,2]<=.55)
    fragment.n_points = len(fragment.point_index)
    fragment.n_rings = len(np.unique(rings[fragment.point_index]))
    fragment.h_min,fragment.h_max = .04,.4
    fragment.low = fragment.rail_local = True
    tracker.update([fragment],0.,lev=lev,ring=rings,geo=geo)
    t = tracker.update([],0.,lev=lev,ring=rings,geo=geo)[0]
    assert t.confirmed and t.misses==0
    assert not t.cand.low and not t.cand.rail_local
    assert t.cand.h_min >= .19


def test_path_correction_rebases_existing_object_before_association():
    from dataclasses import replace
    tracker = Tracker(Config())
    for _ in range(3):
        tracker.update([_candidate()],0.)
    ident = tracker.tracks[0].id
    old = _straight_track()
    current = replace(old,c=old.c+.8,z=old.z-.3)
    measured = _candidate(s=18.,d=-.8)
    measured.h_min, measured.h_max = .5,2.0
    track = tracker.update([measured],2.,path_prior=old,path_current=current)[0]
    assert track.id == ident and track.confirmed and track.misses == 0
    assert track.d == pytest.approx(-.8)


def test_rebasing_does_not_change_a_height_measured_from_local_rail():
    from dataclasses import replace
    tracker = Tracker(Config())
    c = _low_body()
    c.rail_local = True
    tracker.update([c],0.)
    old = _straight_track()
    current = replace(old,c=old.c+.2,z=old.z+.3)
    track = tracker.update([],0.,path_prior=old,path_current=current)[0]
    assert track.cand.h_min == c.h_min and track.cand.h_max == c.h_max
    assert track.cand.lateral == pytest.approx(c.lateral-.2)


def test_sensor_height_recalibration_moves_a_predicted_body_in_levelled_coordinates():
    from dataclasses import replace
    tracker = Tracker(Config())
    for _ in range(3):
        tracker.update([_candidate()],0.)
    old = replace(_straight_track(),calibration={'height_m':1.1,'pitch_deg':0.,'roll_deg':0.})
    new = replace(old,calibration={'height_m':1.5,'pitch_deg':0.,'roll_deg':0.})
    t = tracker.update([],0.,path_prior=old,path_current=new)[0]
    assert t.cand.h_min == pytest.approx(.6)
    assert t.cand.h_max == pytest.approx(2.1)
    assert t.s == pytest.approx(20.)


def test_sensor_height_recalibration_preserves_local_rail_height_reference():
    from dataclasses import replace
    tracker = Tracker(Config())
    c = _low_body()
    c.rail_local = True
    tracker.update([c],0.)
    old = replace(_straight_track(),calibration={'height_m':1.1,'pitch_deg':0.,'roll_deg':0.})
    new = replace(old,calibration={'height_m':1.5,'pitch_deg':0.,'roll_deg':0.})
    t = tracker.update([],0.,path_prior=old,path_current=new)[0]
    assert t.cand.h_min == c.h_min and t.cand.h_max == c.h_max


def _low_body(s=20., single=False):
    c = _candidate(s=s, d=.8, low=True)
    c.h_min, c.h_max = .04, .10
    c.n_points, c.n_rings, c.single_beam = 8, 1 if single else 2, single
    return c


def test_short_confirmation_uses_only_multibeam_body_hits():
    tracker = Tracker(Config())
    for _ in range(2):
        assert not tracker.update([_low_body(single=True)], 0., allow_single=True)[0].confirmed
    for _ in range(2):
        assert not tracker.update([_low_body()], 0.)[0].confirmed
    assert tracker.update([_low_body()], 0.)[0].confirmed


def test_far_low_body_keeps_the_long_confirmation_window():
    tracker = Tracker(Config())
    for _ in range(5):
        assert not tracker.update([_low_body(50.)], 0.)[0].confirmed
    assert tracker.update([_low_body(50.)], 0.)[0].confirmed


def test_near_bump_with_truncated_rail_baseline_keeps_long_confirmation():
    tracker = Tracker(Config())
    for _ in range(5):
        assert not tracker.update([_low_body(7.)],0.)[0].confirmed
    assert tracker.update([_low_body(7.)],0.)[0].confirmed


def test_single_beam_low_return_requires_standing_train_and_persistence():
    tracker = Tracker(Config())
    assert not tracker.update([_low_body(single=True)], .2, allow_single=True)
    assert not tracker.update([_low_body(single=True)], 0.)
    for _ in range(Config().tracking.low_single_confirm_hits-1):
        assert not tracker.update([_low_body(single=True)], 0., allow_single=True)[0].confirmed
    assert tracker.update([_low_body(single=True)], 0., allow_single=True)[0].confirmed


def test_persistent_narrow_rail_echo_never_confirms_without_body_width():
    tracker = Tracker(Config())
    for _ in range(20):
        c = _low_body(s=45.)
        c.d_min, c.d_max = .76, .84
        c.h_min, c.h_max = .29, .41
        assert not tracker.update([c], 0.)[0].confirmed


def test_held_path_expires_and_does_not_claim_fresh_measurements():
    cfg = Config()
    estimator = TrackEstimator(cfg)
    _, model = estimator.estimate(from_raw(tunnel_scan(cols=600)).xyz)
    observed = model.observed_until
    assert observed > 60.0
    original = model.c.copy()
    for i in range(1, cfg.track.hold_missing_frames + 1):
        _, model = estimator.estimate(np.zeros((0, 3), np.float32), model)
        assert model.coast_frames == i
        assert model.observed_until == observed
        assert model.current_observed_until == cfg.track.start
        assert not model.lat_obs.any() and not model.rail.any()
        assert np.isfinite(model.c).all() and np.isfinite(model.z).all()
        within = model.s <= observed
        assert np.allclose(model.c[within], original[within])
    _, expired = estimator.estimate(np.zeros((0, 3), np.float32), model)
    assert expired.observed_until == cfg.track.start
    assert expired.coast_frames == 0


def test_sparse_startup_cannot_invent_an_observed_path_and_reset_clears_turnout():
    estimator = TrackEstimator(Config())
    _, model = estimator.estimate(np.zeros((0, 3), np.float32))
    assert model.observed_until == estimator.cfg.track.start
    assert not model.lat_obs.any() and model.coast_frames == 0
    estimator.turnout_at = 20.0
    estimator.reset()
    assert estimator.turnout_at is None


def test_far_path_tail_is_retained_when_only_near_returns_remain():
    estimator = TrackEstimator(Config())
    points = from_raw(tunnel_scan(cols=600)).xyz
    _, old = estimator.estimate(points)
    _, partial = estimator.estimate(points[points[:, 1] < 35.0], old)
    assert partial.lat_obs.any()
    assert partial.current_observed_until < 40.0
    assert partial.observed_until == old.observed_until
    assert partial.coast_frames == 1
    tail = (partial.s > 45.0) & (partial.s <= old.observed_until)
    assert np.allclose(partial.c[tail], old.c[tail])
    _, recovered = estimator.estimate(points, partial)
    assert recovered.current_observed_until > 60.0
    assert recovered.coast_frames == 0


def test_filter_rejects_isolated_duplicates_and_nonfinite_points():
    p = np.array([[0.0, 10.0, 0.8], [0.0, 10.0, 0.8],
                  [1.0, 20.0, 1.0], [np.nan, 5.0, 0.0]])
    assert not support_mask(p).any()
    assert support_mask(p, FilterConfig(enabled=False)).tolist() == [True, True, True, False]
    assert support_mask(np.zeros((0, 3))).shape == (0,)


def test_filter_preserves_sparse_far_body_and_small_near_object():
    distant = np.array([[0.0, 150.0, 0.3], [0.15, 150.0, 0.6],
                        [0.3, 150.0, 0.9], [0.0, 150.0, 1.2]])
    small = np.array([[0.8, 20.0, 0.18], [0.82, 20.0, 0.20], [0.85, 20.0, 0.22]])
    assert support_mask(np.vstack((distant, small))).all()


def test_isolated_dust_is_removed_while_person_and_small_object_are_detected():
    cfg = Config()
    track = _straight_track()
    d, h = np.meshgrid(np.linspace(-0.2, 0.2, 10), np.linspace(0.2, 1.8, 40))
    person = np.column_stack((d.ravel(), np.full(d.size, 20.0), h.ravel()))
    small = np.array([[0.70, 30.0, 0.20], [0.74, 30.0, 0.22], [0.78, 30.0, 0.24]])
    dust = np.array([(d, s, h) for d in (-0.9, 0.9) for s in np.arange(5.0, 46.0, 3.0)
                     for h in (0.6, 1.6)])
    xyz = np.vstack((person, small, dust)).astype(np.float32)
    mask = support_mask(xyz)
    assert mask[:len(person) + len(small)].all()
    assert not mask[len(person) + len(small):].any()
    lev = xyz
    # Distinct sampled beams, with no organized-cloud dependency.
    rings = np.arange(len(xyz), dtype=np.int16)
    stats = {}
    candidates = detect(lev, rings, track_coordinates(lev, track, cfg), track, cfg, filter_stats=stats)
    assert stats["removed"] == len(dust)
    assert stats["examined"] == len(xyz)
    good = [c for c in candidates if c.rejected is None]
    assert len(good) == 2
    assert good[0].s_near == 20.0 and not good[0].low
    assert good[1].s_near == 30.0 and good[1].low
    assert all((c.point_index < len(person) + len(small)).all() for c in good)


def test_filter_does_not_invent_support_across_integer_address_boundaries():
    # Far-apart cells at opposing corners may have adjacent scalar addresses;
    # padding must stop those addresses being treated as spatial neighbors.
    p = np.array([[0.0, 10.0, -2.0], [0.0, 11.0, 2.0], [4.0, 10.0, 2.0]])
    assert not support_mask(p).any()


@pytest.mark.parametrize('height,span,rings,width,expected', [
    (.10, .30, 2, .28, True),  # 300 x 300 x 100 mm item above the local rail head
    (.10, .30, 2, .07, False), # narrow rail-head echoes seen in real negative bags
    (.06, .30, 2, .28, False), # small rail-surface variation
    (.10, .30, 1, .28, True),  # broad one-beam candidate needs verified standstill + eight temporal hits
    (.10, 4.0, 2, .28, False), # extended raised switch/rail structure
])
def test_small_rail_obstacle_survives_height_gate_without_accepting_rail_noise(height, span, rings, width, expected):
    cfg = Config()
    track = _straight_track()
    s = np.arange(10., 45., .05)
    rail = np.column_stack((np.full(len(s), .8), s, np.zeros(len(s))))
    longitudinal, lateral = np.meshgrid(np.arange(20., 20. + span, .04), np.linspace(.8-width/2, .8+width/2, 6))
    item = np.column_stack((lateral.ravel(), longitudinal.ravel(), np.full(lateral.size, height)))
    lev = np.vstack((rail, item)).astype(np.float32)
    beam = np.concatenate((np.zeros(len(rail), dtype=np.int16), np.arange(len(item)) % rings))
    candidates = detect_rail_objects(lev, beam, track_coordinates(lev, track, cfg), track, cfg)
    assert bool(candidates) is expected
    if expected:
        assert candidates[0].h_max == pytest.approx(height)
        assert candidates[0].n_rings == rings


def test_falcon_centreline_obstacle_beyond_300m_is_not_rejected_by_growing_edge_margin():
    cfg = Config.load(Path(__file__).parents[1] / 'config/falcon_prime_ros.yaml')
    track = _straight_track()
    from dataclasses import replace
    track = replace(track, s=track.s * 1.8, observed_until=340., current_observed_until=340.)
    lateral, height = np.meshgrid(np.linspace(-.3, .3, 6), np.linspace(.2, 1.8, 12))
    lev = np.column_stack((lateral.ravel(), np.full(lateral.size, 330.), height.ravel())).astype(np.float32)
    rings = np.repeat(np.arange(12), 6)
    candidates = detect(lev, rings, track_coordinates(lev, track, cfg), track, cfg)
    good = [c for c in candidates if c.rejected is None]
    assert len(good) == 1 and good[0].s_near == 330.


def test_falcon_coarse_scan_density_preserves_resolved_near_person():
    cfg = Config.load(Path(__file__).parents[1] / 'config/falcon_prime_ros.yaml')
    track = _straight_track()
    d,h = np.meshgrid(np.linspace(-.25,.25,10),np.linspace(.2,1.7,30))
    lev = np.column_stack((d.ravel(),np.full(d.size,9.),h.ravel())).astype(np.float32)
    rings = np.repeat(np.arange(30),10)
    def candidates(config):
        return detect(lev,rings,track_coordinates(lev,track,config),track,config)
    assert any(c.rejected is None and c.n_points==300 for c in candidates(cfg))
    fine_only = cfg.with_overrides({'sensor':{'density_azimuth_resolution_deg':0.,
                                             'density_elevation_resolution_deg':0.}})
    assert all(c.rejected=='low_score' for c in candidates(fine_only))


@pytest.mark.parametrize('heights,expected', [
    ([2.75,2.9,3.1,3.3,3.5,3.7], True),
    ([2.75,2.9], False),
    ([2.75,2.9,3.5,3.5,3.5,3.5], False),
    ([3.1,3.3,3.5,3.7,3.9,4.1], False),
])
def test_hanging_tip_needs_continuous_vertical_body_crossing_clearance_boundary(heights,expected):
    cfg = Config()
    track = _straight_track()
    lev = np.column_stack((np.zeros(len(heights)),np.full(len(heights),20.),heights)).astype(np.float32)
    rings = np.arange(len(heights))
    cands = detect(lev,rings,track_coordinates(lev,track,cfg),track,cfg)
    good = [c for c in cands if c.rejected is None]
    assert bool(good) is expected
    if expected:
        assert good[0].n_points == 2 and good[0].n_rings == 2
        assert good[0].h_max < cfg.envelope.height


def test_rail_window_optimizations_preserve_nan_and_boundary_behavior():
    rng = np.random.default_rng(12)
    for n in (1, 12, 80, 130):
        a = rng.normal(size=n)
        a[::3] = np.nan
        for half in (2, 8):
            expected = []
            maxima = []
            for i in range(n):
                finite = a[max(0, i - half):i + half + 1]
                finite = np.sort(finite[np.isfinite(finite)])
                expected.append(finite[int(np.floor((len(finite) - 1) * .3))] if len(finite) else np.nan)
                maxima.append(finite[-1] if len(finite) else -np.inf)
            assert np.allclose(_window_percentile(a, half, .3), expected, equal_nan=True)
            assert np.array_equal(_window_max(a, half), maxima)
    x = [32.5, 33.5, 34.5, 35.5, 36.5, 37.5]
    y = [0.03, 0.01, -0.02, -0.03, -0.06, -0.08]
    assert np.allclose(_short_line(x, y), np.polyfit(x, y, 1), atol=1e-12)


def test_temporal_rail_constraint_keeps_supported_branch_despite_stronger_spurious_peaks():
    xs = np.arange(-2.2,2.21,.04)
    profile = np.zeros(len(xs))
    for x in (-.8,.8):
        profile[np.argmin(abs(xs-x))] = .15
    for x in (-.4,1.2):
        profile[np.argmin(abs(xs-x))] = .5
    unbounded = _rail_pair(xs,profile,0.,1.6,.1,.6,.06)
    constrained = _rail_pair(xs,profile,0.,1.6,.1,.6,.06,(-.25,.25))
    assert unbounded is not None and .5*(unbounded[1]+unbounded[3]) == pytest.approx(.4)
    assert constrained is not None and .5*(constrained[1]+constrained[3]) == pytest.approx(0.,abs=.001)


def test_rail_temporal_guard_requires_persistent_scatter_and_reset_clears_it():
    cfg = Config()
    estimator = TrackEstimator(cfg)
    xyz = from_raw(tunnel_scan(cols=600)).xyz
    _,prior = estimator.estimate(xyz)
    assert estimator.noisy_rail_frames == 0
    scatter = np.array([(x,s,h-1.1) for x in (-.9,.9) for s in np.arange(5.,59.,4.)
                        for h in (.8,1.8)],np.float32)
    disturbed = np.vstack((xyz,scatter))
    for k in (1,2):
        lev,prior = estimator.estimate(disturbed,prior)
        assert len(lev) == len(disturbed)
        assert estimator.rail_noise_points >= 16
        assert estimator.noisy_rail_frames == k
    estimator.reset()
    assert estimator.noisy_rail_frames == 0 and estimator.rail_noise_points == 0


