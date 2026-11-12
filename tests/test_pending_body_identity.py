from dataclasses import replace

import numpy as np

from gabarit.config import Config
from gabarit.detect import Candidate
from gabarit.tracker import Tracker


def low_body(width, s=15.):
    return Candidate(1,s,s+.25,.8-width/2,.8+width/2,.08,.43,s,20,8,.8,1.,
                     low=True,rail_local=True,point_index=np.arange(20))


def test_pending_wide_body_cannot_confirm_a_different_narrow_rail_fragment():
    tracker=Tracker(Config())
    for frame in range(3):
        tracker.update([low_body(.38,15.-frame)],1.)
    original=tracker.tracks[0].id
    for frame in range(3,9):
        tracker.update([low_body(.08,15.-frame)],1.)
        assert not any(t.confirmed for t in tracker.tracks)
    assert all(t.id != original for t in tracker.tracks)


def test_consistent_low_body_still_uses_the_configured_confirmation_window():
    tracker=Tracker(Config())
    for frame in range(6):
        tracker.update([low_body(.3,20.-frame)],1.)
        assert any(t.confirmed for t in tracker.tracks) == (frame==5)


def test_sparse_sampling_alone_does_not_break_pending_body_identity():
    tracker=Tracker(Config())
    tracker.update([low_body(.3)],0.)
    ident=tracker.tracks[0].id
    tracker.update([replace(low_body(.08),n_points=4,n_rings=2)],0.)
    assert tracker.tracks[0].id == ident


def upright(score=1., rejected=None):
    return replace(low_body(.4,30.),low=False,rail_local=False,h_min=.2,h_max=1.8,
                   score=score,rejected=rejected,n_points=40,n_rings=8)


def test_consistent_pending_body_bridges_one_weak_frame_between_two_strong_observations():
    tracker=Tracker(Config())
    first=tracker.update([upright()],0.)[0]
    tracker.update([upright(.3,'low_score')],0.)
    assert tracker.tracks[0].id==first.id and not tracker.tracks[0].confirmed
    tracker.update([upright()],0.)
    assert tracker.tracks[0].id==first.id and tracker.tracks[0].confirmed


def test_one_strong_measurement_followed_only_by_weak_scores_cannot_confirm():
    tracker=Tracker(Config())
    tracker.update([upright()],0.)
    for _ in range(12):
        tracker.update([upright(.3,'low_score')],0.)
        assert not any(t.confirmed for t in tracker.tracks)


def test_weak_rail_body_cannot_borrow_pending_confirmation():
    tracker=Tracker(Config())
    tracker.update([low_body(.3)],0.)
    tracker.update([replace(low_body(.3),score=.3,rejected='low_score')],0.)
    assert tracker.tracks[0].hits==1
