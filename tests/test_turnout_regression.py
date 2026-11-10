"""Temporal policy regression: configured evidence windows must be honoured."""
from __future__ import annotations

import pytest

from gabarit.config import Config
from gabarit.detect import Candidate
from gabarit.tracker import Tracker


def candidate(kind):
    far = kind == "far"
    low = kind in {"low", "low_body", "low_single"}
    shallow = kind in {"low_body", "low_single"}
    s = 100. if far else 20.
    return Candidate(1, s, s + .5, -.2, .2, .1 if low else .2,
                     .2 if shallow else .4 if low else 1.7,
                     s, 20, 4, 0., .9, low=low,
                     single_beam=kind == "low_single")


@pytest.mark.parametrize("kind,keys", [
    ("near", ("confirm_hits", "confirm_window")),
    ("far", ("far_confirm_hits", "far_confirm_window")),
    ("low", ("low_confirm_hits", "low_confirm_window")),
    ("low_body", ("low_body_confirm_hits", "low_body_confirm_window")),
    ("low_single", ("low_single_confirm_hits", "low_single_confirm_window")),
])
def test_long_custom_confirmation_window_can_confirm_after_required_hits(kind, keys):
    cfg = Config().with_overrides({"tracking": {keys[0]: 20, keys[1]: 24}})
    tracker = Tracker(cfg)
    for frame in range(20):
        tracks = tracker.update([candidate(kind)], 0., allow_single=True)
        assert len(tracks) == 1
        assert tracks[0].confirmed == (frame == 19)
    assert tracks[0].hits == 20
    assert tracks[0].confirmed_frame == 20
    # Enlarging the configured memory must not make it grow without a bound.
    for _ in range(40):
        tracker.update([candidate(kind)], 0., allow_single=True)
    for evidence in (tracks[0].history, tracks[0].body_history,
                     tracks[0].wide_body_history, tracks[0].strict_history):
        assert len(evidence) == 24


def test_a_short_custom_window_does_not_count_evidence_outside_its_window():
    cfg = Config().with_overrides({"tracking": {"confirm_hits": 3, "confirm_window": 3}})
    tracker = Tracker(cfg)
    # The stored history is longer than this policy, but only its last 3
    # entries count; alternating misses cannot confirm a new obstacle.
    for _ in range(12):
        tracker.update([candidate("near")], 0.)
        tracks = tracker.update([], 0.)
        assert not any(t.confirmed for t in tracks)
    for _ in range(3):
        tracks = tracker.update([candidate("near")], 0.)
    assert tracks[0].confirmed
