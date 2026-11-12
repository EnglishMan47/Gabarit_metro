from dataclasses import replace
from types import SimpleNamespace

import pytest

from gabarit.config import Config
from gabarit.detect import Candidate
from gabarit.tracker import Tracker


def config(enabled=True, policy="assist"):
    cfg=Config()
    # The helper exercises the Tracker contract independently of model I/O.
    settings={**vars(cfg.models),"component_support_confirmation":enabled,"component_policy":policy}
    return replace(cfg,models=SimpleNamespace(**settings))


def body(**changes):
    candidate=Candidate(1,100.,100.5,-.2,.2,.2,1.8,100.,60,8,0.,.9,
                        reasons=["gabarit_component_supported"])
    return replace(candidate,**changes)


def test_explicit_model_support_uses_normal_confirmation_for_resolved_far_body():
    tracker=Tracker(config())
    for frame in range(3):
        tracks=tracker.update([body()],0.)
        assert tracks[0].confirmed==(frame==2)


@pytest.mark.parametrize('distance',[30.,100.])
def test_explicit_two_scan_model_profile_needs_two_physical_observations(distance):
    cfg=config()
    cfg=replace(cfg,models=SimpleNamespace(**{**vars(cfg.models),'component_min_hits':2}))
    tracker=Tracker(cfg)
    assert not tracker.update([body(s_near=distance,s_far=distance+.5)],0.)[0].confirmed
    assert tracker.update([body(s_near=distance,s_far=distance+.5)],0.)[0].confirmed


@pytest.mark.parametrize("enabled,policy",[(False,"assist"),(True,"score"),(False,"score")])
def test_default_or_score_only_profiles_never_shorten_far_confirmation(enabled,policy):
    tracker=Tracker(config(enabled,policy))
    for frame in range(6):
        tracks=tracker.update([body()],0.)
        assert tracks[0].confirmed==(frame==5)


@pytest.mark.parametrize("changes",[{"reasons":[]},{"transparent":True},{"n_points":5,"n_rings":2}])
def test_tag_without_resolved_eligible_body_cannot_shorten_far_window(changes):
    tracker=Tracker(config())
    for _ in range(3):
        tracks=tracker.update([body(**changes)],0.)
        assert not tracks[0].confirmed


def test_low_object_still_uses_its_dedicated_window():
    tracker=Tracker(config())
    for frame in range(6):
        tracks=tracker.update([body(s_near=50.,s_far=50.5,low=True,h_max=.4)],0.)
        assert tracks[0].confirmed==(frame==5)
