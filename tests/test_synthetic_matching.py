"""Synthetic success requires actual injected evidence, not a nearby alarm."""
from types import SimpleNamespace as NS
import json

import numpy as np

from gabarit.cloud import from_raw, axis_transform
from scripts.synthetic_eval import SyntheticMatches
import scripts.synthetic_eval as synthetic_eval
from scripts.synthetic_reference import reference_frame
from tests.sim import tunnel_scan


def _track(ident, point_index, *, misses=0):
    return NS(id=ident,s=20.,misses=misses,confirmed=True,
              cand=NS(point_index=point_index,rejected=None))


def test_nearby_alarm_cannot_be_counted_as_synthetic_detection():
    matcher = SyntheticMatches()
    result = NS(data={})
    wrong = _track(7,np.array([0]))
    target = np.array([False,True])
    assert matcher.update(result,NS(tracks=[wrong]),target,20.) == (False,False,True)
    correct = _track(8,np.array([1]))
    assert matcher.update(result,NS(tracks=[wrong,correct]),target,20.) == (True,True,True)
    held = _track(8,None,misses=1)
    assert matcher.update(result,NS(tracks=[held]),target,20.) == (False,True,False)
    assert matcher.update(NS(data={'state_reset':'time_gap'}),NS(tracks=[held]),target,20.) == (False,False,True)


def test_background_reassociation_invalidates_target_identity():
    matcher = SyntheticMatches()
    result = NS(data={})
    target = np.array([False, True])
    assert matcher.update(result,NS(tracks=[_track(8,np.array([1]))]),target,20.) == (True,True,False)
    assert matcher.update(result,NS(tracks=[_track(8,np.array([0]))]),target,20.) == (False,False,True)
    assert matcher.update(result,NS(tracks=[_track(8,None,misses=1)]),target,20.) == (False,False,True)


def test_optional_source_indices_survive_working_mask_and_subsetting():
    raw = tunnel_scan(cols=40)
    raw.points['x'][100] = np.nan
    plain = from_raw(raw,working_range=35.)
    tagged = from_raw(raw,working_range=35.,preserve_source_index=True)
    assert plain.source_index is None
    np.testing.assert_array_equal(tagged.xyz,plain.xyz)
    original = np.column_stack([raw.points[k] for k in 'xyz'])
    np.testing.assert_array_equal(tagged.xyz,original[tagged.source_index] @ axis_transform().T)
    chosen = tagged.subset(np.arange(len(tagged))%2==0)
    np.testing.assert_array_equal(chosen.source_index,tagged.source_index[::2])
    assert 100 not in chosen.source_index


def test_frozen_reference_keeps_test_inputs_identical_when_detector_pose_changes(tmp_path,monkeypatch):
    from gabarit.config import Config
    raw = tunnel_scan(cols=300)
    class MemoryReader:
        def __init__(self,*args): pass
        def __len__(self): return 8
        def read(self,index): return index*100_000_000,raw
        def close(self): pass
    monkeypatch.setattr(synthetic_eval,'BagReader',MemoryReader)
    monkeypatch.setattr(synthetic_eval,'open_bag',lambda path:path)
    reference = dict(schema_version=1,config=Config().to_dict(),frames=[
        dict(frame=i,stamp_ns=i*100_000_000,s=[3.,40.,80.],c=[0.,0.,0.],z=[0.,0.,0.],
             displacement_m=0.,calibration=dict(height=1.1,pitch=0.,roll=0.,valid=True,frames=0))
        for i in range(8)])
    (tmp_path/'memory_bag.json').write_text(json.dumps(reference),encoding='utf-8')
    args = ('memory_bag','person',.6,0,20.,Config().to_dict(),8,0.,.01,41,str(tmp_path))
    a = synthetic_eval.run_case(args)
    changed = Config().with_overrides({'sensor':{'auto_calibration':False,'default_pitch_deg':5.}})
    b = synthetic_eval.run_case(args[:5]+(changed.to_dict(),)+args[6:])
    assert a['input_geometry_sha256'] == b['input_geometry_sha256']
    assert a['injected_beams'] > 0 and a['injected_beams'] == b['injected_beams']
    assert a['frames'] == b['frames'] == 8


def test_reference_from_another_timestamp_is_rejected():
    import pytest
    with pytest.raises(ValueError,match='frame/timestamp'):
        reference_frame({'frames':[{'frame':0,'stamp_ns':42}]},0,43)
