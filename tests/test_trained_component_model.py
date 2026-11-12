"""Own forest export parity, calibrated assistance and sensor uncertainty gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gabarit.component_model import ComponentFeatures, ComponentModel, ComponentModelAssistant, FEATURE_NAMES
from gabarit.detect import Candidate


def model_file(tmp_path, calibrated=True):
    path = tmp_path / 'own.json'
    path.write_text(json.dumps({'format':'gabarit-component-forest-v1','features':list(FEATURE_NAMES),
                               'aggregation':'mean_tree_probability','threshold':.99,
                               'trees':[{'negative_weight':0.,'positive_weight':1.}],
                               'training':{'calibration_zero_false_positives':calibrated}}))
    return path


def candidate():
    points = np.random.default_rng(10).uniform(-.2,.2,(100,3)).astype(np.float32)
    points += [0.,70.,0.]
    return Candidate(1,69.8,70.2,-.2,.2,.2,1.5,70.,100,4,0.,.7,
                     point_index=np.arange(100)), points


def apply(assistant, c, points, frame, sigma=.05, **kw):
    return assistant.apply([c],points,stamp=frame*.1,observed_until_m=100.,path_reliable=True,
                           policy='assist',min_hits=3,support_confirmation=True,
                           path=SimpleNamespace(s=np.array([3.,100.]),sig_c=np.full(2,sigma)), **kw)


def test_three_consistent_calibrated_model_scans_support_only_existing_body(tmp_path):
    assistant = ComponentModelAssistant(ComponentModel(model_file(tmp_path)))
    for frame in range(3):
        c, points = candidate()
        report = apply(assistant,c,points,frame)
        assert ('gabarit_component_supported' in c.reasons) == (frame==2)
        assert report['supported'] == int(frame==2)
        assert c.rejected is None and c.score == .7
        assert report['promoted'] == 0
    # Confidence is tied to current path evidence, not copied from history.
    report = apply(assistant,c,points,3,sigma=.5)
    assert 'gabarit_component_supported' not in c.reasons
    assert report['candidates'][0]['reason'] == 'uncertain_path'


@pytest.mark.parametrize('change',[{'low':True},{'single_beam':True},{'transparent':True},
                                   {'rejected':'ground'},{'n_rings':1},{'d_max':1.1}])
def test_model_score_cannot_bypass_geometry_or_special_low_branch(tmp_path, change):
    assistant = ComponentModelAssistant(ComponentModel(model_file(tmp_path)))
    for frame in range(5):
        c,points = candidate()
        for key,value in change.items():setattr(c,key,value)
        before = c.rejected
        report=apply(assistant,c,points,frame)
        assert c.rejected == before and 'gabarit_component_supported' not in c.reasons
        assert report['supported'] == report['promoted'] == 0


def test_missing_or_uncalibrated_weights_fall_back_to_score_only(tmp_path):
    for model in (ComponentModel.optional(tmp_path/'absent.json'),
                  ComponentModel(model_file(tmp_path,calibrated=False))):
        assistant=ComponentModelAssistant(model)
        c,points=candidate()
        for frame in range(5):
            report=apply(assistant,c,points,frame)
            assert report['policy']=='score' and report['requested_policy']=='assist'
            assert report['fallback_reason']
            assert c.rejected is None and not c.reasons


def test_packaged_weights_have_independent_whole_recording_split():
    path=Path(__file__).resolve().parents[1]/'gabarit/models/component_forest_v1.json'
    model=ComponentModel(path)
    training=json.loads(path.read_text())['training']
    split={k:set(v) for k,v in training['whole_bag_split'].items()}
    assert not split['train']&split['test'] and not split['train']&split['calibration']
    assert not split['test']&split['calibration']
    assert training['test_never_selects_threshold'] is True
    assert training['calibration_zero_false_positives'] is True
    assert model.weight_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(model.trees)==31 and .99<=model.threshold<1.


@pytest.mark.skipif(importlib.util.find_spec('sklearn') is None,reason='Optional offline training library')
def test_json_inference_matches_actual_sklearn_export_on_unseen_vectors(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from scripts.train_component_forest import fit
    rng=np.random.default_rng(20261110)
    features=np.column_stack([rng.integers(3,200,150),rng.uniform(.1,2.,(150,7))])
    labels=(features[:,0]>60).astype(int)
    rows=[{'features':list(x),'label':int(y),'rejected':None} for x,y in zip(features,labels)]
    cases=[{'case':{'split':name,'bag':f'/data/{name}','reference':'/reference'},
            'rows':rows,'geometry_sha256':name} for name in ('train','calibration','test')]
    path=tmp_path/'export.json'
    fit(cases,path)
    reference=RandomForestClassifier(n_estimators=31,max_depth=7,min_samples_leaf=5,
                                     class_weight='balanced',random_state=20261110,n_jobs=1)
    reference.fit(features,labels)
    vectors=np.column_stack([rng.integers(3,200,500),rng.uniform(.1,2.,(500,7))])
    actual=ComponentModel(path)
    scores=[actual.predict(ComponentFeatures(*vector)).score for vector in vectors]
    assert np.asarray(scores)==pytest.approx(reference.predict_proba(vectors)[:,1],abs=1e-14)
