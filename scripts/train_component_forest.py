"""Train a compact forest on beam-cast objects with whole-recording splits.

Training requires scikit-learn. Runtime evaluation uses the exported JSON only.
Labels come from the exact changed return indices, never detector predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from gabarit.component_model import ComponentFeatures, FEATURE_NAMES
from gabarit.config import Config
from gabarit.detect import detect
from scripts import synthetic_eval as ev


def collect_case(case: dict) -> dict:
    config = Config().with_overrides({'models': {'component_policy': 'off'}})
    rows = []
    target_raw = None
    process, inject, dust = ev.Pipeline.process_raw, ev.inject, ev.inject_dust

    def capture_inject(raw, *args, **kwargs):
        nonlocal target_raw
        changed, count = inject(raw, *args, **kwargs)
        target_raw = np.logical_or.reduce([changed.points[k] != raw.points[k] for k in 'xyz'])
        return changed, count

    def capture_dust(raw, *args, **kwargs):
        nonlocal target_raw
        changed, count = dust(raw, *args, **kwargs)
        if target_raw is not None:
            target_raw &= np.logical_and.reduce([changed.points[k] == raw.points[k] for k in 'xyz'])
        return changed, count

    def capture_process(pipe, raw, *args, **kwargs):
        nonlocal target_raw
        result = process(pipe, raw, *args, **kwargs)
        if result.data['frame'] >= case['start_frame']:
            target = (target_raw[result.cloud.source_index] if target_raw is not None
                      else np.zeros(len(result.cloud), bool))
            candidates = detect(result.lev, result.cloud.ring, result.geo, result.track,
                                pipe.cfg, result.cloud.split)
            for candidate in candidates:
                ids = candidate.point_index
                if ids is None or len(ids) < 3:
                    continue
                hits = int(target[ids].sum())
                # Mixed bridge components and accidental one-return overlap
                # cannot become convenient positive training examples.
                if hits and (hits < 3 or hits / len(ids) < .5):
                    continue
                feature = ComponentFeatures.from_points(result.cloud.xyz[ids])
                rows.append({'features': feature.vector(), 'label': int(hits > 0),
                             'target_returns': hits, 'frame': result.data['frame'],
                             'rejected': candidate.rejected, 'rings': candidate.n_rings,
                             'low': candidate.low, 'lateral': candidate.lateral,
                             'box': {'s':[candidate.s_near,candidate.s_far],
                                     'd':[candidate.d_min,candidate.d_max],
                                     'h':[candidate.h_min,candidate.h_max]}})
        target_raw = None
        return result

    ev.inject, ev.inject_dust, ev.Pipeline.process_raw = capture_inject, capture_dust, capture_process
    try:
        result = ev.run_case((case['bag'], case['shape'], case['lateral'], case['start_frame'],
                              case['distance'], config.to_dict(), case['frames'], case['yaw'],
                              case['dust'], case['seed'], case['reference']))
    finally:
        ev.inject, ev.inject_dust, ev.Pipeline.process_raw = inject, dust, process
    return {'case': case, 'geometry_sha256': result['input_geometry_sha256'],
            'result': result, 'rows': rows}


def default_cases(data: Path, reference: Path):
    # Neither held-out recording is present in training or calibration.
    groups = [('train', 'doubleT_platform', 24, 100100),
              ('calibration', 'roundT_squareT_pressureGate_squareT', 12, 200200),
              ('test', 'roundT_pressureGate_roundT', 12, 300300)]
    shapes = ['person', 'person_lying', 'person_sitting', 'child', 'box_50cm', 'barrel', 'trolley', 'glass']
    cases = []
    for split, name, count, seed in groups:
        rng = np.random.default_rng(seed)
        for index in range(count):
            cases.append({'split':split, 'bag':str((data/name).resolve()),
                          'shape':shapes[index % len(shapes)] if index % 9 != 8 else 'none',
                          'lateral':float(rng.choice([-.8,-.4,0.,.4,.8])),
                          'start_frame':int(rng.choice([10,40,80])),
                          'distance':float(rng.choice([30.,50.,70.,100.])),
                          'frames':65, 'yaw':float(rng.choice([0.,30.,60.,90.])),
                          'dust':float(rng.choice([0.,.005,.02])), 'seed':seed+index,
                          'reference':str(reference.resolve())})
    return cases


def metrics(model, rows, threshold):
    from sklearn.metrics import roc_auc_score
    x = np.array([r['features'] for r in rows])
    truth = np.array([r['label'] for r in rows])
    score = model.predict_proba(x)[:, 1]
    predicted = score >= threshold
    tp, fp = int((predicted & (truth == 1)).sum()), int((predicted & (truth == 0)).sum())
    return {'rows':len(rows), 'positive':int(truth.sum()), 'true_positive':tp, 'false_positive':fp,
            'precision':tp/max(tp+fp,1), 'recall':tp/max(int(truth.sum()),1),
            'roc_auc':float(roc_auc_score(truth,score)), 'threshold':threshold}


def fit(cases, target):
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    groups = {name:[row for case in cases if case['case']['split']==name for row in case['rows']]
              for name in ('train','calibration','test')}
    bags = {name:{c['case']['bag'] for c in cases if c['case']['split']==name} for name in groups}
    assert not any(bags[a] & bags[b] for a,b in [('train','test'),('train','calibration'),('calibration','test')])
    training = groups['train']
    model = RandomForestClassifier(n_estimators=31, max_depth=7, min_samples_leaf=5,
                                   class_weight='balanced', random_state=20261110, n_jobs=1)
    model.fit([r['features'] for r in training], [r['label'] for r in training])
    # Threshold is selected solely on the separate calibration recording.
    # Require no calibration false positive; test labels never select it.
    thresholds = np.unique(np.r_[np.linspace(.9,1.,101), model.predict_proba([r['features'] for r in groups['calibration']])[:,1]])
    threshold = 1.0
    for value in thresholds:
        score = metrics(model,groups['calibration'],float(value))
        if value >= .9 and score['false_positive']==0 and score['true_positive']:
            threshold = float(value)
            break
    report = {name:metrics(model,rows,threshold) for name,rows in groups.items()}
    report['whole_bag_split'] = {k:sorted(Path(v).name for v in values) for k,values in bags.items()}
    report['feature_names'] = list(FEATURE_NAMES)
    report['candidate_rejections'] = {name:dict(Counter(r['rejected'] or 'accepted' for r in rows if r['label']))
                                     for name,rows in groups.items()}

    def export(tree, index=0):
        weights = np.asarray(tree.value[index]).ravel()
        total = float(weights.sum())
        weights = weights/total if total else np.array([1.,0.])
        node = {'negative_weight':float(weights[0]),'positive_weight':float(weights[1])}
        if tree.children_left[index] != tree.children_right[index]:
            feature = int(tree.feature[index])
            node.update(feature_index=feature,feature=FEATURE_NAMES[feature],threshold=float(tree.threshold[index]),
                        left=export(tree,int(tree.children_left[index])),right=export(tree,int(tree.children_right[index])))
        return node
    portable_rows = [dict(case={k:(Path(v).name if k in ('bag','reference') else v)
                                    for k,v in c['case'].items()},
                          geometry_sha256=c['geometry_sha256'],rows=c['rows']) for c in cases]
    payload = {'format':'gabarit-component-forest-v1','features':list(FEATURE_NAMES),
               'aggregation':'mean_tree_probability','threshold':threshold,
               'trees':[export(est.tree_) for est in model.estimators_],
               'training':{'seed':20261110, 'whole_bag_split':report['whole_bag_split'],
                           'training_data_sha256':hashlib.sha256(json.dumps(portable_rows,sort_keys=True).encode()).hexdigest(),
                           'calibration_zero_false_positives':report['calibration']['false_positive']==0,
                           'sklearn_version':sklearn.__version__,
                           'ground_truth':'exact changed return indices after beam casting and dust occlusion',
                           'test_never_selects_threshold':True}}
    target.write_text(json.dumps(payload,separators=(',',':'))+'\n',encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path('data/extracted/for_hackathon'))
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--features',type=Path,help='Previously exported labelled feature dataset (.json.gz); no bags required')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--reuse',action='store_true')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.features is None and args.reference is None:
        ap.error('--reference is required for cloud collection; --features reproduces training without bags')
    cases = default_cases(args.data,args.reference) if args.reference is not None else []
    (args.out/'manifest.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    result=[]
    if args.features is not None:
        with gzip.open(args.features,'rt',encoding='utf-8') as source:
            result=json.load(source)
    elif args.reuse:
        result=json.loads((args.out/'dataset.json').read_text())
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index,case in enumerate(pool.map(collect_case,cases)):
                result.append(case)
                (args.out/'dataset.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
                print(f"{index+1}/{len(cases)} {case['case']['split']} {case['case']['shape']} rows={len(case['rows'])}",flush=True)
    report=fit(result,args.out/'component_forest.json')
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
