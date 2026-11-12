#!/usr/bin/env python3
"""Replay immutable ray-cast scenes; compare input identity before quality.

python scripts/regression_suite.py --data data/extracted/for_hackathon --out results/regression --workers 4
The fixtures contain estimated clean-background geometry, never surveyed rail
truth. The heldout_augmentation partition shares backgrounds with regression;
whole-bag generalization needs independently annotated recordings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import lzma
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gabarit.config import Config
from scripts.synthetic_eval import run_case


def source_identity(root: Path = ROOT) -> dict:
    paths = sorted((root / 'gabarit').rglob('*.py'))
    paths += sorted((root / 'gabarit/models').glob('*.json'))
    paths += [root/'scripts/synthetic_eval.py', root/'scripts/synthetic_reference.py']
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_suite(path: Path) -> dict:
    suite = json.loads(path.read_text(encoding='utf-8'))
    if suite.get('schema_version') != 1 or not isinstance(suite.get('cases'), list):
        raise ValueError('suite needs schema_version=1 and cases')
    if not suite['cases']:
        raise ValueError('suite is empty')
    for name, digest in suite.get('reference_sha256', {}).items():
        reference = (path.parent/name).resolve()
        if reference.parent != path.parent.resolve():
            raise ValueError('reference must be next to manifest')
        if hashlib.sha256(reference.read_bytes()).hexdigest() != digest:
            raise ValueError(f'reference identity mismatch: {name}')
    return suite


def summarize(runs: list[dict]) -> dict:
    positives = [r for r in runs if not r['negative_control']]
    negatives = [r for r in runs if r['negative_control']]
    return dict(cases=len(runs), positive_cases=len(positives),
                detected=sum(r['detected'] for r in positives),
                eligible_missed_frames=sum(r['eligible_missed_frames'] for r in positives),
                post_confirmation_missed_frames=sum(r['post_confirmation_missed_eligible_frames'] for r in positives),
                extra_alarm_frames=sum(r['other_alarm_frames'] for r in positives),
                negative_cases=len(negatives), negative_alarm_frames=sum(r['other_alarm_frames'] for r in negatives))


def quality_failures(runs: list[dict], cases: list[dict]) -> list[dict]:
    """Compare individual scenes, so aggregate gains cannot hide a lost target."""
    failures=[]
    for run,case in zip(runs,cases):
        baseline=case['baseline']
        reasons=[]
        if not run['negative_control'] and baseline['detected'] and not run['detected']:
            reasons.append('lost_target')
        for metric in ('post_confirmation_missed_eligible_frames','other_alarm_frames'):
            if run[metric]>baseline[metric]:reasons.append(metric)
        if reasons:failures.append(dict(seed=case['seed'],reasons=reasons))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'tests/fixtures/regression118/manifest.json')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='*')
    parser.add_argument('--check', action='store_true', help='fail on a lost target or additional post-confirmation misses/alarms in any scene')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    suite = load_suite(args.manifest)
    cases = [c for c in suite['cases'] if not args.seeds or c['seed'] in args.seeds]
    if not cases:
        parser.error('no selected cases')
    args.out.mkdir(parents=True, exist_ok=True)
    reference = args.out/'reference'
    reference.mkdir(exist_ok=True)
    for source in args.manifest.parent.glob('*.json.xz'):
        (reference/source.stem).write_bytes(lzma.decompress(source.read_bytes()))
    cfg = Config.load(args.config).to_dict()
    jobs = []
    for c in cases:
        bag = args.data/c['bag']
        if not bag.is_dir():
            parser.error(f'missing recording: {bag}')
        jobs.append((str(bag), c['shape'], c['lateral'], c['start_frame'], c['start_distance'],
                     cfg, c['frames'], c['yaw_deg'], c['dust_fraction'], c['seed'], str(reference)))
    identity = source_identity()
    (args.out/'manifest.json').write_text(json.dumps(dict(suite=args.manifest.name,
        config=cfg, cases=cases, source_sha256=identity), indent=2), encoding='utf-8')
    runs, mismatches = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for c, run in zip(cases, pool.map(run_case, jobs)):
            if run['input_geometry_sha256'] != c['expected_input_geometry_sha256']:
                mismatches.append(c['seed'])
            runs.append(run)
            (args.out/'runs.json').write_text(json.dumps(runs, indent=1, allow_nan=False), encoding='utf-8')
            print(f"{len(runs)}/{len(cases)} {c['bag']} {c['shape']} seed={c['seed']} "
                  f"detected={run['detected']} post={run['post_confirmation_missed_eligible_frames']} extra={run['other_alarm_frames']}", flush=True)
    report = dict(current=summarize(runs), baseline=summarize([c['baseline'] for c in cases]),
                  input_identity_mismatch_seeds=mismatches, source_sha256=identity,
                  source_unchanged_during_run=identity == source_identity(),
                  scope=suite.get('note'))
    report['quality_regressions']=quality_failures(runs,cases)
    (args.out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['current']))
    return 1 if mismatches or not report['source_unchanged_during_run'] or (args.check and report['quality_regressions']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
