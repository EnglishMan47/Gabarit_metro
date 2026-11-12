"""Paired temporal evaluation of a fixed optional model on whole held-out rides."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gabarit.config import Config
from scripts.synthetic_eval import run_case
from scripts.train_component_forest import default_cases


def paired(case):
    scene, config_path=case
    default=Config().with_overrides({'models':{'component_policy':'off'}})
    model=Config.load(config_path)
    base=(scene['bag'],scene['shape'],scene['lateral'],scene['start_frame'],scene['distance'],
          default.to_dict(),scene['frames'],scene['yaw'],scene['dust'],scene['seed'],scene['reference'])
    current=list(base);current[5]=model.to_dict()
    before,after=run_case(base),run_case(tuple(current))
    assert before['input_geometry_sha256']==after['input_geometry_sha256'], 'Paired input geometry differs'
    return {'case':scene,'before':before,'after':after}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path('data/extracted/for_hackathon'))
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=Path('config/component_trained.yaml'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--frozen-manifest',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.frozen_manifest:
        data=json.loads(args.frozen_manifest.read_text())
        scenes=[{'bag':row[0],'shape':row[1],'lateral':row[2],'start_frame':row[3],
                 'distance':row[4],'frames':row[6],'yaw':row[7],'dust':row[8],
                 'seed':row[9],'reference':row[10],'split':'frozen-test'} for row in data]
    else:
        scenes=[c for c in default_cases(args.data,args.reference) if c['split']=='test']
        # A fourth recording was absent from training, threshold calibration
        # and the component-level test. New seeds are fixed before inference.
        for index,c in enumerate(list(scenes)):
            c=dict(c,bag=str((args.data/'roundT_doubleT').resolve()),split='external-test',seed=400400+index)
            scenes.append(c)
    (args.out/'manifest.json').write_text(json.dumps(scenes,indent=2),encoding='utf-8')
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(paired,[(s,str(args.config)) for s in scenes]):
            rows.append(item)
            (args.out/'runs.json').write_text(json.dumps(rows,indent=1),encoding='utf-8')
            a,b=item['before'],item['after']
            print(len(rows),'/',len(scenes),a['bag'],a['shape'],a['confirmed_at_m'],'->',b['confirmed_at_m'],
                  'post',b['post_confirmation_missed_eligible_frames'],'extra',b['other_alarm_frames'],flush=True)
    report={}
    for name in ('before','after'):
        items=[row[name] for row in rows]
        report[name]={'cases':len(items),'detected':sum(r['detected'] for r in items if not r['negative_control']),
                      'positive_cases':sum(not r['negative_control'] for r in items),
                      'eligible_missed_frames':sum(r['eligible_missed_frames'] for r in items),
                      'post_confirmation_missed_frames':sum(r['post_confirmation_missed_eligible_frames'] for r in items),
                      'extra_alarm_frames':sum(r['other_alarm_frames'] for r in items),
                      'negative_alarm_frames':sum(r['other_alarm_frames'] for r in items if r['negative_control'])}
    report['all_input_sha256_equal']=True
    report['config']=str(args.config)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
