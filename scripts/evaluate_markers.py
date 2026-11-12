#!/usr/bin/env python3
"""Match detector point evidence to independent organizer marker geometry.

No estimated path is used to decide marker membership. This reports support
of visible annotated objects, NOT binary path-intrusion FP/FN: an active marker
can be outside the train's route, and an unmatched alarm may be a real object.
python scripts/evaluate_markers.py --data data/extracted/finalist_markers_20261019 --out results/marker-eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gabarit.annotation_sources import marker_records, point_membership
from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.rosbag import BagReader, discover_bags


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stop',type=int)
    args=parser.parse_args(argv)
    if args.stop is not None and args.stop<0:parser.error('--stop must be nonnegative')
    cfg=Config.load(args.config)
    bags=discover_bags(args.data)
    if not bags:parser.error('no recordings found')
    args.out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for info in bags:
        annotations={r['timestamp_ns']:r for r in marker_records(info)}
        reader=BagReader(info);pipe=Pipeline(cfg)
        rows=[];objects={};unmatched_alarms=0;annotated=0;frame_mismatch=0
        try:
            for index,stamp,raw in reader.iterate(stop=args.stop):
                result=pipe.process_raw(raw,stamp*1e-9,frame_index=index,preserve_source_index=True)
                record=annotations.get(int(stamp))
                row=dict(frame=index,timestamp_ns=int(stamp),annotation_available=record is not None,objects=[])
                if record is None:
                    rows.append(row);continue
                annotated+=1
                source_indices=result.cloud.source_index
                sensor_points=np.column_stack([raw.points[k][source_indices] for k in 'xyz'])
                supported=[]
                for marker in record['markers']:
                    if marker['action']!=0:continue
                    if marker['frame_id']!=raw.frame_id:
                        frame_mismatch+=1;continue
                    try:target=point_membership(sensor_points,marker)
                    except ValueError:continue
                    # Matching is by actual retained returns, never just boxes
                    # near an annotated marker or the detector's own corridor.
                    count=int(target.sum())
                    matched=[]
                    for track in pipe.output_tracks:
                        ids=track.cand.point_index
                        if track.confirmed and track.misses==0 and ids is not None and target[ids].any():
                            matched.append(track.id)
                    key=f"{marker['ns']}:{marker['id']}"
                    stats=objects.setdefault(key,dict(active_frames=0,visible_frames=0,supported_frames=0,
                        first_supported_range_m=None,first_supported_frame=None))
                    stats['active_frames']+=1;stats['visible_frames']+=count>0;stats['supported_frames']+=bool(matched)
                    distance=float(np.linalg.norm(result.cloud.xyz[target],axis=1).min()) if count else None
                    if matched and stats['first_supported_frame'] is None:
                        stats['first_supported_frame']=index;stats['first_supported_range_m']=distance
                    row['objects'].append(dict(marker=key,visible_points=count,range_m=distance,confirmed_track_ids=matched))
                    supported+=matched
                row['unmatched_confirmed_track_ids']=[t.id for t in pipe.output_tracks if t.confirmed and t.id not in supported]
                unmatched_alarms+=len(row['unmatched_confirmed_track_ids'])
                rows.append(row)
                if index%100==0:print(f'{info.id}: {index}/{len(reader)}',flush=True)
        finally:reader.close()
        target=args.out/info.id
        target.parent.mkdir(parents=True,exist_ok=True)
        target.with_suffix('.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
        report=dict(bag=info.id,frames=len(rows),frames_with_synchronized_annotations=annotated,
            marker_frame_mismatches=frame_mismatch,objects=objects,unmatched_confirmed_observations=unmatched_alarms,
            binary_path_intrusion_metrics_available=False,
            scope='Independent source-marker point-support coverage; marker presence alone is not path intrusion')
        reports.append(report)
        (args.out/'summary.json').write_text(json.dumps(dict(bags=reports),indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':raise SystemExit(main())
