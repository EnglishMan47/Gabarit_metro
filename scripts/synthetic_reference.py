#!/usr/bin/env python3
"""Freeze clean background geometry before comparing detector versions.

python scripts/synthetic_reference.py data/extracted/for_hackathon --out artifacts/reference

Ray-cast targets and their motion must not depend on the version under test.
The reference is an estimated clean background trajectory, not surveyed rail
ground truth. Keep the generated files fixed for a paired comparison.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.rosbag import BagReader,discover_bags
from gabarit.track import Calibration


@lru_cache(maxsize=16)
def load_reference(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('unsupported synthetic reference schema')
    return data


def reference_frame(reference: dict, index: int, stamp_ns: int):
    row = reference['frames'][index]
    if row['frame'] != index or row['stamp_ns'] != stamp_ns:
        raise ValueError('synthetic reference does not match the recording frame/timestamp')
    model = SimpleNamespace(s=np.asarray(row['s']),c=np.asarray(row['c']),z=np.asarray(row['z']))
    calib = Calibration(**row['calibration'])
    return model,calib,float(row['displacement_m'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',type=Path)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bags',nargs='*')
    ap.add_argument('--overwrite',action='store_true')
    args = ap.parse_args()
    cfg = Config.load(args.config)
    args.out.mkdir(parents=True,exist_ok=True)
    for bag in discover_bags(args.data):
        if args.bags and bag.name not in args.bags:
            continue
        target = args.out/(bag.name+'.json')
        if target.exists() and not args.overwrite:
            print('keep',target,flush=True)
            continue
        pipe = Pipeline(cfg)
        reader = BagReader(bag)
        frames = []
        try:
            for i,ts,raw in reader.iterate():
                result = pipe.process_raw(raw,ts*1e-9,frame_index=i)
                track = result.track
                # TrackModel records the pose that produced this levelled
                # frame. The estimator has already updated its live pose for
                # the NEXT frame by the time process_raw returns.
                pose = track.calibration
                calibration = dict(height=pose['height_m'],pitch=math.tan(math.radians(pose['pitch_deg'])),
                                   roll=math.tan(math.radians(pose['roll_deg'])),valid=pose['valid'],frames=0)
                frames.append(dict(frame=i,stamp_ns=int(ts),s=track.s.tolist(),c=track.c.tolist(),
                                   z=track.z.tolist(),calibration=calibration,
                                   displacement_m=result.data['displacement_m']))
                if i%200 == 0:
                    print(bag.name,i,len(reader),flush=True)
        finally:
            reader.close()
        data = dict(schema_version=1,bag=bag.name,config=cfg.to_dict(),frames=frames,
                    note='Frozen clean-background estimate; not surveyed ground truth.')
        target.write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf-8')
        print('saved',target,flush=True)


if __name__ == '__main__':
    main()
