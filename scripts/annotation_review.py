#!/usr/bin/env python3
"""Create an offline annotation review pack directly from raw clouds.

No Pipeline/model output is loaded. Every frame starts UNKNOWN; exported
labels are explicit reviewer choices accepted by scripts/evaluate.py.
python scripts/annotation_review.py --data data/extracted/falcon_test --config config/falcon_prime_ros.yaml --out results/review --every 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gabarit.cloud import from_raw
from gabarit.config import Config
from gabarit.rosbag import BagReader, discover_bags


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--every',type=int,default=100)
    parser.add_argument('--max-frames',type=int,default=60)
    parser.add_argument('--max-points',type=int,default=12000)
    args = parser.parse_args(argv)
    if min(args.every,args.max_frames,args.max_points)<1:
        parser.error('sampling values must be positive')
    cfg = Config.load(args.config)
    frames=[]
    for bag in discover_bags(args.data):
        reader=BagReader(bag)
        try:
            for index in range(0,len(reader),args.every):
                if len(frames)>=args.max_frames:break
                stamp,raw=reader.read(index)
                sensor=cfg.sensor
                cloud=from_raw(raw,sensor.forward_axis,sensor.min_range,up_axis=sensor.up_axis,
                               scan_pattern=sensor.scan_pattern,dual_returns=sensor.dual_returns,
                               elevation_resolution_deg=sensor.elevation_resolution_deg,
                               working_range=cfg.detection.max_range+5.)
                points=cloud.xyz
                if len(points)>args.max_points:
                    points=points[np.linspace(0,len(points)-1,args.max_points).astype(int)]
                frames.append(dict(bag=bag.id,frame=index,timestamp_ns=int(stamp),
                                   frame_id=raw.frame_id,obstacle=None,points=np.round(points,3).tolist()))
        finally:reader.close()
    if not frames:parser.error('no frames found')
    args.out.mkdir(parents=True,exist_ok=True)
    template={'schema_version':1,'annotation_source':'manual_raw_cloud_review',
              'note':'UNKNOWN is excluded from FP/FN. Marker objects do not by themselves establish path intrusion.',
              'sensor':cfg.to_dict()['sensor'],'bags':{}}
    for row in frames:
        template['bags'].setdefault(row['bag'],{'frames':[]})['frames'].append(
            dict(frame=row['frame'],timestamp_ns=row['timestamp_ns'],obstacle=None,reviewed=False))
    (args.out/'annotations.json').write_text(json.dumps(template,indent=2),encoding='utf-8')
    data=json.dumps(dict(frames=frames,annotations=template),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разметка сырых облаков</title>
<style>body{margin:20px;font:16px system-ui;background:#111827;color:#e2e8f0}button,select{padding:9px;margin:4px}canvas{background:#080e18;width:48%;height:440px;border:1px solid #475569}p{max-width:1100px}.unknown{color:#fbbf24}</style>
<h1>Независимая ручная проверка облаков</h1><p>Показаны сырые координаты лидара без траектории, боксов и предсказаний модели. Слева — вид сверху (вперёд/влево), справа — высота/дальность. Подтверждайте препятствие только при достаточной видимости и известном маршруте. При сомнении оставьте «Неизвестно». Отсутствие очевидного объекта не означает доказанный свободный путь.</p>
<button id="prev">←</button><select id="frame"></select><button id="next">→</button><span id="label" class="unknown"></span><div><canvas id="top" width="1000" height="600"></canvas><canvas id="side" width="1000" height="600"></canvas></div>
<button data-label="true">Есть препятствие</button><button data-label="false">Проверено: препятствия нет</button><button data-label="null">Неизвестно</button><button id="save">Скачать annotations.json</button>
<script>const data=DATA;let index=0;const $=id=>document.getElementById(id);data.frames.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.bag+' · кадр '+r.frame;$('frame').append(o)});
function entry(){let r=data.frames[index];return data.annotations.bags[r.bag].frames.find(a=>a.frame===r.frame)}
function draw(){let r=data.frames[index];$('frame').value=index;$('label').textContent=entry().obstacle===null?'НЕИЗВЕСТНО':entry().obstacle?'ЕСТЬ ПРЕПЯТСТВИЕ':'ПРОВЕРЕНО: ПРЕПЯТСТВИЯ НЕТ';for(let [id,axis,min,max]of [['top',0,-20,20],['side',2,-4,6]]){let c=$(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);g.fillStyle='#94b8d8';for(let p of r.points){let x=30+(c.width-50)*p[1]/RANGE,y=c.height-30-(p[axis]-min)/(max-min)*(c.height-50);if(x>=30&&x<c.width&&y>=0&&y<c.height)g.fillRect(x,y,1.5,1.5)}g.fillStyle='#f8fafc';g.fillText('0 → RANGE м; '+(id==='top'?'поперёк ±20 м':'высота от лидара −4…6 м'),20,20)}}
$('frame').onchange=e=>{index=Number(e.target.value);draw()};$('prev').onclick=()=>{index=Math.max(0,index-1);draw()};$('next').onclick=()=>{index=Math.min(data.frames.length-1,index+1);draw()};document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{entry().obstacle=JSON.parse(b.dataset.label);entry().reviewed=true;draw()});$('save').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data.annotations,null,2)],{type:'application/json'}));a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};draw();</script></html>'''
    page=page.replace('DATA',data).replace('RANGE',str(cfg.detection.max_range))
    (args.out/'review.html').write_text(page,encoding='utf-8')
    print(f'{len(frames)} independent raw frames -> {args.out}/review.html')
    return 0


if __name__=='__main__':raise SystemExit(main())
