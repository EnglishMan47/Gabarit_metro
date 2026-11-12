#!/usr/bin/env python3
"""Measure the running ROS + RViz + web detector at the bag's original rate.

Run inside the workspace container. No input downsampling or playback slowdown
is applied. Startup and steady-state timings are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import struct
import hashlib
import asyncio
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def header_stamps(reader, limit=None):
    """Select a chronological prefix even when split files interleave in time."""
    def prefixes():
        selected=reader._ids if limit is None else reader._ids[:limit]
        if all(hasattr(connection,'blobopen') for connection in reader._cons):
            # Python >=3.11 can read 12 bytes without materializing an entire
            # multi-megabyte BLOB. The fallback also supports Humble/Python3.10.
            for _,connection,index in selected:
                with reader._cons[connection].blobopen('messages','data',index,readonly=True) as blob:
                    yield blob.read(12)
        elif limit is not None:
            for _,connection,index in selected:
                yield reader._cons[connection].execute(
                    'SELECT substr(data,1,12) FROM messages WHERE id=?',(index,)).fetchone()[0]
        else:
            for connection in reader._cons:
                query=("SELECT substr(m.data,1,12) FROM messages m JOIN topics t ON t.id=m.topic_id "
                       "WHERE t.name=? AND t.type='sensor_msgs/msg/PointCloud2'")
                for (prefix,) in connection.execute(query,(reader.info.topic,)):yield prefix
    stamps=[]
    for prefix in prefixes():
        endian='<' if prefix[:2]==b'\x00\x01' else '>'
        sec,nsec=struct.unpack_from(endian+'iI',prefix,4)
        stamps.append(sec*1_000_000_000+nsec)
    return np.sort(np.array(stamps,dtype=np.int64))


def summarize(rows, warmup, expected_ns):
    steady = rows[warmup:]
    if len(steady) < 2:
        raise ValueError('Too few steady-state detector messages')
    arrivals = np.array([r[0] for r in steady])
    stamps = np.array([r[1]['stamp'] for r in steady])
    lo = np.searchsorted(expected_ns, round(stamps[0]*1e9)-5000)
    hi = np.searchsorted(expected_ns, round(stamps[-1]*1e9)+5000, side='right')
    expected = int(hi-lo)
    if not expected:
        raise ValueError('Detector timestamps do not match any source frame')
    source_stamps = expected_ns[lo:hi]
    received_ns = np.rint(stamps*1e9).astype(np.int64)
    indices = np.searchsorted(source_stamps, received_ns)
    right = np.minimum(indices, len(source_stamps)-1)
    left = np.maximum(indices-1, 0)
    indices = np.where(abs(source_stamps[left]-received_ns) < abs(source_stamps[right]-received_ns), left, right)
    matched = indices[abs(source_stamps[indices]-received_ns) <= 5000]
    unique = len(np.unique(matched))
    missing = expected-unique
    duplicates = len(matched)-unique
    timing = [r[1].get('ros_timing_ms', {}).get('total', r[1].get('latency_ms')) for r in steady]
    timing = np.asarray([t for t in timing if t is not None], dtype=float)
    phases = {name: {'p50':float(np.percentile([r[1].get('ros_timing_ms',{}).get(name,0) for r in steady],50)),
                     'p95':float(np.percentile([r[1].get('ros_timing_ms',{}).get(name,0) for r in steady],95))}
              for name in ('input_decode','pipeline','outputs','web_pack')}
    hz = (len(steady)-1)/(arrivals[-1]-arrivals[0])
    expected_hz = (expected-1)/(stamps[-1]-stamps[0])
    intact = missing==duplicates==0 and len(matched)==len(steady) and np.all(np.diff(stamps)>0)
    budget_met = bool(np.percentile(timing,95)<=100)
    gaps = np.diff(source_stamps)*1e-9
    return dict(total_messages=len(rows), warmup_messages=warmup, steady_messages=len(steady),
                received_hz=float(hz), source_hz=float((len(steady)-1)/(stamps[-1]-stamps[0])),
                expected_source_hz=float(expected_hz),
                recorded_gap_count=int(np.sum(gaps>1.5*np.median(gaps))),
                max_recorded_gap_ms=float(np.max(gaps)*1000),
                expected_source_messages=expected, missing_source_messages=missing,
                duplicate_source_messages=duplicates, unmatched_source_messages=len(steady)-len(matched),
                callback_p95_ms=float(np.percentile(timing,95)), callback_p99_ms=float(np.percentile(timing,99)),
                callback_deadline_met_share=float(np.mean(timing<=100)), phases_ms=phases,
                latency_10hz_budget_pass=budget_met,
                source_realtime_pass=bool(intact and hz>=.98*expected_hz and budget_met),
                steady_10hz_pass=bool(hz>=9.8 and intact and budget_met),
                first_stamp=float(stamps[0]), last_stamp=float(stamps[-1]))


class WebProbe:
    """Keep a real live viewer connected while the ROS pipeline is measured."""
    def __init__(self, url):
        self.url=url;self.frames=[];self.error=None
        self.ready=threading.Event();self.stop=threading.Event()
        self.thread=threading.Thread(target=lambda:asyncio.run(self.receive()),daemon=True)

    async def receive(self):
        from aiohttp import ClientSession, WSMsgType
        try:
            async with ClientSession() as session:
                async with session.ws_connect(self.url,max_msg_size=64*1024*1024) as ws:
                    self.ready.set()
                    while not self.stop.is_set():
                        try:message=await ws.receive(timeout=.2)
                        except asyncio.TimeoutError:continue
                        if message.type==WSMsgType.BINARY:self.frames.append(time.monotonic())
                        elif message.type in (WSMsgType.CLOSED,WSMsgType.ERROR):
                            raise RuntimeError('Live web viewer disconnected')
        except Exception as error:
            self.error=str(error);self.ready.set()

    def start(self):
        self.thread.start()
        if not self.ready.wait(10) or self.error:
            raise RuntimeError(f'Live web viewer failed: {self.error or "timeout"}')

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:self.thread.join(timeout=3)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bag',type=Path)
    ap.add_argument('--config',default='/opt/gabarit/config/gabarit.yaml')
    ap.add_argument('--frames',type=int,default=180)
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--timeout',type=float,default=90)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--require-10hz',action='store_true')
    ap.add_argument('--web-url',default='http://127.0.0.1:8080/ws')
    ap.add_argument('--no-cache',action='store_true',help='stream an oversized recording from its original mount')
    ap.add_argument('--header-frames',type=int,help='inspect only this storage prefix; fail if playback exceeds it')
    args=ap.parse_args()
    if args.frames<args.warmup+2 or args.warmup<0 or args.timeout<=0:ap.error('invalid sample duration')
    if args.header_frames is not None and args.header_frames<args.frames:ap.error('--header-frames must cover --frames')
    import rclpy
    from std_msgs.msg import String
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from gabarit.rosbag import BagReader, discover_bags
    bags=discover_bags(args.bag)
    if len(bags)!=1:ap.error('bag must identify exactly one recording')
    playback=subprocess.run(['gabarit','bag','play',str(bags[0].path.parent),'--config',str(Path(args.config).resolve()),
                             *([] if args.no_cache else ['--cache']),'--paused'],
                            capture_output=True,text=True)
    if playback.returncode:raise RuntimeError(playback.stderr.strip() or 'Could not start benchmark playback')
    playback_state=json.loads(playback.stdout)
    cached=discover_bags(Path(playback_state.get('playback_bag',str(args.bag))))
    if len(cached)!=1:raise ValueError('Cannot inspect cached playback recording')
    reader=BagReader(cached[0])
    # Bag storage time and device header time may differ by years. Compare
    # against actual input header stamps, reading only 12 CDR header bytes.
    version=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in (cached[0].storage_paths or (cached[0].path,))]
    key=hashlib.sha256(json.dumps([version,cached[0].topic,args.header_frames,
                                 'chronological-prefix' if args.header_frames is not None else None]).encode()).hexdigest()[:16]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stamp_cache=args.out.parent/f'header-times-{key}.npy'
    if stamp_cache.is_file():
        expected_ns=np.load(stamp_cache,allow_pickle=False)
    else:
        expected_ns=header_stamps(reader,args.header_frames)
        np.save(stamp_cache,expected_ns,allow_pickle=False)
    inspected=min(len(reader),args.header_frames) if args.header_frames is not None else len(reader)
    if expected_ns.ndim!=1 or len(expected_ns)!=inspected:raise ValueError('Header timing cache does not match recording')
    reader.close()
    rclpy.init();node=rclpy.create_node('gabarit_realtime_benchmark')
    rows=[];views=[];markers=[]
    node.create_subscription(String,'/gabarit/status',lambda m:rows.append((time.monotonic(),json.loads(m.data))),
                             QoSProfile(depth=300,reliability=ReliabilityPolicy.RELIABLE))
    node.create_subscription(PointCloud2,'/gabarit/view_cloud',lambda m:views.append(1),
                             QoSProfile(depth=4,reliability=ReliabilityPolicy.BEST_EFFORT),raw=True)
    node.create_subscription(MarkerArray,'/gabarit/view_markers',lambda m:markers.append(1),10,raw=True)
    viewer=WebProbe(args.web_url)
    try:
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        viewer.start()
        subprocess.run(['gabarit','bag','resume'],check=True,stdout=subprocess.DEVNULL)
        deadline=time.monotonic()+args.timeout
        while len(rows)<args.frames and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if rows and time.monotonic()-rows[-1][0]>2:
                state=subprocess.run(['gabarit','bag','status'],capture_output=True,text=True)
                if state.returncode==0 and not json.loads(state.stdout).get('running',False):break
        # Keep observing while the CLI pauses the player; otherwise its ROS
        # startup can hide final visualization messages from the counters.
        pause=subprocess.Popen(['gabarit','bag','pause'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        pause_deadline=time.monotonic()+10
        while pause.poll() is None and time.monotonic()<pause_deadline:
            rclpy.spin_once(node,timeout_sec=.05)
        if pause.poll() is None:
            pause.terminate();pause.wait(timeout=3)
            raise RuntimeError('Timed out pausing benchmark playback')
        deadline=time.monotonic()+.3
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.02)
        if rows and round(rows[-1][1]['stamp']*1e9)>int(expected_ns[-1])+5000:
            raise ValueError('Playback exceeded inspected headers; increase --header-frames')
        report=summarize(rows,args.warmup,expected_ns)
        args.out.with_suffix('.frames.jsonl').write_text(''.join(json.dumps(dict(arrival=t,**d))+'\n'
            for t,d in rows),encoding='utf-8')
        report.update(bag=str(args.bag),view_cloud_messages=len(views),marker_messages=len(markers),
                      rviz_running=subprocess.run(['pgrep','-f','/rviz2'],capture_output=True).returncode==0,
                      playback_rate=1.,startup_callback_ms=[r[1].get('latency_ms') for r in rows[:args.warmup]])
        report.update(source_cached=bool(playback_state.get('cached')),inspected_source_headers=len(expected_ns))
        report.update(web_frame_messages=len(viewer.frames),web_viewer_error=viewer.error,
                      web_viewer_connected=viewer.ready.is_set() and viewer.error is None)
        report['steady_10hz_pass'] = bool(report['steady_10hz_pass'] and report['rviz_running']
            and report['web_viewer_connected'] and len(viewer.frames)>=len(rows)-args.warmup-2
            and len(views)>=len(rows)-args.warmup-2 and len(markers)>=len(rows)-args.warmup-2)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
        return int(not report['steady_10hz_pass']) if args.require_10hz else 0
    finally:
        viewer.close()
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
