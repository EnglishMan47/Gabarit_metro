"""Operator-facing status and binary protocol regressions, executed as real JS."""
from pathlib import Path
import shutil
import subprocess

import pytest

NODE = shutil.which('node')
MODULE = Path(__file__).resolve().parents[1] / 'web/static/js/viewer_state.js'


def run_js(body):
    if NODE is None:
        pytest.skip('Node.js is required for the browser-state JavaScript checks')
    program = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
const source = fs.readFileSync(process.argv[2], 'utf8');
const { coverage, decisionView, streamView, decodeFrame } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
""" + body
    result = subprocess.run([NODE, '--input-type=module', '-', str(MODULE)],
                            input=program, text=True, encoding='utf-8', capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_unknown_or_unobserved_frames_never_display_a_free_path():
    run_js("""
for (const status of ['UNKNOWN', 'DEGRADED', 'unrecognized', undefined]) {
  assert.notEqual(decisionView({status, observed_until_m: 100}).kind, 'ok');
}
assert.notEqual(decisionView({status:'CLEAR', observed_until_m:0}).kind, 'ok');
assert.equal(decisionView({status:'CLEAR', observed_until_m:100}).kind, 'ok');
assert.equal(decisionView({status:'OBSTACLE', observed_until_m:0}).kind, 'bad');
""")


def test_held_path_range_cannot_replace_current_observation_range():
    run_js("""
const frame = {status:'CLEAR', analysis_range_m:350, observed_until_m:200,
  current_observed_until_m:80, envelope_until_m:140, envelope_held:true};
assert.deepEqual(coverage(frame), {range:350,measured:80,envelope:140,held:true});
assert.equal(decisionView(frame).kind, 'warn');
assert.equal(coverage({analysis_range_m:NaN,observed_until_m:NaN}).measured, 0);
""")


def test_uncertain_far_path_warns_without_changing_measured_coverage_or_alarm_priority():
    run_js("""
const frame = {status:'CLEAR', analysis_range_m:200, observed_until_m:123,
  envelope_until_m:123, track:{uncertain_from:87}};
const view = decisionView(frame);
assert.equal(view.kind,'warn');
assert.equal(view.label,'ГАБАРИТ УТОЧНЯЕТСЯ');
assert.equal(view.uncertainFrom,87);
assert.equal(view.area.measured,123);
assert.equal(decisionView({...frame,status:'OBSTACLE'}).kind,'bad');
assert.equal(decisionView({...frame,status:'CHECK'}).label,'ПРОВЕРКА ОБЪЕКТА');
assert.equal(decisionView({...frame,track:{uncertain_from:0}}).kind,'warn');
for (const value of [null, undefined, NaN]) {
  assert.equal(decisionView({...frame,track:{uncertain_from:value}}).kind,'ok');
}
assert.equal(decisionView({...frame,track:undefined}).kind,'ok');
""")


def test_a_frozen_live_frame_or_disconnected_server_does_not_look_current():
    run_js("""
const state = {connected:true,mode:'live',playing:false,hasFrame:true,lastArrive:100,now:4000};
assert.equal(streamView(state).stale,true);
assert.equal(streamView({...state,mode:'playback'}).stale,false);
assert.equal(streamView({...state,mode:'playback',playing:true}).stale,true);
assert.equal(streamView({...state,connected:false}).stale,true);
assert.equal(streamView({...state,hasFrame:false}).label,'Ожидание облака');
""")


def test_decoder_rejects_truncation_without_using_an_old_success_status():
    run_js("""
const metadata = {status:'CLEAR',track:{s:[1,2],c:[0,0],z:[0,0]},obstacles:[],pending:[]};
const raw = new TextEncoder().encode(JSON.stringify(metadata));
const length = raw.length + (4-raw.length%4)%4;
const buffer = new ArrayBuffer(8+length+4+14);
const view = new DataView(buffer);
view.setUint32(0,0x47425231,false); view.setUint32(4,length,true);
new Uint8Array(buffer,8,length).fill(32); new Uint8Array(buffer,8,raw.length).set(raw);
view.setUint32(8+length,1,true);
new Float32Array(buffer,12+length,3).set([.2,50,1.]);
new Uint8Array(buffer,24+length,2).set([4,30]);
const decoded = decodeFrame(buffer);
assert.equal(decoded.meta.status,'CLEAR');
assert.equal(decoded.points.n,1);
assert.equal(decoded.points.xyz[1],50);
assert.equal(decoded.points.cls[0],4);
assert.equal(decoded.points.inten[0],30);
assert.throws(()=>decodeFrame(buffer.slice(0,-1)),/Неполный массив/);
assert.throws(()=>decodeFrame(new ArrayBuffer(3)),/Неполный кадр/);
view.setUint32(4,0xffffffff,true);
assert.throws(()=>decodeFrame(buffer),/Повреждён заголовок/);
""")
