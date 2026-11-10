import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gabarit.config import Config
from gabarit.detect import Candidate
from gabarit.pipeline import _sensor_distances
from gabarit.rosbag import BagInfo
from web.session import load_cached_scan, save_cached_scan
from web import server
from web.session import pack_frame
from types import SimpleNamespace as NS
import struct


def info(path, ident):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'first recording')
    return BagInfo(ident, 'same_name', path, '/points', 1, 2, 10, 20)


def test_scan_cache_distinguishes_equal_bag_names_and_modified_storage(tmp_path):
    a = info(tmp_path / 'a' / 'part.db3', 'a/same_name')
    b = info(tmp_path / 'b' / 'part.db3', 'b/same_name')
    cache = tmp_path / 'cache'
    first = [{'i': 0, 'status': 'CLEAR'}, {'i': 1, 'status': 'CHECK'}]
    save_cached_scan(cache, a, 'algorithm1', first)
    assert load_cached_scan(cache, a, 'algorithm1') == first
    assert load_cached_scan(cache, b, 'algorithm1') is None
    assert load_cached_scan(cache, a, 'algorithm2') is None
    a.path.write_bytes(b'replaced recording with different clouds')
    assert load_cached_scan(cache, a, 'algorithm1') is None
    assert not list(cache.glob('*.tmp'))


def test_malformed_or_partial_scan_is_not_analyzed_timeline(tmp_path):
    recording = info(tmp_path / 'bag' / 'part.db3', 'bag')
    cache = tmp_path / 'cache'
    save_cached_scan(cache, recording, 'v1', [{'i': 0}])
    assert load_cached_scan(cache, recording, 'v1') is None
    save_cached_scan(cache, recording, 'v1', [{'i': 1}, {'i': 0}])
    assert load_cached_scan(cache, recording, 'v1') is None


def test_detector_source_update_invalidates_timeline_even_with_same_config(tmp_path, monkeypatch):
    (tmp_path / 'gabarit').mkdir()
    source = tmp_path / 'gabarit' / 'detect.py'
    source.write_text('algorithm version one')
    monkeypatch.setattr(server, 'PROJECT', tmp_path)
    before = server.cfg_hash(Config())
    source.write_text('algorithm version two')
    assert server.cfg_hash(Config()) != before


def test_reported_range_uses_actual_sensor_origin_and_current_member_indices():
    candidate = Candidate(1, 5., 5.5, -.2, .2, .2, 1.5, 99., 2, 2, 0., .9,
                          point_index=np.array([0, 1]))
    # Same physical returns: translated leveling would put z at +1 and +2.
    sensor = np.array([[0., 5., -1.], [.2, 5.5, 0.]], np.float32)
    held = replace(candidate, distance=8., point_index=None)
    _sensor_distances([candidate, held], sensor)
    assert candidate.distance == pytest.approx(np.sqrt(26.))
    assert held.distance == 8.


def test_frame_distinguishes_measured_range_from_uncertain_axis_and_held_tail():
    track = NS(s=np.array([3., 60., 120.]), c=np.zeros(3), z=np.zeros(3),
               sig_c=np.array([.03, .4, .8]), sig_z=np.full(3, .02),
               wall_left=np.full(3,np.nan), wall_right=np.full(3,np.nan),
               observed_until=120., current_observed_until=80., coast_frames=2, rail_until=40.)
    result = NS(lev=np.array([[0.,20.,1.]], np.float32), labels=np.array([4], np.uint8),
                cloud=NS(intensity=np.array([20.])), track=track)
    payload = pack_frame(result, {'analysis_range_m':200.})
    length = struct.unpack_from('<I', payload, 4)[0]
    meta = json.loads(payload[8:8+length])
    assert meta['track']['observed_until'] == 120.
    assert meta['track']['current_observed_until'] == 80.
    assert meta['track']['uncertain_from'] == 60.
    assert meta['track']['sig_c'] == [.03,.4,.8]
