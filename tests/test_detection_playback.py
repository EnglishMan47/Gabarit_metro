"""Confirmed detections remain visible while playback follows user controls."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace as NS

import numpy as np

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from gabarit.config import Config
from web.admin import Access
from web import server
from web.session import pack_frame


def test_web_frame_preserves_falcon_returns_at_330m():
    track = NS(s=np.array([3., 350.]), c=np.zeros(2), z=np.zeros(2),
               wall_left=np.full(2, np.nan), wall_right=np.full(2, np.nan),
               observed_until=340., current_observed_until=340., coast_frames=0, rail_until=60.)
    res = NS(lev=np.array([[0., 330., 1.], [0., 20., 1.]], dtype=np.float32),
             labels=np.array([4, 0], dtype=np.uint8), track=track,
             cloud=NS(intensity=np.array([40., 10.])))
    payload = pack_frame(res, {'analysis_range_m': 350.})
    length = struct.unpack_from('<I', payload, 4)[0]
    meta = json.loads(payload[8:8 + length])
    count = struct.unpack_from('<I', payload, 8 + length)[0]
    points = np.frombuffer(payload, dtype='<f4', offset=12 + length, count=count * 3).reshape(-1, 3)
    assert meta['analysis_range_m'] == 350.
    assert count == 1 and points[:, 1].max() == 330.


def test_obstacle_does_not_change_playback_or_block_manual_controls(monkeypatch):
    class Info:
        id = name = 'obstacle'

        def public(self):
            return {'id': self.id, 'name': self.name}

    class Session:
        def __init__(self, info, cfg):
            self.info, self.cfg = info, cfg
            self.frames = 200
            self.next_index = 0

        def close(self):
            pass

        def seek(self, index):
            self.next_index = index

        def step(self):
            meta = {'frame': self.next_index, 'status': 'OBSTACLE', 'obstacle': True,
                    'obstacles': [{'track_id': 7, 'distance_m': 2., 'lateral_m': 0.}],
                    'nearest_track_distance_m': 2., 'speed_mps': 15., 'displacement_m': 1.5}
            self.next_index += 1
            return json.dumps(meta).encode(), meta

    monkeypatch.setattr(server, 'BagSession', Session)
    monkeypatch.setattr(server, 'load_cached_scan', lambda *a: None)
    monkeypatch.setattr(server, 'sensor_profiles', lambda cfg: {})

    async def scenario():
        app = object.__new__(server.App)
        app.live = None
        app.base_cfg = Config()
        app.bags = {'obstacle': Info()}
        app.uploaded = set()
        app.access = Access()
        app.cache = Path('unused-in-memory-cache')
        app.pool = ThreadPoolExecutor(max_workers=2)
        webapp = web.Application()
        webapp.router.add_get('/ws', app.ws)

        async def receive(ws, kind):
            message = await ws.receive(timeout=2)
            if kind == 'frame':
                assert message.type == WSMsgType.BINARY, message
                meta = json.loads(message.data)
                assert meta['status'] == 'OBSTACLE' and meta['obstacles'][0]['track_id'] == 7
                return meta
            assert message.type == WSMsgType.TEXT, message
            data = message.json()
            assert data['type'] == kind, data
            return data

        try:
            async with TestClient(TestServer(webapp)) as client:
                ws = await client.ws_connect('/ws')
                hello = await receive(ws, 'hello')
                assert 'braking_deceleration' not in hello['config']
                await ws.send_json({'cmd': 'open', 'bag': 'obstacle'})
                assert not (await receive(ws, 'state'))['playing']
                await receive(ws, 'opened')
                assert (await receive(ws, 'frame'))['frame'] == 0
                await ws.send_json({'cmd': 'speed', 'value': 4})
                await ws.send_json({'cmd': 'play'})
                assert (await receive(ws, 'state'))['playing']
                start = time.monotonic()
                for i in range(1, 17):
                    assert (await receive(ws, 'frame'))['frame'] == i
                # The 4x rate stays selected throughout the persistent alarm.
                assert time.monotonic() - start < 1.2
                await ws.send_json({'cmd': 'pause'})
                # Drain at most the in-flight worker frame before pause acknowledgement.
                while True:
                    message = await ws.receive(timeout=2)
                    if message.type == WSMsgType.TEXT:
                        assert message.json() == {'type': 'state', 'playing': False}
                        break
                    assert message.type == WSMsgType.BINARY
                try:
                    await ws.receive(timeout=.15)
                except asyncio.TimeoutError:
                    pass
                else:
                    raise AssertionError('paused player emitted another frame')
                await ws.send_json({'cmd': 'seek', 'index': 50})
                assert (await receive(ws, 'frame'))['frame'] == 50
                await ws.send_json({'cmd': 'step', 'delta': 1})
                assert (await receive(ws, 'frame'))['frame'] == 51
                await ws.send_json({'cmd': 'play'})
                assert (await receive(ws, 'state'))['playing']
                assert (await receive(ws, 'frame'))['frame'] == 52
                await ws.close()
        finally:
            app.pool.shutdown(wait=True)

    asyncio.run(scenario())
