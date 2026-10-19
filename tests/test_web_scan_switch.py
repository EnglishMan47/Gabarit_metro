"""Real WebSocket regression for a late scan worker after switching bags.

The worker and bag session are in-memory fakes; no recordings or cache files
are opened. The HTTP/WebSocket handler and its executor are the production
ones, including cancellation that cannot stop an already running thread.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from gabarit.config import Config
from web.admin import Access
from web import server


def test_late_scan_cannot_publish_or_cache_old_bag_after_switch(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cancelled = []
    saved = []

    class Info:
        def __init__(self, ident):
            self.id = self.name = ident

        def public(self):
            return {"id": self.id, "name": self.name}

    class Session:
        def __init__(self, info, cfg):
            self.info, self.cfg = info, cfg
            self.reader = [None, None]
            self.frames = 2
            self.next_index = 0
            self.closed = False

        def close(self):
            self.closed = True

        def seek(self, index):
            self.next_index = index

        def step(self):
            assert not self.closed
            self.next_index += 1
            return self.info.id.encode(), {"frame": self.next_index - 1, "bag": self.info.id}

    def scan(info, cfg, progress, stop):
        items = [{"i": i, "status": "CLEAR", "ids": []} for i in range(2)]
        if info.id == "a":
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("test failed to release the old scan worker")
            cancelled.append(stop())
            # Deliberately finish after a switch, as a real in-flight frame can.
            progress(items, 2)
            finished.set()
        else:
            progress(items, 2)
        return items

    monkeypatch.setattr(server, "BagSession", Session)
    monkeypatch.setattr(server, "scan_bag", scan)
    monkeypatch.setattr(server, "load_cached_scan", lambda *a: None)
    monkeypatch.setattr(server, "save_cached_scan", lambda cache, info, cfg, items: saved.append(info.id))
    monkeypatch.setattr(server, "sensor_profiles", lambda cfg: {})

    async def scenario():
        application = object.__new__(server.App)
        application.live = None
        application.base_cfg = Config()
        application.bags = {key: Info(key) for key in ("a", "b")}
        application.uploaded = set()
        application.access = Access()
        application.cache = Path("unused-in-memory-cache")
        application.pool = ThreadPoolExecutor(max_workers=4)
        webapp = web.Application()
        webapp.router.add_get("/ws", application.ws)

        async def next_json(ws, kind, *, bag=None):
            while True:
                message = await ws.receive(timeout=3)
                if message.type == WSMsgType.TEXT:
                    value = message.json()
                    if value.get("type") == kind and (bag is None or value.get("bag", {}).get("id") == bag):
                        return value
                elif message.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    raise AssertionError("WebSocket closed before the expected response")

        try:
            async with TestClient(TestServer(webapp)) as client:
                ws = await client.ws_connect("/ws")
                await next_json(ws, "hello")
                await ws.send_json({"cmd": "open", "bag": "a"})
                await next_json(ws, "opened", bag="a")
                assert (await ws.receive(timeout=3)).type == WSMsgType.BINARY
                await ws.send_json({"cmd": "scan"})
                assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=3)
                await ws.send_json({"cmd": "open", "bag": "b"})
                await next_json(ws, "opened", bag="b")
                assert (await ws.receive(timeout=3)).type == WSMsgType.BINARY
                release.set()
                assert await asyncio.wait_for(asyncio.to_thread(finished.wait), timeout=3)
                # The production scan coroutine waits up to 0.5 s for progress.
                # Observe beyond that deadline so a late final message is caught.
                try:
                    message = await ws.receive(timeout=0.7)
                except asyncio.TimeoutError:
                    pass
                else:
                    assert message.type != WSMsgType.TEXT or message.json().get("type") != "timeline"
                assert cancelled == [True]
                assert "a" not in saved
                # The new session can still run and publish its own scan.
                await ws.send_json({"cmd": "scan"})
                timeline = await next_json(ws, "timeline")
                assert timeline["bag"] == "b"
                await ws.close()
        finally:
            release.set()
            application.pool.shutdown(wait=True)

    asyncio.run(scenario())
