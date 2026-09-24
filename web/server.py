"""Web interface of the obstacle detector.

    python -m web.server --data data/extracted/for_hackathon --port 8000

Two modes share the same page:
* playback — the server reads ROS 2 bags itself (no ROS needed), plays them
  with the full pipeline, supports seeking, speed control, synthetic obstacles
  and a fast scan of the whole ride for the timeline;
* live — the ROS 2 node pushes every processed frame (see ros2 package);
  the page then shows the stream coming from ``ros2 bag play``.
"""

from __future__ import annotations

import gabarit  # noqa: F401  (sets single-threaded BLAS before NumPy loads)

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from gabarit.config import Config
from gabarit.rosbag import discover_bags
from gabarit.synth import SHAPES
from web.session import BagSession, load_cached_scan, save_cached_scan, scan_bag

LOG = logging.getLogger("gabarit.web")
ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PROJECT = ROOT.parent


def cfg_hash(cfg: Config) -> str:
    return hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


class Journal:
    """Operator decisions (confirmed / false alarm) in JSON lines."""

    FIELDS = ["time", "bag", "frame", "track_id", "verdict", "distance_m", "lateral_m", "height_m", "synthetic", "comment"]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, entry: dict) -> dict:
        entry = {k: entry.get(k) for k in self.FIELDS}
        entry["time"] = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=self.FIELDS, delimiter=";")
        w.writeheader()
        for row in self.all():
            w.writerow({k: row.get(k) for k in self.FIELDS})
        return buf.getvalue()


class LiveHub:
    """Frames pushed by the ROS 2 node, broadcast to every live client."""

    def __init__(self) -> None:
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last: bytes | None = None
        self.frames = 0

    def publish_threadsafe(self, payload: bytes) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(asyncio.ensure_future, self._broadcast(payload))

    async def _broadcast(self, payload: bytes) -> None:
        self.last = payload
        self.frames += 1
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_bytes(payload)
            except (ConnectionError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class App:
    def __init__(self, data_root: Path | None, config_path: Path | None, live: bool = False):
        self.data_root = data_root
        self.base_cfg = Config.load(config_path)
        self.bags = {b.id: b for b in discover_bags(data_root)} if data_root else {}
        self.journal = Journal(PROJECT / "artifacts" / "operator_journal.jsonl")
        self.cache = PROJECT / "artifacts" / "scan_cache"
        self.live = LiveHub() if live else None
        self.pool = ThreadPoolExecutor(max_workers=4)

    # ---------------------------------------------------------------- http
    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html")

    async def api_bags(self, request: web.Request) -> web.Response:
        return web.json_response({
            "mode": "live" if self.live else "playback",
            "bags": [b.public() for b in self.bags.values()],
            "shapes": {k: {"kind": v[0], "size": v[1:4]} for k, v in SHAPES.items()},
            "config": self.base_cfg.to_dict(),
        })

    async def api_journal(self, request: web.Request) -> web.Response:
        return web.json_response(self.journal.all())

    async def api_journal_csv(self, request: web.Request) -> web.Response:
        return web.Response(text=self.journal.csv(), content_type="text/csv", charset="utf-8",
                            headers={"Content-Disposition": "attachment; filename=operator_journal.csv"})

    async def api_feedback(self, request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(self.journal.add(body))

    async def api_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "bags": len(self.bags), "live": bool(self.live)})

    # ------------------------------------------------------------------ ws
    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        if self.live is not None:
            return await self._ws_live(ws)
        return await self._ws_playback(ws)

    async def _ws_live(self, ws: web.WebSocketResponse) -> web.WebSocketResponse:
        self.live.loop = asyncio.get_running_loop()
        self.live.clients.add(ws)
        await ws.send_json({"type": "hello", "mode": "live"})
        if self.live.last:
            await ws.send_bytes(self.live.last)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("cmd") == "feedback":
                    await ws.send_json({"type": "journal", "entry": self.journal.add(data)})
        self.live.clients.discard(ws)
        return ws

    async def _ws_playback(self, ws: web.WebSocketResponse) -> web.WebSocketResponse:
        loop = asyncio.get_running_loop()
        state: dict[str, Any] = {"session": None, "playing": False, "speed": 1.0, "task": None,
                                 "scan_stop": False, "cfg": self.base_cfg}
        send_lock = asyncio.Lock()

        async def send_json(obj: dict) -> None:
            async with send_lock:
                if not ws.closed:
                    await ws.send_json(obj)

        async def send_frame(sess: BagSession) -> dict | None:
            out = await loop.run_in_executor(self.pool, sess.step)
            if out is None:
                return None
            payload, meta = out
            async with send_lock:
                if not ws.closed:
                    await ws.send_bytes(payload)
            return meta

        async def player() -> None:
            try:
                while state["playing"] and state["session"] is not None:
                    sess = state["session"]
                    t0 = time.perf_counter()
                    meta = await send_frame(sess)
                    if meta is None:
                        state["playing"] = False
                        await send_json({"type": "state", "playing": False, "ended": True})
                        break
                    period = 0.1 / max(state["speed"], 0.05)
                    await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))
            except (ConnectionError, RuntimeError) as err:  # client went away
                LOG.info("player stopped: %s", err)

        def stop_player() -> None:
            state["playing"] = False
            task = state.get("task")
            if task is not None and not task.done():
                task.cancel()

        async def open_bag(bag_id: str, index: int = 0) -> None:
            stop_player()
            info = self.bags.get(bag_id)
            if info is None:
                await send_json({"type": "error", "message": f"нет такой записи: {bag_id}"})
                return
            if state["session"] is not None:
                state["session"].close()
            sess = BagSession(info, state["cfg"])
            state["session"] = sess
            await send_json({"type": "opened", "bag": info.public(), "frames": sess.frames})
            cached = load_cached_scan(self.cache, info, cfg_hash(state["cfg"]))
            if cached:
                await send_json({"type": "timeline", "items": cached, "total": sess.frames, "done": True})
            await loop.run_in_executor(self.pool, sess.seek, index)
            await send_frame(sess)

        async def scan() -> None:
            sess = state["session"]
            if sess is None:
                return
            info, cfg = sess.info, state["cfg"]
            state["scan_stop"] = False
            q: asyncio.Queue = asyncio.Queue()

            def progress(items, total):
                loop.call_soon_threadsafe(q.put_nowait, (list(items[-10:]), total, len(items)))

            fut = loop.run_in_executor(self.pool, lambda: scan_bag(info, cfg, progress, lambda: state["scan_stop"]))
            while not fut.done():
                try:
                    items, total, n = await asyncio.wait_for(q.get(), 0.5)
                    await send_json({"type": "timeline", "items": items, "total": total, "done": False, "n": n})
                except asyncio.TimeoutError:
                    pass
            items = await fut
            if len(items) == len(sess.reader):
                save_cached_scan(self.cache, info, cfg_hash(cfg), items)
            await send_json({"type": "timeline", "items": items, "total": len(sess.reader), "done": True})

        await send_json({"type": "hello", "mode": "playback", "bags": [b.public() for b in self.bags.values()]})
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                cmd = data.get("cmd")
                sess: BagSession | None = state["session"]
                try:
                    if cmd == "open":
                        await open_bag(data["bag"], int(data.get("index", 0)))
                    elif cmd == "play" and sess is not None:
                        if not state["playing"]:
                            state["playing"] = True
                            state["task"] = asyncio.ensure_future(player())
                        await send_json({"type": "state", "playing": True})
                    elif cmd == "pause":
                        stop_player()
                        await send_json({"type": "state", "playing": False})
                    elif cmd == "speed":
                        state["speed"] = float(data.get("value", 1.0))
                    elif cmd == "step" and sess is not None:
                        stop_player()
                        target = sess.next_index - 1 + int(data.get("delta", 1))
                        if int(data.get("delta", 1)) != 1:
                            await loop.run_in_executor(self.pool, sess.seek, target)
                        await send_frame(sess)
                    elif cmd == "seek" and sess is not None:
                        was = state["playing"]
                        stop_player()
                        await loop.run_in_executor(self.pool, sess.seek, int(data["index"]))
                        await send_frame(sess)
                        if was:
                            state["playing"] = True
                            state["task"] = asyncio.ensure_future(player())
                    elif cmd == "add_obstacle" and sess is not None:
                        ob = sess.add_obstacle(data.get("shape", "person"), float(data.get("s", 100)),
                                               float(data.get("d", 0.0)))
                        await send_json({"type": "synthetic", "added": ob})
                        if not state["playing"]:
                            await send_frame(sess)
                    elif cmd == "clear_obstacles" and sess is not None:
                        sess.clear_obstacles()
                        await send_json({"type": "synthetic", "cleared": True})
                    elif cmd == "config":
                        cfg = self.base_cfg.with_overrides(data.get("overrides") or {})
                        state["cfg"] = cfg
                        if sess is not None:
                            await loop.run_in_executor(self.pool, sess.set_config, cfg)
                            if not state["playing"]:
                                await send_frame(sess)
                        await send_json({"type": "config", "config": cfg.to_dict()})
                    elif cmd == "scan":
                        asyncio.ensure_future(scan())
                    elif cmd == "scan_stop":
                        state["scan_stop"] = True
                    elif cmd == "feedback":
                        await send_json({"type": "journal", "entry": self.journal.add(data)})
                except (KeyError, ValueError, TypeError) as err:
                    await send_json({"type": "error", "message": str(err)})
        finally:
            stop_player()
            state["scan_stop"] = True
            if state["session"] is not None:
                state["session"].close()
        return ws

    def make(self) -> web.Application:
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_get("/", self.index)
        app.router.add_get("/api/bags", self.api_bags)
        app.router.add_get("/api/health", self.api_health)
        app.router.add_get("/api/journal", self.api_journal)
        app.router.add_get("/api/journal.csv", self.api_journal_csv)
        app.router.add_post("/api/feedback", self.api_feedback)
        app.router.add_get("/ws", self.ws)
        app.router.add_static("/static/", STATIC, show_index=False)
        return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Gabarit web interface")
    ap.add_argument("--data", type=Path, default=PROJECT / "data" / "extracted" / "for_hackathon",
                    help="directory with ROS 2 bags (searched recursively)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.data.exists():
        LOG.warning("data directory %s does not exist: no bags to play", args.data)
    app = App(args.data, args.config)
    LOG.info("found %d bags in %s", len(app.bags), args.data)
    print(f"\n  Gabarit: откройте http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}\n")
    web.run_app(app.make(), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
