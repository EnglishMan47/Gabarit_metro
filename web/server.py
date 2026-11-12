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
import dataclasses
import hashlib
import io
import json
import logging
import os
import shutil
import socket
import sqlite3
import threading
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from aiohttp import WSMsgType, web

from gabarit import mlops
from gabarit.config import Config
from gabarit.rosbag import discover_bags
from gabarit.synth import SHAPES
from web.admin import (PRESETS, UPLOAD_GROUP, Access, check_admin, config_yaml, mlflow_status, parse_config,
                       receive_upload)
from web.session import BagSession, load_cached_scan, save_cached_scan, scan_bag

LOG = logging.getLogger("gabarit.web")
ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PROJECT = ROOT.parent


def cfg_hash(cfg: Config) -> str:
    digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode())
    # Config alone cannot identify detector output after a source update.
    for path in sorted((PROJECT / "gabarit").rglob("*.py")):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    paths = ([cfg.models.far_path_weights] if cfg.models.learned_path else [])
    if cfg.models.component_weights:
        paths.append(cfg.models.component_weights)
    for model_path in paths:
        weights = Path(model_path)
        if not weights.is_absolute():
            weights = PROJECT / weights
        if weights.is_file():
            stat = weights.stat()
            digest.update(f"{weights.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def sensor_profiles(cfg: Config) -> dict:
    """Explicit conventions: bag frame names alone cannot determine axes."""
    profiles = {"current": {"title": "Из конфигурации запуска", "sensor": dataclasses.asdict(cfg.sensor),
                            "max_range_m": min(cfg.track.horizon, cfg.detection.max_range)},
                "pandar": {"title": "Pandar · вперёд −Y, вверх +Z", "sensor": dataclasses.asdict(Config().sensor),
                           "max_range_m": 200.0}}
    for key, title in (("falcon_prime", "Falcon native · вперёд +Z, вверх +X"),
                       ("falcon_prime_ros", "Falcon ROS · вперёд +X, вверх +Z")):
        path = PROJECT / "config" / (key + ".yaml")
        if path.exists():
            profile = Config.load(path)
            profiles[key] = {"title": title, "sensor": dataclasses.asdict(profile.sensor),
                             "max_range_m": min(profile.track.horizon, profile.detection.max_range)}
    return profiles


class Journal:
    """Read-only archive of annotations created by older versions."""

    FIELDS = ["time", "bag", "frame", "track_id", "verdict", "distance_m", "lateral_m", "height_m", "synthetic", "comment"]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._pending = {}
        self._senders = {}
        self._generation = 0
        self._packing = False

    def clear(self):
        with self._frame_lock:
            self.last = self._latest_frame = None
            self._generation += 1

    def publish_frame_threadsafe(self, result, meta) -> None:
        # No viewer: retain one immutable frame for a paused-stream newcomer,
        # without serializing a large point payload in the ROS callback.
        with self._frame_lock:
            self._latest_frame = (result, dict(meta))
            self.last = None
            self._generation += 1
            schedule = bool(self.clients and self.loop is not None and not self._packing)
            if schedule:
                self._packing = True
        if schedule:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self._pack_latest()))

    async def _pack_latest(self):
        # A single encoder owns at most one in-flight frame and one latest
        # replacement. Rendering never serializes the ROS callback or queues
        # an unbounded history when a viewer/network falls behind.
        from web.session import pack_frame
        try:
            while True:
                with self._frame_lock:
                    frame, generation = self._latest_frame, self._generation
                    if frame is None or not self.clients:
                        self._packing = False
                        return
                payload = await asyncio.get_running_loop().run_in_executor(None, pack_frame, *frame)
                await self._broadcast(payload, generation)
                with self._frame_lock:
                    if generation == self._generation:
                        self._packing = False
                        return
        except BaseException:
            with self._frame_lock:
                self._packing = False
            raise

    def latest_payload(self):
        with self._frame_lock:
            payload, frame = self.last, self._latest_frame
        if payload is not None:
            return payload
        if frame is not None:
            from web.session import pack_frame
            return pack_frame(*frame)
        return None

    def publish_threadsafe(self, payload: bytes, generation=None) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(asyncio.ensure_future, self._broadcast(payload, generation))

    async def _broadcast(self, payload: bytes, generation=None) -> None:
        with self._frame_lock:
            if generation is not None and generation != self._generation:
                return
            self.last = payload
        self.frames += 1
        for ws in list(self.clients):
            self._pending[ws] = payload
            if ws not in self._senders:
                self._senders[ws] = asyncio.create_task(self._send_latest(ws))

    async def _send_latest(self, ws):
        try:
            while ws in self.clients and ws in self._pending:
                payload = self._pending.pop(ws)
                await asyncio.wait_for(ws.send_bytes(payload), timeout=2.0)
        except (ConnectionError, RuntimeError, asyncio.TimeoutError):
            self.clients.discard(ws)
        finally:
            self._pending.pop(ws, None)
            self._senders.pop(ws, None)


class App:
    def __init__(self, data_root: Path | list[Path] | None, config_path: Path | None, live: bool = False,
                 upload_root: Path | None = None, max_users: int = 1, idle_s: float = 600.0,
                 max_upload_gb: float = 20.0):
        roots = [] if data_root is None else [data_root] if isinstance(data_root, Path) else list(data_root)
        self.roots = roots
        self.data_root = roots[0] if roots else None
        self.upload_root = upload_root
        self.base_cfg = Config.load(config_path)
        self.access = Access(max_users, idle_s)
        self.max_upload = int(max_upload_gb * 2 ** 30)
        self.started = time.time()
        self.jobs: dict[str, dict] = {}
        self.versions_file = PROJECT / "artifacts" / "admin" / "versions.json"
        self.mlflow_proc: subprocess.Popen | None = None
        self.bags: dict = {}
        self.uploaded: set[str] = set()
        self.rescan()
        self.journal = Journal(PROJECT / "artifacts" / "operator_journal.jsonl")
        self.cache = PROJECT / "artifacts" / "scan_cache"
        self.live = LiveHub() if live else None
        self.pool = ThreadPoolExecutor(max_workers=4)

    def rescan(self) -> None:
        """(Re)discover the recordings: data roots, then the uploads folder."""
        bags: dict = {}
        uploaded: set[str] = set()
        roots = [(r, False) for r in self.roots]
        if self.upload_root is not None:
            roots.append((self.upload_root, True))
        for root, up in roots:
            for b in discover_bags(root):
                if up:
                    b = dataclasses.replace(b, id=f"uploads/{b.id}")
                elif b.id in bags:   # same relative path under two roots
                    b = dataclasses.replace(b, id=f"{Path(root).name}/{b.id}")
                bags[b.id] = b
                if up:
                    uploaded.add(b.id)
        self.bags, self.uploaded = bags, uploaded

    def bags_public(self) -> list[dict]:
        out = []
        for b in self.bags.values():
            d = b.public()
            if b.id in self.uploaded:
                d["group"] = UPLOAD_GROUP
                d["uploaded"] = True
            out.append(d)
        return out

    # ---------------------------------------------------------------- http
    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html")

    async def api_bags(self, request: web.Request) -> web.Response:
        return web.json_response({
            "mode": "live" if self.live else "playback",
            "bags": self.bags_public(),
            "shapes": {k: {"kind": v[0], "size": v[1:4]} for k, v in SHAPES.items()},
            "config": self.base_cfg.to_dict(),
            "sensor_profiles": sensor_profiles(self.base_cfg),
        })

    async def api_journal(self, request: web.Request) -> web.Response:
        return web.json_response(self.journal.all())

    async def api_journal_csv(self, request: web.Request) -> web.Response:
        return web.Response(text=self.journal.csv(), content_type="text/csv", charset="utf-8",
                            headers={"Content-Disposition": "attachment; filename=operator_journal.csv"})

    async def api_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "bags": len(self.bags), "live": bool(self.live)})

    # ------------------------------------------------------------------ ws
    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        if self.live is not None:
            return await self._ws_live(ws)
        return await self._ws_playback(ws, request)

    async def _ws_live(self, ws: web.WebSocketResponse) -> web.WebSocketResponse:
        self.live.loop = asyncio.get_running_loop()
        await ws.send_json({"type": "hello", "mode": "live", "config": self.base_cfg.to_dict(),
                            "sensor_profiles": sensor_profiles(self.base_cfg)})
        payload = await asyncio.get_running_loop().run_in_executor(self.pool, self.live.latest_payload)
        if payload:
            await ws.send_bytes(payload)
        self.live.clients.add(ws)
        async for _ in ws:
            pass
        self.live.clients.discard(ws)
        return ws

    async def _ws_playback(self, ws: web.WebSocketResponse, request: web.Request) -> web.WebSocketResponse:
        loop = asyncio.get_running_loop()
        seat = await self.access.admit(ws, request)
        if seat is None:   # another user is working with the service
            await ws.send_json(self.access.busy_info())
            await ws.close()
            return ws
        state: dict[str, Any] = {"session": None, "playing": False, "speed": 1.0, "task": None,
                                 "scan_stop": False, "cfg": self.base_cfg}
        send_lock = asyncio.Lock()

        async def send_json(obj: dict) -> None:
            async with send_lock:
                if not ws.closed:
                    await ws.send_json(obj)

        async def send_frame(sess: BagSession) -> dict | None:
            out = await loop.run_in_executor(self.pool, sess.step)
            if out is None or state["session"] is not sess:
                return None
            payload, meta = out
            async with send_lock:
                if not ws.closed:
                    await ws.send_bytes(payload)
            return meta

        async def resume_play() -> None:
            if not state["playing"] and state["session"] is not None:
                state["playing"] = True
                state["task"] = asyncio.ensure_future(player())
            await send_json({"type": "state", "playing": True})

        async def player() -> None:
            try:
                while state["playing"] and state["session"] is not None:
                    sess = state["session"]
                    self.access.touch(ws)
                    t0 = time.perf_counter()
                    meta = await send_frame(sess)
                    if meta is None:
                        state["playing"] = False
                        await send_json({"type": "state", "playing": False, "ended": True})
                        break
                    speed = max(state["speed"], 0.05)
                    period = 0.1 / speed
                    await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))
            except (ConnectionError, RuntimeError) as err:  # client went away
                state["playing"] = False
                LOG.info("player stopped: %s", err)
                await send_json({"type": "state", "playing": False})
                await send_json({"type": "error", "message": f"Не удалось продолжить просмотр: {err}"})
            except (OSError, ValueError, sqlite3.Error) as err:
                state["playing"] = False
                await send_json({"type": "state", "playing": False})
                await send_json({"type": "error", "message": f"Ошибка чтения записи: {err}"})

        def stop_player() -> None:
            state["playing"] = False
            task = state.get("task")
            if task is not None and not task.done():
                task.cancel()

        async def open_bag(bag_id: str, index: int = 0) -> None:
            stop_player()
            state["scan_generation"] = state.get("scan_generation", 0) + 1
            await send_json({"type": "state", "playing": False})
            info = self.bags.get(bag_id)
            if info is None:
                await send_json({"type": "error", "message": f"нет такой записи: {bag_id}"})
                return
            sess = await loop.run_in_executor(self.pool, BagSession, info, state["cfg"])
            if state["session"] is not None:
                await loop.run_in_executor(self.pool, state["session"].close)
            state["session"] = sess
            await send_json({"type": "opened", "bag": info.public(), "frames": sess.frames})
            cached = load_cached_scan(self.cache, info, cfg_hash(state["cfg"]))
            if cached:
                await send_json({"type": "timeline", "bag": info.id, "items": cached, "total": sess.frames, "done": True})
            await loop.run_in_executor(self.pool, sess.seek, index)
            await send_frame(sess)

        async def scan() -> None:
            sess = state["session"]
            if sess is None:
                return
            info, cfg = sess.info, state["cfg"]
            generation = state["scan_generation"] = state.get("scan_generation", 0) + 1
            q: asyncio.Queue = asyncio.Queue()

            def cancelled():
                return ws.closed or generation != state.get("scan_generation") or state["session"] is not sess

            def progress(items, total):
                if not cancelled():
                    loop.call_soon_threadsafe(q.put_nowait, (list(items[-10:]), total, len(items)))

            fut = loop.run_in_executor(self.pool, lambda: scan_bag(info, cfg, progress, cancelled))
            while not fut.done():
                try:
                    items, total, n = await asyncio.wait_for(q.get(), 0.5)
                    if not cancelled():
                        await send_json({"type": "timeline", "bag": info.id, "items": items, "total": total,
                                         "done": False, "n": n})
                except asyncio.TimeoutError:
                    pass
            items = await fut
            if cancelled():
                return
            if len(items) == len(sess.reader):
                save_cached_scan(self.cache, info, cfg_hash(cfg), items)
            await send_json({"type": "timeline", "bag": info.id, "items": items, "total": len(sess.reader), "done": True})

        await send_json({"type": "hello", "mode": "playback", "bags": self.bags_public(), "token": seat["token"],
                         "config": self.base_cfg.to_dict(), "sensor_profiles": sensor_profiles(self.base_cfg),
                         "idle_min": round(self.access.idle_s / 60)})
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    await send_json({"type": "error", "message": "ожидается объект команды"})
                    continue
                cmd = data.get("cmd")
                self.access.touch(ws)
                sess: BagSession | None = state["session"]
                try:
                    if cmd == "open":
                        await open_bag(data["bag"], int(data.get("index", 0)))
                    elif cmd == "play" and sess is not None:
                        await resume_play()
                    elif cmd == "pause":
                        stop_player()
                        await send_json({"type": "state", "playing": False})
                    elif cmd == "speed":
                        speed = float(data.get("value", 1.0))
                        if not 0.05 <= speed <= 5.0:
                            raise ValueError("скорость просмотра должна быть от 0.05 до 5")
                        state["speed"] = speed
                    elif cmd == "step" and sess is not None:
                        stop_player()
                        await send_json({"type": "state", "playing": False})
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
                        ob = await loop.run_in_executor(self.pool, sess.add_obstacle, data.get("shape", "person"),
                                                        float(data.get("s", 100)), float(data.get("d", 0.0)),
                                                        float(data.get("yaw_deg", 0.0)))
                        await send_json({"type": "synthetic", "added": ob})
                        if not state["playing"]:
                            await send_frame(sess)
                    elif cmd == "clear_obstacles" and sess is not None:
                        await loop.run_in_executor(self.pool, sess.clear_obstacles)
                        await send_json({"type": "synthetic", "cleared": True})
                    elif cmd == "config":
                        state["scan_generation"] = state.get("scan_generation", 0) + 1
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
                        state["scan_generation"] = state.get("scan_generation", 0) + 1
                    elif cmd == "list":
                        await send_json({"type": "bags", "bags": self.bags_public()})
                except (KeyError, ValueError, TypeError, OSError, RuntimeError, sqlite3.Error) as err:
                    await send_json({"type": "error", "message": str(err)})
        finally:
            self.access.release(ws)
            stop_player()
            state["scan_generation"] = state.get("scan_generation", 0) + 1
            if state["session"] is not None:
                await loop.run_in_executor(self.pool, state["session"].close)
        return ws

    # ---------------------------------------------------------------- upload
    async def api_upload(self, request: web.Request) -> web.Response:
        """New recording from the operator (session token) or the admin."""
        if not self.access.token_ok(request.headers.get("X-Gabarit-Token")):
            check_admin(request)
        if self.upload_root is None:
            raise web.HTTPForbidden(text="загрузка записей отключена")
        try:
            dest, name = await receive_upload(request, self.upload_root, self.max_upload, self.pool)
        except ValueError as err:
            return web.json_response({"ok": False, "error": str(err)}, status=400)
        await asyncio.get_running_loop().run_in_executor(self.pool, self.rescan)
        bag = next((b for b in self.bags_public() if b["id"] == f"uploads/{name}"), None)
        LOG.info("uploaded recording %s (%s)", name, dest)
        return web.json_response({"ok": True, "bag": bag})

    # ---------------------------------------------------------------- admin
    async def admin_page(self, request: web.Request) -> web.StreamResponse:
        check_admin(request)
        return web.FileResponse(STATIC / "admin.html")

    def _versions(self) -> list[dict]:
        try:
            return json.loads(self.versions_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _mlflow_running(self) -> bool:
        url = mlops.ui_url()
        host = url.split("//")[-1].split(":")[0].split("/")[0] or "127.0.0.1"
        try:
            port = int(url.rsplit(":", 1)[-1].split("/")[0])
        except ValueError:
            port = 80
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            return False

    async def api_admin_status(self, request: web.Request) -> web.Response:
        check_admin(request)
        bags = []
        for d in self.bags_public():
            info = self.bags[d["id"]]
            try:
                size = sum(f.stat().st_size for f in Path(info.path).parent.glob("*.db3"))
            except OSError:
                size = 0
            bags.append({**d, "size_mb": round(size / 2 ** 20), "path": str(Path(info.path).parent)})
        ml = mlflow_status()
        ml["running"] = self._mlflow_running()
        return web.json_response({
            "uptime_s": round(time.time() - self.started),
            "sessions": self.access.public(), "max_users": self.access.max_users,
            "idle_min": round(self.access.idle_s / 60), "refused": self.access.refused,
            "bags": bags, "upload_root": str(self.upload_root) if self.upload_root else None,
            "max_upload_gb": round(self.max_upload / 2 ** 30),
            "config_yaml": config_yaml(self.base_cfg), "presets": list(PRESETS),
            "mlflow": ml, "runs": mlops.recent_runs(), "versions": list(reversed(self._versions()))[:20],
            "jobs": list(self.jobs.values())[-10:], "journal": len(self.journal.all()),
        })

    async def api_admin_kick(self, request: web.Request) -> web.Response:
        check_admin(request)
        n = await self.access.kick_all("Сеанс завершён администратором")
        return web.json_response({"ok": True, "kicked": n})

    async def api_admin_limit(self, request: web.Request) -> web.Response:
        check_admin(request)
        body = await request.json()
        self.access.max_users = max(0, int(body.get("max_users", self.access.max_users)))
        self.access.idle_s = max(60.0, float(body.get("idle_min", self.access.idle_s / 60)) * 60)
        return web.json_response({"ok": True, "max_users": self.access.max_users})

    async def api_admin_config(self, request: web.Request) -> web.Response:
        """New detector parameters for the next sessions (optionally a new model version)."""
        check_admin(request)
        body = await request.json()
        try:
            if body.get("preset"):
                cfg = Config().with_overrides(PRESETS[body["preset"]])
            else:
                cfg = parse_config(body.get("yaml", ""))
        except (KeyError, ValueError, TypeError, yaml.YAMLError) as err:
            return web.json_response({"ok": False, "error": f"ошибка в параметрах: {err}"}, status=400)
        self.base_cfg = cfg
        entry = None
        if body.get("register", True):
            entry = await asyncio.get_running_loop().run_in_executor(
                self.pool, mlops.register_detector, cfg, str(body.get("note", "")))
            vers = self._versions()
            vers.append(entry)
            self.versions_file.parent.mkdir(parents=True, exist_ok=True)
            self.versions_file.write_text(json.dumps(vers[-100:], ensure_ascii=False, indent=1), encoding="utf-8")
        return web.json_response({"ok": True, "version": entry, "config_yaml": config_yaml(cfg)})

    async def api_admin_evaluate(self, request: web.Request) -> web.Response:
        """Evaluate the current parameters on a recording in the background."""
        check_admin(request)
        body = await request.json()
        info = self.bags.get(body.get("bag", ""))
        if info is None:
            return web.json_response({"ok": False, "error": "нет такой записи"}, status=404)
        job_id = f"job{len(self.jobs) + 1}"
        cfg = self.base_cfg
        vers = self._versions()
        cfg_version = (vers[-1].get("version") or vers[-1].get("config_file")) if vers else None
        job = {"id": job_id, "bag": info.name, "state": "running", "done": 0, "total": 0, "result": None}
        self.jobs[job_id] = job

        def progress(done, total):
            job["done"], job["total"] = done, total

        def work():
            res = mlops.evaluate_bag(info, cfg, progress)
            return mlops.log_evaluation(res, cfg, str(cfg_version) if cfg_version else None)

        async def run():
            try:
                job["result"] = await asyncio.get_running_loop().run_in_executor(self.pool, work)
                job["state"] = "done"
            except Exception as err:   # reported to the panel
                job["state"], job["error"] = "error", str(err)[:300]

        asyncio.ensure_future(run())
        return web.json_response({"ok": True, "job": job})

    async def api_admin_delete_bag(self, request: web.Request) -> web.Response:
        check_admin(request)
        bag_id = request.match_info["bag_id"]
        if bag_id not in self.uploaded:
            return web.json_response({"ok": False, "error": "удалять можно только загруженные записи"}, status=400)
        folder = Path(self.bags[bag_id].path).parent
        if self.upload_root is None or self.upload_root.resolve() not in folder.resolve().parents:
            return web.json_response({"ok": False, "error": "запись вне папки загрузок"}, status=400)
        shutil.rmtree(folder, ignore_errors=True)
        await asyncio.get_running_loop().run_in_executor(self.pool, self.rescan)
        return web.json_response({"ok": True})

    async def api_admin_mlflow_start(self, request: web.Request) -> web.Response:
        check_admin(request)
        if not mlops.available():
            return web.json_response({"ok": False, "error": "MLflow не установлен (pip install mlflow)"}, status=400)
        if self._mlflow_running():
            return web.json_response({"ok": True, "url": mlops.ui_url(), "already": True})
        port = mlops.ui_url().rsplit(":", 1)[-1].split("/")[0]
        self.mlflow_proc = subprocess.Popen([sys.executable, "-m", "gabarit", "mlflow", "--port", port],
                                            cwd=str(PROJECT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            await asyncio.sleep(0.5)
            if self._mlflow_running():
                break
        return web.json_response({"ok": self._mlflow_running(), "url": mlops.ui_url()})

    def make(self) -> web.Application:
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_get("/", self.index)
        app.router.add_get("/api/bags", self.api_bags)
        app.router.add_get("/api/health", self.api_health)
        app.router.add_get("/api/journal", self.api_journal)
        app.router.add_get("/api/journal.csv", self.api_journal_csv)
        app.router.add_post("/api/upload", self.api_upload)
        app.router.add_get("/admin", self.admin_page)
        app.router.add_get("/api/admin/status", self.api_admin_status)
        app.router.add_post("/api/admin/kick", self.api_admin_kick)
        app.router.add_post("/api/admin/limit", self.api_admin_limit)
        app.router.add_post("/api/admin/config", self.api_admin_config)
        app.router.add_post("/api/admin/evaluate", self.api_admin_evaluate)
        app.router.add_post("/api/admin/mlflow/start", self.api_admin_mlflow_start)
        app.router.add_delete("/api/admin/bags/{bag_id:.+}", self.api_admin_delete_bag)
        app.router.add_get("/ws", self.ws)
        app.router.add_static("/static/", STATIC, show_index=False)

        async def revalidate(request: web.Request, response: web.StreamResponse) -> None:
            # the page must not run stale scripts after an update; unchanged
            # files are still answered with 304 Not Modified
            if request.path == "/" or request.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache"

        app.on_response_prepare.append(revalidate)
        return app


DEFAULT_DATA = [PROJECT / "data" / "extracted"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Gabarit web interface")
    ap.add_argument("--data", type=Path, action="append",
                    help="directory with ROS 2 bags, searched recursively; may be repeated "
                         "(default: data/extracted, including new finalist recordings)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--uploads", type=Path, default=PROJECT / "data" / "uploads",
                    help="folder for recordings uploaded from the interface (default: data/uploads)")
    ap.add_argument("--max-users", type=int, default=int(os.environ.get("GABARIT_MAX_USERS", 1)),
                    help="simultaneous users of the interface, 0 = unlimited (default: 1)")
    ap.add_argument("--idle-min", type=float, default=10.0,
                    help="an idle session is released after this many minutes (default: 10)")
    ap.add_argument("--max-upload-gb", type=float, default=20.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    roots = args.data or [d for d in DEFAULT_DATA if d.exists()] or DEFAULT_DATA[:1]
    for d in roots:
        if not d.exists():
            LOG.warning("data directory %s does not exist: no bags to play", d)
    app = App(roots, args.config, upload_root=args.uploads, max_users=args.max_users,
              idle_s=args.idle_min * 60, max_upload_gb=args.max_upload_gb)
    LOG.info("found %d bags in %s", len(app.bags), ", ".join(str(d) for d in roots))
    host = 'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host
    print(f"\n  Gabarit: откройте http://{host}:{args.port}   (администрирование: http://{host}:{args.port}/admin)\n")
    web.run_app(app.make(), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
