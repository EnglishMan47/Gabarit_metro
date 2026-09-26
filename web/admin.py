"""Service administration: one user at a time, upload of recordings and the
admin panel API (sessions, recordings, detector parameters, MLflow)."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import secrets
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from gabarit import mlops
from gabarit.config import Config
from gabarit.rosbag import open_bag

UPLOAD_GROUP = "Загруженные записи"
PRESETS = {
    "balanced": {},
    "strict": {"tracking": {"confirm_hits": 4, "confirm_window": 5, "far_confirm_hits": 8, "far_confirm_window": 10},
               "detection": {"min_score": 0.65}},
    "sensitive": {"tracking": {"confirm_hits": 2, "confirm_window": 3, "far_confirm_hits": 4, "far_confirm_window": 6},
                  "detection": {"min_score": 0.4}},
}


# ---------------------------------------------------------------- access
class Access:
    """Limits the playback interface to ``max_users`` simultaneous users.

    The service replays heavy recordings with the full pipeline per user; with
    one CPU core per session several parallel users would slow everyone down
    and look like a broken service. Extra visitors get a "busy" message and
    retry; an idle session is released after ``idle_s`` seconds.
    """

    def __init__(self, max_users: int = 1, idle_s: float = 600.0):
        self.max_users = max_users
        self.idle_s = idle_s
        self.sessions: dict[Any, dict] = {}
        self.refused = 0

    def _prune(self) -> None:
        for ws in [w for w in self.sessions if w.closed]:
            self.sessions.pop(ws, None)

    async def admit(self, ws: web.WebSocketResponse, request: web.Request) -> dict | None:
        self._prune()
        now = time.time()
        if self.max_users and len(self.sessions) >= self.max_users:
            for w, s in list(self.sessions.items()):
                if now - s["last"] > self.idle_s:
                    await self.kick(w, "Сеанс завершён из-за неактивности")
        if self.max_users and len(self.sessions) >= self.max_users:
            self.refused += 1
            return None
        s = {"token": secrets.token_urlsafe(16), "ip": request.remote or "?", "since": now, "last": now,
             "agent": request.headers.get("User-Agent", "")[:120]}
        self.sessions[ws] = s
        return s

    def touch(self, ws) -> None:
        s = self.sessions.get(ws)
        if s is not None:
            s["last"] = time.time()

    def release(self, ws) -> None:
        self.sessions.pop(ws, None)

    async def kick(self, ws, reason: str) -> None:
        self.sessions.pop(ws, None)
        try:
            await ws.send_json({"type": "kicked", "reason": reason})
            await ws.close()
        except (ConnectionError, RuntimeError):
            pass

    async def kick_all(self, reason: str) -> int:
        ws_list = list(self.sessions)
        for ws in ws_list:
            await self.kick(ws, reason)
        return len(ws_list)

    def token_ok(self, token: str | None) -> bool:
        return bool(token) and any(s["token"] == token for s in self.sessions.values())

    def busy_info(self) -> dict:
        self._prune()
        now = time.time()
        oldest = min((s["since"] for s in self.sessions.values()), default=now)
        return {"type": "busy", "users": len(self.sessions), "max_users": self.max_users,
                "busy_for_s": round(now - oldest), "retry_s": 10}

    def public(self) -> list[dict]:
        self._prune()
        now = time.time()
        return [{"ip": s["ip"], "since": time.strftime("%H:%M:%S", time.localtime(s["since"])),
                 "active_s": round(now - s["since"]), "idle_s": round(now - s["last"]), "agent": s["agent"]}
                for s in self.sessions.values()]


# ---------------------------------------------------------------- upload
_SAFE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё_.-]+")


def safe_name(name: str) -> str:
    name = _SAFE.sub("_", Path(name).name).strip("._") or "record"
    for ext in (".tar.gz", ".tgz", ".tar", ".zip", ".db3", ".yaml"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name[:60] or "record"


def _extract_members(archive: Path, dest: Path) -> int:
    """Only *.db3 and metadata.yaml, flattened (no paths from the archive)."""
    n = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                base = Path(m.filename).name
                if m.is_dir() or not (base.endswith(".db3") or base == "metadata.yaml"):
                    continue
                with z.open(m) as src, open(dest / base, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                n += base.endswith(".db3")
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                base = Path(m.name).name
                if not m.isfile() or not (base.endswith(".db3") or base == "metadata.yaml"):
                    continue
                src = t.extractfile(m)
                if src is None:
                    continue
                with src, open(dest / base, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                n += base.endswith(".db3")
    else:
        raise ValueError("архив не распознан: нужен .zip или .tar/.tar.gz")
    return n


async def receive_upload(request: web.Request, root: Path, max_bytes: int, pool) -> tuple[Path, str]:
    """Stream the multipart upload to ``root/<name>/``; returns (bag dir, name)."""
    reader = await request.multipart()
    name_hint = request.query.get("name", "")
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".incoming-{secrets.token_hex(4)}"
    tmp.mkdir()
    received = 0
    files: list[Path] = []
    try:
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name != "file" or not part.filename:
                continue
            fname = Path(part.filename).name
            low = fname.lower()
            if not low.endswith((".db3", ".yaml", ".zip", ".tar", ".tgz", ".tar.gz")):
                raise ValueError(f"файл {fname}: нужен .db3 (с metadata.yaml или без) либо архив .zip/.tar")
            name_hint = name_hint or fname
            target = tmp / (fname if not low.endswith(".yaml") else "metadata.yaml")
            with open(target, "wb") as fh:
                while True:
                    chunk = await part.read_chunk(1 << 20)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise ValueError(f"файл больше допустимого размера ({max_bytes / 2**30:.0f} ГБ)")
                    fh.write(chunk)
            files.append(target)
        if not files:
            raise ValueError("файл не передан")
        loop = asyncio.get_running_loop()
        for f in list(files):
            if f.name.lower().endswith((".zip", ".tar", ".tgz", ".tar.gz")):
                n = await loop.run_in_executor(pool, _extract_members, f, tmp)
                f.unlink()
                if not n:
                    raise ValueError("в архиве нет файла .db3")
        if not list(tmp.glob("*.db3")):
            raise ValueError("нет файла .db3 (запись ROS 2 bag в формате sqlite3)")
        name = safe_name(name_hint)
        dest = root / name
        k = 2
        while dest.exists():
            dest = root / f"{name}_{k}"
            k += 1
        tmp.rename(dest)
        try:
            await loop.run_in_executor(pool, open_bag, dest)   # readable PointCloud2 bag?
        except Exception as err:
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(f"не удалось прочитать запись: {err}") from err
        return dest, dest.name
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- admin auth
def admin_password() -> str:
    return os.environ.get("GABARIT_ADMIN_PASSWORD", "admin")


def check_admin(request: web.Request) -> None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            user, _, pwd = base64.b64decode(auth[6:]).decode("utf-8").partition(":")
            if user == "admin" and secrets.compare_digest(pwd, admin_password()):
                return
        except (ValueError, UnicodeDecodeError):
            pass
    raise web.HTTPUnauthorized(headers={"WWW-Authenticate": 'Basic realm="Gabarit admin", charset="UTF-8"'})


def config_yaml(cfg: Config) -> str:
    return yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)


def parse_config(text: str, base: Config | None = None) -> Config:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("ожидается YAML-словарь параметров")
    return (base or Config()).with_overrides(data)


def mlflow_status() -> dict:
    return {"installed": mlops.available(), "ui_url": mlops.ui_url(), "tracking_uri": mlops.tracking_uri(),
            "model": mlops.MODEL_NAME, "experiment": mlops.EXPERIMENT}
