"""Service features: transparent obstacles,
one user at a time, upload of recordings."""

from __future__ import annotations

import asyncio
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gabarit  # noqa: E402,F401
from gabarit.cloud import dual_return_split  # noqa: E402
from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.synth import SyntheticObstacle, inject  # noqa: E402
from tests.sim import tunnel_scan  # noqa: E402
from web.admin import Access, _extract_members, safe_name  # noqa: E402
def test_dual_return_split_marks_the_near_echo():
    rings, cols = 4, 6
    r = np.full((cols, rings), 20.0)
    r[2, 1], r[3, 1] = 12.0, 30.0          # firing 1, ring 1: glass at 12 m, wall at 30 m
    r[5, 3], r[4, 3] = 9.0, 9.02           # identical echoes: solid surface
    split = dual_return_split((r ** 2).reshape(-1), np.ones(rings * cols, bool), rings, cols).reshape(cols, rings)
    assert split[2, 1] and not split[3, 1]
    assert split.sum() == 1


def test_glass_pane_is_detected_as_transparent():
    pipe = Pipeline(Config())
    glass = SyntheticObstacle(shape="glass", s=30.0, d=0.0)
    rng = np.random.default_rng(1)
    flagged = False
    for k in range(10):
        raw = tunnel_scan(stamp=0.1 * k)
        if pipe.track.prev is not None:
            raw, _ = inject(raw, [glass], pipe.track.prev, pipe.track.calib, rng=rng)
        res = pipe.process_raw(raw, 0.1 * k, frame_index=k)
        flagged |= any(c.get("transparent") for c in res.data["candidates"])
    assert flagged
    assert res.data["obstacle"]
    assert abs(res.data["nearest_track_distance_m"] - 30.0) < 1.5


class _WS:
    def __init__(self):
        self.closed = False
        self.sent = []

    async def send_json(self, obj):
        self.sent.append(obj)

    async def close(self):
        self.closed = True


class _Req:
    remote = "127.0.0.1"
    headers = {"User-Agent": "test"}


def test_one_user_at_a_time_and_idle_release():
    async def scenario():
        acc = Access(max_users=1, idle_s=600)
        a, b = _WS(), _WS()
        seat = await acc.admit(a, _Req())
        assert seat is not None and acc.token_ok(seat["token"])
        assert await acc.admit(b, _Req()) is None and acc.refused == 1
        acc.sessions[a]["last"] -= 601                  # the first user went idle
        assert await acc.admit(b, _Req()) is not None
        assert a.closed and a.sent[-1]["type"] == "kicked"
        acc.release(b)
        assert await acc.admit(_WS(), _Req()) is not None
    asyncio.run(scenario())


def test_upload_names_and_archive_extraction(tmp_path):
    assert safe_name("../../evil name.db3") == "evil_name"
    assert safe_name("Запись 1.tar.gz") == "Запись_1"
    arc = tmp_path / "rec.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("folder/rec_0.db3", b"sqlite")
        z.writestr("folder/metadata.yaml", "x: 1")
        z.writestr("../../outside.txt", "no")
    dest = tmp_path / "out"
    dest.mkdir()
    assert _extract_members(arc, dest) == 1
    assert sorted(p.name for p in dest.iterdir()) == ["metadata.yaml", "rec_0.db3"]
