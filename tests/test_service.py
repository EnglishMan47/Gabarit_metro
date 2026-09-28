"""Service features: emergency braking of the demo, transparent obstacles,
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
from web.session import BrakeGuard, BrakeSim  # noqa: E402


def _meta(ids=(), status=None, v=15.0, dist=150.0):
    obs = [{"track_id": i, "distance_m": dist, "lateral_m": 0.0} for i in ids]
    return {"status": status or ("OBSTACLE" if ids else "CLEAR"), "obstacles": obs,
            "speed_mps": v, "nearest_track_distance_m": dist if ids else None}


def _drive_to_stop(g: BrakeGuard, meta: dict) -> None:
    while not g.brake.done:
        g.brake.advance(1.0)
    g.hold(meta)


def test_guard_brakes_without_operator_and_holds_while_occupied():
    g = BrakeGuard(clear_frames=3)
    assert g.start(_meta([7]), 0.3, 2.5) is not None      # no operator input needed
    _drive_to_stop(g, _meta([7]))
    assert g.holding
    for _ in range(10):                                    # obstacle stays: train stays
        assert not g.check_clear(_meta([7]))
    assert g.holding


def test_guard_resumes_by_itself_when_the_track_clears():
    g = BrakeGuard(clear_frames=3)
    g.start(_meta([7]), 0.3, 2.5)
    _drive_to_stop(g, _meta([7]))
    assert not g.check_clear(_meta())
    assert not g.check_clear(_meta([7]))                  # it came back: counter restarts
    assert [g.check_clear(_meta()) for _ in range(3)] == [False, False, True]
    assert not g.holding and g.brake is None and g.last_release == "auto"
    assert g.start(_meta([7]), 0.3, 2.5) is not None      # not exempt: brakes again


def test_guard_override_exempts_only_those_objects():
    g = BrakeGuard()
    g.start(_meta([7]), 0.3, 2.5)
    _drive_to_stop(g, _meta([7]))
    assert g.override(_meta([7])) == {7}
    assert not g.holding and g.last_release == "override"
    assert g.start(_meta([7]), 0.3, 2.5) is None           # the operator took responsibility for #7
    assert g.start(_meta([7, 9]), 0.3, 2.5) is not None    # a new obstacle brakes the train again
    assert g.hold_ids == {9}


def _drive(b: BrakeSim, dx: float) -> float:
    t = 0.0
    for _ in range(10000):
        if b.done:
            break
        t += b.advance(dx)
    return t


def test_brake_stops_before_a_far_obstacle():
    cfg = Config()
    v = 80 / 3.6
    b = BrakeSim(v, 140.0, cfg.reaction_time, cfg.braking_deceleration)
    t = _drive(b, v * 0.1)
    expected = v * cfg.reaction_time + v * v / (2 * cfg.braking_deceleration)
    assert not b.collision and b.public()["stopped"]
    assert abs(b.x - expected) < 0.05 and abs(b.stop_m - expected) < 1e-6
    assert abs(t - (cfg.reaction_time + v / cfg.braking_deceleration)) < 0.05
    assert b.public()["gap_m"] > 30


def test_brake_reports_impact_when_too_close():
    b = BrakeSim(60 / 3.6, 40.0, 0.3, 2.5)
    _drive(b, 60 / 3.6 * 0.1)
    p = b.public()
    assert b.collision and not p["stopped"]
    assert 30 < p["impact_kmh"] < 45          # braking still lowers the speed
    assert b.x <= 39.0 + 1e-6


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
