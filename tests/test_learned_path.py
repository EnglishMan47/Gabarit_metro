"""Frame conversion, abstention and optional real-checkpoint inference."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gabarit.learned_path import DEFAULT_MODEL, FarBevAdvisor, prepare_bev
from gabarit.vendor.attention_please.geometry import LAYERS, stations


def _track(center=0., height=0., rails=40., reach=100.):
    s = np.arange(1., 202., 2.)
    return SimpleNamespace(s=s, c=np.full_like(s, center), z=np.full_like(s, height),
                           observed_until=reach, rail_until=rails)


def _cloud(center=0., height=0.):
    s = np.arange(10., 181., .5)
    # Both walls, bed and low rail returns.
    points = np.concatenate([np.column_stack((np.full_like(s, d + center), s, np.full_like(s, u + height)))
                             for d, u in ((-2.53, 1.05), (2.53, 1.05), (.03, -.2), (.83, .03))])
    return points, np.full(len(points), 30., np.float32)


def _fake_advisor(dc=0.2, sigma=.1, dz=.05, scale=.05):
    advisor = FarBevAdvisor()
    advisor._attempted = True
    advisor._net = SimpleNamespace(s1=170.)
    shape = stations(170.).shape
    advisor._predict = lambda *_: tuple(np.full(shape, v) for v in (dc, sigma, dz, scale))
    return advisor


def test_import_keeps_torch_optional():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import gabarit.learned_path; assert 'torch' not in sys.modules"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bev_is_invariant_to_calibrated_height_and_lateral_origin():
    points, intensity = _cloud()
    shifted, _ = _cloud(center=1.25, height=2.4)
    original = prepare_bev(points, intensity, _track(), 170.)[0]
    moved = prepare_bev(shifted, intensity, _track(1.25, 2.4), 170.)[0]
    assert original.shape == (9, 320, 200)
    np.testing.assert_array_equal(original, moved)
    # Rail-head returns at forward 10 m occupy the first row, not the back.
    assert original[:len(LAYERS), 0].sum() == 4


def test_nonfinite_and_behind_points_do_not_pollute_bev():
    xyz, intensity = _cloud()
    image = prepare_bev(xyz, intensity, _track(), 170.)[0]
    bad = np.array([[np.nan, 20., 1.], [0., np.inf, 1.], [0., -20., 1.], [0., 20., np.nan]])
    mixed = prepare_bev(np.vstack((xyz, bad)), np.r_[intensity, [1, 2, 3, 4]], _track(), 170.)[0]
    np.testing.assert_array_equal(image, mixed)


def test_missing_checkpoint_falls_back_and_reset_retries(tmp_path):
    advisor = FarBevAdvisor(tmp_path / "absent.pt")
    xyz, intensity = _cloud()
    assert advisor.observe(xyz, intensity, _track()) is None
    assert advisor.status()["reason"] == "checkpoint_missing"
    advisor.reset()
    assert advisor.status()["state"] == "not_loaded"


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_cpu_thread_count_must_be_a_positive_integer(threads):
    with pytest.raises(ValueError, match="cpu_threads"):
        FarBevAdvisor(cpu_threads=threads)


def test_measurements_preserve_rails_and_absolute_frame_without_mutation():
    advisor = _fake_advisor()
    xyz, intensity = _cloud(center=1.25, height=2.4)
    old_xyz = xyz.copy()
    track = _track(1.25, 2.4)
    out = advisor.observe(xyz, intensity, track)
    assert out is not None and out.valid.any()
    assert not out.valid[out.s <= 40.].any()
    assert not out.valid[out.s > 160.].any()
    np.testing.assert_allclose(out.c[out.valid & (out.s >= 50)], 1.45)
    np.testing.assert_allclose(out.z[out.valid & (out.s >= 50)], 2.45)
    np.testing.assert_array_equal(xyz, old_xyz)
    np.testing.assert_allclose(track.c, 1.25)
    np.testing.assert_allclose(track.z, 2.4)


@pytest.mark.parametrize("outputs", [dict(sigma=1.), dict(scale=.3), dict(dc=8.), dict(dz=1.)])
def test_uncertainty_and_bounded_corrections_abstain(outputs):
    xyz, intensity = _cloud()
    out = _fake_advisor(**outputs).observe(xyz, intensity, _track())
    assert out is not None and not out.valid.any()


def test_large_obstacle_must_not_steer_path_around_it_when_rails_hidden():
    xyz, intensity = _cloud()
    out = _fake_advisor(dc=2.).observe(xyz, intensity, _track(rails=20.))
    assert out is not None and not out.valid.any()


def test_absent_context_and_backend_failure_remain_safe():
    advisor = _fake_advisor()
    assert advisor.observe(np.zeros((0, 3)), np.zeros(0), _track()) is None
    assert advisor.status()["reason"] == "insufficient_visible_context"
    advisor._predict = lambda *_: (_ for _ in ()).throw(RuntimeError("bad backend"))
    xyz, intensity = _cloud()
    assert advisor.observe(xyz, intensity, _track()) is None
    assert advisor.status()["reason"] == "inference_failed"
    assert advisor.observe(xyz, intensity, _track()) is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or not DEFAULT_MODEL.is_file(),
                    reason="Optional PyTorch and pinned downloaded checkpoint required")
def test_real_pinned_checkpoint_cpu_inference():
    import torch
    from gabarit.vendor.attention_please.far_bev_net import load, predict
    from scripts.fetch_path_models import verify

    verify(DEFAULT_MODEL)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(4, previous_threads))
        net = load(DEFAULT_MODEL, "cpu")
        xyz, intensity = _cloud()
        image, *_ = prepare_bev(xyz, intensity, _track(), net.s1)
        outputs = predict(net, image, 100.)
        assert len(outputs) == 4
        for output in outputs:
            assert output.shape == stations(net.s1).shape
            assert np.isfinite(output).all()
        assert (outputs[1] >= .03).all() and (outputs[3] >= .01).all()
        # Run the public adapter as well: real outputs pass all frame/gate code.
        # Pipeline.__init__ limits OpenMP to one; the advisor's explicit CPU
        # setting must still be effective, with direct API users able to use 1.
        torch.set_num_threads(1)
        advisor = FarBevAdvisor(DEFAULT_MODEL, device="cpu", cpu_threads=4)
        result = advisor.observe(xyz, intensity, _track())
        assert result is not None
        assert advisor.status()["state"] == "ready"
        assert advisor.status()["cpu_threads"] == torch.get_num_threads() == 4
        assert np.isfinite(result.c).all() and np.isfinite(result.z).all()
        assert not result.valid[result.s <= 40.].any()
        single = FarBevAdvisor(DEFAULT_MODEL, device="cpu", cpu_threads=1)
        assert single._load()
        assert single.status()["cpu_threads"] == torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous_threads)
