"""Optional, uncertainty-gated FarBevNet path observations.

The host always passes calibrated coordinates (left, forward, up) with the
rail plane near z=0. TrackModel.z is in that same calibrated frame. The
network sees normal distance and height *relative to that path*, so sensor
mounting height and input-axis choices do not leak into the learned image.
The advisor never mutates TrackModel, declares rail detections or changes the
obstacle list. The caller decides how to fuse accepted observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import time

import numpy as np

from gabarit.vendor.attention_please.geometry import (
    DS, LAYERS, S0, bev_grid, extend_axis, n_rows, raster, stations, straighten,
)

if TYPE_CHECKING:
    from gabarit.track import TrackModel

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / ".models" / "attention_please" / "far_bev.pt"


@dataclass(frozen=True)
class LearnedPathObservation:
    """Absolute path samples in the host's calibrated frame, at 2 m stations.

    valid masks uncertain/unsupported stations. sig_c is conservatively the
    raw network spread, *not* the winner's empirical 0.2 multiplier: that
    multiplier has not been calibrated for this host's sensors/datasets.
    """

    s: np.ndarray
    c: np.ndarray
    z: np.ndarray
    sig_c: np.ndarray
    sig_z: np.ndarray
    valid: np.ndarray
    visible_until: float
    input_reach: float
    elapsed_ms: float

    @property
    def accepted_until(self) -> float:
        return float(self.s[self.valid][-1]) if self.valid.any() else 0.0


def prepare_bev(xyz: np.ndarray, intensity: np.ndarray, track: "TrackModel", s1: float):
    """Exact winner raster from leveled host points; no Torch import needed.

    Upstream uses (left, -forward, up). Its straighten() immediately negates
    the second axis; the conversion below therefore preserves positive-ahead
    host distances. Neither the cloud nor the track arrays are modified.
    """
    points = np.asarray(xyz)
    inten = np.asarray(intensity)
    if points.ndim != 2 or points.shape[1] != 3 or inten.shape != (points.shape[0],):
        raise ValueError("Expected xyz (N,3) and intensity (N,)")
    grid = np.asarray(track.s, np.float64)
    center, height = np.asarray(track.c, np.float64), np.asarray(track.z, np.float64)
    if (grid.size < 2 or center.shape != grid.shape or height.shape != grid.shape
            or not np.isfinite(np.concatenate((grid, center, height))).all()
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("Track needs at least two finite, increasing path samples")
    if not np.isfinite(s1) or not 42 <= s1 <= 512 or n_rows(s1) % 4:
        raise ValueError("Unsupported checkpoint longitudinal extent")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(inten)
    p = points[finite].astype(np.float64, copy=True)
    p[:, 1] *= -1.0
    i = np.clip(inten[finite], 0, 255)
    reach = float(np.clip(track.observed_until, grid[0], grid[-1]))
    gb = bev_grid(s1)
    n_ax, z_ax = extend_axis(grid, center, height, reach, gb)
    s, d, u = straighten(p, gb, n_ax, z_ax)
    image = raster(s, d, u, i, n_rows(s1))
    return image, gb, n_ax, z_ax, reach


def _support(image: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, float]:
    # The winner's visibility cutoff is the third-from-last occupied row.
    hits = image[:len(LAYERS)].sum(axis=(0, 2)) > 0
    rows = np.flatnonzero(hits)
    visible = float(S0 + (rows[-3] + 1) * DS) if rows.size >= 3 else 0.0
    # In addition require local evidence, so disconnected returns cannot make
    # an unobserved interval count as a measured path.
    centres = S0 + (np.arange(hits.size) + .5) * DS
    prefix = np.concatenate(([0], np.cumsum(hits)))
    lo = np.searchsorted(centres, st - 4.0)
    hi = np.searchsorted(centres, st + 4.0, side="right")
    return prefix[hi] - prefix[lo] >= 3, visible


class FarBevAdvisor:
    """Lazy optional inference. Absence/failure is exposed by status().

    A missing dependency or weight file returns None and leaves the detector
    operational. reset() clears any failure latch, allowing an installed or
    newly downloaded model to be discovered on the next observation.
    The single-frame adapter deliberately does not accumulate scans without
    measured full-frame poses (translation alone distorts bends).
    """

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", *,
                 sigma_max: float = .45, far_sigma_max: float = .65,
                 z_sigma_max: float = .2, max_correction: float = 3.0,
                 max_height_correction: float = .5, cpu_threads: int = 4):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        limits = (sigma_max, far_sigma_max, z_sigma_max, max_correction, max_height_correction)
        if not all(np.isfinite(v) and v > 0 for v in limits):
            raise ValueError("All confidence/correction limits must be positive and finite")
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, (int, np.integer)) or cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL
        self.device = device
        self.cpu_threads = int(cpu_threads)
        self.sigma_max, self.far_sigma_max = float(sigma_max), float(far_sigma_max)
        self.z_sigma_max = float(z_sigma_max)
        self.max_correction, self.max_height_correction = float(max_correction), float(max_height_correction)
        self._net = None
        self._predict = None
        self._attempted = False
        self._status = {"state": "not_loaded", "source": "Attention Please / BelousovArts/fod",
                        "model": str(self.model_path)}

    def reset(self) -> None:
        self._net = self._predict = None
        self._attempted = False
        self._status = dict(self._status, state="not_loaded", reason="")

    def status(self) -> dict:
        return dict(self._status)

    def _load(self) -> bool:
        if self._attempted:
            return self._net is not None
        self._attempted = True
        if not self.model_path.is_file():
            self._status.update(state="unavailable", reason="checkpoint_missing")
            return False
        try:
            import torch
            from gabarit.vendor.attention_please.far_bev_net import load, predict
            device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
            if device == "auto":
                device = "cpu"
            # The host limits small-matrix BLAS/OpenMP work to one thread.
            # Override PyTorch's intra-op pool explicitly for the much larger
            # CNN convolutions; this does not change NumPy's BLAS limit.
            actual_device = "cpu" if device == "cuda" and not torch.cuda.is_available() else device
            if actual_device == "cpu":
                torch.set_num_threads(self.cpu_threads)
            net = load(self.model_path, device)
            # Checkpoint shape must match the encoded raster geometry.
            if not np.isfinite(net.s1) or not 42 <= net.s1 <= 512 or n_rows(net.s1) % 4:
                raise ValueError("Unsupported checkpoint longitudinal extent")
            self._net, self._predict = net, predict
            self._status.update(state="ready", reason="", device=str(next(net.parameters()).device), s1=net.s1,
                                cpu_threads=torch.get_num_threads() if actual_device == "cpu" else None)
            return True
        except ImportError as error:
            self._status.update(state="unavailable", reason="torch_unavailable", detail=str(error))
        except Exception as error:
            self._status.update(state="unavailable", reason="checkpoint_load_failed", detail=f"{type(error).__name__}: {error}")
        return False

    def observe(self, xyz_leveled: np.ndarray, intensity: np.ndarray, track: "TrackModel") -> LearnedPathObservation | None:
        """Infer path measurements without changing the track or input cloud."""
        if not self._load():
            return None
        start = time.perf_counter()
        try:
            if self._status.get("device") == "cpu":
                import torch
                # Constructing another geometric Pipeline can reset the
                # process-wide OpenMP limit after this advisor was loaded.
                if torch.get_num_threads() != self.cpu_threads:
                    torch.set_num_threads(self.cpu_threads)
            image, gb, n_ax, z_ax, reach = prepare_bev(xyz_leveled, intensity, track, self._net.s1)
            st = stations(self._net.s1)
            support, visible = _support(image, st)
            if visible < 30.0:
                self._status.update(state="ready", reason="insufficient_visible_context", accepted=0)
                return None
            dc, sigma, dz, scale = self._predict(self._net, image, reach)
            dc, sigma, dz, scale = [np.asarray(v, np.float64) for v in (dc, sigma, dz, scale)]
            if any(v.shape != st.shape for v in (dc, sigma, dz, scale)):
                raise ValueError("Checkpoint output shape does not match its longitudinal extent")
            blend = np.clip((st - 30.0) / 20.0, 0.0, 1.0)
            slope = np.interp(st, gb, np.gradient(n_ax, gb))
            dn = blend * dc * np.sqrt(1.0 + slope * slope)
            height_shift = blend * dz
            limit = np.where(st > 220.0, self.far_sigma_max, self.sigma_max)
            # The older 170 m checkpoint used a 0.6 raw-spread threshold.
            if self._net.s1 <= 170.0:
                limit = np.full_like(st, max(self.sigma_max, .6))
            center_out = np.interp(st, gb, n_ax) + dn
            height_out = np.interp(st, gb, z_ax) + height_shift
            center_change = center_out - np.interp(st, track.s, track.c)
            height_change = height_out - np.interp(st, track.s, track.z)
            valid = (support & (st > max(30.0, float(track.rail_until)))
                     & (st <= min(float(track.s[-1]), self._net.s1 - 10.0, visible))
                     & np.isfinite(dc) & np.isfinite(sigma) & np.isfinite(dz) & np.isfinite(scale)
                     & (sigma >= 0) & (sigma <= limit) & (scale > 0) & (scale <= self.z_sigma_max)
                     & (np.abs(dn) <= self.max_correction) & (np.abs(height_shift) <= self.max_height_correction)
                     & (np.abs(center_change) <= self.max_correction)
                     & (np.abs(height_change) <= self.max_height_correction))
            # Upstream protection against a large obstacle steering the net
            # around it when near rails are hidden. Reject the entire frame.
            if float(track.rail_until) < 30 and np.any((st <= reach) & (np.abs(dn) > 1.0)):
                valid[:] = False
            out = LearnedPathObservation(st, center_out, height_out,
                                         np.maximum(sigma, .03), np.maximum(scale, .01), valid,
                                         visible, reach, 1000 * (time.perf_counter() - start))
            self._status.update(state="ready", reason="" if valid.any() else "confidence_rejected",
                                accepted=int(valid.sum()), visible_until=visible,
                                elapsed_ms=round(out.elapsed_ms, 2))
            return out
        except Exception as error:
            # A broken optional backend is latched off until reset; geometric
            # estimation continues and diagnostics preserve the cause.
            self._net = None
            self._status.update(state="unavailable", reason="inference_failed", detail=f"{type(error).__name__}: {error}")
            return None
