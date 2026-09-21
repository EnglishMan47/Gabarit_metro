"""Optional ONNX execution with deterministic geometry fallback."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

from webapp.bev import BEV_CHANNELS, DEFAULT_BEV_SPEC, BevSpec


MODES = ("auto", "gpu", "cpu", "geometry")


class HybridInferenceEngine:
    def __init__(
        self,
        model_path: Path,
        *,
        spec: BevSpec = DEFAULT_BEV_SPEC,
        ort_module: Any | None = None,
        import_runtime: bool = True,
    ):
        self.model_path = model_path.resolve()
        self.spec = spec
        self._sessions: dict[str, Any] = {}
        self._session_errors: dict[str, str] = {}
        self._backend_probe: dict[str, bool] = {}
        self._backend_latency_ms: dict[str, float] = {}
        self._ort = ort_module
        self._runtime_error: str | None = None
        if self._ort is None and import_runtime:
            try:
                self._ort = importlib.import_module("onnxruntime")
            except (ImportError, OSError) as error:
                self._runtime_error = str(error)
        self._prepare_gpu_runtime()

    def _prepare_gpu_runtime(self) -> None:
        """Make pip-installed CUDA/cuDNN DLLs discoverable on Windows."""
        if self._ort is None or os.name != "nt" or not hasattr(self._ort, "preload_dlls"):
            return
        module_file = getattr(self._ort, "__file__", None)
        if not module_file:
            return
        site_packages = Path(module_file).resolve().parents[1]
        cuda_directory = site_packages / "nvidia" / "cu13" / "bin" / "x86_64"
        cudnn_directory = site_packages / "nvidia" / "cudnn" / "bin"
        try:
            if cuda_directory.is_dir():
                self._ort.preload_dlls(cuda=True, cudnn=False, directory=str(cuda_directory))
            if cudnn_directory.is_dir():
                self._ort.preload_dlls(cuda=False, cudnn=True, directory=str(cudnn_directory))
        except Exception as error:  # CUDA is optional; CPU must remain available
            self._runtime_error = f"GPU libraries could not be preloaded: {error}"

    @property
    def model_available(self) -> bool:
        return self.model_path.is_file()

    def providers(self) -> list[str]:
        if self._ort is None:
            return []
        try:
            return list(self._ort.get_available_providers())
        except Exception as error:  # runtime discovery is an optional boundary
            self._runtime_error = str(error)
            return []

    def capabilities(self) -> dict[str, Any]:
        providers = self.providers()
        if not self.model_available or self._ort is None:
            recommended = "geometry"
        else:
            gpu_ready = "CUDAExecutionProvider" in providers and self._probe_backend("gpu")
            cpu_ready = "CPUExecutionProvider" in providers and self._probe_backend("cpu")
            if gpu_ready and cpu_ready:
                recommended = (
                    "gpu"
                    if self._backend_latency_ms["gpu"] < self._backend_latency_ms["cpu"] * 0.9
                    else "cpu"
                )
            elif gpu_ready:
                recommended = "gpu"
            elif cpu_ready:
                recommended = "cpu"
            else:
                recommended = "geometry"
        return {
            "model_available": self.model_available,
            "model_path": str(self.model_path),
            "onnxruntime_available": self._ort is not None,
            "providers": providers,
            "recommended_mode": recommended,
            "probe_latency_ms": dict(self._backend_latency_ms),
            "fallback_available": True,
            "bev": self.spec.public(),
            "runtime_error": self._runtime_error,
        }

    def should_build_bev(self, mode: str) -> bool:
        self.validate_mode(mode)
        return mode != "geometry" and self.model_available and self._ort is not None

    @staticmethod
    def validate_mode(mode: str) -> str:
        normalized = str(mode).lower()
        if normalized not in MODES:
            raise ValueError(f"inference_mode must be one of {MODES}")
        return normalized

    def _session(self, backend: str) -> Any:
        if backend in self._sessions:
            return self._sessions[backend]
        if backend in self._session_errors:
            raise RuntimeError(self._session_errors[backend])
        provider = "CUDAExecutionProvider" if backend == "gpu" else "CPUExecutionProvider"
        if provider not in self.providers():
            message = f"{provider} is unavailable"
            self._session_errors[backend] = message
            raise RuntimeError(message)
        try:
            session = self._ort.InferenceSession(str(self.model_path), providers=[provider])
            actual_providers = session.get_providers() if hasattr(session, "get_providers") else [provider]
            if provider not in actual_providers:
                raise RuntimeError(f"{provider} failed to initialize; active providers: {actual_providers}")
        except Exception as error:
            self._session_errors[backend] = str(error)
            raise RuntimeError(str(error)) from error
        self._sessions[backend] = session
        return session

    def _probe_backend(self, backend: str) -> bool:
        if backend in self._backend_probe:
            return self._backend_probe[backend]
        probe = np.zeros((len(BEV_CHANNELS), self.spec.height, self.spec.width), dtype=np.float32)
        try:
            self._run(backend, probe)
            samples = [self._run(backend, probe)[1] for _ in range(3)]
            self._backend_latency_ms[backend] = float(median(samples))
            result = True
        except Exception:
            result = False
        self._backend_probe[backend] = result
        return result

    def _run(self, backend: str, bev: np.ndarray) -> tuple[np.ndarray, float]:
        session = self._session(backend)
        input_meta = session.get_inputs()[0]
        batch = np.ascontiguousarray(bev[None, ...], dtype=np.float32)
        started = perf_counter()
        output = session.run(None, {input_meta.name: batch})[0]
        elapsed_ms = (perf_counter() - started) * 1000
        probabilities = np.asarray(output, dtype=np.float32)
        if probabilities.ndim == 4:
            probabilities = probabilities[0, 0]
        elif probabilities.ndim == 3:
            probabilities = probabilities[0]
        if probabilities.shape != (self.spec.height, self.spec.width):
            raise RuntimeError(
                f"unexpected model output shape {probabilities.shape}; expected {(self.spec.height, self.spec.width)}"
            )
        return np.clip(probabilities, 0.0, 1.0), elapsed_ms

    def infer(self, bev: np.ndarray | None, requested_mode: str) -> tuple[dict[str, Any], np.ndarray | None]:
        mode = self.validate_mode(requested_mode)
        base = {
            "requested_mode": mode,
            "active_backend": "geometry",
            "model_used": False,
            "model_latency_ms": None,
            "fallback_reason": None,
            "advisory_only": True,
        }
        if mode == "geometry":
            return base, None
        if not self.model_available:
            return {**base, "fallback_reason": "model_missing"}, None
        if self._ort is None:
            return {**base, "fallback_reason": "onnxruntime_missing"}, None
        if bev is None:
            return {**base, "fallback_reason": "bev_missing"}, None

        available = self.providers()
        auto_recommendation = self.capabilities()["recommended_mode"] if mode == "auto" else None
        order = {
            "auto": ["gpu", "cpu"] if auto_recommendation == "gpu" else ["cpu", "gpu"],
            "gpu": ["gpu", "cpu"],
            "cpu": ["cpu"],
        }[mode]
        failures: list[str] = []
        for backend in order:
            provider = "CUDAExecutionProvider" if backend == "gpu" else "CPUExecutionProvider"
            if provider not in available:
                failures.append("cuda_unavailable" if backend == "gpu" else "cpu_unavailable")
                continue
            try:
                probabilities, elapsed_ms = self._run(backend, bev)
                fallback = failures[0] if failures else None
                return {
                    **base,
                    "active_backend": "cuda" if backend == "gpu" else "cpu",
                    "model_used": True,
                    "model_latency_ms": elapsed_ms,
                    "fallback_reason": fallback,
                    "probability_max": float(np.max(probabilities)),
                    "positive_cell_count": int(np.count_nonzero(probabilities >= 0.5)),
                }, probabilities
            except Exception:
                failures.append("session_failed" if backend == "gpu" else "inference_failed")
        return {**base, "fallback_reason": failures[0] if failures else "provider_unavailable"}, None
