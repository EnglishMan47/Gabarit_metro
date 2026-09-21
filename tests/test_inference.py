from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from webapp.bev import DEFAULT_BEV_SPEC
from webapp.inference import HybridInferenceEngine
from webapp.server import Application


class _FakeInput:
    name = "bev"


class _FakeSession:
    def __init__(self, providers: list[str]):
        self.providers = providers

    def get_inputs(self):
        return [_FakeInput()]

    def get_providers(self):
        return self.providers

    def run(self, _outputs, feed):
        batch = feed["bev"]
        return [np.full((1, 1, batch.shape[2], batch.shape[3]), 0.75, dtype=np.float32)]


class _FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def InferenceSession(_path, providers):
        return _FakeSession(providers)


class InferenceTests(unittest.TestCase):
    def test_missing_model_uses_geometry_fallback(self) -> None:
        engine = HybridInferenceEngine(Path("missing-model.onnx"), import_runtime=False)
        result, probabilities = engine.infer(None, "auto")
        self.assertEqual(result["active_backend"], "geometry")
        self.assertEqual(result["fallback_reason"], "model_missing")
        self.assertIsNone(probabilities)

    def test_gpu_mode_uses_available_cuda_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"placeholder")
            engine = HybridInferenceEngine(model, ort_module=_FakeOrt())
            bev = np.zeros((7, DEFAULT_BEV_SPEC.height, DEFAULT_BEV_SPEC.width), dtype=np.float32)
            result, probabilities = engine.infer(bev, "gpu")
            self.assertEqual(result["active_backend"], "cuda")
            self.assertTrue(result["model_used"])
            self.assertEqual(probabilities.shape, (DEFAULT_BEV_SPEC.height, DEFAULT_BEV_SPEC.width))

    def test_frame_packet_is_aligned_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = Application(root / "bags", root / "feedback.jsonl")
            app.frame = lambda _query: {"frame_index": 4, "points": [[1.0, 2.0, 3.0, 4.0]]}  # type: ignore[method-assign]
            packet = app.frame_packet({})
            header_length = struct.unpack_from("<I", packet)[0]
            header = json.loads(packet[4:4 + header_length])
            point_offset = (4 + header_length + 3) & ~3
            points = np.frombuffer(packet, dtype="<f4", offset=point_offset)
            self.assertEqual(header["point_count"], 1)
            self.assertEqual(header["point_stride"], 4)
            np.testing.assert_array_equal(points, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
