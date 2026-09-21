#!/usr/bin/env python3
"""Verify that the isolated PyTorch environment can execute work on CUDA."""

from __future__ import annotations

import json
import time

import torch


def main() -> int:
    result: dict[str, object] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        print(json.dumps(result, indent=2))
        return 1

    device = torch.device("cuda")
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    product = left @ right
    torch.cuda.synchronize()
    result["matrix_multiply_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["finite"] = bool(torch.isfinite(product).all().item())
    print(json.dumps(result, indent=2))
    return 0 if result["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
