#!/usr/bin/env python3
"""Measure complete local HTTP frame responses from the running web app."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--response", choices=("packet", "json"), default="packet")
    parser.add_argument("--inference-mode", choices=("auto", "gpu", "cpu", "geometry"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/http-benchmark.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with urlopen(f"{args.base_url}/api/bags", timeout=10) as response:
        bags = json.load(response)["bags"]
    if not bags:
        raise SystemExit("server returned no bags")
    bag = bags[0]
    count = min(args.frames, bag["frame_count"])
    if args.warmup < 0:
        raise SystemExit("warmup must not be negative")
    for index in range(min(args.warmup, count)):
        query = urlencode({
            "bag": bag["id"], "index": index, "response": args.response,
            "inference_mode": args.inference_mode,
        })
        with urlopen(f"{args.base_url}/api/frame?{query}", timeout=10) as response:
            response.read()
    times = []
    sizes = []
    last = None
    started = perf_counter()
    for index in range(count):
        query = urlencode({
            "bag": bag["id"],
            "index": index,
            "response": args.response,
            "inference_mode": args.inference_mode,
        })
        request_started = perf_counter()
        with urlopen(f"{args.base_url}/api/frame?{query}", timeout=10) as response:
            body = response.read()
        times.append((perf_counter() - request_started) * 1000)
        sizes.append(len(body))
        if args.response == "packet":
            header_length = struct.unpack_from("<I", body)[0]
            last = json.loads(body[4 : 4 + header_length])
        else:
            last = json.loads(body)
    wall_s = perf_counter() - started
    report = {
        "base_url": args.base_url,
        "bag": bag["name"],
        "frames": count,
        "warmup_frames": min(args.warmup, count),
        "response": args.response,
        "inference_mode": args.inference_mode,
        "http_ms": {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
        },
        "response_bytes": {"mean": float(np.mean(sizes)), "max": max(sizes)},
        "sequential_fps": count / wall_s,
        "last_frame": last["frame_index"] if last else None,
        "scene_model_loaded": bool(last and last["diagnostics"]["scene_model_loaded"]),
        "active_backend": last["inference"]["active_backend"] if last else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
