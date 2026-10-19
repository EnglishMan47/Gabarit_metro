"""Command line of the project (no Docker, no ROS needed).

    python -m gabarit web      [--data DIR]          web interface, http://localhost:8000
    python -m gabarit run      BAG [--realtime]      process one bag, print the decision per frame
    python -m gabarit evaluate DIR [--out DIR]       all bags: per-frame JSONL + summary
    python -m gabarit synthetic DIR [--out DIR]      detection range with synthetic obstacles
    python -m gabarit check                          environment and data check
    python -m gabarit mlflow   [--port 5000]         MLflow server: evaluation runs and detector versions
"""

from __future__ import annotations

import gabarit  # noqa: F401  (single-threaded BLAS before NumPy)

import argparse
import json
import math
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "extracted" / "for_hackathon"


def cmd_run(args: argparse.Namespace) -> int:
    from gabarit.config import Config
    from gabarit.pipeline import Pipeline
    from gabarit.rosbag import BagReader, open_bag

    cfg = Config.load(args.config)
    reader = BagReader(open_bag(args.bag))
    pipe = Pipeline(cfg)
    out = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    wall_start = time.perf_counter()
    first_stamp = None
    print(f"{reader.info.name}: {len(reader)} frames, topic {reader.info.topic}")
    print("frame  time_s  status    obstacle  distance_m  free_m  speed_kmh  latency_ms")
    try:
        for i, ts, raw in reader.iterate():
            if first_stamp is None:
                first_stamp = ts
            if args.realtime:
                # Anchor playback to recording time: sleeping a whole frame
                # period *after* processing would turn 10 Hz into ~6-7 Hz.
                target = wall_start + (ts - first_stamp) * 1e-9 / args.rate
                time.sleep(max(0.0, target - time.perf_counter()))
            d = pipe.process_raw(raw, ts * 1e-9, frame_index=i).data
            dist = "-" if d["nearest_distance_m"] is None else f"{d['nearest_distance_m']:.1f}"
            print(f"{i:5d}  {(ts - reader.info.start_ns) / 1e9:6.2f}  {d['status']:8s}  {str(d['obstacle']):8s}  "
                  f"{dist:>10s}  {d['free_distance_m']:6.0f}  {d['speed_kmh']:9.1f}  {d['timing_ms']['total']:10.1f}")
            if out:
                out.write(json.dumps(d, ensure_ascii=False, default=float) + "\n")
    finally:
        reader.close()
        if out:
            out.close()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    import platform

    import numpy as np

    ok = True
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    for mod in ("aiohttp", "yaml"):
        try:
            __import__(mod)
            print(f"  {mod}: ok")
        except ImportError:
            print(f"  {mod}: MISSING (pip install -r requirements.txt)")
            ok = False
    try:
        import rclpy  # noqa: F401
        print("  ROS 2 (rclpy): available")
    except ImportError:
        print("  ROS 2 (rclpy): not installed - not needed for the web interface and batch processing")
    from gabarit.rosbag import discover_bags

    data = Path(args.data)
    bags = discover_bags(data)
    print(f"Data: {data} - {len(bags)} bags")
    for b in bags:
        print(f"  {b.name}: {b.frame_count} frames, {b.duration_s:.1f} s, topic {b.topic}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gabarit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("web", help="web interface")
    w.add_argument("--data", action="append",
                   help="folder with bags, may be repeated (default: data/extracted recursively)")
    w.add_argument("--config", default=None)
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", default="8000")
    w.add_argument("--max-users", default=None, help="simultaneous users, 0 = unlimited (default 1)")
    w.add_argument("--uploads", default=None, help="folder for uploaded recordings (default data/uploads)")
    r = sub.add_parser("run", help="process one bag and print the decision per frame")
    r.add_argument("bag", help="bag directory or .db3 file")
    r.add_argument("--config", default=None)
    r.add_argument("--realtime", action="store_true", help="pace frames like the sensor (10 Hz)")
    r.add_argument("--rate", type=float, default=1.0, help="playback speed with --realtime (e.g. 0.5)")
    r.add_argument("--jsonl", default=None, help="also write the full result of every frame")
    e = sub.add_parser("evaluate", help="all bags of a directory")
    e.add_argument("data", nargs="?", default=str(DEFAULT_DATA))
    e.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("synthetic", help="detection range with synthetic obstacles")
    s.add_argument("data", nargs="?", default=str(DEFAULT_DATA))
    s.add_argument("rest", nargs=argparse.REMAINDER)
    m = sub.add_parser("mlflow", help="MLflow tracking server with the project store (model management)")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", default="5000")
    c = sub.add_parser("check", help="check the environment and the data directory")
    c.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args(argv)
    if args.cmd == "run" and (not math.isfinite(args.rate) or args.rate <= 0):
        ap.error("--rate must be finite and positive")

    sys.path.insert(0, str(ROOT))
    if args.cmd == "web":
        sys.argv = ["web.server", "--host", args.host, "--port", args.port]
        for d in args.data or []:
            sys.argv += ["--data", d]
        if args.max_users is not None:
            sys.argv += ["--max-users", args.max_users]
        if args.uploads:
            sys.argv += ["--uploads", args.uploads]
        if args.config:
            sys.argv += ["--config", args.config]
        runpy.run_module("web.server", run_name="__main__")
        return 0
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "mlflow":
        from gabarit.mlops import serve_ui
        return serve_ui(args.host, int(args.port))
    if args.cmd in ("evaluate", "synthetic"):
        script = ROOT / "scripts" / ("evaluate.py" if args.cmd == "evaluate" else "synthetic_eval.py")
        sys.argv = [str(script), args.data, *args.rest]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    if args.cmd == "check":
        return cmd_check(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
