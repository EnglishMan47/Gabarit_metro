#!/usr/bin/env python3
"""Diagnostic rendering of one frame (needs matplotlib): BEV + side view in
track coordinates with the estimated path, walls, envelope and candidates.

    python scripts/diag_frame.py data/extracted/for_hackathon/<bag> 120 --warmup 10 -o frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabarit.config import Config  # noqa: E402
from gabarit.pipeline import Pipeline  # noqa: E402
from gabarit.rosbag import BagReader, open_bag  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--fmax", type=float, default=200)
    ap.add_argument("--config", type=Path, default=None, help="sensor profile, including Falcon axes")
    ap.add_argument("-o", "--out", type=Path, default=Path("diag.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reader = BagReader(open_bag(args.bag))
    fig, axes = plt.subplots(2 * len(args.frames), 1, figsize=(26, 9 * len(args.frames)))
    for k, target in enumerate(args.frames):
        pipe = Pipeline(Config.load(args.config))
        res = None
        for i in range(max(0, target - args.warmup), target + 1):
            ts, raw = reader.read(i)
            res = pipe.process_raw(raw, ts * 1e-9, frame_index=i)
        lev, tr, geo, d = res.lev, res.track, res.geo, res.data
        ax = axes[2 * k]
        m = (lev[:, 1] > 0) & (lev[:, 1] < args.fmax) & (np.abs(lev[:, 0] - np.interp(lev[:, 1], tr.s, tr.c)) < 8) \
            & (lev[:, 2] > 0.2) & (lev[:, 2] < 3.2)
        ax.scatter(lev[m, 1], lev[m, 0], s=0.4, c=lev[m, 2], cmap="viridis", vmin=0, vmax=3)
        g = (res.labels == 1)
        ax.scatter(lev[g, 1], lev[g, 0], s=0.2, c="0.7")
        for cls, col in ((2, "orange"), (3, "red"), (4, "magenta")):
            mm = res.labels == cls
            ax.scatter(lev[mm, 1], lev[mm, 0], s=4 if cls > 2 else 2, c=col)
        ax.plot(tr.s, tr.c, "k-", lw=1)
        hw = geo.half_width
        ax.plot(tr.s, tr.c + hw, "m-", lw=0.7)
        ax.plot(tr.s, tr.c - hw, "m-", lw=0.7)
        ax.plot(tr.s, tr.wall_left, "r-", lw=1.5, alpha=0.5)
        ax.plot(tr.s, tr.wall_right, "b-", lw=1.5, alpha=0.5)
        ax.plot(tr.s[tr.rail], tr.c[tr.rail], "g.", ms=4)
        ax.axvline(tr.observed_until, color="orange")
        lo = np.nanmin(tr.c[tr.s < args.fmax]) - 5
        hi = np.nanmax(tr.c[tr.s < args.fmax]) + 5
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, args.fmax)
        ax.set_xticks(range(0, int(args.fmax) + 1, 10))
        ax.grid(True)
        ax.set_title(f"{args.bag.name} #{target}: {d['status']} v={d['speed_kmh']} km/h, observed {d['observed_until_m']} m, "
                     f"calib {d['calibration']}")
        for c in d["candidates"]:
            if c["rejected"] is None:
                ax.annotate(f"{c['id']}:{c['distance_m']:.0f}m", (c["s_m"], np.interp(c["s_m"], tr.s, tr.c) + c["lateral_m"]),
                            color="red", fontsize=8)
        ax2 = axes[2 * k + 1]
        inside = geo.in_range & (np.abs(geo.d) < hw + 0.5) & (geo.h > -0.6) & (geo.h < 4.5)
        ax2.scatter(geo.s[inside], geo.h[inside], s=0.4, c=np.abs(geo.d[inside]), cmap="coolwarm", vmin=0, vmax=hw + 0.5)
        for cls, col in ((3, "red"), (4, "magenta")):
            mm = res.labels == cls
            ax2.scatter(geo.s[mm], geo.h[mm], s=5, c=col)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axhline(3.0, color="m", lw=0.5)
        ax2.plot(tr.s, np.zeros_like(tr.s), "k-")
        ax2.set_xlim(0, args.fmax)
        ax2.set_xticks(range(0, int(args.fmax) + 1, 10))
        ax2.grid(True)
        ax2.set_title("side view in track frame (|d| < W/2 + 0.5)")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=55)
    reader.close()
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
