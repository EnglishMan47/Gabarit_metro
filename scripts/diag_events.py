#!/usr/bin/env python3
"""Gallery of confirmed alarm events of one bag (needs matplotlib).

One sequential pass through the bag; the frame where every new track gets
confirmed is rendered as a zoomed BEV + side view around the object.
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
    ap.add_argument("-o", "--out", type=Path, default=Path("events.png"))
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reader = BagReader(open_bag(args.bag))
    pipe = Pipeline(Config())
    seen = set()
    shots = []
    for i, ts, raw in reader.iterate(0, args.stop):
        res = pipe.process_raw(raw, ts * 1e-9, frame_index=i)
        for ob in res.data["obstacles"]:
            if ob["track_id"] in seen:
                continue
            seen.add(ob["track_id"])
            shots.append((i, ob, res))
        if len(shots) >= args.max:
            break
    n = max(len(shots), 1)
    fig, axes = plt.subplots(n, 2, figsize=(22, 4.2 * n), squeeze=False)
    for row, (i, ob, res) in enumerate(shots):
        lev, tr, geo = res.lev, res.track, res.geo
        s0 = ob["s_m"]
        win = (geo.s > s0 - 25) & (geo.s < s0 + 25)
        ax = axes[row, 0]
        m = win & (np.abs(geo.d) < 6) & (geo.h > -0.3) & (geo.h < 4.5)
        ax.scatter(geo.s[m], geo.d[m], s=1.5, c=geo.h[m], cmap="viridis", vmin=-0.3, vmax=4)
        mm = m & (res.labels == 4)
        ax.scatter(geo.s[mm], geo.d[mm], s=8, c="red")
        for y in (geo.half_width, -geo.half_width):
            ax.axhline(y, color="m", lw=0.8)
        ax.set_title(f"frame {i} #{ob['track_id']} {ob['distance_m']} m d={ob['lateral_m']} h={ob['box']['h']} n={ob['points']} "
                     f"obs={res.data['observed_until_m']} v={res.data['speed_kmh']}")
        ax.set_xlim(s0 - 25, s0 + 25)
        ax.set_ylim(-6, 6)
        ax.grid(True)
        ax2 = axes[row, 1]
        m2 = win & (np.abs(geo.d) < geo.half_width + 1.0) & (geo.h > -0.5) & (geo.h < 4.5)
        ax2.scatter(geo.s[m2], geo.h[m2], s=1.5, c=np.abs(geo.d[m2]), cmap="coolwarm", vmin=0, vmax=geo.half_width + 1)
        ax2.scatter(geo.s[mm], geo.h[mm], s=8, c="red")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axhline(3, color="m", lw=0.5)
        ax2.set_xlim(s0 - 25, s0 + 25)
        ax2.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=50)
    print("events:", len(shots), "saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
