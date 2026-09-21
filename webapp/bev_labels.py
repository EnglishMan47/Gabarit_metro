"""Rasterize provisional 3D annotations onto the fixed BEV grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from webapp.bev import DEFAULT_BEV_SPEC, BevSpec


def load_frame_boxes(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[int, list[dict[str, Any]]] = {}
    for name, label in payload.get("labels", {}).items():
        if name == "secondary_person":
            continue
        for item in label.get("frames", []):
            if name == "primary_person" and not item.get("inside_gauge", False):
                continue
            if "min_m" not in item or "max_m" not in item:
                continue
            lookup.setdefault(int(item["frame"]), []).append({
                "name": name,
                "min_m": item["min_m"],
                "max_m": item["max_m"],
            })
    return lookup


def rasterize_boxes(boxes: list[dict[str, Any]], spec: BevSpec = DEFAULT_BEV_SPEC) -> np.ndarray:
    mask = np.zeros((spec.height, spec.width), dtype=np.uint8)
    for box in boxes:
        low, high = box["min_m"], box["max_m"]
        x0 = int(np.floor((float(low["x"]) - spec.min_x) / spec.resolution_x))
        x1 = int(np.floor((float(high["x"]) - spec.min_x) / spec.resolution_x))
        y0 = int(np.floor((float(low["forward"]) - spec.min_forward) / spec.resolution_forward))
        y1 = int(np.floor((float(high["forward"]) - spec.min_forward) / spec.resolution_forward))
        x0, x1 = max(0, x0), min(spec.width - 1, x1)
        y0, y1 = max(0, y0), min(spec.height - 1, y1)
        if x0 <= x1 and y0 <= y1:
            mask[y0:y1 + 1, x0:x1 + 1] = 1
    return mask
