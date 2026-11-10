# Adapted from BelousovArts/fod, commit baebf57a4169697aa61d017fcea8e2374ef135a0.
# No upstream license was published at that revision; see NOTICE.md.
"""Дальний вид сверху, выпрямленный вдоль текущей оси и накопленный по позам.

Ось — вариант «тоннель + карта» с парами головок от сети по range image, за её
дальностью — продолжение с постоянной кривизной. Кадры копятся в мире по позам
small_gicp и переносятся в текущий кадр. Точка → (s, d, u): вдоль оси, поперёк неё
по нормали, высота над головками.

Растр: s 10…`S1` м через 0.5 м, d ±10 м через 0.1 м (200 столбцов; сети идёт ±8 м).
Набор для обучения — шире, ±`D_WIDE` м: при обучении растр выпрямляется заново вдоль
эталона с шумом (как второй проход каскада), а дальше 200 м ось выпрямления уходит от пути
на 5…20 м. Каналы, uint8: число точек в слоях высоты, средняя
интенсивность, средняя высота низких точек и наибольшая высота в ячейке (по ним сеть
видит, насколько пол и свод съехали от высоты оси). Станции оси — центры полос 2 м:
11, 13, …, S1 − 1 м; ось выдаётся до S1 − 10 м, дальше — запас по контексту.

Длина растра — свойство сети (`s1` в чекпойнте): новые наборы и сети — до `S1` = 330 м,
сеть без `s1` — первая, до 170 м.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np

GRID = np.arange(0.0, 160.5, 0.5)
S0, S1, DS = 10.0, 330.0, 0.5
S1_V1 = 170.0
GRID_BEV = np.arange(0.0, S1 + 0.5, 0.5)
D_STORE, D_NET, DD = 10.0, 8.0, 0.1
D_WIDE = 40.0
N_S = int(round((S1 - S0) / DS))
N_D = int(round(2 * D_STORE / DD))
N_D_NET = int(round(2 * D_NET / DD))
MARGIN = (N_D - N_D_NET) // 2
# Слои высоты над головками, м: лоток, полотно с рельсами, низ стен и КР, стены, верх стен, свод.
LAYERS = ((-1.5, -0.3), (-0.3, 0.25), (0.25, 1.2), (1.2, 2.5), (2.5, 4.0), (4.0, 8.0))
# Средняя высота точек ниже LOW_TOP и наибольшая высота: 0 — пусто, 1…255 — от LAYERS[0][0] до верха.
LOW_TOP = 1.0
N_CH = len(LAYERS) + 3
ST_DS = 2.0
STATIONS = np.arange(S0 + ST_DS / 2, S1, ST_DS)
K_MAX = 10
GAP_S = 0.55
# Вбок от лидара в буфере: на дуге к 330 м путь уходит на десятки метров, растр — ±D_WIDE от оси.
SIDE = 100.0


def n_rows(s1: float = S1) -> int:
    return int(round((s1 - S0) / DS))


def stations(s1: float = S1) -> np.ndarray:
    return np.arange(S0 + ST_DS / 2, s1, ST_DS)


def bev_grid(s1: float = S1) -> np.ndarray:
    return np.arange(0.0, s1 + 0.5, 0.5)


def extend_curved(grid: np.ndarray, values: np.ndarray, reach: float, span: float = 40.0, k_max: float = 1.0 / 250.0):
    """За `reach` — парабола по последним `span` м, кривизна не круче `k_max`."""
    out = values.copy()
    last = np.flatnonzero(grid <= reach)
    if last.size < 2:
        return out
    last = int(last[-1])
    fit = (grid > reach - span) & (grid <= reach)
    x = grid[fit] - grid[last]
    if fit.sum() >= 6 and -x[0] >= 15.0:
        c, b, _a = np.polyfit(x, values[fit], 2)
        c = float(np.clip(c, -0.5 * k_max, 0.5 * k_max))
    elif fit.sum() >= 2:
        (b, _a), c = np.polyfit(x, values[fit], 1), 0.0
    else:
        return out
    far = grid > grid[last]
    d = grid[far] - grid[last]
    out[far] = values[last] + b * d + c * d * d
    return out


def extend_axis(grid: np.ndarray, n: np.ndarray, z: np.ndarray, reach: float, out: np.ndarray = GRID_BEV):
    """Ось до `reach` → n и z на `out`: вбок парабола, по высоте прямая."""
    reach = min(reach, float(grid[-1]))
    n2, z2 = np.interp(out, grid, n), np.interp(out, grid, z)
    return extend_curved(out, n2, reach), extend_curved(out, z2, reach, k_max=0.0)


def encode_u(u: np.ndarray, top: float) -> np.ndarray:
    lo = LAYERS[0][0]
    return np.clip(np.rint(1.0 + 254.0 * (u - lo) / (top - lo)), 1, 255)


def straighten(pts: np.ndarray, grid: np.ndarray, n_ax: np.ndarray, z_ax: np.ndarray):
    """Точки кадра → (s, d, u)."""
    s = -pts[:, 1]
    slope = np.gradient(n_ax, grid)
    d = (pts[:, 0] - np.interp(s, grid, n_ax)) / np.sqrt(1.0 + np.interp(s, grid, slope) ** 2)
    u = pts[:, 2] - np.interp(s, grid, z_ax)
    return s, d, u


def normal_offset(s: np.ndarray, n: np.ndarray, grid: np.ndarray, n_ax: np.ndarray) -> np.ndarray:
    """Боковое отклонение линии (s, n) от оси по нормали."""
    slope = np.gradient(n_ax, grid)
    return (n - np.interp(s, grid, n_ax)) / np.sqrt(1.0 + np.interp(s, grid, slope) ** 2)


def raster(s: np.ndarray, d: np.ndarray, u: np.ndarray, intensity: np.ndarray, n_s: int = N_S, d_store: float = D_STORE) -> np.ndarray:
    """(N_CH, n_s, n_d) uint8: число точек по слоям (до 255), средняя интенсивность,
    средняя высота точек ниже `LOW_TOP` и наибольшая высота."""
    n_d = int(round(2 * d_store / DD))
    r = np.floor((s - S0) / DS).astype(np.int64)
    c = np.floor((d + d_store) / DD).astype(np.int64)
    ok = (r >= 0) & (r < n_s) & (c >= 0) & (c < n_d) & (u >= LAYERS[0][0]) & (u < LAYERS[-1][1])
    r, c, u, inten = r[ok], c[ok], u[ok], intensity[ok].astype(np.float64)
    flat = r * n_d + c
    layer = np.searchsorted([lo for lo, _ in LAYERS[1:]], u, side="right")
    cnt = np.bincount(layer * (n_s * n_d) + flat, minlength=len(LAYERS) * n_s * n_d).reshape(len(LAYERS), n_s, n_d)
    tot = cnt.sum(axis=0)
    i_sum = np.bincount(flat, weights=inten, minlength=n_s * n_d).reshape(n_s, n_d)
    low = u < LOW_TOP
    n_low = np.bincount(flat[low], minlength=n_s * n_d)
    u_low = np.bincount(flat[low], weights=u[low], minlength=n_s * n_d) / np.maximum(n_low, 1)
    u_max = np.full(n_s * n_d, -np.inf)
    np.maximum.at(u_max, flat, u)
    out = np.empty((N_CH, n_s, n_d), np.uint8)
    n_l = len(LAYERS)
    out[:n_l] = np.minimum(cnt, 255)
    out[n_l] = np.clip(np.rint(i_sum / np.maximum(tot, 1)), 0, 255)
    out[n_l + 1] = np.where(n_low > 0, encode_u(u_low, LOW_TOP), 0).reshape(n_s, n_d)
    out[n_l + 2] = np.where(np.isfinite(u_max), encode_u(np.where(np.isfinite(u_max), u_max, 0.0), LAYERS[-1][1]), 0).reshape(n_s, n_d)
    return out


