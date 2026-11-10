"""Small fixed-band solvers for the path's third-difference penalty.

The smoothing normal matrix has seven diagonals. A banded LDL factorisation
and selected inverse calculate exactly the same normal-equation solution and
covariance diagonal without constructing or inverting a dense square matrix.
Only NumPy is needed; no optional native compiler or runtime is required.
"""

from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=32)
def third_difference_bands(n: int) -> np.ndarray:
    """Return the four lower bands of D3.T @ D3, indexed [offset, row]."""
    if n < 1:
        raise ValueError("the profile grid must not be empty")
    bands = np.zeros((4, n), dtype=np.float64)
    count = max(0, n - 3)
    coefficients = (-1., 3., -3., 1.)
    for i in range(4):
        for j in range(i + 1):
            bands[i - j, i:i + count] += coefficients[i] * coefficients[j]
    bands.flags.writeable = False
    return bands


class BandedLDL:
    """Factor a symmetric positive definite matrix with half-bandwidth three.

    ``bands[d, i]`` is A[i, i-d]; missing lower diagonals must be zero.
    The three unrolled recurrences avoid tiny NumPy calls in the inner loop.
    """

    def __init__(self, bands: np.ndarray):
        bands = np.asarray(bands, dtype=np.float64)
        if bands.ndim != 2 or bands.shape[0] != 4 or not bands.shape[1]:
            raise ValueError("expected four bands of a non-empty matrix")
        if not np.isfinite(bands).all():
            raise np.linalg.LinAlgError("non-finite smoothing matrix")
        self.n = n = bands.shape[1]
        a0, a1, a2, a3 = bands.tolist()
        # Three leading padding entries make the boundary recurrence identical
        # to the interior. Trailing entries are for the reverse recurrences.
        d = [1.] * 3 + [0.] * n + [1.] * 3
        l1, l2, l3 = ([0.] * (n + 6) for _ in range(3))
        for r in range(n):
            i = r + 3
            c3 = a3[r] / d[i - 3]
            c2 = (a2[r] - c3 * d[i - 3] * l1[i - 2]) / d[i - 2]
            c1 = (a1[r] - c2 * d[i - 2] * l1[i - 1]
                  - c3 * d[i - 3] * l2[i - 1]) / d[i - 1]
            pivot = a0[r] - (c1 * c1 * d[i - 1] + c2 * c2 * d[i - 2]
                              + c3 * c3 * d[i - 3])
            if not pivot > 0. or not math.isfinite(pivot):
                raise np.linalg.LinAlgError("smoothing matrix is not positive definite")
            l1[i], l2[i], l3[i], d[i] = c1, c2, c3, pivot
        self.d, self.l1, self.l2, self.l3 = d, l1, l2, l3

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (self.n,):
            raise ValueError("right hand side does not match the profile grid")
        n = self.n
        d, l1, l2, l3 = self.d, self.l1, self.l2, self.l3
        y = [0.] * 3 + rhs.tolist() + [0.] * 3
        for i in range(3, n + 3):
            y[i] -= l1[i] * y[i - 1] + l2[i] * y[i - 2] + l3[i] * y[i - 3]
        x = [0.] * (n + 6)
        for i in range(n + 2, 2, -1):
            x[i] = (y[i] / d[i] - l1[i + 1] * x[i + 1]
                    - l2[i + 2] * x[i + 2] - l3[i + 3] * x[i + 3])
        return np.array(x[3:n + 3])

    def inverse_diagonal(self) -> np.ndarray:
        """Selected-inverse recursion; compute diag(A^-1) in linear space/time."""
        n = self.n
        d, l1, l2, l3 = self.d, self.l1, self.l2, self.l3
        z0, z1, z2, z3 = ([0.] * (n + 6) for _ in range(4))
        for i in range(n + 2, 2, -1):
            z3[i] = -(l1[i + 1] * z2[i + 1] + l2[i + 2] * z1[i + 2]
                      + l3[i + 3] * z0[i + 3])
            z2[i] = -(l1[i + 1] * z1[i + 1] + l2[i + 2] * z0[i + 2]
                      + l3[i + 3] * z1[i + 2])
            z1[i] = -(l1[i + 1] * z0[i + 1] + l2[i + 2] * z1[i + 1]
                      + l3[i + 3] * z2[i + 1])
            z0[i] = (1. / d[i] - l1[i + 1] * z1[i] - l2[i + 2] * z2[i]
                     - l3[i + 3] * z3[i])
        return np.array(z0[3:n + 3])
