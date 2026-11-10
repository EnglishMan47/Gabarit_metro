"""Numerical regression of the path solver against dense normal equations."""

import numpy as np
import pytest

from gabarit.numerics import BandedLDL, third_difference_bands
import gabarit.track as track


def _dense(bands):
    n = bands.shape[1]
    matrix = np.diag(bands[0])
    for offset in range(1, 4):
        if offset < n:
            matrix += np.diag(bands[offset, offset:], -offset)
            matrix += np.diag(bands[offset, offset:], offset)
    return matrix


@pytest.mark.parametrize("n", [1, 2, 3, 4, 31, 205, 355])
def test_banded_solution_and_uncertainty_match_dense(n):
    rng = np.random.default_rng(783 + n)
    differences = np.diff(np.eye(n), 3, axis=0)
    expected_penalty = differences.T @ differences
    bands = third_difference_bands(n)
    np.testing.assert_array_equal(_dense(bands), expected_penalty)
    assert not bands.flags.writeable
    weights = rng.uniform(0.3, 200., n)
    normal = 3e6 * expected_penalty + np.diag(weights)
    regularized = 3e6 * bands.copy()
    regularized[0] += weights
    rhs = rng.normal(size=n)
    solver = BandedLDL(regularized)
    solution = solver.solve(rhs)
    variance = solver.inverse_diagonal()
    np.testing.assert_allclose(solution, np.linalg.solve(normal, rhs), rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(variance, np.diag(np.linalg.inv(normal)), rtol=3e-8, atol=2e-10)
    # A minimum of the weighted quadratic has a vanishing energy gradient.
    np.testing.assert_allclose(normal @ solution, rhs, rtol=2e-8, atol=2e-8)
    assert np.all(variance > 0.)


@pytest.mark.parametrize("side_steps", [False, True])
@pytest.mark.parametrize("horizon", [200., 350.])
def test_robust_profile_parity_with_priors_outliers_and_gaps(monkeypatch, horizon, side_steps):
    x = np.r_[np.arange(3.5, 70., 1.), np.arange(70., horizon, 2.)]
    rng = np.random.default_rng(17)
    truth = 0.01 * x + 0.00008 * x ** 2
    prior = truth + 0.05 * np.sin(x / 20.)
    prior_sig = np.full(len(x), 0.2)
    meas = []
    for i in range(len(x)):
        if 75. < x[i] < 110. or i % 7 == 0:
            continue
        for side in (-1, 1):
            value = truth[i] + rng.normal(0., 0.025)
            if i % 11 == 0:
                value += 0.8  # IRLS must reject the same outliers.
            if side_steps and side == 1 and x[i] > 130.:
                value += 0.3
            meas.append([i, value, 0.04 ** 2, side])
    args = (x, meas, prior, prior_sig, truth, 3e6, 2.5, side_steps)
    actual = track._smooth_profile(*args)

    class DenseReference:
        def __init__(self, bands):
            self.matrix = _dense(bands)

        def solve(self, rhs):
            return np.linalg.solve(self.matrix, rhs)

        def inverse_diagonal(self):
            return np.diag(np.linalg.inv(self.matrix))

    monkeypatch.setattr(track, "BandedLDL", DenseReference)
    expected = track._smooth_profile(*args)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(actual[2], expected[2])


def test_profile_without_measurements_retains_fallback():
    fallback = np.array([0.1, 0.2, 0.4])
    profile, sigma, supported = track._smooth_profile(
        np.array([3., 4., 5.]), [], None, None, fallback, 3e6, 2.5, False)
    np.testing.assert_array_equal(profile, fallback)
    np.testing.assert_array_equal(sigma, [0.5, 0.5, 0.5])
    assert not supported.any()
    assert profile is not fallback


def test_invalid_system_is_rejected():
    with pytest.raises(ValueError):
        BandedLDL(np.empty((4, 0)))
    with pytest.raises(np.linalg.LinAlgError):
        BandedLDL(np.zeros((4, 5)))
    with pytest.raises(np.linalg.LinAlgError):
        BandedLDL(np.full((4, 5), np.nan))
