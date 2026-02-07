"""Tests for online_estimators.simulation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.simulation.utils import abs_residual, closest_spd, rel_residual


class TestResiduals:
    def test_abs_residual_exact(self):
        A = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        b = A @ x
        assert abs_residual(A, x, b) == pytest.approx(0.0, abs=1e-12)

    def test_abs_residual_nonzero(self):
        A = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        r = abs_residual(A, x, b)
        assert r > 0

    def test_rel_residual_exact(self):
        A = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        b = A @ x
        assert rel_residual(A, x, b) == pytest.approx(0.0, abs=1e-12)

    def test_rel_residual_nonzero(self):
        A = np.eye(3)
        x = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        r = rel_residual(A, x, b)
        assert 0 < r < 1


class TestClosestSPD:
    def test_preserves_mass(self):
        theta = np.array([0.04, 0.0, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        result = closest_spd(theta)
        assert result[0] == pytest.approx(theta[0])

    def test_output_has_spd_inertia(self):
        theta = np.array([0.04, 0.0, 0.0, 0.0, 2e-5, 2e-5, 1e-5, 0.0, 0.0, 0.0])
        result = closest_spd(theta)
        J = np.array(
            [
                [result[4], result[7], result[8]],
                [result[7], result[5], result[9]],
                [result[8], result[9], result[6]],
            ]
        )
        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals > 0)

    def test_handles_negative_eigenvalue(self):
        # Construct a theta with non-PD inertia
        theta = np.array([0.04, 0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5, 5e-5, 0.0, 0.0])
        result = closest_spd(theta)
        J = np.array(
            [
                [result[4], result[7], result[8]],
                [result[7], result[5], result[9]],
                [result[8], result[9], result[6]],
            ]
        )
        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals > 0)
