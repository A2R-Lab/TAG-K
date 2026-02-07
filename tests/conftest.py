"""Shared fixtures for TAG-K tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def simple_system():
    """A small consistent linear system ``Ax = b`` with known solution.

    System: 6×3, A is full column rank, x_true = [1, 2, 3].
    """
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    x_true = np.array([1.0, 2.0, 3.0])
    b = A @ x_true
    return A, b, x_true


@pytest.fixture()
def overdetermined_noisy(rng):
    """A noisy overdetermined system (100×5)."""
    m, n = 100, 5
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    return A, b, x_true
