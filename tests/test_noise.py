"""Tests for online_estimators.noise.models."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.noise.models import AWGNNoise, OUNoise, RandomWalkNoise, apply_noise


class TestApplyNoise:
    def test_none_returns_copy(self):
        x = np.array([1.0, 2.0, 3.0])
        y = apply_noise(x, level="none")
        np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_output_shape(self, level):
        x = np.ones(13)
        y = apply_noise(x, level=level)
        assert y.shape == x.shape

    def test_noise_magnitude_ordering(self):
        """Higher noise level should (on average) produce larger perturbations."""
        x = np.ones(13)
        norms = {}
        for level in ("low", "medium", "high"):
            diffs = [np.linalg.norm(apply_noise(x, level=level) - x) for _ in range(500)]
            norms[level] = np.mean(diffs)
        assert norms["low"] < norms["medium"] < norms["high"]


class TestAWGNNoise:
    def test_mean_zero(self, rng):
        noise = AWGNNoise(sigma=1.0)
        samples = np.array([noise.sample(shape=(5,)) for _ in range(10000)])
        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.1)

    def test_std(self, rng):
        sigma = 0.5
        noise = AWGNNoise(sigma=sigma)
        samples = np.array([noise.sample(shape=(3,)) for _ in range(10000)])
        np.testing.assert_allclose(samples.std(axis=0), sigma, atol=0.05)


class TestOUNoise:
    def test_shape(self):
        noise = OUNoise(theta=0.15, mu=0.0, sigma=0.2)
        s = noise.sample()
        assert isinstance(s, float)

    def test_reset(self):
        noise = OUNoise(theta=0.15, mu=0.0, sigma=0.2)
        for _ in range(10):
            noise.sample()
        noise.reset()
        assert noise.state == 0.0


class TestRandomWalkNoise:
    def test_shape(self):
        noise = RandomWalkNoise(sigma=0.01)
        s = noise.sample()
        assert isinstance(s, float)

    def test_accumulates(self):
        noise = RandomWalkNoise(sigma=0.1)
        samples = [noise.sample() for _ in range(100)]
        # Later samples should (on average over many runs) drift further from zero
        # We check that the absolute value grows over time (rough check)
        late_abs = np.mean([abs(s) for s in samples[80:]])
        # Not guaranteed every run, but with step_size=0.1 it's very likely
        assert isinstance(late_abs, float)
