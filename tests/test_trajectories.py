"""Tests for online_estimators.trajectories.generators."""

from __future__ import annotations

import numpy as np
import pytest

from online_estimators.trajectories.generators import (
    TRAJECTORY_REGISTRY,
    make_traj_circle,
    make_traj_ellipse,
    make_traj_figure8,
    make_traj_helix,
    make_traj_hover,
    make_traj_lissajous,
    make_traj_spiral,
)

GENERATORS = [
    make_traj_hover,
    make_traj_figure8,
    make_traj_circle,
    make_traj_lissajous,
    make_traj_ellipse,
    make_traj_helix,
    make_traj_spiral,
]


class TestTrajectoryRegistry:
    def test_all_registered(self):
        expected = {"hover", "figure8", "circle", "lissajous", "ellipse", "helix", "spiral"}
        assert set(TRAJECTORY_REGISTRY.keys()) == expected


class TestAllTrajectories:
    @pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.__name__)
    def test_output_shapes(self, gen, rng):
        T, dt = 5.0, 0.01
        t, pos, vel = gen(T, dt, rng)
        N = len(t)
        assert pos.shape == (N, 3)
        assert vel.shape == (N, 3)

    @pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.__name__)
    def test_time_array(self, gen, rng):
        T, dt = 2.0, 0.01
        t, _, _ = gen(T, dt, rng)
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(T - dt, abs=dt)
        np.testing.assert_allclose(np.diff(t), dt, atol=1e-10)

    @pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.__name__)
    def test_finite_values(self, gen, rng):
        t, pos, vel = gen(3.0, 0.01, rng)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))


class TestHover:
    def test_stationary(self, rng):
        _, pos, vel = make_traj_hover(2.0, 0.01, rng)
        # Hover: all positions identical, velocity zero
        np.testing.assert_allclose(vel, 0.0, atol=1e-10)
        for i in range(3):
            np.testing.assert_allclose(pos[:, i], pos[0, i], atol=1e-10)


class TestFigure8:
    def test_periodic_xy(self, rng):
        T = 10.0
        _, pos, _ = make_traj_figure8(T, 0.01, rng)
        # x-y should vary
        assert np.std(pos[:, 0]) > 0.01
        assert np.std(pos[:, 1]) > 0.01
