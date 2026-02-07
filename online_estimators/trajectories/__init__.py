"""
Reference trajectory generators for quadrotor simulations.
"""

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

__all__ = [
    "TRAJECTORY_REGISTRY",
    "make_traj_hover",
    "make_traj_figure8",
    "make_traj_circle",
    "make_traj_lissajous",
    "make_traj_ellipse",
    "make_traj_helix",
    "make_traj_spiral",
]
