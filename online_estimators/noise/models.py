"""
Noise models for sensor simulation.

Provides additive noise generators with different temporal correlation
structures, suitable for Monte-Carlo simulation of measurement systems.
"""

from __future__ import annotations

import numpy as np


def apply_noise(
    x: np.ndarray,
    level: str = "none",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply position/velocity noise to a 13-dimensional quadrotor state.

    Parameters
    ----------
    x : np.ndarray, shape ``(13,)``
        State vector ``[r(3), q(4), v(3), omega(3)]``.
    level : str
        One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
    rng : np.random.Generator, optional
        Random number generator.  Falls back to the global NumPy RNG
        when *None*.

    Returns
    -------
    np.ndarray, shape ``(13,)``
        Noisy state.
    """
    x_noisy = x.copy()
    if level == "none":
        return x_noisy

    noise_cfg = {
        "low": (1e-5, 1e-6),
        "medium": (1e-4, 1e-5),
        "high": (5e-4, 5e-5),
    }
    if level not in noise_cfg:
        raise ValueError(f"Unknown noise level '{level}'. Choose from {list(noise_cfg)}")

    pos_std, vel_std = noise_cfg[level]
    if rng is not None:
        x_noisy[0:3] += rng.normal(0.0, pos_std, size=3)
        x_noisy[7:10] += rng.normal(0.0, vel_std, size=3)
    else:
        x_noisy[0:3] += np.random.normal(0.0, pos_std, size=3)
        x_noisy[7:10] += np.random.normal(0.0, vel_std, size=3)
    return x_noisy


class AWGNNoise:
    """Additive White Gaussian Noise (i.i.d. per sample).

    Parameters
    ----------
    sigma : float
        Standard deviation.

    Examples
    --------
    >>> noise = AWGNNoise(sigma=0.01)
    >>> sample = noise.sample(shape=(3,))
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def reset(self) -> None:
        """No-op (stateless noise model)."""
        pass

    def sample(self, shape: tuple = ()) -> np.ndarray:
        """Draw a noise sample.

        Parameters
        ----------
        shape : tuple
            Shape of the output array.

        Returns
        -------
        np.ndarray
        """
        return np.random.normal(0.0, self.sigma, size=shape)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally-correlated noise.

    The OU process is a mean-reverting stochastic process:

        ``dx = theta (mu - x) dt + sigma sqrtdt * N(0,1)``

    Parameters
    ----------
    theta : float
        Mean-reversion rate.
    mu : float
        Long-run mean.
    sigma : float
        Volatility.
    dt : float
        Time-step.
    """

    def __init__(self, theta: float, mu: float, sigma: float, dt: float = 0.01) -> None:
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.state: float = float(mu)

    def reset(self) -> None:
        """Reset the process to its mean."""
        self.state = float(self.mu)

    def sample(self) -> float:
        """Advance one step and return the new state.

        Returns
        -------
        float
        """
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += dx
        return self.state


class RandomWalkNoise:
    """Random walk (Brownian motion) noise model.

    ``b_{t+1} = b_t + sigma sqrtdt * N(0,1)``

    Parameters
    ----------
    sigma : float
        Step standard deviation.
    dt : float
        Time-step.
    """

    def __init__(self, sigma: float, dt: float = 0.01) -> None:
        self.sigma = sigma
        self.dt = dt
        self.state: float = 0.0

    def reset(self) -> None:
        """Reset the walk to zero."""
        self.state = 0.0

    def sample(self) -> float:
        """Advance one step and return the accumulated bias.

        Returns
        -------
        float
        """
        step = self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += step
        return self.state
