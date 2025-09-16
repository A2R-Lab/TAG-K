import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_noise(x, level="none"):
    """
    Apply physically meaningful noise to a 13D state vector.
    
    ref traj vel: 0.1 - 0.6 m/s, acc 0.02 - 0.8 m/s^2
    """
    x_noisy = x.copy()

    if level == "none":
        return x_noisy
    elif level == "low":
        pos_std = 1e-5
        vel_std = 1e-6
    elif level == "medium":
        pos_std = 1e-4
        vel_std = 1e-5
    elif level == "high":
        pos_std = 5e-4
        vel_std = 5e-5

    x_noisy[0:3] += np.random.normal(0.0, pos_std, size=3)  # position noise
    x_noisy[7:10] += np.random.normal(0.0, vel_std, size=3)  # velocity noise

    return x_noisy

class AWGNNoise:
    """
    Additive White Gaussian Noise (AWGN)
    Uncorrelated zero-mean Gaussian noise.
    """

    def __init__(self, sigma: float):
        """
        Args:
            sigma: Standard deviation of the noise.
        """
        self.sigma = sigma

    def reset(self):
        """No internal state to reset for white noise."""
        pass

    def sample(self, shape=()):
        """
        Generate AWGN samples.
        
        Args:
            shape: Tuple indicating the output shape (default scalar).
        
        Returns:
            np.ndarray or float: Noise sample(s).
        """
        return np.random.normal(0.0, self.sigma, size=shape)


class OUNoise:
    """
    Ornstein–Uhlenbeck (OU) process for temporally correlated noise.
    x_{t+1} = x_t + theta*(mu - x_t)*dt + sigma*sqrt(dt)*N(0,1)
    """

    def __init__(self, theta: float, mu: float, sigma: float, dt: float = 0.01):
        """
        Args:
            theta: Rate of mean reversion.
            mu: Long-term mean.
            sigma: Noise intensity.
            dt: Time step.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()

    def reset(self):
        """Reset the process to its mean."""
        self.state = np.array(self.mu, dtype=float)

    def sample(self):
        """
        Advance one time step and return OU noise.
        
        Returns:
            float: New noise value.
        """
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += dx
        return self.state


class RandomWalkNoise:
    """
    Random walk noise: b_{t+1} = b_t + sigma * sqrt(dt) * N(0,1)
    Models slowly drifting biases.
    """

    def __init__(self, sigma: float, dt: float = 0.01):
        """
        Args:
            sigma: Standard deviation of increments.
            dt: Time step.
        """
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()

    def reset(self):
        """Reset the bias to zero."""
        self.state = 0.0

    def sample(self):
        """
        Advance one time step and return random-walk bias.
        
        Returns:
            float: New bias value.
        """
        step = self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.state += step
        return self.state
