import numpy as np

def make_traj_hover(T: float, dt: float, rng: np.random.Generator):
    t = np.arange(0, T, dt)
    pos = np.zeros((len(t), 3))
    vel = np.zeros_like(pos)
    return t, pos, vel


def make_traj_figure8(T: float, dt: float, rng: np.random.Generator):
    A = rng.uniform(0.4, 0.8)
    B = rng.uniform(0.2, 0.4)
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2 * np.pi / 20.0) * rng.uniform(0.8, 1.25)
    t = np.arange(0, T, dt)
    x = A * np.sin(omega * t)
    y = B * np.sin(2 * omega * t)
    z = z0 * np.ones_like(t)
    vx = A * omega * np.cos(omega * t)
    vy = 2 * B * omega * np.cos(2 * omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T


def make_traj_circle(T: float, dt: float, rng: np.random.Generator):
    Rr = rng.uniform(0.3, 0.6)
    omega = (2 * np.pi / 15.0) * rng.uniform(0.8, 1.2)
    z0 = rng.uniform(-0.05, 0.05)
    t = np.arange(0, T, dt)
    x = Rr * np.cos(omega * t)
    y = Rr * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    vx = -Rr * omega * np.sin(omega * t)
    vy = Rr * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T


def make_traj_lissajous(T: float, dt: float, rng: np.random.Generator):
    Ax = rng.uniform(0.35, 0.7)
    Ay = rng.uniform(0.2, 0.5)
    wx = (2 * np.pi / 18.0) * rng.uniform(0.8, 1.2)
    wy = 3 * wx
    phi = rng.uniform(0, np.pi / 2)
    z0 = rng.uniform(-0.05, 0.05)
    t = np.arange(0, T, dt)
    x = Ax * np.sin(wx * t + phi)
    y = Ay * np.sin(wy * t)
    z = z0 * np.ones_like(t)
    vx = Ax * wx * np.cos(wx * t + phi)
    vy = Ay * wy * np.cos(wy * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T


def make_traj_ellipse(T: float, dt: float, rng: np.random.Generator):
    Ax = rng.uniform(0.5, 0.9)  # semi-major
    By = rng.uniform(0.2, 0.5)  # semi-minor
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2 * np.pi / 18.0) * rng.uniform(0.9, 1.2)
    t = np.arange(0, T, dt)
    x = Ax * np.cos(omega * t)
    y = By * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    vx = -Ax * omega * np.sin(omega * t)
    vy = By * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T


def make_traj_helix(T: float, dt: float, rng: np.random.Generator):
    Rr = rng.uniform(0.3, 0.6)
    omega_xy = (2 * np.pi / 16.0) * rng.uniform(0.9, 1.2)
    z0 = rng.uniform(-0.05, 0.05)
    z_amp = rng.uniform(0.04, 0.10)
    omega_z = omega_xy * rng.uniform(0.4, 0.8)
    t = np.arange(0, T, dt)
    x = Rr * np.cos(omega_xy * t)
    y = Rr * np.sin(omega_xy * t)
    z = z0 + z_amp * np.sin(omega_z * t)
    vx = -Rr * omega_xy * np.sin(omega_xy * t)
    vy = Rr * omega_xy * np.cos(omega_xy * t)
    vz = z_amp * omega_z * np.cos(omega_z * t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T


def make_traj_spiral(T: float, dt: float, rng: np.random.Generator):
    R0 = rng.uniform(0.2, 0.35)
    R1 = rng.uniform(0.45, 0.7)
    z0 = rng.uniform(-0.05, 0.05)
    omega = (2 * np.pi / 18.0) * rng.uniform(0.8, 1.2)
    t = np.arange(0, T, dt)
    k = (R1 - R0) / max(T, 1e-6)  # radial growth rate
    R_t = R0 + k * t
    x = R_t * np.cos(omega * t)
    y = R_t * np.sin(omega * t)
    z = z0 * np.ones_like(t)
    vx = k * np.cos(omega * t) - R_t * omega * np.sin(omega * t)
    vy = k * np.sin(omega * t) + R_t * omega * np.cos(omega * t)
    vz = np.zeros_like(t)
    return t, np.vstack([x, y, z]).T, np.vstack([vx, vy, vz]).T