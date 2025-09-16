import numpy as np
import matplotlib.pyplot as plt
from python.common.estimation_methods import RLS, KF, RK, TARK, IGRK, MGRK, REK

# ------------------------------------------------------------
# Simulation utilities
# ------------------------------------------------------------

def generate_system(m, n, noise_level=0.0, drift=0.01, T=200, seed=0):
    """
    Simulate time-varying system A x_true(t) = b(t).
    - A: fixed (m, n)
    - x_true(t): drifts over time
    - b(t): consistent = A @ x_true; inconsistent = A @ x_true + noise
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n))

    # true x starts random, then drifts
    x_true = rng.normal(size=n)

    xs, bs = [], []
    for t in range(T):
        # drift true solution
        drift_vec = drift * rng.normal(size=n)
        x_true = x_true + drift_vec

        b = A @ x_true
        if noise_level > 0:
            b = b + noise_level * rng.normal(size=m)

        xs.append(x_true.copy())
        bs.append(b)

    return A, np.array(xs), np.array(bs)


def run_estimators(A, xs, bs, estimators):
    """
    Run all estimators online.
    Returns dict[name -> trajectory of estimates]
    """
    T = len(xs)
    n = A.shape[1]
    results = {name: np.zeros((T, n)) for name in estimators}

    for t in range(T):
        b = bs[t]
        for name, est in estimators.items():
            x_hat = est.iterate(A, b)
            results[name][t] = x_hat
    return results


def compute_errors(results, xs):
    """Return dict[name -> trajectory of squared errors]."""
    errors = {}
    for name, traj in results.items():
        errors[name] = np.linalg.norm(traj - xs, axis=1) ** 2
    return errors


def plot_errors(errors, title):
    plt.figure(figsize=(8, 5))
    for name, err in errors.items():
        plt.plot(err, label=name)
    plt.xlabel("time step")
    plt.ylabel("squared error ||x - x_true||^2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

if __name__ == "__main__":
    m, n, T = 50, 20, 10

    # Consistent system (drifting solution, no noise)
    A, xs_cons, bs_cons = generate_system(m, n, noise_level=0.0, drift=0.1, T=T)
    # Inconsistent system (drifting solution + noisy b)
    A2, xs_incons, bs_incons = generate_system(m, n, noise_level=0.5, drift=0.1, T=T, seed=1)

    # Define estimators
    estimators_cons = {
        "RLS": RLS(n),
        "KF": KF(n, process_noise=0.01*np.eye(n), measurement_noise=0.1*np.eye(m)),
        "RK": RK(n),
        "TARK": TARK(n),
        "IGRK": IGRK(n),
        "MGRK": MGRK(n, alpha=1.0, beta=0.5, theta=0.5),
        "REK": REK(n),
    }
    # copy fresh instances for inconsistent system
    from copy import deepcopy
    estimators_incons = {k: deepcopy(v) for k,v in estimators_cons.items()}

    # Run consistent system
    results_cons = run_estimators(A, xs_cons, bs_cons, estimators_cons)
    errors_cons = compute_errors(results_cons, xs_cons)

    # Run inconsistent system
    results_incons = run_estimators(A2, xs_incons, bs_incons, estimators_incons)
    errors_incons = compute_errors(results_incons, xs_incons)

    # Plot
    plot_errors(errors_cons, "Consistent drifting system")
    plot_errors(errors_incons, "Inconsistent drifting system (noisy b)")
    plt.show()
