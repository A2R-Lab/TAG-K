#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max([x_range, y_range, z_range]) / 2.0
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

def _load_method_dicts(npz_path, methods=None):
    """Load your saved file (dict_<method> -> python dict)."""
    data = np.load(npz_path, allow_pickle=True)
    keys = [k for k in data.files if k.startswith("dict_")]
    if methods:
        keys = [f"dict_{m}" for m in methods if f"dict_{m}" in data.files]
    method_dicts = {}
    for k in keys:
        d = data[k]
        # saved via np.savez(**results) where values are dicts -> 0-d object arrays
        if isinstance(d, np.ndarray) and d.dtype == object and d.shape == ():
            d = d.item()
        method = k[len("dict_"):]
        method_dicts[method] = d
    return method_dicts

def _approx_t_est(t, U):
    """
    Reconstruct times for estimator snapshots:
    updates occurred every est_freq steps; est_freq wasn't stored, so approximate.
    """
    if U <= 0 or len(t) <= 1:
        return np.array([])
    T = len(t)
    est_step = max(1, int(round((T - 1) / U)))
    idxs = np.arange(1, U + 1) * est_step
    idxs = np.clip(idxs, 0, T - 1)
    return t[idxs]

def viz_results(npz_path, methods=None):
    npz_path = Path(npz_path)
    outdir = npz_path.parent
    mdicts = _load_method_dicts(npz_path, methods=methods)
    if not mdicts:
        raise RuntimeError("No method dicts found in the npz (expected keys like 'dict_rk', 'dict_kf', ...).")

    # ---------- 3D trajectory: all methods + dotted reference ----------
    # take reference from the first method (they all used same seed/trajectory in your script)
    first_method = next(iter(mdicts))
    t_ref = mdicts[first_method]["t"]
    xr = mdicts[first_method]["x_ref_traj"]  # (T,13)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xr[:, 0], xr[:, 1], xr[:, 2], linestyle="--", linewidth=2.0, alpha=0.8, label="Reference")

    for m, d in mdicts.items():
        xm = d["x_meas_traj"]  # (T,13)
        ax.plot(xm[:, 0], xm[:, 1], xm[:, 2], label=m)

    ax.set_title("3D Trajectories — all methods")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal_3d(ax)
    plt.tight_layout()
    out_path = outdir / "traj3d_all_methods.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

    # ---------- Residual: ||A θ - b||^2 over (approx) time ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in mdicts.items():
        A_snap = d.get("A_snapshots", None)   # (U, m, n)
        b_snap = d.get("b_snapshots", None)   # (U, m)
        theta_est = d.get("theta_est_traj", None)  # (U, n)
        t = d["t"]
        if A_snap is None or b_snap is None or theta_est is None:
            continue
        U = A_snap.shape[0]
        t_est = _approx_t_est(t, U)
        r2 = []
        for i in range(U):
            Ai = A_snap[i]
            bi = b_snap[i]
            th = theta_est[i]
            if not np.all(np.isfinite(Ai)) or not np.all(np.isfinite(bi)) or not np.all(np.isfinite(th)):
                r2.append(np.nan); continue
            ri = Ai @ th - bi
            r2.append(float(ri @ ri))
        r2 = np.array(r2)
        ax.plot(t_est, r2, label=m)
    ax.set_title(r"Residual $\|A\theta - b\|^2$ over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|A\theta - b\|^2$")
    ax.grid(True, which="both")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    out_path = outdir / "residual_sq_all_methods.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

    # ---------- Mean position error over time ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in mdicts.items():
        t = d["t"]
        mpe = d["abs_mean_pos_err_t"]  # (T,)
        ax.plot(t, np.clip(mpe, 1e-12, None), label=m)  # safe for log if you want
    ax.set_title("Mean |position error| over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean |pos err| [m]")
    ax.grid(True, which="both")
    ax.legend()
    plt.tight_layout()
    out_path = outdir / "mean_pos_err_all_methods.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to payload_trials_parallel.npz")
    ap.add_argument("--methods", nargs="*", default=None, help="Subset of methods to plot (names without 'dict_')")
    args = ap.parse_args()
    viz_results(args.npz, methods=args.methods)

if __name__ == "__main__":
    main()
