#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def summarize_err(err_2d, t, stat="mean", tmin=None, tmax=None):
    """
    Summarize an error-timeseries array across time & trials.

    err_2d: (N_trials, T) for a single label & metric (e.g., pos)
    t:      (T,) time vector from the NPZ
    stat:   "mean" | "median" | "p95" | "max"
    tmin,tmax: time window in seconds (None = use full)
    """
    if err_2d.ndim != 2:
        raise ValueError(f"expected (N,T), got {err_2d.shape}")
    T = err_2d.shape[1]
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.01

    k0 = 0 if tmin is None else max(0, int(np.floor(tmin / dt)))
    k1 = T if tmax is None else min(T, int(np.ceil(tmax / dt)))
    if k1 <= k0:  # degenerate, fall back to full
        k0, k1 = 0, T

    windowed = err_2d[:, k0:k1]
    if stat == "mean":
        return float(np.nanmean(windowed))
    if stat == "median":
        return float(np.nanmedian(windowed))
    if stat == "p95":
        return float(np.nanpercentile(windowed, 95))
    if stat == "max":
        return float(np.nanmax(windowed))
    raise ValueError(f"Unknown stat={stat}")

def build_summary_frame(npz, metrics=("pos","vel","ori"), stat="mean", tmin=None, tmax=None):
    """
    Returns a DataFrame with one row per label and columns:
    [label, gamma, alpha, tol, pos_<stat>, vel_<stat>, ori_<stat>, est_mean]
    """
    labels = [str(x) for x in npz["labels"]]
    gammas = np.asarray(npz["deka_gamma"], dtype=float)
    alphas = np.asarray(npz["deka_alpha"], dtype=float)
    tols   = np.asarray(npz["deka_tol"],   dtype=float)
    t      = np.asarray(npz["t"],          dtype=float)

    rows = []
    for lab, g, a, tol in zip(labels, gammas, alphas, tols):
        row = {"label": lab, "gamma": g, "alpha": a, "tol": tol}
        for m in metrics:
            arr = np.asarray(npz[f"{lab}__{m}"])   # shape: (N_trials, T)
            row[f"{m}_{stat}"] = summarize_err(arr, t, stat=stat, tmin=tmin, tmax=tmax)
        # also summarize estimator residuals (per-update series): mean of last half
        if f"{lab}__est" in npz:
            est = np.asarray(npz[f"{lab}__est"])   # (N_trials, U)
            U = est.shape[1]
            est_tail = est[:, U//2:] if U >= 4 else est
            row["est_mean"] = float(np.nanmean(est_tail))
        rows.append(row)

    df = pd.DataFrame(rows)
    # add a simple composite score (lower is better); tweak weights if you like
    df["score"] = df[[f"{m}_{stat}" for m in metrics]].mean(axis=1)
    return df

def heatmap_grid(df, outdir, metrics=("pos","vel","ori"), stat="mean",
                 annotate=True, cmap="viridis_r", fmt=".3g", title_suffix="",
                 x_key="gamma", y_key="alpha", panel_key="tol"):
    """
    For each unique `panel_key` (tol), plot heatmaps of metric_{stat} with
    x = gamma, y = alpha. Saves one PNG per panel.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    uniq_panels = np.sort(df[panel_key].unique())
    for p in uniq_panels:
        sub = df[df[panel_key] == p].copy()

        xs = np.sort(sub[x_key].unique())
        ys = np.sort(sub[y_key].unique())

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for j, m in enumerate(metrics):
            Z = np.full((len(ys), len(xs)), np.nan)
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    match = sub[(sub[x_key] == x) & (sub[y_key] == y)]
                    if not match.empty:
                        Z[iy, ix] = float(match[f"{m}_{stat}"].iloc[0])

            ax = axes[j]
            im = ax.imshow(Z, origin="lower", cmap=cmap,
                           extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                           aspect="auto")
            ax.set_xlabel(x_key)
            if j == 0:
                ax.set_ylabel(y_key)
            ax.set_title(f"{m} ({stat})")

            if annotate:
                # annotate on an evenly-spaced grid of indices, not data coords
                for iy in range(len(ys)):
                    for ix in range(len(xs)):
                        val = Z[iy, ix]
                        if np.isfinite(val):
                            ax.text(xs[ix], ys[iy], format(val, fmt),
                                    ha="center", va="center", fontsize=9,
                                    color="black",
                                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(f"{m} {stat}", rotation=90, va="center")

        fig.suptitle(f"DEKA grid — {panel_key}={p:g}  {title_suffix}")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = Path(outdir) / f"deka_grid_heatmap_tol{p:g}_{stat}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

def _center_band(E, stat="mean", band="iqr"):
    """
    E: (N_trials, T)
    Returns (center, lo, hi) where lo/hi define the band.
    band in {"none","iqr","p90","std"}
    """
    import numpy as np
    if stat == "mean":
        center = np.nanmean(E, axis=0)
    elif stat == "median":
        center = np.nanmedian(E, axis=0)
    else:
        raise ValueError("stat must be 'mean' or 'median'")

    if band == "none":
        return center, None, None
    if band == "iqr":
        lo = np.nanpercentile(E, 25, axis=0)
        hi = np.nanpercentile(E, 75, axis=0)
        return center, lo, hi
    if band == "p90":
        lo = np.nanpercentile(E, 5, axis=0)
        hi = np.nanpercentile(E, 95, axis=0)
        return center, lo, hi
    if band == "std":
        std = np.nanstd(E, axis=0)
        return center, center - std, center + std
    raise ValueError("band must be one of {'none','iqr','p90','std'}")


def _moving_average(y, nwin):
    if nwin is None or nwin <= 1:
        return y
    k = int(nwin)
    k = max(1, k)
    # causal-ish symmetric smoothing
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(y, kernel, mode="same")


def plot_timeseries_grid(npz, df, outdir,
                         metric="pos",         # "pos" | "vel" | "ori"
                         stat="mean",          # "mean" | "median"
                         band="iqr",           # "none" | "iqr" | "p90" | "std"
                         topk=5,
                         rank_by="score",      # "score" or e.g. "pos_mean"
                         tmin=None, tmax=None,
                         smooth_sec=0.0,
                         ymax=None):
    """
    For each tol, overlay the top-K configs' error timeseries for a chosen metric.
    Saves: deka_grid_timeseries_<metric>_tol<tol>.png
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    Path(outdir).mkdir(parents=True, exist_ok=True)
    t = np.asarray(npz["t"], dtype=float)
    titles = {"pos":"Position error [m]","vel":"Velocity error [m/s]","ori":"Orientation error [deg]"}
    uniq_tol = np.sort(df["tol"].unique())

    # time window indices
    T = len(t)
    dt = float(t[1]-t[0]) if T > 1 else 0.01
    k0 = 0 if tmin is None else max(0, int(np.floor(tmin / dt)))
    k1 = T if tmax is None else min(T, int(np.ceil(tmax / dt)))
    if k1 <= k0:
        k0, k1 = 0, T
    t_win = t[k0:k1]
    nwin = int(round(float(smooth_sec) / dt)) if smooth_sec and smooth_sec > 0 else None

    for tol in uniq_tol:
        sub = df[df["tol"] == tol].copy()

        # ranking
        if rank_by not in sub.columns:
            if rank_by != "score":
                raise ValueError(f"rank_by='{rank_by}' not found in df; available: {list(sub.columns)}")
            sub = sub.sort_values("score")
        else:
            sub = sub.sort_values(rank_by)

        pick = sub.head(int(topk))

        plt.figure(figsize=(10, 6))
        for _, row in pick.iterrows():
            lab = str(row["label"])
            g, a = row["gamma"], row["alpha"]

            E = np.asarray(npz[f"{lab}__{metric}"])   # (N_trials, T)
            E = E[:, k0:k1]                           # time window

            # center & band across trials
            center, lo, hi = _center_band(E, stat=stat, band=band)

            # optional smoothing
            c_s = _moving_average(center, nwin)
            if lo is not None and hi is not None:
                lo_s = _moving_average(lo, nwin)
                hi_s = _moving_average(hi, nwin)
                plt.fill_between(t_win, lo_s, hi_s, alpha=0.20, linewidth=0)
            plt.plot(t_win, c_s, label=f"γ={g:g}, α={a:g}")

        plt.title(f"{metric.upper()} timeseries — tol={tol:g}   (center: {stat}, band: {band}"
                  + (f", smooth={smooth_sec:.2f}s" if smooth_sec and smooth_sec > 0 else "") + ")")
        plt.xlabel("Time [s]")
        plt.ylabel(titles[metric])
        if ymax is not None:
            plt.ylim(top=float(ymax))
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        out = Path(outdir) / f"deka_grid_timeseries_{metric}_tol{tol:g}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved {out}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="results_deka_grid.npz file")
    ap.add_argument("--outdir", default=None, help="where to save figures (default: alongside NPZ)")
    ap.add_argument("--stat", default="mean", choices=["mean","median","p95","max"],
                    help="summary statistic for errors")
    ap.add_argument("--tmin", type=float, default=None, help="start time (s) for summary window")
    ap.add_argument("--tmax", type=float, default=None, help="end time (s) for summary window")
    ap.add_argument("--no_annot", action="store_true", help="disable numeric annotations in cells")
    ap.add_argument("--csv", action="store_true", help="also write a CSV with the summary by config")

    ap.add_argument("--plot_timeseries", action="store_true",
                    help="Plot timeseries of error per tol for the top-K configs.")
    ap.add_argument("--ts_metric", type=str, default="pos", choices=["pos","vel","ori"])
    ap.add_argument("--ts_stat", type=str, default="mean", choices=["mean","median"])
    ap.add_argument("--ts_band", type=str, default="iqr", choices=["none","iqr","p90","std"])
    ap.add_argument("--ts_topk", type=int, default=5)
    ap.add_argument("--ts_rank_by", type=str, default="score",
                    help="Column to rank by, e.g., 'score' or 'pos_mean/vel_mean/ori_mean'.")
    ap.add_argument("--ts_tmin", type=float, default=None)
    ap.add_argument("--ts_tmax", type=float, default=None)
    ap.add_argument("--ts_smooth_sec", type=float, default=0.0,
                    help="Moving-average smoothing window (seconds). 0 = off.")
    ap.add_argument("--ts_ymax", type=float, default=None,
                    help="Clip y-axis top to this value.")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    outdir = Path(args.outdir) if args.outdir else npz_path.parent

    data = np.load(npz_path, allow_pickle=False)
    # Sanity-check that this is a DEKA grid NPZ (has the keys your runner writes)
    for key in ("labels", "deka_gamma", "deka_alpha", "deka_tol"):
        if key not in data.files:
            raise RuntimeError(f"NPZ missing '{key}'. Are you loading the grid NPZ?")

    df = build_summary_frame(
        data, metrics=("pos","vel","ori"),
        stat=args.stat, tmin=args.tmin, tmax=args.tmax
    )

    # Show the grid that was run (unique values per hyper-param)
    print("\nGrid values discovered:")
    print(" gamma:", sorted(df["gamma"].unique()))
    print(" alpha:", sorted(df["alpha"].unique()))
    print(" tol:  ", sorted(df["tol"].unique()))

    # Top 10 configs by the composite score (lower=better)
    top = df.sort_values("score").head(10)
    print("\nTop 10 configs by composite score (mean of pos/vel/ori stats):")
    print(top[["label","gamma","alpha","tol","pos_"+args.stat,"vel_"+args.stat,"ori_"+args.stat,"score"]])

    if args.csv:
        csv_path = outdir / f"deka_grid_summary_{args.stat}.csv"
        top.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    # Heatmaps: one panel per tol, gamma on x, alpha on y
    title_suffix = f"stat={args.stat}, window=[{args.tmin if args.tmin is not None else 0}s, {args.tmax if args.tmax is not None else 'end'}s]"
    heatmap_grid(
        df, outdir,
        metrics=("pos","vel","ori"),
        stat=args.stat,
        annotate=not args.no_annot,
        title_suffix=title_suffix,
    )

    if args.plot_timeseries:
        plot_timeseries_grid(
            data, df, outdir,
            metric=args.ts_metric,
            stat=args.ts_stat,
            band=args.ts_band,
            topk=int(args.ts_topk),
            rank_by=args.ts_rank_by,
            tmin=args.ts_tmin, tmax=args.ts_tmax,
            smooth_sec=args.ts_smooth_sec,
            ymax=args.ts_ymax,
        )

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
