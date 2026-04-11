"""
Plot fisher_speed_dn(N) and log_det_fisher(N) for every readout curve.

Expects ``results/fisher_ricci_and_embedding_vs_n.csv`` from
``compute_fisher_ricci_curve.py`` (``data_source=measured``).

Optional vertical line at estimated N* from ``fisher_estimated_thresholds_per_readout_all_readouts.csv``.

Usage:
  python code/plot_fisher_embedding_curves.py
  python code/plot_fisher_embedding_curves.py --embedding-csv results/fisher_ricci_and_embedding_vs_n.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-csv",
        default=os.path.join(RESULTS_DIR, "fisher_ricci_and_embedding_vs_n.csv"),
    )
    ap.add_argument(
        "--thresholds-csv",
        default=os.path.join(RESULTS_DIR, "fisher_estimated_thresholds_per_readout_all_readouts.csv"),
        help="Optional; omit file to skip N* lines",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(RESULTS_DIR, "fisher_embedding_speed_logdet_28readouts.png"),
    )
    ap.add_argument(
        "--include-demo",
        action="store_true",
        help="Include synthetic_demo rows (default: only data_source==measured)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.embedding_csv)
    if not args.include_demo and "data_source" in df.columns:
        df = df[df["data_source"].astype(str) == "measured"]
    if df.empty:
        raise SystemExit(f"No rows to plot in {args.embedding_csv}")

    thr = None
    if args.thresholds_csv and os.path.isfile(args.thresholds_csv):
        thr = pd.read_csv(args.thresholds_csv)

    files = sorted(df["file"].unique())
    n_files = len(files)
    ncols = 4
    nrows = int(np.ceil(n_files / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, fname in zip(axes_flat, files):
        g = df[df["file"] == fname].sort_values("n_points")
        n = g["n_points"].values.astype(float)
        ax2 = ax.twinx()
        (l1,) = ax.plot(n, g["fisher_speed_dn"].values, "b.-", markersize=3, label="fisher_speed_dn")
        (l2,) = ax2.plot(n, g["log_det_fisher"].values, "r.-", markersize=3, label="log_det_fisher")
        ql = g["q_label"].iloc[0]
        ax.set_title(f"{ql}", fontsize=9)
        ax.set_ylabel("speed", color="b", fontsize=8)
        ax2.set_ylabel("log det G", color="r", fontsize=8)
        ax.tick_params(axis="y", labelcolor="b", labelsize=7)
        ax2.tick_params(axis="y", labelcolor="r", labelsize=7)
        if thr is not None and "file" in thr.columns and "estimated_N_threshold" in thr.columns:
            sub = thr[thr["file"] == fname]
            if len(sub) == 1:
                nx = float(sub["estimated_N_threshold"].iloc[0])
                if np.isfinite(nx):
                    ax.axvline(nx, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for j in range(len(files), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Fisher path speed ‖dθ/dN‖_G and log det G(N)  (blue / red)",
        fontsize=12,
    )
    fig.supxlabel("N (data length)", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
