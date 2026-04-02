"""
Publication-style figures for the Fisher threshold study (Sycamore 28 readouts, seeds 0–2).

Reads CSVs from results/ and writes PDF + PNG to results/fisher_figures/ (or --out-dir).

Figures:
  1) Median N* by topology (median per readout, grouped boxplot).
  2) Fisher trace vs data length N for three representative readouts (seed 0 metrics CSV).
  3) Median N* with IQR whiskers (all readouts, sorted).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")


def fig_median_by_topology(pub: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["1D_Snake", "2D_Block", "Bulk_Full"]
    data = [pub[pub["topology"] == t]["median_N_star"].values for t in order if (pub["topology"] == t).any()]
    labels = [t for t in order if (pub["topology"] == t).any()]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("#4C72B0")
        p.set_alpha(0.65)
    ax.set_ylabel("Median estimated N* (samples)")
    ax.set_xlabel("Sycamore layout family (topology label)")
    ax.set_title("Fisher-derived threshold N* by topology (28 readouts, median over seeds 0–2)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig01_median_Nstar_by_topology.{ext}"), dpi=200)
    plt.close(fig)


def fig_fisher_trace_examples(metrics_csv: str, out_dir: str) -> None:
    df = pd.read_csv(metrics_csv)
    picks = [
        ("12q_readout_raw_data.txt", "12q 2D_Block"),
        ("20q_readout_raw_data.txt", "20q 2D_Block"),
        ("53q_readout_raw_data.txt", "53q Bulk_Full"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
    for ax, (fn, title) in zip(axes, picks):
        g = df[df["file"] == fn].sort_values("n_points")
        if g.empty:
            ax.text(0.5, 0.5, "missing", ha="center")
            continue
        x = g["n_points"].values
        y = np.asarray(g["fisher_trace"].values, dtype=float)
        ypos = np.maximum(y, 1e-300)
        ax.plot(x, ypos, color="#C44E52", lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (data length)")
        ax.set_ylabel("Fisher trace")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fisher information trace vs data length (grammar transition matrix; seed 0)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig02_fisher_trace_vs_N_examples_seed0.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_median_iqr_bars(pub: pd.DataFrame, out_dir: str) -> None:
    pub = pub.copy()
    pub["q_int"] = pub["q_label"].str.replace("q", "", regex=False).astype(int)
    pub = pub.sort_values(["topology", "q_int"])
    n = len(pub)
    x = np.arange(n)
    med = pub["median_N_star"].values
    lo = pub["N_star_q25"].values
    hi = pub["N_star_q75"].values
    err_lo = med - lo
    err_hi = hi - med
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(x, med, color="#55A868", alpha=0.85, label="median N*")
    ax.errorbar(x, med, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pub["q_label"].values, rotation=90, fontsize=7)
    ax.set_ylabel("Estimated N* (median over 3 seeds)")
    ax.set_title("Per-readout Fisher threshold: median and IQR (seeds 0, 1, 2)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig03_median_Nstar_with_IQR_all_readouts.{ext}"), dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(RESULTS, "fisher_figures"))
    p.add_argument(
        "--publication-csv",
        default=os.path.join(RESULTS, "fisher_threshold_median_iqr_publication.csv"),
    )
    p.add_argument(
        "--metrics-seed0",
        default=os.path.join(RESULTS, "fisher_metric_vs_datalength_all_readouts_seed0.csv"),
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pub = pd.read_csv(args.publication_csv)
    # normalize column names (build_fisher_median_iqr_publication_table uses median_N_star etc.)
    if "median_N_star" not in pub.columns:
        raise SystemExit(f"Expected median_N_star in {args.publication_csv}")

    fig_median_by_topology(pub, args.out_dir)
    if os.path.isfile(args.metrics_seed0):
        fig_fisher_trace_examples(args.metrics_seed0, args.out_dir)
    else:
        print("Skip fig02: missing", args.metrics_seed0)
    fig_median_iqr_bars(pub, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
