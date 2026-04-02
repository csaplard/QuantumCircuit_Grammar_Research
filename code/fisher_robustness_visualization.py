"""
Post-process Fisher sweep CSVs: topology boxplots + robust log-Fisher curves.

Reads:
  results/fisher_estimated_thresholds_per_readout_all_readouts.csv
  results/fisher_metric_vs_datalength_all_readouts.csv

With --quick, uses the non-_all_readouts filenames from a 3-file sweep.

Outputs:
  results/fisher_threshold_topology_boxplot.png
  results/fisher_trace_robust_curves.png
  results/fisher_robustness_summary.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

TOPO_ORDER = ["1D_Snake", "2D_Block", "Bulk_Full"]
TOPO_COLORS = {"1D_Snake": "#e74c3c", "2D_Block": "#3498db", "Bulk_Full": "#27ae60"}


def _default_paths(quick: bool) -> tuple[str, str]:
    suf = "" if quick else "_all_readouts"
    thresh = os.path.join(RESULTS_DIR, f"fisher_estimated_thresholds_per_readout{suf}.csv")
    long_csv = os.path.join(RESULTS_DIR, f"fisher_metric_vs_datalength{suf}.csv")
    return thresh, long_csv


def _q25(s: pd.Series) -> float:
    return float(s.quantile(0.25))


def _q75(s: pd.Series) -> float:
    return float(s.quantile(0.75))


def plot_threshold_boxplot(thresh_df: pd.DataFrame, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    series_list = []
    labels = []
    positions = []
    topos_used: list[str] = []
    pos = 1
    for topo in TOPO_ORDER:
        vals = thresh_df.loc[thresh_df["topology"] == topo, "estimated_N_threshold"].dropna().values
        if len(vals) == 0:
            continue
        series_list.append(vals)
        topos_used.append(topo)
        labels.append(topo.replace("_", "\n"))
        positions.append(pos)
        pos += 1

    bp = ax.boxplot(series_list, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
    for patch, topo in zip(bp["boxes"], topos_used):
        patch.set_facecolor(TOPO_COLORS[topo])
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for i, topo in enumerate(topos_used):
        vals = thresh_df.loc[thresh_df["topology"] == topo, "estimated_N_threshold"].dropna().values
        x = positions[i] + 0.08 * (rng.random(len(vals)) - 0.5)
        ax.scatter(x, vals, color="0.2", s=22, alpha=0.75, zorder=3)

    ax.axhline(8000, color="gray", linestyle="--", linewidth=1.2, label="N ≈ 8000 (grammar ref.)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated N (max |d(Fisher trace)/dN|)")
    ax.set_title("Per-readout Fisher threshold by topology")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def add_log_and_smooth(long_df: pd.DataFrame, roll: int) -> pd.DataFrame:
    df = long_df.copy()
    df["log10_fisher_trace"] = np.log10(np.maximum(df["fisher_trace"].astype(float), 1e-300))
    parts = []
    for _, g in df.groupby("file", sort=False):
        g = g.sort_values("n_points")
        sm = g["log10_fisher_trace"].rolling(window=roll, center=True, min_periods=1).mean()
        parts.append(g.assign(log10_fisher_trace_smooth=sm))
    return pd.concat(parts, ignore_index=True)


def plot_robust_curves(df: pd.DataFrame, out_path: str, roll: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    def draw_panel(ax, col: str, title: str) -> None:
        for topo in TOPO_ORDER:
            sub = df[df["topology"] == topo]
            if sub.empty:
                continue
            agg = (
                sub.groupby("n_points", sort=True)[col]
                .agg(median="median", q1=_q25, q3=_q75)
                .reset_index()
            )
            n = agg["n_points"].values
            ax.fill_between(n, agg["q1"], agg["q3"], color=TOPO_COLORS[topo], alpha=0.22)
            ax.plot(n, agg["median"], "o-", color=TOPO_COLORS[topo], linewidth=2, label=topo, markersize=4)
        ax.axvline(8000, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Data length N")
        ax.set_ylabel("log10(Fisher trace)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

    draw_panel(axes[0], "log10_fisher_trace", "Across readouts: median ± IQR (raw log10 trace)")
    draw_panel(
        axes[1],
        "log10_fisher_trace_smooth",
        f"Same after per-file rolling mean (window={roll} N-steps)",
    )
    plt.suptitle("Robust aggregate Fisher trace vs N (28 Sycamore readouts)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_summary(thresh_df: pd.DataFrame, long_df: pd.DataFrame, roll: int, out_path: str) -> None:
    lines = [
        "=== Fisher robustness summary ===",
        "",
        "Estimated N threshold (from discrete |d(trace)/dN| max), by topology:",
        "",
    ]
    for topo in TOPO_ORDER:
        v = thresh_df.loc[thresh_df["topology"] == topo, "estimated_N_threshold"].dropna().astype(float)
        if v.empty:
            continue
        lines.append(
            f"{topo}: n={len(v)}, min={v.min():.0f}, Q1={v.quantile(0.25):.0f}, "
            f"median={v.median():.0f}, Q3={v.quantile(0.75):.0f}, max={v.max():.0f}, "
            f"mean={v.mean():.1f}, std={v.std():.1f}"
        )
    lines.extend(
        [
            "",
            f"log10(Fisher trace): per-file rolling mean window = {roll} (by sorted N grid).",
            "Aggregate plots: median and IQR across readouts at each N.",
            "",
        ]
    )
    n_grid = [int(x) for x in sorted(long_df["n_points"].unique())]
    lines.append(f"N grid: {n_grid}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Fisher robustness plots from sweep CSVs")
    p.add_argument("--quick", action="store_true", help="Use fisher_* without _all_readouts suffix.")
    p.add_argument("--rolling", type=int, default=3, help="Rolling window size on log10 trace (per file).")
    p.add_argument("--threshold-csv", default=None)
    p.add_argument("--metric-csv", default=None)
    args = p.parse_args()

    t_path, m_path = _default_paths(args.quick)
    if args.threshold_csv:
        t_path = args.threshold_csv
    if args.metric_csv:
        m_path = args.metric_csv

    for path, name in ((t_path, "threshold"), (m_path, "metric")):
        if not os.path.isfile(path):
            print(f"Missing {name} CSV: {path}", file=sys.stderr)
            sys.exit(1)

    thresh_df = pd.read_csv(t_path)
    long_df = pd.read_csv(m_path)

    df_smooth = add_log_and_smooth(long_df, max(1, int(args.rolling)))

    box_out = os.path.join(RESULTS_DIR, "fisher_threshold_topology_boxplot.png")
    curves_out = os.path.join(RESULTS_DIR, "fisher_trace_robust_curves.png")
    summary_out = os.path.join(RESULTS_DIR, "fisher_robustness_summary.txt")

    plot_threshold_boxplot(thresh_df, box_out)
    plot_robust_curves(df_smooth, curves_out, max(1, int(args.rolling)))
    write_summary(thresh_df, long_df, max(1, int(args.rolling)), summary_out)

    print(f"Saved: {box_out}")
    print(f"Saved: {curves_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
