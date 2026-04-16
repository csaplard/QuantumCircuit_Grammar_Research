"""
Posthoc layer-separation analysis for the internal Fisher PoC.

For each monitored layer (early/mid/late) and each regime (factual / math / creative),
compute the per-step Fisher trace summary and the between-regime separation strength.

We use only the *baseline* runs (no temperature feedback) so that the layer
comparison is not contaminated by the adaptive controller.

Outputs (under results/internal_fisher_poc/):
  - layer_separation.csv             : per (layer, regime)  mean / std / n
  - layer_separation_pairs.csv       : per (layer, regime_pair)  Cohen's d
  - layer_separation_plot.png        : 1x3 grid, F(t) overlay per layer for the 3 regimes
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ["factual", "mathematical", "creative"]
LAYERS = ["early", "mid", "late"]
WARMUP = 32  # skip the sliding-window warm-up region


def load_baseline_traces(results_dir: Path) -> dict[str, list[pd.DataFrame]]:
    """Load all baseline traces for each regime.

    Supports both the old single-prompt naming (baseline_{regime}_fisher_trace.csv)
    and the new multi-prompt naming (baseline_{regime}_p{idx}_fisher_trace.csv).
    Returns a list of DataFrames per regime so per-prompt traces stay separable.
    """
    out: dict[str, list[pd.DataFrame]] = {}
    for regime in REGIMES:
        # Multi-prompt files first
        multi = sorted(results_dir.glob(f"baseline_{regime}_p*_fisher_trace.csv"))
        if multi:
            out[regime] = [pd.read_csv(p) for p in multi]
        else:
            single = results_dir / f"baseline_{regime}_fisher_trace.csv"
            if not single.exists():
                raise FileNotFoundError(
                    f"no baseline traces found for regime={regime} in {results_dir}"
                )
            out[regime] = [pd.read_csv(single)]
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / max(1, na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/internal_fisher_poc",
                    help="results directory containing baseline_*_fisher_trace.csv files")
    args = ap.parse_args()
    results_dir = Path(args.dir)
    traces = load_baseline_traces(results_dir)

    # For each (layer, regime), pool *all* prompts: concatenate post-warmup
    # F-values across prompts. This mixes within-prompt step variability with
    # between-prompt variability, which is the right denominator for Cohen's d
    # if we want "regime separation in the population of likely prompts".
    rows = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    per_prompt_arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for layer in LAYERS:
        col = f"fisher_{layer}"
        for regime in REGIMES:
            dfs = traces[regime]
            chunks: list[np.ndarray] = []
            for df in dfs:
                if col not in df.columns:
                    raise RuntimeError(
                        f"missing column {col} in baseline_{regime} csv"
                    )
                chunks.append(df[col].to_numpy(dtype=float)[WARMUP:])
            per_prompt_arrays[(layer, regime)] = chunks
            x = np.concatenate(chunks) if chunks else np.array([])
            arrays[(layer, regime)] = x
            rows.append({
                "layer": layer,
                "regime": regime,
                "n_prompts": len(chunks),
                "n_steps_total": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "layer_separation.csv", index=False)

    # Pairwise Cohen's d per layer (pooled across prompts)
    pair_rows = []
    for layer in LAYERS:
        for r1, r2 in combinations(REGIMES, 2):
            d = cohens_d(arrays[(layer, r1)], arrays[(layer, r2)])
            pair_rows.append({
                "layer": layer,
                "regime_a": r1,
                "regime_b": r2,
                "cohens_d": d,
                "abs_d": abs(d),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(results_dir / "layer_separation_pairs.csv", index=False)

    # Aggregate per-layer separation = mean |d| across the 3 regime pairs
    agg = pairs.groupby("layer", as_index=False)["abs_d"].mean().rename(
        columns={"abs_d": "mean_abs_d"}
    )

    # Per-prompt mean F (regime × prompt × layer): used for a between-prompt
    # robustness check. If means are tightly clustered within a regime and
    # well-separated between regimes, the signal is robust.
    perp_rows = []
    for layer in LAYERS:
        for regime in REGIMES:
            for i, x in enumerate(per_prompt_arrays[(layer, regime)]):
                perp_rows.append({
                    "layer": layer,
                    "regime": regime,
                    "prompt_idx": i,
                    "mean_F": float(np.mean(x)),
                    "std_F": float(np.std(x, ddof=1)),
                })
    pd.DataFrame(perp_rows).to_csv(
        results_dir / "layer_separation_per_prompt.csv", index=False
    )

    # Plot: overlay every prompt's trace, lightly, plus the regime-pooled
    # mean-as-a-line for visual clarity.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    colors = {"factual": "#1f77b4", "mathematical": "#d62728", "creative": "#2ca02c"}
    for ax, layer in zip(axes, LAYERS):
        for regime in REGIMES:
            for x in per_prompt_arrays[(layer, regime)]:
                ax.plot(x, lw=0.6, color=colors[regime], alpha=0.45)
            # Plot a label-only marker so legend has clean entries
            ax.plot([], [], color=colors[regime], lw=2, label=regime)
        d_mean = float(agg.loc[agg["layer"] == layer, "mean_abs_d"].iloc[0])
        ax.set_title(f"{layer}  (mean |d| = {d_mean:.2f})")
        ax.set_xlabel("step (post-warmup)")
        ax.set_ylabel("F(t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Internal Fisher trace by layer — baseline runs, all prompts overlaid")
    fig.tight_layout()
    fig.savefig(results_dir / "layer_separation_plot.png", dpi=150)
    plt.close(fig)

    print("=== per (layer, regime) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== pairwise Cohen's d ===")
    print(pairs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== mean |d| per layer ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[write] {results_dir / 'layer_separation.csv'}")
    print(f"[write] {results_dir / 'layer_separation_pairs.csv'}")
    print(f"[write] {results_dir / 'layer_separation_per_prompt.csv'}")
    print(f"[write] {results_dir / 'layer_separation_plot.png'}")


if __name__ == "__main__":
    main()
