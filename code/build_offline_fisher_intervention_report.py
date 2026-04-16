"""
Build offline Fisher intervention plots/reports for factual+mathematical 512-token runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sliding_fisher import rolling_fisher_trace_series


def _analyze_one(
    *,
    regime: str,
    input_csv: Path,
    window: int,
    epsilon: float,
    check_every: int,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(input_csv)
    entropy = df["entropy"].astype(float).values
    trace = rolling_fisher_trace_series(
        entropy_series=entropy,
        window_size=window,
        alphabet_size=7,
        sax_behavior="quantile",
        fisher_epsilon=epsilon,
    )
    mean = float(np.nanmean(trace))
    std = float(np.nanstd(trace))
    low, high = mean - std, mean + std

    steps = np.arange(len(trace))
    check_idx = np.arange(check_every - 1, len(trace), check_every)
    check_trace = trace[check_idx]
    decisions = np.where(
        check_trace < low,
        "down",
        np.where(check_trace > high, "up", "hold"),
    )

    out = pd.DataFrame(
        {
            "regime": regime,
            "step": steps,
            "fisher_trace": trace,
            "is_check_step": np.isin(steps, check_idx),
        }
    )

    for i, s in enumerate(check_idx):
        out.loc[out["step"] == s, "decision"] = decisions[i]
    out["decision"] = out["decision"].fillna("")
    out["low_threshold"] = low
    out["high_threshold"] = high

    summary = {
        "regime": regime,
        "n_tokens": int(len(entropy)),
        "n_valid_trace": int(np.isfinite(trace).sum()),
        "window_size": int(window),
        "fisher_epsilon": float(epsilon),
        "check_every": int(check_every),
        "mean_trace": mean,
        "std_trace": std,
        "low_threshold": low,
        "high_threshold": high,
        "n_checkpoints": int(len(check_idx)),
        "n_interventions": int((decisions != "hold").sum()),
        "n_down": int((decisions == "down").sum()),
        "n_up": int((decisions == "up").sum()),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline Fisher intervention plots for factual+mathematical")
    ap.add_argument("--factual-csv", default="results/llm_entropy/runs/factual_p0_seed42_none.csv")
    ap.add_argument("--mathematical-csv", default="results/llm_entropy/runs/mathematical_p0_seed42_none.csv")
    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--fisher-epsilon", type=float, default=0.01)
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--out-dir", default="results/llm_entropy/fisher_runs/offline_intervention_report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factual_df, factual_summary = _analyze_one(
        regime="factual",
        input_csv=Path(args.factual_csv),
        window=args.window_size,
        epsilon=args.fisher_epsilon,
        check_every=args.check_every,
    )
    math_df, math_summary = _analyze_one(
        regime="mathematical",
        input_csv=Path(args.mathematical_csv),
        window=args.window_size,
        epsilon=args.fisher_epsilon,
        check_every=args.check_every,
    )

    detail_csv = out_dir / "offline_fisher_intervention_detail.csv"
    summary_csv = out_dir / "offline_fisher_intervention_summary.csv"
    pd.concat([factual_df, math_df], ignore_index=True).to_csv(detail_csv, index=False)
    pd.DataFrame([factual_summary, math_summary]).to_csv(summary_csv, index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for ax, regime_df, summary in (
            (axes[0], factual_df, factual_summary),
            (axes[1], math_df, math_summary),
        ):
            reg = summary["regime"]
            ax.plot(regime_df["step"], regime_df["fisher_trace"], linewidth=1.3, label=f"{reg} trace")
            ax.axhline(summary["low_threshold"], linestyle="--", linewidth=1, label="mean-σ")
            ax.axhline(summary["high_threshold"], linestyle="--", linewidth=1, label="mean+σ")

            c = regime_df[regime_df["is_check_step"] == True].copy()  # noqa: E712
            downs = c[c["decision"] == "down"]
            ups = c[c["decision"] == "up"]
            if not downs.empty:
                ax.scatter(downs["step"], downs["fisher_trace"], marker="v", s=40, label="down")
            if not ups.empty:
                ax.scatter(ups["step"], ups["fisher_trace"], marker="^", s=40, label="up")

            ax.set_title(
                f"{reg}: interventions={summary['n_interventions']} "
                f"(down={summary['n_down']}, up={summary['n_up']})"
            )
            ax.set_ylabel("Fisher trace")
            ax.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("Token index")
        fig.tight_layout()
        plot_path = out_dir / "offline_fisher_intervention_plot.png"
        fig.savefig(plot_path, dpi=220)
        print(f"Wrote {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
