"""
Build quantitative table + plot from transformers baseline/adaptive traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _token_repeat_metrics(text: str) -> tuple[float, int]:
    toks = re.findall(r"\S+", text)
    if len(toks) < 2:
        return 0.0, 1
    same_prev = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    frac = same_prev / (len(toks) - 1)
    max_run = 1
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return float(frac), int(max_run)


def _summarize_one(regime: str, mode: str, trace_csv: Path, text_path: Path) -> dict:
    df = pd.read_csv(trace_csv)
    trace = pd.to_numeric(df["fisher_trace"], errors="coerce")
    valid = trace.dropna().values
    action = df["action"].astype(str)
    txt = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    rep_frac, rep_run = _token_repeat_metrics(txt)
    return {
        "regime": regime,
        "mode": mode,
        "n_steps": int(len(df)),
        "n_fisher_points": int(np.isfinite(trace).sum()),
        "fisher_mean": float(np.mean(valid)) if len(valid) else np.nan,
        "fisher_std": float(np.std(valid)) if len(valid) else np.nan,
        "down_events": int((action == "down").sum()),
        "up_events": int((action == "up").sum()),
        "hold_events": int((action == "hold").sum()),
        "repeat_token_fraction": rep_frac,
        "max_repeat_run": rep_run,
        "trace_csv": str(trace_csv).replace("\\", "/"),
        "text_path": str(text_path).replace("\\", "/"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create transformers A/B summary table and figure")
    ap.add_argument("--factual-adaptive-trace", required=True)
    ap.add_argument("--factual-adaptive-text", required=True)
    ap.add_argument("--factual-baseline-trace", required=True)
    ap.add_argument("--factual-baseline-text", required=True)
    ap.add_argument("--math-adaptive-trace", required=True)
    ap.add_argument("--math-adaptive-text", required=True)
    ap.add_argument("--math-baseline-trace", required=True)
    ap.add_argument("--math-baseline-text", required=True)
    ap.add_argument("--out-dir", default="results/llm_entropy/fisher_runs/transformers_ab_report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _summarize_one("factual", "adaptive", Path(args.factual_adaptive_trace), Path(args.factual_adaptive_text)),
        _summarize_one("factual", "baseline", Path(args.factual_baseline_trace), Path(args.factual_baseline_text)),
        _summarize_one("mathematical", "adaptive", Path(args.math_adaptive_trace), Path(args.math_adaptive_text)),
        _summarize_one("mathematical", "baseline", Path(args.math_baseline_trace), Path(args.math_baseline_text)),
    ]
    sdf = pd.DataFrame(rows)
    summary_csv = out_dir / "transformers_ab_summary.csv"
    sdf.to_csv(summary_csv, index=False)

    # Human-readable markdown snapshot
    summary_md = out_dir / "transformers_ab_summary.md"
    cols = ["regime", "mode", "n_steps", "n_fisher_points", "fisher_mean", "fisher_std", "down_events", "up_events", "hold_events", "repeat_token_fraction", "max_repeat_run"]
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Transformers Fisher A/B Summary\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, r in sdf[cols].iterrows():
            vals = [str(r[c]) for c in cols]
            f.write("| " + " | ".join(vals) + " |\n")

    # Plot traces
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        data = {
            "factual": {
                "adaptive": pd.read_csv(args.factual_adaptive_trace),
                "baseline": pd.read_csv(args.factual_baseline_trace),
            },
            "mathematical": {
                "adaptive": pd.read_csv(args.math_adaptive_trace),
                "baseline": pd.read_csv(args.math_baseline_trace),
            },
        }
        colors = {"adaptive": "#4da3ff", "baseline": "#9aa0a6"}

        for ax, regime in zip(axes, ["factual", "mathematical"]):
            adf = data[regime]["adaptive"]
            bdf = data[regime]["baseline"]
            ax.plot(adf["step"], pd.to_numeric(adf["fisher_trace"], errors="coerce"), color=colors["adaptive"], lw=1.6, label="adaptive")
            ax.plot(bdf["step"], pd.to_numeric(bdf["fisher_trace"], errors="coerce"), color=colors["baseline"], lw=1.1, alpha=0.8, label="baseline")

            d = adf[adf["action"] == "down"]
            u = adf[adf["action"] == "up"]
            ax.scatter(d["step"], pd.to_numeric(d["fisher_trace"], errors="coerce"), s=24, marker="v", label="down")
            ax.scatter(u["step"], pd.to_numeric(u["fisher_trace"], errors="coerce"), s=24, marker="^", label="up")
            ax.set_title(regime.capitalize())
            ax.set_ylabel("Fisher trace")
            ax.legend(fontsize=8)

        axes[-1].set_xlabel("Token index")
        fig.tight_layout()
        plot_png = out_dir / "transformers_ab_trace_plot.png"
        fig.savefig(plot_png, dpi=220)
        print(f"Wrote {plot_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
