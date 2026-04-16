"""
Simple A/B evaluator for baseline vs Fisher-adaptive generations.

Expected input CSV columns:
  mode, prompt_id, run_id, score_coherence, score_hallucination, score_structure, fisher_trace_std
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build A/B comparison table")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-csv", default="results/llm_entropy/fisher_runs/fisher_ab_comparison.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    needed = {
        "mode",
        "score_coherence",
        "score_hallucination",
        "score_structure",
        "fisher_trace_std",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    agg = (
        df.groupby("mode", as_index=False)
        .agg(
            n_runs=("mode", "count"),
            coherence_mean=("score_coherence", "mean"),
            hallucination_mean=("score_hallucination", "mean"),
            structure_mean=("score_structure", "mean"),
            fisher_trace_std_mean=("fisher_trace_std", "mean"),
        )
        .sort_values("mode")
    )
    agg["overall_quality_mean"] = (
        agg["coherence_mean"] + agg["hallucination_mean"] + agg["structure_mean"]
    ) / 3.0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(agg)} rows)", flush=True)


if __name__ == "__main__":
    main()
