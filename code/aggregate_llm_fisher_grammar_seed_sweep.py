"""
Median / IQR of N* (n_star_trace_max) across grammar-seed Fisher summary CSVs.

Sycamore-style: fixed LLM readout (entropy series), vary LSTM / grammar seed only.

Grammar-seed sweep (fixed LLM seed 42, 10k series), then aggregate:

  foreach ($gs in 0,1,2) {
    python code/run_llm_entropy_grammar_fisher.py --grammar-seed $gs \\
      --glob "results/llm_entropy/runs/*_p0_seed42_none.csv" \\
      --glob "results/llm_entropy/runs/*_p0_seed42_shuffled.csv" \\
      --summary-csv "llm_entropy_fisher_summary_gsweep_gram${gs}_llm42_none_shuf.csv"
  }
  python code/aggregate_llm_fisher_grammar_seed_sweep.py \\
    results/llm_entropy/fisher_runs/llm_entropy_fisher_summary_gsweep_gram0_llm42_none_shuf.csv \\
    results/llm_entropy/fisher_runs/llm_entropy_fisher_summary_gsweep_gram1_llm42_none_shuf.csv \\
    results/llm_entropy/fisher_runs/llm_entropy_fisher_summary_gsweep_gram2_llm42_none_shuf.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Median N* + IQR over grammar-seed Fisher summaries")
    ap.add_argument(
        "summaries",
        nargs="+",
        help="Two or more llm_entropy_fisher_summary_*.csv files (same LLM data, different --grammar-seed)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            REPO_ROOT, "results", "llm_entropy", "fisher_runs", "grammar_seed_sweep_nstar_median_iqr.csv"
        ),
        help="Output CSV path",
    )
    args = ap.parse_args()

    frames = []
    for i, path in enumerate(args.summaries):
        df = pd.read_csv(path)
        df["_grammar_sweep_idx"] = i
        df["_summary_path"] = os.path.basename(path)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    if "regime" not in all_df.columns or "n_star_trace_max" not in all_df.columns:
        sys.exit("Summaries must contain regime and n_star_trace_max columns")

    rows = []
    for (regime, control), g in all_df.groupby(["regime", "control"], sort=True):
        vals = g["n_star_trace_max"].values.astype(float)
        med = float(np.median(vals))
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        rows.append(
            {
                "regime": regime,
                "control": control,
                "n_star_median": med,
                "n_star_q1": q1,
                "n_star_q3": q3,
                "n_star_iqr": q3 - q1,
                "grammar_seeds_n": len(vals),
                "n_star_values": ";".join(str(int(v)) if v == int(v) else str(v) for v in sorted(vals)),
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
