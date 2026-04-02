"""
Combine several Fisher per-readout threshold CSVs (different RNG seeds) into one
CSV with median (and optional IQR) estimated N* per readout file / q_label.

Usage (after running fisher_information_analysis.py with --tag-with-seed and different --seed):
  python code/fisher_information_analysis.py --tag-with-seed --seed 0
  python code/fisher_information_analysis.py --tag-with-seed --seed 1
  python code/aggregate_fisher_threshold_seeds.py \\
    results/fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv \\
    results/fisher_estimated_thresholds_per_readout_all_readouts_seed1.csv \\
    --out results/fisher_estimated_thresholds_median_seeds.csv

Then:
  python code/fit_threshold_transfer_model.py --sycamore-csv results/fisher_estimated_thresholds_median_seeds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Median Fisher N* across seed CSVs")
    p.add_argument("inputs", nargs="+", type=Path, help="CSV paths with columns file,q_label,...,estimated_N_threshold")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/fisher_estimated_thresholds_median_seeds.csv"),
    )
    p.add_argument("--value-col", default="estimated_N_threshold")
    args = p.parse_args()

    frames = []
    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"Missing: {path}")
        frames.append(pd.read_csv(path))

    key_cols = ["file", "q_label"]
    for df in frames:
        for c in key_cols + [args.value_col]:
            if c not in df.columns:
                raise SystemExit(f"Column {c!r} missing in {df}")

    cols0 = key_cols + [args.value_col]
    if "topology" in frames[0].columns:
        cols0.append("topology")
    merged = frames[0][cols0].rename(columns={args.value_col: "v0"})
    for i, df in enumerate(frames[1:], start=1):
        sub = df[key_cols + [args.value_col]].rename(columns={args.value_col: f"v{i}"})
        merged = merged.merge(sub, on=key_cols, how="outer")

    vcols = [c for c in merged.columns if c.startswith("v")]
    mat = merged[vcols].to_numpy(dtype=np.float64)
    merged["estimated_N_threshold"] = np.nanmedian(mat, axis=1)
    merged["estimated_N_threshold_q25"] = np.nanpercentile(mat, 25, axis=1)
    merged["estimated_N_threshold_q75"] = np.nanpercentile(mat, 75, axis=1)

    out_cols = [
        "file",
        "q_label",
        "topology",
        "estimated_N_threshold",
        "estimated_N_threshold_q25",
        "estimated_N_threshold_q75",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(merged)} rows, {len(frames)} inputs)")


if __name__ == "__main__":
    main()
