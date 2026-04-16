"""
Calibrate Fisher baseline threshold profiles from non-adaptive runs.

Input: CSV logs with at least columns [regime, fisher_trace]
Output: JSON profile with regime-wise quantile thresholds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fisher_baseline_profiles.json from trace CSVs")
    ap.add_argument(
        "--glob",
        action="append",
        default=["results/llm_entropy/fisher_runs/*baseline*.csv"],
        help="Repeatable glob for baseline trace CSVs",
    )
    ap.add_argument(
        "--out-json",
        default="results/llm_entropy/fisher_runs/fisher_baseline_profiles.json",
    )
    ap.add_argument("--low-quantile", type=float, default=0.2)
    ap.add_argument("--high-quantile", type=float, default=0.8)
    args = ap.parse_args()

    files = sorted({p for g in args.glob for p in glob.glob(g)})
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "fisher_trace" not in df.columns:
            continue
        if "regime" not in df.columns:
            df["regime"] = "default"
        frames.append(df[["regime", "fisher_trace"]].copy())

    if not frames:
        raise SystemExit("No usable fisher_trace columns found")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[np.isfinite(all_df["fisher_trace"].values)]
    if all_df.empty:
        raise SystemExit("All fisher_trace values are NaN/inf")

    regimes: dict[str, dict[str, float]] = {}
    for regime, g in all_df.groupby("regime"):
        vals = g["fisher_trace"].values.astype(float)
        regimes[str(regime)] = {
            "n_samples": int(vals.size),
            "median_trace": float(np.median(vals)),
            "low_trace_threshold": float(np.quantile(vals, args.low_quantile)),
            "high_trace_threshold": float(np.quantile(vals, args.high_quantile)),
            "std_trace": float(np.std(vals)),
        }

    global_vals = all_df["fisher_trace"].values.astype(float)
    profile = {
        "description": "Regime-wise Fisher trace baseline profile for adaptive sampling",
        "low_quantile": float(args.low_quantile),
        "high_quantile": float(args.high_quantile),
        "default_low_trace_threshold": float(np.quantile(global_vals, args.low_quantile)),
        "default_high_trace_threshold": float(np.quantile(global_vals, args.high_quantile)),
        "global_median_trace": float(np.median(global_vals)),
        "regimes": regimes,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (regimes={len(regimes)})", flush=True)


if __name__ == "__main__":
    main()
