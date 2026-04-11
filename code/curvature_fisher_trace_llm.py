"""
d^2 Tr / dN^2 alignment with N* = argmax(Tr) for LLM Fisher curves (per CSV).

Each file is one curve (e.g. fisher_metric_factual_p0_seed42_none.csv).

Example (seed 42, factual + mathematical, all controls):
  python code/curvature_fisher_trace_llm.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from curvature_fisher_trace_sycamore import (  # noqa: E402
    maybe_smooth_uniform,
    n_star_trace_max,
    second_derivative_irregular,
    second_derivative_spline,
    _prepare_curve,
)

FISHER_RUNS = os.path.join(REPO_ROOT, "results", "llm_entropy", "fisher_runs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Curvature vs N* for LLM Fisher metric CSVs")
    ap.add_argument(
        "--glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob(s) under fisher_runs; default: factual+mathematical seed42",
    )
    ap.add_argument(
        "--method",
        choices=("finite", "spline"),
        default="spline",
    )
    ap.add_argument("--smooth-sg", type=int, default=0, metavar="W")
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument(
        "--summary-out",
        default=os.path.join(FISHER_RUNS, "fisher_curvature_llm_factual_math_seed42_summary.csv"),
    )
    ap.add_argument("--no-per-n", action="store_true")
    ap.add_argument(
        "--per-n-out",
        default=os.path.join(FISHER_RUNS, "fisher_curvature_llm_factual_math_seed42_per_n.csv"),
    )
    args = ap.parse_args()

    patterns = args.glob
    if not patterns:
        patterns = [
            os.path.join(FISHER_RUNS, "fisher_metric_factual_p0_seed42_*.csv"),
            os.path.join(FISHER_RUNS, "fisher_metric_mathematical_p0_seed42_*.csv"),
        ]

    files = sorted({os.path.normpath(p) for pat in patterns for p in glob.glob(pat)})
    fn_d2 = second_derivative_spline if args.method == "spline" else second_derivative_irregular

    summary_rows: list[dict] = []
    per_n_rows: list[dict] = []

    for path in files:
        base = os.path.basename(path)
        if "_meta" in base or not base.endswith(".csv"):
            continue
        df = pd.read_csv(path)
        if "n_points" not in df.columns or "fisher_trace" not in df.columns:
            continue
        n = df["n_points"].values
        y = df["fisher_trace"].values
        try:
            n, y = _prepare_curve(np.asarray(n), np.asarray(y))
        except ValueError:
            continue
        if args.smooth_sg and args.smooth_sg > 0:
            n, y = maybe_smooth_uniform(n, y, window=args.smooth_sg, polyorder=args.sg_poly)
        d2 = fn_d2(n, y)
        nstar = n_star_trace_max(n, y)
        j_abs = int(np.nanargmax(np.abs(d2)))
        j_pos = int(np.nanargmax(d2))
        j_neg = int(np.nanargmin(d2))

        regime = df["regime"].iloc[0] if "regime" in df.columns else ""
        control = df["control"].iloc[0] if "control" in df.columns else ""

        row = {
            "source_csv": os.path.relpath(path, REPO_ROOT),
            "curve_id": base.replace(".csv", ""),
            "regime": regime,
            "control": control,
            "llm_seed": 42,
            "n_star_trace_max": nstar,
            "n_peak_abs_d2": float(n[j_abs]),
            "max_abs_d2_tr_dn2": float(d2[j_abs]),
            "n_peak_max_d2": float(n[j_pos]),
            "max_d2_tr_dn2": float(d2[j_pos]),
            "n_peak_min_d2": float(n[j_neg]),
            "min_d2_tr_dn2": float(d2[j_neg]),
            "abs_delta_peak_abs_to_nstar": abs(float(n[j_abs]) - nstar),
            "method": args.method,
            "smooth_sg": int(args.smooth_sg) if args.smooth_sg else "",
        }
        summary_rows.append(row)

        if not args.no_per_n:
            for i in range(len(n)):
                per_n_rows.append(
                    {
                        "curve_id": row["curve_id"],
                        "regime": regime,
                        "control": control,
                        "n_points": float(n[i]),
                        "fisher_trace": float(y[i]),
                        "d2_tr_dn2": float(d2[i]),
                    }
                )

    s_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    s_df.to_csv(args.summary_out, index=False)
    print(f"Wrote {args.summary_out} ({len(s_df)} rows)")

    if not args.no_per_n and per_n_rows:
        pd.DataFrame(per_n_rows).to_csv(args.per_n_out, index=False)
        print(f"Wrote {args.per_n_out} ({len(per_n_rows)} rows)")


if __name__ == "__main__":
    main()
