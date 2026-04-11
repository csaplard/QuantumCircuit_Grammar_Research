"""
Numerical curvature of Fisher trace vs data length N (Sycamore readout curves).

Reads long-form CSVs such as results/fisher_metric_vs_datalength_all_readouts.csv
(columns include n_points, fisher_trace, file, q_label, topology).

For each readout curve Tr(N), computes d^2 Tr / dN^2 on the discrete N grid using
either irregular three-point finite differences or a natural cubic spline (SciPy).

Outputs:
  - summary: peak location of |d^2 Tr / dN^2| (and signed peak), compared to
    N* from argmax(Tr) on the same grid and optional estimated_N_threshold CSV.

Example:
  python code/curvature_fisher_trace_sycamore.py
  python code/curvature_fisher_trace_sycamore.py --method spline --smooth-sg 7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")


def _prepare_curve(n: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by N, drop duplicate N (keep last), require len >= 3."""
    order = np.argsort(n)
    n, y = n[order].astype(float), y[order].astype(float)
    if len(n) < 3:
        raise ValueError("Need at least 3 points per curve")
    # collapse duplicate N
    uniq_n: list[float] = []
    uniq_y: list[float] = []
    for i in range(len(n)):
        if not uniq_n or n[i] != uniq_n[-1]:
            uniq_n.append(n[i])
            uniq_y.append(y[i])
        else:
            uniq_y[-1] = y[i]
    n2 = np.asarray(uniq_n, dtype=float)
    y2 = np.asarray(uniq_y, dtype=float)
    if len(n2) < 3:
        raise ValueError("Need at least 3 unique N per curve")
    return n2, y2


def second_derivative_irregular(n: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interior standard irregular-grid second derivative; endpoints via quadratic fit."""
    n, y = _prepare_curve(n, y)
    d2 = np.full_like(y, np.nan, dtype=float)
    for i in range(1, len(y) - 1):
        h1 = n[i] - n[i - 1]
        h2 = n[i + 1] - n[i]
        if h1 <= 0 or h2 <= 0:
            continue
        d2[i] = (2.0 / (h1 + h2)) * ((y[i + 1] - y[i]) / h2 - (y[i] - y[i - 1]) / h1)
    # endpoints: fit y = a n^2 + b n + c to first/last three points -> y'' = 2a
    def quad_second_at_end(n3: np.ndarray, y3: np.ndarray, at_start: bool) -> float:
        A = np.column_stack([n3**2, n3, np.ones(3)])
        a, _, _, _ = np.linalg.lstsq(A, y3, rcond=None)
        return 2.0 * a[0]

    d2[0] = quad_second_at_end(n[:3], y[:3], True)
    d2[-1] = quad_second_at_end(n[-3:], y[-3:], False)
    return d2


def second_derivative_spline(n: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Natural cubic spline second derivative evaluated at knots."""
    n, y = _prepare_curve(n, y)
    cs = CubicSpline(n, y, bc_type="natural")
    return cs(n, 2)


def maybe_smooth_uniform(n: np.ndarray, y: np.ndarray, window: int, polyorder: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample to uniform grid in N, apply Savitzky–Golay, return same length as n."""
    n, y = _prepare_curve(n, y)
    n_min, n_max = float(n.min()), float(n.max())
    if n_max <= n_min:
        return n, y
    m = max(len(n), window + 2)
    nu = np.linspace(n_min, n_max, m)
    yu = np.interp(nu, n, y)
    if window % 2 == 0:
        window += 1
    window = min(window, len(yu) // 2 * 2 - 1)
    if window < polyorder + 2:
        return n, y
    yu_s = savgol_filter(yu, window_length=window, polyorder=polyorder)
    y_back = np.interp(n, nu, yu_s)
    return n, y_back


def n_star_trace_max(n: np.ndarray, y: np.ndarray) -> float:
    n, y = _prepare_curve(n, y)
    j = int(np.argmax(y))
    return float(n[j])


def main() -> None:
    ap = argparse.ArgumentParser(description="d^2 Tr/dN^2 on Sycamore Fisher curves")
    ap.add_argument(
        "--input",
        default=os.path.join(RESULTS, "fisher_metric_vs_datalength_all_readouts.csv"),
        help="Long-form Fisher vs N CSV",
    )
    ap.add_argument(
        "--thresholds",
        default=os.path.join(RESULTS, "fisher_estimated_thresholds_per_readout_all_readouts.csv"),
        help="Optional CSV with estimated_N_threshold (merge on file)",
    )
    ap.add_argument(
        "--method",
        choices=("finite", "spline"),
        default="spline",
        help="finite: irregular 3-point; spline: natural cubic spline second derivative",
    )
    ap.add_argument(
        "--smooth-sg",
        type=int,
        default=0,
        metavar="W",
        help="If >0 (odd), smooth Tr on a uniform N grid with Savitzky–Golay window W before d^2/dN^2",
    )
    ap.add_argument(
        "--sg-poly",
        type=int,
        default=3,
        help="Savitzky–Golay polynomial order (default 3)",
    )
    ap.add_argument(
        "--summary-out",
        default=os.path.join(RESULTS, "fisher_curvature_sycamore_summary.csv"),
    )
    ap.add_argument(
        "--per-n-out",
        default=os.path.join(RESULTS, "fisher_curvature_sycamore_per_n.csv"),
        help="Long table of N, Tr, d2Tr/dN2 for every readout",
    )
    ap.add_argument("--no-per-n", action="store_true", help="Skip writing per-N long CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "n_points" not in df.columns or "fisher_trace" not in df.columns:
        raise SystemExit("Expected columns n_points, fisher_trace")
    group_cols = [c for c in ("file", "q_label", "topology") if c in df.columns]
    if not group_cols:
        group_cols = ["file"]

    thr = None
    if args.thresholds and os.path.isfile(args.thresholds):
        thr = pd.read_csv(args.thresholds)

    fn_d2 = second_derivative_spline if args.method == "spline" else second_derivative_irregular

    summary_rows: list[dict] = []
    per_n_rows: list[dict] = []

    for key, g in df.groupby(group_cols, sort=False):
        n = g["n_points"].values
        y = g["fisher_trace"].values
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

        row: dict = {
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
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row[group_cols[0]] = key

        if thr is not None and "estimated_N_threshold" in thr.columns:
            merge_on = [c for c in group_cols if c in thr.columns]
            sub = thr
            for c in merge_on:
                if c not in row:
                    sub = pd.DataFrame()
                    break
                sub = sub[sub[c] == row[c]]
            if len(sub) == 1:
                est = float(sub["estimated_N_threshold"].iloc[0])
                row["estimated_N_threshold"] = est
                row["delta_peak_abs_to_est"] = float(n[j_abs]) - est
                row["delta_nstar_to_est"] = float(nstar) - est

        summary_rows.append(row)

        if not args.no_per_n:
            meta = {c: row[c] for c in group_cols if c in row}
            for i in range(len(n)):
                r = {
                    **meta,
                    "n_points": float(n[i]),
                    "fisher_trace": float(y[i]),
                    "d2_tr_dn2": float(d2[i]),
                    "method": args.method,
                }
                per_n_rows.append(r)

    s_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    s_df.to_csv(args.summary_out, index=False)
    print(f"Wrote {args.summary_out} ({len(s_df)} rows)")

    if not args.no_per_n and per_n_rows:
        pd.DataFrame(per_n_rows).to_csv(args.per_n_out, index=False)
        print(f"Wrote {args.per_n_out} ({len(per_n_rows)} rows)")


if __name__ == "__main__":
    main()
