"""
Phenomenological fits to Fisher trace Tr(N) vs data length N (fast path, no full Fisher matrix).

Reads long-form CSVs with columns n_points, fisher_trace (e.g. results/fisher_metric_vs_datalength_all_readouts.csv).

Fits per curve:
  - saturating exponential: Tr ≈ B + A * (1 - exp(-k * x))
  - logistic (sigmoid):     Tr ≈ L + (U-L) / (1 + exp(-k * (x - x0)))

where x is either N or log10(N) (--abscissa), optionally after Savitzky–Golay smoothing of Tr (--smooth-sg).

Writes a summary CSV with RMSE, parameters, and interpretable scales:
  - n_at_max_slope: inflection x0 (logistic) or characteristic scale 1/k (exp heuristic)
  - n_star_empirical: argmax(Tr) on the grid used for fitting

Use together with curvature_fisher_trace_sycamore.py for d²Tr/dN² peaks; this script summarizes
low-parameter functional shapes (Foti-style transition / saturation language).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from curvature_fisher_trace_sycamore import (  # noqa: E402
    _prepare_curve,
    maybe_smooth_uniform,
    n_star_trace_max,
)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")


def _x_from_n(n: np.ndarray, abscissa: str) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    if abscissa == "log10_n":
        return np.log10(np.maximum(n, 1.0))
    return n


def model_saturating_exp(x: np.ndarray, b: float, a: float, k: float) -> np.ndarray:
    """Tr ~ b + a * (1 - exp(-k x)), a>=0, k>=0."""
    return b + a * (1.0 - np.exp(-k * x))


def model_logistic(x: np.ndarray, L: float, dU: float, k: float, x0: float) -> np.ndarray:
    """Tr ~ L + dU / (1 + exp(-k*(x-x0)))."""
    return L + dU / (1.0 + np.exp(-k * (x - x0)))


def fit_one_curve(
    n: np.ndarray,
    y: np.ndarray,
    abscissa: str,
    smooth_sg: int,
    sg_poly: int,
) -> list[dict]:
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        n, y = _prepare_curve(n, y)
    except ValueError:
        return []

    if smooth_sg and smooth_sg > 0:
        n, y = maybe_smooth_uniform(n, y, window=smooth_sg, polyorder=sg_poly)

    x = _x_from_n(n, abscissa)
    y = np.maximum(y, 1e-30)

    nstar = n_star_trace_max(n, y)

    rows: list[dict] = []

    # --- saturating exponential (4 params reduced to 3: b, a, k)
    b0 = float(np.min(y))
    a0 = float(max(np.max(y) - b0, 1.0))
    k0 = 1.0 / max(float(np.median(n)), 1.0) if abscissa == "n" else 1.0 / max(float(np.median(x)), 1e-6)
    try:
        popt, _ = curve_fit(
            model_saturating_exp,
            x,
            y,
            p0=[b0, a0, k0],
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        b, a, k = (float(popt[0]), float(popt[1]), float(popt[2]))
        pred = model_saturating_exp(x, b, a, k)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        # Half-saturation of (1-exp(-k x)): x_half = ln(2)/k -> N_half = 10^x_half or ln(2)/k
        if k > 0:
            if abscissa == "log10_n":
                x_half = np.log(2.0) / k
                # avoid 10**overflow when k is tiny
                x_half = float(np.clip(x_half, -12.0, 40.0))
                n_char = float(np.power(10.0, x_half))
                n_scale_note = "N_half_sat_in_log10_x"
            else:
                n_char = float(np.log(2.0) / k)
                n_scale_note = "N_half_sat"
        else:
            n_char = float("nan")
            n_scale_note = ""
        rows.append(
            {
                "model": "saturating_exp",
                "abscissa": abscissa,
                "param_b": b,
                "param_a": a,
                "param_k": k,
                "param_L": "",
                "param_dU": "",
                "param_x0": "",
                "rmse": rmse,
                "rmse_over_median_y": rmse / max(float(np.median(y)), 1e-30),
                "n_star_empirical": nstar,
                "n_at_max_slope": n_char,
                "scale_note": n_scale_note,
            }
        )
    except (RuntimeError, ValueError):
        pass

    # --- logistic (tight x0 bounds in x-space to avoid overflow / meaningless scales)
    L0 = float(np.min(y))
    dU0 = float(max(np.max(y) - L0, 1.0))
    x0_0 = float(np.median(x))
    k0_l = 1.0 / max(float(np.std(x)), 1e-6) if len(x) > 2 else 1.0
    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = max(xmax - xmin, 1e-9)
    x0_lo, x0_hi = xmin - 2 * span, xmax + 2 * span
    try:
        popt, _ = curve_fit(
            model_logistic,
            x,
            y,
            p0=[L0, dU0, k0_l, x0_0],
            bounds=(
                [0.0, 0.0, 1e-12, x0_lo],
                [np.inf, np.inf, 500.0, x0_hi],
            ),
            maxfev=20000,
        )
        L, dU, k, x0 = (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
        pred = model_logistic(x, L, dU, k, x0)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        if abscissa == "log10_n":
            n_inflect = float(10**x0)
        else:
            n_inflect = float(x0)
        rows.append(
            {
                "model": "logistic",
                "abscissa": abscissa,
                "param_b": "",
                "param_a": "",
                "param_k": k,
                "param_L": L,
                "param_dU": dU,
                "param_x0": x0,
                "rmse": rmse,
                "rmse_over_median_y": rmse / max(float(np.median(y)), 1e-30),
                "n_star_empirical": nstar,
                "n_at_max_slope": n_inflect,
                "scale_note": "inflection_N",
            }
        )
    except (RuntimeError, ValueError):
        pass

    return rows


def pick_best(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    # Scale-free: raw RMSE is dominated by large-Tr regimes on wide dynamic range
    return min(rows, key=lambda r: r["rmse_over_median_y"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Functional fits Tr(N) — fast trace-only path")
    ap.add_argument(
        "--input",
        default=os.path.join(RESULTS, "fisher_metric_vs_datalength_all_readouts.csv"),
        help="Long-form Fisher vs N CSV",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(RESULTS, "fisher_trace_functional_fit_summary.csv"),
        help="Output summary CSV",
    )
    ap.add_argument(
        "--abscissa",
        choices=("n", "log10_n"),
        default="log10_n",
        help="Fit in N or log10(N); log10 often more stable for wide N range",
    )
    ap.add_argument(
        "--smooth-sg",
        type=int,
        default=0,
        metavar="W",
        help="If >0 (odd), smooth Tr with Savitzky–Golay window W before fitting",
    )
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument(
        "--best-only",
        action="store_true",
        help="Only write the lowest-RMSE model per curve (else all attempted models)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "n_points" not in df.columns or "fisher_trace" not in df.columns:
        raise SystemExit("Expected columns n_points, fisher_trace")

    group_cols = [c for c in ("file", "q_label", "topology", "regime", "control") if c in df.columns]
    if not group_cols:
        group_cols = ["file"]

    out_rows: list[dict] = []

    for key, g in df.groupby(group_cols, sort=False):
        n = g["n_points"].values
        y = g["fisher_trace"].values
        attempted = fit_one_curve(
            n,
            y,
            abscissa=args.abscissa,
            smooth_sg=args.smooth_sg,
            sg_poly=args.sg_poly,
        )
        if not attempted:
            continue

        meta: dict = {}
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                meta[c] = v
        else:
            meta[group_cols[0]] = key

        if args.best_only:
            best = pick_best(attempted)
            if best:
                out_rows.append({**meta, **best})
        else:
            for r in attempted:
                out_rows.append({**meta, **r})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
