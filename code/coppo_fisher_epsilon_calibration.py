"""
Map exponential Fisher-speed fits v(N)=v∞+A·exp(−N/τ) to a tolerance threshold N⋆(ε).

Operational definition (overlap-style tolerance on the *excess* speed above the floor):
  A·exp(−N⋆/τ) = ε   ⇒   N⋆(ε) = τ·(ln A − ln ε)   (requires A > ε > 0).

This exhibits the same **linear dependence on ln ε** as Coppo-type threshold formulas where
the underlying quantity decays exponentially in N (cf. GCS overlap); the slope ∂N⋆/∂(ln ε) = −τ.

Coppo (su(2), schematic): N_t = ln(ε) / ln[cos(δ/2)] + n — also linear in ln ε.
Bosonic (Eq. 27 form): N_t = −ln(ε)/δ² — linear in ln ε with a different prefactor.

This script does **not** identify δ or ε_phys; it exports N⋆(ε) for a grid of numerical ε and
compares the implied slope to −τ from the exponential fit (sanity check).

Inputs: ``results/fisher_speed_foti_scale_fit.csv`` (must contain exp_tau, exp_A for exp-preferred rows).

Usage:
  python code/coppo_fisher_epsilon_calibration.py
  python code/coppo_fisher_epsilon_calibration.py --eps 1e-6 1e-4 1e-2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def n_threshold_from_exp(A: float, tau: float, eps: float) -> float:
    if not (np.isfinite(A) and np.isfinite(tau) and np.isfinite(eps)):
        return float("nan")
    if A <= 0 or tau <= 0 or eps <= 0:
        return float("nan")
    if eps >= A:
        return float("nan")
    return float(tau * (np.log(A) - np.log(eps)))


def main() -> None:
    ap = argparse.ArgumentParser(description="N⋆(ε) from exponential Fisher-speed fit vs ln ε")
    ap.add_argument(
        "--fit-csv",
        default=os.path.join(RESULTS_DIR, "fisher_speed_foti_scale_fit.csv"),
    )
    ap.add_argument(
        "--out-long",
        default=os.path.join(RESULTS_DIR, "coppo_style_Nstar_vs_epsilon_long.csv"),
    )
    ap.add_argument(
        "--out-slopes",
        default=os.path.join(RESULTS_DIR, "coppo_style_ln_eps_slope_check.csv"),
    )
    ap.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="ε grid (default: 1e-8 1e-6 1e-4 1e-3 1e-2)",
    )
    args = ap.parse_args()

    eps_list = (
        args.eps
        if args.eps
        else [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    )

    df = pd.read_csv(args.fit_csv)
    sub = df[df["preferred_model"].astype(str) == "exp"].copy()
    if sub.empty:
        raise SystemExit("No exp-preferred rows in fit CSV; run fit_fisher_speed_foti_scale.py first.")

    long_rows: list[dict] = []
    slope_rows: list[dict] = []

    for _, row in sub.iterrows():
        A = float(row["exp_A"])
        tau = float(row["exp_tau"])
        fname = str(row["file"])
        for eps in eps_list:
            nstar = n_threshold_from_exp(A, tau, eps)
            long_rows.append(
                {
                    "file": fname,
                    "q_label": row.get("q_label", ""),
                    "topology": row.get("topology", ""),
                    "epsilon": eps,
                    "ln_epsilon": np.log(eps),
                    "N_star_epsilon": nstar,
                    "exp_tau": tau,
                    "exp_A": A,
                }
            )

        # slope of N_star vs ln_epsilon over valid eps (finite N_star)
        xs, ys = [], []
        for eps in eps_list:
            ns = n_threshold_from_exp(A, tau, eps)
            if np.isfinite(ns):
                xs.append(np.log(eps))
                ys.append(ns)
        if len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)  # ys = c1*x + c0
            slope = float(coef[0])
        else:
            slope = float("nan")
        slope_rows.append(
            {
                "file": fname,
                "q_label": row.get("q_label", ""),
                "topology": row.get("topology", ""),
                "slope_N_vs_ln_eps": slope,
                "minus_tau": float(-tau),
                "abs_slope_plus_tau": abs(slope + tau) if np.isfinite(slope) else float("nan"),
            }
        )

    pd.DataFrame(long_rows).to_csv(args.out_long, index=False)
    pd.DataFrame(slope_rows).to_csv(args.out_slopes, index=False)
    print(f"Wrote {args.out_long}")
    print(f"Wrote {args.out_slopes}")
    print(
        "Check: N*(eps)=tau*(ln A - ln eps) => dN*/d(ln eps) = -tau ; "
        "compare slope_N_vs_ln_eps with minus_tau on the epsilon grid."
    )


if __name__ == "__main__":
    main()
