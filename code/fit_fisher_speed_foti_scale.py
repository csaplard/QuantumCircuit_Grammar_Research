"""
Fit phenomenological decays to Fisher path speed v(N) = fisher_speed_dn along each readout curve.

Models (least squares on positive speeds):
  (exp)   v(N) = v_inf + A * exp(-N / tau),   tau > 0  (exponential relaxation scale)
  (pow)   v(N) = v_inf + a * N^(-b),         a,b > 0  (power-law tail toward floor v_inf)

Model choice: lower AIC (Gaussian noise on v; same n, comparable).

Cross-scale comparison (narrative / PaW line from Fisher_Curvature_Conjecture_Note.tex):
  - N*_grammar: ``estimated_N_threshold`` from Fisher-trace analysis (same pipeline family).
  - N_foti_anchor: default 8000 (Grammar classification / informational scale cited alongside
    Foti et al. 2021 PaW discussion in-repo — override with --foti-anchor-n).

Outputs per-readout CSV with tau, exponent b, RMSE, AIC, ratios tau/N*, tau/N_anchor.

For exponential fits, also reports **A×τ** and the test invariant ``k*(k-1)*π/2`` (--alphabet-k),
motivated by cross-alphabet comparisons (same seed / pipeline).

Usage:
  python code/fit_fisher_speed_foti_scale.py
  python code/fit_fisher_speed_foti_scale.py --alphabet-k 7
  python code/fit_fisher_speed_foti_scale.py --n-min 2000 --embedding-csv results/fisher_ricci_embedding_alphabet5.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def _qubits_from_label(q_label: str) -> int | None:
    m = re.match(r"^(\d+)q$", str(q_label).strip())
    return int(m.group(1)) if m else None


def k_invariant_pred(k: int) -> float:
    """Hypothesis scale k*(k-1)*π/2 (simplex dimension k-1 coupled to k)."""
    if k < 2:
        return float("nan")
    return float(k * (k - 1) * np.pi / 2.0)


def model_exp(N: np.ndarray, v_inf: float, A: float, tau: float) -> np.ndarray:
    return v_inf + A * np.exp(-N / tau)


def model_pow(N: np.ndarray, v_inf: float, a: float, b: float) -> np.ndarray:
    return v_inf + a * np.power(N, -b)


def aic_gaussian(n: int, rss: float, k: int) -> float:
    """AIC for i.i.d. Gaussian errors: n*log(RSS/n) + 2k (+ const dropped)."""
    if n <= 0 or rss <= 0:
        return float("nan")
    return float(n * np.log(rss / n) + 2 * k)


def fit_one_curve(
    N: np.ndarray,
    v: np.ndarray,
) -> dict:
    """Return best of exp vs pow + diagnostics."""
    N = np.asarray(N, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    m = np.isfinite(N) & np.isfinite(v) & (v > 0)
    N, v = N[m], v[m]
    if len(v) < 5:
        return {
            "status": "too_few_points",
            "n_points_used": len(v),
            "preferred_model": "none",
            "delta_aic": float("nan"),
        }

    vmin = float(np.min(v))
    # Asymptotic floor must stay strictly below observed speeds if A>0 and exp is decreasing.
    v_inf_hi = max(vmin * (1.0 - 1e-6), 1e-24)
    v_inf_lo = 0.0

    out: dict = {"status": "ok", "n_points_used": len(v)}

    # --- exponential
    try:
        A0 = float(np.max(v) - np.min(v))
        tau0 = float(np.median(N))
        p0_v = min(float(np.min(v) * 0.25), v_inf_hi * 0.95)
        p0 = [p0_v, max(A0, 1e-15), max(tau0, 300.0)]
        bounds = ([v_inf_lo, 0.0, 50.0], [v_inf_hi, 1e6, 5e5])
        popt, _ = curve_fit(
            model_exp,
            N,
            v,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        v_inf_e, A_e, tau_e = (float(popt[0]), float(popt[1]), float(popt[2]))
        pred_e = model_exp(N, v_inf_e, A_e, tau_e)
        rss_e = float(np.sum((v - pred_e) ** 2))
        rmse_e = float(np.sqrt(rss_e / len(v)))
        aic_e = aic_gaussian(len(v), rss_e, k=3)
        out.update(
            {
                "exp_v_inf": v_inf_e,
                "exp_A": A_e,
                "exp_tau": tau_e,
                "exp_rmse": rmse_e,
                "exp_aic": aic_e,
            }
        )
    except (RuntimeError, ValueError):
        out["exp_tau"] = float("nan")
        out["exp_rmse"] = float("nan")
        out["exp_aic"] = float("nan")

    # --- power law
    try:
        p0_v2 = min(float(np.min(v) * 0.25), v_inf_hi * 0.95)
        p0p = [p0_v2, max(float(np.max(v) - np.min(v)) * (float(np.min(N)) ** 0.5), 1e-15), 0.8]
        bounds_p = ([v_inf_lo, 0.0, 0.05], [v_inf_hi, 1e6, 5.0])
        poptp, _ = curve_fit(
            model_pow,
            N,
            v,
            p0=p0p,
            bounds=bounds_p,
            maxfev=20000,
        )
        v_inf_p, a_p, b_p = (float(poptp[0]), float(poptp[1]), float(poptp[2]))
        pred_p = model_pow(N, v_inf_p, a_p, b_p)
        rss_p = float(np.sum((v - pred_p) ** 2))
        rmse_p = float(np.sqrt(rss_p / len(v)))
        aic_p = aic_gaussian(len(v), rss_p, k=3)
        out.update(
            {
                "pow_v_inf": v_inf_p,
                "pow_a": a_p,
                "pow_b": b_p,
                "pow_rmse": rmse_p,
                "pow_aic": aic_p,
            }
        )
    except (RuntimeError, ValueError):
        out["pow_b"] = float("nan")
        out["pow_rmse"] = float("nan")
        out["pow_aic"] = float("nan")

    # pick winner
    ae, ap = out.get("exp_aic", float("nan")), out.get("pow_aic", float("nan"))
    if np.isfinite(ae) and np.isfinite(ap):
        out["preferred_model"] = "exp" if ae < ap else "pow"
        out["delta_aic"] = float(abs(ae - ap))
    elif np.isfinite(ae):
        out["preferred_model"] = "exp"
        out["delta_aic"] = float("nan")
    elif np.isfinite(ap):
        out["preferred_model"] = "pow"
        out["delta_aic"] = float("nan")
    else:
        out["preferred_model"] = "none"
        out["delta_aic"] = float("nan")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit power/exp decay to fisher_speed_dn(N)")
    ap.add_argument(
        "--embedding-csv",
        default=os.path.join(RESULTS_DIR, "fisher_ricci_and_embedding_vs_n.csv"),
        help="Must contain file, n_points, fisher_speed_dn; measured rows used by default.",
    )
    ap.add_argument(
        "--thresholds-csv",
        default=os.path.join(RESULTS_DIR, "fisher_estimated_thresholds_per_readout_all_readouts.csv"),
        help="Merge estimated_N_threshold (N*_grammar) on file.",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(RESULTS_DIR, "fisher_speed_foti_scale_fit.csv"),
    )
    ap.add_argument(
        "--n-min",
        type=float,
        default=500.0,
        help="Use only grid points with N >= this (after dropping NaN speeds).",
    )
    ap.add_argument(
        "--foti-anchor-n",
        type=float,
        default=8000.0,
        help="Reference scale N for ratio columns (PaW/grammar narrative anchor; override freely).",
    )
    ap.add_argument(
        "--include-non-measured",
        action="store_true",
        help="Do not filter data_source == measured",
    )
    ap.add_argument(
        "--alphabet-k",
        type=int,
        default=7,
        metavar="K",
        help="SAX alphabet size for A*tau vs k*(k-1)*pi/2 comparison (LLM usually 7).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.embedding_csv)
    if not args.include_non_measured and "data_source" in df.columns:
        df = df[df["data_source"].astype(str) == "measured"].copy()

    thr = pd.read_csv(args.thresholds_csv) if os.path.isfile(args.thresholds_csv) else None

    rows: list[dict] = []

    for fname, g in df.groupby("file", sort=False):
        g = g.sort_values("n_points")
        N = g["n_points"].values.astype(float)
        v = g["fisher_speed_dn"].values.astype(float)
        m = np.isfinite(v) & (N >= args.n_min)
        Nf, vf = N[m], v[m]

        q_label = str(g["q_label"].iloc[0])
        topo = str(g["topology"].iloc[0])
        qubits = _qubits_from_label(q_label)

        fit = fit_one_curve(Nf, vf)
        row: dict = {
            "file": fname,
            "q_label": q_label,
            "topology": topo,
            "qubits": qubits if qubits is not None else "",
            "n_min_filter": args.n_min,
        }
        row.update(fit)

        n_star = float("nan")
        if thr is not None and "file" in thr.columns and "estimated_N_threshold" in thr.columns:
            sub = thr[thr["file"] == fname]
            if len(sub) == 1:
                n_star = float(sub["estimated_N_threshold"].iloc[0])
        row["N_star_grammar"] = n_star

        tau = row.get("exp_tau", float("nan"))
        if np.isfinite(tau) and np.isfinite(n_star) and n_star > 0:
            row["ratio_tau_over_N_star"] = float(tau / n_star)
        else:
            row["ratio_tau_over_N_star"] = float("nan")

        if np.isfinite(tau) and args.foti_anchor_n > 0:
            row["ratio_tau_over_N_foti_anchor"] = float(tau / args.foti_anchor_n)
        else:
            row["ratio_tau_over_N_foti_anchor"] = float("nan")

        row["N_foti_anchor"] = float(args.foti_anchor_n)

        A_e = row.get("exp_A", float("nan"))
        tau_e = row.get("exp_tau", float("nan"))
        if np.isfinite(A_e) and np.isfinite(tau_e):
            row["A_times_tau"] = float(A_e * tau_e)
        else:
            row["A_times_tau"] = float("nan")
        row["alphabet_k"] = int(args.alphabet_k)
        inv = k_invariant_pred(args.alphabet_k)
        row["invariant_k_kminus1_pi_half"] = inv
        at = row["A_times_tau"]
        if np.isfinite(at) and np.isfinite(inv) and inv > 0:
            row["ratio_A_tau_over_invariant"] = float(at / inv)
        else:
            row["ratio_A_tau_over_invariant"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["topology", "q_label"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    # brief stdout summary
    print(f"Wrote {args.out} ({len(out)} rows)")
    if "preferred_model" in out.columns:
        pref = out["preferred_model"].value_counts(dropna=False).to_dict()
        print(f"Preferred model counts: {pref}")
    med_tau = out["exp_tau"].median(skipna=True) if "exp_tau" in out.columns else float("nan")
    med_ns = out["N_star_grammar"].median(skipna=True) if "N_star_grammar" in out.columns else float("nan")
    if np.isfinite(med_tau) and np.isfinite(med_ns) and med_ns > 0:
        print(f"Median exp_tau: {med_tau:.1f}  |  median N*_grammar: {med_ns:.1f}  |  median tau/N*: {med_tau/med_ns:.3f}")
    med_at = out["A_times_tau"].median(skipna=True) if "A_times_tau" in out.columns else float("nan")
    inv = k_invariant_pred(args.alphabet_k)
    if np.isfinite(med_at) and np.isfinite(inv) and inv > 0:
        print(
            f"Median A*tau: {med_at:.4g}  |  k(k-1)*pi/2 for k={args.alphabet_k}: {inv:.6g}  |  median ratio: {med_at/inv:.4f}"
        )


if __name__ == "__main__":
    main()
