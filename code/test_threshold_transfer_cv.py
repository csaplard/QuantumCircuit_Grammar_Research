"""Cross-validation for the log-linear threshold transfer model (LOOCV + grouped holds)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fit_threshold_transfer_model import RESULTS, build_design_matrix, load_data

RESULTS_DIR = RESULTS


def loocv(df: pd.DataFrame) -> pd.DataFrame:
    X, y, _ = build_design_matrix(df)
    n = len(df)
    log_pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        log_pred[i] = float(X[i] @ beta)
    out = df.copy().reset_index(drop=True)
    out["logN"] = y
    out["logN_pred_loo"] = log_pred
    out["N_pred_loo"] = np.exp(log_pred)
    out["abs_err_log"] = np.abs(out["logN"] - out["logN_pred_loo"])
    out["ape_pct"] = (np.abs(out["N_star"] - out["N_pred_loo"]) / np.maximum(out["N_star"], 1e-12)) * 100.0
    return out


def grouped_holdout(df: pd.DataFrame, column: str, value) -> tuple[float, float] | None:
    """Train without rows where df[column]==value; test on those rows. Returns (rmse_log, mape)."""
    train = df[df[column] != value].copy()
    test = df[df[column] == value].copy()
    if len(train) < 3 or len(test) < 1:
        return None
    X_tr, y_tr, _ = build_design_matrix(train)
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    X_te, y_te, _ = build_design_matrix(test)
    log_pred = X_te @ beta
    rmse = float(np.sqrt(np.mean((y_te - log_pred) ** 2)))
    n_obs = np.exp(y_te)
    n_pr = np.exp(log_pred)
    mape = float(np.mean(np.abs(n_obs - n_pr) / np.maximum(n_obs, 1e-12)) * 100.0)
    return rmse, mape


def main() -> None:
    p = argparse.ArgumentParser(description="LOOCV / grouped CV for threshold transfer model")
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "threshold_transfer_loocv.csv")
    p.add_argument("--out-report", type=Path, default=RESULTS_DIR / "threshold_transfer_cv_report.txt")
    args = p.parse_args()

    df = load_data()
    X, y, names = build_design_matrix(df)
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))

    loo = loocv(df)
    rmse_loo = float(np.sqrt(np.mean(loo["abs_err_log"] ** 2)))
    mape_loo = float(loo["ape_pct"].mean())
    med_ape = float(loo["ape_pct"].median())

    rank_note = "" if rank == k else " (WARNING: rank<p, collinear columns)"
    lines = [
        "=== Threshold transfer model - cross-validation ===",
        "",
        f"Design: n={n}, p={k}, rank(X)={rank}{rank_note}",
        f"features: {names}",
        "",
        "--- Leave-one-out (all points) ---",
        f"RMSE (log): {rmse_loo:.4f}",
        f"MAPE (linear %, mean): {mape_loo:.2f}",
        f"MAPE (linear %, median): {med_ape:.2f}",
        "",
    ]

    for col, val, label in [
        ("source", "Sycamore", "Hold out all Sycamore (3 pts), train on IBM"),
        ("backend_group", "marrakesh", "Hold out Marrakesh, train on rest"),
        ("backend_group", "torino", "Hold out Torino, train on rest"),
    ]:
        gh = grouped_holdout(df, col, val)
        if gh is None:
            lines.append(f"--- {label} ---")
            lines.append("  (skipped: insufficient rows)")
        else:
            rmse_g, mape_g = gh
            lines.append(f"--- {label} ---")
            lines.append(f"  RMSE (log) on held-out: {rmse_g:.4f}")
            lines.append(f"  MAPE (%) on held-out: {mape_g:.2f}")
        lines.append("")

    lines.append(
        "Note: 'Hold out Sycamore' removes the reference-regime anchor; "
        "errors are expected to be large unless IBM alone identifies the intercept."
    )

    report_text = "\n".join(lines)
    args.out_report.write_text(report_text, encoding="utf-8")
    loo.to_csv(args.out_csv, index=False)

    print(args.out_report)
    print(args.out_csv)
    print()
    print(report_text)


if __name__ == "__main__":
    main()
