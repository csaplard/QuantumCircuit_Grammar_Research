from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

SYC_THRESHOLD_CSV = RESULTS / "fisher_estimated_thresholds_per_readout_all_readouts.csv"
IBM_THRESHOLD_CSV = RESULTS / "ibm_fisher_thresholds_normalized.csv"

RIDGE_ALPHAS = np.logspace(-2, 5, 40)


def load_ibm_rows(*, coarse_regimes: bool = False) -> pd.DataFrame:
    ibm = pd.read_csv(IBM_THRESHOLD_CSV)
    ibm = ibm.copy()
    ibm["source"] = "IBM"
    ibm["backend_group"] = ibm["backend"].astype(str)
    if coarse_regimes:
        ibm["regime_family"] = ibm["circuit"].map(
            {"ghz": "GHZ", "hadamard": "HadamardLayers", "layers": "HadamardLayers", "identity": "Identity"}
        )
    else:
        ibm["regime_family"] = ibm["circuit"].astype(str).str.lower()
    ibm = ibm.rename(columns={"n_qubits": "Q", "threshold_estimate": "N_star"})
    ibm = ibm[["source", "backend_group", "regime_family", "Q", "N_star", "file"]]
    before = len(ibm)
    ibm = ibm.drop_duplicates(subset=["backend_group", "regime_family", "Q"], keep="first")
    if len(ibm) < before:
        ibm = ibm.reset_index(drop=True)
    return ibm


def load_sycamore_rows(path: Path | None = None) -> pd.DataFrame:
    p = path or SYC_THRESHOLD_CSV
    if not p.is_file():
        raise FileNotFoundError(f"Missing Sycamore Fisher thresholds: {p}")
    s = pd.read_csv(p)
    s = s.copy()
    s["Q"] = s["q_label"].astype(str).str.replace("q", "", case=False).astype(int)
    s["source"] = "Sycamore"
    s["backend_group"] = "sycamore"
    s["regime_family"] = s["topology"].astype(str)
    s["N_star"] = s["estimated_N_threshold"].astype(float)
    s["file"] = s["file"].astype(str)
    return s[["source", "backend_group", "regime_family", "Q", "N_star", "file"]]


def load_legacy_sycamore_three() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": "Sycamore", "backend_group": "sycamore", "regime_family": "SycamoreRandom", "Q": 26, "N_star": 6500.0, "file": "legacy26"},
            {"source": "Sycamore", "backend_group": "sycamore", "regime_family": "SycamoreRandom", "Q": 14, "N_star": 8500.0, "file": "legacy14"},
            {"source": "Sycamore", "backend_group": "sycamore", "regime_family": "SycamoreRandom", "Q": 12, "N_star": 9500.0, "file": "legacy12"},
        ]
    )


def winsorize_series(s: pd.Series, low_pct: float, high_pct: float) -> tuple[pd.Series, float, float]:
    lo = float(s.quantile(low_pct))
    hi = float(s.quantile(high_pct))
    return s.clip(lo, hi), lo, hi


def build_design_matrix_legacy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = df.copy()
    d["logN"] = np.log(np.maximum(d["N_star"].astype(float), 1.0))
    d["logQ"] = np.log(np.maximum(d["Q"].astype(float), 1.0))

    cols = [np.ones(len(d)), d["logQ"].to_numpy()]
    names: list[str] = ["intercept", "logQ"]

    for b in ["marrakesh", "torino"]:
        cols.append((d["backend_group"] == b).astype(float).to_numpy())
        names.append(f"backend:{b}")
    for r in ["GHZ", "HadamardLayers"]:
        cols.append((d["regime_family"] == r).astype(float).to_numpy())
        names.append(f"regime:{r}")

    X = np.column_stack(cols)
    y = d["logN"].to_numpy()
    return X, y, names


def build_design_matrix_full(df: pd.DataFrame, *, granular_ibm: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """log(N*) ~ intercept + logQ + backends + regimes (ref sycamore, ref 2D_Block)."""
    d = df.copy()
    d["logN"] = np.log(np.maximum(d["N_star"].astype(float), 1.0))
    d["logQ"] = np.log(np.maximum(d["Q"].astype(float), 1.0))

    marrakesh = (d["backend_group"] == "marrakesh").astype(float).to_numpy()
    torino = (d["backend_group"] == "torino").astype(float).to_numpy()
    rf = d["regime_family"].astype(str)
    if granular_ibm:
        rl = rf.str.lower()
        ibm_cols = [
            (rl == "ghz").astype(float).to_numpy(),
            (rl == "hadamard").astype(float).to_numpy(),
            (rl == "layers").astype(float).to_numpy(),
            (rl == "identity").astype(float).to_numpy(),
        ]
        ibm_names = ["regime:ghz", "regime:hadamard", "regime:layers", "regime:identity"]
    else:
        ibm_cols = [
            (rf == "GHZ").astype(float).to_numpy(),
            (rf == "HadamardLayers").astype(float).to_numpy(),
            (rf == "Identity").astype(float).to_numpy(),
        ]
        ibm_names = ["regime:GHZ", "regime:HadamardLayers", "regime:Identity"]
    syc1d = (rf == "1D_Snake").astype(float).to_numpy()
    syc_bulk = (rf == "Bulk_Full").astype(float).to_numpy()

    cols = [np.ones(len(d)), d["logQ"].to_numpy(), marrakesh, torino, *ibm_cols, syc1d, syc_bulk]
    names = ["intercept", "logQ", "backend:marrakesh", "backend:torino", *ibm_names, "regime:1D_Snake", "regime:Bulk_Full"]
    X = np.column_stack(cols)
    y = d["logN"].to_numpy()
    return X, y, names


def build_feature_matrix_no_intercept(
    df: pd.DataFrame, include_log_q: bool, *, granular_ibm: bool
) -> tuple[np.ndarray, list[str]]:
    """For Ridge: sklearn adds intercept. Optionally omit logQ for scaled target N*sqrt(Q)."""
    d = df.copy()
    marrakesh = (d["backend_group"] == "marrakesh").astype(float).to_numpy()
    torino = (d["backend_group"] == "torino").astype(float).to_numpy()
    rf = d["regime_family"].astype(str)
    if granular_ibm:
        rl = rf.str.lower()
        ibm_cols = [
            (rl == "ghz").astype(float).to_numpy(),
            (rl == "hadamard").astype(float).to_numpy(),
            (rl == "layers").astype(float).to_numpy(),
            (rl == "identity").astype(float).to_numpy(),
        ]
        ibm_names = ["regime:ghz", "regime:hadamard", "regime:layers", "regime:identity"]
    else:
        ibm_cols = [
            (rf == "GHZ").astype(float).to_numpy(),
            (rf == "HadamardLayers").astype(float).to_numpy(),
            (rf == "Identity").astype(float).to_numpy(),
        ]
        ibm_names = ["regime:GHZ", "regime:HadamardLayers", "regime:Identity"]
    syc1d = (rf == "1D_Snake").astype(float).to_numpy()
    syc_bulk = (rf == "Bulk_Full").astype(float).to_numpy()
    names: list[str] = []
    cols: list[np.ndarray] = []
    if include_log_q:
        logq = np.log(np.maximum(d["Q"].astype(float).to_numpy(), 1.0))
        cols.append(logq)
        names.append("logQ")
    cols.extend([marrakesh, torino, *ibm_cols, syc1d, syc_bulk])
    names.extend(["backend:marrakesh", "backend:torino", *ibm_names, "regime:1D_Snake", "regime:Bulk_Full"])
    return np.column_stack(cols), names


def fit_ls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def n_from_scaled_target(y_log_scaled: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """y = log(N * sqrt(Q)) -> N = exp(y) / sqrt(Q)."""
    Qs = np.maximum(Q.astype(float), 1.0)
    return np.exp(y_log_scaled) / np.sqrt(Qs)


def logn_metrics(y_true_log_n: np.ndarray, n_pred: np.ndarray) -> tuple[float, float]:
    log_p = np.log(np.maximum(n_pred, 1.0))
    rmse = float(np.sqrt(np.mean((y_true_log_n - log_p) ** 2)))
    mae = float(np.mean(np.abs(y_true_log_n - log_p)))
    return rmse, mae


def make_folds(df: pd.DataFrame, n_splits: int, seed: int, legacy: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    if legacy:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        folds = []
        for k in range(n_splits):
            te = parts[k]
            tr = np.concatenate([parts[i] for i in range(n_splits) if i != k])
            folds.append((tr, te))
        return folds
    y_strat = (df["source"] == "IBM").astype(int).to_numpy()
    min_class = int(np.minimum(np.sum(y_strat == 0), np.sum(y_strat == 1)))
    if min_class < n_splits:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.zeros(n)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(n), y_strat))


def cv_baseline_ols(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    legacy: bool,
    granular_ibm: bool,
) -> pd.DataFrame:
    rows = []
    for k, (tr, te) in enumerate(folds):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        if legacy:
            X_tr, y_tr, _ = build_design_matrix_legacy(d_tr)
            X_te, y_te, _ = build_design_matrix_legacy(d_te)
        else:
            X_tr, y_tr, _ = build_design_matrix_full(d_tr, granular_ibm=granular_ibm)
            X_te, y_te, _ = build_design_matrix_full(d_te, granular_ibm=granular_ibm)
        beta = fit_ls(X_tr, y_tr)
        pred = X_te @ beta
        rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        mae = float(np.mean(np.abs(pred - y_te)))
        rows.append({"fold": k, "n_train": len(tr), "n_test": len(te), "rmse_logN": rmse, "mae_logN": mae})
    return pd.DataFrame(rows)


def _ridge_train_predict_eval(
    d_tr: pd.DataFrame,
    d_te: pd.DataFrame,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> tuple[float, float, float]:
    raw_tr = d_tr["N_star"].astype(float)
    lo, hi = raw_tr.quantile(winsor_low), raw_tr.quantile(winsor_high)
    d_tr = d_tr.copy()
    d_te = d_te.copy()
    d_tr["N_w"] = d_tr["N_star"].astype(float).clip(lo, hi)

    Q_tr = np.maximum(d_tr["Q"].astype(float).to_numpy(), 1.0)
    Q_te = np.maximum(d_te["Q"].astype(float).to_numpy(), 1.0)
    Nw_tr = np.maximum(d_tr["N_w"].to_numpy(), 1.0)

    if scaled_target:
        y_tr = np.log(Nw_tr * np.sqrt(Q_tr))
    else:
        y_tr = np.log(Nw_tr)

    X_tr, _ = build_feature_matrix_no_intercept(d_tr, include_log_q, granular_ibm=granular_ibm)
    X_te, _ = build_feature_matrix_no_intercept(d_te, include_log_q, granular_ibm=granular_ibm)

    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
    ridge.fit(X_tr, y_tr)
    y_pred_te = ridge.predict(X_te)

    if scaled_target:
        n_pred = n_from_scaled_target(y_pred_te, Q_te)
    else:
        n_pred = np.exp(y_pred_te)

    y_true_log_n = np.log(np.maximum(d_te["N_star"].astype(float).to_numpy(), 1.0))
    rmse, mae = logn_metrics(y_true_log_n, n_pred)
    return rmse, mae, float(ridge.alpha_)


def cv_improved(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> pd.DataFrame:
    """Train on winsorized N; report RMSE in log(N_raw) space."""
    rows = []
    for k, (tr, te) in enumerate(folds):
        d_tr = df.iloc[tr]
        d_te = df.iloc[te]
        rmse, mae, alpha = _ridge_train_predict_eval(
            d_tr, d_te, winsor_low, winsor_high, scaled_target, include_log_q, granular_ibm
        )
        rows.append(
            {
                "fold": k,
                "n_train": len(tr),
                "n_test": len(te),
                "rmse_logN": rmse,
                "mae_logN": mae,
                "ridge_alpha": alpha,
            }
        )
    return pd.DataFrame(rows)


def cv_leave_one_sycamore_topology(
    df: pd.DataFrame,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> pd.DataFrame:
    """Train on IBM + Sycamore rows excluding one layout topology; test only on held-out topology."""
    rows = []
    for held in ("1D_Snake", "2D_Block", "Bulk_Full"):
        test_mask = (df["source"] == "Sycamore") & (df["regime_family"] == held)
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        rmse, mae, alpha = _ridge_train_predict_eval(
            df.iloc[tr],
            df.iloc[te],
            winsor_low,
            winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        rows.append(
            {
                "held_out_topology": held,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "rmse_logN": rmse,
                "mae_logN": mae,
                "ridge_alpha": alpha,
            }
        )
    return pd.DataFrame(rows)


def fit_full_baseline(
    df: pd.DataFrame, legacy: bool, granular_ibm: bool
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray, list[str]]:
    d = df.copy()
    if legacy:
        X, y, names = build_design_matrix_legacy(d)
    else:
        X, y, names = build_design_matrix_full(d, granular_ibm=granular_ibm)
    beta = fit_ls(X, y)
    coef = dict(zip(names, beta))
    log_pred = X @ beta
    d["logN"] = y
    d["logN_pred"] = log_pred
    d["N_pred"] = np.exp(log_pred)
    d["residual_log"] = y - log_pred

    rmse_log = float(np.sqrt(np.mean((y - log_pred) ** 2)))
    r2 = float(1 - np.sum((y - log_pred) ** 2) / np.sum((y - y.mean()) ** 2))

    factors: dict[str, float] = {"alpha_logQ": float(coef.get("logQ", 0.0)), "RMSE_log_full": rmse_log, "R2_log_full": r2}
    if not legacy:
        factors["H0_baseline"] = float(np.exp(coef["intercept"]))
        factors["B_marrakesh"] = float(np.exp(coef["backend:marrakesh"]))
        factors["B_torino"] = float(np.exp(coef["backend:torino"]))
        for key, val in coef.items():
            if key.startswith("regime:"):
                factors[f"exp({key})"] = float(np.exp(val))
        if not granular_ibm:
            factors["C_2D_Block_ref"] = 1.0
    else:
        factors["H0_baseline"] = float(np.exp(coef["intercept"]))
        factors["B_marrakesh"] = float(np.exp(coef["backend:marrakesh"]))
        factors["B_torino"] = float(np.exp(coef["backend:torino"]))
        factors["C_GHZ"] = float(np.exp(coef["regime:GHZ"]))
        factors["C_HadamardLayers"] = float(np.exp(coef["regime:HadamardLayers"]))

    return d, factors, beta, names


def fit_full_improved(
    df: pd.DataFrame,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> tuple[pd.DataFrame, dict[str, float], RidgeCV, list[str]]:
    s = df["N_star"].astype(float)
    lo, hi = s.quantile(winsor_low), s.quantile(winsor_high)
    d = df.copy()
    d["N_star_winsor"] = s.clip(lo, hi)
    d["N_star_raw"] = s

    Q = np.maximum(d["Q"].astype(float).to_numpy(), 1.0)
    Nw = np.maximum(d["N_star_winsor"].to_numpy(), 1.0)
    if scaled_target:
        y = np.log(Nw * np.sqrt(Q))
    else:
        y = np.log(Nw)

    X, fnames = build_feature_matrix_no_intercept(d, include_log_q, granular_ibm=granular_ibm)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
    ridge.fit(X, y)

    y_hat = ridge.predict(X)
    if scaled_target:
        n_pred = n_from_scaled_target(y_hat, Q)
    else:
        n_pred = np.exp(y_hat)

    log_n_true = np.log(np.maximum(d["N_star_raw"].to_numpy(), 1.0))
    log_n_pred = np.log(np.maximum(n_pred, 1.0))

    d["logN"] = log_n_true
    d["logN_pred"] = log_n_pred
    d["N_pred"] = n_pred
    d["residual_log"] = log_n_true - log_n_pred

    rmse_log = float(np.sqrt(np.mean((log_n_true - log_n_pred) ** 2)))
    r2 = float(1 - np.sum((log_n_true - log_n_pred) ** 2) / np.sum((log_n_true - log_n_true.mean()) ** 2))

    factors: dict[str, float] = {
        "RMSE_log_full": rmse_log,
        "R2_log_full": r2,
        "ridge_alpha": float(ridge.alpha_),
        "winsor_low_q": winsor_low,
        "winsor_high_q": winsor_high,
        "scaled_target_log_N_sqrt_Q": float(scaled_target),
        "include_logQ": float(include_log_q),
        "granular_ibm": float(granular_ibm),
    }
    for j, name in enumerate(fnames):
        factors[f"coef:{name}"] = float(ridge.coef_[j])
    factors["intercept"] = float(ridge.intercept_)

    return d, factors, ridge, fnames


def rmse_by_source(df_pred: pd.DataFrame) -> dict[str, float]:
    out = {}
    for src in df_pred["source"].unique():
        g = df_pred[df_pred["source"] == src]
        e = g["logN"] - g["logN_pred"]
        out[f"RMSE_log_{src}"] = float(np.sqrt(np.mean(e**2)))
    return out


def cv_within_source(
    df: pd.DataFrame,
    source_name: str,
    n_splits: int,
    seed: int,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> pd.DataFrame:
    sub = df[df["source"] == source_name].reset_index(drop=True)
    n = len(sub)
    if n < 4:
        return pd.DataFrame()
    k = min(n_splits, n // 2)
    if k < 2:
        return pd.DataFrame()
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for fold_id, (tr, te) in enumerate(kf.split(np.zeros(n))):
        rmse, mae, alpha = _ridge_train_predict_eval(
            sub.iloc[tr],
            sub.iloc[te],
            winsor_low,
            winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        rows.append(
            {
                "source": source_name,
                "fold": fold_id,
                "n_train": len(tr),
                "n_test": len(te),
                "rmse_logN": rmse,
                "mae_logN": mae,
                "ridge_alpha": alpha,
            }
        )
    return pd.DataFrame(rows)


def permutation_test_r2(
    df: pd.DataFrame,
    n_reps: int,
    seed: int,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> tuple[float, float]:
    _, fac_obs, _, _ = fit_full_improved(
        df, winsor_low, winsor_high, scaled_target, include_log_q, granular_ibm
    )
    r2_obs = float(fac_obs["R2_log_full"])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_reps):
        d = df.copy()
        d["N_star"] = rng.permutation(d["N_star"].values)
        _, fac, _, _ = fit_full_improved(
            d, winsor_low, winsor_high, scaled_target, include_log_q, granular_ibm
        )
        if float(fac["R2_log_full"]) >= r2_obs:
            exceed += 1
    p = (1 + exceed) / (n_reps + 1)
    return r2_obs, p


def bootstrap_r2_ci(
    df: pd.DataFrame,
    n_reps: int,
    seed: int,
    winsor_low: float,
    winsor_high: float,
    scaled_target: bool,
    include_log_q: bool,
    granular_ibm: bool,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(df)
    samples: list[float] = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        d = df.iloc[idx].copy().reset_index(drop=True)
        _, fac, _, _ = fit_full_improved(
            d, winsor_low, winsor_high, scaled_target, include_log_q, granular_ibm
        )
        samples.append(float(fac["R2_log_full"]))
    arr = np.array(samples, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(np.mean(arr))


def save_plot(df_pred: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    color_map = {"Sycamore": "#1f77b4", "IBM": "#ff7f0e"}
    for src, g in df_pred.groupby("source"):
        ax.scatter(
            g["N_star"],
            g["N_pred"],
            s=50 if src == "IBM" else 28,
            alpha=0.8,
            label=src,
            color=color_map.get(src, "#333333"),
        )

    mn = float(min(df_pred["N_star"].min(), df_pred["N_pred"].min()))
    mx = float(max(df_pred["N_star"].max(), df_pred["N_pred"].max()))
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed N* (raw)")
    ax.set_ylabel("Predicted N*")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def write_onepager(
    df_pred: pd.DataFrame,
    factors: dict[str, float],
    factors_baseline: dict[str, float],
    cv_imp: pd.DataFrame,
    cv_base: pd.DataFrame,
    cv_topo: pd.DataFrame | None,
    by_src: dict[str, float],
    out_path: Path,
    n_ibm: int,
    n_syc: int,
    legacy: bool,
) -> None:
    lines = [
        "=== Threshold transfer model — improved pipeline ===",
        "",
        "Improvements vs plain OLS on log(N):",
        "  - Stratified K-fold (IBM vs Sycamore balance in each fold).",
        "  - Winsorize N* on training fold (default 5–95% quantiles) to cut spike leverage.",
        "  - Ridge regression with CV-chosen alpha (less overfit on small IBM set).",
        "  - Default target: log(N*sqrt(Q)) with only backend+regime features (theory: N* ~ 1/sqrt(Q));",
        "    predictions mapped back to N; metrics in log(N_raw) vs log(N_pred).",
        "",
        f"Data: IBM rows (dedup): {n_ibm}; Sycamore: {n_syc}. Legacy mode: {legacy}.",
        "IBM regimes: separate ghz / hadamard / layers / identity unless --coarse-ibm-regimes.",
        "",
        "Improved — full-sample fit on raw N (reported in log-space):",
        f"  RMSE log N: {factors.get('RMSE_log_full', 0):.4f}",
        f"  R^2 log: {factors.get('R2_log_full', 0):.4f}",
        f"  chosen ridge alpha (full fit): {factors.get('ridge_alpha', float('nan')):.4f}",
    ]
    for k, v in sorted(by_src.items()):
        lines.append(f"  {k}: {v:.4f}")
    lines.extend(
        [
            "",
            "Baseline OLS — full-sample fit (same design matrix; target log(N*) linear regression):",
            f"  R^2 log: {factors_baseline.get('R2_log_full', float('nan')):.4f}",
            f"  RMSE log: {factors_baseline.get('RMSE_log_full', float('nan')):.4f}",
            "",
            "Improved — CV mean RMSE (log N_raw):",
            f"  {cv_imp['rmse_logN'].mean():.4f} ± {cv_imp['rmse_logN'].std():.4f}",
            "",
            "Baseline OLS — CV mean RMSE (same folds, log N in training space):",
            f"  {cv_base['rmse_logN'].mean():.4f} ± {cv_base['rmse_logN'].std():.4f}",
            "",
        ]
    )
    if cv_topo is not None and not cv_topo.empty:
        lines.extend(
            [
                "Leave-one-Sycamore-topology-out (IBM always in train):",
                f"  mean RMSE log N: {cv_topo['rmse_logN'].mean():.4f} ± {cv_topo['rmse_logN'].std():.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "Caveat: regimes differ across platforms; this is still a phenomenological fit.",
            "",
            "Outputs: threshold_transfer_predictions.csv, threshold_transfer_cv_folds.csv,",
            "threshold_transfer_cv_baseline_ols.csv, threshold_transfer_cv_sycamore_loo_topology.csv,",
            "threshold_transfer_cv_ibm_only.csv, threshold_transfer_cv_sycamore_only.csv,",
            "threshold_transfer_uncertainty.txt, threshold_transfer_model_report.txt",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    df_pred: pd.DataFrame,
    factors_imp: dict[str, float],
    cv_imp: pd.DataFrame,
    cv_base: pd.DataFrame,
    cv_topo: pd.DataFrame | None,
    cv_ibm: pd.DataFrame | None,
    cv_syc: pd.DataFrame | None,
    plot_path: Path,
    pred_csv: Path,
    legacy: bool,
    fnames: list[str],
    ridge: RidgeCV,
    extra_footer: str,
) -> None:
    report = RESULTS / "threshold_transfer_model_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("=== Threshold Transfer Model ===\n\n")
        if legacy:
            f.write("Sycamore: legacy 3-point bucket.\n\n")
        else:
            f.write(
                "Sycamore: 28 Fisher N*; topologies 1D/2D/Bulk. IBM: fine circuits (ghz/hadamard/layers/identity) "
                "unless coarse mode.\n\n"
            )

        f.write("--- Improved: Ridge + winsor + stratified CV; target log(N*sqrt(Q)); no logQ term ---\n")
        f.write(f"Ridge alpha (full data): {ridge.alpha_:.6f}\n")
        f.write(f"Intercept: {ridge.intercept_:.6f}\n")
        for j, name in enumerate(fnames):
            f.write(f"  {name}: {ridge.coef_[j]:.6f}\n")
        f.write(f"\nFull fit vs raw N_star (log space): RMSE={factors_imp['RMSE_log_full']:.4f}, R2={factors_imp['R2_log_full']:.4f}\n")

        f.write("\nImproved CV (RMSE on log raw N):\n")
        f.write(cv_imp.to_string(index=False))
        f.write("\n\nBaseline OLS CV (same folds; RMSE on log N OLS target):\n")
        f.write(cv_base.to_string(index=False))
        if cv_topo is not None and not cv_topo.empty:
            f.write("\n\nLeave-one-Sycamore-topology-out (train IBM + other two topologies; test held-out):\n")
            f.write(cv_topo.to_string(index=False))
            f.write(
                f"\n  mean RMSE log N: {cv_topo['rmse_logN'].mean():.4f} ± {cv_topo['rmse_logN'].std():.4f}\n"
            )
        if cv_ibm is not None and not cv_ibm.empty:
            f.write("\n\nWithin-source CV — IBM only (Ridge, same target):\n")
            f.write(cv_ibm.to_string(index=False))
            f.write(f"\n  mean RMSE log N: {cv_ibm['rmse_logN'].mean():.4f} ± {cv_ibm['rmse_logN'].std():.4f}\n")
        if cv_syc is not None and not cv_syc.empty:
            f.write("\n\nWithin-source CV — Sycamore only:\n")
            f.write(cv_syc.to_string(index=False))
            f.write(f"\n  mean RMSE log N: {cv_syc['rmse_logN'].mean():.4f} ± {cv_syc['rmse_logN'].std():.4f}\n")
        if extra_footer.strip():
            f.write("\n\n")
            f.write(extra_footer)
        f.write("\n\nFiles:\n")
        f.write(f"  predictions: {pred_csv}\n")
        f.write(f"  cv improved: {RESULTS / 'threshold_transfer_cv_folds.csv'}\n")
        f.write(f"  cv baseline OLS: {RESULTS / 'threshold_transfer_cv_baseline_ols.csv'}\n")
        if cv_topo is not None:
            f.write(f"  cv Sycamore LOO topology: {RESULTS / 'threshold_transfer_cv_sycamore_loo_topology.csv'}\n")
        f.write(f"  cv IBM-only / Syc-only: {RESULTS / 'threshold_transfer_cv_ibm_only.csv'}, ")
        f.write(f"{RESULTS / 'threshold_transfer_cv_sycamore_only.csv'}\n")
        f.write(f"  uncertainty: {RESULTS / 'threshold_transfer_uncertainty.txt'}\n")
        f.write(f"  one-pager: {RESULTS / 'threshold_transfer_onepager.txt'}\n")
        f.write(f"  figure: {plot_path}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Transfer model for Fisher N* (IBM + Sycamore)")
    ap.add_argument("--legacy", action="store_true", help="3 Sycamore points + old 7-column OLS design.")
    ap.add_argument("--sycamore-csv", type=Path, default=SYC_THRESHOLD_CSV)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=42)
    ap.add_argument("--winsor-low", type=float, default=0.05)
    ap.add_argument("--winsor-high", type=float, default=0.95)
    ap.add_argument(
        "--coarse-ibm-regimes",
        action="store_true",
        help="Merge hadamard+layers into one IBM regime (old behaviour). Default: separate circuits.",
    )
    ap.add_argument(
        "--no-scaled-target",
        action="store_true",
        help="Use log(N) target instead of log(N*sqrt(Q)); add logQ feature.",
    )
    ap.add_argument("--perm-reps", type=int, default=199, help="Permutation test reps for R^2 (0=skip).")
    ap.add_argument("--bootstrap-reps", type=int, default=199, help="Bootstrap reps for R^2 CI (0=skip).")
    ap.add_argument("--uncertainty-seed", type=int, default=43)
    args = ap.parse_args()

    coarse_ibm = args.legacy or args.coarse_ibm_regimes
    granular_ibm = not coarse_ibm and not args.legacy
    ibm = load_ibm_rows(coarse_regimes=coarse_ibm)
    syc = load_legacy_sycamore_three() if args.legacy else load_sycamore_rows(args.sycamore_csv)
    df = pd.concat([ibm, syc], ignore_index=True)

    scaled_target = not args.no_scaled_target
    include_log_q = not scaled_target

    folds = make_folds(df, n_splits=args.cv_folds, seed=args.cv_seed, legacy=args.legacy)
    cv_base = cv_baseline_ols(df, folds, legacy=args.legacy, granular_ibm=granular_ibm)
    cv_imp = cv_improved(
        df,
        folds,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
        scaled_target=scaled_target,
        include_log_q=include_log_q,
        granular_ibm=granular_ibm,
    )

    cv_topo: pd.DataFrame | None = None
    cv_ibm_only: pd.DataFrame | None = None
    cv_syc_only: pd.DataFrame | None = None
    if not args.legacy:
        cv_topo = cv_leave_one_sycamore_topology(
            df,
            winsor_low=args.winsor_low,
            winsor_high=args.winsor_high,
            scaled_target=scaled_target,
            include_log_q=include_log_q,
            granular_ibm=granular_ibm,
        )
        cv_topo.to_csv(RESULTS / "threshold_transfer_cv_sycamore_loo_topology.csv", index=False)
        cv_ibm_only = cv_within_source(
            df,
            "IBM",
            args.cv_folds,
            args.cv_seed + 1,
            args.winsor_low,
            args.winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        cv_syc_only = cv_within_source(
            df,
            "Sycamore",
            args.cv_folds,
            args.cv_seed + 2,
            args.winsor_low,
            args.winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        if not cv_ibm_only.empty:
            cv_ibm_only.to_csv(RESULTS / "threshold_transfer_cv_ibm_only.csv", index=False)
        if not cv_syc_only.empty:
            cv_syc_only.to_csv(RESULTS / "threshold_transfer_cv_sycamore_only.csv", index=False)

    _, factors_baseline, _, _ = fit_full_baseline(df, legacy=args.legacy, granular_ibm=granular_ibm)

    df_pred, factors_imp, ridge, fnames = fit_full_improved(
        df,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
        scaled_target=scaled_target,
        include_log_q=include_log_q,
        granular_ibm=granular_ibm,
    )
    by_src = rmse_by_source(df_pred)
    factors_imp.update(by_src)

    unc_lines: list[str] = []
    if args.perm_reps > 0 and not args.legacy:
        r2_obs, p_perm = permutation_test_r2(
            df,
            args.perm_reps,
            args.uncertainty_seed,
            args.winsor_low,
            args.winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        unc_lines.append(f"Permutation test (shuffle N*): R2_obs={r2_obs:.4f}, p≈{p_perm:.4f} ({args.perm_reps} reps)")
    if args.bootstrap_reps > 0 and not args.legacy:
        lo, hi, mean_r2 = bootstrap_r2_ci(
            df,
            args.bootstrap_reps,
            args.uncertainty_seed + 100,
            args.winsor_low,
            args.winsor_high,
            scaled_target,
            include_log_q,
            granular_ibm,
        )
        unc_lines.append(
            f"Bootstrap R2 (row resample): 95% CI [{lo:.4f}, {hi:.4f}], mean={mean_r2:.4f} ({args.bootstrap_reps} reps)"
        )
    extra_footer = ""
    if unc_lines:
        extra_footer = "=== Uncertainty (improved Ridge model) ===\n" + "\n".join(unc_lines) + "\n"
        (RESULTS / "threshold_transfer_uncertainty.txt").write_text(extra_footer, encoding="utf-8")

    pred_csv = RESULTS / "threshold_transfer_predictions.csv"
    df_pred.to_csv(pred_csv, index=False)
    cv_imp.to_csv(RESULTS / "threshold_transfer_cv_folds.csv", index=False)
    cv_base.to_csv(RESULTS / "threshold_transfer_cv_baseline_ols.csv", index=False)

    plot_path = RESULTS / "threshold_transfer_pred_vs_obs.png"
    title = "Ridge + winsor (legacy)" if args.legacy else "Ridge + winsor + N√Q (28 Syc + IBM, fine IBM circuits)"
    save_plot(df_pred, plot_path, title=title)

    write_report(
        df_pred,
        factors_imp,
        cv_imp,
        cv_base,
        cv_topo,
        cv_ibm_only,
        cv_syc_only,
        plot_path,
        pred_csv,
        args.legacy,
        fnames,
        ridge,
        extra_footer,
    )
    write_onepager(
        df_pred,
        factors_imp,
        factors_baseline,
        cv_imp,
        cv_base,
        cv_topo,
        by_src,
        RESULTS / "threshold_transfer_onepager.txt",
        n_ibm=len(ibm),
        n_syc=len(syc),
        legacy=args.legacy,
    )

    print(RESULTS / "threshold_transfer_model_report.txt")
    print(f"CV improved RMSE log N: {cv_imp['rmse_logN'].mean():.4f} ± {cv_imp['rmse_logN'].std():.4f}")
    print(f"CV baseline OLS:        {cv_base['rmse_logN'].mean():.4f} ± {cv_base['rmse_logN'].std():.4f}")
    if cv_topo is not None and not cv_topo.empty:
        print(
            f"CV Syc LOO topology:    {cv_topo['rmse_logN'].mean():.4f} ± {cv_topo['rmse_logN'].std():.4f}"
        )
    if cv_ibm_only is not None and not cv_ibm_only.empty:
        print(
            f"CV IBM-only:            {cv_ibm_only['rmse_logN'].mean():.4f} ± {cv_ibm_only['rmse_logN'].std():.4f}"
        )
    if cv_syc_only is not None and not cv_syc_only.empty:
        print(
            f"CV Syc-only:            {cv_syc_only['rmse_logN'].mean():.4f} ± {cv_syc_only['rmse_logN'].std():.4f}"
        )


if __name__ == "__main__":
    main()
