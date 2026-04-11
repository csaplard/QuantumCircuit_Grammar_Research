"""
N*/N_curv as a complexity metric: aggregate curvature analysis across substrates.

- Sycamore: enrich fisher_curvature_sycamore_summary.csv with ratio; group by topology.
- LLM: all fisher_metric_*_p0_seed42_*.csv curves (4 regimes × 3 controls).
- IBM: fisher sweep CSVs (Marrakesh, Torino) — one curve per shot file.

N* = argmax_N Tr(F), N_curv = argmax_N |d^2 Tr/dN^2| (natural cubic spline on Tr(N)).

Outputs under results/:
  - fisher_curvature_nstar_ratio_all_curves.csv
  - fisher_curvature_nstar_ratio_sycamore_by_topology.csv
  - fisher_curvature_nstar_ratio_llm_seed42_by_regime.csv (none-only summary)
  - fisher_curvature_nstar_ratio_ibm_by_backend.csv
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
RESULTS = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, SCRIPT_DIR)

from curvature_fisher_trace_sycamore import (  # noqa: E402
    _prepare_curve,
    n_star_trace_max,
    second_derivative_spline,
)


def metrics_for_curve(n: np.ndarray, y: np.ndarray) -> dict:
    n, y = _prepare_curve(np.asarray(n, dtype=float), np.asarray(y, dtype=float))
    d2 = second_derivative_spline(n, y)
    j_abs = int(np.nanargmax(np.abs(d2)))
    n_curv = float(n[j_abs])
    n_star = n_star_trace_max(n, y)
    max_abs_d2 = float(np.abs(d2[j_abs]))
    ratio = (n_star / n_curv) if n_curv > 0 else float("nan")
    return {
        "n_star_trace_max": n_star,
        "n_curv_abs_d2_peak": n_curv,
        "max_abs_d2_tr_dn2": max_abs_d2,
        "n_star_over_n_curv": ratio,
        "reliable_curvature": max_abs_d2 >= 1.0,
    }


def process_long_csv(df: pd.DataFrame, group_cols: list[str], source: str) -> list[dict]:
    rows: list[dict] = []
    for key, g in df.groupby(group_cols, sort=False):
        n = g["n_points"].values
        y = g["fisher_trace"].values
        try:
            m = metrics_for_curve(n, y)
        except ValueError:
            continue
        row = {"source": source, **m}
        if isinstance(key, tuple):
            for c, v in zip(group_cols, key):
                row[c] = v
        else:
            row[group_cols[0]] = key
        if "file" in row:
            row["substrate_detail"] = row["file"]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="N*/N_curv ratio across Sycamore, LLM, IBM")
    ap.add_argument(
        "--out-dir",
        default=RESULTS,
        help="Directory for output CSVs",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_rows: list[dict] = []

    # --- Sycamore (28 readouts) ---
    syc_path = os.path.join(out_dir, "fisher_curvature_sycamore_summary.csv")
    if os.path.isfile(syc_path):
        sdf = pd.read_csv(syc_path)
        for _, r in sdf.iterrows():
            ns = float(r["n_star_trace_max"])
            nc = float(r["n_peak_abs_d2"])
            mad = float(r["max_abs_d2_tr_dn2"])
            ratio = (ns / nc) if nc > 0 else np.nan
            all_rows.append(
                {
                    "source": "sycamore",
                    "substrate_detail": r.get("file", ""),
                    "topology": r.get("topology", ""),
                    "q_label": r.get("q_label", ""),
                    "regime": "",
                    "control": "",
                    "llm_seed": "",
                    "backend": "",
                    "n_star_trace_max": ns,
                    "n_curv_abs_d2_peak": nc,
                    "max_abs_d2_tr_dn2": mad,
                    "n_star_over_n_curv": ratio,
                    "reliable_curvature": abs(mad) >= 1.0,
                }
            )
    else:
        print(f"Warning: missing {syc_path}, skip Sycamore", flush=True)

    # --- LLM seed 42: all regimes × controls ---
    llm_pat = os.path.join(REPO_ROOT, "results", "llm_entropy", "fisher_runs", "fisher_metric_*_p0_seed42_*.csv")
    for path in sorted(glob.glob(llm_pat)):
        base = os.path.basename(path)
        if "_meta" in base:
            continue
        df = pd.read_csv(path)
        if "n_points" not in df.columns:
            continue
        n = df["n_points"].values
        y = df["fisher_trace"].values
        try:
            m = metrics_for_curve(n, y)
        except ValueError:
            continue
        regime = df["regime"].iloc[0] if "regime" in df.columns else ""
        control = df["control"].iloc[0] if "control" in df.columns else ""
        all_rows.append(
            {
                "source": "llm",
                "substrate_detail": os.path.relpath(path, REPO_ROOT),
                "topology": "",
                "q_label": "",
                "regime": regime,
                "control": control,
                "llm_seed": 42,
                "backend": "",
                **m,
            }
        )

    # --- IBM Marrakesh + Torino ---
    for fname in ("ibm_fisher_sweep_marrakesh40960.csv", "ibm_fisher_sweep_torino40960.csv"):
        p = os.path.join(out_dir, fname)
        if not os.path.isfile(p):
            print(f"Warning: missing {p}, skip", flush=True)
            continue
        df = pd.read_csv(p)
        group_cols = ["file", "backend"] if "backend" in df.columns else ["file"]
        for r in process_long_csv(df, group_cols, "ibm"):
            r.setdefault("topology", "")
            r.setdefault("q_label", "")
            r.setdefault("regime", "")
            r.setdefault("control", "")
            r.setdefault("llm_seed", "")
            all_rows.append(r)

    full = pd.DataFrame(all_rows)
    full_path = os.path.join(out_dir, "fisher_curvature_nstar_ratio_all_curves.csv")
    full.to_csv(full_path, index=False)
    print(f"Wrote {full_path} ({len(full)} rows)")

    # --- Sycamore by topology ---
    syc = full[full["source"] == "sycamore"].copy()
    if len(syc):
        agg = []
        for topo, g in syc.groupby("topology"):
            ratios = g["n_star_over_n_curv"].replace([np.inf, -np.inf], np.nan).dropna()
            rel = g[g["reliable_curvature"]]
            rr = rel["n_star_over_n_curv"].replace([np.inf, -np.inf], np.nan).dropna()
            agg.append(
                {
                    "topology": topo,
                    "n_readouts": len(g),
                    "median_n_star_over_n_curv_all": float(ratios.median()) if len(ratios) else np.nan,
                    "mean_n_star_over_n_curv_all": float(ratios.mean()) if len(ratios) else np.nan,
                    "median_n_star_over_n_curv_reliable": float(rr.median()) if len(rr) else np.nan,
                    "mean_n_star_over_n_curv_reliable": float(rr.mean()) if len(rr) else np.nan,
                }
            )
        by_topo = pd.DataFrame(agg).sort_values("topology")
        p2 = os.path.join(out_dir, "fisher_curvature_nstar_ratio_sycamore_by_topology.csv")
        by_topo.to_csv(p2, index=False)
        print(f"Wrote {p2}")
        print(by_topo.to_string(index=False))

    # --- LLM seed 42 by regime (none only) ---
    llm = full[(full["source"] == "llm") & (full["control"] == "none")].copy()
    if len(llm):
        order = ["mathematical", "philosophical", "creative", "factual"]
        llm["_ord"] = llm["regime"].map({r: i for i, r in enumerate(order)})
        llm = llm.sort_values("_ord", na_position="last").drop(columns=["_ord"])
        p3 = os.path.join(out_dir, "fisher_curvature_nstar_ratio_llm_seed42_none_by_regime.csv")
        llm_out = llm[
            [
                "regime",
                "n_star_trace_max",
                "n_curv_abs_d2_peak",
                "n_star_over_n_curv",
                "max_abs_d2_tr_dn2",
                "reliable_curvature",
            ]
        ]
        llm_out.to_csv(p3, index=False)
        print(f"Wrote {p3}")
        print(llm_out.to_string(index=False))

    llm_all = full[full["source"] == "llm"].copy()
    if len(llm_all):
        ro = ["mathematical", "philosophical", "creative", "factual"]
        llm_all["_ord"] = llm_all["regime"].map({r: i for i, r in enumerate(ro)})
        ctrl_order = {"none": 0, "shuffled": 1, "random_uniform": 2}
        llm_all["_c"] = llm_all["control"].map(lambda x: ctrl_order.get(str(x), 9))
        llm_all = llm_all.sort_values(["_ord", "_c"]).drop(columns=["_ord", "_c"])
        p3b = os.path.join(out_dir, "fisher_curvature_nstar_ratio_llm_seed42_all_controls.csv")
        llm_all.to_csv(p3b, index=False)
        print(f"Wrote {p3b} ({len(llm_all)} rows)")

    # --- IBM by backend ---
    ibm = full[full["source"] == "ibm"].copy()
    if len(ibm):
        agg_i = []
        for bk, g in ibm.groupby("backend"):
            ratios = g["n_star_over_n_curv"].replace([np.inf, -np.inf], np.nan).dropna()
            agg_i.append(
                {
                    "backend": bk,
                    "n_curves": len(g),
                    "median_n_star_over_n_curv": float(ratios.median()) if len(ratios) else np.nan,
                    "mean_n_star_over_n_curv": float(ratios.mean()) if len(ratios) else np.nan,
                }
            )
        ibm_sum = pd.DataFrame(agg_i)
        p4 = os.path.join(out_dir, "fisher_curvature_nstar_ratio_ibm_by_backend.csv")
        ibm_sum.to_csv(p4, index=False)
        print(f"Wrote {p4}")
        print(ibm_sum.to_string(index=False))

        per_file_path = os.path.join(out_dir, "fisher_curvature_nstar_ratio_ibm_per_file.csv")
        ibm_out = ibm[
            ["backend", "file", "n_star_trace_max", "n_curv_abs_d2_peak", "n_star_over_n_curv", "max_abs_d2_tr_dn2", "reliable_curvature"]
        ].copy()
        ibm_out.to_csv(per_file_path, index=False)
        print(f"Wrote {per_file_path} ({len(ibm_out)} rows)")


if __name__ == "__main__":
    main()
