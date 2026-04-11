"""
Build N-dependent geometric scalars from saved transition matrices T(N).

Reads ``transition_matrices_sycamore*.npz`` + matching ``transition_matrices_index*.csv``,
groups by readout ``file``, sorts by ``n_points``, and for each grid point writes:

  - ricci_ambient: scalar Ricci of the **product** Fisher manifold (independent rows).
    On each multinomial simplex factor the Fisher–Rao Ricci is **almost constant** in
    the interior for fixed dimension, so this column is nearly flat in N.

  - log_det_fisher: log det G(θ) — varies with T(N).

  - fisher_speed_dn: ‖Δθ‖_{Ḡ}/|ΔN| between consecutive N (first point NaN).

If no ``.npz`` exists yet, use ``--demo-synthetic`` to build a **placeholder** T(N)
curve from the Fisher CSV grid (same files / N) for pipeline testing — column
``data_source`` marks ``measured`` vs ``synthetic_demo``.

Usage:
  python code/compute_fisher_ricci_curve.py --skip-ricci
  python code/compute_fisher_ricci_curve.py --demo-synthetic --skip-ricci
  python code/compute_fisher_ricci_curve.py --npz results/transition_matrices_sycamore_all_readouts.npz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fisher_riemann_simplex import (  # noqa: E402
    fisher_speed_wrt_N,
    log_det_fisher_block,
    ricci_scalar_transition_matrix,
)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def _file_seed(fname: str) -> int:
    return int.from_bytes(hashlib.md5(fname.encode("utf-8")).digest()[:4], "little")


def synthetic_transition_matrix(n: float, fname: str) -> np.ndarray:
    """
    Deterministic row-stochastic T(N): two regimes mixed by a logistic in N centered ~8k.
    Not measured data — only for empty-npz workflow / plotting pipeline tests.
    """
    rng = np.random.default_rng(_file_seed(fname) & 0xFFFFFFFF)
    early = rng.dirichlet(np.ones(7), size=7)
    late = rng.dirichlet(np.ones(7), size=7)
    # logistic transition on linear N scale (grid spans 500–40k)
    w = 1.0 / (1.0 + np.exp(-(float(n) - 8200.0) / 1400.0))
    T = (1.0 - w) * early + w * late
    T = np.maximum(T, 1e-9)
    T /= T.sum(axis=1, keepdims=True)
    return T.astype(np.float64)


def build_demo_stack_from_fisher_csv(path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return T_all (K,7,7) and meta with row_index 0..K-1 from fisher_metric CSV."""
    df = pd.read_csv(path)
    need = {"file", "n_points", "q_label", "topology"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} must contain columns {sorted(need)}")
    df = df[list(need)].copy()
    df = df.sort_values(["file", "n_points"], kind="mergesort")
    K = len(df)
    T_all = np.zeros((K, 7, 7), dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        T_all[i] = synthetic_transition_matrix(float(row.n_points), str(row.file))
    meta = pd.DataFrame(
        {
            "row_index": np.arange(K, dtype=int),
            "file": df["file"].values,
            "q_label": df["q_label"].values,
            "topology": df["topology"].values,
            "n_points": df["n_points"].values,
        }
    )
    return T_all, meta


def compute_rows(
    meta: pd.DataFrame,
    T_all: np.ndarray,
    *,
    skip_ricci: bool,
    per_row_ricci: bool,
    data_source: str,
) -> pd.DataFrame:
    rows_out: list[dict] = []
    for fname, g in meta.groupby("file", sort=False):
        sub = g.sort_values("n_points")
        idxs = sub["row_index"].values.astype(int)
        npts = sub["n_points"].values.astype(float)
        T_sub = T_all[idxs]

        prev_T = None
        prev_n = None
        for i in range(len(sub)):
            T = T_sub[i]
            n = float(npts[i])
            row: dict = {
                "data_source": data_source,
                "file": fname,
                "q_label": sub["q_label"].iloc[i],
                "topology": sub["topology"].iloc[i],
                "n_points": int(n) if n == int(n) else n,
            }
            if not skip_ricci:
                rtot, rpr = ricci_scalar_transition_matrix(T)
                row["ricci_ambient"] = rtot
                if per_row_ricci:
                    for j, rv in enumerate(rpr):
                        row[f"ricci_row_{j}"] = float(rv)
            else:
                row["ricci_ambient"] = float("nan")

            row["log_det_fisher"] = log_det_fisher_block(T)

            if prev_T is not None and prev_n is not None:
                row["fisher_speed_dn"] = fisher_speed_wrt_N(prev_T, T, prev_n, n)
            else:
                row["fisher_speed_dn"] = float("nan")

            rows_out.append(row)
            prev_T, prev_n = T, n

    return pd.DataFrame(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ricci + Fisher geometry along T(N) curves")
    ap.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="Ignore npz: build deterministic synthetic T(N) from --fisher-csv grid (testing only).",
    )
    ap.add_argument(
        "--fisher-csv",
        default=os.path.join(RESULTS_DIR, "fisher_metric_vs_datalength_all_readouts.csv"),
        help="Grid for --demo-synthetic (and default measured outputs live here too).",
    )
    ap.add_argument(
        "--npz",
        default=os.path.join(RESULTS_DIR, "transition_matrices_sycamore_all_readouts.npz"),
        help="Stacked T matrices from fisher_information_analysis",
    )
    ap.add_argument(
        "--index-csv",
        default=None,
        help="Index CSV (default: paired name next to npz)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(RESULTS_DIR, "fisher_ricci_and_embedding_vs_n.csv"),
        help="Output long-form CSV",
    )
    ap.add_argument(
        "--skip-ricci",
        action="store_true",
        help="Skip expensive per-matrix Ricci (only log-det and speeds).",
    )
    ap.add_argument(
        "--per-row-ricci",
        action="store_true",
        help="Include ricci_row_0..ricci_row_6 columns (ambient Ricci is usually enough).",
    )
    args = ap.parse_args()

    if args.demo_synthetic:
        if not os.path.isfile(args.fisher_csv):
            raise SystemExit(f"Missing {args.fisher_csv} (needed for --demo-synthetic)")
        T_all, meta = build_demo_stack_from_fisher_csv(args.fisher_csv)
        data_source = "synthetic_demo"
        print(
            "DEMO mode: synthetic T(N); replace with measured transition_matrices_sycamore*.npz when available."
        )
    else:
        if not os.path.isfile(args.npz):
            raise SystemExit(
                f"Missing {args.npz}\n"
                "  Run: python code/fisher_information_analysis.py\n"
                "  Or use: python code/compute_fisher_ricci_curve.py --demo-synthetic --skip-ricci"
            )
        index_csv = args.index_csv
        if not index_csv:
            bn = os.path.basename(args.npz)
            d = os.path.dirname(os.path.abspath(args.npz))
            if bn.startswith("transition_matrices_sycamore") and bn.endswith(".npz"):
                suffix = bn[len("transition_matrices_sycamore") : -len(".npz")]
                index_csv = os.path.join(d, f"transition_matrices_index{suffix}.csv")
            else:
                base, _ = os.path.splitext(args.npz)
                index_csv = base + "_index.csv"

        z = np.load(args.npz)
        T_all = z["T"]
        meta = pd.read_csv(index_csv)
        if len(meta) != len(T_all):
            raise SystemExit(f"Index rows ({len(meta)}) != T stack ({len(T_all)})")
        data_source = "measured"

    out_df = compute_rows(
        meta,
        T_all,
        skip_ricci=args.skip_ricci,
        per_row_ricci=args.per_row_ricci,
        data_source=data_source,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
