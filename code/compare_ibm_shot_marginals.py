"""
Compare two IBM shot runs using low-dimensional summaries of outcomes.

Assumes Qiskit-style LSB indexing: qubit q is bit (output >> q) & 1, masked to n qubits.

Per file pair:
  - Per-qubit P(bit=1): mean absolute delta and max absolute delta across qubits.
  - Hamming weight (popcount on masked outcome): TV and Jensen–Shannon on {0,…,n}.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")

Q_RE = re.compile(r"_(\d+)q_\d+shots\.txt$", re.IGNORECASE)


def n_qubits_from_name(name: str) -> int | None:
    m = Q_RE.search(name)
    return int(m.group(1)) if m else None


def _load_outputs(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", engine="python", on_bad_lines="skip")
    if "output" not in df.columns:
        raise ValueError(f"No 'output' column in {path}")
    return df["output"].astype(np.int64).values


def masked_outputs(raw: np.ndarray, n: int) -> np.ndarray:
    m = (1 << n) - 1 if n < 63 else np.iinfo(np.int64).max
    return raw.astype(np.int64) & m


def hamming_weights(v: np.ndarray, n: int) -> np.ndarray:
    v = masked_outputs(v, n).astype(np.uint64)
    return np.vectorize(int.bit_count, otypes=[np.int32])(v)


def marginal_p1(v: np.ndarray, n: int) -> np.ndarray:
    v = masked_outputs(v, n)
    out = np.zeros(n, dtype=np.float64)
    for q in range(n):
        out[q] = float(np.mean((v >> q) & 1))
    return out


def tv_js_hamming(wa: np.ndarray, wb: np.ndarray, n: int, eps: float) -> dict[str, float]:
    """Empirical PMF on {0..n}, TV and JS."""
    ca = pd.Series(wa).value_counts()
    cb = pd.Series(wb).value_counts()
    keys = np.arange(n + 1, dtype=np.int64)
    pa = np.array([ca.get(int(k), 0) for k in keys], dtype=np.float64)
    qa = np.array([cb.get(int(k), 0) for k in keys], dtype=np.float64)
    pa = pa + eps
    qa = qa + eps
    pa /= pa.sum()
    qa /= qa.sum()
    tv = 0.5 * float(np.sum(np.abs(pa - qa)))
    js = float(jensenshannon(pa, qa, base=np.e))
    return {"tv_hamming": tv, "js_hamming": js}


def main() -> None:
    p = argparse.ArgumentParser(description="Marginal & Hamming comparison for IBM shot pairs")
    p.add_argument(
        "--reference-dir",
        default=os.path.join(RESULTS, "ibm_raw_shots", "archive_ibm_torino_20260329_122743"),
    )
    p.add_argument("--current-dir", default=os.path.join(RESULTS, "ibm_raw_shots"))
    p.add_argument(
        "--filename-filter",
        default="",
        help="If set, only basenames containing this substring (e.g. 20q).",
    )
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument(
        "--out-csv",
        default=os.path.join(RESULTS, "ibm_torino_marginals_hamming_compare.csv"),
    )
    p.add_argument(
        "--out-txt",
        default=os.path.join(RESULTS, "ibm_torino_marginals_hamming_compare.txt"),
    )
    args = p.parse_args()

    ref_dir = Path(args.reference_dir)
    cur_dir = Path(args.current_dir)
    if not ref_dir.is_dir():
        print(f"Missing reference dir: {ref_dir}", file=sys.stderr)
        sys.exit(2)

    ref_files = sorted(
        f
        for f in ref_dir.glob("ibm_*_*shots.txt")
        if f.is_file() and "metadata" not in f.name.lower()
    )
    if args.filename_filter:
        ref_files = [f for f in ref_files if args.filename_filter in f.name]

    rows: list[dict] = []
    for rf in ref_files:
        n = n_qubits_from_name(rf.name)
        if n is None:
            continue
        cf = cur_dir / rf.name
        if not cf.is_file():
            continue

        a = _load_outputs(str(rf))
        b = _load_outputs(str(cf))
        pa = marginal_p1(a, n)
        pb = marginal_p1(b, n)
        d = np.abs(pa - pb)
        wh = tv_js_hamming(hamming_weights(a, n), hamming_weights(b, n), n, args.eps)

        rows.append(
            {
                "file": rf.name,
                "n_qubits": n,
                "mean_abs_marginal_dp": float(np.mean(d)),
                "max_abs_marginal_dp": float(np.max(d)),
                **wh,
            }
        )

    if not rows:
        print("No paired files.", file=sys.stderr)
        sys.exit(3)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    sub20 = out_df[out_df["n_qubits"] == 20]
    lines = [
        "IBM Torino: per-qubit marginal drift + Hamming-weight distance",
        f"reference_dir: {ref_dir.resolve()}",
        f"current_dir:   {cur_dir.resolve()}",
        "",
        "mean_abs_marginal_dp: average over qubits of |P_ref(bit=1) - P_cur(bit=1)|",
        "tv_hamming / js_hamming: TV and JS on Hamming weight (0..n).",
        "",
        out_df.to_string(index=False),
        "",
    ]
    if len(sub20):
        lines.extend(
            [
                "--- n_qubits == 20 only ---",
                sub20.to_string(index=False),
                "",
                f"20q mean TV Hamming: {sub20['tv_hamming'].mean():.6f}",
                f"20q mean mean_abs_marginal_dp: {sub20['mean_abs_marginal_dp'].mean():.6f}",
            ]
        )
    text = "\n".join(lines) + "\n"
    with open(args.out_txt, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_txt}")


if __name__ == "__main__":
    main()
