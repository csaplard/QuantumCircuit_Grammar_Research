"""
Compare two IBM raw-shot runs (same filenames): empirical outcome distributions.

Reads space-separated shot files with an `output` column (integers). Reports
total variation distance, Jensen–Shannon distance (scipy, base e), and
symmetric KL on the union outcome support with small additive smoothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results")


def _load_outputs(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", engine="python", on_bad_lines="skip")
    if "output" not in df.columns:
        raise ValueError(f"No 'output' column in {path}")
    return df["output"].astype(np.int64).values


def _aligned_pmfs(a: np.ndarray, b: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.union1d(np.unique(a), np.unique(b))
    ca = pd.Series(a).value_counts()
    cb = pd.Series(b).value_counts()
    pa = np.array([ca.get(int(k), 0) for k in keys], dtype=np.float64)
    qa = np.array([cb.get(int(k), 0) for k in keys], dtype=np.float64)
    pa = pa + eps
    qa = qa + eps
    pa /= pa.sum()
    qa /= qa.sum()
    return pa, qa


def metrics(pa: np.ndarray, qa: np.ndarray) -> dict[str, float]:
    tv = 0.5 * float(np.sum(np.abs(pa - qa)))
    js = float(jensenshannon(pa, qa, base=np.e))
    kl_pq = float(np.sum(pa * np.log(pa / qa)))
    kl_qp = float(np.sum(qa * np.log(qa / pa)))
    skl = 0.5 * (kl_pq + kl_qp)
    return {"tv": tv, "js_distance": js, "sym_kl": skl, "kl_pq": kl_pq, "kl_qp": kl_qp}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare IBM shot runs by outcome distribution")
    p.add_argument(
        "--reference-dir",
        default=os.path.join(RESULTS, "ibm_raw_shots", "archive_ibm_torino_20260329_122743"),
        help="Older run (directory with ibm_*_*shots.txt)",
    )
    p.add_argument(
        "--current-dir",
        default=os.path.join(RESULTS, "ibm_raw_shots"),
        help="Newer run (same basenames)",
    )
    p.add_argument("--eps", type=float, default=1e-12, help="Additive smoothing for KL alignment")
    p.add_argument(
        "--out-csv",
        default=os.path.join(RESULTS, "ibm_torino_run_compare_distributions.csv"),
    )
    p.add_argument(
        "--out-txt",
        default=os.path.join(RESULTS, "ibm_torino_run_compare_distributions.txt"),
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
    rows: list[dict] = []
    for rf in ref_files:
        cf = cur_dir / rf.name
        if not cf.is_file():
            print(f"Skip (no current file): {rf.name}", file=sys.stderr)
            continue
        old = _load_outputs(str(rf))
        new = _load_outputs(str(cf))
        if len(old) != len(new):
            print(
                f"Warning: length mismatch {rf.name}: ref={len(old)} cur={len(new)}",
                file=sys.stderr,
            )
        pa, qa = _aligned_pmfs(old, new, args.eps)
        m = metrics(pa, qa)
        uo = int(len(np.unique(old)))
        un = int(len(np.unique(new)))
        rows.append(
            {
                "file": rf.name,
                "n_shots_ref": len(old),
                "n_shots_cur": len(new),
                "n_unique_ref": uo,
                "n_unique_cur": un,
                **m,
            }
        )

    if not rows:
        print("No paired shot files found.", file=sys.stderr)
        sys.exit(3)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    lines = [
        "IBM Torino run comparison (empirical outcome distributions)",
        f"reference_dir: {ref_dir.resolve()}",
        f"current_dir:   {cur_dir.resolve()}",
        "",
        "tv = total variation; js_distance = sqrt(JS divergence), base e;",
        "sym_kl = mean of KL(ref||cur) and KL(cur||ref) on smoothed union support.",
        "",
        out_df.to_string(index=False),
        "",
        f"mean TV: {out_df['tv'].mean():.6f}",
        f"mean JS distance: {out_df['js_distance'].mean():.6f}",
        f"mean sym KL: {out_df['sym_kl'].mean():.6f}",
    ]
    text = "\n".join(lines) + "\n"
    with open(args.out_txt, "w", encoding="utf-8") as fh:
        fh.write(text)

    print(text)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_txt}")


if __name__ == "__main__":
    main()
