"""
SAX → LSTM → transition matrix → Fisher trace vs subsample length N (entropy time series).

Reads CSVs from results/llm_entropy/runs/ (step, entropy) produced by collect_llm_entropy_series.py.

Examples:
  python code/run_llm_entropy_grammar_fisher.py --smoke
  python code/run_llm_entropy_grammar_fisher.py --grammar-seed 0 --glob "results/llm_entropy/runs/*_none.csv"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from fisher_information_analysis import (  # noqa: E402
    compute_fisher_matrix,
    compute_kl_divergence,
    fisher_scalar,
)
from grammar_learner import extract_grammar, train_model  # noqa: E402


def _default_n_grid(max_n: int, smoke: bool) -> list[int]:
    if smoke:
        pts = [32, 40, max_n]
    else:
        pts = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
        # Dense sweep for long series (e.g. L=2048): step 256 above 512, then max_n.
        n = 768
        while n < max_n:
            pts.append(n)
            n += 256
    out = sorted({p for p in pts if p <= max_n and p >= 24})
    if not out and max_n >= 24:
        out = [max_n]
    elif max_n not in out and max_n >= 32:
        out.append(max_n)
    return sorted(set(out))


def _effective_seq_len(n_pts: int, requested: int) -> int:
    """Keep enough points in the 20% val split for at least one strided batch."""
    # val_len ≈ 0.2 * n_pts; need val_len >= seq_len + 2
    max_seq = max(4, int(0.2 * n_pts) - 3)
    if max_seq < 4:
        max_seq = 4
    return max(4, min(requested, max_seq))


def _parse_run_id(path: str) -> tuple[str, str, str]:
    """Return (regime, control, seed) from filename .../factual_p0_seed42_none.csv"""
    base = os.path.basename(path)
    m = re.match(r"^([a-z_]+)_p(\d+)_seed(\d+)_(.+)\.csv$", base)
    if not m:
        return ("unknown", "unknown", "unknown")
    regime, _, seed, control = m.group(1), m.group(2), m.group(3), m.group(4)
    return regime, control, seed


def _load_entropy_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "entropy" not in df.columns:
        raise ValueError(f"Expected 'entropy' column in {path}")
    return df["entropy"].values.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grammar + Fisher on LLM entropy series")
    _default_glob = os.path.join(REPO_ROOT, "results", "llm_entropy", "runs", "*_none.csv")
    ap.add_argument(
        "--glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help=f"Glob for CSV files; repeat for multiple patterns. Default: {_default_glob}",
    )
    ap.add_argument("--grammar-seed", type=int, default=42, help="LSTM / numpy seed")
    ap.add_argument("--alphabet-size", type=int, default=7)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--smoke", action="store_true", help="Fewer N grid points and epochs=15")
    ap.add_argument("--include-controls", action="store_true", help="Also match *_shuffled.csv and *_random_uniform.csv")
    ap.add_argument(
        "--summary-csv",
        default="llm_entropy_fisher_summary.csv",
        metavar="NAME",
        help="Summary filename inside fisher_runs/ (default: llm_entropy_fisher_summary.csv).",
    )
    args = ap.parse_args()

    params = {
        "hidden_dim": args.hidden_dim,
        "seq_len": args.seq_len,
        "epochs": 15 if args.smoke else args.epochs,
        "lr": args.lr,
    }

    patterns = args.glob if args.glob else [_default_glob]
    files = sorted(
        {os.path.normpath(p) for pattern in patterns for p in glob.glob(pattern)},
        key=lambda x: os.path.basename(x),
    )
    # Default: only *_none.csv. If globs already match shuffled / random_uniform, keep them
    # without requiring --include-controls (avoids silently dropping shuffled-only globs).
    if not args.include_controls:
        has_control_csv = any(
            re.search(r"_(shuffled|random_uniform)\.csv$", os.path.basename(f)) for f in files
        )
        if not has_control_csv:
            files = [f for f in files if re.search(r"_(none)\.csv$", os.path.basename(f))]

    if not files:
        print(f"No files matched: {patterns}")
        print("Run: python code/collect_llm_entropy_series.py --quick --smoke")
        return

    out_dir = os.path.join(REPO_ROOT, "results", "llm_entropy", "fisher_runs")
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for path in files:
        full = np.asarray(_load_entropy_csv(path), dtype=np.float64)
        max_n = int(len(full))
        regime, control, seed_run = _parse_run_id(path)
        label = os.path.basename(path).replace(".csv", "")

        n_grid = _default_n_grid(max_n, args.smoke)
        print(f"\n{'='*60}\n{label} | len={max_n} | N grid={n_grid}\n{'='*60}", flush=True)

        rows = []
        prev_T = None
        for n_pts in n_grid:
            if n_pts > max_n:
                continue
            signal = full[:n_pts]
            seq_len = _effective_seq_len(n_pts, args.seq_len)
            min_n = max(32, seq_len * 5)
            if n_pts < min_n:
                print(f"  N={n_pts}: skip (need n >= {min_n} for seq_len={seq_len})")
                continue

            tag = f"{label}_N{n_pts}"
            run_params = {**params, "seq_len": seq_len}
            print(f"  N={n_pts} (seq_len={seq_len}) ...", end=" ", flush=True)
            _, _, model, val_data = train_model(
                signal,
                tag,
                alphabet_size=args.alphabet_size,
                data_is_array=True,
                seed=args.grammar_seed,
                **run_params,
            )
            if model is None or val_data is None or len(val_data) < seq_len + 2:
                print("FAIL")
                continue
            T = extract_grammar(model, val_data, seq_len=seq_len)
            F = compute_fisher_matrix(T)
            tr = fisher_scalar(F)
            kl = 0.0
            if prev_T is not None:
                kl = compute_kl_divergence(prev_T, T)
            prev_T = T
            rows.append(
                {
                    "file": label,
                    "regime": regime,
                    "control": control,
                    "n_points": n_pts,
                    "fisher_trace": tr,
                    "kl_vs_prev": kl,
                }
            )
            print(f"Tr(F)={tr:.4f}", flush=True)

        if rows:
            out_csv = os.path.join(out_dir, f"fisher_metric_{label}.csv")
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            # N* heuristic: max Fisher trace on grid (same as publication-style threshold)
            dfm = pd.DataFrame(rows)
            j = int(dfm["fisher_trace"].values.argmax())
            n_star = float(dfm["n_points"].values[j])
            summary_rows.append(
                {
                    "source_csv": os.path.relpath(path, REPO_ROOT),
                    "regime": regime,
                    "control": control,
                    "max_len": max_n,
                    "n_star_trace_max": n_star,
                    "fisher_at_n_star": float(dfm["fisher_trace"].values[j]),
                }
            )
            with open(os.path.join(out_dir, f"fisher_metric_{label}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "grammar_seed": args.grammar_seed,
                        "alphabet_size": args.alphabet_size,
                        "params": params,
                        "n_grid_used": n_grid,
                    },
                    f,
                    indent=2,
                )

    if summary_rows:
        s_path = os.path.join(out_dir, os.path.basename(args.summary_csv))
        pd.DataFrame(summary_rows).to_csv(s_path, index=False)
        print(f"\nSummary: {s_path}")


if __name__ == "__main__":
    main()
