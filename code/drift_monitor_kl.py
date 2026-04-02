"""
KL-based drift monitor (grammar transition matrix vs reference).

Uses the same KL definition as ``fisher_information_analysis.compute_kl_divergence``:
row-averaged KL( T_ref_row || T_cur_row ) style (see implementation).

Typical use:
  1) Build a golden fingerprint once (e.g. ``build_sycamore_readout_fingerprints.py``).
  2) Point this script at the saved matrix in the .npz and at fresh readout data.
  3) If KL exceeds ``--threshold``, treat as calibration / noise drift (heuristic).

``--window-size`` splits the *current* readout into consecutive windows; each window
re-trains a small model and emits one KL value — useful to simulate rolling monitoring
(batch jobs), not sub-second latency.

Real-time deployment would wrap the same ``train_model`` + ``extract_grammar`` + KL
step on buffers fed by your acquisition stack; this script is an offline / batch prototype.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fisher_information_analysis import compute_kl_divergence
from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def load_reference_matrix(npz_path: str, key: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=False)
    if key not in data.files:
        raise KeyError(f"Key {key!r} not in {npz_path}. Available: {list(data.files)}")
    return np.asarray(data[key], dtype=np.float64)


def train_transition_matrix(
    signal: np.ndarray,
    label: str,
    *,
    alphabet_size: int,
    hidden_dim: int,
    seq_len: int,
    epochs: int,
    lr: float,
    seed: int,
) -> np.ndarray | None:
    val_loss, _ppl, model, val_data = train_model(
        signal,
        label,
        alphabet_size=alphabet_size,
        data_is_array=True,
        seed=seed,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        epochs=epochs,
        lr=lr,
    )
    if model is None:
        return None
    return extract_grammar(model, val_data, seq_len=seq_len)


def main() -> None:
    p = argparse.ArgumentParser(description="KL drift vs reference grammar matrix")
    p.add_argument(
        "--reference-npz",
        default=os.path.join(RESULTS_DIR, "sycamore_readout_grammar_fingerprints.npz"),
        help="NPZ containing transition matrices (e.g. from build_sycamore_readout_fingerprints).",
    )
    p.add_argument(
        "--reference-key",
        required=True,
        help="Array key in NPZ, e.g. sycamore_12q",
    )
    p.add_argument(
        "--current-readout",
        required=True,
        help="Path to *_readout_raw_data.txt (or compatible CSV with 'output' column).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="If >0, split current readout into consecutive windows of this many samples (each re-trains). 0 = use full file once.",
    )
    p.add_argument(
        "--window-stride",
        type=int,
        default=0,
        help="If >0 with --window-size, step by this many samples (default: window-size, no overlap).",
    )
    p.add_argument("--threshold", type=float, default=None, help="Exit 1 if any KL > threshold.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphabet-size", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--out-csv",
        default=None,
        help="Write per-window KL (default: results/drift_kl_monitor.csv if windowed).",
    )
    args = p.parse_args()

    T_ref = load_reference_matrix(args.reference_npz, args.reference_key)

    df = pd.read_csv(args.current_readout, sep=" ", on_bad_lines="skip")
    raw = df["output"].astype(np.float64).values
    n = len(raw)
    if n < args.seq_len * 10:
        print(f"Too few samples: {n}", file=sys.stderr)
        sys.exit(2)

    params = dict(
        alphabet_size=args.alphabet_size,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    rows: list[dict] = []
    ws = args.window_size
    stride = args.window_stride if args.window_stride > 0 else ws

    if ws <= 0:
        T_cur = train_transition_matrix(raw, "drift_current", **params)
        if T_cur is None:
            print("Training failed for current readout.", file=sys.stderr)
            sys.exit(3)
        kl = compute_kl_divergence(T_ref, T_cur)
        print(f"KL(reference || current) = {kl:.6f}")
        rows.append({"window_index": 0, "start": 0, "end": n, "kl": kl})
        max_kl = kl
    else:
        max_kl = 0.0
        wi = 0
        start = 0
        while start + ws <= n:
            chunk = raw[start : start + ws]
            T_cur = train_transition_matrix(chunk, f"drift_w{wi}", **params)
            if T_cur is None:
                print(f"  window {wi} [{start}:{start+ws}] FAILED", flush=True)
                rows.append({"window_index": wi, "start": start, "end": start + ws, "kl": float("nan")})
            else:
                kl = compute_kl_divergence(T_ref, T_cur)
                max_kl = max(max_kl, kl)
                print(f"  window {wi} [{start}:{start+ws}] KL = {kl:.6f}", flush=True)
                rows.append({"window_index": wi, "start": start, "end": start + ws, "kl": kl})
            wi += 1
            start += stride
        if not rows:
            print("No full windows fit in signal.", file=sys.stderr)
            sys.exit(4)
        print(f"max KL = {max_kl:.6f}")

    out_csv = args.out_csv or (os.path.join(RESULTS_DIR, "drift_kl_monitor.csv") if ws > 0 else None)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")

    if args.threshold is not None and max_kl > args.threshold:
        print(f"ALERT: KL {max_kl:.6f} > threshold {args.threshold}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
