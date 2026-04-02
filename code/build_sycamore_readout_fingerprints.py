"""
Build Grammar Fingerprint .npz from Google Sycamore-style readout files.

Default input folder: repo-root ``readout_raw_data/`` (full 28-file set). If that
folder is missing or empty, falls back to ``results/readout_raw_data/``.

Default training matches the published Sycamore validation pipeline
(``run_validation_pipeline.py``): alphabet_size=7, hidden_dim=32, seq_len=16,
**epochs=50**, lr=0.01. IBM shot fingerprints from ``run_ibm_grammar_learning.py``
use epochs=20 by design (sufficient there); Sycamore typically needs more epochs
for sharp transition matrices before Frobenius clustering.

Use ``--epochs 20`` only for quick smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def default_readout_dir() -> str:
    """Prefer ``readout_raw_data`` at repo root (28 files); else ``results/readout_raw_data``."""
    root = os.path.join(REPO_ROOT, "readout_raw_data")
    nested = os.path.join(RESULTS_DIR, "readout_raw_data")
    pat_root = os.path.join(root, "*_readout_raw_data.txt")
    pat_nested = os.path.join(nested, "*_readout_raw_data.txt")
    n_root = len(glob.glob(pat_root)) if os.path.isdir(root) else 0
    n_nested = len(glob.glob(pat_nested)) if os.path.isdir(nested) else 0
    if n_root >= n_nested and n_root > 0:
        return root
    if n_nested > 0:
        return nested
    return root


def label_from_filename(path: str) -> str:
    """e.g. 12q_readout_raw_data.txt -> 12q"""
    base = os.path.basename(path)
    m = re.match(r"^(\d+)q_?readout", base, re.IGNORECASE)
    if m:
        return f"{m.group(1)}q"
    return base.replace(".txt", "")


def main() -> None:
    p = argparse.ArgumentParser(description="Sycamore readout -> grammar fingerprints (IBM-matched params)")
    p.add_argument(
        "--input-dir",
        default=None,
        help="Folder with *_readout_raw_data.txt (default: repo readout_raw_data/ if non-empty, else results/readout_raw_data/).",
    )
    p.add_argument(
        "--out-npz",
        default=os.path.join(RESULTS_DIR, "sycamore_readout_grammar_fingerprints.npz"),
    )
    p.add_argument(
        "--out-csv",
        default=os.path.join(RESULTS_DIR, "sycamore_readout_grammar_learning_results.csv"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-pts",
        type=int,
        default=40_960,
        help="Truncate each readout (default 40960 to align with common IBM shot files; use 100000 for full protocol).",
    )
    p.add_argument(
        "--only-qubits",
        type=str,
        default=None,
        help="Comma-separated qubit counts, e.g. '12,14,16' — keeps only Nq_readout_raw_data.txt for those N.",
    )
    p.add_argument("--epochs", type=int, default=50, help="LSTM epochs (default 50 = original Sycamore validation).")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()
    input_dir = args.input_dir if args.input_dir is not None else default_readout_dir()

    # Canonical naming only (avoids accidentally picking unrelated *readout*.txt files).
    pattern = os.path.join(input_dir, "*_readout_raw_data.txt")
    files = sorted(glob.glob(pattern))
    if args.only_qubits:
        allowed = {x.strip() for x in args.only_qubits.split(",") if x.strip()}
        filtered: list[str] = []
        for f in files:
            m = re.match(r"^(\d+)q_readout_raw_data\.txt$", os.path.basename(f), re.I)
            if m and m.group(1) in allowed:
                filtered.append(f)
        files = filtered
    if not files:
        print(f"No files matching {pattern}", file=sys.stderr)
        sys.exit(1)
    print(f"Input dir: {input_dir}  ({len(files)} x *_readout_raw_data.txt)", flush=True)
    print(
        f"LSTM: hidden_dim={args.hidden_dim} seq_len={args.seq_len} epochs={args.epochs} lr={args.lr} alphabet=7",
        flush=True,
    )

    params = {
        "hidden_dim": int(args.hidden_dim),
        "seq_len": int(args.seq_len),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
    }
    matrices: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for path in files:
        q_label = label_from_filename(path)
        key = f"sycamore_{q_label}"
        df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
        raw = df["output"].astype(np.float64).values
        if len(raw) > args.max_pts:
            raw = raw[: args.max_pts]

        val_loss, ppl, model, val_data = train_model(
            raw,
            key,
            alphabet_size=7,
            data_is_array=True,
            seed=args.seed,
            **params,
        )
        if model is None:
            continue
        fp = extract_grammar(model, val_data, seq_len=params["seq_len"])
        matrices[key] = fp
        rows.append(
            {
                "file": os.path.basename(path),
                "key": key,
                "n_samples": int(len(raw)),
                "hidden_dim": params["hidden_dim"],
                "seq_len": params["seq_len"],
                "epochs": params["epochs"],
                "lr": params["lr"],
                "val_loss": float(val_loss),
                "perplexity": float(ppl),
            }
        )

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, **matrices)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "key",
                "n_samples",
                "hidden_dim",
                "seq_len",
                "epochs",
                "lr",
                "val_loss",
                "perplexity",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(matrices)} fingerprints -> {args.out_npz}")
    print(f"Summary -> {args.out_csv}")


if __name__ == "__main__":
    main()
