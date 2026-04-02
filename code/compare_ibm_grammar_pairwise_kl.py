"""
Pairwise grammar fingerprint comparison: archive IBM shots vs current run.

For each matching *shots.txt basename, trains the same CharLSTM + SAX pipeline as
``run_ibm_grammar_learning.py`` on reference then current data (same seed each
train for comparable init), extracts transition matrices, and reports
``compute_kl_divergence`` (row-averaged KL) in both directions plus Frobenius
distance between matrices.
"""

from __future__ import annotations

import argparse
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
RESULTS = os.path.join(REPO_ROOT, "results")


def load_signal(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
    if "output" not in df.columns:
        raise ValueError(f"No 'output' column in {path}")
    return df["output"].astype(np.float64).values


def train_fingerprint(
    signal: np.ndarray,
    label: str,
    *,
    alphabet_size: int,
    hidden_dim: int,
    seq_len: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray | None, float, float]:
    val_loss, ppl, model, val_data = train_model(
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
    if model is None or val_data is None:
        return None, float(val_loss), float(ppl)
    mat = extract_grammar(model, val_data, seq_len=seq_len)
    return np.asarray(mat, dtype=np.float64), float(val_loss), float(ppl)


def main() -> None:
    p = argparse.ArgumentParser(description="Grammar KL archive vs current IBM shots")
    p.add_argument(
        "--reference-dir",
        default=os.path.join(RESULTS, "ibm_raw_shots", "archive_ibm_torino_20260329_122743"),
    )
    p.add_argument("--current-dir", default=os.path.join(RESULTS, "ibm_raw_shots"))
    p.add_argument("--seed", type=int, default=0, help="Same seed per train (ref then cur).")
    p.add_argument("--alphabet-size", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--out-csv",
        default=os.path.join(RESULTS, "ibm_torino_grammar_pairwise_kl.csv"),
    )
    p.add_argument(
        "--out-txt",
        default=os.path.join(RESULTS, "ibm_torino_grammar_pairwise_kl.txt"),
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

        print(f"\n{'=' * 60}\nPair: {rf.name}\n{'=' * 60}", flush=True)
        sig_ref = load_signal(str(rf))
        sig_cur = load_signal(str(cf))

        T_ref, vl_r, pp_r = train_fingerprint(
            sig_ref,
            f"{rf.stem}_archive",
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )
        T_cur, vl_c, pp_c = train_fingerprint(
            sig_cur,
            f"{rf.stem}_current",
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )

        if T_ref is None or T_cur is None:
            rows.append(
                {
                    "file": rf.name,
                    "kl_ref_to_cur": float("nan"),
                    "kl_cur_to_ref": float("nan"),
                    "sym_kl": float("nan"),
                    "frob_diff": float("nan"),
                    "val_loss_ref": vl_r,
                    "val_loss_cur": vl_c,
                    "ppl_ref": pp_r,
                    "ppl_cur": pp_c,
                    "ok": 0,
                }
            )
            continue

        kl_rc = compute_kl_divergence(T_ref, T_cur)
        kl_cr = compute_kl_divergence(T_cur, T_ref)
        sym = 0.5 * (kl_rc + kl_cr)
        frob = float(np.linalg.norm(T_ref - T_cur, ord="fro"))

        rows.append(
            {
                "file": rf.name,
                "kl_ref_to_cur": kl_rc,
                "kl_cur_to_ref": kl_cr,
                "sym_kl": sym,
                "frob_diff": frob,
                "val_loss_ref": vl_r,
                "val_loss_cur": vl_c,
                "ppl_ref": pp_r,
                "ppl_cur": pp_c,
                "ok": 1,
            }
        )
        print(
            f"  KL(archive||current)={kl_rc:.6f}  KL(current||archive)={kl_cr:.6f}  "
            f"sym={sym:.6f}  ||T_ref-T_cur||_F={frob:.6f}",
            flush=True,
        )

    if not rows:
        print("No paired files.", file=sys.stderr)
        sys.exit(3)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    ok = out_df[out_df["ok"] == 1]
    lines = [
        "IBM Torino grammar fingerprint pairwise comparison",
        f"reference_dir: {ref_dir.resolve()}",
        f"current_dir:   {cur_dir.resolve()}",
        f"train: alphabet={args.alphabet_size} hidden={args.hidden_dim} seq_len={args.seq_len} "
        f"epochs={args.epochs} lr={args.lr} seed={args.seed}",
        "",
        out_df.to_string(index=False),
        "",
    ]
    if len(ok):
        lines.extend(
            [
                f"mean sym_kl (ok rows): {ok['sym_kl'].mean():.6f}",
                f"mean frob_diff:       {ok['frob_diff'].mean():.6f}",
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
