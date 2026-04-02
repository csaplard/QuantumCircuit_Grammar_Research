import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from grammar_learner import extract_grammar, train_model

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(REPO_ROOT, "results", "ibm_raw_shots")
OUT_CSV = os.path.join(REPO_ROOT, "results", "ibm_grammar_learning_results.csv")
OUT_NPZ = os.path.join(REPO_ROOT, "results", "ibm_grammar_fingerprints.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-prefix", type=str, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument(
        "--filename-substr",
        type=str,
        default=None,
        help="If set, keep only shot files whose basename contains this (e.g. 163840 for reps=20).",
    )
    args = parser.parse_args()

    file_glob = "*shots.txt" if not args.backend_prefix else f"{args.backend_prefix}_*shots.txt"
    files = sorted(glob.glob(os.path.join(INPUT_DIR, file_glob)))
    if args.filename_substr:
        files = [f for f in files if args.filename_substr in os.path.basename(f)]
    if not files:
        print(f"No input files found in: {INPUT_DIR}")
        return

    rows = []
    matrices = {}
    params = {"hidden_dim": 32, "seq_len": 16, "epochs": 20, "lr": 0.01}

    print(f"Found {len(files)} IBM shot files.")
    for path in files:
        name = os.path.basename(path)
        label = name.replace(".txt", "")
        df = pd.read_csv(path, sep=" ", on_bad_lines="skip")
        signal = df["output"].astype(np.float64).values

        val_loss, ppl, model, val_data = train_model(
            signal,
            label,
            alphabet_size=7,
            data_is_array=True,
            seed=0,
            **params,
        )
        if model is None:
            continue

        fp = extract_grammar(model, val_data, seq_len=params["seq_len"])
        matrices[label] = fp
        rows.append(
            {
                "file": name,
                "n_samples": int(len(signal)),
                "val_loss": float(val_loss),
                "perplexity": float(ppl),
            }
        )

    out_csv = OUT_CSV
    out_npz = OUT_NPZ
    if args.output_tag:
        out_csv = os.path.join(REPO_ROOT, "results", f"ibm_grammar_learning_results_{args.output_tag}.csv")
        out_npz = os.path.join(REPO_ROOT, "results", f"ibm_grammar_fingerprints_{args.output_tag}.npz")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n_samples", "val_loss", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(out_npz, **matrices)
    print(f"Saved summary: {out_csv}")
    print(f"Saved fingerprints: {out_npz}")


if __name__ == "__main__":
    main()
