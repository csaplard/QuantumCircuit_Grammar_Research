"""
Single-qubit-bit control for Sycamore readout pipeline.

Uses only one bit of each multi-qubit readout integer per timestep, e.g. LSB
(output_bit_index=0), so every file is a 0/1 time series of the same format.

Same stack as validation: SAX (via grammar_learner) + LSTM + Frobenius + Ward + 3 clusters.

Interpretation (see results file header):
  - High blind accuracy → structure visible at single-bit level.
  - Near-chance vs full-integer pipeline → signal may live in the full state word.

Does not use the `input` column — only `output`, then bit extraction.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_validation_pipeline import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    run_single_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LSB/single-bit readout topology experiment")
    parser.add_argument(
        "--output-bit-index",
        type=int,
        default=0,
        help="Which bit of int(output) to keep (0 = LSB).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-pts", type=int, default=10_000)
    parser.add_argument("--alphabet", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 123, 999])
    parser.add_argument(
        "--out",
        default=os.path.join(RESULTS_DIR, "readout_lsb_bit_control_sax7.txt"),
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines: list[str] = []
    lines.append("=== SINGLE-BIT READOUT CONTROL (SAX=7) ===\n")
    lines.append(
        "Per file: signal[t] = (int(output[t]) >> k) & 1  (same pipeline as validation otherwise).\n"
    )
    lines.append(f"output_bit_index k = {args.output_bit_index} (0 = LSB)\n")
    lines.append(f"epochs={args.epochs}, max_pts={args.max_pts}, seeds={list(args.seeds)}\n\n")

    for seed in args.seeds:
        acc, _, cluster_lines = run_single_validation(
            seed=seed,
            alphabet_size=args.alphabet,
            epochs=args.epochs,
            max_pts=args.max_pts,
            shuffle=False,
            data_dir=DATA_DIR,
            output_bit_index=args.output_bit_index,
        )
        lines.append(f"--- seed={seed} accuracy={acc:.2f}% ---\n")
        for cl in cluster_lines:
            lines.append(f"  {cl}\n")
        lines.append("\n")
        print(f"seed={seed} accuracy={acc:.2f}%", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
