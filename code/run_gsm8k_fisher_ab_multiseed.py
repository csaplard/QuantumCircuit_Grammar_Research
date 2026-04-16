"""
Run GSM8K Fisher A/B across multiple seeds and aggregate results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed GSM8K A/B runner")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--answer-style", default="final_number", choices=("reasoning", "final_number"))
    ap.add_argument("--seed-start", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--out-dir", default="results/llm_entropy/fisher_runs/gsm8k_ab_multiseed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict] = []
    for k in range(args.n_seeds):
        seed = args.seed_start + k
        seed_dir = out_dir / f"seed_{seed}"
        cmd = [
            sys.executable,
            "code/run_gsm8k_fisher_ab.py",
            "--model-id",
            args.model_id,
            "--n-questions",
            str(args.n_questions),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--answer-style",
            args.answer_style,
            "--seed",
            str(seed),
            "--out-dir",
            str(seed_dir),
        ]
        _run(cmd)
        sdf = pd.read_csv(seed_dir / "gsm8k_ab_summary.csv").iloc[0].to_dict()
        sdf["seed"] = seed
        per_seed_rows.append(sdf)

    ps = pd.DataFrame(per_seed_rows).sort_values("seed")
    per_seed_csv = out_dir / "multiseed_per_seed_summary.csv"
    ps.to_csv(per_seed_csv, index=False)

    total_q = int(ps["n_questions"].sum())
    total_base = int(ps["baseline_correct"].sum())
    total_adap = int(ps["adaptive_correct"].sum())
    agg = {
        "n_seeds": int(args.n_seeds),
        "questions_per_seed": int(args.n_questions),
        "total_questions": total_q,
        "baseline_correct_total": total_base,
        "adaptive_correct_total": total_adap,
        "baseline_accuracy_total": total_base / total_q if total_q else 0.0,
        "adaptive_accuracy_total": total_adap / total_q if total_q else 0.0,
        "adaptive_minus_baseline_accuracy_total": (total_adap - total_base) / total_q if total_q else 0.0,
        "baseline_accuracy_mean_per_seed": float(ps["baseline_accuracy"].mean()),
        "adaptive_accuracy_mean_per_seed": float(ps["adaptive_accuracy"].mean()),
        "adaptive_minus_baseline_mean_per_seed": float(ps["adaptive_minus_baseline_accuracy"].mean()),
        "adaptive_minus_baseline_std_per_seed": float(ps["adaptive_minus_baseline_accuracy"].std(ddof=1)) if len(ps) > 1 else 0.0,
    }
    agg_csv = out_dir / "multiseed_aggregate_summary.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)

    print("\nDone.")
    print(f"Per-seed summary: {per_seed_csv}")
    print(f"Aggregate summary: {agg_csv}")
    print(agg)


if __name__ == "__main__":
    main()
