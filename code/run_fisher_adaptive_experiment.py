"""
End-to-end runner:
  baseline traces -> threshold calibration -> adaptive traces -> A/B scoring template.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from llm_entropy.regimes import REGIMES


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _trace_std(path: Path) -> float:
    df = pd.read_csv(path)
    if "fisher_trace" not in df.columns:
        return float("nan")
    vals = pd.to_numeric(df["fisher_trace"], errors="coerce").dropna().values
    if vals.size == 0:
        return float("nan")
    return float(np.std(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline+adaptive Fisher A/B experiment")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--regimes", default="factual,mathematical,creative,philosophical")
    ap.add_argument("--runs-per-regime", type=int, default=2, help="2 => 8 baseline + 8 adaptive runs")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--fisher-check-every", type=int, default=16)
    ap.add_argument("--chunk-tokens", type=int, default=16)
    ap.add_argument("--fisher-epsilon", type=float, default=1e-10)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/llm_entropy/fisher_runs/ab_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_list = [r.strip() for r in args.regimes.split(",") if r.strip()]
    bad = [r for r in regime_list if r not in REGIMES]
    if bad:
        raise SystemExit(f"Unknown regimes: {bad}")

    baseline_rows: list[dict] = []
    adaptive_rows: list[dict] = []

    # Phase A: baseline runs
    for regime in regime_list:
        prompts = REGIMES[regime]
        for i in range(args.runs_per_regime):
            prompt = prompts[i % len(prompts)]
            run_id = f"{regime}_r{i}"
            out_csv = out_dir / f"{run_id}_baseline_trace.csv"
            seed = args.seed_base + i
            _run(
                [
                    sys.executable,
                    "code/fisher_trace_baseline_runner.py",
                    "--ollama-url",
                    args.ollama_url,
                    "--model",
                    args.model,
                    "--prompt",
                    prompt,
                    "--regime",
                    regime,
                    "--run-id",
                    run_id,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top-p",
                    str(args.top_p),
                    "--window-size",
                    str(args.window_size),
                    "--seed",
                    str(seed),
                    "--chunk-tokens",
                    str(args.chunk_tokens),
                    "--fisher-epsilon",
                    str(args.fisher_epsilon),
                    "--out-csv",
                    str(out_csv),
                ]
            )
            baseline_rows.append(
                {
                    "mode": "baseline",
                    "regime": regime,
                    "prompt_id": f"{regime}_p{i % len(prompts)}",
                    "run_id": run_id,
                    "trace_csv": str(out_csv).replace("\\", "/"),
                }
            )

    # Phase B: calibration
    profile_json = out_dir / "fisher_baseline_profiles.json"
    _run(
        [
            sys.executable,
            "code/calibrate_fisher_baseline_profiles.py",
            "--glob",
            str(out_dir / "*baseline_trace.csv"),
            "--out-json",
            str(profile_json),
        ]
    )

    # Phase C: adaptive runs
    for regime in regime_list:
        prompts = REGIMES[regime]
        for i in range(args.runs_per_regime):
            prompt = prompts[i % len(prompts)]
            run_id = f"{regime}_r{i}"
            out_csv = out_dir / f"{run_id}_adaptive_trace.csv"
            out_txt = out_dir / f"{run_id}_adaptive_text.txt"
            seed = args.seed_base + i
            _run(
                [
                    sys.executable,
                    "code/fisher_adaptive_sampler.py",
                    "--ollama-url",
                    args.ollama_url,
                    "--model",
                    args.model,
                    "--prompt",
                    prompt,
                    "--regime",
                    regime,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--window-size",
                    str(args.window_size),
                    "--fisher-check-every",
                    str(args.fisher_check_every),
                    "--temperature-init",
                    str(args.temperature),
                    "--top-p-init",
                    str(args.top_p),
                    "--seed",
                    str(seed),
                    "--chunk-tokens",
                    str(args.chunk_tokens),
                    "--fisher-epsilon",
                    str(args.fisher_epsilon),
                    "--profile-json",
                    str(profile_json),
                    "--out-csv",
                    str(out_csv),
                    "--out-text",
                    str(out_txt),
                ]
            )
            adaptive_rows.append(
                {
                    "mode": "adaptive",
                    "regime": regime,
                    "prompt_id": f"{regime}_p{i % len(prompts)}",
                    "run_id": run_id,
                    "trace_csv": str(out_csv).replace("\\", "/"),
                    "text_path": str(out_txt).replace("\\", "/"),
                }
            )

    # Phase D: scoring template (manual 1-5 fields + prefilled Fisher std)
    score_csv = out_dir / "ab_scoring_template.csv"
    with score_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "regime",
                "prompt_id",
                "run_id",
                "trace_csv",
                "score_coherence",
                "score_hallucination",
                "score_structure",
                "fisher_trace_std",
            ]
        )
        for row in baseline_rows + adaptive_rows:
            stdv = _trace_std(Path(row["trace_csv"]))
            w.writerow(
                [
                    row["mode"],
                    row["regime"],
                    row["prompt_id"],
                    row["run_id"],
                    row["trace_csv"],
                    "",
                    "",
                    "",
                    f"{stdv:.8f}" if np.isfinite(stdv) else "",
                ]
            )

    print(f"\nDone.\nOutput dir: {out_dir}\nProfile: {profile_json}\nScoring template: {score_csv}", flush=True)


if __name__ == "__main__":
    main()
