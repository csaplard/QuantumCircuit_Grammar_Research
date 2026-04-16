"""
Reproducible transformers A/B runner for Fisher-guided adaptive decoding.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROMPTS = {
    "factual": "List the capital cities of European Union member states. Be concise, one per line.",
    "mathematical": "Prove that the square root of 2 is irrational. Show each logical step clearly.",
}


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline vs adaptive transformers A/B")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--window-size", type=int, default=32)
    ap.add_argument("--fisher-epsilon", type=float, default=0.01)
    ap.add_argument("--temperature-init", type=float, default=0.7)
    ap.add_argument("--temperature-low", type=float, default=0.4)
    ap.add_argument("--temperature-high", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/llm_entropy/fisher_runs/transformers_ab_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["regime", "mode", "trace_csv", "text_path"])

    for regime, prompt in PROMPTS.items():
        # Adaptive
        trace_a = out_dir / f"{regime}_adaptive_trace.csv"
        text_a = out_dir / f"{regime}_adaptive_output.txt"
        _run(
            [
                sys.executable,
                "code/transformers_fisher_adaptive_minimal.py",
                "--model-id",
                args.model_id,
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--window-size",
                str(args.window_size),
                "--fisher-epsilon",
                str(args.fisher_epsilon),
                "--temperature-init",
                str(args.temperature_init),
                "--temperature-low",
                str(args.temperature_low),
                "--temperature-high",
                str(args.temperature_high),
                "--top-p",
                str(args.top_p),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--seed",
                str(args.seed),
                "--out-csv",
                str(trace_a),
                "--out-text",
                str(text_a),
            ]
        )
        with manifest.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([regime, "adaptive", str(trace_a).replace("\\", "/"), str(text_a).replace("\\", "/")])

        # Baseline (fixed temp)
        trace_b = out_dir / f"{regime}_baseline_trace.csv"
        text_b = out_dir / f"{regime}_baseline_output.txt"
        _run(
            [
                sys.executable,
                "code/transformers_fisher_adaptive_minimal.py",
                "--model-id",
                args.model_id,
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--window-size",
                str(args.window_size),
                "--fisher-epsilon",
                str(args.fisher_epsilon),
                "--temperature-init",
                str(args.temperature_init),
                "--temperature-low",
                str(args.temperature_init),
                "--temperature-high",
                str(args.temperature_init),
                "--stats-warmup-checks",
                "9999",
                "--top-p",
                str(args.top_p),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--seed",
                str(args.seed),
                "--out-csv",
                str(trace_b),
                "--out-text",
                str(text_b),
            ]
        )
        with manifest.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([regime, "baseline", str(trace_b).replace("\\", "/"), str(text_b).replace("\\", "/")])

    print(f"\nDone. Output dir: {out_dir}", flush=True)
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
