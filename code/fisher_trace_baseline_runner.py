"""
Collect non-adaptive sliding Fisher traces for baseline calibration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from llm_entropy.entropy import entropy_from_top_logprobs
from fisher_adaptive_sampler import _step_generate_chunk
from sliding_fisher import SlidingFisherTrace


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline run: fixed sampler + sliding Fisher logging")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--regime", default="default")
    ap.add_argument("--run-id", default="baseline_run")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--top-logprobs", type=int, default=15)
    ap.add_argument("--chunk-tokens", type=int, default=16)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--alphabet-size", type=int, default=7)
    ap.add_argument("--sax-behavior", choices=("quantile", "gaussian"), default="quantile")
    ap.add_argument("--fisher-epsilon", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="results/llm_entropy/fisher_runs/fisher_baseline_trace.csv")
    args = ap.parse_args()

    tracker = SlidingFisherTrace(
        window_size=args.window_size,
        alphabet_size=args.alphabet_size,
        sax_behavior=args.sax_behavior,
        fisher_epsilon=args.fisher_epsilon,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["mode", "regime", "run_id", "step", "token", "entropy", "fisher_trace"])

    context = None
    prompt_use = args.prompt
    step = 0
    while step < args.max_tokens:
        need = args.max_tokens - step
        resp = _step_generate_chunk(
            base_url=args.ollama_url,
            model=args.model,
            prompt=prompt_use,
            context=context,
            top_logprobs=args.top_logprobs,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            num_ctx=None,
            num_predict=min(args.chunk_tokens, need),
        )
        lps = resp.get("logprobs") or []
        if not lps:
            break
        for lp in lps:
            token = str(lp.get("token", ""))
            entropy = float(entropy_from_top_logprobs(lp))
            fisher_trace = tracker.update(entropy)

            with out_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "baseline",
                        args.regime,
                        args.run_id,
                        step,
                        token,
                        f"{entropy:.8f}",
                        "" if fisher_trace is None else f"{fisher_trace:.8f}",
                    ]
                )
            step += 1
            if step >= args.max_tokens:
                break
        context = resp.get("context")
        if step >= args.max_tokens:
            break
        if context is not None:
            prompt_use = ""
            continue
        piece = str(resp.get("response", ""))
        if not piece:
            break
        prompt_use += piece

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
