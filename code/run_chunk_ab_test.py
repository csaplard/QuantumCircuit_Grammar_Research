"""
Chunk-based A/B test for Fisher-adaptive sampling (Ollama).

Baseline:
  fixed temperature across chunks.
Adaptive:
  after each chunk, compute sliding Fisher trace over accumulated entropy stream
  and set temperature by regime thresholds:
    trace < low  -> temp_down
    trace > high -> temp_up
    else         -> temp_hold (keep current)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_entropy.entropy import entropy_from_top_logprobs
from llm_entropy.ollama_client import _post_json
from llm_entropy.regimes import REGIMES
from sliding_fisher import SlidingFisherTrace


REGIME_THRESHOLDS = {
    "factual": {"low": 343.0, "high": 401.0},
    "mathematical": {"low": 351.0, "high": 407.0},
}


def _generate_chunk(
    *,
    base_url: str,
    model: str,
    prompt: str,
    context: list[int] | None,
    num_predict: int,
    temperature: float,
    top_p: float,
    top_logprobs: int,
    seed: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_predict": int(num_predict),
        "stop": [],
    }
    if seed is not None and context is None:
        options["seed"] = int(seed)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "options": options,
    }
    if context is not None:
        payload["context"] = context
    return _post_json(base_url.rstrip("/") + "/api/generate", payload, timeout_s=240.0)


def _run_one(
    *,
    mode: str,
    regime: str,
    prompt: str,
    run_id: str,
    out_dir: Path,
    ollama_url: str,
    model: str,
    seed: int,
    chunk_tokens: int,
    n_chunks: int,
    top_p: float,
    baseline_temp: float,
    temp_low: float,
    temp_high: float,
    fisher_window: int,
    fisher_epsilon: float,
    top_logprobs: int,
) -> tuple[Path, Path, Path]:
    thresholds = REGIME_THRESHOLDS[regime]
    tracker = SlidingFisherTrace(
        window_size=fisher_window,
        alphabet_size=7,
        sax_behavior="quantile",
        fisher_epsilon=fisher_epsilon,
    )

    temperature = float(baseline_temp)
    context: list[int] | None = None
    prompt_use = prompt
    text_parts: list[str] = []

    trace_csv = out_dir / f"{run_id}_{mode}_trace.csv"
    text_path = out_dir / f"{run_id}_{mode}_text.txt"
    chunk_csv = out_dir / f"{run_id}_{mode}_chunks.csv"

    with trace_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["mode", "regime", "run_id", "step", "token", "entropy", "fisher_trace", "temperature"])
    with chunk_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["mode", "regime", "run_id", "chunk_idx", "temp_before", "fisher_after_chunk", "decision"])

    step = 0
    for chunk_idx in range(n_chunks):
        resp = _generate_chunk(
            base_url=ollama_url,
            model=model,
            prompt=prompt_use,
            context=context,
            num_predict=chunk_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
            seed=seed,
        )
        lps = resp.get("logprobs") or []
        if not lps:
            break

        fisher_after_chunk: float | None = None
        for lp in lps:
            token = str(lp.get("token", ""))
            entropy = float(entropy_from_top_logprobs(lp))
            text_parts.append(token)
            fisher_after_chunk = tracker.update(entropy)
            with trace_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        mode,
                        regime,
                        run_id,
                        step,
                        token,
                        f"{entropy:.8f}",
                        "" if fisher_after_chunk is None else f"{fisher_after_chunk:.8f}",
                        f"{temperature:.4f}",
                    ]
                )
            step += 1

        decision = "hold"
        if mode == "adaptive" and fisher_after_chunk is not None:
            if fisher_after_chunk < thresholds["low"]:
                temperature = float(temp_low)
                decision = "down"
            elif fisher_after_chunk > thresholds["high"]:
                temperature = float(temp_high)
                decision = "up"
            else:
                decision = "hold"

        with chunk_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    mode,
                    regime,
                    run_id,
                    chunk_idx,
                    f"{temperature:.4f}",
                    "" if fisher_after_chunk is None else f"{fisher_after_chunk:.8f}",
                    decision,
                ]
            )

        context = resp.get("context")
        if context is not None:
            prompt_use = ""
            continue
        break

    text_path.write_text("".join(text_parts), encoding="utf-8")
    return trace_csv, chunk_csv, text_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk-based A/B test: baseline vs Fisher-adaptive")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--regimes", default="factual,mathematical")
    ap.add_argument("--runs-per-mode", type=int, default=4)
    ap.add_argument("--chunk-tokens", type=int, default=32)
    ap.add_argument("--n-chunks", type=int, default=8)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--baseline-temp", type=float, default=0.7)
    ap.add_argument("--temp-low", type=float, default=0.4)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--fisher-window", type=int, default=32)
    ap.add_argument("--fisher-epsilon", type=float, default=0.01)
    ap.add_argument("--top-logprobs", type=int, default=15)
    ap.add_argument("--seed-base", type=int, default=200)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/llm_entropy/fisher_runs/chunk_ab_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regimes:
        if r not in REGIMES:
            raise SystemExit(f"Unknown regime: {r}")
        if r not in REGIME_THRESHOLDS:
            raise SystemExit(f"No thresholds configured for regime: {r}")

    manifest = out_dir / "chunk_ab_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["mode", "regime", "run_id", "trace_csv", "chunk_csv", "text_path"])

    for regime in regimes:
        prompts = REGIMES[regime]
        for mode in ("baseline", "adaptive"):
            for i in range(args.runs_per_mode):
                run_id = f"{regime}_r{i}"
                prompt = prompts[i % len(prompts)]
                seed = args.seed_base + i
                trace_csv, chunk_csv, text_path = _run_one(
                    mode=mode,
                    regime=regime,
                    prompt=prompt,
                    run_id=run_id,
                    out_dir=out_dir,
                    ollama_url=args.ollama_url,
                    model=args.model,
                    seed=seed,
                    chunk_tokens=args.chunk_tokens,
                    n_chunks=args.n_chunks,
                    top_p=args.top_p,
                    baseline_temp=args.baseline_temp,
                    temp_low=args.temp_low,
                    temp_high=args.temp_high,
                    fisher_window=args.fisher_window,
                    fisher_epsilon=args.fisher_epsilon,
                    top_logprobs=args.top_logprobs,
                )
                with manifest.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [mode, regime, run_id, str(trace_csv).replace("\\", "/"), str(chunk_csv).replace("\\", "/"), str(text_path).replace("\\", "/")]
                    )
                print(f"{mode} {regime} {run_id} -> {text_path}", flush=True)

    print(f"\nDone. Output dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
