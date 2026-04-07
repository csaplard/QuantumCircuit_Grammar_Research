"""
Collect token-level entropy time series for Grammar+Fisher (fixed max length per regime).

Backends:
  - ollama: chunked generate + context continuation; optional --ollama-raw; empty-logprob retries.
  - llama-server: llama.cpp HTTP /completion with ignore_eos (recommended for exact fixed length).

Fixed token budget (Sycamore-style same max_pts):
  - --fixed-tokens: target generated-token steps (e.g. 20000).
  - Ollama: --chunk-tokens caps each request; continuation may still stall on EOS.
  - llama-server: one request with n_predict=fixed-tokens and ignore_eos=true (raise server -c if truncated).

Examples:
  python code/collect_llm_entropy_series.py --quick --smoke
  python code/collect_llm_entropy_series.py --backend llama-server --llama-server-url http://127.0.0.1:8080 --quick --fixed-tokens 1024
  python code/collect_llm_entropy_series.py --ollama-raw --fixed-tokens 8192 --num-ctx 32768
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(REPO_ROOT, "results", "llm_entropy", "runs")

sys.path.insert(0, SCRIPT_DIR)
from llm_entropy.entropy import series_from_response_logprobs  # noqa: E402
from llm_entropy.llama_server_client import complete_fixed_tokens  # noqa: E402
from llm_entropy.ollama_client import generate_fixed_token_logprobs  # noqa: E402
from llm_entropy.regimes import REGIMES  # noqa: E402
from llm_entropy.repetition import repeat_step_flags, repetition_events  # noqa: E402


def _write_series(
    path: str,
    entropies: list[float],
    meta: dict,
    repeat_flags: list[int] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if repeat_flags is not None and len(repeat_flags) == len(entropies):
            w.writerow(["step", "entropy", "in_repeat_run"])
            for i, h in enumerate(entropies):
                w.writerow([i, f"{h:.8f}", int(repeat_flags[i])])
        else:
            w.writerow(["step", "entropy"])
            for i, h in enumerate(entropies):
                w.writerow([i, f"{h:.8f}"])
    meta_path = path.replace(".csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _append_manifest(manifest_path: str, row: dict) -> None:
    file_exists = os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect LLM entropy time series (fixed length)")
    ap.add_argument(
        "--backend",
        choices=("ollama", "llama-server"),
        default="ollama",
        help="llama-server: use llama.cpp /completion with ignore_eos (best for exact length). Default: ollama.",
    )
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--llama-server-url",
        default=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
        help="Base URL of llama-server (not Ollama). Env: LLAMA_SERVER_URL.",
    )
    ap.add_argument(
        "--llama-timeout",
        type=float,
        default=7200.0,
        help="HTTP timeout (seconds) for a single llama-server /completion call.",
    )
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument(
        "--fixed-tokens",
        type=int,
        default=8192,
        help="Target number of generated tokens (entropy steps) per run; same for all regimes (default 8192).",
    )
    ap.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        help="Max tokens per HTTP request; smaller chunks = more continuation rounds (default 512).",
    )
    ap.add_argument("--top-logprobs", type=int, default=15)
    ap.add_argument("--quick", action="store_true", help="One prompt per regime only")
    ap.add_argument(
        "--regimes",
        default=None,
        metavar="LIST",
        help="Comma-separated subset of regimes (default: all). Example: factual,mathematical",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: fixed-tokens=48, chunk-tokens=48 (implies --quick)",
    )
    ap.add_argument("--no-controls", action="store_true", help="Skip shuffled and random baselines")
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument(
        "--min-repeat-run",
        type=int,
        default=12,
        help="Flag consecutive identical tokens as repetition if run length >= this (default 12).",
    )
    ap.add_argument(
        "--keep-string-stops",
        action="store_true",
        help="Do not pass stop: [] (use model default stop strings).",
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        metavar="N",
        help="Optional Ollama num_ctx only (e.g. 32768). For llama-server, set context via server -c flag.",
    )
    ap.add_argument(
        "--ollama-raw",
        action="store_true",
        help="Ollama only: set raw=true (no chat template; may alter EOS behaviour).",
    )
    ap.add_argument(
        "--max-empty-chunk-retries",
        type=int,
        default=4,
        help="Ollama only: retries when a continuation returns empty logprobs (default 4).",
    )
    ap.add_argument(
        "--num-predict",
        type=int,
        default=None,
        metavar="N",
        help="Deprecated alias for --fixed-tokens (if set, overrides --fixed-tokens).",
    )
    args = ap.parse_args()

    fixed_tokens = args.num_predict if args.num_predict is not None else args.fixed_tokens
    chunk_tokens = args.chunk_tokens
    if args.smoke:
        fixed_tokens = 48
        chunk_tokens = 48
        args.quick = True

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(os.path.dirname(os.path.normpath(args.out_dir)), "manifest.csv")

    rng = np.random.default_rng(args.seed)

    regime_filter: set[str] | None = None
    if args.regimes:
        regime_filter = {x.strip() for x in args.regimes.split(",") if x.strip()}
        unknown = regime_filter - set(REGIMES.keys())
        if unknown:
            ap.error(f"Unknown regime(s): {sorted(unknown)}. Known: {sorted(REGIMES.keys())}")

    for regime, prompts in REGIMES.items():
        if regime_filter is not None and regime not in regime_filter:
            continue
        plist = prompts[:1] if args.quick else prompts
        for pi, prompt in enumerate(plist):
            base_id = f"{regime}_p{pi}_seed{args.seed}"
            print(
                f"\n=== {base_id} (backend={args.backend}, fixed_tokens={fixed_tokens}) ===",
                flush=True,
            )
            try:
                if args.backend == "llama-server":
                    resp = complete_fixed_tokens(
                        base_url=args.llama_server_url,
                        prompt=prompt,
                        target_tokens=fixed_tokens,
                        top_logprobs=args.top_logprobs,
                        temperature=args.temperature,
                        seed=args.seed,
                        timeout_s=args.llama_timeout,
                        ignore_eos=True,
                    )
                else:
                    resp = generate_fixed_token_logprobs(
                        base_url=args.ollama_url,
                        model=args.model,
                        prompt=prompt,
                        target_tokens=fixed_tokens,
                        chunk_tokens=chunk_tokens,
                        top_logprobs=args.top_logprobs,
                        temperature=args.temperature,
                        seed=args.seed,
                        clear_stop_sequences=not args.keep_string_stops,
                        num_ctx=args.num_ctx,
                        use_raw=args.ollama_raw,
                        max_empty_chunk_retries=args.max_empty_chunk_retries,
                    )
            except Exception as e:
                print(f"FAILED generate: {e}", flush=True)
                continue

            if resp.get("truncated_before_target"):
                if args.backend == "llama-server":
                    print(
                        f"  WARNING: only {resp.get('actual_tokens')} / {fixed_tokens} tokens "
                        f"(llama-server: increase -c / n_ctx on server or reduce fixed-tokens; "
                        f"stop_type={resp.get('stop_type')!r})",
                        flush=True,
                    )
                else:
                    print(
                        f"  WARNING: only {resp.get('actual_tokens')} / {fixed_tokens} tokens "
                        f"(Ollama: try --backend llama-server, --ollama-raw, or larger --num-ctx)",
                        flush=True,
                    )
            logprobs = resp.get("logprobs") or []
            series = series_from_response_logprobs(logprobs)
            if len(series) < 8:
                print(f"Too few tokens ({len(series)}), skip", flush=True)
                continue

            rep_ev = repetition_events(logprobs, min_run=args.min_repeat_run)
            rflags = repeat_step_flags(len(series), rep_ev)

            meta_base = {
                "backend": args.backend,
                "regime": regime,
                "prompt_index": pi,
                "prompt": prompt,
                "model": args.model,
                "llama_server_url": args.llama_server_url if args.backend == "llama-server" else None,
                "temperature": args.temperature,
                "fixed_tokens": fixed_tokens,
                "chunk_tokens": chunk_tokens if args.backend == "ollama" else None,
                "top_logprobs": args.top_logprobs,
                "seed": args.seed,
                "continuation_rounds": resp.get("continuation_rounds"),
                "round_meta": resp.get("round_meta"),
                "actual_tokens": resp.get("actual_tokens"),
                "truncated_before_target": resp.get("truncated_before_target"),
                "clear_stop_sequences": not args.keep_string_stops if args.backend == "ollama" else None,
                "ollama_raw": bool(args.ollama_raw) if args.backend == "ollama" else None,
                "num_ctx": args.num_ctx if args.backend == "ollama" else None,
                "stop_type": resp.get("stop_type"),
                "repetition_events": rep_ev,
                "response_excerpt": (resp.get("response") or "")[:800],
            }

            def save_variant(control: str, values: list[float], flags: list[int] | None) -> None:
                fname = f"{base_id}_{control}.csv"
                path = os.path.join(args.out_dir, fname)
                m = {**meta_base, "control": control, "n_steps": len(values)}
                _write_series(path, values, m, repeat_flags=flags if control == "none" else None)
                _append_manifest(
                    manifest_path,
                    {
                        "run_id": base_id,
                        "control": control,
                        "backend": args.backend,
                        "regime": regime,
                        "prompt_index": pi,
                        "seed": args.seed,
                        "model": args.model,
                        "n_steps": len(values),
                        "fixed_tokens": fixed_tokens,
                        "csv_path": os.path.relpath(path, REPO_ROOT),
                    },
                )
                print(f"  wrote {path} ({len(values)} steps)", flush=True)

            save_variant("none", series, rflags)

            if not args.no_controls:
                shuffled = series.copy()
                rng.shuffle(shuffled)
                save_variant("shuffled", shuffled, None)

                lo, hi = float(np.min(series)), float(np.max(series))
                if hi <= lo:
                    hi = lo + 1e-6
                random_series = [float(rng.uniform(lo, hi)) for _ in range(len(series))]
                save_variant("random_uniform", random_series, None)

    print(f"\nDone. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
