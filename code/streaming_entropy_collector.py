"""
Streaming token-by-token entropy collector from Ollama /api/generate.

Unlike collect_llm_entropy_series.py (offline fixed-length batch),
this script yields per-token entropy in real time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Iterable
import urllib.error
import urllib.request

from llm_entropy.entropy import entropy_from_top_logprobs


def _stream_generate_events(
    *,
    base_url: str,
    model: str,
    prompt: str,
    options: dict[str, Any],
    top_logprobs: int,
    context: list[int] | None = None,
    timeout_s: float = 1800.0,
) -> Iterable[dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "options": options,
    }
    if context is not None:
        payload["context"] = context

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                yield json.loads(line)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama stream request failed: {e}") from e


def _entropy_steps_from_event(event: dict[str, Any]) -> list[tuple[str, float]]:
    """
    Return list of (token_text, entropy) items from one stream event.
    Works with both per-event 'logprobs' list and single-step 'top_logprobs' payloads.
    """
    out: list[tuple[str, float]] = []
    logprobs = event.get("logprobs") or []
    if isinstance(logprobs, list) and logprobs:
        for step in logprobs:
            token = str(step.get("token", ""))
            out.append((token, float(entropy_from_top_logprobs(step))))
        return out

    if event.get("top_logprobs") is not None or event.get("logprob") is not None:
        token = str(event.get("response", ""))
        out.append((token, float(entropy_from_top_logprobs(event))))
    return out


def _append_row(path: str, row: list[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time entropy stream from Ollama tokens")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-logprobs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=None, help="Optional CSV output path")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    args = ap.parse_args()

    out_csv = args.out_csv
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "token", "entropy"])

    options = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_predict": int(args.max_tokens),
        "seed": int(args.seed),
    }

    n = 0
    final_context: list[int] | None = None
    text_parts: list[str] = []

    for event in _stream_generate_events(
        base_url=args.ollama_url,
        model=args.model,
        prompt=args.prompt,
        options=options,
        top_logprobs=args.top_logprobs,
        timeout_s=args.timeout_s,
    ):
        steps = _entropy_steps_from_event(event)
        for token, entropy in steps:
            text_parts.append(token)
            print(f"{n}\t{entropy:.8f}\t{token!r}", flush=True)
            if out_csv:
                _append_row(out_csv, [n, token, f"{entropy:.8f}"])
            n += 1
            if n >= args.max_tokens:
                break
        if n >= args.max_tokens:
            break
        if event.get("done"):
            final_context = event.get("context")
            break

    if final_context is not None:
        print(f"\nDone: {n} tokens, context_length={len(final_context)}", flush=True)
    else:
        print(f"\nDone: {n} tokens", flush=True)

    if out_csv:
        print(f"Wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
