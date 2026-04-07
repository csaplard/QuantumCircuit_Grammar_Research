"""Ollama HTTP client: logprobs with fixed output length via chunked generate + context continuation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed ({url}): {e}") from e
    return json.loads(body)


def generate_with_logprobs(
    *,
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    top_logprobs: int,
    temperature: float,
    seed: int | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Single-shot generate (backward compatible). Prefer generate_fixed_token_logprobs for fair cross-regime N*."""
    return generate_fixed_token_logprobs(
        base_url=base_url,
        model=model,
        prompt=prompt,
        target_tokens=num_predict,
        chunk_tokens=num_predict,
        top_logprobs=top_logprobs,
        temperature=temperature,
        seed=seed,
        timeout_per_chunk_s=timeout_s,
        use_raw=False,
    )


def generate_fixed_token_logprobs(
    *,
    base_url: str,
    model: str,
    prompt: str,
    target_tokens: int,
    chunk_tokens: int = 512,
    top_logprobs: int,
    temperature: float,
    seed: int | None = None,
    timeout_per_chunk_s: float = 900.0,
    clear_stop_sequences: bool = True,
    num_ctx: int | None = None,
    use_raw: bool = False,
    max_empty_chunk_retries: int = 4,
) -> dict[str, Any]:
    """
    Accumulate exactly `target_tokens` logprob steps (Sycamore-style fixed max_pts).

    Ollama often stops at EOS before num_predict; we continue with `context` from the
    previous response (empty prompt) until we reach the target or hit a safety cap.
    `options.stop: []` reduces extra string stop sequences; EOS may still end a chunk early.
    `use_raw: True` sets Ollama `raw` (no chat template); may change EOS behaviour (model-dependent).
    Empty `logprobs` chunks: retry with a whitespace nudge to the prompt (same `context`).

    Returns:
      response (full text), logprobs (length == target_tokens), context, continuation_rounds, round_meta
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    chunk_tokens = max(1, min(int(chunk_tokens), int(target_tokens)))

    url = base_url.rstrip("/") + "/api/generate"
    all_logprobs: list[dict[str, Any]] = []
    response_parts: list[str] = []
    round_meta: list[dict[str, Any]] = []
    context: list[int] | None = None
    prompt_use = prompt
    max_rounds = (target_tokens + chunk_tokens - 1) // chunk_tokens + 60 + max_empty_chunk_retries * 3
    rounds = 0
    empty_retries = 0

    while len(all_logprobs) < target_tokens and rounds < max_rounds:
        rounds += 1
        need = target_tokens - len(all_logprobs)
        n_predict = min(chunk_tokens, need)

        options: dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(n_predict),
        }
        if clear_stop_sequences:
            options["stop"] = []
        if seed is not None and context is None:
            options["seed"] = int(seed)
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt_use,
            "stream": False,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
            "options": options,
        }
        if use_raw:
            payload["raw"] = True
        if context is not None:
            payload["context"] = context

        out = _post_json(url, payload, timeout_per_chunk_s)

        chunk_lps = out.get("logprobs") or []
        if not chunk_lps:
            if context is not None and empty_retries < max_empty_chunk_retries:
                empty_retries += 1
                prompt_use = " " if prompt_use == "" else prompt_use + " "
                round_meta.append(
                    {
                        "round": rounds,
                        "done_reason": out.get("done_reason"),
                        "n_logprobs": 0,
                        "note": f"empty logprobs; retry {empty_retries}/{max_empty_chunk_retries} with prompt nudge",
                    }
                )
                continue
            round_meta.append(
                {
                    "round": rounds,
                    "done_reason": out.get("done_reason"),
                    "n_logprobs": 0,
                    "note": "empty logprobs chunk; stopping",
                }
            )
            break

        empty_retries = 0
        all_logprobs.extend(chunk_lps)
        response_parts.append(out.get("response") or "")
        context = out.get("context")
        round_meta.append(
            {
                "round": rounds,
                "done_reason": out.get("done_reason"),
                "eval_count": out.get("eval_count"),
                "n_logprobs": len(chunk_lps),
            }
        )
        prompt_use = ""
        if context is None:
            round_meta[-1]["note"] = "missing context; cannot continue"
            break

    all_logprobs = all_logprobs[:target_tokens]
    return {
        "backend": "ollama",
        "model": model,
        "response": "".join(response_parts),
        "logprobs": all_logprobs,
        "context": context,
        "continuation_rounds": rounds,
        "round_meta": round_meta,
        "target_tokens": target_tokens,
        "actual_tokens": len(all_logprobs),
        "truncated_before_target": len(all_logprobs) < target_tokens,
        "done": True,
    }
