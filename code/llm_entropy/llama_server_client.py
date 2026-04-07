"""
llama.cpp HTTP server (/completion) with ignore_eos — fixed-length generation without EOS stop.

Run llama-server separately, e.g. (Windows, adjust paths):
  llama-server.exe -m C:\\path\\to\\model.gguf -c 32768 --port 8080 --host 127.0.0.1

Use a context (-c) at least: prompt_tokens + fixed_tokens + margin.

API reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

import json
import math
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
        raise RuntimeError(f"llama-server request failed ({url}): {e}") from e
    return json.loads(body)


def _probs_array(out: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("completion_probabilities", "probs"):
        v = out.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def _normalize_logprob(x: Any) -> float:
    if x is None:
        return -1e9
    try:
        v = float(x)
    except (TypeError, ValueError):
        return -1e9
    if not math.isfinite(v):
        return -1e9
    return v


def llama_probs_to_ollama_logprobs(probs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map llama-server probability entries to Ollama-shaped dicts for entropy.py."""
    ollama_style: list[dict[str, Any]] = []
    for p in probs:
        top = p.get("top_logprobs") or p.get("top_probs") or []
        top_ollama = []
        for t in top:
            if "logprob" in t:
                lp = _normalize_logprob(t.get("logprob"))
            else:
                pr = float(t.get("prob", 0.0))
                lp = math.log(max(pr, 1e-300))
            top_ollama.append(
                {
                    "token": str(t.get("token", "")),
                    "logprob": lp,
                }
            )
        main_lp = _normalize_logprob(p.get("logprob"))
        if main_lp <= -1e8 and top_ollama:
            main_lp = top_ollama[0]["logprob"]
        ollama_style.append(
            {
                "token": str(p.get("token", "")),
                "logprob": main_lp,
                "top_logprobs": top_ollama,
            }
        )
    return ollama_style


def complete_fixed_tokens(
    *,
    base_url: str,
    prompt: str,
    target_tokens: int,
    top_logprobs: int,
    temperature: float,
    seed: int | None = None,
    timeout_s: float = 7200.0,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    """
    Single /completion call: n_predict == target_tokens, ignore_eos + optional EOS logit ban.

    Some servers still stop on an instruct-model EOS id despite ignore_eos; we then retry once
    with logit_bias ``[[eos_token_id, false]]`` using the last token id from the first response.

    Does not use Ollama; `base_url` is llama-server root (e.g. http://127.0.0.1:8080).
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    url = base_url.rstrip("/") + "/completion"
    body: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": int(target_tokens),
        "ignore_eos": bool(ignore_eos),
        "n_probs": int(max(1, top_logprobs)),
        "temperature": float(temperature),
        "stream": False,
        "stop": [],
    }
    if seed is not None:
        body["seed"] = int(seed)

    def _one_completion(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        o = _post_json(url, payload, timeout_s)
        pr = _probs_array(o)
        if not pr:
            raise RuntimeError(
                "llama-server returned no completion_probabilities/probs. "
                "Ensure n_probs > 0 in request and server version supports probability output."
            )
        return o, pr

    out, probs = _one_completion(body)
    stop_type = out.get("stop_type", "")
    actual = len(probs)

    # llama-server maps ignore_eos to extra EOG logit bias only; instruct models (e.g. Qwen2.5)
    # often still sample a dedicated EOS id (e.g. 151645) and stop with stop_type=eos.
    # If so, retry once banning the last emitted token id (the EOS step is included in probs).
    if (
        ignore_eos
        and stop_type == "eos"
        and actual < int(target_tokens)
        and probs
        and isinstance(probs[-1].get("id"), int)
    ):
        eos_tok = int(probs[-1]["id"])
        body_retry = {**body, "logit_bias": [[eos_tok, False]]}
        out, probs = _one_completion(body_retry)
        stop_type = out.get("stop_type", "")
        actual = len(probs)

    logprobs = llama_probs_to_ollama_logprobs(probs)
    content = out.get("content", "") or ""
    truncated = bool(out.get("truncated", False))
    actual = len(logprobs)

    return {
        "backend": "llama-server",
        "response": content,
        "logprobs": logprobs,
        "target_tokens": target_tokens,
        "actual_tokens": actual,
        "truncated_before_target": actual < target_tokens or truncated,
        "stop_type": stop_type,
        "generation_settings": out.get("generation_settings"),
        "continuation_rounds": 1,
        "round_meta": [
            {
                "round": 1,
                "n_logprobs": actual,
                "stop_type": stop_type,
                "truncated": truncated,
            }
        ],
        "done": True,
    }
