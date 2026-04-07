"""Shannon entropy (nats) from Ollama top_logprobs + remainder bucket."""

from __future__ import annotations

import math
from typing import Any


def entropy_from_top_logprobs(step: dict[str, Any]) -> float:
    """
    One generation step: Ollama returns { token, logprob, top_logprobs: [{token, logprob}, ...] }.
    Probabilities p_i = exp(logprob_i) for each distinct token in top_logprobs; remainder r = 1 - sum(p_i).
    H = -sum_i p_i log p_i - r log r (natural log, nats).
    """
    top = step.get("top_logprobs") or []
    if not top:
        lp = float(step.get("logprob", -1e9))
        p = math.exp(min(lp, 0.0))
        p = min(max(p, 0.0), 1.0)
        r = max(0.0, 1.0 - p)
        return _binary_entropy(p, r)

    probs: list[float] = []
    seen: set[str] = set()
    for item in top:
        tok = item.get("token", "")
        if tok in seen:
            continue
        seen.add(tok)
        lp = float(item.get("logprob", -1e9))
        p = math.exp(min(lp, 0.0))
        probs.append(p)

    s = sum(probs)
    if s > 1.0 + 1e-5:
        probs = [p / s for p in probs]
        s = 1.0
    r = max(0.0, 1.0 - s)
    if r < 1e-15 and s < 1e-15:
        return 0.0

    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    if r > 0:
        h -= r * math.log(r)
    return float(h)


def _binary_entropy(p: float, r: float) -> float:
    tot = p + r
    if tot <= 0:
        return 0.0
    p, r = p / tot, r / tot
    h = 0.0
    if p > 0:
        h -= p * math.log(p)
    if r > 0:
        h -= r * math.log(r)
    return float(h)


def series_from_response_logprobs(logprobs: list[dict[str, Any]]) -> list[float]:
    return [entropy_from_top_logprobs(step) for step in logprobs]
