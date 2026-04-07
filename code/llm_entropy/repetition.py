"""Detect token repetition in Ollama logprob streams (mark steps; do not truncate)."""

from __future__ import annotations

from typing import Any


def repetition_events(logprobs: list[dict[str, Any]], min_run: int = 12) -> list[dict[str, Any]]:
    """
    Consecutive identical generated tokens (string match on `token` field).
    Returns event dicts with step_start, step_end, token, run_length.
    """
    if not logprobs or min_run < 2:
        return []
    events: list[dict[str, Any]] = []
    i = 0
    n = len(logprobs)
    while i < n:
        tok = logprobs[i].get("token", "")
        j = i + 1
        while j < n and logprobs[j].get("token", "") == tok:
            j += 1
        run = j - i
        if run >= min_run:
            events.append(
                {
                    "step_start": i,
                    "step_end": j - 1,
                    "token": tok[:200],
                    "run_length": run,
                }
            )
        i = j
    return events


def repeat_step_flags(n_steps: int, events: list[dict[str, Any]]) -> list[int]:
    """1 if step index lies inside any repetition run (for CSV marking)."""
    flags = [0] * n_steps
    for ev in events:
        a, b = int(ev["step_start"]), int(ev["step_end"])
        for k in range(max(0, a), min(n_steps, b + 1)):
            flags[k] = 1
    return flags
