"""
Cognitive regime prompts (factual / creative / mathematical / philosophical).
Multiple prompts per regime for stability across prompt variation.
"""

from __future__ import annotations

REGIMES: dict[str, list[str]] = {
    "factual": [
        "List the capital cities of European Union member states. Be concise, one per line.",
        "Name the planets of the solar system in order from the Sun. Short answers only.",
    ],
    "creative": [
        "Write freely about solitude — images, tone, and form are yours to choose.",
        "Imagine a place that only exists at twilight. Describe it without a fixed length or structure.",
    ],
    "mathematical": [
        "Prove that the square root of 2 is irrational. Show each logical step clearly.",
        "Show step by step: derive the quadratic formula from ax^2+bx+c=0.",
    ],
    "philosophical": [
        "Explore the nature of time. Follow your reasoning wherever it leads.",
        "Think about whether moral truth is objective. Let the argument develop freely.",
    ],
}

# Optional one-line system-style prefix (kept minimal; model-dependent)
REGIME_TAGS: dict[str, str] = {
    "factual": "",
    "creative": "",
    "mathematical": "",
    "philosophical": "",
}
