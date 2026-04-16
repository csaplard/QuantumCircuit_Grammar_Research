"""
Sliding-window Fisher trace from entropy time series (Markov direct estimate).

Pipeline per window:
  entropy window -> SAX symbols (K bins, quantile or gaussian) -> KxK Markov transition matrix
  -> Fisher matrix -> trace(F)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque

import numpy as np

from fisher_information_analysis import compute_fisher_matrix, fisher_scalar


def _z_norm(values: np.ndarray) -> np.ndarray:
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sigma


def _breakpoints_from_window(
    z_values: np.ndarray,
    alphabet_size: int,
    sax_behavior: str,
) -> np.ndarray:
    if sax_behavior == "gaussian":
        from scipy.stats import norm

        probs = np.linspace(0.0, 1.0, alphabet_size + 1)[1:-1]
        return np.asarray(norm.ppf(probs), dtype=np.float64)
    if sax_behavior == "quantile":
        probs = np.linspace(0.0, 100.0, alphabet_size + 1)[1:-1]
        return np.asarray(np.percentile(z_values, probs), dtype=np.float64)
    raise ValueError(f"Unknown sax_behavior={sax_behavior!r}, expected 'quantile' or 'gaussian'")


def _sax_indices_from_window(
    window_values: np.ndarray,
    alphabet_size: int,
    sax_behavior: str,
) -> np.ndarray:
    z_values = _z_norm(window_values)
    breakpoints = _breakpoints_from_window(z_values, alphabet_size, sax_behavior)
    return np.digitize(z_values, breakpoints).astype(np.int32)


def estimate_markov_transition(
    sax_indices: np.ndarray,
    alphabet_size: int,
    laplace_alpha: float = 1e-6,
) -> np.ndarray:
    """
    Estimate row-stochastic transition matrix P(next|current) directly from counts.
    """
    k = int(alphabet_size)
    counts = np.full((k, k), float(laplace_alpha), dtype=np.float64)
    if sax_indices.size >= 2:
        src = sax_indices[:-1]
        dst = sax_indices[1:]
        np.add.at(counts, (src, dst), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(row_sums, 1e-12)


def fisher_trace_from_entropy_window(
    entropy_window: np.ndarray,
    alphabet_size: int = 7,
    sax_behavior: str = "quantile",
    laplace_alpha: float = 1e-6,
    fisher_epsilon: float = 1e-10,
) -> float:
    sax_indices = _sax_indices_from_window(
        window_values=np.asarray(entropy_window, dtype=np.float64),
        alphabet_size=alphabet_size,
        sax_behavior=sax_behavior,
    )
    transition = estimate_markov_transition(
        sax_indices=sax_indices,
        alphabet_size=alphabet_size,
        laplace_alpha=laplace_alpha,
    )
    fisher = compute_fisher_matrix(transition, epsilon=float(fisher_epsilon))
    return float(fisher_scalar(fisher))


@dataclass
class SlidingFisherTrace:
    window_size: int = 128
    alphabet_size: int = 7
    sax_behavior: str = "quantile"
    laplace_alpha: float = 1e-6
    fisher_epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if self.window_size < 4:
            raise ValueError("window_size must be >= 4")
        self._window: Deque[float] = deque(maxlen=self.window_size)

    def update(self, entropy_value: float) -> float | None:
        """
        Push one entropy value and return current Fisher trace when window is full.
        """
        self._window.append(float(entropy_value))
        if len(self._window) < self.window_size:
            return None
        window = np.asarray(self._window, dtype=np.float64)
        return fisher_trace_from_entropy_window(
            entropy_window=window,
            alphabet_size=self.alphabet_size,
            sax_behavior=self.sax_behavior,
            laplace_alpha=self.laplace_alpha,
            fisher_epsilon=self.fisher_epsilon,
        )

    @property
    def is_ready(self) -> bool:
        return len(self._window) >= self.window_size

    @property
    def size(self) -> int:
        return len(self._window)


def rolling_fisher_trace_series(
    entropy_series: np.ndarray,
    window_size: int = 128,
    alphabet_size: int = 7,
    sax_behavior: str = "quantile",
    laplace_alpha: float = 1e-6,
    fisher_epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Batch helper for offline checks. Returns array of same length as input with NaN warmup.
    """
    entropy_series = np.asarray(entropy_series, dtype=np.float64)
    out = np.full(entropy_series.shape[0], np.nan, dtype=np.float64)
    tracker = SlidingFisherTrace(
        window_size=window_size,
        alphabet_size=alphabet_size,
        sax_behavior=sax_behavior,
        laplace_alpha=laplace_alpha,
        fisher_epsilon=fisher_epsilon,
    )
    for i, h in enumerate(entropy_series):
        tr = tracker.update(float(h))
        if tr is not None and not math.isnan(tr):
            out[i] = float(tr)
    return out
