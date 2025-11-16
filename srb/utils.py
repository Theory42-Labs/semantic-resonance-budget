"""
Utility helpers for the Semantic Resonance Budget (SRB) library.

This module contains small, broadly useful helpers that are shared across
SRB components. The goal is to keep these utilities lightweight and
pure-Python, without introducing heavy dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import math


def safe_shannon_entropy(probs: Sequence[float]) -> float:
    """Compute Shannon entropy for a probability distribution.

    This helper gracefully skips zero-probability entries and assumes
    that `probs` are non-negative and sum to approximately 1.0.
    """
    H = 0.0
    for p in probs:
        if p > 0.0:
            H += -p * math.log(p)
    return H


def normalize_series_minmax(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Apply simple min–max normalization to a numeric series.

    None values are preserved as None. If all non-None values are
    identical, they are mapped to 0.0.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return [None for _ in values]

    v_min = min(non_null)
    v_max = max(non_null)

    if v_max == v_min:
        # Degenerate case: constant series.
        return [0.0 if v is not None else None for v in values]

    span = v_max - v_min
    out: List[Optional[float]] = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append((v - v_min) / span)
    return out


def rolling_window_std(values: Sequence[float], window: int) -> List[float]:
    """Compute a rolling standard deviation over a fixed window.

    The result has len(values) entries. For the first `window - 1`
    positions where a full window is not yet available, the value is
    computed over the available prefix.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        if not segment:
            out.append(0.0)
            continue
        m = sum(segment) / float(len(segment))
        var = sum((x - m) ** 2 for x in segment) / float(len(segment))
        out.append(math.sqrt(var))
    return out