"""
Resonance metrics built on top of SRB kinematics.

This module provides helpers for:
- Normalizing entropy into [0, 1]
- Computing resonance amplitude from entropy + coherence
- Estimating a simple “collapse” step from a resonance series
- Building resonance views directly from SRBSequenceMetrics outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .types import SRBSequenceMetrics
from .utils import normalize_series_minmax, rolling_window_std


@dataclass
class ResonanceStep:
    """Resonance-oriented view of a single step.

    This is a higher-level representation than SRBStepMetrics, focused on
    entropy normalization and resonance amplitude.
    """

    index: int

    entropy: Optional[float] = None
    entropy_norm: Optional[float] = None
    coherence: Optional[float] = None
    resonance: Optional[float] = None
    resonance_delta: Optional[float] = None

    extras: Dict[str, object] = field(default_factory=dict)


def normalize_entropy(
    entropies: Sequence[Optional[float]],
    max_entropy: Optional[float] = None,
) -> List[Optional[float]]:
    """Normalize raw entropies into [0, 1].

    If `max_entropy` is not provided, it is inferred as the maximum
    non-None entropy in the sequence. Values are mapped as:

        H_norm = H / max_entropy

    If max_entropy <= 0 or no valid entropies exist, all non-None
    entries are mapped to 0.0.
    """
    non_null = [h for h in entropies if h is not None]
    if not non_null:
        return [None for _ in entropies]

    if max_entropy is None:
        max_entropy = max(non_null)

    if max_entropy <= 0.0:
        return [0.0 if h is not None else None for h in entropies]

    out: List[Optional[float]] = []
    for h in entropies:
        if h is None:
            out.append(None)
        else:
            out.append(h / max_entropy)
    return out


def compute_resonance_amplitude(
    entropy_norm: Sequence[Optional[float]],
    coherence: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Compute resonance amplitude R_t from normalized entropy + coherence.

    A simple and interpretable definition:

        R_t = (1 - H_norm_t) * max(C_t, 0)

    where:
        - H_norm_t ∈ [0, 1] is normalized entropy
        - C_t is a coherence signal (e.g., cosine similarity / composite)
    """
    if len(entropy_norm) != len(coherence):
        raise ValueError("entropy_norm and coherence must have the same length")

    out: List[Optional[float]] = []
    for Hn, C in zip(entropy_norm, coherence):
        if Hn is None or C is None:
            out.append(None)
            continue
        C_pos = max(C, 0.0)
        out.append((1.0 - Hn) * C_pos)
    return out


def compute_resonance_series(
    entropies: Sequence[Optional[float]],
    coherence: Sequence[Optional[float]],
    max_entropy: Optional[float] = None,
) -> List[ResonanceStep]:
    """Build a ResonanceStep series from entropy + coherence sequences."""
    if len(entropies) != len(coherence):
        raise ValueError("entropies and coherence must have the same length")

    Hn = normalize_entropy(entropies, max_entropy=max_entropy)
    R = compute_resonance_amplitude(Hn, coherence)

    # Compute discrete derivative of resonance
    dR: List[Optional[float]] = []
    prev: Optional[float] = None
    for r in R:
        if prev is None or r is None:
            dR.append(None)
        else:
            dR.append(r - prev)
        prev = r

    steps: List[ResonanceStep] = []
    for i, (H, H_norm, C, res, dr) in enumerate(zip(entropies, Hn, coherence, R, dR)):
        steps.append(
            ResonanceStep(
                index=i,
                entropy=H,
                entropy_norm=H_norm,
                coherence=C,
                resonance=res,
                resonance_delta=dr,
            )
        )
    return steps


def coherence_from_nsm_divergence(
    nsm_divergence: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Derive a simple coherence signal from NSM Divergence.

    NSM Divergence measures distance from a normative semantic baseline.
    As a heuristic, we invert a min–max normalized divergence series so
    that:

        high divergence  -> low coherence
        low divergence   -> high coherence

    Steps where divergence is ``None`` yield coherence = ``None``.
    """

    # Normalize divergence into [0, 1]. Higher values mean more drift.
    norm = normalize_series_minmax(nsm_divergence)

    coherence: List[Optional[float]] = []
    for d in norm:
        if d is None:
            coherence.append(None)
        else:
            coherence.append(1.0 - d)
    return coherence


def build_resonance_from_sequence(
    seq: SRBSequenceMetrics,
    *,
    max_entropy: Optional[float] = None,
) -> List[ResonanceStep]:
    """Construct a resonance series from an SRBSequenceMetrics object.

    This helper interprets the existing SRB metrics as follows:

    - ``entropy`` comes from ``SRBStepMetrics.semantic_entropy``.
    - ``coherence`` is derived from ``SRBStepMetrics.nsm_divergence``
      using :func:`coherence_from_nsm_divergence`.

    The resulting :class:`ResonanceStep` list can be used for further
    analysis (e.g., collapse estimation) or visualization.
    """

    entropies: List[Optional[float]] = []
    divergences: List[Optional[float]] = []

    for step in seq.steps:
        entropies.append(step.semantic_entropy)
        divergences.append(step.nsm_divergence)

    coherence = coherence_from_nsm_divergence(divergences)
    return compute_resonance_series(entropies, coherence, max_entropy=max_entropy)


def estimate_collapse_step(
    resonance: Sequence[Optional[float]],
    *,
    window: int = 5,
    eps: float = 1e-3,
) -> Optional[int]:
    """Estimate the first step where resonance has “collapsed”.

    A simple heuristic: compute the rolling standard deviation over the
    non-None resonance values, and return the first index where the
    std-dev over the last `window` steps falls below `eps`.

    Returns:
        The index of the first collapse step, or None if no collapse
        is detected.
    """
    # Filter out None for the rolling computation, but we need to keep
    # alignment with original indices.
    numeric: List[float] = []
    idx_map: List[int] = []

    for i, r in enumerate(resonance):
        if r is not None:
            idx_map.append(i)
            numeric.append(r)

    if len(numeric) < window:
        return None

    stds = rolling_window_std(numeric, window=window)
    for local_idx, std in enumerate(stds):
        if std < eps and local_idx >= window - 1:
            return idx_map[local_idx]

    return None