

"""Public API for running SRB analysis on token-level traces.

This module exposes a high-level entry point, ``analyze_sequence``, which
accepts a list of ``TokenStep`` objects and returns structured SRB metrics
for each step along with aggregate statistics.

It composes the core semantic kinematics implemented in ``srb.kinematics``
using the configuration defined in ``srb.types.SRBConfig``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .types import SRBConfig, TokenStep, SRBStepMetrics, SRBSequenceMetrics
from . import kinematics


# ---------------------------------------------------------------------------
# Helper functions for aggregate statistics
# ---------------------------------------------------------------------------


def _filter_valid(values: Sequence[Optional[float]]) -> List[float]:
    """Return a list of non-None values from a sequence of optionals."""

    return [v for v in values if v is not None]


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = _filter_valid(values)
    if not valid:
        return None
    return sum(valid) / float(len(valid))


def _max(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = _filter_valid(values)
    if not valid:
        return None
    return max(valid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_sequence(
    steps: Sequence[TokenStep],
    config: Optional[SRBConfig] = None,
) -> SRBSequenceMetrics:
    """Run SRB analysis over a generated token sequence.

    Parameters
    ----------
    steps:
        A sequence of ``TokenStep`` objects representing token-level
        generation traces. At minimum, embeddings should be provided for
        metrics that depend on them (e.g., velocity, NSM divergence).
    config:
        Optional :class:`SRBConfig` controlling which metrics are computed
        and how certain distances are interpreted.

    Returns
    -------
    SRBSequenceMetrics
        An object containing per-step SRB metrics and aggregate statistics
        over the entire sequence.
    """

    if config is None:
        config = SRBConfig()

    n = len(steps)
    step_metrics: List[SRBStepMetrics] = [SRBStepMetrics(index=i) for i in range(n)]
    aggregates: Dict[str, float] = {}

    # Pre-computed series that may be reused across metrics.
    velocities: Optional[List[Optional[float]]] = None
    entropies: Optional[List[Optional[float]]] = None
    surprisals: Optional[List[Optional[float]]] = None
    surprisal_gradients: Optional[List[Optional[float]]] = None
    nsm_refs: Optional[List[Optional[kinematics.Vector]]] = None  # type: ignore[attr-defined]
    nsm_divs: Optional[List[Optional[float]]] = None
    accelerations: Optional[List[Optional[float]]] = None

    # ------------------------------------------------------------------
    # Semantic Velocity
    # ------------------------------------------------------------------

    if "semantic_velocity" in config.metrics or "semantic_acceleration" in config.metrics:
        velocities = kinematics.compute_semantic_velocity(
            steps, use_cosine_distance=config.use_cosine_distance
        )
        # Populate step metrics
        for m, v in zip(step_metrics, velocities):
            m.semantic_velocity = v

        mv = _mean(velocities)
        if mv is not None:
            aggregates["mean_velocity"] = mv
        mx = _max(velocities)
        if mx is not None:
            aggregates["max_velocity"] = mx

    # ------------------------------------------------------------------
    # Semantic Entropy
    # ------------------------------------------------------------------

    if "semantic_entropy" in config.metrics:
        entropies = kinematics.compute_semantic_entropy(steps)
        for m, H in zip(step_metrics, entropies):
            m.semantic_entropy = H

        mH = _mean(entropies)
        if mH is not None:
            aggregates["mean_entropy"] = mH
        MH = _max(entropies)
        if MH is not None:
            aggregates["max_entropy"] = MH

    # ------------------------------------------------------------------
    # Surprisal + Surprisal Gradient
    # ------------------------------------------------------------------

    if "surprisal_gradient" in config.metrics:
        surprisals = kinematics.compute_surprisal(steps)
        surprisal_gradients = kinematics.compute_surprisal_gradient(surprisals)

        for m, s, ds in zip(step_metrics, surprisals, surprisal_gradients):
            m.surprisal = s
            m.surprisal_gradient = ds

        mg = _max(surprisal_gradients)
        if mg is not None:
            aggregates["max_surprisal_gradient"] = mg

    # ------------------------------------------------------------------
    # NSM Divergence (Coherence Drift)
    # ------------------------------------------------------------------

    if "nsm_divergence" in config.metrics:
        nsm_refs = kinematics.compute_nsm_reference_embeddings(steps)
        nsm_divs = kinematics.compute_nsm_divergence(
            steps,
            nsm_refs,
            use_cosine_distance=config.use_cosine_distance,
        )

        for m, d in zip(step_metrics, nsm_divs):
            m.nsm_divergence = d

        md = _mean(nsm_divs)
        if md is not None:
            aggregates["mean_nsm_divergence"] = md
        Md = _max(nsm_divs)
        if Md is not None:
            aggregates["max_nsm_divergence"] = Md

    # ------------------------------------------------------------------
    # Semantic Acceleration
    # ------------------------------------------------------------------

    if "semantic_acceleration" in config.metrics and config.compute_acceleration:
        # Ensure we have velocities to differentiate.
        if velocities is None:
            velocities = kinematics.compute_semantic_velocity(
                steps, use_cosine_distance=config.use_cosine_distance
            )
            for m, v in zip(step_metrics, velocities):
                # Only fill if not already set.
                if m.semantic_velocity is None:
                    m.semantic_velocity = v

        accelerations = kinematics.compute_semantic_acceleration(velocities)
        for m, a in zip(step_metrics, accelerations):
            m.semantic_acceleration = a

        ma = _max(accelerations)
        if ma is not None:
            aggregates["max_acceleration"] = ma

    return SRBSequenceMetrics(steps=step_metrics, aggregates=aggregates)