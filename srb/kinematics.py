

"""Core semantic kinematics metrics for SRB.

This module implements the primary low-level metrics described in
``docs/Metrics_Definitions.md``:

- Semantic Velocity
- Semantic Entropy
- Surprisal and Surprisal Gradient
- NSM Divergence (Coherence Drift)
- Semantic Acceleration

These functions operate on token-level traces (``TokenStep``) and return
per-step metric series that can be composed by higher-level APIs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import math

from .types import TokenStep, Vector


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------


def _dot(a: Vector, b: Vector) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: Vector) -> float:
    return math.sqrt(float(sum(x * x for x in a)))


def _cosine_distance(a: Vector, b: Vector) -> float:
    """1 - cosine similarity.

    Falls back to 1.0 (maximal distance) if either vector has zero norm.
    """

    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        # Degenerate case: undefined cosine; treat as maximally distant.
        return 1.0
    return 1.0 - (_dot(a, b) / (na * nb))


def _euclidean_distance(a: Vector, b: Vector) -> float:
    return math.sqrt(float(sum((x - y) * (x - y) for x, y in zip(a, b))))


def _distance(a: Vector, b: Vector, *, use_cosine: bool) -> float:
    if use_cosine:
        return _cosine_distance(a, b)
    return _euclidean_distance(a, b)


# ---------------------------------------------------------------------------
# Semantic Velocity
# ---------------------------------------------------------------------------


def compute_semantic_velocity(
    steps: Sequence[TokenStep], *, use_cosine_distance: bool = True
) -> List[Optional[float]]:
    """Compute Semantic Velocity for each token step.

    Velocity is defined as the distance between consecutive embeddings in
    semantic space. The first step has no previous embedding and therefore
    returns ``None``.

    Parameters
    ----------
    steps:
        A sequence of ``TokenStep`` objects with ``embedding`` populated.
    use_cosine_distance:
        If True, use cosine distance (1 - cosine similarity). Otherwise,
        Euclidean (L2) distance is used.
    """

    velocities: List[Optional[float]] = []

    if not steps:
        return velocities

    velocities.append(None)  # v_0 is undefined

    for i in range(1, len(steps)):
        e = steps[i].embedding
        e_prev = steps[i - 1].embedding
        if e is None or e_prev is None:
            velocities.append(None)
            continue
        delta = _distance(e, e_prev, use_cosine=use_cosine_distance)
        velocities.append(float(delta))

    return velocities


# ---------------------------------------------------------------------------
# Semantic Entropy
# ---------------------------------------------------------------------------


def compute_semantic_entropy(steps: Sequence[TokenStep]) -> List[Optional[float]]:
    """Compute Semantic Entropy H_i for each token step.

    Uses the Shannon entropy of the next-token probability distribution
    provided in ``TokenStep.probs``. If probabilities are missing for a
    given step, the entropy is reported as ``None``.
    """

    entropies: List[Optional[float]] = []

    for step in steps:
        probs = step.probs
        if probs is None:
            entropies.append(None)
            continue

        H = 0.0
        for p in probs:
            if p > 0.0:
                H += -p * math.log(p)
        entropies.append(H)

    return entropies


# ---------------------------------------------------------------------------
# Surprisal and Surprisal Gradient
# ---------------------------------------------------------------------------


def compute_surprisal(steps: Sequence[TokenStep]) -> List[Optional[float]]:
    """Compute token-level surprisal S_i = -log P(t_i | context).

    If ``TokenStep.logprob`` is missing, the surprisal is reported as
    ``None``.
    """

    surprisals: List[Optional[float]] = []

    for step in steps:
        if step.logprob is None:
            surprisals.append(None)
        else:
            surprisals.append(-step.logprob)

    return surprisals


def compute_surprisal_gradient(
    surprisals: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Compute the Surprisal Gradient dS/dt.

    This is the discrete temporal derivative of the surprisal sequence:

    .. math::

        dS/dt_i = S_i - S_{i-1}

    The first step has no previous value and therefore returns ``None``.
    If either of the required surprisal values is ``None``, the gradient
    at that position is also ``None``.
    """

    gradients: List[Optional[float]] = []

    if not surprisals:
        return gradients

    gradients.append(None)  # gradient at step 0 is undefined

    for i in range(1, len(surprisals)):
        prev = surprisals[i - 1]
        curr = surprisals[i]
        if prev is None or curr is None:
            gradients.append(None)
        else:
            gradients.append(curr - prev)

    return gradients


# ---------------------------------------------------------------------------
# NSM Divergence (Coherence Drift)
# ---------------------------------------------------------------------------


def compute_nsm_reference_embeddings(
    steps: Sequence[TokenStep],
) -> List[Optional[Vector]]:
    """Compute a simple reference embedding baseline for NSM Divergence.

    In the full SRB framework, the reference embedding can be derived from
    top-k next-token candidates, normative semantic clusters, or domain-
    specific baselines. As a practical and model-agnostic default, we use a
    rolling exponential mean of past embeddings.

    Steps without embeddings yield ``None`` references.
    """

    refs: List[Optional[Vector]] = []
    running: Optional[Vector] = None

    for step in steps:
        e = step.embedding
        if e is None:
            refs.append(running)
            continue

        if running is None:
            running = list(e)
        else:
            # Exponential moving average with a fixed smoothing factor.
            alpha = 0.1
            running = [
                (1.0 - alpha) * r + alpha * v
                for r, v in zip(running, e)
            ]

        refs.append(list(running))

    return refs


def compute_nsm_divergence(
    steps: Sequence[TokenStep],
    reference_embeddings: Sequence[Optional[Vector]],
    *,
    use_cosine_distance: bool = True,
) -> List[Optional[float]]:
    """Compute NSM Divergence (Coherence Drift) for each token step.

    NSM Divergence is defined as the distance between the current embedding
    and a reference embedding representing the expected semantic direction.

    If either the current embedding or its reference is missing, the
    divergence is reported as ``None``.
    """

    divergences: List[Optional[float]] = []

    for step, ref in zip(steps, reference_embeddings):
        e = step.embedding
        if e is None or ref is None:
            divergences.append(None)
            continue
        d = _distance(e, ref, use_cosine=use_cosine_distance)
        divergences.append(float(d))

    return divergences


# ---------------------------------------------------------------------------
# Semantic Acceleration
# ---------------------------------------------------------------------------


def compute_semantic_acceleration(
    velocities: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Compute Semantic Acceleration from a series of velocities.

    Acceleration is the discrete derivative of velocity:

    .. math::

        a_i = v_i - v_{i-1}

    The first step has no previous velocity and therefore returns ``None``.
    If either of the required velocities is ``None``, the acceleration at
    that position is also ``None``.
    """

    accelerations: List[Optional[float]] = []

    if not velocities:
        return accelerations

    accelerations.append(None)  # a_0 is undefined

    for i in range(1, len(velocities)):
        v_prev = velocities[i - 1]
        v_curr = velocities[i]
        if v_prev is None or v_curr is None:
            accelerations.append(None)
        else:
            accelerations.append(v_curr - v_prev)

    return accelerations
