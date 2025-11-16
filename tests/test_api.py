

"""Basic tests for the SRB public API and kinematics.

These tests use synthetic ``TokenStep`` sequences to verify that:
- ``analyze_sequence`` runs without error on well-formed input.
- Core metrics are populated when the necessary fields are present.
- Metrics gracefully handle missing data (e.g., absent probs or logprobs).
"""

from __future__ import annotations

import math

from srb.types import TokenStep, SRBConfig
from srb.api import analyze_sequence
from srb import kinematics


def _make_simple_steps():
    """Construct a small deterministic sequence of TokenStep objects.

    The sequence is designed so that:
    - embeddings move linearly in one dimension (for simple velocity),
    - probabilities are valid distributions,
    - logprobs align with the chosen index.
    """

    steps = []

    # Simple 2D embeddings along a line: (0, 0), (1, 0), (2, 0), (3, 0)
    embeddings = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ]

    # Use the same probability distribution for simplicity.
    probs = [0.7, 0.3]
    chosen_idx = 0
    logprob = math.log(probs[chosen_idx])

    for i, emb in enumerate(embeddings):
        steps.append(
            TokenStep(
                index=i,
                token=f"tok_{i}",
                logprob=logprob,
                probs=probs,
                embedding=emb,
            )
        )

    return steps


def test_analyze_sequence_basic_metrics_populated():
    """Core metrics should be populated for a simple, well-formed sequence."""

    steps = _make_simple_steps()
    config = SRBConfig()  # default: all core metrics enabled

    result = analyze_sequence(steps, config=config)

    # We expect one SRBStepMetrics per TokenStep.
    assert len(result.steps) == len(steps)

    # Check per-step fields on a few positions.
    first = result.steps[0]
    second = result.steps[1]

    # Index alignment
    assert first.index == 0
    assert second.index == 1

    # Velocity should be None for the first step and positive for the second.
    assert first.semantic_velocity is None
    assert second.semantic_velocity is not None
    assert second.semantic_velocity > 0.0

    # Entropy should be defined for all steps (we provided probs).
    for s in result.steps:
        assert s.semantic_entropy is not None
        assert s.semantic_entropy >= 0.0

    # Surprisal should be defined and consistent with logprob.
    expected_surprisal = -steps[0].logprob  # type: ignore[arg-type]
    for s in result.steps:
        assert s.surprisal is not None
        assert math.isclose(s.surprisal, expected_surprisal, rel_tol=1e-6)

    # Surprisal gradient should be zero after the first undefined element
    # because the surprisal is constant across steps.
    assert result.steps[0].surprisal_gradient is None
    for s in result.steps[1:]:
        assert s.surprisal_gradient is not None
        assert math.isclose(s.surprisal_gradient, 0.0, abs_tol=1e-6)

    # NSM divergence and acceleration should be computed where possible.
    # We don't assert exact values, but we expect non-None entries after
    # sufficient context exists.
    assert result.steps[0].semantic_acceleration is None
    assert result.steps[0].nsm_divergence is not None

    # Aggregates should contain some summary statistics.
    assert "mean_velocity" in result.aggregates
    assert "max_velocity" in result.aggregates
    assert "mean_entropy" in result.aggregates
    assert "max_entropy" in result.aggregates
    assert "max_surprisal_gradient" in result.aggregates
    assert "mean_nsm_divergence" in result.aggregates
    assert "max_nsm_divergence" in result.aggregates
    assert "max_acceleration" in result.aggregates


def test_kinematics_handle_missing_fields_gracefully():
    """Kinematic metrics should return None when required data is missing."""

    # Step 0: missing embedding and probs/logprob
    s0 = TokenStep(index=0, token="tok_0")

    # Step 1: has embedding but no probs/logprob
    s1 = TokenStep(index=1, token="tok_1", embedding=[1.0, 0.0])

    steps = [s0, s1]

    # Velocity requires embeddings; first step should be None, second also
    # None because previous embedding is missing.
    v = kinematics.compute_semantic_velocity(steps)
    assert v == [None, None]

    # Entropy requires probs; both should be None.
    H = kinematics.compute_semantic_entropy(steps)
    assert H == [None, None]

    # Surprisal requires logprob; both should be None.
    S = kinematics.compute_surprisal(steps)
    assert S == [None, None]

    # Surprisal gradient should also be None everywhere.
    dS = kinematics.compute_surprisal_gradient(S)
    assert dS == [None, None]

    # NSM reference embeddings: first step has no embedding, second adopts
    # the first available embedding.
    refs = kinematics.compute_nsm_reference_embeddings(steps)
    assert refs[0] is None
    assert refs[1] is not None

    # NSM divergence: both should be None because at least one of the
    # operands (embedding or reference) is missing.
    D = kinematics.compute_nsm_divergence(steps, refs)
    assert D == [None, None]

    # Acceleration requires velocities; with all None, all accelerations
    # should also be None.
    a = kinematics.compute_semantic_acceleration(v)
    assert a == [None, None]