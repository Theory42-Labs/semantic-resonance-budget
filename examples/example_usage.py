

"""Minimal example of running SRB on a synthetic token sequence.

This script demonstrates how to:

1. Construct a list of ``TokenStep`` objects with dummy data.
2. Run SRB's high-level ``analyze_sequence`` API.
3. Inspect per-step metrics and aggregate statistics.

In a real integration, the ``TokenStep`` objects would be created by a
model adapter (e.g., wrapping an OpenAI or Hugging Face generation call)
that records token text, logprobs, probabilities, and embeddings.
"""

from __future__ import annotations

import math
import random
from typing import List

from srb.types import TokenStep, SRBConfig
from srb.api import analyze_sequence


def _softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def build_synthetic_sequence(n_tokens: int = 8, embedding_dim: int = 4) -> List[TokenStep]:
    """Construct a simple synthetic token sequence for demonstration.

    This does *not* rely on any real model backend. It simply generates:
    - token strings ("tok_0", "tok_1", ...)
    - random logprob values
    - a small probability distribution over a fictitious vocabulary
    - low-dimensional embeddings with slight drift across steps
    """

    random.seed(42)

    steps: List[TokenStep] = []

    # Start from a random base embedding.
    base_embedding = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]

    for i in range(n_tokens):
        token = f"tok_{i}"

        # Simulate a logprob and a tiny vocabulary distribution.
        logits = [random.uniform(-2.0, 2.0) for _ in range(5)]
        probs = _softmax(logits)
        # Choose an index as the "generated" token probability.
        chosen_idx = random.randrange(len(probs))
        logprob = math.log(probs[chosen_idx])

        # Create a slightly drifted embedding from the base.
        drift = [random.uniform(-0.2, 0.2) for _ in range(embedding_dim)]
        embedding = [b + d for b, d in zip(base_embedding, drift)]
        # Update base_embedding for the next step to simulate a trajectory.
        base_embedding = embedding

        steps.append(
            TokenStep(
                index=i,
                token=token,
                logprob=logprob,
                probs=probs,
                embedding=embedding,
            )
        )

    return steps


def main() -> None:
    # Build a synthetic sequence of token steps.
    steps = build_synthetic_sequence()

    # Use default SRB configuration (all core metrics enabled).
    config = SRBConfig()

    # Run SRB analysis.
    result = analyze_sequence(steps, config=config)

    print("Per-step SRB metrics:\n")
    for s in result.steps:
        print(
            f"idx={s.index:2d} | token={steps[s.index].token:6s} | "
            f"v={s.semantic_velocity!r:>8} | H={s.semantic_entropy!r:>8} | "
            f"S={s.surprisal!r:>8} | dS={s.surprisal_gradient!r:>8} | "
            f"NSM={s.nsm_divergence!r:>8} | a={s.semantic_acceleration!r:>8}"
        )

    print("\nAggregate metrics:\n")
    for name, value in result.aggregates.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()