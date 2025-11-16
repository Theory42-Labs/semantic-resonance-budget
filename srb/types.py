"""Core data types for the Semantic Resonance Budget (SRB) library.

This module defines the structured types used throughout the SRB API,
including configuration, token-level traces, and per-sequence metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A simple alias for embedding-like vectors. We intentionally avoid depending
# on a specific numeric library (e.g., numpy) at the type level so that SRB
# can be used in a variety of environments.
Vector = Sequence[float]


# Names of metrics that can be requested via SRBConfig.
MetricName = Literal[
    "semantic_velocity",
    "semantic_entropy",
    "surprisal_gradient",
    "nsm_divergence",
    "semantic_acceleration",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SRBConfig:
    """Configuration for SRB analysis.

    This controls which metrics are computed and how certain derived
    quantities (e.g., NSM reference baselines) are parameterized.
    """

    metrics: List[MetricName] = field(default_factory=list)
    """List of metrics to compute. If empty, a sensible default set is used."""

    top_k_for_nsm: int = 5
    """Number of top-k candidates to use when building NSM reference baselines."""

    use_cosine_distance: bool = True
    """Whether to interpret Vector distances as cosine-derived rather than raw L2.

    This flag is advisory; concrete metric implementations may respect or
    ignore it depending on how embeddings are provided.
    """

    compute_acceleration: bool = True
    """Whether to compute semantic acceleration when velocity is available."""

    def __post_init__(self) -> None:
        # If no metrics were explicitly requested, enable the core set.
        if not self.metrics:
            self.metrics = [
                "semantic_velocity",
                "semantic_entropy",
                "surprisal_gradient",
                "nsm_divergence",
                "semantic_acceleration",
            ]


# ---------------------------------------------------------------------------
# Token-level traces
# ---------------------------------------------------------------------------

@dataclass
class TokenStep:
    """A single token-generation step emitted by a model.

    This is the fundamental trace unit SRB operates on. It captures the
    generated token, its probability information, and its embedding in
    semantic space.
    """

    index: int
    """Position of this token in the generated sequence (0-based)."""

    token: str
    """The text of the generated token (or subword)."""

    logprob: Optional[float] = None
    """Log-probability of the generated token under the model, if available."""

    probs: Optional[Sequence[float]] = None
    """Full or truncated next-token probability distribution, if available.

    This may be the complete vocabulary distribution or a top-k slice,
    depending on the adapter used.
    """

    embedding: Optional[Vector] = None
    """Embedding representation of the token or its contextual state.

    Adapters are free to choose whether this is a token embedding, a
    hidden-state embedding, or a pooled representation, as long as it is
    used consistently within a sequence.
    """

    extras: Dict[str, object] = field(default_factory=dict)
    """Arbitrary additional fields supplied by adapters (e.g., timing info)."""


# ---------------------------------------------------------------------------
# Per-step and per-sequence SRB metrics
# ---------------------------------------------------------------------------

@dataclass
class SRBStepMetrics:
    """SRB metrics computed for a single token-generation step."""

    index: int

    semantic_velocity: Optional[float] = None
    semantic_entropy: Optional[float] = None

    surprisal: Optional[float] = None
    surprisal_gradient: Optional[float] = None

    nsm_divergence: Optional[float] = None
    semantic_acceleration: Optional[float] = None

    # Room for future or custom metrics without changing the schema.
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass
class SRBSequenceMetrics:
    """Collection of SRB metrics for an entire generated sequence.

    ``steps`` contains per-token metrics, while ``aggregates`` stores
    sequence-level statistics (e.g., mean velocity, max entropy, etc.).
    """

    steps: List[SRBStepMetrics]
    aggregates: Dict[str, float] = field(default_factory=dict)
