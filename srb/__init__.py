

"""Semantic Resonance Budget (SRB) core library.

This package provides model-agnostic tools for measuring the internal
semantic dynamics of language models during inference.

The main public entry points are:

- :func:`analyze_sequence` — run SRB metrics over a sequence of TokenStep
- :class:`SRBConfig` — configure which metrics are computed
- :class:`TokenStep` — token-level trace input structure
- :class:`SRBSequenceMetrics` — container for per-step and aggregate output

Additional functionality is available via submodules:

- ``srb.kinematics`` — low-level semantic kinematics metrics
- ``srb.adapters`` — model adapters (OpenAI, etc.)
"""

from __future__ import annotations

from .types import SRBConfig, TokenStep, SRBStepMetrics, SRBSequenceMetrics
from .api import analyze_sequence

__all__ = [
    "SRBConfig",
    "TokenStep",
    "SRBStepMetrics",
    "SRBSequenceMetrics",
    "analyze_sequence",
]

# Keep the package version in sync with pyproject.toml
__version__ = "0.1.0"