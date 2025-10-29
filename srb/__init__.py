"""
Semantic Resonance Budgeting (SRB) — public API.

This package exposes:
- Core runtime (model loader and SRB dynamic run)
- Metrics helpers (entropy/coherence → resonance; collapse detection; summaries)
- Tracing utilities (CSV/JSON logging)
- Plotting helpers (resonance / entropy / coherence)
"""

from __future__ import annotations

__version__ = "1.0.0"

def get_version() -> str:
    """Return the SRB package version."""
    return __version__

# Core runtime
from .srb_wrapper import (
    load_model,
    run_static,
    run_srb_dynamic,
    DEFAULT_TOPK,
)

# Metrics
from .metrics import (
    StepSignals,
    normalize_entropy,
    resonance_amplitude,
    step_signals,
    collapse_rate,
    completion_detected,
    resonance_summary,
    aggregate_run_metrics,
    to_arrays,
)

# Tracing
from .tracing import (
    TraceLogger,
    load_trace,
    save_summary,
)

# Plotting
from .plotting import (
    plot_resonance,
    plot_entropy_coherence,
    plot_from_arrays,
)

__all__ = [
    "__version__",
    "get_version",
    # core
    "load_model",
    "run_static",
    "run_srb_dynamic",
    "DEFAULT_TOPK",
    # metrics
    "StepSignals",
    "normalize_entropy",
    "resonance_amplitude",
    "step_signals",
    "collapse_rate",
    "completion_detected",
    "resonance_summary",
    "aggregate_run_metrics",
    "to_arrays",
    # tracing
    "TraceLogger",
    "load_trace",
    "save_summary",
    # plotting
    "plot_resonance",
    "plot_entropy_coherence",
    "plot_from_arrays",
]