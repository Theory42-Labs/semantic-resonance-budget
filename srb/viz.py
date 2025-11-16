"""
Visualization helpers for SRB metrics.

These functions provide simple, opinionated plotting utilities for
exploring SRB metric and resonance series. They import matplotlib lazily
so that the core SRB library does not depend on it at runtime.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .types import SRBSequenceMetrics
from .resonance import ResonanceStep


def _require_matplotlib():
    """Import and return matplotlib.pyplot, or raise a helpful error.

    Matplotlib is treated as an optional dependency. Users who do not
    need visualization are not required to install it.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "matplotlib is required for srb.viz functions. "
            "Install it with `pip install matplotlib`."
        ) from exc
    return plt


# ---------------------------------------------------------------------------
# Core metric plotting
# ---------------------------------------------------------------------------


def plot_metric_series(
    seq: SRBSequenceMetrics,
    metric_name: str,
    *,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot a single per-step metric from an SRBSequenceMetrics object.

    Parameters
    ----------
    seq:
        SRBSequenceMetrics instance returned by ``analyze_sequence``.
    metric_name:
        Name of the ``SRBStepMetrics`` field to plot
        (e.g., ``"semantic_velocity"``, ``"semantic_entropy"``).
    title:
        Optional plot title. If not provided, one is constructed.
    show:
        If True, call ``plt.show()`` at the end.
    """

    plt = _require_matplotlib()

    xs = []
    ys = []
    for step in seq.steps:
        xs.append(step.index)
        val = getattr(step, metric_name, None)
        ys.append(val)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Step index")
    ax.set_ylabel(metric_name)
    if title is None:
        title = f"SRB metric: {metric_name}"
    ax.set_title(title)
    ax.grid(True)

    if show:
        plt.show()

    return fig, ax


def plot_multiple_metrics(
    seq: SRBSequenceMetrics,
    metric_names: Sequence[str],
    *,
    title: Optional[str] = None,
    show: bool = True,
):
    """Overlay multiple per-step metrics on a single plot.

    Only metrics that have at least one non-None value are plotted.

    Parameters
    ----------
    seq:
        SRBSequenceMetrics instance returned by ``analyze_sequence``.
    metric_names:
        Iterable of ``SRBStepMetrics`` attribute names to plot.
    title:
        Optional plot title. If not provided, one is constructed.
    show:
        If True, call ``plt.show()`` at the end.
    """

    plt = _require_matplotlib()

    xs = [step.index for step in seq.steps]

    fig, ax = plt.subplots()

    for name in metric_names:
        ys = [getattr(step, name, None) for step in seq.steps]
        # Skip metrics that are entirely None.
        if all(v is None for v in ys):
            continue
        ax.plot(xs, ys, marker="o", label=name)

    ax.set_xlabel("Step index")
    if title is None:
        title = "SRB metrics: " + ", ".join(metric_names)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# Resonance plotting
# ---------------------------------------------------------------------------


def plot_resonance_series(
    steps: Sequence[ResonanceStep],
    *,
    title: Optional[str] = None,
    collapse_idx: Optional[int] = None,
    show: bool = True,
):
    """Plot resonance, coherence, and normalized entropy over time.

    Parameters
    ----------
    steps:
        A sequence of :class:`ResonanceStep` objects, typically produced
        by ``build_resonance_from_sequence``.
    title:
        Optional plot title. If not provided, a default is used.
    collapse_idx:
        Optional token index at which resonance collapse was detected.
        If provided, a vertical line is drawn at this index.
    show:
        If True, call ``plt.show()`` at the end.
    """

    plt = _require_matplotlib()

    xs = [s.index for s in steps]
    resonance = [s.resonance for s in steps]
    coherence = [s.coherence for s in steps]
    entropy_norm = [s.entropy_norm for s in steps]

    fig, ax = plt.subplots()

    # Only plot lines that have some non-None values.
    if any(r is not None for r in resonance):
        ax.plot(xs, resonance, marker="o", label="resonance")
    if any(c is not None for c in coherence):
        ax.plot(xs, coherence, marker="o", label="coherence")
    if any(h is not None for h in entropy_norm):
        ax.plot(xs, entropy_norm, marker="o", label="entropy_norm")

    if collapse_idx is not None:
        ax.axvline(collapse_idx, linestyle="--", label="collapse")

    ax.set_xlabel("Step index")
    ax.set_ylabel("value")
    if title is None:
        title = "SRB resonance series"
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return fig, ax