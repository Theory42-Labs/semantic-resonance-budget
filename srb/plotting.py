"""
srb/plotting.py

Lightweight plotting utilities for SRB traces.
- Uses matplotlib only (no seaborn).
- Each plot uses its own figure (no subplots).
- Does not set any explicit colors or styles.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _read_trace_csv(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Read a SRB trace.csv into column arrays.
    Expected columns include at least:
      step, token, text_so_far, entropy, coherence, resonance, d_resonance, temperature, top_p
    Missing columns are tolerated and returned as empty arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace CSV not found: {path}")

    cols: Dict[str, List[Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in cols:
                    cols[k] = []
                cols[k].append(v)

    # Convert known numeric columns to float arrays where possible
    def to_float_array(key: str) -> np.ndarray:
        if key not in cols:
            return np.array([], dtype=float)
        out: List[float] = []
        for v in cols[key]:
            try:
                out.append(float(v))
            except Exception:
                # non-numeric or missing
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    out = {
        "step": to_float_array("step"),
        "entropy": to_float_array("entropy"),
        "coherence": to_float_array("coherence"),
        "resonance": to_float_array("resonance"),
        "d_resonance": to_float_array("d_resonance"),
    }
    return out


def plot_resonance(trace_csv: str | Path, out_png: str | Path, title: str | None = None) -> None:
    """
    Plot resonance (R_t) across decoding steps.

    Parameters
    ----------
    trace_csv : Path to a trace.csv file.
    out_png   : Output path for the PNG image.
    title     : Optional title for the plot.
    """
    data = _read_trace_csv(trace_csv)
    steps = data["step"]
    R = data["resonance"]

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(steps, R, linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Resonance (R_t)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_entropy_coherence(trace_csv: str | Path, out_png: str | Path, title: str | None = None) -> None:
    """
    Plot entropy (H_t) and coherence (C_t) across decoding steps
    on two separate figures (one per chart), saved as:
      - out_png for entropy
      - sibling file with suffix '_coherence.png' for coherence
    This maintains the constraint: one chart per figure.
    """
    data = _read_trace_csv(trace_csv)
    steps = data["step"]
    H = data["entropy"]
    C = data["coherence"]

    out_png = Path(out_png)

    # Entropy plot
    fig_H = plt.figure()
    axH = fig_H.gca()
    axH.plot(steps, H, linewidth=2.0)
    axH.set_xlabel("Step")
    axH.set_ylabel("Entropy (H_t)")
    if title:
        axH.set_title(f"{title} — Entropy")
    fig_H.tight_layout()
    fig_H.savefig(out_png, dpi=180)
    plt.close(fig_H)

    # Coherence plot (save next to entropy file)
    coh_png = out_png.with_name(out_png.stem + "_coherence.png")
    fig_C = plt.figure()
    axC = fig_C.gca()
    axC.plot(steps, C, linewidth=2.0)
    axC.set_xlabel("Step")
    axC.set_ylabel("Coherence (C_t)")
    if title:
        axC.set_title(f"{title} — Coherence")
    fig_C.tight_layout()
    fig_C.savefig(coh_png, dpi=180)
    plt.close(fig_C)


def plot_from_arrays(
    steps: np.ndarray,
    values: np.ndarray,
    out_png: str | Path,
    y_label: str,
    title: str | None = None,
) -> None:
    """
    Generic single-series plotter from arrays.
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(steps, values, linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
