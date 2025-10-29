# srb/metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class StepSignals:
    """Per-step telemetry captured during SRB decoding."""
    step: int
    token: str
    text_so_far: str
    entropy: float          # H_t (raw, unnormalized)
    entropy_norm: float     # H_t normalized to [0,1] by log(k)
    coherence: float        # C_t in [-1,1] (cosine similarity)
    resonance: float        # R_t in [0,1]  (default: (1 - Hn) * C+)
    d_resonance: float      # ΔR_t

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep floats concise if downstream writes CSV
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = float(v)
        return d

# -----------------------------
# Core computations
# -----------------------------

def normalize_entropy(H: float, top_k: int) -> float:
    """
    Normalize Shannon entropy by log(k) so values land in [0,1].
    Falls back to [0,1] clamp if k is degenerate.
    """
    denom = max(1e-12, math.log(max(2, top_k)))
    x = H / denom
    return max(0.0, min(1.0, x))

def resonance_amplitude(entropy_norm: float, coherence: float, alpha: float = 1.0) -> float:
    """
    Default resonance definition:
        R_t = (1 - Hn) * C+
    where Hn is normalized entropy in [0,1] and C+ is coherence clamped to [0,1].
    """
    C_pos = max(0.0, min(1.0, coherence))  # clamp cosine to [0,1]
    R = (1.0 - entropy_norm) * C_pos
    R = max(0.0, min(1.0, R * alpha))
    return float(R)

def step_signals(
    step_idx: int,
    token_str: str,
    text_so_far: str,
    entropy_raw: float,
    entropy_topk: int,
    coherence: float,
    prev_R: Optional[float],
    alpha: float = 1.0,
) -> StepSignals:
    """
    Convenience helper to compute normalized entropy, resonance, and ΔR_t.
    """
    Hn = normalize_entropy(entropy_raw, entropy_topk)
    R_t = resonance_amplitude(Hn, coherence, alpha=alpha)
    dR_t = 0.0 if prev_R is None else (R_t - prev_R)
    return StepSignals(
        step=step_idx,
        token=token_str,
        text_so_far=text_so_far,
        entropy=float(entropy_raw),
        entropy_norm=float(Hn),
        coherence=float(coherence),
        resonance=float(R_t),
        d_resonance=float(dR_t),
    )

# -----------------------------
# Stability & completion metrics
# -----------------------------

def collapse_rate(res_trace: np.ndarray, eps: float = 0.01, window: int = 5) -> int:
    """
    Steps until |ΔR_t| < eps over a sliding window (semantic stabilization).
    Returns the length of the trace if stabilization is not reached.
    """
    n = int(res_trace.shape[0]) if hasattr(res_trace, "shape") else len(res_trace)
    if n <= window:
        return n
    diffs = np.abs(np.diff(res_trace))
    for t in range(window, len(diffs)):
        if np.all(diffs[t - window : t] < eps):
            return t
    return n

def completion_detected(
    res_trace: np.ndarray,
    eps: float = 0.01,
    window: int = 5,
    min_tokens: int = 32,
) -> bool:
    """
    True if resonance has stabilized (|ΔR_t| small across a window) after at least min_tokens.
    """
    n = int(res_trace.shape[0]) if hasattr(res_trace, "shape") else len(res_trace)
    if n < min_tokens:
        return False
    cr = collapse_rate(res_trace, eps=eps, window=window)
    return cr < n

# -----------------------------
# Aggregations & summaries
# -----------------------------

def resonance_summary(R: np.ndarray) -> Dict[str, float]:
    """Basic descriptive stats for a resonance trace."""
    R = np.asarray(R, dtype=float)
    if R.size == 0:
        return {"R_mean": 0.0, "R_var": 0.0, "R_std": 0.0, "R_final": 0.0, "R_collapse_steps": 0}
    return {
        "R_mean": float(np.mean(R)),
        "R_var":  float(np.var(R)),
        "R_std":  float(np.std(R)),
        "R_final": float(R[-1]),
        "R_collapse_steps": int(collapse_rate(R)),
    }

def aggregate_run_metrics(
    entropies: List[float],
    coherences: List[float],
    resonances: List[float],
) -> Dict[str, float]:
    """
    Aggregate per-step signals into a concise summary suitable for summary.json.
    """
    E = np.asarray(entropies, dtype=float) if len(entropies) else np.array([], dtype=float)
    C = np.asarray(coherences, dtype=float) if len(coherences) else np.array([], dtype=float)
    R = np.asarray(resonances, dtype=float) if len(resonances) else np.array([], dtype=float)

    out: Dict[str, float] = {}
    # Entropy (raw) stats
    if E.size:
        out.update({
            "entropy_mean": float(np.mean(E)),
            "entropy_std": float(np.std(E)),
            "entropy_final": float(E[-1]),
        })
    else:
        out.update({"entropy_mean": 0.0, "entropy_std": 0.0, "entropy_final": 0.0})

    # Coherence stats
    if C.size:
        out.update({
            "coherence_mean": float(np.mean(C)),
            "coherence_std": float(np.std(C)),
            "coherence_final": float(C[-1]),
        })
    else:
        out.update({"coherence_mean": 0.0, "coherence_std": 0.0, "coherence_final": 0.0})

    # Resonance stats (includes collapse)
    if R.size:
        out.update(resonance_summary(R))
    else:
        out.update({"R_mean": 0.0, "R_var": 0.0, "R_std": 0.0, "R_final": 0.0, "R_collapse_steps": 0})

    return out

# -----------------------------
# Convenience: convert StepSignals list → arrays
# -----------------------------

def to_arrays(steps: List[StepSignals]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract entropies, coherences, resonances as numpy arrays from a list of StepSignals.
    """
    if not steps:
        z = np.array([], dtype=float)
        return z, z, z
    E = np.array([s.entropy for s in steps], dtype=float)
    C = np.array([s.coherence for s in steps], dtype=float)
    R = np.array([s.resonance for s in steps], dtype=float)
    return E, C, R