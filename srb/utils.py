"""
SRB Entropy & Plateau Utilities
--------------------------------
Lightweight helpers for computing top-k entropy from logits and for detecting
entropy plateaus suitable for early-exit rules.

All entropies are returned in nats by default (natural logarithm). If you want
bits, set base=2 in the functions that support it.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Union

import math

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None
    F = None

__all__ = [
    "topk_entropy_from_logits",
    "normalize_topk_from_logits",
    "rolling_mean",
    "slope_last",
    "entropy_plateau",
    "should_soft_stop",
]

# -------------------------
# Entropy / Top-k Utilities
# -------------------------

def normalize_topk_from_logits(logits: "torch.Tensor", k: int = 20) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (probs, log_probs) over the top-k tokens of `logits`.

    The returned distribution is renormalized *within the top-k set* so that
    probs.sum() == 1. This gives a stable, comparable entropy estimate even when
    we only peek at a truncated distribution.

    Parameters
    ----------
    logits : torch.Tensor
        1D tensor of shape [vocab].
    k : int
        Number of highest-logit tokens to keep.

    Returns
    -------
    probs : torch.Tensor
        Probabilities over the top-k items (sum to 1.0).
    log_probs : torch.Tensor
        Log-probabilities over the top-k items (matching the normalization of
        `probs`).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for normalize_topk_from_logits")

    if logits.dim() != 1:
        logits = logits.view(-1)

    k = int(max(1, min(k, logits.numel())))
    vals, _ = torch.topk(logits, k=k, dim=-1)
    log_probs = F.log_softmax(vals, dim=-1)
    probs = log_probs.exp()
    return probs, log_probs


def topk_entropy_from_logits(logits: "torch.Tensor", k: int = 20, base: Union[float, str] = math.e) -> float:
    """Compute top-k entropy H = -Σ p log p from logits.

    By default returns **nats**. Pass base=2 to get **bits**.

    Notes
    -----
    • This computes entropy on the **renormalized top-k** distribution, which is
      efficient and consistent with server runtimes that only return top-k
      probabilities.
    • If you want exact entropy, set k to the full vocab size (may be expensive).
    """
    probs, log_probs = normalize_topk_from_logits(logits, k=k)
    H_nats = -(probs * log_probs).sum().item()
    if base == 2:
        return H_nats / math.log(2)
    if isinstance(base, (int, float)) and base != math.e:
        return H_nats / math.log(float(base))
    return float(H_nats)


# -------------------------
# Rolling Stats & Plateau
# -------------------------

def rolling_mean(values: Sequence[float], window: int) -> List[float]:
    """Compute a simple rolling mean with a growing warm-up.

    For the first `window-1` points, the average is taken over the available
    prefix (length 1..window-1). Returns a list of the same length as `values`.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float] = []
    acc = 0.0
    q: List[float] = []
    for v in values:
        q.append(float(v))
        acc += float(v)
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def slope_last(values: Sequence[float], span: int = 3) -> float:
    """Return a simple finite-difference slope over the last `span` points.

    If fewer than `span` values are available, returns +inf to discourage early
    stopping during warm-up.
    """
    if span <= 1:
        raise ValueError("span must be >= 2")
    n = len(values)
    if n < span:
        return float("inf")
    a = float(values[-span])
    b = float(values[-1])
    return (b - a) / (span - 1)


def entropy_plateau(
    entropies: Sequence[float],
    window: int = 16,
    slope_eps: float = 0.01,
    span: int = 3,
) -> Tuple[bool, float, float]:
    """Detect whether a rolling-mean entropy series has plateaued.

    Parameters
    ----------
    entropies : Sequence[float]
        Per-token entropy values (nats by default) for the *generated* tokens so far.
    window : int
        Rolling mean window size (tokens). Larger → smoother & stricter.
    slope_eps : float
        Absolute slope threshold considered "flat enough" to stop.
    span : int
        Number of tail points to compute the slope over.

    Returns
    -------
    plateau : bool
        True if |slope| < slope_eps over the rolling-mean tail.
    slope : float
        The computed slope value.
    last_mean : float
        The last rolling-mean entropy value.
    """
    if len(entropies) == 0:
        return False, float("inf"), float("nan")

    means = rolling_mean(entropies, window)
    sl = slope_last(means, span=span)
    plateau = abs(sl) < float(slope_eps)
    return plateau, sl, float(means[-1])


# -------------------------
# Text Heuristics
# -------------------------

def should_soft_stop(text: str) -> bool:
    """Lightweight completion heuristic.

    Returns True when `text` appears to end a thought cleanly—by default when
    the trimmed text ends with a terminal punctuation mark. Extend this with
    domain-specific cues (e.g., closing backticks, headings) if desired.
    """
    trimmed = (text or "").strip()
    if not trimmed:
        return False
    return trimmed.endswith((".", "!", "?", "…"))