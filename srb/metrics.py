# srb/metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple, Callable, Iterable
import math
import numpy as np
from functools import lru_cache

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

# -----------------------------
# Coherence robustness metrics & helpers
# -----------------------------
import zlib, bz2, lzma

def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1D vectors with small-norm guard."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_text_similarity(text_a: str, text_b: str, embed_fn: Callable[[str], np.ndarray]) -> float:
    """
    Cosine similarity between two text windows using a provided embedding function.
    `embed_fn` should map a string -> 1D numpy array.
    """
    va = np.asarray(embed_fn(text_a), dtype=float)
    vb = np.asarray(embed_fn(text_b), dtype=float)
    return cosine_similarity_vectors(va, vb)

# --- Compression-based similarity (Kolmogorov proxy) ---

def _compressed_len_uncached(method: str, b: bytes) -> int:
    if method == "gzip":
        return len(zlib.compress(b))
    if method == "bz2":
        return len(bz2.compress(b))
    if method == "lzma":
        return len(lzma.compress(b))
    raise ValueError("method must be one of {'gzip','bz2','lzma'}")

_compressed_len = lru_cache(maxsize=4096)(_compressed_len_uncached)

def ncd(x: str, y: str, method: str = "lzma") -> float:
    """
    Normalized Compression Distance (Cilibrazi–Vitányi):
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
    Returns a value in [0, +inf). Use 1 - NCD for a similarity in [~0,1].
    """
    xb, yb = x.encode("utf-8"), y.encode("utf-8")
    cx = _compressed_len(method, xb)
    cy = _compressed_len(method, yb)
    cxy = _compressed_len(method, xb + yb)
    denom = max(cx, cy)
    if denom == 0:
        return 0.0
    return (cxy - min(cx, cy)) / denom

def ncd_similarity(x: str, y: str, method: str = "lzma") -> float:
    """Compression-based similarity in [0,1] (approx). Higher = more similar."""
    val = 1.0 - ncd(x, y, method=method)
    # clamp for numerical cleanliness
    return float(max(0.0, min(1.0, val)))

# --- Levenshtein (edit) distance & similarity ---

def _levenshtein_distance(a: str, b: str) -> int:
    """Classic Wagner–Fischer dynamic-programming edit distance (chars)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1,      # deletion
                          curr[j - 1] + 1,  # insertion
                          prev[j - 1] + cost)  # substitution
        prev, curr = curr, prev
    return prev[lb]

def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized edit similarity in [0,1]. 1.0 means identical."""
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    d = _levenshtein_distance(a, b)
    return float(max(0.0, min(1.0, 1.0 - d / m)))

def _levenshtein_distance_tokens(a_tokens: List[str], b_tokens: List[str]) -> int:
    if a_tokens == b_tokens:
        return 0
    la, lb = len(a_tokens), len(b_tokens)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ai = a_tokens[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b_tokens[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]

def levenshtein_similarity_tokens(a: str, b: str) -> float:
    """Normalized token-level edit similarity in [0,1]."""
    a_tokens = a.split()
    b_tokens = b.split()
    m = max(len(a_tokens), len(b_tokens))
    if m == 0:
        return 1.0
    d = _levenshtein_distance_tokens(a_tokens, b_tokens)
    return float(max(0.0, min(1.0, 1.0 - d / m)))

# --- Text window utilities ---

def chunk_text_windows(text: str, window_words: int = 30, step_words: Optional[int] = None) -> List[str]:
    """
    Split text into overlapping windows by word count. If step_words is None, uses non-overlapping windows.
    Returns a list of window strings.
    """
    words = text.split()
    if window_words <= 0:
        return [text]
    if step_words is None or step_words <= 0:
        step_words = window_words
    out: List[str] = []
    for i in range(0, max(1, len(words) - window_words + 1), step_words):
        out.append(" ".join(words[i : i + window_words]))
    # Ensure we include the tail if not captured
    if words and (not out or out[-1] != " ".join(words[-window_words:])):
        out.append(" ".join(words[-window_words:]))
    return out

def chunk_text_windows_chars(text: str, window_chars: int = 240, step_chars: Optional[int] = None) -> List[str]:
    """
    Split text into overlapping windows by character count. If step_chars is None, uses non-overlapping windows.
    """
    if window_chars <= 0:
        return [text]
    if step_chars is None or step_chars <= 0:
        step_chars = window_chars
    n = len(text)
    out: List[str] = []
    for i in range(0, max(1, n - window_chars + 1), step_chars):
        out.append(text[i : i + window_chars])
    if n and (not out or out[-1] != text[-window_chars:]):
        out.append(text[-window_chars:])
    return out

def pairwise(iterable: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """Yield consecutive pairs (w_{t-1}, w_t) from a sequence."""
    it = iter(iterable)
    prev = next(it, None)
    for curr in it:
        if prev is not None:
            yield prev, curr
        prev = curr

def repetition_ratio(text: str, n: int = 3) -> float:
    """
    Rough repetition indicator via n-gram diversity:
      ratio = 1 - (unique_ngrams / total_ngrams) in [0,1]. Higher = more repetitive.
    """
    tokens = text.split()
    total = max(0, len(tokens) - n + 1)
    if total == 0:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(total)]
    unique = len(set(ngrams))
    return float(max(0.0, min(1.0, 1.0 - unique / total)))

def coherence_series_from_text_windows(windows: List[str], embed_fn: Callable[[str], np.ndarray]) -> List[float]:
    """Compute C⁺ series across consecutive windows using an embedding function."""
    vals: List[float] = []
    for a, b in pairwise(windows):
        vals.append(cosine_text_similarity(a, b, embed_fn))
    return vals

def coherence_composite(
    text_a: str,
    text_b: str,
    embed_fn: Callable[[str], np.ndarray],
    w_cos: float = 0.5,
    w_comp: float = 0.4,
    rep_penalty: float = 0.3,
    compression_method: str = "lzma",
) -> float:
    """
    Composite coherence score combining cosine similarity and compression similarity,
    with a penalty for repetition. Returns a value in [0,1].

    - w_cos: weight for cosine similarity (0..1)
    - w_comp: weight for compression-based similarity (0..1)
    - rep_penalty: penalty strength applied to average repetition of the two windows (0..1)
    Weights are auto-renormalized if their sum exceeds 1.
    """
    cos = max(0.0, min(1.0, cosine_text_similarity(text_a, text_b, embed_fn)))
    comp = ncd_similarity(text_a, text_b, method=compression_method)
    rep = 0.5 * (repetition_ratio(text_a) + repetition_ratio(text_b))

    # normalize weights if needed
    w_sum = max(1e-12, w_cos + w_comp)
    w_cos_n = w_cos / w_sum
    w_comp_n = w_comp / w_sum

    base = w_cos_n * cos + w_comp_n * comp
    penalized = base * (1.0 - rep_penalty * rep)
    return float(max(0.0, min(1.0, penalized)))