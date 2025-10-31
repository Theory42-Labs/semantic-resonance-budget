import math
from collections import deque

class TransferEntropy:
    """Estimate transfer entropy between sentence-level features of prompt→response.
    Skeleton supports plug-in estimators (e.g., KSG). Here we provide a stub with
    histogram approximation for categorical/quantized features.
    """
    def __init__(self, history_window: int = 3, estimator: str = "histogram"):
        self.w = history_window
        self.estimator = estimator
        self.hist_states = deque(maxlen=self.w)

    def _quantize(self, seq: list[float], bins: int = 16) -> list[int]:
        if not seq:
            return []
        lo, hi = min(seq), max(seq)
        if hi == lo:
            return [0] * len(seq)
        return [min(bins - 1, int((x - lo) / (hi - lo + 1e-12) * bins)) for x in seq]

    def compute(self, prompt_feats: list[float], response_feats: list[float]) -> float:
        """Return a scalar TE estimate from prompt→response."""
        # TODO: replace with KSG estimator on continuous features.
        qp = self._quantize(prompt_feats)
        qr = self._quantize(response_feats)
        joint = {}
        for a, b in zip(qp, qr):
            joint[(a, b)] = joint.get((a, b), 0) + 1
        pa = {}
        pb = {}
        for a, b in joint:
            pa[a] = pa.get(a, 0) + joint[(a, b)]
            pb[b] = pb.get(b, 0) + joint[(a, b)]
        n = max(1, sum(joint.values()))
        te = 0.0
        for (a, b), c in joint.items():
            p_ab = c / n
            p_a = pa[a] / n
            p_b = pb[b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                te += p_ab * math.log((p_ab / (p_a * p_b + 1e-12)) + 1e-12)
        return max(0.0, te)