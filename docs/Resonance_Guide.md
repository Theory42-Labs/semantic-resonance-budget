

# 🜁 SRB Resonance Guide  
*A conceptual and mathematical overview of the resonance layer in Semantic Resonance Budget (SRB).*

---

## 1. Overview

Resonance is SRB’s higher‑order interpretive layer.  
Where the kinematics layer measures **entropy, semantic velocity, acceleration, drift, and divergence**, the resonance layer asks:

> **How stable is the model’s internal meaning-state as it generates text?**

Resonance integrates:
- **Entropy** (uncertainty)
- **Coherence** (semantic conformity)
- **Change over time** (derivatives)

Together, these produce a signal that reflects the internal tension, stability, or collapse of the reasoning process.

Resonance is *not* metaphysical—it is a technical signal representing the interaction of surprise and structure during generation.

---

## 2. Required Inputs

The resonance layer uses two signals derived from SRB kinematics:

### 2.1 Semantic Entropy  
From `SRBStepMetrics.semantic_entropy`

A normalized measure of local uncertainty in the probability distribution for each token.

### 2.2 NSM Divergence  
From `SRBStepMetrics.nsm_divergence`

Distance from a normative semantic baseline (e.g., Natural Semantic Metalanguage expectations).

This is interpreted as **semantic drift**:
- high divergence → low coherence  
- low divergence → high coherence

---

## 3. Coherence Model

Resonance requires a coherence signal.  
SRB constructs this using:

```python
coherence = 1 - normalize(nsm_divergence)
```

Where:
- `normalize` is min–max normalization over the sequence  
- `coherence` ∈ [0, 1]  
- Higher = more semantically consistent with expected structure

This coherence layer can evolve in the future (e.g., discourse models, topic adherence, embedding‑space clustering).

---

## 4. Resonance Amplitude

Resonance amplitude reflects the *interaction* of surprise (entropy) and structure (coherence).

### 4.1 Normalized Entropy
Entropy is normalized:

```
H_norm = entropy / max_entropy
```

where `max_entropy` is detected automatically or specified manually.

### 4.2 Resonance Formula

The resonance amplitude is defined as:

```
R = (1 - H_norm) * max(coherence, 0)
```

Interpretation:
- Low entropy AND high coherence → high resonance  
- High entropy OR low coherence → low resonance  
- Negative coherence is floored at zero (cannot add energy)

This maps cleanly onto an intuitive meaning:
> **Resonance is the model “holding its shape.”**

---

## 5. Resonance Series Construction

From an `SRBSequenceMetrics`:

```python
from srb.resonance import build_resonance_from_sequence

res_steps = build_resonance_from_sequence(result)
```

This yields a list of:

```python
ResonanceStep(
    entropy,
    entropy_norm,
    coherence,
    resonance,
    resonance_derivative,
)
```

Where:
- `entropy_norm` is normalized entropy
- `resonance` is amplitude R
- `resonance_derivative` captures sudden changes over time

The derivative is especially valuable for collapse detection.

---

## 6. Detecting Resonance Collapse

A collapse event happens when resonance destabilizes sharply.

This is implemented using:

```python
from srb.resonance import estimate_collapse_step
collapse_idx = estimate_collapse_step(
    [s.resonance for s in res_steps],
    window=5,
    eps=1e-3,
)
```

### 6.1 Intuition

A collapse may indicate:
- Loss of coherence in reasoning
- Phase transition in the model’s internal attention patterns
- Start of hallucination
- Breakdown of structured reasoning
- Transition into boilerplate, filler, or exhausted states

### 6.2 Algorithm

The detector:
1. Computes a rolling standard deviation (volatility)
2. Flags the earliest point where volatility drops below a small threshold (`eps`)
3. Returns the index of the collapse or `None` if no stable collapse is detected

This finds **stability after chaos**, indicating a structural “shift” in the reasoning process.

---

## 7. Visualizing Resonance

Visualization is not required, but highly clarifying.

Typical resonance plots include:
- Entropy vs. Coherence
- Resonance amplitude over time
- Resonance derivative
- Collapse markers

Once `srb/viz.py` is implemented, recommended helpers will be:

```
plot_resonance(...)
plot_kinematics(...)
plot_collapse(...)
```

---

## 8. Interpretation Guidance

### 8.1 High Resonance
Indicates:
- Strong semantic structure  
- Low uncertainty  
- High conformity to expected norms  
- Stable reasoning  

Often seen in:
- Clear explanations  
- Well‑structured arguments  
- Procedural or step-by-step reasoning  

### 8.2 Low Resonance
Indicates:
- High uncertainty OR semantic drift  
- Structural instability  
- Weak internal coherence  

Often seen in:
- “Waffling” responses  
- Hallucinations  
- Topic shifts  
- Degenerate or repetitive output  

### 8.3 Sudden Changes (Derivative Spikes)
Indicate:
- Abrupt reasoning transitions  
- Internal conflict or reformulation  
- Model “changing direction” mid‑sentence  

### 8.4 Collapse Event
Marks:
- The point where reasoning ceases to be structured
- OR transitions into low-entropy boilerplate text

This is especially useful for:
- Detecting overshoot in chain‑of‑thought
- Identifying where the reasoning “ran out of steam”
- Training truncation and routing systems

---

## 9. Future Extensions

Planned or possible enhancements:

- Multi-scale resonance using different entropy bands  
- Embedding‑space coherence models  
- Cross-head or cross-layer resonance at attention-level  
- Multi-sequence resonance for comparing candidate answers  
- Real-time resonance-aware generation  

---

## 10. Summary

Resonance transforms SRB from a kinematic measurement framework into an interpretive lens for understanding model behavior.

It captures:
- Stability  
- Structure  
- Transitions  
- Collapse  

And provides a powerful, intuitive signal for both research and applied inference routing.

*Resonance makes the invisible structure of reasoning visible.*
