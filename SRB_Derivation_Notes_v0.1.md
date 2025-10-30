# Semantic Resonance Budgeting (SRB) — Derivation Notes v0.1

This document provides a compact formal outline of the entropy and coherence signals used in Semantic Resonance Budgeting (SRB), and the initial control logic coupling them to inference modulation.

---

## 1. Token Entropy

At token step *t* with context \(x_{\le t}\), the model’s next‑token distribution \(p_t(k)\) gives:

```math
H_t = - \sum_{k \in V} p_t(k) \log p_t(k)
```

We track raw entropy and normalized entropy:

```math
\hat{H}_t = rac{H_t}{\log(|V_{	ext{eff},t}|)}
```

Where \(V_{\text{eff},t}\) may be:
- Full vocabulary size
- Effective support (tokens with probability ≥ ε)

---

## 2. Entropy Dynamics

We compute first and second derivatives to track collapse rate and curvature:

```math
\Delta H_t = H_t - H_{t-1}
\Delta^2 H_t = \Delta H_t - \Delta H_{t-1}
```

We also apply an EMA smooth:

```math
	ilde{H}_t = lpha H_t + (1 - lpha)	ilde{H}_{t-1}
```

---

## 3. Coherence / Semantic Resonance

Let \(e_t\) be an embedding vector (e.g., mean of token embeddings over a sliding window):

```math
C^+_t = \cos(e_{t-1}, e_t) = rac{\langle e_{t-1}, e_t angle}{\|e_{t-1}\|\|e_t\|}
```

Interpretation: **directional stability in semantic space** as the model generates text.

---

## 4. Control Policy

We treat SRB as an online control rule over inference continuation:

```math
a_t = f(	ilde{H}_t, \Delta	ilde{H}_t, C^+_t)
```

### Example early‑exit rule

Stop generation when:

- \(	ilde{H}_t \le 	heta_H\)
- \(|\Delta	ilde{H}_t| \le 	heta_\Delta\)
- \(C^+_t \ge 	heta_C\)
- For \(	au\) consecutive steps

If conditions are met → **early‑exit** (semantic collapse achieved).  
Otherwise → continue inference, with optional **deepen** step if entropy is high but coherence rising.

---

## 5. Research Questions

1. Do \((H_t, \Delta H_t, C^+_t)\) predict quality and failure modes?
2. Can entropy/coherence signals allow **compute savings** without quality loss?
3. Do collapse curves reveal **model‑ and task‑invariant structure**?
4. Can this serve as a formal instrument for **semantic thermodynamics**?

---

### Notes

- Future extensions include KL drift and MI proxies.
- EEG coupling for Resonance Loom is in progress.
- This is a working draft — feedback welcome.

---

*Version:* v0.1 — experimental derivation notes  
*Author:* Joey Stafford (Theory 42 Labs)

