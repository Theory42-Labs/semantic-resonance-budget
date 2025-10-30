# Semantic Resonance Budgeting (SRB) — Derivation Notes v0.1

This document provides a compact formal outline of the entropy and coherence signals used in
Semantic Resonance Budgeting (SRB), and the initial control logic coupling them to inference modulation.

---

## 1) Token Entropy

At token step \(t\) with context \(x_{\le t}\), the model’s next-token distribution \(p_t(k)\) gives:

$$
H_t \;=\; - \sum_{k \in V} p_t(k)\,\log p_t(k)
$$

We track raw entropy and a normalized version:

$$
\hat{H}_t \;=\; \frac{H_t}{\log\!\big(\lvert V_{\text{eff},t}\rvert\big)}
$$

Where \(V_{\text{eff},t}\) may be:
- the full vocabulary size \(|V|\) (simple, comparable), or
- an **effective support** \(\{k : p_t(k)\ge \epsilon\}\) (focus on active mass).

---

## 2) Entropy Dynamics (Trajectory)

We compute first and second differences to track collapse rate and curvature:

$$
\Delta H_t = H_t - H_{t-1}, \qquad
\Delta^2 H_t = \Delta H_t - \Delta H_{t-1}
$$

We also apply an EMA smooth to stabilize control:

$$
\tilde{H}_t \;=\; \alpha\,H_t \;+\; (1-\alpha)\,\tilde{H}_{t-1}
\quad\text{with}\quad \alpha \in (0,1]
$$

---

## 3) Coherence / Semantic Resonance

Let \(\mathbf{e}_t\) be an embedding of the recent output
(e.g., mean of token embeddings over a sliding window of length \(w\)).
Define directional stability between successive windows:

$$
C^+_t \;=\; \cos\!\big(\mathbf{e}_{t-1},\, \mathbf{e}_t\big)
\;=\; \frac{\langle \mathbf{e}_{t-1}, \mathbf{e}_t \rangle}
{\lVert \mathbf{e}_{t-1}\rVert\,\lVert \mathbf{e}_t\rVert}
$$

**Interpretation:** higher \(C^+_t\) ≈ more stable semantic direction as text unfolds.

---

## 4) Control Policy (Online Inference Modulation)

We treat SRB as an online control rule over inference continuation/depth:

$$
a_t \;=\; f\!\big(\tilde{H}_t,\, \Delta\tilde{H}_t,\, C^+_t\big),
\quad
a_t \in \{\text{continue},\, \text{deepen},\, \text{early-exit}\}
$$

### Example early-exit guard

Stop generation when, for \(\tau\) consecutive steps:

- \(\tilde{H}_t \le \theta_H\)  *(uncertainty low)*  
- \(|\Delta\tilde{H}_t| \le \theta_\Delta\)  *(trajectory settled)*  
- \(C^+_t \ge \theta_C\)  *(semantic direction stable)*

If met → **early-exit** (semantic collapse judged stable).
Otherwise continue; optionally **deepen** if \(\tilde{H}_t\) is high but \(C^+_t\) is increasing (seek clarity rather than exit).

---

## 5) Research Questions (What We’re Testing)

1. **Predictive validity:** do \(\tilde{H}_t\), \(\Delta\tilde{H}_t\), \(C^+_t\) predict quality, stability, and failure modes *during* generation?  
2. **Utility:** can the joint signal enable **compute savings** (early-exit / adaptive depth) **without quality loss**?  
3. **Generalization:** do collapse curves reveal **model- and task-invariant structure** (semantic thermodynamics signatures)?  
4. **Ablations/nulls:** does randomizing local logits or removing coherence degrade control performance as expected?

---

### Notes & Extensions

- Add KL drift \(D_{\mathrm{KL}}(p_t \parallel p_{t-1})\) as a “surprise” measure alongside \(\Delta H_t\).  
- Explore MI proxies linking prompt features/attention tags to local collapse rate.  
- EEG coupling (Resonance Loom) planned for human–model **dual-controller** studies.

---

**Version:** v0.1 — experimental derivation notes  
**Author:** Joey Stafford (Theory 42 Labs)