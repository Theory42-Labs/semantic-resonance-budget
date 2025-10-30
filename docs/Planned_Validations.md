

# Planned Validations & Falsification Criteria for SRB

This file tracks the planned validation experiments for Semantic Resonance Budgeting (SRB), along with falsification criteria.  
The objective is to rigorously test whether SRB provides meaningful control signals for generative models, and to document failure cases clearly when they arise.

SRB is not assumed to be correct — it must *earn* validity through reproducible experiments.

---

## 1. Entropy Trajectory Predictive Power

**Hypothesis:**  
Token‑wise entropy dynamics (`H_t`, `ΔH_t`, `Δ²H_t`) correlate with output quality and reasoning stability.

**Tests:**
- Apply SRB metrics on tasks: reasoning, QA, summarization, creative writing.
- Score outputs via rubric and/or human evaluation.
- Compare correlation vs baselines: perplexity, output length.

**Falsification Criteria:**
- SRB features do not outperform simple baselines across tasks and models.
- No consistent statistical signal across seeds.

---

## 2. Coherence Signal Robustness (C⁺)

**Hypothesis:**  
Cosine‑based coherence (`C⁺_t`) captures semantic stability, not just repetition.

**Tests:**
- Compare `C⁺_t` with:
  - NCD similarity (`1 − NCD`)
  - normalized Levenshtein similarity
- Evaluate on:
  - repetition‑heavy outputs
  - bilingual / creative outputs
  - long‑form responses

**Falsification Criteria:**
- Cosine similarity collapses to “repetition detector”
- After controlling for repetition and token length, `C⁺_t` provides no meaningful additional signal

---

## 3. Compute Savings Without Quality Loss

**Hypothesis:**  
Early‑exit / deepen policy improves compute efficiency without harming quality.

**Tests:**
- Paired runs: SRB vs no‑SRB
- Track tokens, latency, and performance metrics

**Falsification Criteria:**
- Quality degradation exceeds accepted tolerance (e.g., >1/10 on rubric)
- Compute savings negligible or negative

---

## 4. Normalization Stability

**Hypothesis:**  
Entropy normalization choices do not invalidate conclusions.

**Variants:**
- raw entropy
- `H / log(|V|)`
- `H / log(|V_eff|)` (ε‑support)
- top‑k normalized entropy

**Falsification Criteria:**
- Core findings reverse depending on normalization
- Instability across comparable token‑wise entropy measures

---

## 5. Generalization Across Models & Seeds

**Hypothesis:**  
SRB features generalize across architectures and sampling noise.

**Tests:**
- Llama‑3.x, Qwen‑7B, Mistral‑7B (initial set)
- 3–5 seeds per model

**Falsification Criteria:**
- SRB effects fail to appear consistently across models or seeds

---

## 6. Long‑Form Degeneration Guard

**Hypothesis:**  
SRB does not trigger premature termination during coherent long‑form generation.

**Tests:**
- Narrative and multi‑paragraph evaluation
- Add degeneration flags (repetition rise, entropy rebound, topic drift)

**Falsification Criteria:**
- Early‑stopping damages narrative structure
- Degeneration detectors routinely override SRB (suggesting SRB naive to long‑form structure)

---

## 7. ENA / Redundancy Analogy Exploration

**Goal:**  
Explore parallels to ecological network ascendency & redundancy (Ulanowicz).

**Non‑critical — exploratory.**  
Results will be documented but not counted as pass/fail.

---

## Status Tracking

| Validation | Status | Notes |
|---|---|---|
Entropy → Quality | ☐ Not started | |
Cosine vs NCD/Lev | ☐ Not started | |
Compute Savings | ☐ Not started | |
Normalization Study | ☐ Not started | |
Cross‑model Seeds | ☐ Not started | |
Long‑form Guard | ☐ Not started | |
ENA Analogy | ☐ Exploratory | |

---

## Philosophy

> *We are not proving SRB — we are actively trying to break it.*

If SRB survives these tests, it will have earned confidence.  
If it fails, those failure modes will directly guide refinement and future work.