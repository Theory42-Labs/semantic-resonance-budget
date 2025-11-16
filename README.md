# Semantic Resonance Budget (SRB)

Semantic Resonance Budget (SRB) is a model‑agnostic framework for measuring the *internal semantic dynamics* of large language models during inference.  
It provides a physics‑inspired set of metrics—velocity, acceleration, entropy, divergence, resonance amplitude—that quantify how a model *moves through meaning* as it generates text.

SRB can be used for:
- Research on reasoning processes in LLMs  
- Detecting hallucination drift and instability  
- Evaluating model efficiency and coherence  
- Real‑time inference monitoring  
- Adaptive inference strategies  
- Multi‑model comparison and benchmarking  

This repository contains the **clean, official SRB core library**, designed for use in research projects and production inference pipelines.

> For the original Phase 1–5 experimental codebase, datasets, and exploratory notebooks, see the companion repo:  
> **semantic-resonance-budget-lab**

---

## 🧠 Core Concepts

SRB models semantic inference as a *trajectory* through high‑dimensional embedding space.  
Each generated token is treated as a point on this trajectory; SRB measures the “semantic kinematics” of that motion.

### **Primary Metrics**
These are fully implemented in the SRB API:

| Metric | Description |
|--------|-------------|
| **Semantic Velocity** | How far the model moves in embedding space per token. |
| **Semantic Entropy** | Token‑level uncertainty from the model’s probability distribution. |
| **Surprisal Gradient** | Change in internal expectation from one token to the next. |
| **NSM Divergence** | Deviation from the model’s expected semantic direction. |
| **Semantic Acceleration** | Second derivative of the semantic trajectory; reasoning phase shifts. |

### **Resonance Metrics**
SRB also incorporates resonance‑based measures:

- Normalized entropy  
- Coherence (cosine + compression + repetition penalties)  
- Resonance amplitude  
- Resonance collapse rate  

These originated in early SRB experiments and remain fully supported.

---

## 📦 Repository Structure

```
semantic-resonance-budget/
│
├── srb/
│   ├── api.py            # Public API for running SRB analysis
│   ├── kinematics.py     # Velocity, acceleration, entropy, divergence, gradient
│   ├── resonance.py      # Resonance amplitude, coherence, collapse metrics
│   ├── types.py          # Data classes for structured SRB output
│   ├── adapters/         # Model wrappers (OpenAI, HF, local models)
│   └── utils.py          # Shared helpers
│
├── examples/
│   ├── notebook.ipynb    # Demonstration of SRB usage
│   └── example_usage.py
│
├── tests/
│
├── README.md
└── pyproject.toml
```

The structure is designed for clarity, extendibility, and publication readiness.

---

## 🚀 Quick Start

### Install (local dev)
```
pip install -e .
```

### Basic Usage

```python
from srb.api import analyze_sequence
from srb.adapters.openai import OpenAIChatAdapter

adapter = OpenAIChatAdapter(model="gpt-4.1-mini")
steps = adapter.generate_with_traces("Explain SRB in one paragraph.")

result = analyze_sequence(steps)

for step in result.steps:
    print(step.index, step.semantic_velocity, step.semantic_entropy)
```

### Adding Resonance Analysis

You can extend SRB kinematics with resonance metrics derived from entropy and semantic coherence.

```python
from srb.api import analyze_sequence
from srb.adapters.openai import OpenAIChatAdapter
from srb.resonance import (
    build_resonance_from_sequence,
    estimate_collapse_step,
)

adapter = OpenAIChatAdapter(model="gpt-4.1-mini")
steps = adapter.generate_with_traces("Explain resonance in SRB.")

# Core SRB metrics
result = analyze_sequence(steps)

# Build resonance series from SRB output
resonance_steps = build_resonance_from_sequence(result)

# Find the first resonance collapse point
collapse_idx = estimate_collapse_step(
    [s.resonance for s in resonance_steps],
    window=5,
    eps=1e-3,
)

print(f"Resonance collapse detected at token index: {collapse_idx}")
```

This produces:
- A resonance amplitude per token  
- A coherence-normalized entropy signal  
- A collapse index indicating where the model's internal structure destabilizes
```

Outputs per‑token semantic metrics along with aggregate summary statistics.

---

## 📘 Documentation

Full metric definitions live in:

```
docs/Metrics_Definitions.md
```

This includes:
- Mathematical formulations
- Visual interpretation guidance
- Examples of behavior patterns
- Use cases for each metric

Resonance methods and interpretation guidelines are documented in `docs/Resonance_Guide.md` (if present) and will expand as SRB’s resonance layer develops.

---

## 🧪 Experimental Codebase

The SRB experimentation environment (Phase 1–5), including:
- UMAP/TDA prototypes  
- Entropy topology experiments  
- Mirror tests and falsification trials  
- Research notebooks  
- Raw generation logs  

…has been moved into the archive repository:

**semantic-resonance-budget-lab**

This preserves reproducibility while keeping the SRB core library clean and focused.

---

## 📄 License

SRB is distributed under a **dual‑license model** designed to support open scientific research while protecting commercial usage.

- **Academic & Research License** — for universities, non‑profits, independent researchers, and open scientific work.  
  See: `LICENSE-ACADEMIC.md`

- **Commercial License** — required for any commercial, enterprise, proprietary, or revenue‑generating use.  
  See: `LICENSE-COMMERCIAL.md`

For a summary of allowed use cases, see the top‑level `LICENSE.md`.

To obtain a Commercial License, contact:  
**theory42labs@gmail.com**

---

## 🤝 Contributing

SRB welcomes contributors interested in:
- Semantic dynamics of LLMs  
- Adaptive inference  
- AI reasoning research  
- Visualization of internal model states  

Please open an issue or submit a pull request.

---

## 🌌 About the Project

SRB is part of a broader exploration of:
- Semantic Cartography  
- Entropic reasoning processes  
- AI self‑reflection metrics  
- Emergent coherence patterns in language models  

The goal is to make the invisible structure of model reasoning *measurable*, *visualizable*, and *scientifically rigorous*.

---

## ✨ Acknowledgments

This project was created through the collaborative reasoning work of  
**Joey Stafford & Aevum**,  
integrating engineering, mathematical intuition, and semantic physics research.