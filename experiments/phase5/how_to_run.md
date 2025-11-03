# Phase 5 — How to Run (TSIF: Token‑Step Inference Fingerprinting)

This guide explains how to run the **Phase 5** pipeline: token‑step generation, telemetry capture (logprobs, surprisal), semantic embeddings, **UMAP** trajectory, **semantic velocity/curvature**, and **persistent homology** via Ripser. It also covers environment setup, model choices, outputs, and common issues.

---

## 0) Prerequisites

- **Python** ≥ 3.10  
- **macOS (Apple Silicon)** or Linux. (Windows WSL works but not tested here.)
- For Apple Silicon GPU: PyTorch with **MPS** support (`torch.backends.mps.is_available()` → `True`)
- For NVIDIA GPU (optional): CUDA‑enabled PyTorch (not required for macOS).
- A Hugging Face account/token if you use gated models.

> **Repo layout (Phase 5)**
>
> ```
> experiments/phase5/
> ├── runners/
> │   └── run_phase5.py
> ├── analysis/
> │   └── analyze_phase5.py
> ├── config/
> │   └── defaults.yaml
> └── reports/
>     └── … (timestamped run folders get created here)
> ```

---

## 1) Environment Setup

From the repo root:

```bash
# 1) Create and activate virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt

# 3) (Optional) Hugging Face auth if you use gated models
# Either set env var or run CLI login
export HF_TOKEN=hf_xxx_your_token
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" || true

# 4) (Optional) quiet tokenizers warning
export TOKENIZERS_PARALLELISM=false
```

**Requirements** (already in `requirements.txt`, but listed here for clarity):
- `torch`, `transformers`, `accelerate`, `sentencepiece`
- `sentence-transformers`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `umap-learn`, `ripser`
- `matplotlib`, `tqdm`, `pyyaml`

---

## 2) Choose a Model

Phase 5 works best with small‑to‑mid models on a MacBook:

- ✅ `microsoft/Phi-4-mini-instruct` (fast, stable)
- ✅ `microsoft/Phi-3.5-mini-instruct` (fast, stable)
- ✅ `meta-llama/Llama-3.2-3B-Instruct` (OK on MPS, slower)
- ⚠️ `mistralai/Mistral-7B-Instruct-v0.3` (often slow on laptops)
- ❌ Very large models (70B) → use cloud (see Phase 4 cloud guide).

---

## 3) Configure (optional)

Edit `experiments/phase5/config/defaults.yaml` for:
- `model.name`, `max_new_tokens`, `temperature`, `top_p`
- telemetry flags (e.g., surprisal/logprobs on)
- seed, device (`cpu|mps|cuda`)

You can also override via CLI flags.

---

## 4) Run the Phase 5 Runner

From repo root:

```bash
# Example: Apple Silicon (MPS) with Phi‑4‑mini
python experiments/phase5/runners/run_phase5.py \
  --config experiments/phase5/config/defaults.yaml \
  --device mps \
  --model microsoft/Phi-4-mini-instruct \
  --tag tsif_local_phi4_mini
```

Other examples:

```bash
# Llama 3.2 3B (MPS)
python experiments/phase5/runners/run_phase5.py \
  --config experiments/phase5/config/defaults.yaml \
  --device mps \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --tag tsif_local_llama3b

# CPU only (slow but works)
python experiments/phase5/runners/run_phase5.py \
  --config experiments/phase5/config/defaults.yaml \
  --device cpu \
  --model microsoft/Phi-3.5-mini-instruct \
  --tag tsif_cpu_phi35mini
```

### What the runner does

For each prompt sequence it:

1. Generates **token by token** (causal decoding) and records:
   - token ids, logprobs, **surprisal**
   - cumulative text after each step
2. Embeds each step with `sentence-transformers`
3. Builds a **UMAP** 2D trajectory of the semantic path
4. Computes **semantic velocity** & **semantic curvature**
5. Runs **Ripser** for persistent homology on the trajectory (H₀/H₁)
6. Saves a **timestamped folder** under `experiments/phase5/reports/` with:
   - `config.yml` (resolved configuration for the run)
   - `phase5_records.jsonl` (all stepwise telemetry)
   - Plot images (trajectory, velocity, curvature, surprisals, TDA)
   - `analysis_index.csv` (written by the analysis step)

Output folder pattern:
```
experiments/phase5/reports/<tag>_<YYYYMMDD_HHMMSS>/
```

---

## 5) Analyze & Plot

You can analyze the **latest** run automatically:

```bash
python experiments/phase5/analysis/analyze_phase5.py
```

Or point to a specific run dir:

```bash
python experiments/phase5/analysis/analyze_phase5.py \
  --run-dir experiments/phase5/reports/tsif_local_phi4_mini_20251103_093012
```

Artifacts written to the run folder include:

- `*_umap_trajectory.png` — 2D semantic path over time
- `*_trace_surprisal.png` — token‑step surprisal curve
- `*_trace_cos_prompt.png` — cosine similarity vs prompt
- `*_trace_cos_prev.png` — cosine similarity vs previous step
- `*_semantic_velocity.png` — ||Δembedding||/Δt (z‑scored)
- `*_semantic_curvature.png` — frame‑local turn of the path
- `*_persistent_homology.png` — Ripser diagram (H₀/H₁)
- `analysis_index.csv` — list of all generated figures

---

## 6) Reproducibility

- Set `--seed` or `seed:` in YAML.
- The run folder is timestamped; `config.yml` captures resolved config.
- Rerunning with the same seed + config should produce similar geometry (stochastic decoding may still vary slightly).

---

## 7) Performance Tips

- Prefer **Phi‑4‑mini** or **Phi‑3.5‑mini** on laptops.
- Reduce `max_new_tokens` during iteration/experimentation.
- Keep `temperature` modest (e.g., 0.6) and `top_p` ≤ 0.95 to avoid NaN probs.
- On macOS, ensure:
  ```python
  import torch; print(torch.__version__, hasattr(torch.backends,'mps'), torch.backends.mps.is_available())
  ```
- If very slow, switch to a smaller model or move to a CUDA cloud VM.

---

## 8) Common Warnings & Errors (and what to do)

- **UMAP**  
  - `n_jobs value 1 overridden …` — benign.  
  - If you see spectral init warnings, UMAP falls back to random init — also fine.

- **scikit‑learn matmul: divide/overflow/invalid**  
  - Usually harmless when working with small sample counts; results still render.  
  - If frequent, reduce dimensionality noise (normalize embeddings) or increase sample size.

- **Ripser: “point cloud has more columns than rows; transpose?”**  
  - Benign; your path samples can be short relative to embedding dimensionality.

- **`RuntimeError: probability tensor contains NaN/inf/<0`**  
  - Use safer decoding: lower `temperature` (e.g., 0.6), clamp `top_p` to 0.9–0.95.

- **“NLL computation failed, using stub”**  
  - Happens if a model backend doesn’t expose token‑level losses in the expected way. The pipeline falls back to a stable stub; geometry plots still work.

---

## 9) Interpreting the Figures (30‑second cheat sheet)

- **UMAP Trajectory**: the *shape* of the model’s evolving representation. Smooth arcs = stable reasoning; sharp bends = context shifts.
- **Semantic Velocity**: speed of conceptual change; spikes often precede decisive answer tokens.
- **Semantic Curvature**: how sharply the path turns; peaks can indicate branch/repair or creative pivots.
- **Surprisal Trace**: token‑level “shock”; correlates with difficult steps or novelty.
- **Persistent Homology**: loops/holes suggest revisitation or multi‑threaded reasoning motifs.

---

## 10) Re‑running with a Different Model

Just swap `--model` (and optionally `--tag`) on the run command:

```bash
python experiments/phase5/runners/run_phase5.py \
  --config experiments/phase5/config/defaults.yaml \
  --device mps \
  --model microsoft/Phi-4-mini-instruct \
  --tag tsif_local_phi4_mini
```

---

## 11) Contact / Issues

If something breaks:
- Capture the exact command + console output
- Note your OS, Python, PyTorch versions
- Open an issue with your run folder contents (config + logs + plots)

Happy mapping. 🧭
