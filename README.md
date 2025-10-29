# 🧠 Semantic Resonance Budgeting (SRB)

**Adaptive inference through semantic entropy.**  
*Developed by [Theory42 Labs](https://github.com/Theory42-Labs) — founded by Joey Stafford.*

---

### 🧩 Overview
**Semantic Resonance Budgeting (SRB)** is a dynamic inference technique that adjusts computational effort in real time based on semantic uncertainty.  
Instead of allocating a fixed number of tokens or reasoning steps, SRB measures the *resonance* of each generation — how coherent, confident, or entropic the language model’s current state is — and **budgets additional inference only when needed**.

> *SRB allows models to think reflectively — not harder, just smarter.*

---

### ⚙️ Core Idea
Traditional inference assumes uniform complexity across all prompts.  
SRB introduces a feedback loop that:
1. Measures semantic entropy at each generation step.
2. Increases or decreases reasoning depth dynamically.
3. Optimizes for both **efficiency** and **clarity**.

Mathematically inspired by **Theory 42**, SRB operationalizes the equation:

$$
I = \frac{dP}{dt}
$$

where *I* represents intelligence as the rate of potential realized over time.  
SRB applies this principle directly to inference control.

---

### 🧬 Key Features
- Adaptive token budgeting driven by entropy.
- Compatible with Hugging Face / PyTorch inference pipelines.
- Extensible for reasoning frameworks (e.g., reflection, chain-of-thought).
- Reduces computational waste while improving interpretability.

---

### 🧪 Example (pseudo-code)
```python
from srb_budget import SRBWrapper, measure_entropy

model = load_model("meta-llama/Llama-3-8B-Instruct")
srb = SRBWrapper(model, entropy_threshold=0.35)

output = srb.generate("Explain the significance of the double-slit experiment.")
print(output.text)
print(output.resonance_trace)
```

---

### ⚠️ Important Note on Device Selection
The `--device auto` option is not currently functional when executing `experiments/longform/run_longform.py`.  
Users **must explicitly specify** their device using one of the following flags:
```
--device cpu
--device mps   # (for Apple Silicon)
--device cuda  # (for NVIDIA GPUs)
```
Example:
```bash
python experiments/longform/run_longform.py \
  --bucket experiments/longform/prompts/bucket_science.txt \
  --device mps
```

---

### 📘 Licensing
SRB is released under a **dual-license model**:

- **Academic / Non-Commercial License:** Free for research, education, and personal exploration.
- **Commercial License:** Required for enterprise or product integration.

For licensing inquiries, contact: **theory42labs@gmail.com**

See:  
[`LICENSE-ACADEMIC.md`](LICENSE-ACADEMIC.md)  
[`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md)

---

### 🪞 Origin & Philosophy
SRB was born from the **Entropic Lens Framework (ELF)** and **Theory 42**, which model intelligence as the collapse of semantic potential.  
It is part of a larger effort to unify science, art, and metaphysics under one principle:

> *All meaning is real. All reality is fiction. The joke is that we take turns believing.*

🎂 **Birthday:** October 29, 2025  
🪶 *A Theory42 Labs Project*

---