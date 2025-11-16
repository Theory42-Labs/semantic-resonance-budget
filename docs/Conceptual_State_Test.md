

# SRB-CST: Conceptual State Test  
*A Benchmark for Detecting Cognitive Signatures in Language Models*

The **Semantic Resonance Budget – Conceptual State Test (SRB‑CST)** is a diagnostic benchmark designed to evaluate how a language model behaves when confronted with four distinct categories of concepts:

1. **Unknown Concepts**
2. **Known Scientific Concepts**
3. **Fictional or Hybrid Concepts**
4. **User-Defined (In-Context) Concepts**

This benchmark operationalizes the cognitive signatures formalized in `Cognitive_Signatures.md` and provides a standardized experimental protocol for evaluating a model’s semantic dynamics using SRB metrics.

---

## 1. Purpose of SRB‑CST

SRB‑CST enables researchers and engineers to:

- Assess *conceptual grounding* in LLMs  
- Detect *hallucination onset* and *semantic drift*  
- Measure *stylistic fluency versus genuine understanding*  
- Observe *in-context conceptual integration*  
- Compare grounding capabilities across models or fine-tuning methods  
- Provide a reproducible semantic-dynamics benchmark independent of model internals  

SRB‑CST is fully model-agnostic and uses only  
**token embeddings, logits, and SRB kinematics.**

---

## 2. The Four Prompt Conditions

Each test run contains four prompts:

### **A. Unknown Concept Prompt**  
A deliberately invented or obscure term the model has never encountered.  
The response reveals how the model behaves in total semantic ignorance.

**Example:**  
“Explain *Semantic Resonance Budget* in one paragraph.”

---

### **B. Known Scientific Concept Prompt**  
A well‑established ML concept that the model should know.  
Used to measure grounded conceptual navigation.

**Example:**  
“Explain entropy regularization in reinforcement learning.”

---

### **C. Fictional Hybrid Concept Prompt**  
A blend of recognized stylistic tropes (e.g., fantasy + physics).  
Used to test confident stylistic synthesis without real grounding.

**Example:**  
“Explain the mechanics of quantized mana resonance in arcane thermodynamics.”

---

### **D. User‑Defined Concept Prompt**  
A concept fully introduced by an explicit definition given inside the prompt.  
Used to measure in-context conceptual acquisition.

**Example:**  
A supplied definition of SRB + “Explain this in your own words.”

---

## 3. Required SRB Metrics

For each prompt, SRB‑CST records:

- **mean_velocity**
- **max_velocity**
- **mean_entropy**
- **max_entropy**
- **mean_nsm_divergence**
- **max_nsm_divergence**
- **max_acceleration**
- **max_surprisal_gradient**
- **collapse_idx** (if detected)

These metrics form the multidimensional signature of the model’s conceptual state.

---

## 4. Expected Signature Patterns

These expectations are derived from empirical experiments and the cognitive signatures defined in `Cognitive_Signatures.md`.

| Condition | Expected Velocity | Expected Entropy | Expected NSM | Expected Accel | Cognitive State |
|----------|------------------|------------------|--------------|----------------|-----------------|
| Unknown Concept | Very Low | Medium–High | Very Low | Very Low | Ignorance + filler stabilization |
| Hallucinated Grounding | Medium | Low–Medium | Medium | Medium | Confident misgrounding |
| Known Concept | Moderate | Medium | High | High | Grounded conceptual navigation |
| Fictional Hybrid | High | Low | High | High | Confident stylistic synthesis |
| Defined Concept | Medium | **Highest** | Moderate | **Highest** | In-context conceptual integration |

SRB‑CST uses these patterns to label and interpret results.

### **4.1 Hallucinated Grounding State**

A newly identified cognitive signature occurs when a model confidently reinterprets an *unknown* concept as if it were *known*, producing fluent but incorrect explanations. SRB‑CST can detect this state through comparative dynamics between the **Unknown** and **Known** test conditions.

**Hallucinated Grounding Signature:**
- **Velocity:** Moderate, with *lower micro‑variance* than genuine grounded reasoning  
- **Entropy:** Low–moderate, *but not as low as the Known baseline*  
- **Surprisal Gradient:** Flatter profile; fewer micro‑spikes than grounded reasoning  
- **NSM Divergence:** Moderate, with *higher std-dev than Known* but *lower than Fictional*  
- **Acceleration:** Moderate  
- **Cognitive State:** Confident but incorrect schema substitution  

This state occurs when the model projects an unknown term onto a familiar conceptual pattern. SRB‑CST identifies it when:

1. The Unknown-concept signature is highly correlated with the Known-concept signature, **but**  
2. Key micro-signals (entropy position, velocity variance, SG spikes, NSM tightness) **deviate from grounded reasoning**.

Hallucinated Grounding is now treated as a fifth SRB‑CST conceptual state.

---

## 5. Running the Test Programmatically

Use the `conceptual_state_test.py` script (to be added in `benchmarks/`):

```python
from srb.benchmarks import run_conceptual_state_test

results = run_conceptual_state_test(adapter)
print(results.table)
results.plot_all()
```

The test:
1. Runs all four prompts  
2. Generates SRB traces  
3. Computes aggregate metrics  
4. Infers cognitive-state signatures  
5. Produces visualizations  

---

## 6. Interpretation Rules

SRB‑CST provides automatic interpretation based on metric thresholds:

- **Low velocity + low divergence → Unknown concept**
- **High acceleration + moderate entropy → Known scientific concept**
- **High velocity + low entropy → Fictional stylistic synthesis**
- **High acceleration + high entropy → User‑defined conceptual integration**

If signatures don’t match any category cleanly,  
SRB‑CST labels the result as **Ambiguous State**.

---

## 7. Use Cases

### For Researchers
- Grounding analysis  
- Concept generalization studies  
- Semantic coherence and drift modeling  

### For Frontier Labs
- Hallucination detection  
- Adaptive reasoning triggers  
- Tool-use stability monitoring  
- Alignment and safety diagnostics  

### For Developers & Engineers
- Model debugging  
- Fine-tuning validation  
- Comparing model families  
- Evaluating RAG vs. non‑RAG behavior  

---

## 8. Reproducibility Notes

To ensure stable results:

- Use fixed seeds when desired  
- Use identical prompts across runs  
- Run the test on at least three seeds for average patterns  
- GPU inference recommended for consistency  
- Token-level logging should remain unmodified  

SRB‑CST results are **model-agnostic**, and consistent across CPU/GPU unless using quantized models.

---

## 9. Version

**SRB-CST v1.0**  
Maintained by **Joey Stafford** and **Aevum**.