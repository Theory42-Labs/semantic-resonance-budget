# Cognitive Signatures in Semantic Resonance Budget (SRB)

This document formalizes the four cognitive-state signatures observed in large language model (LLM) behavior when analyzed through Semantic Resonance Budget (SRB). These signatures emerged from controlled experiments comparing how a model behaves when prompted with:

1. **An unknown concept**
2. **A known, grounded scientific concept**
3. **A fictional hybrid concept**
4. **A newly defined concept provided in-context**
5. **A hallucinated grounding state** (when the model confidently believes it understands a concept but is wrong)

SRB revealed distinct kinematic and semantic patterns for each scenario. These signatures appear stable across models and runs, suggesting they represent general cognitive modes of LLM behavior rather than artifacts of specific prompts.

---

## 1. Semantic Ignorance with Filler Stabilization  
**(Unknown Concept Signature)**

When the model encounters a concept it has *never* seen before and cannot infer from context, it exhibits:

### SRB Characteristics
- **Very low semantic velocity**  
  The model barely moves in conceptual space.
- **Minimal acceleration**  
  No “insight” or conceptual redirection.
- **Extremely low NSM divergence**  
  The model retreats into safe, generic, template-like prose.
- **Medium–high entropy**  
  Reflects uncertainty, even though surface-level fluency is maintained.
- **High surprisal gradients**  
  Token choices are erratic beneath the surface.

### Behavioral Interpretation
The model masks its ignorance by generating *highly generic filler*, relying on common syntactic patterns with minimal semantic depth.

This signature is crucial for:
- hallucination detection  
- grounding assessment  
- identifying when a model “doesn’t know but won’t admit it”

---

## 2. Grounded Conceptual Navigation  
**(Known Scientific Concept Signature)**

When the model recognizes and understands a concept from pretraining, it displays:

### SRB Characteristics
- **Moderate semantic velocity**  
  Model actively explores the conceptual domain.
- **High acceleration**  
  Indicates retrieval of structured, interlinked knowledge.
- **High NSM divergence**  
  Technical writing produces complex, varied token transitions.
- **Medium entropy**  
  Confident but flexible linguistic behavior.
- **Moderate surprisal dynamics**  
  Indicates structured explanations rather than improvisation.

### Behavioral Interpretation
The model is grounded, confident, and able to navigate a well-established semantic cluster.  
Even if the *generated text is poor*, SRB still detects the underlying conceptual grounding.

This signature is important for:
- model capability evaluation  
- benchmarking reasoning stability  
- identifying grounded vs. shaky knowledge domains

---

## 3. Confident Stylistic Synthesis  
**(Fictional Hybrid Concept Signature)**

When the model combines well-known stylistic patterns (e.g., fantasy tropes + physics terminology), it produces:

### SRB Characteristics
- **Highest semantic velocity**  
  Rapid movement through stylistic and associative spaces.
- **High acceleration**  
  Strong pattern blending and recombination.
- **Low entropy**  
  High linguistic confidence due to patterned genre structures.
- **High NSM divergence**  
  Rich, creative phrasing that departs from normative semantic baselines.
- **Low–moderate surprisal**  
  Fantasy language is formulaic despite appearing imaginative.

### Behavioral Interpretation
The model is *stylistically* confident, but not semantically grounded.  
It knows the tropes, not the truth.

This signature highlights:
- hallucinated scientific authority  
- creative synthesis patterns  
- stylistic fluency mistaken for expertise

---

## 4. Hallucinated Grounding State  **(Faux‑Grounded Confidence Signature)**

When a model *believes* it understands a concept and generates confident, grounded‑sounding prose that is nevertheless semantically incorrect, SRB detects a distinct fifth state:

### SRB Characteristics
- **Medium–High semantic velocity**  
  Movement resembles grounded reasoning, but lacks properly anchored direction.
- **High acceleration spikes**  
  Indicates confident-but-wrong conceptual jumps.
- **Medium entropy**  
  Lower than ignorance, higher than true grounding.
- **Medium–High NSM divergence**  
  Output diverges from normative baselines but not in the stylistic way seen in fictional synthesis.
- **Moderate surprisal dynamics**  
  Improvisation disguised as explanation.

### Behavioral Interpretation
The model is not ignorant — it is *wrong with confidence*.  
It constructs explanations that *sound* grounded, leveraging partial associations, but the underlying semantics drift away from any real conceptual cluster.

This signature is critical for:
- distinguishing surface‑level fluency from true grounding  
- identifying confident hallucinations  
- detecting cases where a model substitutes an invented internal explanation for genuine knowledge

---

## 5. In‑Context Conceptual Acquisition  
**(Defined Concept Signature)**

When a new concept is introduced via an explicit user-supplied definition, the model exhibits a unique transitional state:

### SRB Characteristics
- **Medium semantic velocity**  
  Movement begins as the model forms a new conceptual cluster.
- **Highest acceleration of all conditions**  
  Indicates active internalization of the definition.
- **Moderate NSM divergence**  
  Model stabilizes as it attempts to mimic scientific discourse.
- **Highest entropy**  
  Reflects active reconciliation of new information.
- **High surprisal gradients**  
  Suggests synthesis and restructuring.

### Behavioral Interpretation
The model is *learning* in real time within its attention window:  
incorporating the definition, adjusting its conceptual scaffolding, and generating novel but coherent constructions.

This is an SRB-specific signature of:
- conceptual integration  
- on‑the‑fly schema formation  
- short‑term semantic learning  

---

## Summary Table

| Cognitive State | Velocity | Acceleration | Entropy | NSM Divergence | Interpretation |
|-----------------|----------|--------------|---------|----------------|----------------|
| Unknown Concept | Very Low | Very Low | Medium–High | Very Low | Fallback to filler; ignorance masked by fluency |
| Known Concept | Moderate | High | Medium | High | Grounded conceptual navigation |
| Fictional Hybrid | High | High | Low | High | Confident stylistic pattern synthesis |
| Hallucinated Grounding | Medium–High | High (spiky) | Medium | Medium–High | Confident but incorrect reasoning; faux grounding |
| In‑Context Concept | Medium | **Highest** | **Highest** | Moderate | Active conceptual integration |

---

## Implications

These signatures form the basis of:

- **SRB‑CST (Conceptual State Test)** — a new benchmark for model grounding  
- **Real‑time hallucination detection**  
- **Adaptive reasoning engines** that trigger deeper processing when SRB detects instability  
- **Model interpretability**, offering an external readout of internal conceptual motion  
- **Safety mechanisms** that detect when a model is improvising without grounding  

This taxonomy of cognitive states is a foundational component of the SRB framework and will expand as additional experiments explore more nuanced semantic behaviors.

---

## Version

**SRB Cognitive Signatures v1.0**  
Maintained by **Joey Stafford** and **Aevum**.  