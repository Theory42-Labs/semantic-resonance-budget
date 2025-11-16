# Semantic Velocity

## Definition
Semantic Velocity is a measure of how quickly a language model moves through embedding space as it generates tokens. It captures the *rate of semantic change* between consecutive token embeddings, providing a kinematic view of the model’s internal reasoning trajectory.

Formally, given an embedding function $f$ that maps each generated token $t_i$ to its embedding $\mathbf{e}_i = f(t_i)$, Semantic Velocity $v_i$ at step $i$ is defined as:

```math
v_i = \frac{\lVert \mathbf{e}_i - \mathbf{e}_{i-1} \rVert}{\Delta t}
```

where:
- $\\lVert \cdot \\rVert$ denotes Euclidean or cosine distance,
- $\\Delta t$ is treated as 1 token-generation step.

## Intuition
Semantic Velocity describes how “fast” the model’s meaning representation is shifting.  
- **Low velocity** → the model is refining, elaborating, or staying on topic.  
- **Moderate velocity** → the model is progressing through a reasoning chain.  
- **High velocity** → the model may be shifting topics, making leaps, or entering unstable/hallucinatory states.

This metric reveals the *shape* of reasoning rather than just the output.

## Implementation (Pseudocode)

```
embeddings = []
velocities = []

for each token in generated_sequence:
    e = model.get_embedding(token)
    embeddings.append(e)

for i in range(1, len(embeddings)):
    delta = distance(embeddings[i], embeddings[i-1])
    velocities.append(delta)
```

## Visualization
Common ways to visualize Semantic Velocity:
- **Line plot** showing velocity over time  
- **Scatter plot** with velocity vs surprisal  
- **Overlay with entropy** to identify reasoning phases  

## Interpretive Patterns
Key recognizable patterns in real LLM behavior:

### 1. **Reasoning Phase**
- Smooth, steady velocities.
- Indicates stable exploration of semantic space.

### 2. **Decision Pivot**
- Sudden spike followed by stabilization.
- Often corresponds to the model committing to an answer.

### 3. **Hallucination Drift**
- Chaotic high-variance velocity.
- Often coincides with rising entropy and NSM divergence.

### 4. **Loop or Stall**
- Velocity approaches zero.
- Indicates repetition, stuck states, or collapsed search.

## Applications
Semantic Velocity enables:
- Detection of early hallucination
- Dynamic inference budgeting (stop when velocity stabilizes)
- Understanding model “thought motion”
- Comparative analysis across models and interventions
- Identification of internal reasoning phases

This is one of the core metrics in SRB because it exposes structure that is invisible when only examining token probabilities.

# Semantic Entropy

## Definition
Semantic Entropy measures the uncertainty in the model’s token probability distribution at each generation step. It reflects how many semantic possibilities the model is considering at a given moment and quantifies the “energy” of its reasoning dynamics.

Formally, given a probability distribution over the next-token candidates $P(t_k \mid \text{context})$, Semantic Entropy $H_i$ at step $i$ is defined as:

```math
H_i = - \sum_{k} P(t_k \mid \text{context}) \log P(t_k \mid \text{context})
```

## Intuition
Semantic Entropy reveals how the model navigates uncertainty during generation.

- **Low entropy** → The model is confident, focused, and committed to a narrow semantic trajectory.
- **Moderate entropy** → The model is exploring multiple semantic possibilities.
- **High entropy** → The model is uncertain, unstable, or drifting into unpredictable territory.
- **Entropy collapse** → A sudden drop often marks a decision point or reasoning pivot.
- **Entropy spike** → The model temporarily reevaluates its internal direction or hesitates.

Entropy provides a complementary view to Semantic Velocity: if velocity shows *movement*, entropy shows *tension*.

## Implementation (Pseudocode)

```
entropies = []

for each token_step in generation:
    probs = model.get_token_probabilities()
    H = 0
    for p in probs:
        if p > 0:
            H += -p * log(p)
    entropies.append(H)
```

## Visualization
Useful ways to visualize Semantic Entropy include:
- **Line plot** of entropy over time
- **Overlay with Semantic Velocity** to observe phase relationships
- **Entropy vs surprisal** scatter plots for uncertainty analysis
- **Entropy + NSM divergence** overlays to identify hallucination states

## Interpretive Patterns
Common patterns that appear in real LLM behavior:

### 1. **Reasoning Chain**
Entropy gradually decreases as the model narrows semantic options.

### 2. **Decision Pivot**
Entropy collapses sharply, often aligned with a spike in surprisal or Semantic Velocity.

### 3. **Hallucination Drift**
Entropy fluctuates chaotically or rises steadily, especially when NSM divergence increases.

### 4. **Loop or Degenerate Mode**
Entropy approaches zero while output becomes repetitive—indicating collapse rather than confidence.

## Applications
Semantic Entropy enables:
- Real-time uncertainty tracking
- Early detection of hallucination-like states
- Dynamic resource allocation (longer or shorter reasoning chains)
- Confidence-aware output scoring
- Cross-model comparison of stability and reasoning behavior

Semantic Entropy is a foundational SRB metric because it reveals internal uncertainty modulation that is invisible from token sequences alone.

# Surprisal Gradient

## Definition
The Surprisal Gradient measures the rate of change in token-level surprisal across consecutive generation steps. Surprisal reflects how unexpected a token is under the model’s internal probability distribution, and its gradient reveals how abruptly the model shifts its internal expectations.

Given token surprisal at step $i$ defined as:

```math
S_i = - \log P(t_i \mid \text{context})
```

the Surprisal Gradient is the discrete temporal derivative:

```math
\frac{dS}{dt}_i = S_i - S_{i-1}
```

## Intuition
Surprisal captures how “surprised” the model is by its own next token.  
The Surprisal Gradient captures how quickly that surprise is *changing*.

- **Small gradient** → Stable internal expectations, coherent flow.
- **Large positive gradient** → Sudden uncertainty or semantic shift.
- **Large negative gradient** → Decisive disambiguation or commitment.
- **Oscillatory gradients** → Internal hesitation, reevaluation, or unstable reasoning.

This metric often aligns with internal decision boundaries and cognitive pivot points.

## Implementation (Pseudocode)

```
surprisals = []
gradients = []

for each generated token:
    p = model.get_probability(token)
    S = -log(p)
    surprisals.append(S)

for i in range(1, len(surprisals)):
    dS = surprisals[i] - surprisals[i-1]
    gradients.append(dS)
```

## Visualization
Effective visual representations include:
- **Line plot** of surprisal and its gradient together
- **Gradient spikes overlaid with entropy collapse**
- **Scatter plot** of gradient vs Semantic Velocity
- **Gradient heatmaps** for long outputs or multi-sample comparisons

## Interpretive Patterns

### 1. **Decision Pivot**
A sharp negative gradient corresponds to the model resolving uncertainty and committing to a direction.

### 2. **Uncertainty Surge**
A sudden positive gradient indicates increased confusion, ambiguity, or destabilization.

### 3. **Reasoning Plateau**
A near-zero gradient for many steps suggests a steady, coherent reasoning chain.

### 4. **Hallucination Onset**
Chaotic gradient oscillations frequently appear before or during hallucinated content, especially when paired with rising entropy or NSM divergence.

## Applications
The Surprisal Gradient supports:
- Detection of reasoning transitions
- Early warning signals of hallucination
- Real-time dynamic inference adjustments
- Multi-model comparison of stability and decision making
- Identification of internal “thought boundary” events

The Surprisal Gradient is a high-sensitivity metric that captures microstructure in model behavior that remains invisible when examining token probabilities alone.

# NSM Divergence (Coherence Drift)

## Definition
NSM Divergence measures how far the model’s current semantic trajectory deviates from a baseline “normative” semantic flow. It quantifies how much the model’s output at each generation step drifts away from typical, high-probability semantic continuations.

Given:
- an embedding for the generated token at step $i$, $\mathbf{e}_i$
- a reference embedding representing the expected semantic direction, $\mathbf{r}_i$

NSM Divergence is defined as:

```math
D_i = \lVert \mathbf{e}_i - \mathbf{r}_i \rVert
```

The reference $\mathbf{r}_i$ may be computed using:
- averaged embeddings from top-k probable next tokens  
- centroid of a normative semantic cluster  
- task-specific semantic baselines  

NSM Divergence does **not** measure correctness—only **semantic deviation** from expected flow.

## Intuition
NSM Divergence is the closest SRB metric to a “hallucination vector.”

- **Low divergence** → The model remains aligned with expected semantic structure.
- **Moderate divergence** → Creativity or contextual expansion.
- **High divergence** → Instability, hallucination, or semantic drift.
- **Sudden divergence spike** → Topic jump, reasoning failure, or confusion.
- **Divergence collapse** → Model realigns with coherent semantic direction.

NSM answers the question:  
**“Is the model staying on the semantic rails?”**

## Implementation (Pseudocode)

```
divergences = []

for each token_step:
    e = model.get_embedding(generated_token)
    r = compute_reference_embedding()  # baseline or top‑k centroid
    D = distance(e, r)
    divergences.append(D)
```

The reference baseline may be computed using:
- mean embedding of top‑k highest-probability tokens
- rolling average of past embeddings
- domain-specific curated vectors

## Visualization
Useful plots include:
- **Line plot** of divergence over time
- **Entropy + divergence overlay** (excellent hallucination detector)
- **Scatter plot** divergence vs surprisal gradient
- **Divergence heatmaps** for multi-sample comparison

## Interpretive Patterns

### 1. **Coherent Flow**
Low, smooth divergence across steps.

### 2. **Creative Expansion**
Moderate divergence with stable entropy and velocity.

### 3. **Hallucination Drift**
Divergence rises sharply while entropy fluctuates wildly.

### 4. **Semantic Collapse**
Divergence suddenly drops (realignment) after a spike.

### 5. **Topic Jump**
Large one-step spike, often paired with a positive surprisal gradient.

## Applications
NSM Divergence enables:
- Early hallucination detection  
- Real-time semantic stability scoring  
- Adaptive guardrails and output validation  
- Comparison of model stability across interventions  
- Semantic anomaly detection  

This metric is central to SRB because it captures *semantic deviation*, a dimension of model behavior invisible to token probabilities alone.

# Semantic Acceleration

## Definition
Semantic Acceleration measures the change in Semantic Velocity across consecutive generation steps. While Semantic Velocity captures how quickly the model moves through embedding space, Semantic Acceleration captures how that speed itself is shifting—revealing transitions between reasoning modes.

Given Semantic Velocity at step $i$, denoted as $v_i$, Semantic Acceleration is defined as:

```math
a_i = v_i - v_{i-1}
```

or equivalently as the discrete second derivative of the embedding trajectory:

```math
a_i = 
\lVert \mathbf{e}_i - \mathbf{e}_{i-1} \rVert 
\, - \,
\lVert \mathbf{e}_{i-1} - \mathbf{e}_{i-2} \rVert
```

Semantic Acceleration evaluates *curvature* in semantic movement.

## Intuition
Acceleration reveals the “phase changes” of model reasoning.

- **Low, stable acceleration** → Model is in a consistent reasoning flow.
- **Positive acceleration** → Model is speeding up, making leaps, or shifting topics.
- **Negative acceleration** → Model is stabilizing or converging onto a decision.
- **Large oscillations** → Instability, confusion, or hallucination‑adjacent behavior.

If Velocity shows *motion*, Acceleration shows *momentum of thought*.

## Implementation (Pseudocode)

```
accelerations = []

# assume velocities[] has already been computed
for i in range(1, len(velocities)):
    a = velocities[i] - velocities[i-1]
    accelerations.append(a)
```

Acceleration depends entirely on Semantic Velocity, making it a natural secondary metric.

## Visualization
Common visualization strategies:
- **Velocity + acceleration overlay** to show phase transitions
- **Acceleration spikes aligned with surprisal gradient spikes**
- **Scatter plot** acceleration vs entropy
- **Acceleration bands** to highlight stable vs unstable reasoning segments

## Interpretive Patterns

### 1. **Reasoning Phase**
Acceleration is near zero; the model maintains consistent semantic motion.

### 2. **Insight or Leap**
A strong positive acceleration spike—often when the model “jumps” to a conclusion.

### 3. **Convergence**
Negative acceleration as the model stabilizes and locks into a final answer.

### 4. **Instability / Hallucination Drift**
Irregular, chaotic acceleration often coincides with rising entropy or divergence.

### 5. **Looping or Degenerate Mode**
Acceleration collapses toward zero because velocity is near zero—indicative of semantic stagnation.

## Applications
Semantic Acceleration enables:
- Detection of reasoning phase transitions
- Identification of moments when the model “makes a leap”
- Real‑time stability assessment
- Comparison of reasoning smoothness across models
- Enhanced detection of hallucination onset when paired with velocity and entropy

Semantic Acceleration completes the kinematic trio (Position → Velocity → Acceleration), giving SRB a physics‑inspired view of the model’s semantic trajectory.