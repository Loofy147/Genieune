# Dynamic Entropy Genuineness Framework (Version 1.0)

This repository implements the **Dynamic Entropy Genuineness Framework**, a mechanistic interpretability approach for analyzing Transformer models. The framework maps attention heads into a 2D Phase Space to distinguish between mechanical pattern matching and genuine computation.

## Core Metrics

The framework evaluates attention heads using two primary axes:

### 1. Token Cost (X-Axis)
Represents the "external anchor" or the information density being processed.
- **Definition**: Weighted average surprisal of the tokens attended to by the head.
- **Surprisal**: $I(t) = -\log_2 P(t)$
- **Calculation**: $X_{head} = \sum_{i,j} A_{i,j} \cdot I(j)$, where $A$ is the attention pattern.

### 2. Dynamic Genuineness (Y-Axis)
Measures the internal complexity and "collapse" dynamics of the head's attention.
- **Definition**: $Y = \text{Var}(H) + \frac{\text{Collapse Count}}{L}$
- **Shannon Entropy (H)**: $H(i) = -\sum_j A_{i,j} \log_2(A_{i,j})$
- **Collapse Event**: A sudden drop in entropy between adjacent tokens, $\Delta H < -0.20$.

## Phase Space Quadrants

| Quadrant | Thresholds | Typical Archetype |
| :--- | :--- | :--- |
| **GENUINE_DIFFUSE** | $Y \ge 0.55, X < 0.5$ | Name Mover Heads (Logic Engines) |
| **MECHANICAL_COMMITTED** | $Y \le 0.35, X \ge 0.5$ | Induction Heads (Pattern Retrieval) |
| **MECHANICAL_DIFFUSE** | $Y \le 0.35, X < 0.5$ | Broadcast / Uniform Attention |
| **GENUINE_COMMITTED** | $Y \ge 0.55, X \ge 0.5$ | Rare in trained weights (High cost logic) |

## Differential Dynamics (Circuit Asymmetry)

The framework models the "Elaboration Pull"—the decay of genuine computation into pattern repetition—using empirical differential equations:

- **Degradation (Pattern pull)**: $\frac{dG}{dt} = -0.8129 \cdot G$
- **Recovery (Genuine computation)**: $\frac{dG}{dt} = +1.2371 \cdot (G_{max} - G)$

## Installation & Usage

### Dependencies
- `transformer-lens`
- `torch`
- `numpy`
- `scipy`
- `matplotlib`

### Running Analysis
```bash
PYTHONPATH=. python3 phase_dynamics.py
```

## Sustained Genuineness (Version 2.0 Architectural Blueprint)

*Note: Version 2.0 components are currently implemented as architectural simulations and prototypes.*

To move beyond "Chain-of-Thought" behavioral hacks, the framework proposes a structural evolution of the Transformer architecture.

### 1. Thermodynamic Regularization
A new loss function that rewards $Var(H)$ and `collapse_count` in reasoning blocks while penalizing static, low-entropy attention patterns. This wires the model to favor deep synthesis over shallow retrieval.

### 2. Dual-Stream Decoupling
Isolates "Thinking" from "Talking" via a structural split:
- **Latent Reasoner (Layers 1-32)**: Processes pure math vectors in high-variance states ($G > 0.55$).
- **Phase-Space Gate**: Monitors $G$ score and holds the activation until a final collapse event occurs.
- **Syntax Decoder (Layers 33-40)**: Dedicated blocks of pattern heads translate logic into English syntax only after the thought is complete.

### 3. Mechanistic Recurrence
Instead of fixed-compute forward passes, the model uses real-time monitoring of the $G$ score. If an **Elaboration Pull** (drop in $G$ score) is detected within the logic block, the activation is routed back to the beginning of the reasoning layers (Layer 21), forcing a re-diffusion of attention.

## Testing
```bash
PYTHONPATH=. python3 tests/test_phase_dynamics.py
PYTHONPATH=. python3 tests/test_v2_prototypes.py
```
