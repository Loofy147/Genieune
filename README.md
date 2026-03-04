# Dynamic Entropy Genuineness Framework (Version 2.1 Refined)

This repository implements the **Dynamic Entropy Genuineness Framework**, a mechanistic interpretability and architectural evolution approach for analyzing and training Transformers. The framework uses a 2D Phase Space to distinguish between mechanical pattern matching and genuine computation.

## Core Metrics

The framework evaluates attention heads and model trajectories using two primary axes:

### 1. Token Cost (X-Axis)
Represents the "external anchor" or the information density being processed.
- **Definition**: Weighted average surprisal of the tokens attended to by the head.
- **Surprisal**: $I(t) = -\log_2 P(t)$
- **Calculation**: $X_{head} = \sum_{i,j} A_{i,j} \cdot I(j)$

### 2. Dynamic Genuineness (Y-Axis / G-score)
Measures the internal complexity and attention variance.
- **Definition**: $Y = \text{Var}(H) + \text{Collapse Count} / L$
- **Shannon Entropy (H)**: $H(i) = -\sum_j A_{i,j} \log_2(A_{i,j})$
- **Collapse Event**: Sudden drop in entropy between layers or tokens, $\Delta H < -0.20$.

## Full Version 2.1 Implementation

The repository has transitioned from simulations to a functional **Genuine Transformer** (V2.1) architecture with the following features:

### 1. Rotary Positional Embeddings (RoPE)
Integrated relative positional encoding for enhanced sequence modeling and attention stability.

### 2. Mechanistic Recurrence (Dynamic Routing)
The model monitors the **G-score** (entropy variance) in real-time during the forward pass.
- **Elaboration Pull Detection**: If the G-score drops significantly between reasoning layers, the hidden state is routed back through the reasoning block to "re-diffuse" attention and stabilize the thought.
- **Dynamic Exit**: The model exits the reasoning loop once a sustained high G-score threshold is met.

### 3. Thermodynamic Regularization
An advanced loss function that:
- Rewards **high attention variance** (internal complexity).
- Penalizes **static, low-entropy states** (mechanical pattern matching).
- Penalizes **premature entropy collapse** between layers.

## Repository Structure

- `genuine_model.py`: Core architecture (Version 2.1) including RoPE, Recurrence, and the Regularizer.
- `sustained_genuineness.py`: Logic utilities for G-score tracking and routing.
- `kaggle_analysis.py`: Full analysis pipeline. Evaluates prompts using V1 interpretability and V2.1 model metrics.
- `train_v2_advanced.py`: Advanced training pipeline for V2.1 models on complex reasoning tasks.
- `phase_dynamics.py`: Original V1 mechanistic interpretability tools.
- `AGENTS.md`: Technical instructions and codebase overview.

## Installation & Usage

### Dependencies
- `transformer-lens`, `torch`, `numpy`, `scipy`, `matplotlib`

### Running Analysis
```bash
python3 kaggle_analysis.py
```
This script evaluates a prompt, generates Phase Space plots for GPT-2 (V1), and runs the refined V2.1 model to track its G-trajectory and thermodynamic loss.

### Running Tests
```bash
PYTHONPATH=. python3 -m unittest discover tests
```

### Training
```bash
python3 train_v2_advanced.py
```

## Kaggle Deployment
Automated via `deploy_kaggle.sh`. Requires `KAGGLE_API_TOKEN` environment variable and `kaggle` CLI.

## Phase Space Quadrants (V1)

| Quadrant | Typical Archetype |
| :--- | :--- |
| **GENUINE_DIFFUSE** | Name Mover Heads (Logic Engines) |
| **MECHANICAL_COMMITTED** | Induction Heads (Pattern Retrieval) |
| **MECHANICAL_DIFFUSE** | Broadcast / Uniform Attention |
| **GENUINE_COMMITTED** | High-cost reasoning (Rare) |

---
*Developed under the Dynamic Entropy Genuineness Framework (Version 2.1).*

## Roadmap & Implementation Checklist

### 🚀 Roadmap
- [ ] **V2.2: Advanced Recurrence Control**: Implement adaptive gating for the reasoning loop based on gradient-signal monitoring.
- [ ] **V3.0: Latent Reasoner Decoupling**: Fully separate the latent reasoning space from the syntax decoder, using a cross-attention bridge.
- [ ] **Interpretability Suite**: Add support for path-patching and logit lens visualization within the Genuineness Phase Space.
- [ ] **G-score Stability Optimization**: Refine the Thermodynamic Regularizer to handle longer sequences and prevent training instabilities.

### ✅ Implementation Checklist
- [ ] **Architecture Check**: Ensure RoPE frequency precomputation matches the target sequence length.
- [ ] **Training Stability**: Verify the Thermodynamic loss weight ($\lambda$) is tuned to prevent gradient explosion.
- [ ] **Analysis Validation**: Confirm `kaggle_analysis.py` produces both V1 Phase Space and V2 G-trajectory plots.
- [ ] **Recurrence Verification**: Test that the `max_loops` parameter is correctly bounded in the model configuration.
