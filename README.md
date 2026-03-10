---
title: Dynamic Entropy Genuineness Framework V2.2
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
language:
- en
tags:
- mechanistic-interpretability
- genuineness
- transformer
- dynamic-entropy
metrics:
- genuineness-score
- token-cost
---

# Dynamic Entropy Genuineness Framework (Version 2.2 Advanced)

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

## Version 2.2 Advanced Implementation

The framework has evolved into **Version 2.2 Advanced**, featuring:

### 1. Learned Genuineness Gate (Adaptive Recurrence)
A neural gating mechanism that monitors hidden states and entropy to determine if additional reasoning passes are required.

### 2. Global G-Budgeting
Computational expenditure management across reasoning steps to ensure efficiency and stability.

### 3. Layer-Wise Thermodynamic Regularization
An advanced loss function that:
- Rewards **high attention variance** (internal complexity).
- Penalizes **static, low-entropy states** (mechanical pattern matching).
- Implements **layer-wise decay** to prioritize complexity in early reasoning layers.

## Repository Structure

- `genuine_model.py`: Version 2.2 architecture (Gating, Budgeting, Regularizer).
- `sustained_genuineness.py`: Logic utilities for G-score tracking.
- `kaggle_analysis.py`: Main analysis pipeline for evaluating prompt trajectories.
- `train_v2_advanced.py`: Advanced training for V2.2 on Contextual Parity Pointer tasks.
- `app.py`: Interactive Streamlit UI for the Hugging Face Space.
- `V2_2_TECHNICAL_REPORT.md`: Detailed theory and experimental results.

## Installation & Usage

### Dependencies
- `transformer-lens`, `torch`, `numpy`, `scipy`, `matplotlib`, `streamlit`

### Running the UI
```bash
streamlit run app.py
```

### Training
```bash
python3 train_v2_advanced.py
```

## Phase Space Quadrants (V1)

| Quadrant | Typical Archetype |
| :--- | :--- |
| **GENUINE_DIFFUSE** | Name Mover Heads (Logic Engines) |
| **MECHANICAL_COMMITTED** | Induction Heads (Pattern Retrieval) |
| **MECHANICAL_DIFFUSE** | Broadcast / Uniform Attention |
| **GENUINE_COMMITTED** | High-cost reasoning (Rare) |

---
*Developed under the Dynamic Entropy Genuineness Framework (Version 2.2).*
