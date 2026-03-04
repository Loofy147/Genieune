# Dynamic Entropy Genuineness Framework (Version 2.1 Refined)

This project implements a structural evolution of the Transformer architecture, moving beyond simple "Chain-of-Thought" behavioral hacks towards internal, mechanistic reasoning.

## Core Innovations

### 1. Dynamic Genuineness (G-score)
Instead of relying on token prediction accuracy alone, this framework evaluates model states using the **G-score**—a metric derived from the **Shannon Entropy variance** of attention patterns. High G-scores indicate genuine synthesis and reasoning, while low G-scores indicate mechanical pattern matching.

### 2. Mechanistic Recurrence & Elaboration Pull Detection
The **Genuine Transformer** (V2.1) architecture features a real-time monitoring system in its forward pass. If it detects an "Elaboration Pull" (a sudden drop in attention complexity), the state is dynamically routed back through reasoning layers to sustain its computational depth.

### 3. Thermodynamic Regularization
A custom loss function designed to "wire" the model for reasoning by rewarding high-variance attention states and penalizing premature entropy collapse.

## Components in this Kernel

- **GenuineTransformer**: A V2.1 architecture with RoPE and dynamic recurrence logic.
- **ThermodynamicRegularizer**: The advanced loss function for Sustained Genuineness training.
- **Analysis Pipeline**: Evaluates input prompts across the 2D Genuineness Phase Space.
- **Complex Reasoning Task**: A training task that requires non-trivial parity-based logic across sequences.

## How to use

1. **Run Training**: Execute `train_v2_advanced.py` to observe the model evolving its reasoning capacity via thermodynamic penalties.
2. **Run Analysis**: Execute `kaggle_analysis.py` to analyze specific prompts and visualize their trajectory in the Genuineness Phase Space.

---
*This project represents a step towards models that 'think' internally rather than just 'talking' predictably.*
