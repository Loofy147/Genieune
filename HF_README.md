---
license: apache-2.0
tags:
- mechanistic-interpretability
- transformer
- reasoning
- rope
metrics:
- entropy
---

# Dynamic Entropy Genuineness Framework (Version 2.1 Refined)

This model card covers the **Dynamic Entropy Genuineness Framework V2.1**, which implements a structural evolution of the Transformer architecture to favor internal reasoning over mechanical pattern matching.

## Model Description

- **Architecture**: Genuine Transformer (V2.1 Refined)
- **Key Features**: Rotary Positional Embeddings (RoPE), Mechanistic Recurrence (Dynamic Routing), and Thermodynamic Regularization.
- **Goal**: Sustained genuineness in attention states, monitored via Shannon Entropy variance (G-score).

## Intended Use

Evaluating the reasoning depth of Transformer models and exploring architectural variants that sustain "Genuine States" through dynamic loops.

## How to use

```python
import torch
from genuine_model import GenuineTransformer

model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=1000)
model.load_state_dict(torch.load("advanced_genuine_model_v2_1.pt", map_location="cpu"))
model.eval()

# Run inference with dynamic recurrence
tokens = torch.randint(0, 1000, (1, 16))
logits, entropies = model(tokens, g_threshold=0.6, max_loops=3)
```

## Source and Provenance

- **Original Implementation**: [hichambedrani](https://huggingface.co/LOOFYYLO)
- **Framework**: Dynamic Entropy Genuineness Framework
- **Public Notebook**: [Genuine Transformer V2.1 Advanced Training](https://www.kaggle.com/code/hichambedrani/genuine-transformer-v2-advanced)
- **Kaggle Model**: [Dynamic Entropy Genuineness V2.1](https://www.kaggle.com/models/hichambedrani/dynamic-entropy-genuineness-v2-1)
