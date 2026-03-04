import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.0 Consolidated)
Self-contained script for Kaggle execution with extended training duration.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: GENUINE TRANSFORMER ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class GenuineAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)
        self.head_dim = head_dim

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Shannon Entropy H(A) = -sum(p * log2(p))
        entropy = -torch.sum(attn_weights * torch.log2(attn_weights + 1e-9), dim=-1)

        out = torch.matmul(attn_weights, v)
        return out, attn_weights, entropy

class GenuineLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.heads = nn.ModuleList([GenuineAttentionHead(d_model, self.head_dim) for _ in range(n_heads)])
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.ln1(x)

        head_outputs = [h(x) for h in self.heads]
        head_outs = [out[0] for out in head_outputs]
        attn_weights = [out[1] for out in head_outputs]
        entropies = [out[2] for out in head_outputs]

        x = torch.cat(head_outs, dim=-1)
        x = x + res

        x = x + self.mlp(self.ln2(x))
        return x, attn_weights, entropies

class GenuineTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_reasoner_layers=2, n_decoder_layers=2, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.reasoner = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_reasoner_layers)])
        self.decoder = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, use_recurrence=False, max_loops=3):
        x = self.embedding(x)
        all_entropies = []

        for loop in range(max_loops):
            loop_entropies = []
            for i, layer in enumerate(self.reasoner):
                x, attn, entropies = layer(x)
                loop_entropies.append(entropies)

                if use_recurrence and loop < max_loops - 1:
                    # Detect Elaboration Pull (Simplified G check)
                    layer_g = torch.stack([torch.var(e, dim=-1).mean() for e in entropies]).mean()

            all_entropies.extend(loop_entropies)
            if not use_recurrence: break

        for layer in self.decoder:
            x, attn, entropies = layer(x)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

# ══════════════════════════════════════════════════════════════════
# PART 2: THERMODYNAMIC REGULARIZER
# ══════════════════════════════════════════════════════════════════

class ThermodynamicRegularizer:
    def __init__(self, genuine_threshold=0.55, mechanical_threshold=0.35):
        self.genuine_threshold = genuine_threshold
        self.mechanical_threshold = mechanical_threshold

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=entropies[0].device)
        for head_ent in entropies:
            # Variance Reward
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 0.5 * var_h

            # Static Penalty
            mean_h = head_ent.mean()
            if mean_h < self.mechanical_threshold:
                total_loss = total_loss + torch.pow(self.mechanical_threshold - mean_h, 2)
        return total_loss

# ══════════════════════════════════════════════════════════════════
# PART 3: TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Initialize Model & Regularizer
    model = GenuineTransformer(d_model=128, n_heads=4, n_reasoner_layers=2, n_decoder_layers=2).to(device)
    regularizer = ThermodynamicRegularizer()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def get_batch(batch_size=8, seq_len=10):
        data = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
        target = (data + 1) % 1000
        return data, target

    # 3. Training Loop (Significantly extended duration)
    n_epochs = 5000
    for epoch in range(1, n_epochs + 1):
        model.train()
        data, target = get_batch()

        optimizer.zero_grad()
        logits, entropies = model(data)

        # Calculate standard loss
        task_loss = criterion(logits.view(-1, 1000), target.view(-1))

        # 4. Calculate Differentiable Thermodynamic Loss
        flat_entropies = []
        for layer_ent in entropies:
            flat_entropies.extend(layer_ent)

        thermo_loss = regularizer.calculate_loss(flat_entropies)

        # Combine losses: task_loss + lambda * thermo_loss
        total_loss = task_loss + 0.1 * thermo_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 250 == 0:
            print(f"Epoch {epoch}/{n_epochs} | Task Loss: {task_loss.item():.4f} | Thermo Loss: {thermo_loss.item():.4f}")

    print("Training Complete.")
    torch.save(model.state_dict(), "genuine_model_v2.pt")

if __name__ == "__main__":
    train()
