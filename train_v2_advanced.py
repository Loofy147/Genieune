import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import OneCycleLR

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.1 Advanced)
Featuring Rotary Positional Embeddings (RoPE) and Dynamic Recurrence Control.
Self-contained script for Kaggle with optimized training.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: ADVANCED ARCHITECTURAL COMPONENTS
# ══════════════════════════════════════════════════════════════════

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]].to(x.device)
    # Expand freqs_cis to match batch and head dimensions
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

class AdvancedGenuineAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis):
        batch, seq, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(batch, seq, self.n_heads, self.head_dim)
        k = k.view(batch, seq, self.n_heads, self.head_dim)
        v = v.view(batch, seq, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Scaled Dot-Product Attention
        # (batch, heads, seq, seq)
        attn_scores = torch.einsum("bihd,bjhd->bihj", q, k) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Entropy Monitoring for G-score calculation
        # Shannon Entropy H(A) = -sum(p * log2(p))
        entropy = -torch.sum(attn_weights * torch.log2(attn_weights + 1e-9), dim=-1)

        out = torch.einsum("bihj,bjhd->bihd", attn_weights, v)
        out = out.reshape(batch, seq, -1)
        return self.wo(out), attn_weights, entropy

class AdvancedGenuineLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = AdvancedGenuineAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, freqs_cis):
        res = x
        x, weights, entropies = self.attn(self.ln1(x), freqs_cis)
        x = x + res
        x = x + self.mlp(self.ln2(x))
        return x, weights, entropies

class AdvancedGenuineTransformer(nn.Module):
    """
    Version 2.1: Includes RoPE and Dynamic Recurrence Logic.
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=6, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([AdvancedGenuineLayer(d_model, n_heads) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, 128) # Max seq len 128

    def forward(self, x, g_threshold=0.6, max_loops=2):
        x = self.embedding(x)
        all_entropies = []

        # Dynamic Recurrence Logic
        # Reasoning blocks (first half) can loop if G-score is low
        reasoning_layers = len(self.layers) // 2

        for loop in range(max_loops):
            loop_entropies = []
            for i in range(reasoning_layers):
                x, attn, entropies = self.layers[i](x, self.freqs_cis)
                loop_entropies.append(entropies)

            # Simplified G-score: Layer Variance
            current_g = torch.stack([torch.var(e, dim=-1).mean() for e in loop_entropies]).mean()
            all_entropies.extend(loop_entropies)

            if current_g >= g_threshold:
                break # Thought completed

        # Decoding blocks (second half)
        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, self.freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

# ══════════════════════════════════════════════════════════════════
# PART 2: ADVANCED THERMODYNAMIC REGULARIZER (V2.1)
# ══════════════════════════════════════════════════════════════════

class AdvancedRegularizer:
    def __init__(self, target_g=0.65, mechanical_penalty=0.4):
        self.target_g = target_g
        self.mechanical_penalty = mechanical_penalty

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=entropies[0].device)
        for head_ent in entropies:
            # 1. Sustained Variance Reward
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 1.0 * var_h

            # 2. Premature Collapse Penalty
            # Penalize sudden drops to low entropy (Mechanical state)
            mean_h = head_ent.mean()
            if mean_h < self.mechanical_penalty:
                total_loss = total_loss + torch.pow(self.mechanical_penalty - mean_h, 2) * 5.0

        return total_loss

# ══════════════════════════════════════════════════════════════════
# PART 3: ADVANCED TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Advanced V2.1 on {device}")

    # Configuration
    batch_size = 16
    seq_len = 16
    n_epochs = 10000
    vocab_size = 1000

    # Initialize
    model = AdvancedGenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=vocab_size).to(device)
    regularizer = AdvancedRegularizer()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    def get_complex_batch():
        # Task: Selective Reversal and Increment
        # If token is even, next token is incremented
        # If token is odd, next token is decremented
        data = torch.randint(10, 900, (batch_size, seq_len)).to(device)
        target = torch.where(data % 2 == 0, (data + 1) % vocab_size, (data - 1) % vocab_size)
        return data, target

    for epoch in range(1, n_epochs + 1):
        model.train()
        data, target = get_complex_batch()

        optimizer.zero_grad()
        logits, entropies = model(data)

        task_loss = criterion(logits.view(-1, vocab_size), target.view(-1))

        # Flatten entropies [layer][batch, seq, head]
        flat_entropies = []
        for layer_ent in entropies:
            # layer_ent is (batch, head, seq) from SDA logic above
            # reshaped for regularizer
            flat_entropies.append(layer_ent)

        thermo_loss = regularizer.calculate_loss(flat_entropies)
        total_loss = task_loss + 0.15 * thermo_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{n_epochs} | Task: {task_loss.item():.4f} | Thermo: {thermo_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Advanced Training Complete.")
    torch.save(model.state_dict(), "advanced_genuine_model_v2_1.pt")

if __name__ == "__main__":
    train()
