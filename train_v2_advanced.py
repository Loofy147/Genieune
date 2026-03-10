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
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.1 Advanced Training)
Self-contained script for Kaggle with RoPE, Recurrence, and Thermo-Regularization.
Refined Version: Complex Multi-Step Parity Task.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: CORE ARCHITECTURE COMPONENTS
# ══════════════════════════════════════════════════════════════════

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]]
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

class GenuineAttention(nn.Module):
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
        # [batch, seq, n_heads, head_dim]
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Matmul optimization: [batch, n_heads, seq, head_dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Scaled Dot-Product Attention using matmul
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Efficient Entropy Monitoring (Shannon Entropy H(A) = -sum(p * log_softmax(p)))
        # log2(x) = ln(x) / ln(2) approx ln(x) * 1.4427
        log_attn_weights = F.log_softmax(attn_scores, dim=-1)
        entropy = -torch.sum(attn_weights * log_attn_weights, dim=-1) * 1.44269504
        entropy = entropy.transpose(1, 2) # [batch, seq, n_heads]

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.wo(out), attn_weights, entropy

class GenuineLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = GenuineAttention(d_model, n_heads)
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

class GenuineTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        # Register freqs_cis as buffer for automatic device management
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // n_heads, 128))

    def forward(self, x, g_threshold=0.6, max_loops=3):
        x = self.embedding(x)
        all_entropies = []
        reasoning_layers = len(self.layers) // 2

        g_history = []
        for loop in range(max_loops):
            loop_entropies = []
            for i in range(reasoning_layers):
                x, attn, entropies = self.layers[i](x, self.freqs_cis)
                loop_entropies.append(entropies)

            current_g = torch.stack([torch.var(e, dim=-1).mean() for e in loop_entropies]).mean()
            all_entropies.extend(loop_entropies)
            g_history.append(float(current_g.detach()))

            if current_g >= g_threshold:
                break

            if len(g_history) >= 2:
                delta_g = g_history[-1] - g_history[-2]
                if delta_g < -0.15: continue

        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, self.freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

class ThermodynamicRegularizer:
    def __init__(self, mechanical_penalty=0.4, collapse_penalty=5.0):
        self.mechanical_penalty = mechanical_penalty
        self.collapse_penalty = collapse_penalty

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=entropies[0].device)
        prev_mean_h = None

        for head_ent in entropies:
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 1.0 * var_h

            mean_h = head_ent.mean()
            if mean_h < self.mechanical_penalty:
                total_loss = total_loss + torch.pow(self.mechanical_penalty - mean_h, 2) * self.collapse_penalty

            if prev_mean_h is not None:
                delta_h = mean_h - prev_mean_h
                if delta_h < -0.2:
                    total_loss = total_loss + torch.pow(delta_h, 2) * self.collapse_penalty
            prev_mean_h = mean_h

        return total_loss

# ══════════════════════════════════════════════════════════════════
# PART 2: TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Refined V2.1 on {device}")

    batch_size = 16
    seq_len = 16
    n_epochs = 10000
    vocab_size = 1000

    model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=vocab_size).to(device)
    regularizer = ThermodynamicRegularizer()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    def get_complex_batch():
        # Task: Multi-Step Selective Logic
        # 1. Base token T
        # 2. If T % 2 == 0, Target = (T * 2 + 1) % vocab_size
        # 3. If T % 2 != 0, Target = (T // 2 - 1) % vocab_size
        data = torch.randint(10, 900, (batch_size, seq_len)).to(device)
        target = torch.where(data % 2 == 0, (data * 2 + 1) % vocab_size, (data // 2 - 1) % vocab_size)
        return data, target

    for epoch in range(1, n_epochs + 1):
        model.train()
        data, target = get_complex_batch()

        optimizer.zero_grad()
        logits, entropies = model(data)

        task_loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        thermo_loss = regularizer.calculate_loss(entropies)
        total_loss = task_loss + 0.15 * thermo_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{n_epochs} | Task: {task_loss.item():.4f} | Thermo: {thermo_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Refined Training Complete.")
    torch.save(model.state_dict(), "advanced_genuine_model_v2_1.pt")

if __name__ == "__main__":
    train()
