import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.2 Advanced)
Full Version: Includes Rotary Positional Embeddings (RoPE),
Learned Genuineness Gate (Adaptive Recurrence), Global G-Budgeting,
and Layer-Wise Thermodynamic Regularization.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ══════════════════════════════════════════════════════════════════

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # Ensure x is reshaped correctly for complex view
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis should be pre-sliced to seq_len
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

# ══════════════════════════════════════════════════════════════════
# PART 2: CORE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

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
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        log_attn_weights = F.log_softmax(attn_scores, dim=-1)
        # Shannon Entropy (H) in bits
        entropy = -torch.sum(attn_weights * log_attn_weights, dim=-1) * 1.44269504
        entropy = entropy.transpose(1, 2)

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
        x_norm = self.ln1(x)
        x_attn, weights, entropies = self.attn(x_norm, freqs_cis)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x, weights, entropies

class GenuinenessGate(nn.Module):
    """
    V2.2: Learned Gating mechanism that monitors hidden states and entropy
    to determine if more reasoning loops are needed.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model + n_heads, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        # Initialize gate with a positive bias to favor reasoning initially
        nn.init.constant_(self.gate_fc[2].bias, 1.5)

    def forward(self, x, entropy):
        # x: [batch, seq, d_model], entropy: [batch, seq, n_heads]
        # Aggregate across sequence for global decision
        feat = torch.cat([x.mean(dim=1), entropy.mean(dim=1)], dim=-1)
        return self.gate_fc(feat)

class GenuineTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=12, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_layers)])
        self.gate = GenuinenessGate(d_model, n_heads)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // n_heads, 256))
        self.n_heads = n_heads

    def forward(self, x, g_budget=12):
        batch_size, seq_len = x.shape
        freqs_cis = self.freqs_cis[:seq_len].view(1, seq_len, 1, -1)

        x = self.embedding(x)
        all_entropies = []
        reasoning_layers = len(self.layers) // 2

        # V2.2: Global G-Budgeting
        total_steps = 0

        while total_steps < g_budget:
            loop_entropies = []
            for i in range(reasoning_layers):
                x, attn, entropies = self.layers[i](x, freqs_cis)
                loop_entropies.append(entropies)
                total_steps += 1

            all_entropies.extend(loop_entropies)

            # Gating check
            last_entropy = loop_entropies[-1]
            gate_signal = self.gate(x, last_entropy) # [batch, 1]

            if gate_signal.mean() < 0.5:
                break

        # Final decoding layers
        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

# ══════════════════════════════════════════════════════════════════
# PART 3: THERMODYNAMIC REGULARIZER (V2.2)
# ══════════════════════════════════════════════════════════════════

class ThermodynamicRegularizer:
    def __init__(self, variance_weight=15.0, mechanical_penalty=0.45, collapse_penalty=10.0, layer_decay=0.92):
        self.variance_weight = variance_weight
        self.mechanical_penalty = mechanical_penalty
        self.collapse_penalty = collapse_penalty
        self.layer_decay = layer_decay

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        if not entropies:
            return torch.tensor(0.0, device="cpu")

        stack = torch.stack(entropies) # [layers, batch, seq, heads]

        # 1. Variance Reward (Internal Complexity)
        # Reward high entropy variance across heads for each token/layer
        var_h = torch.var(stack, dim=-1) # [layers, batch, seq]

        # Apply layer-wise decay to prioritize early-layer complexity
        layer_weights = torch.pow(self.layer_decay, torch.arange(len(entropies), device=stack.device, dtype=torch.float)).view(-1, 1, 1)
        weighted_var = (var_h * layer_weights).mean()

        total_loss = -self.variance_weight * weighted_var

        # 2. Mechanical Penalty (Anti-Pattern Matching)
        means_h = stack.mean(dim=-1) # [layers, batch, seq]
        static_diff = self.mechanical_penalty - means_h
        total_loss += torch.where(static_diff > 0, static_diff.pow(2), torch.zeros_like(static_diff)).mean() * self.collapse_penalty

        # 3. Collapse Penalty (Sudden drop in entropy)
        if len(entropies) > 1:
            diffs = means_h[1:] - means_h[:-1]
            collapse_diff = -0.2 - diffs
            total_loss += torch.where(collapse_diff > 0, collapse_diff.pow(2), torch.zeros_like(collapse_diff)).mean() * self.collapse_penalty

        return total_loss
