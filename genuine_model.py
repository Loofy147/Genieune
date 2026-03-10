import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.1 Refined)
Full Version: Includes Rotary Positional Embeddings (RoPE),
Mechanistic Recurrence (Elaboration Routing), and Thermodynamic Regularization.
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
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]]
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
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

            # G-score: Variance of attention entropy (Mean across heads and sequence)
            current_g = torch.stack([torch.var(e, dim=-1).mean() for e in loop_entropies]).mean()
            all_entropies.extend(loop_entropies)
            g_history.append(float(current_g.detach()))

            # Dynamic routing: Stop if thought is sustained high genuineness
            if current_g >= g_threshold:
                break

            # Elaboration Pull Check: If G-score dropped significantly, loop to stabilize
            if len(g_history) >= 2:
                delta_g = g_history[-1] - g_history[-2]
                if delta_g < -0.15: # Pull detected
                    continue # Try another reasoning pass

        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, self.freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

# ══════════════════════════════════════════════════════════════════
# PART 3: THERMODYNAMIC REGULARIZER
# ══════════════════════════════════════════════════════════════════

class ThermodynamicRegularizer:
    def __init__(self, target_g=0.65, mechanical_penalty=0.4, collapse_penalty=5.0):
        self.target_g = target_g
        self.mechanical_penalty = mechanical_penalty
        self.collapse_penalty = collapse_penalty

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=entropies[0].device)

        # Track previous entropy mean to penalize collapse
        prev_mean_h = None

        for head_ent in entropies:
            # 1. Variance Reward (Sustain internal complexity)
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 1.0 * var_h

            # 2. Static Penalty (Avoid prolonged low entropy)
            mean_h = head_ent.mean()
            if mean_h < self.mechanical_penalty:
                total_loss = total_loss + torch.pow(self.mechanical_penalty - mean_h, 2) * self.collapse_penalty

            # 3. Collapse Penalty (Penalize sudden drop in entropy between layers)
            if prev_mean_h is not None:
                delta_h = mean_h - prev_mean_h
                if delta_h < -0.2: # Significant collapse
                    total_loss = total_loss + torch.pow(delta_h, 2) * self.collapse_penalty

            prev_mean_h = mean_h

        return total_loss
