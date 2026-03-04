import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.1)
Full Version: Includes Rotary Positional Embeddings (RoPE),
Mechanistic Recurrence, and Thermodynamic Regularization.
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
    freqs_cis = freqs_cis[:x.shape[1]].to(x.device)
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
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(batch, seq, self.n_heads, self.head_dim)
        k = k.view(batch, seq, self.n_heads, self.head_dim)
        v = v.view(batch, seq, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Scaled Dot-Product Attention
        attn_scores = torch.einsum("bihd,bjhd->bihj", q, k) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Entropy Monitoring
        entropy = -torch.sum(attn_weights * torch.log2(attn_weights + 1e-9), dim=-1)

        out = torch.einsum("bihj,bjhd->bihd", attn_weights, v)
        out = out.reshape(batch, seq, -1)
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
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, 128)

    def forward(self, x, g_threshold=0.6, max_loops=2):
        x = self.embedding(x)
        all_entropies = []
        reasoning_layers = len(self.layers) // 2

        for loop in range(max_loops):
            loop_entropies = []
            for i in range(reasoning_layers):
                x, attn, entropies = self.layers[i](x, self.freqs_cis)
                loop_entropies.append(entropies)

            # G-score: Variance of attention entropy
            current_g = torch.stack([torch.var(e, dim=-1).mean() for e in loop_entropies]).mean()
            all_entropies.extend(loop_entropies)

            if current_g >= g_threshold:
                break

        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, self.freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

# ══════════════════════════════════════════════════════════════════
# PART 3: THERMODYNAMIC REGULARIZER
# ══════════════════════════════════════════════════════════════════

class ThermodynamicRegularizer:
    def __init__(self, target_g=0.65, mechanical_penalty=0.4):
        self.target_g = target_g
        self.mechanical_penalty = mechanical_penalty

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=entropies[0].device)
        for head_ent in entropies:
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 1.0 * var_h

            mean_h = head_ent.mean()
            if mean_h < self.mechanical_penalty:
                total_loss = total_loss + torch.pow(self.mechanical_penalty - mean_h, 2) * 5.0

        return total_loss
