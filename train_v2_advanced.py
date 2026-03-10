import sys
import os
import subprocess

def setup_environment():
    """Install compatible versions for Kaggle environment."""
    try:
        import transformer_lens
        print("transformer-lens already installed.")
    except ImportError:
        print("Installing transformer-lens and compatible transformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformer-lens==2.1.0", "transformers==4.44.0"])

if __name__ == "__main__":
    if "KAGGLE_URL_BASE" in os.environ:
        setup_environment()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import OneCycleLR

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.2 Advanced Training)
Self-contained script for Kaggle with RoPE, Learned Gating, Global G-Budgeting,
and Layer-Wise Thermodynamic Regularization.
Task: Contextual Parity Pointer Task.
Optimization: Complexity Scaling (Sequence Length 8 -> 32).
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: CORE ARCHITECTURE COMPONENTS (V2.2 Advanced)
# ══════════════════════════════════════════════════════════════════

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
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
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.n_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        log_attn_weights = F.log_softmax(attn_scores, dim=-1)
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
        x_attn, weights, entropies = self.attn(self.ln1(x), freqs_cis)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x, weights, entropies

class GenuinenessGate(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model + n_heads, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_fc[2].bias, 1.5)

    def forward(self, x, entropy):
        feat = torch.cat([x.mean(dim=1), entropy.mean(dim=1)], dim=-1)
        return self.gate_fc(feat)

class GenuineTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_layers)])
        self.gate = GenuinenessGate(d_model, n_heads)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // n_heads, 128))

    def forward(self, x, g_budget=12):
        batch_size, seq_len = x.shape
        freqs_cis = self.freqs_cis[:seq_len].view(1, seq_len, 1, -1)

        x = self.embedding(x)
        all_entropies = []
        reasoning_layers = len(self.layers) // 2
        total_steps = 0

        while total_steps < g_budget:
            loop_entropies = []
            for i in range(reasoning_layers):
                x, attn, entropies = self.layers[i](x, freqs_cis)
                loop_entropies.append(entropies)
                total_steps += 1
            all_entropies.extend(loop_entropies)
            gate_signal = self.gate(x, loop_entropies[-1])
            if gate_signal.mean() < 0.5:
                break

        for i in range(reasoning_layers, len(self.layers)):
            x, attn, entropies = self.layers[i](x, freqs_cis)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies

class ThermodynamicRegularizer:
    def __init__(self, variance_weight=3.0, mechanical_penalty=0.45, collapse_penalty=10.0, layer_decay=0.92):
        self.variance_weight = variance_weight
        self.mechanical_penalty = mechanical_penalty
        self.collapse_penalty = collapse_penalty
        self.layer_decay = layer_decay

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        if not entropies:
            return torch.tensor(0.0, device="cpu")
        stack = torch.stack(entropies)
        var_h = torch.var(stack, dim=-1)
        layer_weights = torch.pow(self.layer_decay, torch.arange(len(entropies), device=stack.device, dtype=torch.float)).view(-1, 1, 1)
        weighted_var = (var_h * layer_weights).mean()
        total_loss = -self.variance_weight * weighted_var
        means_h = stack.mean(dim=-1)
        static_diff = self.mechanical_penalty - means_h
        total_loss += torch.where(static_diff > 0, static_diff.pow(2), torch.zeros_like(static_diff)).mean() * self.collapse_penalty
        if len(entropies) > 1:
            diffs = means_h[1:] - means_h[:-1]
            collapse_diff = -0.2 - diffs
            total_loss += torch.where(collapse_diff > 0, collapse_diff.pow(2), torch.zeros_like(collapse_diff)).mean() * self.collapse_penalty
        return total_loss

# ══════════════════════════════════════════════════════════════════
# PART 2: TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Advanced V2.2 on {device}")

    batch_size = 16
    n_epochs = 10000
    vocab_size = 1000

    model = GenuineTransformer(d_model=256, n_heads=8, n_layers=6, vocab_size=vocab_size).to(device)
    regularizer = ThermodynamicRegularizer()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    def get_pointer_parity_batch(current_seq_len):
        """
        CONTEXTUAL PARITY POINTER TASK:
        - Token 0 is the pointer.
        - Pointer value P = data[0] % current_seq_len.
        - Target is the parity of data[P].
        Forces the model to first resolve the pointer, then attend to the target token.
        """
        data = torch.randint(10, 900, (batch_size, current_seq_len)).to(device)
        pointer_idx = (data[:, 0] % (current_seq_len - 1)) + 1

        # Gather target tokens based on pointer
        target_tokens = data.gather(1, pointer_idx.unsqueeze(1)).squeeze(1)

        # Parity logic for target
        target = torch.where(target_tokens % 2 == 0, (target_tokens * 2 + 1) % vocab_size, (target_tokens // 2 - 1) % vocab_size)

        # Expand target to all positions for simplicity (or just last token)
        full_target = target.unsqueeze(1).repeat(1, current_seq_len)
        return data, full_target

    print("Starting Training with Complexity Scaling...")
    for epoch in range(1, n_epochs + 1):
        # Complexity Scaling: Sequence length increases from 8 to 32
        current_seq_len = 8 + int(24 * (epoch / n_epochs))

        model.train()
        data, target = get_pointer_parity_batch(current_seq_len)

        optimizer.zero_grad()
        logits, entropies = model(data)

        task_loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        thermo_loss = regularizer.calculate_loss(entropies)

        total_loss = task_loss + 0.25 * thermo_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            avg_g = float(torch.var(torch.stack(entropies), dim=-1).mean().detach())
            print(f"Epoch {epoch}/{n_epochs} [Seq: {current_seq_len}] | Task: {task_loss.item():.4f} | Thermo: {thermo_loss.item():.4f} | G-Score: {avg_g:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print("V2.2 Advanced Training Complete.")
    torch.save(model.state_dict(), "advanced_genuine_model_v2_2.pt")

if __name__ == "__main__":
    train()
