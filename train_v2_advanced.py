import sys
import os
import subprocess

def setup_environment():
    """Install compatible versions for Kaggle environment."""
    try:
        import transformer_lens
        import beartype
        print("Dependencies already installed.")
    except ImportError:
        print("Installing dependencies...")
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
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 3.0 - Sparsity Evolution)
Task: Contextual Parity Pointer Task.
Evolution: Entropy-Gated Sparsity + Task-Dominant Equilibrium.
"""

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Version 3.0 (Sparsity Evolution) on {device}")

    batch_size = 32
    n_epochs = 15000
    vocab_size = 1000

    # V3.0 Configuration
    model = GenuineTransformer(d_model=512, n_heads=8, n_layers=12, vocab_size=vocab_size, sparsity_threshold=0.15).to(device)
    regularizer = ThermodynamicRegularizer(variance_weight=5.0)

    # Lower LR for deeper model stability
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-2)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, steps_per_epoch=1, epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    def get_pointer_parity_batch(current_seq_len):
        data = torch.randint(10, 900, (batch_size, current_seq_len)).to(device)
        pointer_idx = (data[:, 0] % (current_seq_len - 1)) + 1
        target_tokens = data.gather(1, pointer_idx.unsqueeze(1)).squeeze(1)

        # Binary Parity Target
        target = target_tokens % 2
        full_target = target.unsqueeze(1).repeat(1, current_seq_len)
        return data, full_target

    print("Starting Training with Genuineness Warmup Curriculum...")
    for epoch in range(1, n_epochs + 1):
        current_seq_len = 8 + int(56 * (epoch / n_epochs))

        model.train()
        data, target = get_pointer_parity_batch(current_seq_len)

        optimizer.zero_grad()
        logits, entropies = model(data, g_budget=24)

        task_loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        thermo_loss = regularizer.calculate_loss(entropies)

        # V5 Task-Dominant Equilibrium with Warmup
        # 5x Task Weighting to force logical signal over internal noise
        thermo_weight = 0.01 if epoch > 1000 else 0.0
        total_loss = 5.0 * task_loss + thermo_weight * thermo_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            avg_g = float(torch.var(torch.stack(entropies), dim=-1).mean().detach())
            log_str = f"Epoch {epoch}/{n_epochs} [Seq: {current_seq_len}] | Task: {task_loss.item():.4f} | Thermo: {thermo_loss.item():.4f} | G-Score: {avg_g:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            print(log_str)
            with open("training_log.txt", "a") as f:
                f.write(log_str + "\n")

    print("V3.0 Sparsity Evolution Training Complete.")
    torch.save(model.state_dict(), "advanced_genuine_model_v3_0.pt")

if __name__ == "__main__":
    train()
