import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import OneCycleLR
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.1 Advanced Training)
Optimized training pipeline using the consolidated GenuineModel.
"""

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Advanced V2.1 on {device}")

    # Configuration
    batch_size = 16
    seq_len = 16
    n_epochs = 10000
    vocab_size = 1000

    # Initialize from consolidated module
    model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=vocab_size).to(device)
    regularizer = ThermodynamicRegularizer()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=1, epochs=n_epochs)
    criterion = nn.CrossEntropyLoss()

    def get_complex_batch():
        # Task: Selective Reversal and Increment
        data = torch.randint(10, 900, (batch_size, seq_len)).to(device)
        target = torch.where(data % 2 == 0, (data + 1) % vocab_size, (data - 1) % vocab_size)
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

    print("Advanced Training Complete.")
    torch.save(model.state_dict(), "advanced_genuine_model_v2_1.pt")

if __name__ == "__main__":
    train()
