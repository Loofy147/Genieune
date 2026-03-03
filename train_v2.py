import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from genuine_model import GenuineTransformer
from sustained_genuineness import ThermodynamicRegularizer

"""
TRAINING PIPELINE (Version 2.0)
Optimizing for Sustained Genuineness using Differentiable Thermodynamic Regularization.
"""

def train():
    print(f'Current working directory: {os.getcwd()}')
    print(f'Files in directory: {os.listdir(os.getcwd())}')
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

    # 3. Training Loop
    for epoch in range(1, 201):
        model.train()
        data, target = get_batch()

        optimizer.zero_grad()
        logits, entropies = model(data)

        # Calculate standard loss
        task_loss = criterion(logits.view(-1, 1000), target.view(-1))

        # 4. Calculate Differentiable Thermodynamic Loss
        # Flatten the nested entropies list [layer][head]
        flat_entropies = []
        for layer_ent in entropies:
            flat_entropies.extend(layer_ent)

        thermo_loss = regularizer.calculate_loss(flat_entropies)

        # Combine losses: task_loss + lambda * thermo_loss
        total_loss = task_loss + 0.1 * thermo_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Task: {task_loss.item():.4f} | Thermo: {thermo_loss.item():.4f}")

    print("Training Complete.")
    torch.save(model.state_dict(), "genuine_model_v2.pt")

if __name__ == "__main__":
    train()
