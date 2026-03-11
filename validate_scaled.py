import torch
import torch.nn as nn
import torch.optim as optim
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

def validate():
    device = torch.device("cpu")
    print(f"Validating Scaled V2.2 - Equilibrium Mode")

    model = GenuineTransformer(d_model=512, n_heads=8, n_layers=12, vocab_size=1000).to(device)
    # Reduced variance_weight 10.0
    regularizer = ThermodynamicRegularizer(variance_weight=10.0)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 8
    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        data = torch.randint(0, 1000, (batch_size, 32))
        # Simple binary parity target
        target = (data[:, 0] % 2).unsqueeze(1).repeat(1, 32)

        optimizer.zero_grad()
        logits, entropies = model(data, g_budget=8)

        task_loss = criterion(logits.view(-1, 1000), target.view(-1))
        thermo_loss = regularizer.calculate_loss(entropies)

        # Reduced thermo influence 0.6
        total_loss = task_loss + 0.6 * thermo_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            avg_g = float(torch.var(torch.stack(entropies), dim=-1).mean().detach())
            print(f"Epoch {epoch} | Total: {total_loss.item():.4f} | Task: {task_loss.item():.4f} | G: {avg_g:.4f}")

if __name__ == "__main__":
    validate()
