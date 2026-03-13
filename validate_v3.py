import torch
import torch.nn as nn
from genuine_model import GenuineTransformer

def validate():
    print(f"Final Logic Check: Binary Classification")
    model = GenuineTransformer(d_model=32, n_heads=2, n_layers=2, vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Can the model learn to predict token 0?
    for epoch in range(1, 51):
        data = torch.randint(0, 5, (16, 4))
        target = data[:, 0].unsqueeze(1).repeat(1, 4)

        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits.view(-1, 10), target.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    validate()
