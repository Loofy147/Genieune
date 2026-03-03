import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GenuineAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim)
        self.k = nn.Linear(d_model, head_dim)
        self.v = nn.Linear(d_model, head_dim)
        self.head_dim = head_dim

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Shannon Entropy H(A) = -sum(p * log2(p))
        entropy = -torch.sum(attn_weights * torch.log2(attn_weights + 1e-9), dim=-1)

        out = torch.matmul(attn_weights, v)
        return out, attn_weights, entropy

class GenuineLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.heads = nn.ModuleList([GenuineAttentionHead(d_model, self.head_dim) for _ in range(n_heads)])
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.ln1(x)

        head_outputs = [h(x) for h in self.heads]
        head_outs = [out[0] for out in head_outputs]
        attn_weights = [out[1] for out in head_outputs]
        entropies = [out[2] for out in head_outputs]

        x = torch.cat(head_outs, dim=-1)
        x = x + res

        x = x + self.mlp(self.ln2(x))
        return x, attn_weights, entropies

class GenuineTransformer(nn.Module):
    """
    Implementation of the Sustained Genuineness Architecture (v2.0).
    Features Mechanistic Recurrence to sustain high-variance attention.
    """
    def __init__(self, d_model=256, n_heads=8, n_reasoner_layers=4, n_decoder_layers=2, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.reasoner = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_reasoner_layers)])
        self.decoder = nn.ModuleList([GenuineLayer(d_model, n_heads) for _ in range(n_decoder_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, use_recurrence=False, max_loops=3):
        x = self.embedding(x)
        all_entropies = []

        for loop in range(max_loops):
            loop_entropies = []
            for i, layer in enumerate(self.reasoner):
                x, attn, entropies = layer(x)
                loop_entropies.append(entropies)

                if use_recurrence and loop < max_loops - 1:
                    # Detect Elaboration Pull (Simplified G check)
                    # Calculate mean G score for the layer
                    layer_g = torch.stack([torch.var(e, dim=-1).mean() for e in entropies]).mean()
                    # In a real v2, we would track delta G and break/loop

            all_entropies.extend(loop_entropies)
            if not use_recurrence: break # Single pass by default

        for layer in self.decoder:
            x, attn, entropies = layer(x)
            all_entropies.append(entropies)

        logits = self.fc_out(x)
        return logits, all_entropies
