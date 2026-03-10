import unittest
import torch
import numpy as np
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

class TestV2FullVersion(unittest.TestCase):
    def test_regularizer(self):
        reg = ThermodynamicRegularizer()
        # Mock entropies [batch, seq, heads]
        entropies = [torch.rand(1, 10, 8) for _ in range(4)]
        loss = reg.calculate_loss(entropies)
        self.assertTrue(torch.is_tensor(loss))

    def test_genuine_transformer_forward(self):
        # Updated for V2.2: g_budget instead of max_loops/g_threshold
        model = GenuineTransformer(d_model=64, n_heads=4, n_layers=2, vocab_size=100)
        x = torch.randint(0, 100, (1, 10))
        logits, entropies = model(x, g_budget=4)

        self.assertEqual(logits.shape, (1, 10, 100))
        # At least 1 reasoning layer * 1 loop + 1 decoder layer = 2 entropy tensors
        self.assertGreaterEqual(len(entropies), 2)

if __name__ == "__main__":
    unittest.main()
