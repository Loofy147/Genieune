import unittest
import torch
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

class TestV2_2Advanced(unittest.TestCase):
    def test_genuine_transformer_v2_2(self):
        # Initialize V2.2 Model
        model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=100)
        x = torch.randint(0, 100, (1, 10))

        # Test forward pass with G-budget
        logits, entropies = model(x, g_budget=6)

        self.assertEqual(logits.shape, (1, 10, 100))
        # At least 2 (one loop) + 2 (decoder) = 4 entropy tensors
        self.assertGreaterEqual(len(entropies), 4)

    def test_thermodynamic_regularizer_v2_2(self):
        reg = ThermodynamicRegularizer(variance_weight=3.0, layer_decay=0.9)
        # Mock entropies: [batch, seq, heads]
        entropies = [torch.rand(1, 10, 4) for _ in range(4)]
        loss = reg.calculate_loss(entropies)

        self.assertTrue(torch.is_tensor(loss))
        self.assertFalse(torch.isnan(loss))

    def test_genuineness_gate(self):
        from genuine_model import GenuinenessGate
        gate = GenuinenessGate(d_model=128, n_heads=4)
        x = torch.randn(1, 10, 128)
        entropy = torch.rand(1, 10, 4)
        signal = gate(x, entropy)

        self.assertEqual(signal.shape, (1, 1))
        self.assertTrue(0 <= signal.item() <= 1)

if __name__ == "__main__":
    unittest.main()
