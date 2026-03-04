import unittest
import torch
import numpy as np
from genuine_model import GenuineTransformer, ThermodynamicRegularizer
from sustained_genuineness import MechanisticRecurrence

class TestV2FullVersion(unittest.TestCase):
    def test_regularizer(self):
        reg = ThermodynamicRegularizer()
        # Mock entropies [head][batch, seq]
        entropies = [torch.rand(1, 10) for _ in range(4)]
        loss = reg.calculate_loss(entropies)
        self.assertTrue(torch.is_tensor(loss))

    def test_genuine_transformer_forward(self):
        model = GenuineTransformer(d_model=64, n_heads=4, n_layers=2, vocab_size=100)
        x = torch.randint(0, 100, (1, 10))
        # Set g_threshold very high to force max_loops
        logits, entropies = model(x, g_threshold=10.0, max_loops=2)

        self.assertEqual(logits.shape, (1, 10, 100))
        # 1 reasoner layer * 2 loops + 1 decoder layer = 3 entropy tensors
        self.assertEqual(len(entropies), 3)

    def test_recurrence_trigger(self):
        recur = MechanisticRecurrence(recurrence_layer=5)
        self.assertFalse(recur.check_and_route(10, [0.6, 0.7]))
        self.assertTrue(recur.check_and_route(10, [0.7, 0.4]))
        self.assertFalse(recur.check_and_route(2, [0.7, 0.4]))

if __name__ == "__main__":
    unittest.main()
