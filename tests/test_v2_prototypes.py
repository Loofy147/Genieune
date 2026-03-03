import unittest
import torch
import numpy as np
from sustained_genuineness import ThermodynamicRegularizer, DualStreamSimulator, MechanisticRecurrence

class TestV2Prototypes(unittest.TestCase):
    def test_regularizer(self):
        reg = ThermodynamicRegularizer()
        # Mock entropies [head][batch, seq]
        entropies = [torch.rand(1, 10) for _ in range(4)]
        loss = reg.calculate_loss(entropies)
        self.assertTrue(torch.is_tensor(loss))

    def test_dual_stream(self):
        sim = DualStreamSimulator()
        result = sim.forward(initial_state=0.4, max_loops=10)
        self.assertIn("final_g", result)
        self.assertIn("loops_in_reasoner", result)
        self.assertGreater(result["loops_in_reasoner"], 0)

    def test_recurrence_trigger(self):
        recur = MechanisticRecurrence(recurrence_layer=21)
        self.assertFalse(recur.check_and_route(25, [0.6, 0.7]))
        self.assertTrue(recur.check_and_route(25, [0.7, 0.4]))
        self.assertFalse(recur.check_and_route(10, [0.7, 0.4]))

if __name__ == "__main__":
    unittest.main()
