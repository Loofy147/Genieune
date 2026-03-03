import unittest
import numpy as np
from sustained_genuineness import ThermodynamicRegularizer, DualStreamSimulator, MechanisticRecurrence

class TestV2Prototypes(unittest.TestCase):
    def test_regularizer(self):
        reg = ThermodynamicRegularizer()
        # High dynamic scores should result in a "reward" (negative loss)
        dynamic_scores = np.array([0.8, 0.9, 0.7])
        # Low variance, low entropy trajectory (MECHANICAL)
        trajectories = [np.array([0.1, 0.1, 0.1])]

        loss = reg.calculate_loss(dynamic_scores, trajectories)
        # Reward (-0.8*1.5 = -1.2) + Penalty (1.0) = -0.2 (approx)
        self.assertLess(loss, 1.0)

    def test_dual_stream(self):
        sim = DualStreamSimulator()
        result = sim.forward(initial_state=0.4, max_loops=10)
        self.assertIn("final_g", result)
        self.assertIn("loops_in_reasoner", result)
        # Should stay in reasoner until threshold is hit or max loops reached
        self.assertGreater(result["loops_in_reasoner"], 0)

    def test_recurrence_trigger(self):
        recur = MechanisticRecurrence(recurrence_layer=21)

        # Test 1: No pull
        self.assertFalse(recur.check_and_route(25, [0.6, 0.7]))

        # Test 2: Elaboration Pull detected (drop > 0.20)
        self.assertTrue(recur.check_and_route(25, [0.7, 0.4]))

        # Test 3: Early layer (no recurrence)
        self.assertFalse(recur.check_and_route(10, [0.7, 0.4]))

if __name__ == "__main__":
    unittest.main()
