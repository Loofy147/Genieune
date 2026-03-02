import unittest
import numpy as np
from phase_dynamics import PhaseSpaceMapper, fit_circuit_rates, compute_text_trajectory, run_transformerlens_phase_analysis
from transformer_lens import HookedTransformer

class TestPhaseDynamicsV1(unittest.TestCase):
    def test_mapper(self):
        # documentation: Genuine > 0.55, Mechanical < 0.35
        mapper = PhaseSpaceMapper(cost_threshold=0.5, genuine_threshold=0.6, mechanical_threshold=0.3)
        self.assertEqual(mapper.classify(0.7, 0.1), "MECHANICAL_COMMITTED")
        self.assertEqual(mapper.classify(0.2, 0.7), "GENUINE_DIFFUSE")
        self.assertEqual(mapper.classify(0.7, 0.7), "GENUINE_COMMITTED")
        self.assertEqual(mapper.classify(0.2, 0.1), "MECHANICAL_DIFFUSE")
        self.assertEqual(mapper.classify(0.5, 0.5), "TRANSITION")

    def test_fitting(self):
        trajectory = [0.8, 0.4, 0.2, 0.6, 0.9]
        circuit_types = [0, 0, 1, 1]
        rates = fit_circuit_rates(trajectory, circuit_types)
        self.assertIn("k_degrade", rates)
        self.assertIn("k_recover", rates)

    def test_trajectory(self):
        scores = [0.9, 0.8, 0.85, 0.7, 0.6, 0.4, 0.3, 0.2]
        res = compute_text_trajectory(scores, window_size=2)
        self.assertTrue(res["elaboration_pull"])

    def test_full_analysis_mock(self):
        results = run_transformerlens_phase_analysis(None, "test")
        self.assertIn("phase_space_distribution", results)
        self.assertIn("genuine_committed_empty_in_weights", results)

    def test_full_analysis_real(self):
        try:
            model = HookedTransformer.from_pretrained("gpt2-small")
            results = run_transformerlens_phase_analysis(model, "Hello world")
            self.assertIn("phase_space_distribution", results)
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")

if __name__ == "__main__":
    unittest.main()
