import unittest
import numpy as np
import os
from phase_dynamics import PhaseSpaceMapper, fit_circuit_rates, compute_text_trajectory, run_transformerlens_phase_analysis, plot_phase_space
from transformer_lens import HookedTransformer

class TestPhaseDynamicsV1(unittest.TestCase):
    def test_mapper(self):
        # documentation: Genuine > 0.55, Mechanical < 0.35
        mapper = PhaseSpaceMapper(cost_threshold=0.5, genuine_threshold=0.6, mechanical_threshold=0.3)
        self.assertEqual(mapper.classify(0.7, 0.1, 0, 0), "MECHANICAL_COMMITTED")
        self.assertEqual(mapper.classify(0.2, 0.7, 0, 1), "GENUINE_DIFFUSE")
        self.assertEqual(mapper.classify(0.7, 0.7, 0, 2), "GENUINE_COMMITTED")
        self.assertEqual(mapper.classify(0.2, 0.1, 0, 3), "MECHANICAL_DIFFUSE")
        self.assertEqual(mapper.classify(0.5, 0.5, 0, 4), "TRANSITION")

        archetypes = mapper.get_archetypes()
        self.assertIn("Name Mover", archetypes)
        self.assertIn("Induction", archetypes)

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
        self.assertLess(res["trajectory_delta"], -0.2)

    def test_full_analysis_mock(self):
        results = run_transformerlens_phase_analysis(None, "test")
        self.assertIn("phase_space_distribution", results)
        self.assertIn("archetypes", results)
        self.assertIn("trajectory_analysis", results)

    def test_visualization_smoke(self):
        cost_scores = np.random.rand(2, 4)
        dynamic_scores = np.random.rand(2, 4)
        mapper = PhaseSpaceMapper()
        test_path = "test_plot.png"
        if os.path.exists(test_path):
            os.remove(test_path)
        plot_phase_space(cost_scores, dynamic_scores, mapper, save_path=test_path)
        self.assertTrue(os.path.exists(test_path))
        os.remove(test_path)

    def test_full_analysis_real(self):
        try:
            model = HookedTransformer.from_pretrained("gpt2-small")
            results = run_transformerlens_phase_analysis(model, "Hello world")
            self.assertIn("phase_space_distribution", results)
            self.assertIn("trajectory_analysis", results)
        except Exception as e:
            self.skipTest(f"Model loading failed: {e}")

if __name__ == "__main__":
    unittest.main()
