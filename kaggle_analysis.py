import json
import numpy as np
import torch
from transformer_lens import HookedTransformer
from phase_dynamics import run_transformerlens_phase_analysis, PhaseSpaceMapper, plot_phase_space
from sustained_genuineness import DualStreamSimulator, ThermodynamicRegularizer

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Kaggle Analysis Pipeline)
Combines Version 1.0 interpretability with Version 2.0 architectural simulations.
"""

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def run_comprehensive_analysis(prompt: str):
    print(f"--- Starting Analysis on Prompt: '{prompt}' ---")

    # 1. Load Model (Version 1.0 Analysis)
    model = None
    try:
        model = HookedTransformer.from_pretrained("gpt2-small")
    except Exception as e:
        print(f"Could not load model: {e}")

    # 2. Run Phase Space Analysis
    results_v1 = run_transformerlens_phase_analysis(model, prompt)

    # Generate Phase Space Plot
    if "raw_scores" in results_v1:
        mapper = PhaseSpaceMapper()
        plot_phase_space(
            np.array(results_v1["raw_scores"]["cost"]),
            np.array(results_v1["raw_scores"]["dynamic"]),
            mapper,
            save_path="v1_phase_space_distribution.png"
        )
        del results_v1["raw_scores"]

    # 3. Simulate Sustained Genuineness (Version 2.0)
    print("--- Running Version 2.0 Structural Simulations ---")
    simulator = DualStreamSimulator()
    initial_g = results_v1["trajectory_analysis"].get("start_G", 0.4)
    results_v2 = simulator.forward(initial_state=initial_g)

    # 4. Thermodynamic Loss Calculation
    regularizer = ThermodynamicRegularizer()
    mock_dynamic = np.random.rand(12, 12)
    mock_trajectories = [np.random.rand(5) for _ in range(5)]
    v2_loss = regularizer.calculate_loss(mock_dynamic, mock_trajectories)

    comprehensive_results = {
        "v1_interpretability": results_v1,
        "v2_architecture_simulation": results_v2,
        "v2_thermodynamic_loss_sample": v2_loss
    }

    return convert_to_serializable(comprehensive_results)

if __name__ == "__main__":
    prompt = "A sequence of logical steps leads to a conclusion. This is the essence of genuine reasoning."
    results = run_comprehensive_analysis(prompt)

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Final Summary ---")
    print(f"Phase Space Distribution: {results['v1_interpretability']['phase_space_distribution']}")
    print(f"Sustained Genuineness Loop Count: {results['v2_architecture_simulation']['loops_in_reasoner']}")
    print(f"V2 Thermodynamic Loss: {results['v2_thermodynamic_loss_sample']}")
    print("Full results saved to analysis_results.json and v1_phase_space_distribution.png")
