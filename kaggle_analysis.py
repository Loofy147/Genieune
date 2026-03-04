import json
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from phase_dynamics import run_transformerlens_phase_analysis, PhaseSpaceMapper, plot_phase_space
from genuine_model import GenuineTransformer, ThermodynamicRegularizer

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Kaggle Analysis Pipeline)
Version 2.1: Full implementation with V2 trajectory visualization.
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
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

def plot_v2_trajectory(g_scores, save_path="v2_genuineness_trajectory.png"):
    """
    Plots the Genuineness (G-score) trajectory across reasoning loops.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(g_scores, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.axhline(y=0.6, color='r', linestyle='--', label='G-Threshold (0.6)')
    plt.xlabel("Layer Processing Step (Loops included)")
    plt.ylabel("Dynamic Genuineness (G-score)")
    plt.title("Version 2.1: Mechanistic Recurrence G-Trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def run_comprehensive_analysis(prompt: str):
    print(f"--- Starting Analysis on Prompt: '{prompt}' ---")

    # 1. Load Model (Version 1.0 Analysis)
    model = None
    try:
        model = HookedTransformer.from_pretrained("gpt2-small")
    except Exception as e:
        print(f"Could not load V1 model: {e}")

    # 2. Run Phase Space Analysis
    results_v1 = run_transformerlens_phase_analysis(model, prompt)

    # Generate V1 Phase Space Plot
    if "raw_scores" in results_v1:
        mapper = PhaseSpaceMapper()
        plot_phase_space(
            np.array(results_v1["raw_scores"]["cost"]),
            np.array(results_v1["raw_scores"]["dynamic"]),
            mapper,
            save_path="v1_phase_space_distribution.png"
        )
        del results_v1["raw_scores"]

    # 3. Run Full Genuineness Model (Version 2.1)
    print("--- Running Version 2.1 Full Model Analysis ---")
    v2_model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=1000)

    if os.path.exists("advanced_genuine_model_v2_1.pt"):
        try:
            v2_model.load_state_dict(torch.load("advanced_genuine_model_v2_1.pt", map_location="cpu"))
            print("Loaded advanced_genuine_model_v2_1.pt weights.")
        except Exception as e:
            print(f"Could not load weights: {e}")

    v2_model.eval()
    with torch.no_grad():
        if model is not None:
            tokens = model.to_tokens(prompt)
            v2_tokens = torch.remainder(tokens, 1000)
        else:
            v2_tokens = torch.randint(0, 1000, (1, 16))

        logits, entropies = v2_model(v2_tokens, g_threshold=0.6, max_loops=3)

        # Calculate G-trajectory from actual model entropies
        g_scores = [float(torch.var(e, dim=-1).mean().detach()) for e in entropies]
        plot_v2_trajectory(g_scores)

        results_v2 = {
            "final_g": round(float(g_scores[-1]), 3) if g_scores else 0.0,
            "g_trajectory": [round(g, 3) for g in g_scores],
            "total_processing_steps": len(entropies)
        }

        # 4. Thermodynamic Loss Calculation
        regularizer = ThermodynamicRegularizer()
        v2_loss = regularizer.calculate_loss(entropies)

    comprehensive_results = {
        "v1_interpretability": results_v1,
        "v2_architecture_analysis": results_v2,
        "v2_thermodynamic_loss": float(v2_loss)
    }

    return convert_to_serializable(comprehensive_results)

if __name__ == "__main__":
    prompt = "Reasoning is a sustained process of complexity. A sequence of logical steps leads to a conclusion."
    results = run_comprehensive_analysis(prompt)

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Final Summary ---")
    print(f"Phase Space Distribution: {results['v1_interpretability']['phase_space_distribution']}")
    print(f"Model Genuineness Score: {results['v2_architecture_analysis']['final_g']}")
    print(f"V2 Processing Steps: {results['v2_architecture_analysis']['total_processing_steps']}")
    print(f"V2 Thermodynamic Loss: {results['v2_thermodynamic_loss']}")
    print("Full results saved to analysis_results.json and visualization PNGs.")
