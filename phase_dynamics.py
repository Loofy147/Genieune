import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import transformer_lens
except ImportError:
    install("transformer-lens")
    import transformer_lens

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 1.0)
Official Implementation of the Mechanistic Interpretability Pipeline.
"""

import numpy as np
import scipy.optimize as opt
from collections import defaultdict
import torch
from transformer_lens import HookedTransformer

# ══════════════════════════════════════════════════════════════════
# PART 1: 2D PHASE SPACE CLASSIFIER
# ══════════════════════════════════════════════════════════════════

class PhaseSpaceMapper:
    """
    Maps attention heads or text outputs into a 2D phase space based on
    Token Cost (X) and Dynamic Genuineness (Y).
    """
    def __init__(self, cost_threshold=0.6, genuine_threshold=0.55, mechanical_threshold=0.35):
        self.cost_threshold = cost_threshold
        self.genuine_threshold = genuine_threshold
        self.mechanical_threshold = mechanical_threshold
        self.quadrants = defaultdict(int)

    def classify(self, cost: float, dynamic_genuineness: float, increment=True) -> str:
        high_cost = cost >= self.cost_threshold
        # Genuineness thresholds from documentation
        if dynamic_genuineness >= self.genuine_threshold:
            if high_cost:
                q = "GENUINE_COMMITTED"    # Usually empty in weights
            else:
                q = "GENUINE_DIFFUSE"      # The logic engines (e.g. Name Movers)
        elif dynamic_genuineness <= self.mechanical_threshold:
            if high_cost:
                q = "MECHANICAL_COMMITTED" # Pure pattern retrieval (e.g. Induction)
            else:
                q = "MECHANICAL_DIFFUSE"   # Uniform context gathering (Broadcast)
        else:
            # Transition zone
            q = "TRANSITION"

        if increment:
            self.quadrants[q] += 1
        return q

    def get_distribution(self):
        total = sum(self.quadrants.values())
        if total == 0: return {}
        return {k: v/total for k, v in self.quadrants.items()}


# ══════════════════════════════════════════════════════════════════
# PART 2: DIFFERENTIAL EQUATION SOLVER (CIRCUIT ASYMMETRY)
# ══════════════════════════════════════════════════════════════════

def fit_circuit_rates(trajectory: list, circuit_types: list):
    """
    Fits k_degrade and k_recover based on the Circuit Asymmetry Equations.
    dG/dt (pattern) = -0.8129 * G
    dG/dt (genuine) = +1.2371 * (G_max - G)
    """
    degradations = []
    recoveries =[]

    for i in range(len(trajectory) - 1):
        G_current = trajectory[i]
        G_next = trajectory[i+1]

        if i < len(circuit_types):
            if circuit_types[i] == 0: # Pattern circuit
                # G_next = G_current * exp(-k_deg)
                if G_current > 0.01:
                    k_deg = -np.log(max(G_next, 1e-5) / G_current)
                    degradations.append(max(0, k_deg))

            elif circuit_types[i] == 1: # Genuine circuit
                # G_next = G_max - (G_max - G_current) * exp(-k_rec)
                if G_current < 0.99:
                    val = (1.0 - G_next) / (1.0 - G_current)
                    k_rec = -np.log(max(val, 1e-5))
                    recoveries.append(max(0, k_rec))

    # Empirical fallbacks from documentation
    empirical_k_deg = np.mean(degradations) if degradations else 0.8129
    empirical_k_rec = np.mean(recoveries) if recoveries else 1.2371

    return {
        "k_degrade": round(float(empirical_k_deg), 4),
        "k_recover": round(float(empirical_k_rec), 4),
        "asymmetry_ratio": round(float(empirical_k_rec / max(empirical_k_deg, 1e-5)), 3)
    }


# ══════════════════════════════════════════════════════════════════
# PART 3: TEXT TRAJECTORY (THE ELABORATION PULL)
# ══════════════════════════════════════════════════════════════════

def compute_text_trajectory(token_scores: list, window_size: int = 5):
    """
    Detects the "elaboration pull" where initial genuine computation
    decays into pattern repetition.
    """
    if len(token_scores) < window_size:
        return {"trajectory_delta": 0.0, "elaboration_pull": False}

    windows =[
        np.mean(token_scores[i:i+window_size])
        for i in range(len(token_scores) - window_size + 1)
    ]

    start_G = windows[0]
    end_G = windows[-1]
    trajectory_delta = end_G - start_G

    return {
        "start_G": round(float(start_G), 3),
        "end_G": round(float(end_G), 3),
        "trajectory_delta": round(float(trajectory_delta), 3),
        "elaboration_pull": trajectory_delta < -0.20 # Sharp drop threshold
    }


# ══════════════════════════════════════════════════════════════════
# PART 4: TRANSFORMERLENS INTEGRATION (VERSION 1.0 METRICS)
# ══════════════════════════════════════════════════════════════════

def extract_metrics(model: HookedTransformer, prompt: str):
    """
    Extracts Token Cost (X) and Dynamic Genuineness (Y) using Version 1.0 protocol.
    """
    logits, cache = model.run_with_cache(prompt)

    # 1. Token Cost (External Anchor / Surprisal)
    probs = torch.softmax(logits, dim=-1)
    tokens = model.to_tokens(prompt)
    log_probs = torch.log(probs[0, :-1, :])
    next_tokens = tokens[0, 1:]
    surprisal = -torch.gather(log_probs, 1, next_tokens.unsqueeze(-1)).squeeze(-1)
    # Convert to log2 surprisal for bit-based measure
    surprisal = surprisal / np.log(2)
    surprisal = torch.cat([torch.tensor([0.0], device=surprisal.device), surprisal])

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    cost_scores = np.zeros((n_layers, n_heads))
    dynamic_scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        pattern = cache[f"blocks.{l}.attn.hook_pattern"][0] # [head, query, key]

        for h in range(n_heads):
            head_attn = pattern[h] # [seq, seq]

            # X: Token Cost (Weighted Surprisal)
            weighted_surprisal = torch.matmul(head_attn, surprisal)
            cost_scores[l, h] = weighted_surprisal.mean().item()

            # Y: Dynamic Genuineness (Var(H) + Collapse Count)
            # Shannon Entropy H(A) = -sum(p * log2(p))
            entropy = -torch.sum(head_attn * torch.log2(head_attn + 1e-9), dim=-1) # [seq]

            # Var(H)
            var_h = torch.var(entropy).item()

            # Collapse Count (delta H < -0.20)
            delta_h = entropy[1:] - entropy[:-1]
            collapse_count = torch.sum(delta_h < -0.20).item()
            # Normalized collapse contribution
            norm_collapses = collapse_count / max(1, len(entropy) - 1)

            dynamic_scores[l, h] = var_h + norm_collapses

    # Normalize to [0, 1] based on theoretical or empirical max for classifier stability
    cost_scores = np.clip(cost_scores / 10.0, 0, 1) # Assume 10 bits is high cost
    dynamic_scores = np.clip(dynamic_scores / 0.5, 0, 1) # Assume 0.5 is high dynamic genuineness

    return cost_scores, dynamic_scores

def run_transformerlens_phase_analysis(model, prompt: str):
    """
    Applies the Version 1.0 Framework to a model.
    """
    if model is None:
        print("Using Mock Model for analysis.")
        cost_scores = np.random.rand(12, 12)
        dynamic_scores = np.random.rand(12, 12)
    else:
        cost_scores, dynamic_scores = extract_metrics(model, prompt)

    mapper = PhaseSpaceMapper()
    n_layers, n_heads = cost_scores.shape

    circuit_types = []
    for l in range(n_layers):
        layer_has_genuine = False
        for h in range(n_heads):
            q = mapper.classify(cost=cost_scores[l, h], dynamic_genuineness=dynamic_scores[l, h])
            if q == "GENUINE_DIFFUSE":
                layer_has_genuine = True
        circuit_types.append(1 if layer_has_genuine else 0)

    distribution = mapper.get_distribution()
    is_weights_empty = "GENUINE_COMMITTED" not in distribution or distribution["GENUINE_COMMITTED"] == 0.0

    # Trajectory of genuineness across layers
    trajectory = np.max(dynamic_scores, axis=1)
    rates = fit_circuit_rates(list(trajectory), circuit_types)

    return {
        "phase_space_distribution": distribution,
        "genuine_committed_empty_in_weights": is_weights_empty,
        "empirical_rates": rates
    }

if __name__ == "__main__":
    import json
    model = None
    try:
        # Version 1.0 targets late layers of large models, but we use gpt2-small for verification
        model = HookedTransformer.from_pretrained("gpt2-small")
    except Exception as e:
        print(f"Could not load model: {e}")

    prompt = "The Quick Brown Fox jumps over the lazy dog. Reasoning is the process of using existing knowledge to draw conclusions."
    results = run_transformerlens_phase_analysis(model, prompt)
    print(json.dumps(results, indent=2))
