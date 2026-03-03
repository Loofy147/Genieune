import torch
import torch.nn as nn
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.0 Prototype)
Differentiable Regularization and Architecture logic.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: THERMODYNAMIC REGULARIZER (THE NEW LOSS FUNCTION)
# ══════════════════════════════════════════════════════════════════

class ThermodynamicRegularizer:
    """
    Calculates a differentiable mechanistic loss based on attention entropy dynamics.
    Rewards high variance (reasoning) and penalizes static states (mechanical).
    """
    def __init__(self, genuine_threshold=0.55, mechanical_threshold=0.35):
        self.genuine_threshold = genuine_threshold
        self.mechanical_threshold = mechanical_threshold

    def calculate_loss(self, entropies: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the differentiable thermodynamic penalty/reward.

        Args:
            entropies: List of [batch, seq] tensors from each head.
        """
        total_loss = torch.tensor(0.0, device=entropies[0].device)

        for head_ent in entropies:
            # 1. Variance Reward (Maximize internal complexity)
            # We want to minimize -Var(H), which is maximizing Var(H)
            var_h = torch.var(head_ent, dim=-1).mean()
            total_loss = total_loss - 0.5 * var_h

            # 2. Static Penalty (Penalize prolonged low entropy)
            # If mean entropy is below threshold, add penalty
            mean_h = head_ent.mean()
            if mean_h < self.mechanical_threshold:
                # Quadratic penalty for staying below threshold
                total_loss = total_loss + torch.pow(self.mechanical_threshold - mean_h, 2)

        return total_loss


# ══════════════════════════════════════════════════════════════════
# PART 2: DUAL-STREAM SIMULATOR (DECOUPLED REASONING)
# ══════════════════════════════════════════════════════════════════

class DualStreamSimulator:
    """
    Logic for the Phase-Space Gate and Decoupled Forward Pass.
    """
    def __init__(self, n_reasoner_layers=32, n_decoder_layers=8):
        self.n_reasoner_layers = n_reasoner_layers
        self.n_decoder_layers = n_decoder_layers
        self.gate_threshold = 0.55

    def forward(self, initial_state: float, max_loops=5) -> Dict:
        """
        Renamed to 'forward' to match usage in kaggle_analysis.py
        """
        current_g = initial_state
        history = []
        loop_count = 0

        is_thinking = True
        while is_thinking and loop_count < max_loops:
            loop_count += 1
            current_g += 1.2371 * (1.0 - current_g) * 0.1
            current_g = min(current_g, 0.95)
            history.append({"state": "Thinking", "G": round(float(current_g), 3)})
            if current_g > 0.85: is_thinking = False

        final_output_g = current_g
        for _ in range(self.n_decoder_layers):
            final_output_g -= 0.8129 * final_output_g * 0.1
            history.append({"state": "Talking", "G": round(float(final_output_g), 3)})

        return {
            "final_g": round(float(final_output_g), 3),
            "loops_in_reasoner": loop_count,
            "trajectory": history
        }


# ══════════════════════════════════════════════════════════════════
# PART 3: MECHANISTIC RECURRENCE (LOOPING LOGIC)
# ══════════════════════════════════════════════════════════════════

class MechanisticRecurrence:
    """
    Detects 'Elaboration Pull' to trigger activation routing.
    """
    def __init__(self, recurrence_layer=21):
        self.recurrence_layer = recurrence_layer
        self.pull_threshold = -0.20

    def check_and_route(self, layer_idx: int, g_scores: List[float]) -> bool:
        if layer_idx < self.recurrence_layer or len(g_scores) < 2:
            return False
        delta_g = g_scores[-1] - g_scores[-2]
        return delta_g < self.pull_threshold
