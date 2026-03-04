import torch
from typing import Dict, List, Tuple

"""
DYNAMIC ENTROPY GENUINENESS FRAMEWORK (Version 2.0 Core Logic)
Support utilities for Mechanistic Recurrence and Genuineness tracking.
"""

# ══════════════════════════════════════════════════════════════════
# PART 1: MECHANISTIC RECURRENCE (LOOPING LOGIC)
# ══════════════════════════════════════════════════════════════════

class MechanisticRecurrence:
    """
    Detects 'Elaboration Pull' (sudden drop in G-score)
    to trigger activation routing in complex reasoning tasks.
    """
    def __init__(self, recurrence_layer=21):
        self.recurrence_layer = recurrence_layer
        self.pull_threshold = -0.20

    def check_and_route(self, layer_idx: int, g_scores: List[float]) -> bool:
        """
        Determines if the model should re-enter reasoning layers.
        """
        if layer_idx < self.recurrence_layer or len(g_scores) < 2:
            return False

        # Delta G: Difference between current and previous G-scores
        delta_g = g_scores[-1] - g_scores[-2]

        # If Genuineness is collapsing (pulling towards mechanical), route back
        return delta_g < self.pull_threshold
