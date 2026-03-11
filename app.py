import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformer_lens import HookedTransformer
from phase_dynamics import run_transformerlens_phase_analysis, PhaseSpaceMapper
from genuine_model import GenuineTransformer

st.set_page_config(page_title="Dynamic Entropy Genuineness Framework", layout="wide")

st.title("Dynamic Entropy Genuineness Framework (Version 2.2 Advanced)")
st.markdown("""
This application analyzes model trajectories using the **Genuineness Phase Space**.
- **Token Cost (X)**: External information density (surprisal).
- **Dynamic Genuineness (Y / G-score)**: Internal complexity (entropy variance).
""")

# Sidebar for controls
st.sidebar.header("Model Configuration")
g_budget = st.sidebar.slider("Global G-Budget", 1, 24, 12)
show_v1 = st.sidebar.checkbox("Show GPT-2 Phase Space (V1)", value=True)

prompt = st.text_area("Input Prompt", value="The quick brown fox jumps over the lazy dog. Reasoning is the process of using existing knowledge to draw conclusions.")

def text_to_tokens(text, vocab_size=1000):
    """Simple deterministic mapping from text to tokens for demo."""
    # Use sum of ordinals or hash for deterministic token mapping in the demo vocab
    tokens = []
    words = text.split()
    for word in words[:16]: # Max 16 tokens for demo
        val = sum(ord(c) for c in word) % vocab_size
        tokens.append(val)
    while len(tokens) < 16:
        tokens.append(0)
    return torch.tensor([tokens])

if st.button("Analyze Genuineness Trajectory"):
    with st.spinner("Analyzing model dynamics..."):
        # 1. Version 2.2 Model Analysis (d_model=256, n_heads=8, n_layers=6 as trained)
        v2_model = GenuineTransformer(d_model=256, n_heads=8, n_layers=6, vocab_size=1000)

        # Load weights
        weights_path = "advanced_genuine_model_v2_2.pt"
        if os.path.exists(weights_path):
            try:
                v2_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                st.sidebar.success(f"Loaded weights: {weights_path}")
            except Exception as e:
                st.sidebar.warning(f"Could not load V2.2 weights: {e}")
        else:
            v1_weights_path = "advanced_genuine_model_v2_1.pt"
            if os.path.exists(v1_weights_path):
                try:
                    v2_model.load_state_dict(torch.load(v1_weights_path, map_location="cpu"), strict=False)
                    st.sidebar.info("Loaded V2.1 weights (Compatibility mode).")
                except Exception as e:
                    st.sidebar.warning(f"Could not load V2.1 weights: {e}")

        v2_model.eval()
        with torch.no_grad():
            v2_tokens = text_to_tokens(prompt)
            logits, entropies = v2_model(v2_tokens, g_budget=g_budget)

            # G-trajectory: var(entropy) per step across heads
            # entropies: list of [batch, seq, heads]
            g_scores = [float(torch.var(e, dim=-1).mean().detach()) for e in entropies]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Version 2.2: Adaptive G-Trajectory")
            if g_scores:
                fig_v2, ax_v2 = plt.subplots(figsize=(10, 6))
                ax_v2.plot(g_scores, marker='o', linestyle='-', color='blue', linewidth=2)
                ax_v2.axhline(y=0.6, color='r', linestyle='--', label='G-Threshold (0.6)')
                ax_v2.set_xlabel("Processing Step (Layers + Loops)")
                ax_v2.set_ylabel("Dynamic Genuineness (G-score)")
                ax_v2.set_title("Learned Adaptive Reasoning Path")
                ax_v2.grid(True, alpha=0.3)
                ax_v2.set_ylim(0, max(1.0, max(g_scores) * 1.2))
                ax_v2.legend()
                st.pyplot(fig_v2)

                st.write(f"**Final G-score**: {round(g_scores[-1], 4)}")
                st.write(f"**Total Reasoning Steps**: {len(entropies)}")
            else:
                st.warning("No G-scores generated.")

        if show_v1:
            try:
                with col2:
                    st.subheader("Version 1.0: GPT-2 Phase Space (Interpretability)")
                    v1_model = HookedTransformer.from_pretrained("gpt2-small")
                    results_v1 = run_transformerlens_phase_analysis(v1_model, prompt)

                    mapper = PhaseSpaceMapper()
                    fig_v1, ax_v1 = plt.subplots(figsize=(10, 8))
                    cost = np.array(results_v1["raw_scores"]["cost"])
                    dynamic = np.array(results_v1["raw_scores"]["dynamic"])

                    ax_v1.scatter(cost, dynamic, alpha=0.5, edgecolors='k')
                    ax_v1.set_xlabel("Token Cost (Surprisal)")
                    ax_v1.set_ylabel("Genuineness (Entropy Variance)")
                    ax_v1.set_title(f"Interpretability Quadrants for GPT-2")
                    # Corrected attribute names
                    ax_v1.axhline(y=mapper.genuine_threshold, color='g', linestyle='--', alpha=0.3, label='Genuine')
                    ax_v1.axhline(y=mapper.mechanical_threshold, color='r', linestyle='--', alpha=0.3, label='Mechanical')
                    ax_v1.axvline(x=mapper.cost_threshold, color='k', linestyle='--', alpha=0.3)
                    ax_v1.set_xlim(0, 1)
                    ax_v1.set_ylim(0, 1)
                    ax_v1.grid(True, alpha=0.2)
                    ax_v1.legend()
                    st.pyplot(fig_v1)

                    st.write("**Quadrant Distribution**:")
                    st.json(results_v1["phase_space_distribution"])
            except Exception as e:
                st.error(f"Error running V1 Analysis: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed under the Dynamic Entropy Genuineness Framework (Version 2.2).")
