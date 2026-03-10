import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformer_lens import HookedTransformer
from phase_dynamics import run_transformerlens_phase_analysis, PhaseSpaceMapper, plot_phase_space
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
g_budget = st.sidebar.slider("Global G-Budget", 1, 16, 12)
show_v1 = st.sidebar.checkbox("Show GPT-2 Phase Space (V1)", value=True)

prompt = st.text_area("Input Prompt", value="Reasoning is a sustained process of complexity. A sequence of logical steps leads to a conclusion.")

if st.button("Analyze Genuineness Trajectory"):
    with st.spinner("Analyzing model dynamics..."):
        # 1. Version 2.2 Model Analysis
        v2_model = GenuineTransformer(d_model=128, n_heads=4, n_layers=4, vocab_size=1000)

        # Load weights if available
        weights_path = "advanced_genuine_model_v2_2.pt"
        if os.path.exists(weights_path):
            try:
                v2_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                st.sidebar.success(f"Loaded weights: {weights_path}")
            except Exception as e:
                st.sidebar.warning(f"Could not load V2.2 weights: {e}")
        else:
            # Fallback to advanced_genuine_model_v2_1.pt
            v1_weights_path = "advanced_genuine_model_v2_1.pt"
            if os.path.exists(v1_weights_path):
                try:
                    v2_model.load_state_dict(torch.load(v1_weights_path, map_location="cpu"), strict=False)
                    st.sidebar.info(f"Loaded V2.1 weights (Compatibility mode).")
                except Exception as e:
                    st.sidebar.warning(f"Could not load weights: {e}")

        v2_model.eval()
        with torch.no_grad():
            # Mock tokens for demonstration (In production, replace with real tokenizer)
            dummy_tokens = torch.randint(0, 1000, (1, 16))
            logits, entropies = v2_model(dummy_tokens, g_budget=g_budget)

            # G-trajectory: var(entropy) per step
            g_scores = [float(torch.var(e, dim=-1).mean().detach()) for e in entropies]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Version 2.2: G-Trajectory")
            fig_v2, ax_v2 = plt.subplots(figsize=(10, 6))
            ax_v2.plot(g_scores, marker='o', linestyle='-', color='blue', linewidth=2)
            ax_v2.axhline(y=0.6, color='r', linestyle='--', label='G-Threshold (0.6)')
            ax_v2.set_xlabel("Processing Step")
            ax_v2.set_ylabel("Genuineness (G-score)")
            ax_v2.set_title("Learned Adaptive Reasoning Trajectory")
            ax_v2.grid(True, alpha=0.3)
            ax_v2.legend()
            st.pyplot(fig_v2)

            st.write(f"**Final G-score**: {round(g_scores[-1], 3)}")
            st.write(f"**Total Reasoning Steps**: {len(entropies)}")

        if show_v1:
            try:
                with col2:
                    st.subheader("Version 1.0: GPT-2 Phase Space")
                    v1_model = HookedTransformer.from_pretrained("gpt2-small")
                    results_v1 = run_transformerlens_phase_analysis(v1_model, prompt)

                    mapper = PhaseSpaceMapper()
                    fig_v1, ax_v1 = plt.subplots(figsize=(10, 8))
                    cost = np.array(results_v1["raw_scores"]["cost"])
                    dynamic = np.array(results_v1["raw_scores"]["dynamic"])

                    # Custom plotting logic for matplotlib integration in Streamlit
                    ax_v1.scatter(cost, dynamic, alpha=0.5, edgecolors='k')
                    ax_v1.set_xlabel("Token Cost (Surprisal)")
                    ax_v1.set_ylabel("Genuineness (Entropy Variance)")
                    ax_v1.set_title(f"Interpretability Quadrants for GPT-2")
                    ax_v1.axhline(y=mapper.GENUINE_THRESHOLD, color='r', linestyle='--', alpha=0.3)
                    ax_v1.axhline(y=mapper.MECHANICAL_THRESHOLD, color='r', linestyle='--', alpha=0.3)
                    ax_v1.grid(True, alpha=0.2)
                    st.pyplot(fig_v1)

                    st.write("**Quadrant Distribution**:")
                    st.json(results_v1["phase_space_distribution"])
            except Exception as e:
                st.error(f"Error running V1 Analysis: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed under the Dynamic Entropy Genuineness Framework (Version 2.2).")
