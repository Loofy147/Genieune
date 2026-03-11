# Genuineness Framework Instructions (Version 2.2 Scaled)

## Codebase Structure
- `genuine_model.py`: Core architecture (Version 2.2 Scaled) including RoPE, Mechanistic Recurrence, and Thermodynamic Regularizer (d_model=512, n_layers=12).
- `sustained_genuineness.py`: Logic utilities for G-score tracking and routing.
- `kaggle_analysis.py`: Main analysis pipeline for evaluating prompts using both V1 Phase Space and V2.1 G-Trajectories.
- `train_v2_advanced.py`: Advanced training pipeline using the **Cumulative Multi-Step Reasoning Task**.
- `push_to_hf.py`: Utility to synchronize model, code, and results to Hugging Face.
- `deploy_kaggle.sh`: Deployment script for Kaggle kernels and model resources.
- `config.json`: Model configuration for Hugging Face download tracking and metadata.

## Operational Checklist: Training & Improvement
- [x] **Task Definition**: Uses cumulative history parity logic to force cross-token reasoning.
- [x] **Model Scaling**: Capacity increased to 512 dimensions and 12 layers.
- [x] **Regularizer Tuning**: `ThermodynamicRegularizer` configured with `variance_weight=2.0` and `mechanical_penalty=0.45`.
- [ ] **Monitoring**: Check `analysis_results.json` after training for G-score > 0.6.

## Verification & Monitoring
- **G-Score**: Measures dynamic complexity. Target > 0.6.
- **G-Trajectory**: Visualized in `v2_genuineness_trajectory.png`.
- **Phase Space**: Baseline visualized in `v1_phase_space_distribution.png`.
- **Downloads**: Tracked via `config.json` on Hugging Face (LOOFYYLO/dynamic-entropy-genuineness-v2-1).

## Tests
- Run tests using `PYTHONPATH=. python3 -m unittest discover tests`.
