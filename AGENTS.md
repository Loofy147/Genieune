# Genuineness Framework Instructions

## Codebase Structure
- `genuine_model.py`: Core architecture (Version 2.1) including RoPE, Mechanistic Recurrence, and Thermodynamic Regularizer.
- `sustained_genuineness.py`: Logic utilities for G-score tracking and routing.
- `kaggle_analysis.py`: Main analysis pipeline for evaluating prompts.
- `train_v2_advanced.py`: Advanced training pipeline for Version 2.1 models.
- `DATA_STRATEGY.md`: Strategic recommendations for task and dataset design.

## Operational Checklist: Training New Versions
- [ ] **Task Definition**: Verify task in `get_complex_batch()` satisfies multi-hop logic.
- [ ] **Regularizer Tuning**: Ensure `ThermodynamicRegularizer` parameters match the layer count.
- [ ] **RoPE Alignment**: Confirm `precompute_freqs_cis` matches `seq_len`.
- [ ] **Convergence Monitoring**: Check that both task loss and thermo loss are decreasing.

## Operational Checklist: Evaluation
- [ ] **V1 Baseline**: Run `phase_dynamics.py` on GPT-2 to establish a baseline phase space.
- [ ] **V2 Trajectory**: Execute `kaggle_analysis.py` and inspect `v2_genuineness_trajectory.png`.
- [ ] **Loop Count Analysis**: Verify the `total_processing_steps` in `analysis_results.json` shows effective recurrence usage.

## Tests
- Run tests using `PYTHONPATH=. python3 -m unittest discover tests`.
