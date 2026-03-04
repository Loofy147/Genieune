# Genuineness Framework Instructions

## Codebase Structure
- `genuine_model.py`: Core architecture (Version 2.1) including RoPE, Mechanistic Recurrence, and Thermodynamic Regularizer.
- `sustained_genuineness.py`: Logic utilities for G-score tracking and routing.
- `kaggle_analysis.py`: Main analysis pipeline for evaluating prompts.
- `train_v2_advanced.py`: Advanced training pipeline for Version 2.1 models.

## Training
- Version 2.1 uses Selective Reversal and Increment as a benchmark task.
- Training should always use `ThermodynamicRegularizer` to sustain attention variance.

## Analysis
- Use `kaggle_analysis.py` to evaluate prompts. It combines V1 interpretability with V2 model metrics.

## Tests
- Run tests using `PYTHONPATH=. python3 -m unittest discover tests`.
