# Genuineness Framework: Standard Operating Methodology

This document defines the iterative loop for improving and evolving the Dynamic Entropy Genuineness Framework. This methodology MUST be followed in every session.

## The Iterative Training Loop

1.  **Deployment**: Push the latest model architecture and training script to Kaggle.
2.  **Autonomous Monitoring**: Wait for the training kernel to complete. Use `kaggle kernels status` to track progress.
3.  **Artifact Extraction**: Once complete, fetch the logs (`training_log.txt`) and output artifacts (`analysis_results.json`, model weights) using `kaggle kernels output`.
4.  **Performance Analysis**:
    *   **Task Accuracy**: Verify if Task Loss converged below target (e.g., < 0.1 for binary parity).
    *   **Genuineness Magnitude**: Analyze G-score stability. (Target: 2.0 - 5.0 for V2.2 Scaled).
    *   **Signal-to-Noise Ratio**: Ensure the thermodynamic loss is regularizing, not dominating (Reward Hacking Check).
5.  **Recalculation of Improvement**: Compare current run metrics against the previous version baseline.
6.  **Concurrent Research**: While training is running, perform research on "System Evolution Concepts" (e.g., adaptive gating refinements, layer-wise probing, or sparsity).
7.  **System Evolution**: Implement fixes or new concepts identified in step 4 and 6 into the next version.

## Versioning & Documentation
*   Always update `AGENTS.md` and the Technical Report after each successful loop.
*   Document specific "Criticisms" and "Assessments" from each run in the Technical Report.

## Evolutionary Leap: Version 3.0 (Sparsity & Task Dominance)

### Findings from V2.2 Scaled Runs
- **Reward Hacking**: High capacity models (d_model=512) will maximize genuineness signals (G-score) by inflating noise if the task gradient is not sufficiently weighted.
- **Sparsity Necessity**: High-entropy noise in attention heads can drown out reasoning. Entropy-Gated Sparsity acts as a selective filter for information.

### V3.0 Methodology Refinement
- **Task Dominance**: Always use a `5.0x` Task Weighting multiplier during the convergence phase.
- **Selective Genuineness**: Apply `0.01x` or lower thermodynamic weight, ensuring it acts as a regularizer, not a driver.
- **Curriculum**: Minimum 1000 epoch warmup for objective logic before introducing complexity priors.
