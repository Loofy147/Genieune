## 2025-05-15 - [Matmul vs Einsum & Entropy Optimization]
**Learning:** In PyTorch, `torch.matmul` with explicit transposes is consistently faster than `torch.einsum` for standard multi-head attention blocks. Additionally, calculating Shannon entropy using `F.log_softmax` instead of `torch.log2(F.softmax(...))` is more numerically stable and slightly faster. Registering positional embeddings as buffers ensures zero-overhead device management.
**Action:** Prefer `matmul` and `register_buffer` for high-frequency tensor operations and positional embeddings.

## 2025-05-15 - [Vectorized Regularizer Loss]
**Learning:** Vectorizing the `ThermodynamicRegularizer` loss by stacking layer-wise entropy tensors into a single [L, B, S, H] tensor and using `torch.where` for conditional penalties yields a ~34% performance improvement. It also makes the code more readable and robust to varying batch/sequence dimensions.
**Action:** Use `torch.stack` and vectorized conditional operations for loss functions that iterate over layers or sequences.

## 2025-05-15 - [Advanced RoPE and G-score Optimizations]
**Learning:** Slicing and reshapping positional embeddings (RoPE) once per forward pass instead of per layer/head reduces memory bandwidth and indexing overhead by ~42% for the embedding step. Vectorizing G-score (variance of entropy) across layers using `torch.stack` provides a significant boost over Python-level list comprehensions.
**Action:** Move all fixed-dimension slicing and reshaping operations to the outermost loop or entry point of a module.
