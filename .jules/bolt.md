## 2025-05-15 - [Matmul vs Einsum & Entropy Optimization]
**Learning:** In PyTorch, `torch.matmul` with explicit transposes is consistently faster than `torch.einsum` for standard multi-head attention blocks. Additionally, calculating Shannon entropy using `F.log_softmax` instead of `torch.log2(F.softmax(...))` is more numerically stable and slightly faster. Registering positional embeddings as buffers ensures zero-overhead device management.
**Action:** Prefer `matmul` and `register_buffer` for high-frequency tensor operations and positional embeddings.
