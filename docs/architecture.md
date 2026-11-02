# Architecture and binary contract

`Tensor` owns one contiguous allocation, is move-only, validates positive dimensions
and overflow, and tracks current/peak CUDA bytes. Stream copies are asynchronous;
callers retain host buffers and synchronize before consuming outputs or releasing
resources. `Workspace` grows only when needed; transformer work buffers are allocated
once for the maximum packed token count. The runtime uses one nonblocking CUDA stream.

`Model` owns validated weights, scratch tensors, and K/V pools. One forward call
accepts packed token IDs, absolute positions, and one block table per packed row.
Several rows may belong to the same request during prefill. All K/V writes precede
attention launches on the same stream. Attention masks future positions even when
later prompt tokens have already been written.

A transformer layer performs input RMSNorm; Q/K/V projections; split-half Llama
RoPE; grouped-query causal attention; output projection and residual; post-attention
RMSNorm; gate/up projections; SiLU(gate)*up; down projection and residual. Final
RMSNorm and LM head produce logits. The runtime supports head dimensions up to 128,
integer grouped-query ratios, unscaled full-head RoPE, no projection biases, and SiLU.
Unsupported configurations fail conversion or loading.

## LLMENG01 file

All integers and floating-point fields are little endian. No padding is inserted.
The reader checks endianness, truncation, names, duplicates, dtype, rank, shape, byte
counts, positive scales, required tensor set, and unexpected trailing bytes.

1. Eight-byte ASCII magic `LLMENG01`.
2. Nine uint32 values: hidden, intermediate, layers, Q heads, KV heads, vocabulary,
   context, BOS ID, EOS ID.
3. Two float32 values: RMS epsilon, RoPE base.
4. uint32 tensor count, exactly `3 + 9*layers`.
5. For each tensor: uint32 UTF-8 name length, name bytes, uint32 dtype
   (`0=FP32,1=FP16,2=INT8`), uint32 rank, uint32 group size (zero for float),
   rank uint64 dimensions, uint64 data-byte count, contiguous data bytes.
6. INT8 tensors immediately append `rows * ceil(cols/group)` float32 scales.

Matrices are `[output,input]`; linear kernels compute `X * W^T`. Native FP16
payloads use IEEE binary16. Embeddings are `[vocab,hidden]`. Norms are FP32 vectors.
No Python process is involved after conversion. Conversion intentionally follows
`--precision`, rather than pretending original BF16 and converted FP16 are identical.

## Verification layers

Host tests check ownership, configuration, binary conversion and allocation behavior.
A **test-only CPU arithmetic backend** compiles the same model and engine orchestration
and compares a deterministic tiny Llama against Hugging Face. It is linked only into
`test_model_reference`, never `llm_inference` or benchmark binaries. Its results cannot
certify CUDA indexing, synchronization, numerical behavior, or performance.
CUDA tests separately cover kernels and direct paged indexing. Full TinyLlama logits,
generation, and batching tests require the NVIDIA execution path.
