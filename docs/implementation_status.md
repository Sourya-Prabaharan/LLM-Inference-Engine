# Implementation status

Updated 2026-11-02. Development host: macOS ARM64, 8 GiB memory,
AppleClang 17, Python 3.13. No nvcc, CUDA toolkit, or NVIDIA device.
Public model assets are reachable with network access outside the sandbox.
Only tokenizer/configuration assets were downloaded; full TinyLlama weights were not.

`Implemented` means source exists and is wired into its caller. It does not mean
CUDA code has compiled or passed GPU tests. No phase requiring GPU validation is
marked complete. A test-only CPU numerical backend is explicitly isolated from
production inference and GPU benchmarks.

| Phase | Implementation | Executed validation | GPU benchmarks | Remaining gate |
|---|---|---|---|---|
| 1 Infrastructure | CMake, move-only tensors, streams, copies, overflow checks, memory counters, grow-only workspace | Host build and systems tests pass | Not applicable | CUDA allocation/copy and memory-check execution |
| 2 Kernels | Naive/tiled GEMM, FP16 GEMM, grouped INT8 GEMM, RMSNorm, split-half RoPE, stable softmax, embedding, residual, SwiGLU | CUDA test source written; host orchestration compiles | Not run | CUDA compile, CPU/reference parity, compute-sanitizer |
| 3 Inference | Validated LLMENG01 weights, native SentencePiece, Llama layers, sampling, EOS, chunked prefill/decode, CLI | Synthetic C++ model against HF passes via explicit CPU reference backend; tokenizer 13 cases pass | Not run | Full TinyLlama CUDA logits and generation |
| 4 Attention | Materialized dense baseline and shared-memory tiled online-softmax attention, GQA and causal masks | GPU test source includes dense/paged parity and permuted blocks | Not run | Numerical CUDA tests, attention speedups |
| 5 Quantization | Offline grouped INT8 converter, C++ quantizer, scales, fused dequantization, loading/runtime integration | Conversion, error bounds, synthetic model paths pass on host | Not run | FP16/INT8 full-model size, GPU throughput, WikiText-2 |
| 6 Cache | Host block pool, atomic extension, direct GPU paged K/V access | Host boundaries, OOM rollback, isolation and reuse; synthetic paged model passes | Analytical accounting only | GPU race/isolation tests, real capacity measurements |
| 7 Batching | Static/continuous FIFO admission, arrival times, packed chunked prefill and decode projections | Admission tests; synthetic single-vs-batched parity for both policies | Not run | CUDA multi-request integration and workload measurement |
| 8 Finalization | Build/test scripts, benchmark programs, reference utilities, README and technical docs | Host tests and Python syntax checks | GPU results deliberately absent | NVIDIA validation and profiling |

## Unimplemented or limited requirements

- No Tensor Core GEMM, autotuning, or cuBLAS inference backend; cuBLAS is a benchmark baseline.
- Runtime activations and KV are FP32. FP16 input GEMM is available separately;
  there is no full FP16-activation transformer. FP16 RMSNorm inputs have a separate entry point.
- Attention processes one query/head per CTA. It is online and tiled, but not a
  reproduction of highly tuned FlashAttention kernels.
- Contiguous cache baseline has accounting and a CUDA allocation benchmark, not an alternate serving backend.
- The synchronous workload runner supports scheduled arrivals, not live concurrent
  external submission. It reports cache OOM per request rather than retrying or swapping.
- Quantization perplexity, real model storage reduction, CUDA numerical accuracy,
  memory leak freedom and every GPU performance target remain unverified.
- Kernel benchmarks currently measure FP32; FP16 GEMMEx comparisons and Nsight
  utilization/traffic collection remain to be implemented.
- End-to-end JSON has TTFT and completion latency, not separate device prefill/decode
  phase timing or a full TPOT distribution.
- Native chat formatting supports a single user turn. Generic model chat templates
  and multi-turn conversation APIs are not implemented.
- Host AddressSanitizer execution stalled before useful output and was stopped.
  UndefinedBehaviorSanitizer is checked separately; see saved test results.

A production-quality completion claim is not justified until these gates are resolved.
