# Reproducible measurement

GPU measurements are unexecuted until an NVIDIA machine is available. CPU tests and
allocation formulas are never labeled GPU performance.

`kernel_benchmark` warms up five times, synchronizes, and records 30 samples using
CUDA events on the execution stream. Inputs and output buffers are allocated before
timing. JSON stores raw milliseconds, mean, median, variance, device name, CUDA
runtime version, dtype, geometry, and maximum output error.

GEMM compares naive custom, tiled custom, and cuBLAS pedantic SGEMM on the same
FP32 inputs. TFLOPS = `2*M*N*K/(median_ms*1e9)`; custom/cuBLAS throughput fraction =
`cublas_ms/custom_ms`. The supplied benchmark does not yet measure FP16 GEMMEx.
Attention compares materialized scores/softmax/value against online attention for
prefill and single-query decode independently. The saved score buffer size is the
baseline intermediate allocation, not a hardware memory-traffic measurement.

End-to-end benchmarks load once per process, perform one warmup request, then time
`Engine::run` using steady wall clock. Each forward synchronizes its output transfer.
The run includes scheduling, prefill, decode, sampling and configured arrival gaps;
it excludes loading and warmup. Scripts repeat five times and retain per-request
TTFT/completion latency. `peak_tracked_gpu_bytes` covers Tensor-owned weights, cache,
and scratch, not CUDA context/driver allocations. GPU utilization and memory traffic
require Nsight and are not guessed. Detailed kernel/phase timing and TPOT distributions
remain to be added; overall generated tokens/s is not pure decode throughput.

Quantization compares actual exported model-file sizes and separately evaluates
WikiText-2 test perplexity. Evaluation uses non-overlapping 512-token windows, scores
next tokens within each window and omits cross-window transitions. FP16 and INT8 must
use the same tokenizer, corpus, context, and token limit. `--max-tokens` is an optional
smoke test limit; never present that as full-corpus perplexity. The evaluator loads the engine once and scores windows with packed prefill, using
FP64 log-sum-exp on the host. Its duration is not an inference performance metric.

Useful commands:

```bash
nsys profile -o results/inference ./build-cuda/llm_inference --model models/tinyllama-fp16 --prompt Hello --max-tokens 128
ncu --set full ./build-cuda/kernel_benchmark gemm 128 2048 2048
python benchmarks/benchmark_attention.py --binary build-cuda/kernel_benchmark
python benchmarks/benchmark_batching.py --binary build-cuda/llm_inference --model models/tinyllama-fp16 --requests configs/workload.tsv
```

The 4096-token synthetic attention case is a kernel stress test, not a claim that
TinyLlama's configured 2048-token model context has been extended.
