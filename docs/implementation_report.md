# Implementation report

## Source delivered

- `include/` and `src/`: tensor/stream ownership, model configuration and binary loader,
  group quantization, paged allocator, scheduling, transformer execution, generation,
  native tokenizer, and command-line interface.
- `kernels/gemm.cu`: naive/tiled floating weights, FP16 inputs, and grouped INT8 reads.
- `kernels/ops.cu`: normalization, rotation, softmax, embedding, activation and residual.
- `kernels/attention.cu`: dense baseline, fused online attention, direct paged K/V access.
- `python/`: pinned download, streaming conversion, HF reference export, strict logit
  comparison and WikiText-2 perplexity evaluation.
- `benchmarks/`: CUDA-event kernel executable and six benchmark/accounting scripts.
- `tests/`: systems, conversion/corruption, native tokenizer parity, CUDA kernels,
  GPU integration, and explicit CPU-reference transformer orchestration tests.
- `configs/`, `scripts/`, `docs/`, `results/`: configuration, reproducible commands,
  design notes, validation status and actual local evidence.

Related operations are grouped into shared CUDA files rather than empty one-operation
files. Transformer orchestration resides in `model.cpp`; this is an intentional layout
adjustment, not a placeholder `transformer.cpp`.

## Verification

See `results/benchmark_results.md` for measured local outcomes and numerical results.
The model and engine C++ sources compile with an explicit test-only CPU arithmetic
backend. This verifies orchestration independently of CUDA. There is no CPU fallback
inside the GPU executable and no CPU result is presented as a GPU measurement.

CUDA compilation, GPU kernel tests, TinyLlama 1.1B inference, real model quantization
metrics and all latency/throughput targets remain unexecuted. There is no NVIDIA
hardware/toolkit on the development machine. Full model weights were not downloaded
because the requested GPU integration cannot run here; the tiny synthetic fixture
is generated locally and native tokenizer assets are downloaded separately.

## Commands

```bash
./scripts/setup.sh
source .venv/bin/activate
./scripts/build.sh cuda
./scripts/test.sh build-cuda
python python/download_model.py
python python/convert_weights.py models/tinyllama-hf models/tinyllama-fp16 --precision fp16
./build-cuda/llm_inference --model models/tinyllama-fp16 --prompt 'Explain CUDA kernels.' --chat --max-tokens 128
./scripts/benchmark.sh models/tinyllama-fp16
```

For this Mac: `./scripts/build.sh host && ./scripts/test.sh build-host-release`.
See README for FP32 validation, INT8 conversion, tokenizer tests, batch workloads,
compute-sanitizer and profiling commands. See `implementation_status.md` for the
specific missing capabilities and remaining validation gates.
