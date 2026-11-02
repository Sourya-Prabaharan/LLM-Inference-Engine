# Results — 2026-11-02

## Execution environment

macOS ARM64, 8 GiB memory, AppleClang 17, Python 3.13.1. No CUDA toolkit,
`nvcc`, or NVIDIA GPU. CUDA CMake configuration correctly fails explicitly.
[environment.json](environment.json) records actual installed package versions.
No full TinyLlama checkpoint was downloaded or executed.

## Executed correctness checks

| Check | Actual outcome |
|---|---|
| Host CMake build | Pass |
| Systems test: tensors, cache, scheduler, quantization | Pass |
| FP32/FP16/INT8 conversion and C++ loading; malformed-file rejection | Pass |
| Synthetic Llama transformer against HF using test-only CPU arithmetic | Pass |
| Batched/chunked generation vs independent generation, static and continuous | Pass on synthetic CPU-reference model |
| Native tokenizer vs HF | 10 raw cases and 3 chat cases pass |
| UndefinedBehaviorSanitizer host suite | 3/3 CTest groups pass, no reported UB |
| Python syntax compilation | Pass |
| AddressSanitizer | Executables compiled; execution stalled and was stopped; no pass claimed |
| CUDA compilation, kernels, compute-sanitizer | Not executed: toolkit/device unavailable |
| Full TinyLlama reference and generation tests | Not executed |

The synthetic model has two layers, hidden dimension 16, four query heads, two KV
heads, vocabulary 32, and a 19-token test sequence. These are **CPU-reference**
measurements of the real C++ transformer orchestration, not GPU correctness results.

| Converted weights | Max absolute logit error against FP32 HF |
|---|---:|
| FP32 | 3.725290298461914e-08 |
| FP16 | 8.813291788101196e-05 |
| Grouped INT8, group size 7 | 0.0010007582604885101 |

FP32 passes a 1e-5 synthetic validation gate. FP16/INT8 errors are reported as
observations; no relaxed TinyLlama accuracy gate is inferred. All three modes pass
synthetic independent-vs-batched generation parity. Raw results:
[cpu_reference.jsonl](cpu_reference.jsonl), [tokenizer_tests.txt](tokenizer_tests.txt),
[ubsan_tests.txt](ubsan_tests.txt).

## KV allocation accounting

Workload: token lengths `[17,31,65,127,255,511,1023]`, maximum context 2048,
block size 16, 22 layers, 4 KV heads, dimension 64, FP32 K/V.

| Quantity | Bytes |
|---|---:|
| Live KV data | 91,418,624 |
| Fixed maximum-context assigned capacity | 645,922,816 |
| Paged assigned capacity | 92,995,584 |
| Fixed-capacity waste | 554,504,192 |
| Paged internal fragmentation | 1,576,960 |

Assigned-capacity waste reduction: **99.715609%**, computed from the stated geometry.
This is an analytical allocation calculation, **not** measured GPU memory consumption.
The GPU pool remains reserved; unassigned pool blocks are reusable capacity. It does
not establish the requested runtime GPU memory target. Raw data: [kv_cache.json](kv_cache.json).

## Performance targets

| Target | Status |
|---|---|
| TinyLlama max logit error ≤1e-3 | Unverified; only tiny synthetic CPU reference passed |
| Custom GEMM ≥80% cuBLAS throughput | Unmeasured |
| Attention ≥3.5× at 2048 tokens | Unmeasured |
| INT8 weight-file reduction approximately 47% | Unmeasured on TinyLlama |
| WikiText-2 perplexity increase <0.1 | Unevaluated |
| Continuous batching approximately 3×, 600→1800 tokens/s | Unmeasured |
| GPU cache waste reduction ≥90% | Unverified on GPU; analytical workload exceeds it |

There are no GPU latency, throughput, utilization, or perplexity numbers to report.
No performance target is certified achieved.

## Reproduce

```bash
./scripts/build.sh host
./scripts/test.sh build-host-release
python tests/test_model_reference.py build-host-release/test_model_reference
python benchmarks/benchmark_kv_cache.py
# NVIDIA machine:
./scripts/build.sh cuda
./scripts/test.sh build-cuda
./scripts/benchmark.sh models/tinyllama-fp16
```

For the UB-sanitized host run:

```bash
cmake -S . -B build-ubsan -DLLM_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS=-fsanitize=undefined -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined
cmake --build build-ubsan --parallel 2
ctest --test-dir build-ubsan --output-on-failure
```
