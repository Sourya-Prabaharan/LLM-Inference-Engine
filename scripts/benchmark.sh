#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 benchmarks/benchmark_gemm.py --binary build-cuda/kernel_benchmark
python3 benchmarks/benchmark_attention.py --binary build-cuda/kernel_benchmark
python3 benchmarks/benchmark_kv_cache.py
if [[ $# -gt 0 ]]; then
 python3 benchmarks/benchmark_end_to_end.py --binary build-cuda/llm_inference --model "$1"
fi
