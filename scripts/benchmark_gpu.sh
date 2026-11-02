#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
mkdir -p results/gpu
exec > >(tee results/gpu/benchmarks.log) 2>&1
trap 'status=$?; printf "exit_code=%s\n" "$status" > results/gpu/benchmark_exit_status.txt' EXIT
git rev-parse HEAD > results/gpu/benchmark_commit.txt
python3 benchmarks/benchmark_gemm.py --binary build-cuda/kernel_benchmark --output results/gpu/gemm.json
python3 benchmarks/benchmark_attention.py --binary build-cuda/kernel_benchmark --output results/gpu/attention.json
python3 benchmarks/benchmark_end_to_end.py --binary build-cuda/llm_inference --model models/tinyllama-fp16 --output results/gpu/end_to_end.json
python3 benchmarks/benchmark_batching.py --binary build-cuda/llm_inference --model models/tinyllama-fp16 --requests configs/workload.tsv --output results/gpu/batching.json
# No smoke-test token limit: evaluate the complete WikiText-2 test split.
python3 benchmarks/benchmark_quantization.py --binary build-cuda/llm_inference --fp16-model models/tinyllama-fp16 --int8-model models/tinyllama-int8 --reference-model models/tinyllama-hf --output results/gpu/quantization.json
./build-cuda/cache_benchmark > results/gpu/kv_gpu.json
python3 benchmarks/benchmark_kv_cache.py --output results/gpu/kv_accounting.json
python3 benchmarks/report_gpu_results.py --directory results/gpu
