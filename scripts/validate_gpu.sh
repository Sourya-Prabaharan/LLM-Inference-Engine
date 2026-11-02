#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'CUDA toolkit required' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU required' >&2; exit 1; }
mkdir -p results/gpu
exec > >(tee results/gpu/validation.log) 2>&1
trap 'status=$?; printf "exit_code=%s\n" "$status" > results/gpu/exit_status.txt' EXIT
git rev-parse HEAD > results/gpu/commit.txt
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > results/gpu/hardware.csv
nvcc --version > results/gpu/nvcc.txt
python3 -m pip freeze > results/gpu/python-packages.txt
./scripts/build.sh cuda
ctest --test-dir build-cuda --output-on-failure
if command -v compute-sanitizer >/dev/null; then
 compute-sanitizer --error-exitcode 1 --tool memcheck ./build-cuda/test_kernels
 compute-sanitizer --error-exitcode 1 --tool racecheck ./build-cuda/test_kernels
else
 echo 'compute-sanitizer is required for full validation' >&2; exit 1
fi
python3 python/download_model.py --output models/tinyllama-hf
for precision in fp32 fp16 int8; do
 python3 python/convert_weights.py models/tinyllama-hf "models/tinyllama-$precision" --precision "$precision"
done
python3 tests/test_tokenizer.py --assets models/tinyllama-hf --binary build-cuda/tokenizer_dump
python3 tests/test_inference.py --binary build-cuda/llm_inference --model models/tinyllama-fp32 --reference-model models/tinyllama-hf --device cuda
python3 python/validate_logits.py --binary build-cuda/llm_inference --model models/tinyllama-fp32 --reference-model models/tinyllama-hf --device cuda --output results/gpu/logits.json
python3 benchmarks/benchmark_gemm.py --binary build-cuda/kernel_benchmark --output results/gpu/gemm.json
python3 benchmarks/benchmark_attention.py --binary build-cuda/kernel_benchmark --output results/gpu/attention.json
python3 benchmarks/benchmark_end_to_end.py --binary build-cuda/llm_inference --model models/tinyllama-fp16 --output results/gpu/end_to_end.json
python3 benchmarks/benchmark_batching.py --binary build-cuda/llm_inference --model models/tinyllama-fp16 --requests configs/workload.tsv --output results/gpu/batching.json
# No smoke-test token limit: evaluate the complete WikiText-2 test split.
python3 benchmarks/benchmark_quantization.py --binary build-cuda/llm_inference --fp16-model models/tinyllama-fp16 --int8-model models/tinyllama-int8 --reference-model models/tinyllama-hf --output results/gpu/quantization.json
./build-cuda/cache_benchmark > results/gpu/kv_gpu.json
python3 benchmarks/benchmark_kv_cache.py --output results/gpu/kv_accounting.json
python3 benchmarks/report_gpu_results.py --directory results/gpu
