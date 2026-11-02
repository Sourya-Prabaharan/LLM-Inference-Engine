# Completing NVIDIA validation

The eight source phases are pushed to GitHub. The Linux CI workflow compiles the
CUDA targets in an NVIDIA CUDA development container. That runner has no GPU;
a successful compile is not a runtime test.

Use an existing authorized Linux NVIDIA machine with CUDA 12.x, `compute-sanitizer`,
CMake, Python 3.11+, and sufficient disk for the source checkpoint and three converted
variants (allow at least 15 GB free). FP32 validation needs enough VRAM for the model,
cache, and activations. The reference test releases its GPU model before launching
the C++ process so two full models do not compete for VRAM.

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements.txt
./scripts/validate_gpu.sh
```

The script checks hardware, captures the Git commit and environment, builds, runs
kernel and sanitizer tests, prepares TinyLlama, validates FP32 logits and generation
through 2048 tokens, tests batching parity, measures GEMM/attention/inference/batching,
and evaluates the entire WikiText-2 test split for FP16 and INT8. It stops on
correctness failures and preserves logs. Perplexity now uses one resident model,
packed prefill, and host FP64 log-sum-exp rather than repeated process/model loading.
Performance targets are reported as met or unmet based only on resulting JSON.

The manual `NVIDIA runtime validation` workflow runs the same script on a self-hosted
runner labeled `linux`, `x64`, and `gpu` and uploads evidence even on failure. It is
not auto-dispatched on pushes, and does not provision or charge for cloud resources.
Do not register an untrusted public runner or use an unrelated machine without authorization.

No accessible GPU was identified in this session: the existing `cs544` SSH host
timed out. An available SSH host or approved GPU provider/budget is required before
runtime validation can actually be completed. No GPU result is inferred from the
availability of this script or workflow.
