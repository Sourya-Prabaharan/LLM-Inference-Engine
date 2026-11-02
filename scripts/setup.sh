#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Activate with: source .venv/bin/activate' 'CUDA builds require a separately installed NVIDIA driver and CUDA toolkit (12.x).'
