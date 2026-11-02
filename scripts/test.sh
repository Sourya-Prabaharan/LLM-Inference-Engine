#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ctest --test-dir "${1:-build-cuda}" --output-on-failure
python3 -m compileall -q python benchmarks tests
