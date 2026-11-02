#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cuda}"
case "$mode" in
 cuda) cmake -S . -B build-cuda -DLLM_CUDA=ON -DCMAKE_BUILD_TYPE=Release "${@:2}"; cmake --build build-cuda --parallel 2 ;;
 host) cmake -S . -B build-host-release -DLLM_CUDA=OFF -DCMAKE_BUILD_TYPE=Release "${@:2}"; cmake --build build-host-release --parallel 2 ;;
 *) echo 'Usage: scripts/build.sh cuda|host [CMake arguments]' >&2; exit 2 ;;
esac
