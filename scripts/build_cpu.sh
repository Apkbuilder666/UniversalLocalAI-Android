#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$project_dir/build}"

cmake -S "$project_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLOCALAI_ENABLE_CUDA=OFF \
  -DLOCALAI_ENABLE_HIP=OFF \
  -DLOCALAI_ENABLE_VULKAN=OFF
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure
