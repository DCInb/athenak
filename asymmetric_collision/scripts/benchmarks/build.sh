#!/usr/bin/env bash
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
site_env=${ATHENAK_ENV:-"$case_dir/../../bashrc_athenak"}
if [[ -f "$site_env" ]]; then source "$site_env"; fi
set -u
output=${BENCHMARK_BUILD_DIR:-"$case_dir/build/benchmarks"}
cuda_root=${CUDAToolkit_ROOT:-$(dirname -- "$(dirname -- "$(command -v nvcc)")")}
mkdir -p -- "$output"
mpicxx -O3 -std=c++17 -I"$cuda_root/include" \
  "$case_dir/scripts/benchmarks/mpi_cuda_bench.cpp" \
  -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart \
  -o "$output/mpi_cuda_bench"
mpicc -O2 -fPIC -shared -Wall -Wextra \
  "$case_dir/scripts/benchmarks/mpi_counts.c" -o "$output/libmpi_counts.so"
printf 'Built communication tools in %s\n' "$output"
