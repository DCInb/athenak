#!/usr/bin/env bash
#SBATCH --job-name=collision_amr_build
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amr-build-%j.out
set -euo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir"
export BACKEND=${BACKEND:-cuda}
# Keep uniform and AMR build configurations in separate directories.
if [[ "$BACKEND" == cuda ]]; then
  export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cuda-amr"}
else
  export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cpu-amr"}
fi
# CHOLMOD factors the coarse matrix; CUDA builds also enable GPU corrections.
# The site installation needs its
# static CHOLMOD/ordering libraries; CMake checks the complete link before building.
options=(-DAthena_ENABLE_CHOLMOD=ON
         "-DCHOLMOD_ROOT=${CHOLMOD_ROOT:-/home/bingxing2/apps/suitesparse/usr}"
         "-DCHOLMOD_USE_STATIC_LIBS=${CHOLMOD_USE_STATIC_LIBS:-ON}"
         "-DBLA_VENDOR=${BLA_VENDOR:-OpenBLAS}")
if [[ "$BACKEND" == cuda ]]; then options+=(-DAthena_ENABLE_CUSPARSE=ON); fi
bash "$case_dir/scripts/make.slurm" "${options[@]}" "$@"
