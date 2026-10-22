#!/usr/bin/env bash
#SBATCH --job-name=collision_amg_build
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amg-build-%j.out
set -euo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir" BACKEND=${BACKEND:-cuda}
export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/$BACKEND-amg"}
export HYPRE_ROOT=${HYPRE_ROOT:-"$case_dir/build/hypre-$BACKEND/install"}
if [[ ! -f "$HYPRE_ROOT/include/HYPRE.h" ]]; then
  bash "$case_dir/scripts/build_hypre.sh"
  if [[ "${CONFIGURE_ONLY:-0}" == 1 ]]; then
    printf 'Install HYPRE before configuring AthenaK; set HYPRE_ROOT to an existing installation.\n' >&2
    exit 1
  fi
fi
bash "$case_dir/scripts/make_amr.slurm" -DAthena_ENABLE_HYPRE=ON \
  "-DHYPRE_ROOT=$HYPRE_ROOT" "$@"
if [[ "${BUILD_SOLVER_TESTS:-0}" == 1 && "${CONFIGURE_ONLY:-0}" != 1 ]]; then
  set +u
  site_env=${ATHENAK_ENV:-"$case_dir/../../bashrc_athenak"}
  if [[ -f "$site_env" ]]; then source "$site_env"; fi
  set -u
  cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-6}" \
    --target thermal_radiation_hypre_driver thermal_radiation_hypre_mpi_driver
  "${SOLVER_TEST_PYTHON:-python3}" "$case_dir/scripts/build_builtin.py" "$BUILD_DIR"
fi
