#!/usr/bin/env bash
#SBATCH --job-name=corona_check
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/corona-validation-%j.out
set -euo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir"
export BUILD_DIR=${BUILD_DIR:-"${case_dir}/build/cuda"}
BACKEND=cuda bash "${case_dir}/scripts/make.slurm"
export RUN_DIR="${case_dir}/runs/corona-validation-${SLURM_JOB_ID}"
BACKEND=cuda bash "${case_dir}/scripts/run.slurm" time/nlim=5 time/ndiag=1 \
  thermal_radiation/implicit_report=true thermal_radiation/source_report=true
python_exe=${SOLVER_TEST_PYTHON:-python3}
"$python_exe" "${case_dir}/scripts/verify_initial.py" "$RUN_DIR" \
  > "${RUN_DIR}/initial_validation.json"
sha256sum "${case_dir}/asymmetric_collision.cpp" "${BUILD_DIR}/src/athena" \
  > "${RUN_DIR}/build.sha256"
