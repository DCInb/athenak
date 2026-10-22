#!/usr/bin/env bash
#SBATCH --job-name=collision_amr
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amr-run-%j.out
set -euo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir"
export BACKEND=${BACKEND:-cuda}
export NRANKS=${NRANKS:-1}
# The sparse coarse solve is small; avoid starting a BLAS team for every solve.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ "$NRANKS" == 1 && "${SLURM_NTASKS:-1}" == 1 ]] || {
  printf 'This 40 GiB AMR memory budget requires exactly one MPI rank.\n' >&2
  exit 1
}
if [[ "$BACKEND" == cuda ]]; then
  export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cuda-amr"}
else
  export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cpu-amr"}
fi
export INPUT_FILE=${INPUT_FILE:-"$case_dir/inputs/asymmetric_collision_amr.athinput"}
export RUN_DIR=${RUN_DIR:-"$case_dir/runs/amr-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}
# Leave time for an atomic checkpoint before the 24-hour Slurm limit.
walltime_args=(-t "${ATHENA_WALLTIME:-23:45:00}")
for argument in "$@"; do
  if [[ "$argument" == -t ]]; then walltime_args=(); fi
done
bash "$case_dir/scripts/run.slurm" "${walltime_args[@]}" "$@"
