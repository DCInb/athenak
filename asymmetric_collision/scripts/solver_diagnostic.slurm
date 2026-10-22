#!/usr/bin/env bash
#SBATCH --job-name=asym_solver_diag
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:05:00
#SBATCH --output=logs/solver-diagnostic-%j.out

# Read-only checkpoint replay. Neither the production deck nor executable is edited.
# Submit from asymmetric_collision with CHECKPOINT and absolute STOP_CYCLE set.
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
case_dir=$(realpath -- "$case_dir")
source "${ATHENAK_ENV:-${case_dir}/../../bashrc_athenak}"
set -u
: "${CHECKPOINT:?Set CHECKPOINT to a complete, immutable restart file}"
: "${STOP_CYCLE:?Set STOP_CYCLE to the checkpoint cycle plus the replay steps}"
[[ "$STOP_CYCLE" =~ ^[0-9]+$ ]]
CHECKPOINT=$(realpath -- "$CHECKPOINT")
[[ -f "$CHECKPOINT" && "$CHECKPOINT" != *.part ]]
export ASYMMETRIC_CASE_DIR="$case_dir"
export ATHENA_REAL_EXE=${ATHENA_REAL_EXE:-"${BUILD_DIR:-$case_dir/build/cuda-amg}/src/athena"}
export NRANKS=1 BACKEND=cuda
diagnostic_dir="${case_dir}/runs/solver-diagnostic-${SLURM_JOB_ID}"
mkdir -p -- "$(dirname -- "$diagnostic_dir")"
mkdir -- "$diagnostic_dir"
printf 'Diagnostic directory: %s\nCheckpoint: %s\nStop cycle: %s\n' \
  "$diagnostic_dir" "$CHECKPOINT" "$STOP_CYCLE"
date --iso-8601=seconds
hostname
stat --format='Checkpoint size=%s bytes modified=%y' "$CHECKPOINT"
sha256sum "$ATHENA_REAL_EXE"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv

args=(-r "$CHECKPOINT" "time/nlim=${STOP_CYCLE}" time/ndiag=1)
# Disable all regular/final output objects, keeping replay files separate and small.
# The production restart has time-based output only. AthenaK CLI overrides must
# name existing parameters, so do not introduce a new dcycle parameter here.
for output in {1..6}; do args+=("output${output}/dt=-1"); done

printf '\n--- Unprofiled replay, production reporting settings ---\n'
RUN_DIR="${diagnostic_dir}/quiet" ATHENA_EXE="$ATHENA_REAL_EXE" \
  bash "${case_dir}/scripts/run.slurm" "${args[@]}" \
  thermal_radiation/implicit_report=false thermal_radiation/source_report=false

printf '\n--- Unprofiled replay, solver reports enabled ---\n'
RUN_DIR="${diagnostic_dir}/reports" ATHENA_EXE="$ATHENA_REAL_EXE" \
  bash "${case_dir}/scripts/run.slurm" "${args[@]}" \
  thermal_radiation/implicit_report=true thermal_radiation/source_report=true

printf '\n--- CUDA-profiled replay, solver reports enabled ---\n'
export NSYS_OUTPUT="${diagnostic_dir}/solver_trace"
RUN_DIR="${diagnostic_dir}/profiled" \
  ATHENA_EXE="${case_dir}/scripts/solver_profile_exec.sh" \
  bash "${case_dir}/scripts/run.slurm" "${args[@]}" \
  thermal_radiation/implicit_report=true thermal_radiation/source_report=true

printf '\nDiagnostic replay complete: %s\n' "$diagnostic_dir"
