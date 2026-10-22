#!/usr/bin/env bash
#SBATCH --job-name=sc_4gpu
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --chdir=/home/bingxing2/home/scx9fy6/mengqi/athenak/asymmetric_collision
#SBATCH --output=logs/sc-4gpu-%j.out

# Submit with sbatch from any directory. Extra arguments go directly to AthenaK,
# e.g. time/nlim=13 for a short run or -r /absolute/path/checkpoint.rst to resume.
set -euo pipefail


if [[ -n "${ASYMMETRIC_CASE_DIR:-}" ]]; then
  case_dir=$ASYMMETRIC_CASE_DIR
elif [[ -f "$PWD/asymmetric_collision.cpp" ]]; then
  # sbatch executes a spool copy; --chdir above identifies the case reliably.
  case_dir=$PWD
else
  # Also support bash scripts/slurms/run_sc_4gpu.sh inside an allocation.
  case_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || {
  printf 'Cannot locate the case; set ASYMMETRIC_CASE_DIR.\n' >&2
  exit 1
}
case_dir=$(cd -- "$case_dir" && pwd)
export ASYMMETRIC_CASE_DIR="$case_dir" NRANKS=4
export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cuda-amg"}
export ATHENA_REAL_EXE=${ATHENA_REAL_EXE:-"$BUILD_DIR/src/athena"}
export INPUT_FILE=${INPUT_FILE:-"$case_dir/inputs/ac_amr_4gpu.athinput"}
export RUN_DIR=${RUN_DIR:-"$case_dir/runs/sc-4gpu-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}

if [[ -z "${ATHENA_EXE:-}" && ! -x "$ATHENA_REAL_EXE" ]]; then
  printf 'Missing executable: %s\nBuild with: sbatch scripts/make_amg.slurm (from %s)\n' \
    "$ATHENA_REAL_EXE" "$case_dir" >&2
  exit 1
fi

# Share GPU-local CPU placement, one GPU per rank, staged UCX transfers, restart
# forwarding and the 23:45 application checkpoint limit with the AMG launcher.
# INPUT_FILE makes its 128-block capacity authoritative; no legacy 108 override.
exec bash "$case_dir/scripts/run_amg.slurm" "$@"
