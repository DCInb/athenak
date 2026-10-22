#!/usr/bin/env bash
#SBATCH --job-name=collision_amg_2gpu
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amg-2gpu-run-%j.out
set -euo pipefail
# Preserve the established two-GPU entry point; share the general AMG launcher.
if [[ "${SLURM_NTASKS:-2}" != 2 || "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
  printf 'This launcher requires two MPI ranks on one node with two GPUs. Use run_amg.slurm for other allocations.\n' >&2
  exit 1
fi
export NRANKS=2
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir"
bash "$case_dir/scripts/run_amg.slurm" "$@"
