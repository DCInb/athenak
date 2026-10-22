#!/usr/bin/env bash
#SBATCH --job-name=collision_amg
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amg-run-%j.out
set -euo pipefail

case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir" BACKEND=cuda
export NRANKS=${NRANKS:-${SLURM_NTASKS:-4}}
if [[ ! "$NRANKS" =~ ^[1-9][0-9]*$ || "$NRANKS" != "${SLURM_NTASKS:-$NRANKS}" ]]; then
  printf 'NRANKS must be positive and match the Slurm task allocation.\n' >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
export OMPI_MCA_pml_ucx_tls=${OMPI_MCA_pml_ucx_tls:-any}
export OMPI_MCA_pml_ucx_devices=${OMPI_MCA_pml_ucx_devices:-any}
# Staged GPU transfers work without peer access. ATHENA_UCX_TLS can select a
# site-validated inter-node transport when using more than one node.
export ATHENA_UCX_TLS=${ATHENA_UCX_TLS:-self,sm,tcp,cuda_copy}
# UCX's automatic TCP selection can choose unreachable auxiliary interfaces on
# this cluster. Use the routable interface for inter-node traffic; honor site overrides.
if (( ${SLURM_JOB_NUM_NODES:-1} > 1 )) && [[ ",$ATHENA_UCX_TLS," == *,tcp,* ]]; then
  export UCX_NET_DEVICES=${ATHENA_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-eth0}}
fi
export BUILD_DIR=${BUILD_DIR:-"$case_dir/build/cuda-amg"}
if [[ -z "${ATHENA_EXE:-}" ]]; then
  export ATHENA_REAL_EXE=${ATHENA_REAL_EXE:-"$BUILD_DIR/src/athena"}
  if [[ "${ATHENA_PIN_GPU_LOCAL:-1}" == 1 ]]; then
    export ATHENA_EXE="$case_dir/scripts/map_gpu_rank.py"
    export OMPI_MCA_hwloc_base_binding_policy=none
  else
    export ATHENA_EXE="$ATHENA_REAL_EXE"
  fi
fi
# Only the unchanged, implicit default deck has the legacy 432-leaf capacity.
# A selected input or restart owns its per-rank capacity unless explicitly overridden.
use_default_capacity=1
use_default_solver=1
if [[ -n "${INPUT_FILE:-}" ]]; then
  use_default_capacity=0
  use_default_solver=0
fi
for argument in "$@"; do
  case "$argument" in
    -i|-r) use_default_capacity=0; use_default_solver=0 ;;
    mesh/*|meshblock/*|mesh_refinement/num_levels=*) use_default_capacity=0 ;;
  esac
done
export INPUT_FILE=${INPUT_FILE:-"$case_dir/inputs/asymmetric_collision_amr.athinput"}
export RUN_DIR=${RUN_DIR:-"$case_dir/runs/amg-${NRANKS}gpu-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}

# Select aggregate AMG only for the implicit original input. Explicit inputs
# and restarts own their solver and batching settings (including cell-AMG).
solver_args=()
if [[ "$use_default_solver" == 1 ]]; then
  solver_args+=(thermal_radiation/implicit_preconditioner=amr-amg)
fi
capacity=${MAX_NMB_PER_RANK:-}
if [[ -z "$capacity" && "$use_default_capacity" == 1 ]]; then
  capacity=$(((432+NRANKS-1)/NRANKS))
fi
if [[ -n "$capacity" ]]; then
  if [[ ! "$capacity" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_NMB_PER_RANK must be a positive integer.\n' >&2
    exit 1
  fi
  solver_args+=("mesh_refinement/max_nmb_per_rank=$capacity")
fi
walltime_args=(-t "${ATHENA_WALLTIME:-23:45:00}")
for argument in "$@"; do
  if [[ "$argument" == -t ]]; then walltime_args=(); fi
done
bash "$case_dir/scripts/run.slurm" "${walltime_args[@]}" "${solver_args[@]}" "$@"
