#!/usr/bin/env bash
#SBATCH --job-name=asym_collision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/run-%j.out

# Based on DCI_3D/run.slurm. Extra arguments are passed to AthenaK unchanged.
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
if [[ ! -f "${case_dir}/asymmetric_collision.cpp" ]]; then
  case_dir=${SLURM_SUBMIT_DIR:-$PWD}
  [[ -f "${case_dir}/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
fi
[[ -f "${case_dir}/asymmetric_collision.cpp" ]] || {
  printf 'Submit from asymmetric_collision, or set ASYMMETRIC_CASE_DIR.\n' >&2
  exit 1
}
case_dir=$(cd -- "$case_dir" && pwd)
repo_dir=$(cd -- "${case_dir}/.." && pwd)
site_env=${ATHENAK_ENV:-"${repo_dir}/../bashrc_athenak"}
if [[ -f "$site_env" ]]; then source "$site_env"; fi
set -u
backend=${BACKEND:-cuda}
case "$backend" in cuda|cpu) ;; *) printf 'Invalid BACKEND.\n' >&2; exit 1 ;; esac
build_dir=${BUILD_DIR:-"${case_dir}/build/${backend}"}
executable=${ATHENA_EXE:-"${build_dir}/src/athena"}
[[ -x "$executable" ]] || { printf 'Build first: %s\n' "$executable" >&2; exit 1; }
executable=$(realpath -- "$executable")
input_file=${INPUT_FILE:-"${case_dir}/inputs/asymmetric_collision.athinput"}
input_file=$(realpath -e -- "$input_file")
input_name=$(basename -- "$input_file")
run_dir=${RUN_DIR:-"${case_dir}/runs/run-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}
mkdir -p -- "$(dirname -- "$run_dir")"
# Refuse to overwrite an existing run, even when RUN_DIR is explicitly supplied.
mkdir -- "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
bash "${case_dir}/scripts/prepare_tables.sh" "${run_dir}/material_tables"
cp -- "$input_file" "${run_dir}/${input_name}"
cp -- "${case_dir}/material_tables.sha256" "${run_dir}/material_tables.sha256"

# Retain the working node-local MPI/UCX settings from DCI_3D/run.slurm.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
if [[ "$backend" == cuda ]]; then
  export UCX_TLS=${ATHENA_UCX_TLS:-self,sm,tcp,cuda_copy}
else
  export UCX_TLS=self,sm,tcp
  export OMPI_MCA_opal_warn_on_missing_libcuda=0
fi
nranks=${NRANKS:-${SLURM_NTASKS:-1}}
device_args=()
if [[ "$backend" == cuda ]]; then device_args+=(--kokkos-map-device-id-by=mpi_rank); fi
input_args=(-i "$input_name")
for argument in "$@"; do
  # A restart carries its own mesh, solver settings, and output counters. Do not
  # overwrite them with the default deck; explicit command-line overrides still work.
  if [[ "$argument" == -r || "$argument" == -i ]]; then input_args=(); fi
done
cd -- "$run_dir"
launch=(mpirun -n "$nranks" "$executable" "${device_args[@]}"
        "${input_args[@]}" "$@")
# Record command-line overrides and the node's driver in the same log as AthenaK.
# The copied input alone does not include overrides such as hydro/rsolver=hllc.
{
  printf 'Run directory: %s\n' "$run_dir"
  printf 'Started: %s\nHost: %s\n' "$(date --iso-8601=seconds)" "$(hostname)"
  printf 'Slurm job: %s\nAllocated nodes: %s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURM_JOB_NODELIST:-none}"
  printf 'Backend: %s\nCUDA_VISIBLE_DEVICES: %s\n' \
    "$backend" "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'OPENBLAS_NUM_THREADS: %s\nOMP_NUM_THREADS: %s\n' \
    "${OPENBLAS_NUM_THREADS:-unset}" "${OMP_NUM_THREADS:-unset}"
  printf 'Executable SHA-256: '
  sha256sum -- "$executable"
  printf 'Input SHA-256: '
  sha256sum -- "$input_name"
  printf 'Command:'
  printf ' %q' "${launch[@]}"
  printf '\n'
  if [[ "$backend" == cuda ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv || true
  fi
  "${launch[@]}"
} 2>&1 | tee run.log
