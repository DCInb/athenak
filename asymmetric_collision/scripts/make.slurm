#!/usr/bin/env bash
#SBATCH --job-name=collision_make
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/build-%j.out

# Based on DCI_3D/make.slurm. Submit from asymmetric_collision, or run with bash.
# BACKEND=cpu bash make.slurm builds the same problem with MPI and serial Kokkos.
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
# Compiler temporaries are short-lived and should stay on node-local scratch.
# Inherited login-node TMPDIR settings can otherwise send every nvcc stage to NFS.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR"
backend=${BACKEND:-cuda}
build_dir=${BUILD_DIR:-"${case_dir}/build/${backend}"}
options=(-DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
         -DAthena_ENABLE_CHOLMOD=OFF -DAthena_ENABLE_HYPRE=OFF
         -DAthena_ENABLE_CUSPARSE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
         "-DCMAKE_C_COMPILER=$(command -v gcc)"
         "-DMPI_C_COMPILER=$(command -v mpicc)"
         "-DMPI_CXX_COMPILER=$(command -v mpicxx)"
         -DPROBLEM=../../asymmetric_collision/asymmetric_collision)
case "$backend" in
  cuda)
    options+=(-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
              "-DCMAKE_CXX_COMPILER=${repo_dir}/kokkos/bin/nvcc_wrapper") ;;
  cpu)
    options+=(-DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_SERIAL=ON
              "-DCMAKE_CXX_COMPILER=$(command -v g++)") ;;
  *) printf 'BACKEND must be cuda or cpu.\n' >&2; exit 1 ;;
esac
bash "${case_dir}/scripts/prepare_tables.sh"
cmake -S "$repo_dir" -B "$build_dir" "${options[@]}" "$@"
if [[ "${CONFIGURE_ONLY:-0}" == 1 ]]; then
  printf 'Configured %s\n' "$build_dir"
  exit 0
fi
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
printf 'Built %s/src/athena\n' "$build_dir"
