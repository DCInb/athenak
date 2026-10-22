#!/bin/bash
#SBATCH --job-name=dci3d_amg_make
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00

# Submit from DCI_3D, the repository root, or the existing build directory.
set -eo pipefail
case_dir=${DCI_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}
if [[ -f "$case_dir/DCI_3D/dci_3d.cpp" ]]; then case_dir+=/DCI_3D; fi
if [[ ! -f "$case_dir/dci_3d.cpp" && -f "$case_dir/../DCI_3D/dci_3d.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/../DCI_3D" && pwd)
fi
[[ -f "$case_dir/dci_3d.cpp" ]] || { printf 'Cannot locate DCI_3D.\n' >&2; exit 1; }
case_dir=$(cd -- "$case_dir" && pwd)
repo_dir=$(cd -- "$case_dir/.." && pwd)
source "${ATHENAK_ENV:-${repo_dir}/../bashrc_athenak}"
set -u

# Keep compiler temporaries on the compute node rather than the shared filesystem.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
build_dir=${BUILD_DIR:-"${repo_dir}/build"}
hypre_root=${HYPRE_ROOT:-"${repo_dir}/asymmetric_collision/build/hypre-cuda/install"}
cholmod_root=${CHOLMOD_ROOT:-/home/bingxing2/apps/suitesparse/usr}
[[ -f "$hypre_root/include/HYPRE.h" ]] || {
  printf 'Set HYPRE_ROOT to the tested CUDA/MPI HYPRE installation.\n' >&2
  exit 1
}
configure_args=()
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
  cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
  cached_cc=$(sed -n 's/^CMAKE_C_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
  # CMake's automatic compiler-change reset discards explicit CUDA/MPI options.
  if [[ "$cached_cxx" != "$repo_dir/kokkos/bin/nvcc_wrapper" ||
        "$cached_cc" != "$(command -v gcc)" ]]; then
    configure_args+=(--fresh)
  fi
fi

cmake "${configure_args[@]}" -S "$repo_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$repo_dir/kokkos/bin/nvcc_wrapper" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_CHOLMOD=ON -DAthena_ENABLE_HYPRE=ON \
  -DAthena_ENABLE_CUSPARSE=ON \
  -DHYPRE_ROOT="$hypre_root" -DCHOLMOD_ROOT="$cholmod_root" \
  -DCHOLMOD_USE_STATIC_LIBS=ON -DBLA_VENDOR=OpenBLAS \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
  -DPROBLEM=../../DCI_3D/dci_3d "$@"
for expected in 'Athena_ENABLE_MPI:BOOL=ON' 'Kokkos_ENABLE_CUDA:BOOL=ON' \
                'Athena_ENABLE_CHOLMOD:BOOL=ON' 'Athena_ENABLE_HYPRE:BOOL=ON' \
                'Athena_ENABLE_CUSPARSE:BOOL=ON' \
                'Kokkos_ARCH_AMPERE80:BOOL=ON' 'PROBLEM:STRING=../../DCI_3D/dci_3d'; do
  if ! grep -q -F -x -- "$expected" "$build_dir/CMakeCache.txt"; then
    printf 'Incorrect build configuration: expected %s\n' "$expected" >&2
    exit 1
  fi
done
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-16}}" \
  --target thermal_radiation_hypre_driver thermal_radiation_hypre_mpi_driver
"$build_dir/src/athena" -c
"$build_dir/thermal_radiation_hypre_driver"
sha256sum "$build_dir/src/athena"
