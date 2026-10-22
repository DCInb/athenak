#!/usr/bin/env bash
# Rebuild the pinned HYPRE dependency without relying on an old experiment directory.
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
repo_dir=$(cd -- "$case_dir/.." && pwd)
site_env=${ATHENAK_ENV:-"$repo_dir/../bashrc_athenak"}
if [[ -f "$site_env" ]]; then source "$site_env"; fi
set -u
backend=${BACKEND:-cuda}
case "$backend" in cuda|cpu) ;; *) printf 'BACKEND must be cuda or cpu.\n' >&2; exit 1 ;; esac
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR"
dependency_dir=${HYPRE_BUILD_ROOT:-"$case_dir/build/hypre-$backend"}
install_dir=${HYPRE_ROOT:-"$dependency_dir/install"}
source_dir=${HYPRE_SOURCE_DIR:-"$dependency_dir/source"}
if [[ ! -f "$source_dir/src/CMakeLists.txt" ]]; then
  if [[ -n "${HYPRE_SOURCE_DIR:-}" ]]; then
    printf 'HYPRE_SOURCE_DIR must contain src/CMakeLists.txt: %s\n' "$source_dir" >&2
    exit 1
  fi
  mkdir -p -- "$dependency_dir"
  archive=${HYPRE_ARCHIVE:-"$dependency_dir/hypre-v2.33.0.tar.gz"}
  if [[ ! -f "$archive" ]]; then
    if [[ -n "${HYPRE_ARCHIVE:-}" ]]; then
      printf 'HYPRE_ARCHIVE does not exist: %s\n' "$archive" >&2
      exit 1
    fi
    curl --fail --location --retry 3 \
      https://codeload.github.com/hypre-space/hypre/tar.gz/refs/tags/v2.33.0 \
      --output "$archive.part"
    mv -- "$archive.part" "$archive"
  fi
  expected=0f9103c34bce7a5dcbdb79a502720fc8aab4db9fd0146e0791cde7ec878f27da
  actual=$(sha256sum -- "$archive")
  [[ "${actual%% *}" == "$expected" ]] || {
    printf 'HYPRE 2.33.0 archive checksum mismatch: %s\n' "$archive" >&2
    exit 1
  }
  mkdir -p -- "$source_dir"
  tar -xzf "$archive" --strip-components=1 -C "$source_dir"
fi
mpi_include_dir=${MPI_INCLUDE_DIR:-"$(dirname -- "$(dirname -- "$(command -v mpicc)")")/include"}
[[ -f "$mpi_include_dir/mpi.h" ]] || {
  printf 'Set MPI_INCLUDE_DIR to the directory containing mpi.h.\n' >&2
  exit 1
}
options=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$install_dir"
         "-DCMAKE_C_COMPILER=$(command -v mpicc)"
         "-DCMAKE_CXX_COMPILER=$(command -v mpicxx)"
         "-DMPI_CXX_COMPILER_INCLUDE_DIR=$mpi_include_dir"
         -DHYPRE_ENABLE_MPI=ON -DHYPRE_ENABLE_OPENMP=OFF
         -DHYPRE_ENABLE_UNIFIED_MEMORY=OFF -DHYPRE_ENABLE_SINGLE=OFF
         -DHYPRE_BUILD_TESTS=OFF -DHYPRE_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF)
if [[ "$backend" == cuda ]]; then
  # Match the dependency used for the validated multi-GPU and capacity runs.
  # These are HYPRE-internal switches; AthenaK's CUDA-aware MPI remains enabled.
  options+=(-DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_GPU_AWARE_MPI=OFF
            -DHYPRE_ENABLE_DEVICE_POOL=OFF -DHYPRE_ENABLE_THRUST_ASYNC=OFF
            "-DCMAKE_CUDA_COMPILER=$(command -v nvcc)"
            "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-80}")
else
  options+=(-DHYPRE_ENABLE_CUDA=OFF -DHYPRE_ENABLE_GPU_AWARE_MPI=OFF)
fi
cmake -S "$source_dir/src" -B "$dependency_dir/build" "${options[@]}" "$@"
if [[ "${CONFIGURE_ONLY:-0}" == 1 ]]; then exit 0; fi
cmake --build "$dependency_dir/build" --parallel "${BUILD_JOBS:-6}"
cmake --install "$dependency_dir/build"
printf 'Installed HYPRE: %s\n' "$install_dir"
