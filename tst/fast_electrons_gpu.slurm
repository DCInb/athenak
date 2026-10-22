#!/usr/bin/env bash
#SBATCH --job-name=fast-electrons-gpu
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=build-fast-electrons-gpu/job-%j.log

set -eo pipefail
repo_dir=${ATHENAK_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}
source "${ATHENAK_ENV:-${repo_dir}/../bashrc_athenak}"
set -u
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="$TMPDIR" TEMP="$TMPDIR" TMP="$TMPDIR"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export UCX_TLS=self,sm,tcp,cuda_copy
build_dir="${repo_dir}/build-fast-electrons-gpu"
python_exe=${ATHENAK_TEST_PYTHON:-/home/bingxing2/home/scx9fy6/anaconda3/bin/python}
cd "$repo_dir"
nvidia-smi
cmake -S "$repo_dir" -B "$build_dir" \
  -DCMAKE_CXX_COMPILER="$repo_dir/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
  -DAthena_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel 16
export ATHENAK_FAST_ELECTRONS_EXE="$build_dir/src/athena"
"$python_exe" -m pytest -q \
  tst/test_suite/nr/test_nr_fast_electrons_cpu.py \
  tst/test_suite/nr/test_nr_fast_electron_physics_cpu.py \
  tst/test_suite/nr/test_nr_fast_electron_hybrid_cpu.py \
  --basetemp="$build_dir/test-${SLURM_JOB_ID}" --tb=short
export ATHENAK_FAST_ELECTRONS_MPI_LAUNCH='mpiexec --oversubscribe -n 2'
"$python_exe" -m pytest -q tst/test_suite/nr/test_nr_fast_electrons_cpu.py \
 tst/test_suite/nr/test_nr_fast_electron_hybrid_cpu.py \
 -k 'mpi_packet or mpi_hybrid' --basetemp="$build_dir/mpi-test-${SLURM_JOB_ID}" --tb=short
sha256sum "$build_dir/src/athena"
