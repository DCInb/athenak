#!/usr/bin/env bash
#SBATCH --job-name=dt-fusion-gpu
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=dt-fusion-gpu-%j.log
set -eo pipefail
repo_dir=${ATHENAK_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}
source "${ATHENAK_ENV:-${repo_dir}/../bashrc_athenak}"
set -u
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export UCX_TLS=self,sm,tcp,cuda_copy
build_dir=${ATHENAK_DT_BUILD:-${repo_dir}/build-dt-fusion-gpu}
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
export ATHENAK_DT_EXE="$build_dir/src/athena"
export ATHENAK_DT_MPI_LAUNCH='mpiexec --oversubscribe -n 2'
"$python_exe" -m pytest -q tst/test_suite/nr/test_nr_dt_fusion_cpu.py \
  --basetemp="$build_dir/test-${SLURM_JOB_ID}" --tb=short
