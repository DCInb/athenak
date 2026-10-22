#!/usr/bin/env bash
#SBATCH --job-name=dt-efficiency
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=dt-efficiency-%j.log
set -eo pipefail
repo_dir=${ATHENAK_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}
source "${ATHENAK_ENV:-${repo_dir}/../bashrc_athenak}"
set -u
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export UCX_TLS=${ATHENAK_DT_UCX_TLS:-self,sm,tcp,cuda_copy}
export OMP_NUM_THREADS=1
cpu_exe=${ATHENAK_DT_CPU_EXE:-${repo_dir}/tst/build/src/athena}
gpu_exe=${ATHENAK_DT_GPU_EXE:-${repo_dir}/build-fast-electrons-gpu/src/athena}
python_exe=${ATHENAK_TEST_PYTHON:-/home/bingxing2/home/scx9fy6/anaconda3/bin/python}
cd "$repo_dir"
"$python_exe" tst/benchmarks/dt_fusion_benchmark.py \
  --cpu-exe "$cpu_exe" --gpu-exe "$gpu_exe" \
  --output "$repo_dir/build-dt-efficiency/job-${SLURM_JOB_ID}" "$@"
