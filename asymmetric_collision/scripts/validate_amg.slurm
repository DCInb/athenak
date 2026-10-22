#!/usr/bin/env bash
#SBATCH --job-name=collision_amg_test
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amg-test-%j.out
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
repo_dir=$(cd -- "$case_dir/.." && pwd)
site_env=${ATHENAK_ENV:-"$repo_dir/../bashrc_athenak"}
if [[ -f "$site_env" ]]; then source "$site_env"; fi
set -u
if (( ${SLURM_NTASKS:-0} < 4 )); then
  printf 'Validation needs an allocation with at least four MPI ranks and one GPU per rank.\n' >&2
  exit 1
fi
build_dir=${BUILD_DIR:-"$case_dir/build/cuda-amg"}
export ATHENA_AMG_TEST_EXE=${ATHENA_AMG_TEST_EXE:-"$build_dir/validation/athena-builtin"}
export ATHENA_AMR_TEST_EXE="$ATHENA_AMG_TEST_EXE"
export ATHENA_AMG_MPI_DRIVER=${ATHENA_AMG_MPI_DRIVER:-"$build_dir/thermal_radiation_hypre_mpi_driver"}
for executable in "$ATHENA_AMG_TEST_EXE" "$ATHENA_AMG_MPI_DRIVER"; do
  [[ -x "$executable" ]] || { printf 'Build first with BUILD_SOLVER_TESTS=1: %s\n' "$executable" >&2; exit 1; }
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export UCX_TLS=${ATHENA_UCX_TLS:-self,sm,tcp,cuda_copy}
if (( ${SLURM_JOB_NUM_NODES:-1} > 1 )) && [[ ",$UCX_TLS," == *,tcp,* ]]; then
  export UCX_NET_DEVICES=${ATHENA_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-eth0}}
fi
export OMPI_MCA_pml=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any
export OMPI_MCA_hwloc_base_binding_policy=none
export TMPDIR=${SLURM_TMPDIR:-/tmp} OMPI_MCA_orte_tmpdir_base=${SLURM_TMPDIR:-/tmp}
export PYTHONPATH="$repo_dir/tst${PYTHONPATH:+:$PYTHONPATH}"
python=${SOLVER_TEST_PYTHON:-python3}
run_dir=${RUN_DIR:-"$case_dir/diagnostics/amg-multigpu-${SLURM_JOB_ID}"}
mkdir -p -- "$(dirname -- "$run_dir")"
mkdir -- "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
cd -- "$run_dir"
# The optional ATHENA_GROUP_BATCH_TEST_TOTAL_MIB enables memory-fallback tests;
# set it to the physical capacity of the smallest allocated CUDA device.
"$python" -m pytest -q --basetemp="$run_dir/pytest" -o cache_dir="$run_dir/cache" \
  "$repo_dir/tst/test_suite/nr/test_nr_thermal_radiation_group_batch_mpicpu.py" \
  "$repo_dir/tst/test_suite/nr/test_nr_thermal_radiation_halo_mpicpu.py" \
  "$repo_dir/tst/test_suite/nr/test_nr_thermal_radiation_amg_mpicpu.py" \
  "$repo_dir/tst/test_suite/nr/test_nr_thermal_radiation_amg_cpu.py"
mpirun -n 4 "$ATHENA_AMG_MPI_DRIVER" coarse-oracle
