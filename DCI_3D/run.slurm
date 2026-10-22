#!/bin/bash
#SBATCH --job-name=dci3d_amr_ch16
#SBATCH --partition=gpu
#SBATCH --qos=gpugpu
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=0

# AMR cell AMG: 84^3 base, CH at 1.786 um and Au at 3.571 um, 12^3 blocks,
# up to four-group batching, and 8 ns without a cycle limit. One A100 per rank.
# The gpu partition/gpugpu QoS permits unlimited wall time. When overriding Slurm
# --time, pass AthenaK -t HH:MM:SS with enough time left for its final checkpoint.
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

build_dir=${BUILD_DIR:-"${repo_dir}/build"}
executable=${ATHENA_EXE:-"${build_dir}/src/athena"}
input_file=${INPUT_FILE:-"${case_dir}/dci_3d.athinput"}
[[ -x "$executable" ]] || { printf 'Build first: %s\n' "$executable" >&2; exit 1; }
executable=$(realpath -e -- "$executable")
input_file=$(realpath -e -- "$input_file")
nranks=${SLURM_NTASKS:-16}
# One rank per GPU; see README.md for memory checks for the current input.
run_dir=${RUN_DIR:-"${case_dir}/runs/run-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}
mkdir -p -- "$(dirname -- "$run_dir")"
mkdir -- "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
cp -- "$executable" "$run_dir/athena"
cp -- "$input_file" "$run_dir/dci_3d.athinput"
cp -r -- "$case_dir/material_tables" "$run_dir/material_tables"
# Stage portable table paths without changing the physical or numerical settings.
sed -i -E 's@^(material[0-9]+_(eos|opacity)_table_file[[:space:]]*=[[:space:]]*).*/([^/[:space:]]+)[[:space:]]*$@\1material_tables/\3@' \
  "$run_dir/dci_3d.athinput"

# The site environment points TMPDIR at a network filesystem.  Open MPI's shared-memory
# backing files must instead live on node-local storage.
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TEMPDIR="${TMPDIR}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export OMPI_MCA_orte_tmpdir_base="${TMPDIR}"

# The qualified same-node transport list avoids the pathological cuda_ipc path while
# retaining CUDA-aware copies and shared-memory host transport.
export UCX_TLS=self,sm,tcp,cuda_copy
# Automatic TCP selection can choose unreachable auxiliary interfaces on this
# cluster. Match the routable interface used by the tested multi-node AMG launcher.
if (( ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}} > 1 )); then
  export UCX_NET_DEVICES=${ATHENA_UCX_NET_DEVICES:-${UCX_NET_DEVICES:-eth0}}
  export OMPI_MCA_oob_tcp_if_include=${OMPI_MCA_oob_tcp_if_include:-eth0}
  export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-eth0}
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any
export OMPI_MCA_hwloc_base_binding_policy=none
export ATHENA_REAL_EXE="$run_dir/athena"
# Reuse the mapper qualified by the four-GPU AMG scaling measurements.
mapper="$repo_dir/asymmetric_collision/scripts/map_gpu_rank.py"
[[ -f "$mapper" ]] || { printf 'Missing GPU/CPU mapper: %s\n' "$mapper" >&2; exit 1; }
cp -- "$mapper" "$run_dir/map_gpu_rank.py"

cd -- "$run_dir"
launch=(mpirun -n "$nranks" python3 ./map_gpu_rank.py --kokkos-map-device-id-by=mpi_rank
        -i dci_3d.athinput "$@")
{
  printf 'Run directory: %s\n' "$run_dir"
  printf 'Started: %s\nHost: %s\n' "$(date --iso-8601=seconds)" "$(hostname)"
  printf 'Slurm job: %s\nCUDA_VISIBLE_DEVICES: %s\n' \
    "${SLURM_JOB_ID:-none}" "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'UCX_TLS: %s\nUCX_NET_DEVICES: %s\n' "$UCX_TLS" "${UCX_NET_DEVICES:-all}"
  sha256sum athena dci_3d.athinput map_gpu_rank.py material_tables/*.2t_eos material_tables/*.opacity
  printf 'Command:'
  printf ' %q' "${launch[@]}"
  printf '\n'
  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
  "${launch[@]}"
} 2>&1 | tee run.log
