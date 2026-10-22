#!/usr/bin/env bash
#SBATCH --job-name=asym_gpu_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --time=00:15:00
#SBATCH --output=logs/gpu-scaling-%j.out

# Replay one immutable checkpoint on one and four GPUs within the same allocation.
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
case_dir=$(realpath -- "$case_dir")
source "${ATHENAK_ENV:-${case_dir}/../../bashrc_athenak}"
set -u
export ASYMMETRIC_CASE_DIR="$case_dir" BACKEND=cuda
export ATHENA_EXE=${ATHENA_EXE:-"${BUILD_DIR:-$case_dir/build/cuda}/src/athena"}
: "${CHECKPOINT:?Set CHECKPOINT to an existing complete restart file}"
checkpoint=$CHECKPOINT
checkpoint=$(realpath -- "$checkpoint")
[[ -f "$checkpoint" && "$checkpoint" != *.part ]]
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export UCX_TLS=self,sm,tcp,cuda_copy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bench_dir="${case_dir}/runs/gpu-scaling-${SLURM_JOB_ID}"
mkdir -p -- "$(dirname -- "$bench_dir")"
mkdir -- "$bench_dir"
cp -- "${case_dir}/scripts/gpu_scaling.slurm" "$bench_dir/submission.slurm"
cp -- "${case_dir}/scripts/analyze_gpu_scaling.py" "$bench_dir/analyze_gpu_scaling.py"
sha256sum "$checkpoint" "$ATHENA_EXE" > "$bench_dir/build.sha256"
printf 'Benchmark directory: %s\nCheckpoint: %s\n' "$bench_dir" "$checkpoint"
hostname
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv

# Verify that the four local ranks can address distinct allocated devices.
mpirun -n 4 /usr/bin/python3 -c '
import ctypes, json, os, socket
lib = ctypes.CDLL("libcudart.so.12")
count = ctypes.c_int()
assert lib.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0
rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
device = rank % count.value
bus = ctypes.create_string_buffer(32)
assert lib.cudaDeviceGetPCIBusId(bus, len(bus), device) == 0
print(json.dumps(dict(host=socket.gethostname(), rank=rank,
                     device=device, pci_bus=bus.value.decode())), flush=True)
' | tee "$bench_dir/gpu_mapping.jsonl"
/usr/bin/python3 - "$bench_dir/gpu_mapping.jsonl" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    mapping = [json.loads(line) for line in stream if line.startswith('{')]
assert sorted(row['rank'] for row in mapping) == [0, 1, 2, 3]
assert len({row['pci_bus'] for row in mapping}) == 4
assert len({row['host'] for row in mapping}) == 1
PY

start_cycle=$(/usr/bin/python3 - "$checkpoint" <<'PY'
from pathlib import Path
import struct, sys
with Path(sys.argv[1]).open('rb') as f:
    h = f.read(40960)
start = h.index(b'<par_end>')+10
print(struct.unpack_from('=i', h, start+8+72+152+16)[0])
PY
)
warmup_steps=2
measured_steps=10
# The last step is excluded because its cycle line follows final output writes.
stop_cycle=$((start_cycle+warmup_steps+measured_steps+1))
args=(-r "$checkpoint" "time/nlim=${stop_cycle}" time/ndiag=1
      thermal_radiation/implicit_report=false thermal_radiation/source_report=false
      output1/dt=1000 output2/dt=-1 output3/dt=-1 output4/dt=-1 output5/dt=-1
      output6/dt=1000)
printf 'Start cycle: %s; stop cycle: %s; timed cycles: %s through %s\n' \
  "$start_cycle" "$stop_cycle" "$((start_cycle+warmup_steps))" \
  "$((start_cycle+warmup_steps+measured_steps))"
# ABBA ordering reduces sensitivity to time-dependent node conditions.
for trial in 1gpu-a 4gpu-a 4gpu-b 1gpu-b; do
  ranks=${trial%%gpu-*}
  NRANKS="$ranks" RUN_DIR="${bench_dir}/${trial}" \
    bash "${case_dir}/scripts/run.slurm" "${args[@]}"
done
sha256sum --check "$bench_dir/build.sha256"
python_exe=${BENCHMARK_PYTHON:-python3}
"$python_exe" "$bench_dir/analyze_gpu_scaling.py" "$bench_dir" \
  --warmup-steps "$warmup_steps" --measured-steps "$measured_steps"
