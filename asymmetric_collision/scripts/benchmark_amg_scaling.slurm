#!/usr/bin/env bash
#SBATCH --job-name=amg_scaling
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amg-scaling-%j.out

# Short matched AMR replays on 1,2,3,4,4,3,2,1 GPUs in one allocation.
set -eo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
case_dir=$(realpath -e -- "$case_dir")
source "${ATHENAK_ENV:-$case_dir/../../bashrc_athenak}"
set -u
: "${CHECKPOINT:?Set CHECKPOINT to a complete, immutable AMR restart}"
checkpoint=$(realpath -e -- "$CHECKPOINT")
[[ "$checkpoint" != *.part ]]
[[ ${SLURM_JOB_NUM_NODES:-1} == 1 && ${SLURM_NTASKS:-4} == 4 ]] || {
  printf 'This comparison requires one node with four tasks and four GPUs.\n' >&2
  exit 1
}
export ASYMMETRIC_CASE_DIR="$case_dir" BACKEND=cuda
export ATHENA_REAL_EXE=${ATHENA_REAL_EXE:-"${BUILD_DIR:-$case_dir/build/cuda-amg}/src/athena"}
ATHENA_REAL_EXE=$(realpath -e -- "$ATHENA_REAL_EXE")
export ATHENA_EXE="$case_dir/scripts/map_gpu_rank.py"
export INPUT_FILE="$case_dir/inputs/asymmetric_collision_amr.athinput"
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_pml=ucx OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any
export ATHENA_UCX_TLS=${ATHENA_UCX_TLS:-self,sm,tcp,cuda_copy}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=${SLURM_TMPDIR:-/tmp} OMPI_MCA_orte_tmpdir_base=${SLURM_TMPDIR:-/tmp}
export BENCH_DIR=${BENCH_DIR:-"$case_dir/runs/amg-scaling-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}
export BENCH_STEPS=${BENCH_STEPS:-6} WARMUP_STEPS=${WARMUP_STEPS:-1}
export MAX_LEAVES=${MAX_LEAVES:-432} CHECKPOINT="$checkpoint"
python_exe=${BENCHMARK_PYTHON:-python3}
mkdir -p -- "$(dirname -- "$BENCH_DIR")"
mkdir -- "$BENCH_DIR"  # Refuse to overwrite any existing benchmark.
BENCH_DIR=$(realpath -- "$BENCH_DIR")
cp -- "$case_dir/scripts/benchmark_amg_scaling.slurm" "$BENCH_DIR/submission.slurm"
cp -- "$case_dir/scripts/analyze_amg_scaling.py" "$case_dir/scripts/compare_states.py" "$BENCH_DIR/"
sha256sum -- "$checkpoint" "$ATHENA_REAL_EXE" > "$BENCH_DIR/inputs.sha256"
hostname > "$BENCH_DIR/host.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv > "$BENCH_DIR/gpus.csv"
nvidia-smi topo -m > "$BENCH_DIR/topology.txt"
scontrol show job -d "$SLURM_JOB_ID" > "$BENCH_DIR/allocation.txt"

"$python_exe" - <<'PY'
import json, os, struct
from pathlib import Path
root, checkpoint = Path(os.environ['BENCH_DIR']), Path(os.environ['CHECKPOINT'])
with checkpoint.open('rb') as stream:
    parameters, section = {}, None
    while True:
        raw = stream.readline()
        assert raw, 'Missing restart parameter terminator'
        line = raw.decode('ascii').split('#', 1)[0].strip()
        if line == '<par_end>':
            break
        if line.startswith('<') and line.endswith('>'):
            section = line[1:-1]
            parameters.setdefault(section, {})
        elif '=' in line:
            key, value = line.split('=', 1)
            parameters[section][key.strip()] = value.strip()
    blocks, level = struct.unpack('=ii', stream.read(8))
    stream.read(72+152)
    start_time, dt, cycle = struct.unpack('=ddi', stream.read(20))
steps, warmup, capacity = (int(os.environ[k]) for k in ('BENCH_STEPS', 'WARMUP_STEPS', 'MAX_LEAVES'))
assert 0 <= warmup < steps and capacity >= blocks
assert parameters['thermal_radiation']['implicit_preconditioner'] == 'amr-amg'
assert parameters['mesh_refinement']['refinement'] == 'adaptive'
outputs = [key for key in parameters if key.startswith('output')]
restarts = [key for key in outputs if parameters[key].get('file_type') == 'rst']
assert len(restarts) == 1, 'Exactly one restart output block is required'
stop = cycle+steps+1
assert stop < 2147483647
order = [(n, 'a') for n in range(1, 5)]+[(n, 'b') for n in range(4, 0, -1)]
for ranks in range(1, 5):
    # Use the current default scheduling: scalar on one GPU, B4 on multiple GPUs.
    overlay = [f'<time>\nnlim = {stop}\nndiag = 1',
               '<thermal_radiation>\nimplicit_report = false\nsource_report = false\n'
               f'solver_timers = false\nimplicit_group_batch_size = {1 if ranks == 1 else 4}',
               f'<mesh_refinement>\nmax_nmb_per_rank = {(capacity+ranks-1)//ranks}']
    for name in outputs:
        # Retain one final restart, with no scheduled outputs in the measured window.
        overlay.append(f'<{name}>\ndt = -1\ndcycle = {2147483647 if name in restarts else 0}')
    (root/f'{ranks}gpu.athinput').write_text('\n\n'.join(overlay)+'\n')
config = dict(checkpoint=str(checkpoint), executable=os.environ['ATHENA_REAL_EXE'],
              job_id=os.environ.get('SLURM_JOB_ID'), start_cycle=cycle,
              start_time=start_time, initial_dt=dt, initial_leaves=blocks,
              stop_cycle=stop, timed_steps=steps, warmup_steps=warmup,
              state_time_relative_tolerance=1e-12,
              max_leaves=capacity, checkpoint_parameters=parameters,
              trials=[f'{n}gpu-{repeat}' for n, repeat in order],
              batching={'1': 1, '2': 4, '3': 4, '4': 4})
(root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
print(json.dumps({k: config[k] for k in ('start_cycle', 'stop_cycle', 'timed_steps',
                                       'warmup_steps', 'initial_leaves', 'trials')}), flush=True)
PY

printf 'Benchmark directory: %s\n' "$BENCH_DIR"
for trial in 1gpu-a 2gpu-a 3gpu-a 4gpu-a 4gpu-b 3gpu-b 2gpu-b 1gpu-b; do
  ranks=${trial%%gpu-*}
  printf 'Starting trial %s at %s\n' "$trial" "$(date --iso-8601=seconds)"
  SECONDS=0
  NRANKS="$ranks" RUN_DIR="$BENCH_DIR/$trial" \
    bash "$case_dir/scripts/run.slurm" -r "$checkpoint" -i "$BENCH_DIR/${ranks}gpu.athinput"
  printf '%s\n' "$SECONDS" > "$BENCH_DIR/$trial/launch_wall_seconds.txt"
done
sha256sum --check "$BENCH_DIR/inputs.sha256"
"$python_exe" "$BENCH_DIR/analyze_amg_scaling.py" "$BENCH_DIR"
