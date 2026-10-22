#!/usr/bin/env bash
#SBATCH --job-name=collision_amr_check
#SBATCH --partition=gpu
#SBATCH --account=scx9fy6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-4
#SBATCH --output=logs/amr-check-%j.out
set -euo pipefail
case_dir=${ASYMMETRIC_CASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$case_dir/asymmetric_collision.cpp" && -f "$case_dir/../asymmetric_collision.cpp" ]]; then
  case_dir=$(cd -- "$case_dir/.." && pwd)
fi
[[ -f "$case_dir/asymmetric_collision.cpp" ]] || case_dir+=/asymmetric_collision
export ASYMMETRIC_CASE_DIR="$case_dir" BACKEND=cuda NRANKS=1
export ATHENA_WALLTIME=${ATHENA_WALLTIME:-01:50:00}
test_dir=${AMR_TEST_DIR:-"$case_dir/runs/amr-validation-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)-$$}"}
mkdir -p -- "$(dirname -- "$test_dir")"
mkdir -- "$test_dir"
cp -- "${INPUT_FILE:-$case_dir/inputs/asymmetric_collision_amr.athinput}" \
  "$test_dir/asymmetric_collision_amr.athinput"
export INPUT_FILE="$test_dir/asymmetric_collision_amr.athinput"
executable=${ATHENA_EXE:-"${BUILD_DIR:-$case_dir/build/cuda-amr}/src/athena"}
cp -- "$executable" "$test_dir/athena"
export ATHENA_EXE="$test_dir/athena"
python_exe=${SOLVER_TEST_PYTHON:-"python3"}
# Keep intermediate checkpoints for diagnosing a later stiff solve without
# repeating startup. Cycle-based output leaves timestep selection unchanged.
"$python_exe" - "$INPUT_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
deck = path.read_text()
if deck.count("<output6>") != 1:
    raise ValueError("expected one output6 restart block")
path.write_text(deck.replace("<output6>", "<output6>\ndcycle = 0", 1))
PY
evolve_cycles=${AMR_EVOLVE_CYCLES:-25}
[[ "$evolve_cycles" =~ ^[1-9][0-9]*$ ]] || {
  printf 'AMR_EVOLVE_CYCLES must be a positive integer.\n' >&2
  exit 1
}
restart_cycles=$((evolve_cycles+2))
# Default validation starts fresh. An explicitly supplied historical checkpoint
# is frozen and physically verified; its prefix is not attributed to this binary.
evolve_start=()
validation_mode=()
if [[ -n ${AMR_START_CHECKPOINT:-} ]]; then
  "$python_exe" "$case_dir/scripts/verify_amr_run.py" "$test_dir" \
    --prepare-start-checkpoint "$AMR_START_CHECKPOINT" --evolve-cycle "$evolve_cycles" \
    > "$test_dir/start-validation.json"
  evolve_start=(-r "$test_dir/start/checkpoint.rst")
  mapfile -t start_solver_overrides < "$test_dir/start/solver-overrides.txt"
  evolve_start+=("${start_solver_overrides[@]}")
  validation_mode=(--checkpoint-start)
fi
# Sample total device use, including the CUDA context, during all validation runs.
nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" \
  --query-gpu=timestamp,uuid,name,memory.total,memory.used \
  --format=csv -l 1 > "$test_dir/gpu-memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
common=(time/ndiag=1 thermal_radiation/implicit_report=true
        thermal_radiation/source_report=true output6/dcycle=0)
for output in {1..6}; do common+=("output${output}/dt=-1"); done

# Preserve the full production geometry, capacity, material tables, and all 20 groups.
RUN_DIR="$test_dir/initial" bash "$case_dir/scripts/run_amr.slurm" "${common[@]}" \
  time/nlim=0 output1/dt=1 output4/dt=1 output5/dt=1
"$python_exe" "$case_dir/scripts/verify_initial.py" "$test_dir/initial" \
  > "$test_dir/initial-validation.json"
RUN_DIR="$test_dir/evolve" bash "$case_dir/scripts/run_amr.slurm" "${evolve_start[@]}" "${common[@]}" \
  "time/nlim=$evolve_cycles" output1/dt=1 output6/dcycle=5
grep -Eq "^time=.* cycle=$evolve_cycles$" "$test_dir/evolve/run.log" || {
  printf 'Evolution stopped before the requested %s cycles.\n' "$evolve_cycles" >&2
  exit 1
}
checkpoint=$(find "$test_dir/evolve/rst" -name '*.rst' -type f | sort | tail -n 1)
[[ -n "$checkpoint" ]] || { printf 'No AMR checkpoint produced.\n' >&2; exit 1; }
RUN_DIR="$test_dir/restart" bash "$case_dir/scripts/run_amr.slurm" -r "$checkpoint" \
  "${common[@]}" "time/nlim=$restart_cycles" output1/dt=1 output6/dcycle=5
grep -Eq "^time=.* cycle=$restart_cycles$" "$test_dir/restart/run.log" || {
  printf 'Restart stopped before the requested %s cycles.\n' "$restart_cycles" >&2
  exit 1
}
# Force every root block to refine, then perform a solve on all 432 leaves.
RUN_DIR="$test_dir/capacity" bash "$case_dir/scripts/run_amr.slurm" "${common[@]}" \
  time/nlim=2 mesh_refinement/refinement_interval=1 amr_criterion1/value_max=0
grep -q 'Current number of MeshBlocks = 432' "$test_dir/capacity/run.log"
grep -Eq '^time=.* cycle=2$' "$test_dir/capacity/run.log"
# The largest mixed topology has 425 leaves. This also exercises the packed
# FGMRES bases, which the uniformly refined 432-leaf CG solve does not need.
"$python_exe" "$case_dir/scripts/prepare_amr_capacity.py" \
  "$INPUT_FILE" "$test_dir/capacity_mixed.athinput"
RUN_DIR="$test_dir/capacity_mixed" INPUT_FILE="$test_dir/capacity_mixed.athinput" \
  bash "$case_dir/scripts/run_amr.slurm" "${common[@]}" time/nlim=1
grep -q 'Total number of MeshBlocks = 425' "$test_dir/capacity_mixed/run.log"
grep -Eq '^time=.* cycle=1$' "$test_dir/capacity_mixed/run.log"
# A failed monitor or an early walltime checkpoint must not count as a pass.
kill -0 "$monitor_pid"
kill "$monitor_pid"
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT
"$python_exe" "$case_dir/scripts/verify_amr_run.py" "$test_dir" \
  --evolve-cycle "$evolve_cycles" --restart-cycle "$restart_cycles" \
  --require-normal-timesteps "${validation_mode[@]}" \
  > "$test_dir/evolution-validation.json"
printf 'AMR GPU validation completed: %s\n' "$test_dir"
