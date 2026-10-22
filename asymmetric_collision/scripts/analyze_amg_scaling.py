#!/usr/bin/env python3
"""Summarize matched 1/2/3/4-GPU AMR replays and validate their final states."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from compare_states import compare, read_restart


def analyze(root):
    config = json.loads((root/'config.json').read_text())
    # Free-running CFL timesteps can differ in their last bits between decompositions.
    # Record this small time allowance; retain the existing field/dt/topology gates.
    config['state_time_relative_tolerance'] = 1e-12
    pattern = re.compile(r'^elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', re.M)
    trials = {}
    steps, warmup = config['timed_steps'], config['warmup_steps']
    for name in config['trials']:
        directory = root/name
        log = (directory/'run.log').read_text()
        assert 'Terminating on cycle limit' in log, name
        ranks = int(name.split('gpu')[0])
        assert f'Number of parallel ranks = {ranks}' in log, name
        mapping = [dict(rank=int(rank), local_rank=int(local), node=node, cpu=int(cpu),
                        gpu_uuid=uuid, pci=pci)
                   for rank, local, node, cpu, uuid, pci in re.findall(
                       r'# GPU-local rank: rank=(\d+) local_rank=(\d+) node=(\S+) '
                       r'pid=\d+ cpu=(\d+) gpu_uuid=(\S+) pci=(\S+)', log)]
        assert sorted(row['rank'] for row in mapping) == list(range(ranks)), name
        assert sorted(row['local_rank'] for row in mapping) == list(range(ranks)), name
        assert len({row['gpu_uuid'] for row in mapping}) == ranks, name
        assert len({row['cpu'] for row in mapping}) == ranks, name
        assert len({row['node'] for row in mapping}) == 1, name
        cycles = [dict(elapsed=float(e), cycle=int(c), time=float(t), dt=float(dt))
                  for e, c, t, dt in pattern.findall(log)]
        assert [c['cycle'] for c in cycles] == list(range(config['start_cycle'],
                                                        config['stop_cycle']+1)), name
        # The final diagnostic follows Finalize's restart write; discard its interval.
        seconds = [b['elapsed']-a['elapsed'] for a, b in zip(cycles[:steps], cycles[1:steps+1])]
        assert len(seconds) == steps and min(seconds) > 0, name
        restarts = list((directory/'rst').glob('*.rst'))
        assert len(restarts) == 1, (name, restarts)
        created, deleted = map(int, re.search(
            r'(\d+) MeshBlocks created, (\d+) deleted by AMR', log).groups())
        # This optional wrapper measurement is not used for solver speedups.
        wall_path = directory/'launch_wall_seconds.txt'
        wall_text = wall_path.read_text().strip() if wall_path.exists() else ''
        trials[name] = dict(gpus=ranks, host=re.search(r'^Host: (.+)$', log, re.M)[1],
                            timed_seconds=sum(seconds), seconds_per_step=seconds,
                            warm_seconds_per_step=statistics.mean(seconds[warmup:]),
                            startup_seconds=cycles[0]['elapsed'],
                            launch_wall_seconds=float(wall_text) if wall_text else None,
                            meshblock_cycles=int(re.search(r'MeshBlock-cycles = (\d+)', log)[1]),
                            final_leaves=int(re.search(r'Current number of MeshBlocks = (\d+)', log)[1]),
                            blocks_created=created, blocks_deleted=deleted,
                            gpu_mapping=mapping, cycles=cycles, restart=str(restarts[0]))
    assert len({trial['host'] for trial in trials.values()}) == 1
    reference = trials['1gpu-a']
    for name, trial in trials.items():
        for key in ('meshblock_cycles', 'final_leaves', 'blocks_created', 'blocks_deleted'):
            assert trial[key] == reference[key], (name, key)
        for a, b in zip(reference['cycles'], trial['cycles']):
            for key in ('time', 'dt'):
                assert math.isclose(a[key], b[key], rel_tol=2e-6, abs_tol=0), (name, key, a, b)
    rows = []
    for ranks in range(1, 5):
        repeats = [trial for trial in trials.values() if trial['gpus'] == ranks]
        assert len(repeats) == 2
        totals = [trial['timed_seconds'] for trial in repeats]
        warmed = [trial['warm_seconds_per_step'] for trial in repeats]
        wall_times = [trial['launch_wall_seconds'] for trial in repeats
                      if trial['launch_wall_seconds'] is not None]
        rows.append(dict(gpus=ranks, batch_size=config['batching'][str(ranks)],
                         mean_timed_seconds=statistics.mean(totals),
                         timed_seconds_range=[min(totals), max(totals)],
                         mean_warm_seconds_per_step=statistics.mean(warmed),
                         warm_seconds_per_step_range=[min(warmed), max(warmed)],
                         mean_startup_seconds=statistics.mean(t['startup_seconds'] for t in repeats),
                         mean_launch_wall_seconds=statistics.mean(wall_times) if wall_times else None,
                         launch_wall_samples=len(wall_times)))
    for row in rows:
        row['speedup'] = rows[0]['mean_timed_seconds']/row['mean_timed_seconds']
        row['warm_speedup'] = rows[0]['mean_warm_seconds_per_step']/row['mean_warm_seconds_per_step']
        row['parallel_efficiency'] = row['warm_speedup']/row['gpus']
    result = dict(config=config, host=reference['host'], summary=rows, trials=trials)
    # Save timings before expensive field comparisons, including if validation fails.
    (root/'amg_scaling.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows, indent=2), flush=True)
    baseline = read_restart(Path(reference['restart']))
    comparisons = {}
    for name, trial in trials.items():
        if name == '1gpu-a':
            continue
        state = compare(baseline, read_restart(Path(trial['restart'])), 5e-7,
                        time_tolerance=config['state_time_relative_tolerance'])
        comparisons[name] = state
        (root/f'comparison-{name}.json').write_text(json.dumps(state, indent=2)+'\n')
        print(f'{name}: final-state validation passed={state["passed"]}', flush=True)
    result['state_comparisons'] = comparisons
    result['validation_passed'] = all(state['passed'] for state in comparisons.values())
    (root/'amg_scaling.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Short AMR GPU scaling benchmark', '',
             f'Job `{config["job_id"]}`, node `{result["host"]}`. Two trials per GPU count, '
             'in order 1,2,3,4,4,3,2,1, with all four GPUs reserved throughout.', '',
             f'Each trial timed {steps} steps from cycle {config["start_cycle"]} to '
             f'{config["start_cycle"]+steps}; the warm result excludes the first {warmup} step(s). '
             'Initialization is excluded. One additional step and final restart I/O are excluded.', '',
             '| GPUs | Batch | Timed steps (s) | Warm s/step | Speedup | Warm speedup | Warm efficiency |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["gpus"]} | {row["batch_size"]} | {row["mean_timed_seconds"]:.3f} | '
                     f'{row["mean_warm_seconds_per_step"]:.3f} | {row["speedup"]:.3f}× | '
                     f'{row["warm_speedup"]:.3f}× | {row["parallel_efficiency"]:.1%} |')
    lines += ['', f'Initial/final leaves: {config["initial_leaves"]}/{reference["final_leaves"]}. '
              'Normal AMR remains enabled. All trials have matching cycle, time, timestep, '
              'final leaf counts and cumulative block-work counts.', '',
              'These are short checkpoint replays using the current default scheduling '
              '(one group on one GPU; four on multiple GPUs), not whole-simulation timings. '
              'The JSON includes trial ranges, per-step timings, startup and launch wall times.', '',
              f'Final-state validation passed: **{result["validation_passed"]}**. '
              'All active hydro, tracer, thermal and radiation fields are compared using '
              'the existing AMR restart comparator. Field limits remain unchanged; timestamps '
              'use an explicit relative tolerance of 1e-12 for CFL roundoff. Actual time '
              'differences and exact-equality flags are recorded. See comparison JSON files '
              'for individual gates.']
    (root/'amg_scaling.md').write_text('\n'.join(lines)+'\n')
    return result['validation_passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_directory', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if analyze(args.benchmark_directory.resolve()) else 1)
