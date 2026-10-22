#!/usr/bin/env python3
"""Measure checkpoint replay scaling and compare the final conserved cell states."""

import argparse
import json
from pathlib import Path
import re
import struct

import numpy as np


def read_restart(path):
    with path.open('rb') as stream:
        header = stream.read(40960)
    start = header.index(b'<par_end>') + 10
    blocks, level = struct.unpack_from('=ii', header, start)
    region = struct.unpack_from('=9d', header, start+8)
    mesh = struct.unpack_from('=19i', header, start+8+72)
    block = struct.unpack_from('=19i', header, start+8+72+76)
    time, dt, cycle = struct.unpack_from('=ddi', header, start+8+72+152)
    ids_offset = start+8+72+152+20
    size_offset = ids_offset+blocks*20
    bytes_per_block, = struct.unpack_from('=Q', header, size_offset)
    payload = size_offset+8
    assert path.stat().st_size == payload+blocks*bytes_per_block
    ng, nx1, nx2, nx3 = block[:4]
    shape = (nx3+2*ng, nx2+2*ng, nx1+2*ng)
    cells = int(np.prod(shape))
    assert bytes_per_block % (cells*8) == 0
    nvar = bytes_per_block//(cells*8)
    assert nvar == 32, 'Expected the five-tracer, two-temperature, 20-group case'
    values = np.memmap(path, mode='r', dtype='=f8', offset=payload,
                       shape=(blocks, nvar)+shape)
    interior = values[:, :, ng:ng+nx3, ng:ng+nx2, ng:ng+nx1]
    info = dict(time=time, dt=dt, cycle=cycle, blocks=blocks, level=level,
                region=region, mesh=mesh, block=block,
                logical_ids=header[ids_offset:size_offset].hex())
    return info, interior


def compare_states(left, right):
    ai, a = read_restart(left)
    bi, b = read_restart(right)
    for key in ('cycle', 'blocks', 'level', 'region', 'mesh', 'block', 'logical_ids'):
        assert ai[key] == bi[key], key
    assert np.isclose(ai['time'], bi['time'], rtol=1e-11, atol=1e-15)
    assert np.isclose(ai['dt'], bi['dt'], rtol=1e-9, atol=1e-15)
    labels = ['density', 'momentum_x1', 'momentum_x2', 'momentum_x3', 'total_energy',
              'CH', 'He', 'upper_CH', 'lower_CH', 'corona_CH', 'ion_energy',
              'electron_energy'] + [f'radiation_{g:02d}' for g in range(20)]
    rows = []
    for field, name in enumerate(labels):
        x, y = np.asarray(a[:, field]), np.asarray(b[:, field])
        assert np.isfinite(x).all() and np.isfinite(y).all(), name
        delta = y-x
        scale = max(float(np.max(np.abs(x))), 1e-300)
        linf = float(np.max(np.abs(delta)))/scale
        l2 = float(np.sqrt(np.sum(delta*delta)))/max(float(np.sqrt(np.sum(x*x))), 1e-300)
        rows.append(dict(field=name, relative_linf=linf, relative_l2=l2,
                         max_absolute_difference=float(np.max(np.abs(delta)))))
    assert np.all(a[:, 0] > 0) and np.all(b[:, 0] > 0)
    for field in range(10, 32):
        assert np.all(a[:, field] >= 0) and np.all(b[:, field] >= 0), labels[field]
    result = dict(left=str(left), right=str(right), cycle=ai['cycle'],
                  left_time_ns=ai['time'], right_time_ns=bi['time'],
                  max_relative_linf=max(r['relative_linf'] for r in rows),
                  max_relative_l2=max(r['relative_l2'] for r in rows),
                  mass_relative_difference=float(np.sum(b[:, 0])/np.sum(a[:, 0])-1),
                  compared_interior_values=int(a.size), fields=rows)
    result['passed'] = result['max_relative_linf'] < 1e-6 and result['max_relative_l2'] < 1e-7
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_directory', type=Path)
    parser.add_argument('--warmup-steps', type=int, default=2)
    parser.add_argument('--measured-steps', type=int, default=10)
    args = parser.parse_args()
    root = args.benchmark_directory.resolve()
    trials = {}
    pattern = re.compile(r'^elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)', re.M)
    for name in ('1gpu-a', '4gpu-a', '4gpu-b', '1gpu-b'):
        directory = root/name
        log = (directory/'run.log').read_text()
        assert 'Terminating on cycle limit' in log, name
        cycles = [dict(elapsed=float(e), cycle=int(c), time_ns=float(t), dt_ns=float(dt))
                  for e, c, t, dt in pattern.findall(log)]
        assert all(b['cycle'] == a['cycle']+1 for a, b in zip(cycles, cycles[1:])), name
        start, count = args.warmup_steps, args.measured_steps
        assert len(cycles) == start+count+2, (name, len(cycles))
        window = cycles[start:start+count+1]
        seconds = [b['elapsed']-a['elapsed'] for a, b in zip(window, window[1:])]
        assert len(seconds) == count and min(seconds) > 0
        ranks = int(name[0])
        assert f'Number of parallel ranks = {ranks}' in log
        restarts = list((directory/'rst').glob('*.rst'))
        assert len(restarts) == 1
        trials[name] = dict(gpus=ranks, host=re.search(r'^Host: (.+)$', log, re.M)[1],
                            mean_seconds_per_step=float(np.mean(seconds)),
                            seconds_per_step=seconds,
                            time_advanced_ns=window[-1]['time_ns']-window[0]['time_ns'],
                            measured_start_cycle=window[0]['cycle'],
                            measured_end_cycle=window[-1]['cycle'],
                            cycles=cycles, restart=str(restarts[0]))
    assert len({v['host'] for v in trials.values()}) == 1
    reference = trials['1gpu-a']['cycles']
    for trial in trials.values():
        for a, b in zip(reference, trial['cycles']):
            assert a['cycle'] == b['cycle']
            assert np.isclose(a['time_ns'], b['time_ns'], rtol=2e-6, atol=1e-12)
            assert np.isclose(a['dt_ns'], b['dt_ns'], rtol=2e-6, atol=1e-12)
    mapping = [json.loads(line) for line in (root/'gpu_mapping.jsonl').read_text().splitlines()
               if line.startswith('{')]
    assert sorted(r['rank'] for r in mapping) == [0, 1, 2, 3]
    assert len({r['pci_bus'] for r in mapping}) == 4
    assert len({r['host'] for r in mapping}) == 1
    one = np.mean([trials[k]['mean_seconds_per_step'] for k in ('1gpu-a', '1gpu-b')])
    four = np.mean([trials[k]['mean_seconds_per_step'] for k in ('4gpu-a', '4gpu-b')])
    comparisons = {}
    baseline = Path(trials['1gpu-a']['restart'])
    for name in ('4gpu-a', '4gpu-b', '1gpu-b'):
        comparisons[name] = compare_states(baseline, Path(trials[name]['restart']))
    result = dict(benchmark_directory=str(root), host=trials['1gpu-a']['host'],
                  warmup_steps=args.warmup_steps, measured_steps=args.measured_steps,
                  one_gpu_seconds_per_step=float(one), four_gpu_seconds_per_step=float(four),
                  speedup=float(one/four), parallel_efficiency=float(one/four/4),
                  four_gpu_cost_ratio=float(4*four/one), trials=trials,
                  gpu_mapping=mapping, state_comparisons=comparisons,
                  validation_passed=all(v['passed'] for v in comparisons.values()))
    (root/'gpu_scaling.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# HLLC/PLM GPU scaling benchmark', '',
             f'Node: `{result["host"]}`. Same executable and checkpoint for every replay.', '',
             'All four GPUs were reserved throughout. Each one-GPU trial used one MPI rank; '
             'each four-GPU trial used four ranks mapped to four distinct devices.', '',
             'Order: 1 GPU, 4 GPUs, 4 GPUs, 1 GPU. Two warmup steps excluded. '
             'Ten steps timed in each trial; the final step and its output writes were excluded.', '',
             '| Trial | Seconds/step |', '|---|---:|']
    lines.extend(f'| {k} | {v["mean_seconds_per_step"]:.6f} |' for k, v in trials.items())
    lines += ['', f'Mean: **{one:.6f} s/step on one GPU**, **{four:.6f} s/step on four GPUs**.',
              f'Speedup: **{one/four:.3f}×**. Parallel efficiency: **{one/four/4:.1%}**. '
              f'GPU-seconds per step relative to one GPU: **{4*four/one:.3f}×**.', '',
              'The 108³ mesh has 27 blocks of 36³ cells (6/7/7/7 blocks across four ranks). '
              'This samples evolution after 0.1 ns and excludes regular output costs; '
              'performance later in the collision can differ.', '',
              'Final double-precision conserved states were compared on physical cells, '
              'excluding ghost zones. Finite fields, positive density, nonnegative component/'
              'radiation energies, matching cycles/times/timesteps, and mesh ordering were checked.', '',
              '| Compared with 1gpu-a | Maximum field-relative L∞ | Maximum field-relative L2 |',
              '|---|---:|---:|']
    lines.extend(f'| {k} | {v["max_relative_linf"]:.3e} | {v["max_relative_l2"]:.3e} |'
                 for k, v in comparisons.items())
    lines += ['', 'Acceptance thresholds: field-relative L∞ < 1e-6 and L2 < 1e-7.',
              f'Validation passed: **{result["validation_passed"]}**.', '',
              'See [gpu_scaling.json](gpu_scaling.json) for per-step timings and all field differences.']
    (root/'gpu_scaling.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ('host', 'one_gpu_seconds_per_step',
                     'four_gpu_seconds_per_step', 'speedup', 'parallel_efficiency',
                     'four_gpu_cost_ratio', 'validation_passed')}, indent=2))
    assert result['validation_passed'], 'State differences exceed benchmark acceptance thresholds'


if __name__ == '__main__':
    main()
