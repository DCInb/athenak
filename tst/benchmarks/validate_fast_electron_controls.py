#!/usr/bin/env python3
"""Two-GPU fixed-emission and simultaneous-memory validation on full pulses.

Writes new output directories only. Includes a source-time refinement and a
field/mesh comparison; these are diagnostics, not an assertion of convergence.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import numpy as np

from fast_electron_cone_benchmark import Setup, POWER_ERG_S, input_text, run, analyze
from fast_electron_bai_benchmark import Checkpoint, capture
from fast_electron_cone_physics_audit import fields_checkpoint
from fast_electron_memory_convergence import monitor


def records(log, kind):
    return [dict(re.findall(r'([a-z0-9_]+)=([^\s]+)', line))
            for line in log.splitlines() if line.startswith(f'# fast_electrons_{kind} ')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'tool', 'profiles', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    exe, tool, profiles, output = [getattr(args, k).resolve() for k in
                                  ('exe', 'tool', 'profiles', 'output')]
    output.mkdir(parents=True, exist_ok=False)
    launcher = ['mpiexec', '--oversubscribe', '--bind-to', 'none', '-n', '2']
    metadata = {'job': os.environ.get('SLURM_JOB_ID'), 'ranks': 2,
                'results_are_not_timing_samples': True,
                'profile_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in profiles.glob('*.dat')},
                'gpu': capture(['nvidia-smi', '-q']), 'hostname': capture(['hostname']),
                'exe': str(exe), 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                'tool': str(tool), 'tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
                'launcher': launcher}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    os.environ['KOKKOS_TOOLS_LIBS'] = str(tool)
    results, fieldsets, table = [], {}, []
    # The last two 16^3 cases have the same midpoint event times and weights:
    # Delta=.002 with P=2 equals Delta=.001 with P=1.
    cases = [('fixed', 16, .25, .002, 1), ('field', 16, .125, .002, 1),
             ('mesh', 24, .25, .002, 1), ('time', 16, .25, .001, 1),
             ('samples', 16, .25, .002, 2)]
    source = 'pic_collimated'
    rows = len(np.loadtxt(profiles/(source+'.dat')))
    for label, nx, cfl, interval, samples in cases:
        setup = Setup(nx=nx, field_cfl=cfl)
        directory, deck = output/label, output/(label+'.athinput')
        text = input_text(source, profiles/(source+'.dat'), setup, True)
        text = text.replace('packets_per_bin = 1\n', f'packets_per_bin = {samples}\n')
        text = text.replace('<fast_electrons>', '<fast_electrons>\n'+
            f'emission_interval = {interval}\nreclaim_packet_memory = true\n'+
            'memory_diagnostics = true\nmemory_budget_mib = 16000\n'+
            'memory_headroom_mib = 512\nparticle_diagnostics_interval = 100\n'+
            'particle_diagnostics_cells = true')
        deck.write_text(text)
        os.environ['KOKKOS_MEMORY_OUTPUT'] = str(directory)
        stop = threading.Event()
        watcher = threading.Thread(target=monitor, args=(output/(label+'-gpu.csv'), stop))
        watcher.start()
        print('START', label, flush=True)
        try:
            result = run(exe, directory, ['-i', deck], 1800, launcher)
        finally:
            stop.set()
            watcher.join()
        result.update(label=label, setup=asdict(setup), interval=interval, samples=samples,
                      input_sha256=hashlib.sha256(deck.read_bytes()).hexdigest())
        result.update(analyze(directory, setup))
        path = sorted((directory/'rst').glob('*.rst'))[-1]
        cp = Checkpoint(path)
        assert cp.header[0] == 2
        expected = rows*setup.radial_order*setup.spatial_azimuths*round(setup.duration_ps/interval)*samples
        assert int(cp.header[6]) == expected, (label, cp.header[6], expected)
        result['emitted_packets'] = expected
        injected = POWER_ERG_S*setup.duration_ps*1e-12
        assert abs(result['ledger']['injected_erg']/injected-1) < 1e-10
        assert abs(result['ledger']['residual_erg']/injected) < 1e-9
        fields, checks = fields_checkpoint(path)
        fieldsets[label] = fields
        result['physical_checks'] = checks
        log = (directory/'run.log').read_text()
        memory_log = '\n'.join((directory/f'fast_electron_memory.rank{r}.log').read_text() for r in (0,1))
        sampling_log = '\n'.join((directory/f'fast_electron_sampling.rank{r}.log').read_text() for r in (0,1))
        capacities, allocations = [records(memory_log, k) for k in ('capacity', 'allocation')]
        sampling = records(sampling_log, 'sampling')
        assert {r['rank'] for r in capacities} == {'0', '1'}
        result['peak_live_packets_at_macro_steps'] = max(map(int, re.findall(
            r'^# fast_electrons .* packets=(\d+)$', log, re.MULTILINE)))
        synchronized = {}
        for r in capacities:
            if r['phase'] in ('injected', 'transported'):
                synchronized.setdefault((r['time'], r['phase']), {})[r['rank']] = int(r['live_packets'])
        assert all(len(v) == 2 for v in synchronized.values())
        result['peak_live_global_at_accepted_boundaries'] = max(sum(v.values()) for v in synchronized.values())
        for r in capacities:
            assert int(r['total_bytes']) == sum(int(r[k+'_bytes']) for k in
                                                ('packet','source','mesh','runtime','diagnostic'))
            assert int(r['live_bytes']) <= int(r['packet_bytes'])
            assert int(r['packet_bytes']) == 104*sum(int(r[k+'_slots']) for k in
                                                    ('storage','scratch','work0','work1','export'))
        result['ranks'] = []
        for rank in (0, 1):
            mem = json.loads((directory/f'memory-rank{rank}.json').read_text())
            assert mem['final'] and mem['duplicate_allocations'] == 0
            allocation_peak = max((r for r in allocations if r['rank'] == str(rank)),
                                  key=lambda r: int(r['total_bytes']))
            observed = int(allocation_peak['total_bytes'])
            independent = next(s['peak_bytes'] for s in mem['spaces'] if s['space'] == 'Cuda')
            assert observed == independent, (label, rank, observed, independent)
            peak = max((r for r in capacities if r['rank'] == str(rank)),
                       key=lambda r: int(r['total_bytes']))
            local = [r for r in sampling if r['rank'] == str(rank) and r['phase'] == 'transported']
            assert local
            occupied = [r for r in local if int(r['occupied_cells'])]
            row = dict(label=label, rank=rank, nx=nx, field_cfl=cfl, emission_interval=interval,
                       packets_per_bin=samples, emitted_global=expected,
                       allocation_peak_live_known=int(allocation_peak['live_known']),
                       allocation_peak_live=int(allocation_peak['live_packets']),
                       allocation_peak_particle_to_mesh=int(allocation_peak['packet_bytes'])/int(allocation_peak['mesh_bytes']),
                       allocated_peak_bytes=observed, snapshot_peak_bytes=int(peak['total_bytes']),
                       snapshot_live=int(peak['live_packets']),
                       capacity_to_live=float(peak['capacity_to_live']),
                       particle_to_mesh=float(peak['particle_to_mesh']),
                       max_sampled_local_ppc=max(int(r['ppc_max']) for r in local),
                       max_sampled_mean_ppc=max(float(r['ppc_mean']) for r in local),
                       min_sampled_mean_neff_ratio=min(float(r['neff_sum'])/int(r['packets'])
                                                       for r in occupied),
                       final_packet_bytes=int([r for r in capacities if r['rank']==str(rank)][-1]['packet_bytes']))
            result['ranks'].append(row)
            table.append(row)
        (directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
        print('DONE', label, 'emitted', expected, 'seconds', result['loop_s'], flush=True)
    # Compare equal-mesh energy-deposition maps using physical volume and injected energy.
    # No hard accuracy claim is made from a single refinement.
    comparisons = []
    # dep is still in code energy density; the 64 micron cube has code volume 64^3.
    from fast_electron_cone_benchmark import ENERGY_UNIT
    volume_energy = (64/16)**3 * ENERGY_UNIT
    for a, b in [('fixed', 'field'), ('fixed', 'time'), ('time', 'samples')]:
        comparisons.append({'a': a, 'b': b,
            'deposition_L1_over_injected': float(np.abs(fieldsets[a]['dep']-fieldsets[b]['dep']).sum()*volume_energy/injected)})
    coarsened = {}
    for label in ('fixed', 'mesh'):
        a = fieldsets[label]['dep']
        factor = a.shape[0]//8
        coarsened[label] = a.reshape(8,factor,8,factor,8,factor).mean(axis=(1,3,5))
    comparisons.append({'a':'fixed', 'b':'mesh', 'comparison_grid':8,
        'deposition_L1_over_injected':float(np.abs(coarsened['fixed']-coarsened['mesh']).sum()*(64/8)**3*ENERGY_UNIT/injected)})
    with (output/'memory_sampling.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (output/'complete.json').write_text(json.dumps({'cases':len(results), 'results':results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
