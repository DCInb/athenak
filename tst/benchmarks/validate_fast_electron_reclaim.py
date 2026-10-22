#!/usr/bin/env python3
"""Compare full-pulse physics and packet-memory reclamation on two GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import numpy as np

from fast_electron_cone_benchmark import Setup, POWER_ERG_S, input_text, run, analyze
from fast_electron_cone_physics_audit import fields_checkpoint
from analyze_fast_electron_scaling_fix import physical_checks
from fast_electron_memory_convergence import monitor
from fast_electron_bai_benchmark import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exe, tool, profiles, output = [p.resolve() for p in
                                  (args.exe, args.tool, args.profiles, args.output)]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'exe': str(exe), 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                'tool': str(tool), 'tool_sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
                'gpus': 2, 'job': os.environ.get('SLURM_JOB_ID'),
                'hostname': capture(['hostname']), 'gpu': capture(['nvidia-smi', '-q']),
                'profile_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in profiles.glob('*.dat')}}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    os.environ['KOKKOS_TOOLS_LIBS'] = str(tool)
    launcher = ['mpiexec', '--oversubscribe', '--bind-to', 'none', '-n', '2']
    setup = Setup()
    results, comparisons = [], []
    for source in ['mono_collimated', 'pic_collimated']:
        pair, fields = [], []
        # Repeat the unchanged setting: GPU atomic accumulation and nonlinear
        # subcycling need a measured repeatability reference, not bitwise claims.
        for mode, enabled in [('retain', False), ('retain-repeat', False), ('reclaim', True)]:
            label = source+'-'+mode
            directory = output/label
            deck = output/(label+'.athinput')
            text = input_text(source, profiles/(source+'.dat'), setup, True)
            text = text.replace('<fast_electrons>', '<fast_electrons>\n'
                                f'reclaim_packet_memory = {str(enabled).lower()}')
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
            result.update(analyze(directory, setup))
            checkpoint = sorted((directory/'rst').glob('*.rst'))[-1]
            values, checks = fields_checkpoint(checkpoint)
            checks.update(physical_checks(checkpoint))
            fields.append(values)
            events = [{key: int(value) for key, value in
                       re.findall(r'([a-z_]+)=(\d+)', line)}
                      for line in (directory/'run.log').read_text().splitlines()
                      if line.startswith('# fast_electrons_memory ')]
            memory = [json.loads((directory/f'memory-rank{rank}.json').read_text())
                      for rank in (0, 1)]
            assert all(m['final'] and m['duplicate_allocations'] == 0 for m in memory)
            for m in memory:
                assert m['untracked_deallocations'] in ([], [
                    {'space': 'Cuda', 'label': '[unlabeled]', 'bytes': 65536}])
            assert result['time'] == 1.5
            ledger = result['ledger']
            expected = POWER_ERG_S*setup.duration_ps*1e-12
            assert abs(ledger['injected_erg']/expected-1) < 1e-10
            assert ledger['packets'] == ledger['active_erg'] == 0
            assert abs(ledger['residual_erg'])/ledger['injected_erg'] < 1e-9
            if enabled:
                assert events and all(e['released_bytes'] > 0 for e in events)
                for rank in (0, 1):
                    local = [e for e in events if e['rank'] == rank]
                    assert local and local[-1]['live_packets'] == 0
                    assert local[-1]['reserved_after'] == 0
            else:
                assert not events
            result.update(source=source, mode=mode, reclaim_packet_memory=enabled,
                          input_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
                          physical_checks=checks, reclamations=events, memory=memory)
            (directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
            pair.append(result)
            results.append(result)
            print('DONE', label, 'reclamations', len(events), flush=True)
        checks = {}
        for index, mode in [(1, 'unchanged_repeat'), (2, 'reclaim')]:
            differences = {}
            for key in ['rho', 'Te', 'Ti', 'B', 'dep', 'ui', 'ue', 'bx', 'by', 'bz']:
                # Normalize vector components by the full magnetic magnitude,
                # avoiding a nearly zero component setting the relative scale.
                scale_key = 'B' if key in ('bx', 'by', 'bz') else key
                scale = max(float(np.max(np.abs(fields[0][scale_key]))), 1e-100)
                if scale_key == 'B' and key != 'B':
                    from fast_electron_cone_physics_audit import BUNIT
                    scale *= 1e7/BUNIT
                differences[key] = float(np.max(np.abs(fields[index][key]-fields[0][key]))/scale)
            a = np.load(output/(source+'-retain')/'fields.npz')['deposition_J_per_um3']
            b = np.load(output/(source+'-'+pair[index]['mode'])/'fields.npz')['deposition_J_per_um3']
            deposition_l1 = float(np.abs(a-b).sum()*(64/setup.nx)**3/
                                  (pair[0]['ledger']['injected_erg']/1e7))
            central_delta = 100*(pair[index]['hotspot_fraction']-pair[0]['hotspot_fraction'])
            emitted_change = (pair[index]['total_emitted_macro_packets']/
                              pair[0]['total_emitted_macro_packets']-1)
            # These full-pulse bounds accompany the much tighter deterministic
            # survivor/RNG/restart regressions. Preserve actual repeat differences.
            assert max(differences.values()) < .01, (source, mode, differences)
            assert deposition_l1 < .001, (source, mode, deposition_l1)
            assert abs(central_delta) < .01, (source, mode, central_delta)
            assert abs(emitted_change) < .001, (source, mode, emitted_change)
            checks[mode] = {'relative_linf_fields': differences,
                            'deposition_l1_over_injected': deposition_l1,
                            'central_deposition_change_pp': central_delta,
                            'relative_emitted_count_change': emitted_change}
        peaks = [[next(s['peak_bytes'] for s in m['spaces'] if s['space'] == 'Cuda')
                  for m in result['memory']] for result in pair]
        comparisons.append({'source': source, 'checks': checks,
                            'tracked_cuda_peak_bytes_retain': peaks[0],
                            'tracked_cuda_peak_bytes_repeat': peaks[1],
                            'tracked_cuda_peak_bytes_reclaim': peaks[2]})
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (output/'complete.json').write_text(json.dumps({'runs': len(results), 'gpus': 2})+'\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
