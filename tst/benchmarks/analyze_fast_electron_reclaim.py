#!/usr/bin/env python3
"""Verify CUDA allocation releases and archive the two-GPU reclamation study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-tests', default='1550036')
    parser.add_argument('--final-tests', default='1550045')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs, output = args.runs.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert json.loads((runs/'complete.json').read_text()) == {'runs': 6, 'gpus': 2}
    metadata = json.loads((runs/'metadata.json').read_text())
    for key in ('exe', 'tool'):
        assert digest(Path(metadata[key])) == metadata[key+'_sha256']
    results = json.loads((runs/'results.json').read_text())
    comparisons = json.loads((runs/'comparisons.json').read_text())
    assert len(results) == 6 and len(comparisons) == 2
    sources = ['mono_collimated', 'pic_collimated']
    labels = ['Monoenergetic', 'Collimated PIC']
    styles = {'retain': ('#777777', '-', 'Retain buffers'),
              'retain-repeat': ('#aaaaaa', '--', 'Retain: repeat'),
              'reclaim': ('#2c7850', '-', 'Reclaim buffers')}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    records = []
    for result in results:
        label = result['source']+'-'+result['mode']
        directory = runs/label
        assert digest(runs/(label+'.athinput')) == result['input_sha256']
        assert digest(root/result['physical_checks']['path']) == result['physical_checks']['sha256']
        for rank in (0, 1):
            with (directory/f'memory-rank{rank}.json.events.csv').open() as stream:
                events = list(csv.DictReader(stream))
            times = [float(e['elapsed_s']) for e in events]
            counts = [int(e['packet_live_bytes']) for e in events]
            live = 0
            for event, count in zip(events, counts):
                live += int(event['bytes'])*(1 if event['event'] == 'allocate' else -1)
                assert live == count and live >= 0
            assert live == 0
            local = [e for e in result['reclamations'] if e['rank'] == rank]
            verified = []
            cursor = 0
            for release in local:
                before = release['reserved_before']*104
                after = release['reserved_after']*104
                start = counts.index(before, cursor)
                end = counts.index(after, start+1)
                delta = sum(int(e['bytes'])*(1 if e['event'] == 'allocate' else -1)
                            for e in events[start+1:end+1])
                assert delta == -release['released_bytes'] == after-before
                verified.append(dict(release, trace_elapsed_s=times[end],
                                     cuda_bytes_after=int(events[end]['cuda_live_bytes'])))
                cursor = end
            memory = result['memory'][rank]
            assert memory['rank'] == rank
            record = {'source': result['source'], 'mode': result['mode'], 'rank': rank,
                      'peak_packet_buffer_bytes': max(counts),
                      'tracked_cuda_peak_bytes': next(s['peak_bytes'] for s in memory['spaces']
                                                      if s['space'] == 'Cuda'),
                      'verified_releases': verified}
            records.append(record)
            ax = axes[sources.index(result['source']), rank]
            color, linestyle, legend = styles[result['mode']]
            ax.step(times, [n/2**30 for n in counts], where='post', color=color,
                    linestyle=linestyle, label=legend)
            if verified:
                ax.scatter([v['trace_elapsed_s'] for v in verified],
                           [v['reserved_after']*104/2**30 for v in verified],
                           color=color, marker='x', s=30, zorder=5)
            ax.set(title=f'{labels[sources.index(result["source"])]}, GPU rank {rank}',
                   xlabel='Elapsed time since profiler initialization [s]',
                   ylabel='Allocated packet buffers [GiB]')
            ax.grid(alpha=.2)
        for filename in ['analysis.json', 'run.log', 'command.json', 'fields.npz',
                         *[f'memory-rank{rank}.json{suffix}' for rank in (0, 1)
                           for suffix in ('', '.events.csv')]]:
            copy(directory/filename, output/'runs'/label/filename)
        for suffix in ('.athinput', '-gpu.csv'):
            copy(runs/(label+suffix), output/'runs'/(label+suffix))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Two GPUs: packet allocation lifetimes; crosses mark verified reclamation\n'
                 'Includes reserved capacity and growth overlap; excludes source templates, mesh and CUDA context',
                 fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'packet_memory_reclamation.{suffix}', dpi=180)
    plt.close(fig)
    for name in ('metadata.json', 'results.json', 'comparisons.json', 'complete.json'):
        copy(runs/name, output/name)
    (output/'allocation_releases.json').write_text(json.dumps(records, indent=2)+'\n')
    work = runs.parent
    logs = {'initial_cpu_tests.log': work/f'cpu-tests-{args.initial_tests}.log',
            'corrected_cpu_tests.log': work/f'cpu-checks-{args.final_tests}.log',
            'gpu_tests.log': work/f'gpu-tests-{args.final_tests}.log'}
    assert '76 passed' in logs['initial_cpu_tests.log'].read_text()
    assert '9 passed' in logs['corrected_cpu_tests.log'].read_text()
    assert '78 passed' in logs['gpu_tests.log'].read_text()
    for name, path in logs.items():
        copy(path, output/'checks'/name)
    for path in (work/'tool-selfcheck').iterdir():
        copy(path, output/'checks/tool-selfcheck'/path.name)
    snapshot = json.loads((work/'source-snapshot.json').read_text())
    for path, expected in snapshot['files'].items():
        assert digest(root/path) == expected, path
        copy(root/path, output/'source'/path)
    for filename in ('source-snapshot.json', 'test.slurm', 'finish-tests.slurm', 'measure.slurm'):
        copy(work/filename, output/'checks'/filename)
    for path in (root/'tst/benchmarks').glob('*.py'):
        if any(key in path.name for key in ('reclaim', 'cone', 'bai', 'memory_convergence',
                                            'scaling_fix', 'mpi_profile')):
            copy(path, output/'reproduction'/path.name)
    copy(root/'tst/benchmarks/kokkos_memory_profile.cpp',
         output/'reproduction/kokkos_memory_profile.cpp')
    for name, expected in metadata['profile_hashes'].items():
        path = root/'inputs/particles/cone_fast_ignition/profiles'/name
        assert digest(path) == expected
        copy(path, output/'inputs/profiles'/name)
    validation = {'cpu_unique_tests_passed': 78, 'gpu_tests_passed': 78,
                  'new_reclamation_test_cases_per_backend': 8, 'full_pulse_gpu_runs': 6,
                  'gpus_per_full_pulse_run': 2,
                  'verified_reclamation_events': sum(len(r['verified_releases']) for r in records),
                  'initial_test_issues': [
                      'The standalone physics test initially used a copied executable without its CMake directory.',
                      'One mixed-population test initially did not cross the quarter-full reclamation threshold; the thermalized fraction was increased.'],
                  'initial_full_pulse_issue': 'The first comparison required 1e-8 field agreement; differences occurred before any reclamation. Final runs include an unchanged-setting repeat and record both differences.',
                  'full_pulse_tolerances': {'relative_field_linf': .01,
                                            'deposition_l1_over_injected': .001,
                                            'central_deposition_change_pp': .01,
                                            'relative_emitted_count_change': .001},
                  'native_checkpoints_included': False, 'raw_runs': str(runs)}
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    manifest = [f'{digest(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*'))
                if p.is_file() and p.name != 'SHA256SUMS']
    (output/'SHA256SUMS').write_text(''.join(manifest))
    print(json.dumps(dict(validation, checksummed_files=len(manifest)), indent=2))


if __name__ == '__main__':
    main()
