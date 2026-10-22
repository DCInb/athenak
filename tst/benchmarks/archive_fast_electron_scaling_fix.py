#!/usr/bin/env python3
"""Archive validated scaling/convergence evidence without native checkpoints/traces."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--convergence', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job, convergence = args.job.resolve(), args.convergence.resolve()
    source, output = args.source.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    job_id = job.name.removeprefix('job-')
    assert (job/'complete.json').is_file()
    scaling = json.loads((output/'scaling.json').read_text())
    sensitivity = json.loads((output/'full_pulse_sensitivity.json').read_text())
    snapshot = json.loads((job.parent/'source-snapshot.json').read_text())
    assert snapshot['executables']['athena-gpu'] == scaling['metadata']['executables']['after']['sha256']
    assert sensitivity['metadata']['exe_sha256'] == snapshot['executables']['athena-gpu']
    # Copy the actual measured source, not whatever happens to be in the worktree.
    for name, expected in snapshot['files'].items():
        path = source/name
        assert digest(path) == expected, f'Measured source mismatch: {name}'
        copy(path, output/'measured-source'/name)
    copy(job.parent/'source-snapshot.json', output/'checks/source-snapshot.json')
    patch = []
    before = job.parent/'before'
    paths = {p.relative_to(source) for p in source.rglob('*') if p.is_file()}
    for name in sorted(paths):
        if not str(name).startswith('src/'):
            continue
        a, b = before/name, source/name
        if a.exists():
            copy(a, output/'baseline-source'/name)
        old = a.read_text().splitlines(keepends=True) if a.exists() else []
        new = b.read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff(old, new, fromfile=f'a/{name}', tofile=f'b/{name}'))
    (output/'implementation.patch').write_text(''.join(patch))
    # Source profiles and their citations are part of the benchmark input.
    profiles = root/'inputs/particles/cone_fast_ignition'
    for name, expected in sensitivity['metadata']['profile_hashes'].items():
        assert digest(profiles/'profiles'/name) == expected, f'Profile changed: {name}'
    for p in profiles.rglob('*'):
        if p.is_file(): copy(p, output/'inputs'/p.relative_to(profiles))
    for suite, destination in [(job, output/'timing-runs'),
                               (convergence, output/'convergence-runs')]:
        for p in suite.iterdir():
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.athinput'):
                copy(p, destination/p.name)
            elif p.is_dir():
                for name in ('run.log', 'command.json', 'analysis.json', 'fields.npz'):
                    if (p/name).is_file(): copy(p/name, destination/p.name/name)
    logs = [f'gpu-tests-{job_id}.log', f'cpu-tests-{job_id}.log',
            'test-cpu-four-ranks-final.log']
    counts = {}
    for name in logs:
        path = job.parent/name
        log = path.read_text()
        matches = re.findall(r'(\d+) passed', log)
        assert matches and not re.search(r'\d+ (?:failed|error)', log), path
        counts[name] = int(matches[-1])
        copy(path, output/'checks'/name)
    for name in [f'build-gpu-{job_id}.log', f'build-cpu-{job_id}.log']:
        copy(job.parent/name, output/'checks'/name)
    for name in ['measure.py', 'benchmark.slurm', 'full-pulse.slurm']:
        copy(job.parent/name, output/'reproduction'/name)
    for p in (root/'tst/benchmarks').glob('*.py'):
        if any(key in p.name for key in ('cone', 'scaling', 'bai', 'mpi_profile',
                                        'fast_electron_convergence')):
            copy(p, output/'reproduction'/p.name)
    complete = json.loads((job/'complete.json').read_text())
    summary = {'timing_runs': complete['timing_runs'], 'science_runs': complete['science_runs'],
               'profile_traces': len(scaling['profiles']),
               'full_pulse_refinement_runs': len(sensitivity['rows']),
               'regression_passes': counts,
               'max_relative_ledger_error_refinements': max(
                   r['max_relative_ledger_error_all_steps'] for r in sensitivity['validation']),
               'native_checkpoints_and_traces_included': False,
               'raw_scaling_directory': str(job), 'raw_convergence_directory': str(convergence),
               'science_comparisons_passed': True, 'spatial_convergence_established': False}
    (output/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    manifest = [f'{digest(p)}  {p.relative_to(output)}\n'
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    (output/'SHA256SUMS').write_text(''.join(manifest))
    print(json.dumps(dict(summary, checksummed_files=len(manifest)), indent=2))


if __name__ == '__main__':
    main()
