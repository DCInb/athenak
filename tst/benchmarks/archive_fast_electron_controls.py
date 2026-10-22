#!/usr/bin/env python3
"""Archive a completed controls validation, including raw per-cell diagnostics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run, output = args.measurement.resolve(), args.output.resolve()
    completed = json.loads((run/'complete.json').read_text())
    meta = json.loads((run/'metadata.json').read_text())
    assert sha(Path(meta['exe'])) == meta['exe_sha256']
    assert sha(Path(meta['tool'])) == meta['tool_sha256']
    assert completed['cases'] == 5
    output.mkdir(parents=True, exist_ok=False)
    for path in run.rglob('*'):
        if not path.is_file():
            continue
        target = output/'runs'/path.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ('.log', '.csv', '.rst'):
            with path.open('rb') as a, gzip.GzipFile(str(target)+'.gz', 'wb', mtime=0) as b:
                shutil.copyfileobj(a, b)
        else:
            shutil.copyfile(path, target)
    shutil.copyfile(run/'memory_sampling.csv', output/'memory_sampling.csv')
    shutil.copyfile(run/'comparisons.json', output/'comparisons.json')
    sources = [*sorted((root/'src').rglob('*')),
               root/'CMakeLists.txt', *sorted((root/'cmake').rglob('*')),
               root/'inputs/particles/fast_electrons.athinput',
               root/'inputs/particles/fast_electrons_hybrid.athinput',
               *sorted((root/'inputs/particles/cone_fast_ignition/profiles').glob('*.dat')),
               *sorted((root/'tst/test_suite/nr').glob('*fast_electron*')),
               *sorted((root/'tst/benchmarks').glob('*.py')),
               root/'tst/benchmarks/kokkos_memory_profile.cpp']
    sources = sorted(set(p for p in sources if p.is_file() and '__pycache__' not in str(p)))
    hashes = {str(p.relative_to(root)):sha(p) for p in sources}
    (output/'source-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(root)))
    expected_checks = {'cpu-all-v4.log':'95 passed', 'cpu-extra-v5.log':'3 passed',
                       'cpu-controls-final.log':'20 passed',
                       'gpu-tests-1550200.log':'97 passed',
                       'gpu-final-1550205.log':'21 passed',
                       'cpu-logs.log':'20 passed', 'gpu-logs-1550213.log':'20 passed'}
    for name, expected in expected_checks.items():
        assert expected in (root/'build-fast-electron-controls'/name).read_text(), name
    checks = output/'checks'
    checks.mkdir()
    for name in ('cpu-all-v4.log', 'cpu-extra-v5.log', 'cpu-controls-final.log',
                 'gpu-tests-1550200.log', 'gpu-final-1550205.log',
                 'cpu-logs.log', 'gpu-logs-1550213.log',
                 'build-cpu-logs.log', 'build-gpu-logs-1550213.log',
                 'validate-logs.slurm', 'validate-logs-1550213.log',
                 'build-cpu-final.log', 'build-gpu-final-1550205.log',
                 'validate.slurm', 'final-check.slurm', 'final-check-1550205.log'):
        shutil.copyfile(root/'build-fast-electron-controls'/name, checks/name)
    for build, label in [('build-hybrid-performance/cpu','cpu'),('build-fast-electrons-gpu','gpu')]:
        for name in ('CMakeCache.txt','defs.hpp'):
            if (root/build/name).is_file():
                shutil.copyfile(root/build/name, checks/(label+'-'+name))
    validation = {'cpu_cases':98, 'gpu_cases':98, 'new_controls_cases':20,
                  'cpu_final_controls':'20 passed',
                  'gpu_initial':'97 passed; standalone physics fixture lacked its CMake path',
                  'gpu_final_controls_and_physics':'21 passed after correcting executable path',
                  'rank_log_followup':'20 CPU and 20 GPU control cases passed',
                  'authoritative_diagnostics':'per-rank logs; MPI stdout may interleave',
                  'two_gpu_full_pulse_cases':5, 'allocation_profiler_peak_matches':10,
                  'source_is_fixed_across_field_and_mesh_refinements':True,
                  'physical_convergence_not_established':True,
                  'measured_executable_sha256':meta['exe_sha256'],
                  'raw_run_directory':str(run)}
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    manifest = [f'{sha(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file()]
    (output/'SHA256SUMS').write_text(''.join(manifest))
    print(json.dumps({'files':len(manifest), 'archive':str(output)}, indent=2))


if __name__ == '__main__':
    main()
