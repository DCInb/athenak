#!/usr/bin/env python3
"""Interleaved one/two-GPU comparisons on one allocated node.

Use the same input decks for both executables. Timings exclude initialization
and outputs; scientific outputs and optional Nsight traces run separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
from fast_electron_bai_benchmark import capture, run
from fast_electron_cone_benchmark import Setup, SOURCES, analyze

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before', type=Path, required=True)
parser.add_argument('--after', type=Path, required=True)
parser.add_argument('--decks', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--timeout', type=int, default=900)
parser.add_argument('--profile', action='store_true', help='Also record MPI/CUDA traces')
args = parser.parse_args()
if args.repeats < 3 or args.timeout <= 0:
    parser.error('Use at least three timing repeats and a positive timeout')
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
EXES = {'before': args.before.resolve(), 'after': args.after.resolve()}
DECKS = args.decks.resolve()
metadata = {'job': os.environ.get('SLURM_JOB_ID'), 'host': capture(['hostname']),
            'gpu': capture(['nvidia-smi', '-q']), 'topology': capture(['nvidia-smi','topo','-m']),
            'environment': {k: os.getenv(k) for k in ['UCX_TLS','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
            'executables': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                            for k, p in EXES.items()},
            'input_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in DECKS.glob('*.athinput')},
            'repeats': args.repeats, 'timing_excludes_initialization': True,
            'timing_has_no_mesh_outputs_or_profiler': True}
(OUT/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
results = []

def launch(ranks):
    return ['mpiexec', '--oversubscribe', '--bind-to', 'none', '-n', str(ranks)] if ranks > 1 else []

def execute(version, name, ranks, label, deck, extra=(), profile=False):
    directory = OUT/label
    launcher = launch(ranks)
    if profile:
        prefix = str(directory/('trace-r%q{OMPI_COMM_WORLD_RANK}' if ranks > 1 else 'trace-r0'))
        launcher += ['nsys','profile','--trace=cuda,nvtx,mpi','--mpi-impl=openmpi',
                     '--sample=none','--cpuctxsw=none','--export=sqlite','--output='+prefix]
    print('START', label, flush=True)
    row = run(EXES[version], directory, ['-i', deck, *extra], args.timeout, launcher)
    row.update(version=version, name=name, ranks=ranks, label=label, profile=profile)
    print(json.dumps(row), flush=True)
    return row

# A short run initializes both devices before the measured, untraced runs.
execute('after', SOURCES[0], 2, 'warmup', DECKS/(SOURCES[0]+'.athinput'), ['time/tlim=0.01'])
for repeat in range(args.repeats):
    offset = repeat % len(SOURCES)
    names = SOURCES[offset:]+SOURCES[:offset]
    variants = [('before',1),('after',1),('after',2),('before',2)]
    offset = repeat % len(variants)
    variants = variants[offset:]+variants[:offset]
    for name in names:
        for version, ranks in variants:
            row = execute(version, name, ranks, f'{name}-{version}-r{ranks}-rep{repeat}',
                          DECKS/(name+'.athinput'))
            row['repeat'] = repeat
            results.append(row)
            (OUT/'measurements.json').write_text(json.dumps(results, indent=2)+'\n')
    summary = []
    for name in SOURCES:
        for version in EXES:
            samples = {r: [x['loop_s'] for x in results if x['name']==name and
                          x['version']==version and x['ranks']==r] for r in (1,2)}
            if not all(samples.values()): continue
            t1,t2 = [statistics.median(samples[r]) for r in (1,2)]
            summary.append(dict(name=name,version=version,samples=samples,
                                one_gpu_s=t1,two_gpu_s=t2,speedup=t1/t2))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

science = []
for name in SOURCES:
    for ranks in (1,2):
        label = f'{name}-after-r{ranks}-science'
        row = execute('after', name, ranks, label, DECKS/(name+'-science.athinput'))
        row['science'] = analyze(OUT/label, Setup())
        science.append(row)
        (OUT/'science.json').write_text(json.dumps(science, indent=2)+'\n')
for name in SOURCES[:2] if args.profile else ():
    for version in EXES:
        execute(version, name, 2, name+'-'+version+'-r2-profile',
                DECKS/(name+'.athinput'), profile=True)
(OUT/'complete.json').write_text(json.dumps({'timing_runs':len(results),'science_runs':len(science)})+'\n')
