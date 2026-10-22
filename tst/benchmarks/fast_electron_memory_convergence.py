#!/usr/bin/env python3
"""Two-GPU memory probes and explicit particle/mesh convergence cases.

Cases are supplied as JSON, with mesh, particle samples per source bin, spatial
quadrature and seed recorded independently. Existing simulation results are never
overwritten. The allocation tool and nvidia-smi measure different memory scopes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from fast_electron_cone_benchmark import Setup, POWER_ERG_S, input_text, run, analyze
from fast_electron_bai_benchmark import capture
from analyze_fast_electron_scaling_fix import physical_checks
from fast_electron_cone_physics_audit import fields_checkpoint


def monitor(path, stop):
    with path.open('w') as stream:
        stream.write('elapsed_s,index,uuid,used_mib,total_mib\n')
        start=time.monotonic()
        while not stop.is_set():
            try:
                p=subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                  '--format=csv,noheader,nounits'], capture_output=True,
                                 text=True, timeout=10)
                for line in p.stdout.splitlines():
                    stream.write(f'{time.monotonic()-start:.3f},{line}\n')
                stream.flush()
            except subprocess.TimeoutExpired:
                pass
            stop.wait(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=7200)
    args=parser.parse_args()
    exe,tool,profiles,cases_path,output=[p.resolve() for p in (
        args.exe,args.tool,args.profiles,args.cases,args.output)]
    cases=json.loads(cases_path.read_text())
    output.mkdir(parents=True,exist_ok=False)
    launcher=['mpiexec','--oversubscribe','--bind-to','none','-n','2']
    metadata={'exe':str(exe),'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
              'tool':str(tool),'tool_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),
              'cases':cases,'cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
              'job':os.environ.get('SLURM_JOB_ID'),'hostname':capture(['hostname']),
              'gpu':capture(['nvidia-smi','-q']),'topology':capture(['nvidia-smi','topo','-m']),
              'launcher':launcher,'ranks':2,'results_are_not_timing_samples':True,
              'profile_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in profiles.glob('*.dat')}}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    os.environ['KOKKOS_TOOLS_LIBS']=str(tool)
    for case in cases:
        name=case['source'];label=case['label'];samples=case.get('packets_per_bin',1)
        setup=Setup(**case.get('setup',{}))
        directory=output/label
        deck=output/(label+'.athinput')
        text=input_text(name,profiles/(name+'.dat'),setup,case.get('outputs',True))
        text=text.replace('packets_per_bin = 1\n',f'packets_per_bin = {samples}\n')
        deck.write_text(text)
        os.environ['KOKKOS_MEMORY_OUTPUT']=str(directory)
        stop=threading.Event()
        watcher=threading.Thread(target=monitor,args=(output/(label+'-gpu.csv'),stop))
        watcher.start()
        print('START',label,flush=True)
        try:
            result=run(exe,directory,['-i',deck],args.timeout,launcher)
        finally:
            stop.set();watcher.join()
        result.update(case=case,setup=asdict(setup),packets_per_bin=samples,
                      input_sha256=hashlib.sha256(deck.read_bytes()).hexdigest())
        memories=[json.loads((directory/f'memory-rank{rank}.json').read_text()) for rank in (0,1)]
        assert all(m['final'] and m['duplicate_allocations']==0 for m in memories)
        # The CUDA runtime's initial scratch allocation precedes tool startup.
        # Its first resize reports an unmatched 64 KiB free. Preserve it in the
        # output instead of attributing it to mesh/particle storage. Other
        # unmatched allocations indicate an accounting error and must stop us.
        for m in memories:
            assert m['untracked_deallocations'] in ([], [
                {'space':'Cuda','label':'[unlabeled]','bytes':65536}])
        assert all(any(s['space']=='Cuda' and s['peak_bytes']>0 for s in m['spaces'])
                   for m in memories), 'CUDA allocation callbacks must be present'
        result['memory']=memories
        expected=POWER_ERG_S*setup.power_scale*min(setup.duration_ps,setup.tlim_ps)*1e-12
        assert math.isclose(result['time'],setup.tlim_ps,rel_tol=2e-6)
        if expected:
            assert math.isclose(result['ledger']['injected_erg'],expected,rel_tol=1e-10)
            assert abs(result['ledger']['residual_erg'])/expected<1e-9
        if case.get('outputs',True):
            result.update(analyze(directory,setup))
            final=sorted((directory/'rst').glob('*.rst'))[-1]
            fields,check=fields_checkpoint(final)
            result['physical_checks']={**check,**physical_checks(final)}
        (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        print('DONE',label,'seconds',result['loop_s'],'peak_packets',
              result['peak_live_packets_at_macro_steps'],flush=True)
    (output/'complete.json').write_text(json.dumps({'cases':len(cases)})+'\n')


if __name__=='__main__':
    main()
