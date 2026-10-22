#!/usr/bin/env python3
"""Archive completed two-GPU memory/convergence evidence and reproducing scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory',type=Path,required=True)
    parser.add_argument('--convergence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    memory,convergence,output=[p.resolve() for p in (args.memory,args.convergence,args.output)]
    report=json.loads((output/'memory_convergence.json').read_text())
    assert (memory/'complete.json').is_file() and (convergence/'complete.json').is_file()
    expected=sum(len(m['cases']) for m in report['metadata'])
    assert len(report['rows'])+len(report['memory_only'])==expected
    assert all(m['ranks']==2 for m in report['metadata'])
    for metadata in report['metadata']:
        assert digest(Path(metadata['exe']))==metadata['exe_sha256']
        assert digest(Path(metadata['tool']))==metadata['tool_sha256']
    for suite,name in [(memory,'memory-runs'),(convergence,'convergence-runs')]:
        for line in (suite/'results.jsonl').read_text().splitlines():
            result=json.loads(line)
            assert digest(suite/(result['case']['label']+'.athinput'))==result['input_sha256']
            if 'physical_checks' in result:
                check=result['physical_checks']
                assert digest(root/check['path'])==check['sha256']
        for p in suite.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.athinput','.csv'):
                copy(p,output/name/p.name)
            elif p.is_dir():
                for filename in ('run.log','command.json','analysis.json','fields.npz',
                                 'memory-rank0.json','memory-rank1.json'):
                    if (p/filename).exists(): copy(p/filename,output/name/p.name/filename)
    scripts=memory.parent/'measured-scripts'
    for p in scripts.iterdir():
        copy(p,output/'measured-scripts'/p.name)
    profiles=root/'inputs/particles/cone_fast_ignition/profiles'
    for name,expected_hash in report['metadata'][0]['profile_hashes'].items():
        assert digest(profiles/name)==expected_hash
        copy(profiles/name,output/'inputs/profiles'/name)
    copy(memory.parent/'script-snapshot.json',output/'checks/script-snapshot.json')
    copy(root/'build-fast-electron-scaling-fix/source-snapshot.json',
         output/'checks/solver-source-snapshot.json')
    for p in (root/'tst/benchmarks').glob('*.py'):
        if any(key in p.name for key in ('memory_convergence','cone','bai','mpi_profile','scaling_fix')):
            copy(p,output/'reproduction'/p.name)
    copy(root/'tst/benchmarks/kokkos_memory_profile.cpp',output/'reproduction/kokkos_memory_profile.cpp')
    for p in (memory.parent/'tool-selfcheck-final').iterdir():
        if p.is_file(): copy(p,output/'checks'/p.name)
    copy(memory.parent/'packet-size.json',output/'checks/packet-size.json')
    copy(memory.parent/'packet_size.cpp',output/'reproduction/packet_size.cpp')
    for suite,prefix in [(memory,'memory'),(convergence,'convergence')]:
        job=suite.name.rsplit('-',1)[-1]
        copy(suite.parent/f'{prefix}-{job}.log',output/'checks'/f'{prefix}-{job}.log')
    validation={'full_pulse_runs':len(report['rows']),'mesh_only_memory_probes':len(report['memory_only']),
                'successive_refinement_comparisons':len(report['successive_refinements']),
                'seed_comparisons':len(report['seed_comparisons']),'gpus_per_run':2,
                'max_relative_packet_ledger_error':max(r['max_relative_ledger_error'] for r in report['rows']),
                'max_normalized_divB':max(r['physical_checks']['divB_dx_over_B'] for r in report['rows']),
                'max_cell_energy_identity_error':max(r['physical_checks']['fluid_energy_identity_error']
                                                     for r in report['rows']),
                'final_time_ps':1.5,'native_checkpoints_included':False,
                'raw_memory_directory':str(memory),'raw_convergence_directory':str(convergence)}
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    manifest=[f'{digest(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*'))
              if p.is_file() and p.name!='SHA256SUMS']
    (output/'SHA256SUMS').write_text(''.join(manifest))
    print(json.dumps(dict(validation,checksummed_files=len(manifest)),indent=2))


if __name__=='__main__':main()
