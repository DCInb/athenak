#!/usr/bin/env python3
"""Audit A/B cone timings, scientific outputs and MPI/CUDA profiles."""
import argparse
import json
from pathlib import Path
import sqlite3
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fast_electron_bai_benchmark import Checkpoint
from fast_electron_cone_benchmark import SOURCES
from fast_electron_cone_physics_audit import fields_checkpoint, LABELS, ROOT
from summarize_fast_electron_mpi_profile import summarize


def physical_checks(path):
    cp=Checkpoint(path)
    dx=64/(2*cp.ind[1]);divmax=bmax=closure=0.
    for g in range(cp.nmb):
        u,b=cp.fluid(g);u=u[(slice(None),*cp.active)]
        bc=[.5*(np.take(a,range(a.shape[2-d]-1),axis=2-d)+
                np.take(a,range(1,a.shape[2-d]),axis=2-d))[cp.active]
            for d,a in enumerate(b)]
        b2=sum(a*a for a in bc)
        bmax=max(bmax,float(np.sqrt(b2).max()))
        div=sum(np.diff(a,axis=2-d)[cp.active]/dx for d,a in enumerate(b))
        divmax=max(divmax,float(np.abs(div).max()))
        total=sum(u[d]**2 for d in (1,2,3))/(2*u[0])+.5*b2+u[5]+u[6]
        closure=max(closure,float(np.max(np.abs(total-u[4])/np.abs(u[4]))))
    normalized=divmax*dx/max(bmax,1e-100)
    assert normalized<1e-9 and closure<1e-12
    return {'divB_dx_over_B':normalized,'fluid_energy_identity_error':closure}


def profiles(job):
    out={}
    for path in sorted(job.glob('*-profile/trace-*.sqlite')):
        r=summarize(path)
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as con:
            r['halo_messages']=[dict(zip(['api','calls','bytes','min_bytes','max_bytes'],row))
                for row in con.execute('''SELECT s.value,count(*),sum(m.size),min(m.size),max(m.size)
                  FROM MPI_P2P_EVENTS m JOIN StringIds s ON m.textId=s.id GROUP BY s.value''')]
            r['halo_batch_kernels']=[dict(zip(['name','calls','seconds'],row))
                for row in con.execute('''SELECT s.value,count(*),sum(k.end-k.start)*1e-9
                  FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                  WHERE s.value LIKE '%BoundaryMPIBatch%' GROUP BY s.value''')]
        out[str(path.relative_to(job))]=r
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();job=args.job.resolve();out=args.output.resolve()
    assert (job/'complete.json').exists(), 'Wait for all runs and trace exports to finish'
    out.mkdir(parents=True,exist_ok=True)
    measurements=json.loads((job/'measurements.json').read_text())
    metadata=json.loads((job/'metadata.json').read_text())
    repeats=metadata.get('repeats',3)
    assert repeats>=3 and len(measurements)==12*repeats
    for r in measurements:
        assert r['time']==1.5 and not r['profile']
        e=r['ledger']
        assert e['packets']==e['active_erg']==0
        assert abs(e['residual_erg'])/e['injected_erg']<1e-9
    timings=[]
    for name in SOURCES:
        row={'name':name}
        for version in ['before','after']:
            samples={r:[x['loop_s'] for x in measurements if x['name']==name and
                        x['version']==version and x['ranks']==r] for r in [1,2]}
            assert all(len(v)==repeats for v in samples.values())
            t1,t2=[statistics.median(samples[r]) for r in [1,2]]
            row[version]={'one_gpu_s':t1,'two_gpu_s':t2,'speedup':t1/t2,
                          'parallel_efficiency':t1/(2*t2),'samples':samples}
        row['two_gpu_improvement']=row['before']['two_gpu_s']/row['after']['two_gpu_s']
        timings.append(row)
    science=[]
    run_science=json.loads((job/'science.json').read_text())
    for name in SOURCES:
        archived=ROOT/'build-fast-electron-cone/final-complete'/(name+'-science')
        old,old_record=fields_checkpoint(sorted((archived/'rst').glob('*.rst'))[-1])
        old_stats=json.loads((archived/'analysis.json').read_text())
        states=[];item={'name':name,'preserved_reference':old_record}
        for ranks in [1,2]:
            path=sorted((job/f'{name}-after-r{ranks}-science/rst').glob('*.rst'))[-1]
            fields,record=fields_checkpoint(path);record.update(physical_checks(path))
            record['relative_linf_vs_archived_one_gpu']={key:float(np.max(np.abs(fields[key]-old[key]))/
                     max(1e-100,np.max(np.abs(old[key])))) for key in ['rho','Te','Ti','Pi','B','dep']}
            stats=next(x['science'] for x in run_science if x['name']==name and x['ranks']==ranks)
            record['science']=stats
            record['box_change_percentage_points']=100*(stats['mesh_deposited_fraction']-old_stats['mesh_deposited_fraction'])
            record['central_change_percentage_points']=100*(stats['hotspot_fraction']-old_stats['hotspot_fraction'])
            # Declared tolerances exceed stochastic rank/order differences but are
            # small compared with the independently measured discretization changes.
            assert abs(record['box_change_percentage_points'])<.01
            assert abs(record['central_change_percentage_points'])<.02
            assert record['relative_linf_vs_archived_one_gpu']['Te']<.01
            assert record['relative_linf_vs_archived_one_gpu']['B']<.01
            states.append(fields);item[f'after_{ranks}_gpu']=record
        item['one_vs_two_gpu_relative_linf']={key:float(np.max(np.abs(states[1][key]-states[0][key]))/
                    max(1e-100,np.max(np.abs(states[0][key])))) for key in ['rho','Te','Ti','Pi','B','dep']}
        science.append(item)
    traces=profiles(job)
    if traces:
        assert len(traces)==8, 'Expected two ranks for each of four instrumented runs'
        for directory in {Path(p).parent for p in traces}:
            devices=[r['devices'] for p,r in traces.items() if Path(p).parent==directory]
            assert len(devices)==2 and all(len(d)==1 for d in devices)
            assert devices[0]!=devices[1], f'Both ranks used the same GPU: {directory}'
    result={'metadata':metadata,'timings':timings,
            'science':science,'profiles':traces,
            'profile_caveat':'API, MPI, kernel and memcpy durations overlap; never add them as an exclusive budget.'}
    (out/'scaling.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,row,label in zip(axes,timings,LABELS):
        values=[row[v][k] for v in ['before','after'] for k in ['one_gpu_s','two_gpu_s']]
        bars=ax.bar(np.arange(4),values,color=['#9aabc0','#506e94','#b1d4bd','#2c7850'])
        ax.bar_label(bars,fmt='%.2f s',fontsize=9,padding=3)
        for q,(v,r) in enumerate([(v,r) for v in ['before','after'] for r in [1,2]]):
            samples=row[v]['samples'][r]
            ax.scatter(np.full(len(samples),q),samples,color='black',s=9,zorder=3)
        ax.set(xticks=np.arange(4),xticklabels=['Old\n1 GPU','Old\n2 GPUs','New\n1 GPU','New\n2 GPUs'],
               ylabel='Evolution time [s]',title=f'{label}\n2-GPU speedup: {row["before"]["speedup"]:.2f} → {row["after"]["speedup"]:.2f}×')
        ax.set_ylim(0,max(values)*1.22)
    fig.suptitle(f'Same-node A100 comparison: median of {repeats} runs; dots show all samples\n16³ cells, identical source/physics/timestep limits; setup, mesh outputs and profiling excluded',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(out/('scaling.'+suffix),dpi=180,bbox_inches='tight')
    print(json.dumps(timings,indent=2))


if __name__=='__main__':main()
