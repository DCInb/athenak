#!/usr/bin/env python3
"""Report measured allocation peaks and independent full-pulse refinements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from fast_electron_cone_benchmark import SOURCES

LABELS=['Monoenergetic','Collimated PIC','Divergent PIC']


def allocation_summary(result, directory):
    case=result['case'];setup=result['setup']
    summary=[]
    for memory in result['memory']:
        device=next(x for x in memory['spaces'] if x['space']=='Cuda')
        assert memory['duplicate_allocations']==0
        at_peak=device['at_peak']
        assert sum(a['bytes'] for a in at_peak)==device['peak_bytes']
        assert all(a['bytes']%104==0 for a in at_peak if 'packet' in a['label'])
        buffers=sum(a['bytes'] for a in at_peak if 'packet' in a['label'])
        # Source templates are an unlabeled view in this preserved executable.
        # All tested spatial rules split equally across the two source owners.
        prototypes=(48 if case['source']=='mono_collimated' else 1152)
        prototypes*=setup['radial_order']*setup['spatial_azimuths']//2*104
        if setup['power_scale']==0: prototypes=0
        if prototypes:
            assert any(a['label'] in ('', 'fe-source-templates') and a['bytes']==prototypes
                       for a in at_peak)
        runtime=sum(a['bytes'] for a in at_peak if a['label'].startswith('Kokkos::'))
        summary.append({'rank':memory['rank'],'cuda_peak_bytes':device['peak_bytes'],
                        'packet_buffers_at_cuda_peak_bytes':buffers,
                        'source_templates_at_cuda_peak_bytes':prototypes,
                        'mesh_and_field_at_cuda_peak_bytes':device['peak_bytes']-buffers-prototypes-runtime,
                        'kokkos_runtime_at_cuda_peak_bytes':runtime,
                        'host_max_rss_bytes':memory['host_max_rss_kib']*1024,
                        'untracked_deallocations':memory['untracked_deallocations']})
    with (directory.parent/(case['label']+'-gpu.csv')).open() as stream:
        samples=[r for r in csv.DictReader(stream) if r.get('index') and r.get('used_mib')]
    sampled={str(i):max(float(r['used_mib'])*2**20 for r in samples
                        if r['index'].strip()==str(i)) for i in (0,1)}
    return {'ranks':summary,'nvidia_sampled_peak_bytes_by_gpu':sampled,
            'nvidia_sampling_interval_s':1}


def remap(cube, n):
    """Conservative volume averaging between any two uniform cubic meshes."""
    m=cube.shape[0]
    assert cube.shape==(m,m,m)
    old=np.linspace(0,1,m+1);new=np.linspace(0,1,n+1)
    overlap=np.maximum(0,np.minimum(new[1:,None],old[None,1:])-
                        np.maximum(new[:-1,None],old[None,:-1]))*n
    np.testing.assert_allclose(overlap.sum(axis=1),1,atol=1e-12)
    mapped=np.einsum('ai,bj,ck,ijk->abc',overlap,overlap,overlap,cube,optimize=True)
    np.testing.assert_allclose(mapped.mean(),cube.mean(),rtol=1e-12,atol=1e-14)
    return mapped


def compare(a,b):
    fa=np.load(Path(a['directory'])/'fields.npz')
    fb=np.load(Path(b['directory'])/'fields.npz')
    n=a['nx']
    out={'source':a['source'],'coarse':a['label'],'fine':b['label'],
         'central_delta_pp':b['central_percent']-a['central_percent'],
         'box_delta_pp':b['box_percent']-a['box_percent'],
         'peak_Te_change_percent':100*(b['peak_Te_keV']/a['peak_Te_keV']-1),
         'peak_B_change_percent':100*(b['peak_B_kT']/a['peak_B_kT']-1)}
    for key,short in [('deposition_J_per_um3','deposition'),
                      ('electron_temperature_keV','Te'),('magnetic_field_kT','B')]:
        old=fa[key];new=remap(fb[key],n)
        difference=np.abs(new-old)
        if short=='deposition':
            out['deposition_L1_over_injected']=float(difference.sum()*(64/n)**3/a['injected_J'])
        elif short=='Te':
            out['Te_L1_over_temperature_rise']=float(difference.sum()/np.abs(new-1).sum())
        else:
            out['B_L1_over_fine_field']=float(difference.sum()/np.abs(new).sum())
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];memory_only=[];metadata=[]
    for root in map(Path.resolve,args.runs):
        if not args.allow_incomplete: assert (root/'complete.json').exists(),root
        metadata.append(json.loads((root/'metadata.json').read_text()))
        for line in (root/'results.jsonl').read_text().splitlines():
            r=json.loads(line);case=r['case'];s=r['setup'];directory=root/case['label']
            memory=allocation_summary(r,directory)
            common={'label':case['label'],'source':case['source'],'nx':s['nx'],
                    'packets_per_bin':r['packets_per_bin'],
                    'source_points':s['radial_order']*s['spatial_azimuths'],'seed':s['seed'],
                    'directory':str(directory),'memory':memory,
                    'peak_live_packets_at_macro_steps':r['peak_live_packets_at_macro_steps']}
            if not s['power_scale']:
                memory_only.append(common);continue
            assert r['time']==1.5 and s['duration_ps']==1 and s['tlim_ps']==1.5
            assert r['physical_checks']['divB_dx_over_B']<1e-9
            assert 0<=r['ledger']['active_erg']<=r['ledger']['injected_erg']
            np.testing.assert_allclose(r['mesh_deposited_fraction'],
                                       r['ledger']['deposited_erg']/r['ledger']['injected_erg'],
                                       rtol=1e-10,atol=1e-12)
            ledger_errors=[]
            for line in (directory/'run.log').read_text().splitlines():
                if line.startswith('# fast_electrons '):
                    d={k:float(v) for k,v in re.findall(r'([a-z_]+)=([-+eE.\d]+)',line)}
                    if d['injected_erg']:
                        ledger_errors.append(abs(d['residual_erg'])/d['injected_erg'])
            assert ledger_errors and max(ledger_errors)<1e-9
            rows.append(dict(common,box_percent=100*r['mesh_deposited_fraction'],
                             central_percent=100*r['hotspot_fraction'],peak_Te_keV=r['max_Te_keV'],
                             peak_B_kT=r['max_B_kT'],min_Ti_keV=r['physical_checks']['ranges']['Ti'][0],
                             min_ion_adiabatic_ratio=r['physical_checks']['ranges']['ion_adiabatic_ratio'][0],
                             injected_J=r['ledger']['injected_erg']/1e7,
                             emitted_packets=r['total_emitted_macro_packets'],
                             remaining_packets=int(r['ledger']['packets']),
                             remaining_energy_fraction=r['ledger']['active_erg']/r['ledger']['injected_erg'],
                             max_relative_ledger_error=max(ledger_errors),loop_s=r['loop_s'],
                             physical_checks=r['physical_checks']))
    assert len({m['exe_sha256'] for m in metadata})==1
    assert len({m['tool_sha256'] for m in metadata})==1
    if rows:
        np.testing.assert_allclose([r['injected_J'] for r in rows],rows[0]['injected_J'],rtol=1e-10)
    families={
        'packet_samples':('packets_per_bin',lambda r:r['nx']==16 and r['source_points']==64),
        'source_points':('source_points',lambda r:r['nx']==16 and r['packets_per_bin']==1),
        'mesh':('nx',lambda r:r['source_points']==64 and r['packets_per_bin']==1)}
    comparisons=[]
    for family,(key,select) in families.items():
        fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
        for index,source in enumerate(SOURCES):
            sequence=sorted([r for r in rows if r['source']==source and select(r) and r['seed']==12345],
                            key=lambda r:r[key])
            for a,b in zip(sequence,sequence[1:]):
                comparisons.append(dict(compare(a,b),family=family))
            for col,(metric,ylabel) in enumerate([
                    ('central_percent','Central deposition [% injected]'),
                    ('peak_Te_keV','Peak electron temperature [keV]'),
                    ('peak_B_kT','Peak magnetic field [kT]')]):
                ax=axes[index,col]
                ax.plot([r[key] for r in sequence],[r[metric] for r in sequence],'o-',label='Seed 12345')
                repeat=sorted([r for r in rows if r['source']==source and select(r) and r['seed']==12346],
                              key=lambda r:r[key])
                if repeat:
                    ax.scatter([r[key] for r in repeat],[r[metric] for r in repeat],marker='x',
                               color='#d37824',label='Seed 12346',zorder=4)
                ax.set(ylabel=ylabel,xlabel={'packet_samples':'Packets per source bin',
                                            'source_points':'Spatial source points',
                                            'mesh':'Cells per dimension'}[family],title=LABELS[index])
                ax.set_xticks([r[key] for r in sequence])
                ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                ax.grid(alpha=.2)
                if index==0 and col==0: ax.legend(fontsize=8)
        fig.suptitle(f'{family.replace("_"," ").capitalize()}: full 1 ps pulse, endpoint 1.5 ps, two GPUs\n'
                     'Each panel uses its own vertical scale',fontsize=12)
        for suffix in ('png','pdf'): fig.savefig(out/f'{family}_convergence.{suffix}',dpi=170)
        plt.close(fig)
    seed_comparisons=[]
    for a in rows:
        if a['seed']!=12345: continue
        matches=[b for b in rows if b['seed']==12346 and all(a[k]==b[k] for k in
                 ('source','nx','source_points','packets_per_bin'))]
        for b in matches: seed_comparisons.append(compare(a,b))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    ordered=sorted(memory_only,key=lambda r:r['nx'])
    if ordered:
        values=[max(m['mesh_and_field_at_cuda_peak_bytes'] for m in r['memory']['ranks'])/2**20
                for r in ordered]
        axes[0].plot([r['nx'] for r in ordered],values,'o-',color='#2c7850')
        for r,value in zip(ordered,values):
            axes[0].annotate(f'{value:.1f} MiB',(r['nx'],value),xytext=(0,8),
                             textcoords='offset points',ha='center',fontsize=9)
        axes[0].set(xscale='log',yscale='log',xlabel='Cells per dimension',
                    ylabel='Mesh and field arrays per GPU [MiB]',title='Particle-free mesh probes')
        axes[0].set_xticks([r['nx'] for r in ordered],[str(r['nx']) for r in ordered])
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].set_xlim(min(r['nx'] for r in ordered)/1.2,max(r['nx'] for r in ordered)*1.2)
        axes[0].set_ylim(min(values)*.6,max(values)*2)
        axes[0].grid(alpha=.2)
    baseline=[next((r for r in rows if r['source']==s and r['nx']==16 and
                    r['packets_per_bin']==1 and r['source_points']==64 and r['seed']==12345),None)
              for s in SOURCES]
    if all(baseline):
        particle=[];mesh=[];runtime=[]
        for r in baseline:
            m=max(r['memory']['ranks'],key=lambda m:m['cuda_peak_bytes'])
            particle.append((m['packet_buffers_at_cuda_peak_bytes']+m['source_templates_at_cuda_peak_bytes'])/2**30)
            mesh.append(m['mesh_and_field_at_cuda_peak_bytes']/2**30)
            runtime.append(m['kokkos_runtime_at_cuda_peak_bytes']/2**30)
        x=np.arange(3)
        axes[1].bar(x,particle,color='#d37824',label='Particle buffers + source templates')
        axes[1].bar(x,mesh,bottom=particle,color='#2c7850',label='Mesh + field arrays')
        total=np.array(particle)+mesh+runtime
        axes[1].bar(x,runtime,bottom=np.array(particle)+mesh,color='#9aabc0',label='Tracked Kokkos scratch')
        for i,v in enumerate(total): axes[1].text(i,v+.09,f'{v:.3f} GiB',ha='center',fontsize=9)
        axes[1].set(xticks=x,xticklabels=['Monoenergetic','Collimated\nPIC','Divergent\nPIC'],
                    ylabel='Peak tracked CUDA allocation per GPU [GiB]',title='16³ full-pulse references',
                    ylim=(0,max(total)*1.35))
        axes[1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Two A100 40 GiB GPUs: allocation lifetimes include buffer-growth overlap\n'
                 'Driver/context allocations are outside Kokkos tracking',fontsize=11)
    for suffix in ('png','pdf'): fig.savefig(out/f'memory_breakdown.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,(family,(key,select)) in zip(axes,families.items()):
        for source,label,color in zip(SOURCES,LABELS,['#1f77b4','#d37824','#36826d']):
            sequence=sorted([r for r in rows if r['source']==source and select(r) and
                             r['seed']==12345],key=lambda r:r[key])
            x=[r[key] for r in sequence]
            low=[min(m['cuda_peak_bytes'] for m in r['memory']['ranks'])/2**30 for r in sequence]
            high=[max(m['cuda_peak_bytes'] for m in r['memory']['ranks'])/2**30 for r in sequence]
            ax.plot(x,high,'o-',color=color,label=label)
            ax.fill_between(x,low,high,color=color,alpha=.2)
        ax.axhline(40,color='#666666',linestyle='--',linewidth=.8)
        ax.set(yscale='log',ylim=(.15,50),title=family.replace('_',' ').capitalize(),
               xlabel={'packet_samples':'Packets per source bin','source_points':'Source points',
                       'mesh':'Cells per dimension'}[family],ylabel='Peak tracked CUDA per GPU [GiB]')
        ax.set_xticks(sorted({r[key] for r in rows if select(r)}))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Two-GPU memory cost: curves show the larger rank peak; shading spans both rank peaks\n'
                 'Dashed line: 40 GiB installed per GPU; driver/context memory is outside CUDA tracking',fontsize=10)
    for suffix in ('png','pdf'): fig.savefig(out/f'memory_scaling.{suffix}',dpi=180)
    plt.close(fig)
    result={'metadata':metadata,'memory_only':memory_only,'rows':rows,
            'successive_refinements':comparisons,'seed_comparisons':seed_comparisons,
            'caveats':['Allocation peaks include simultaneous old/new storage during growth.',
                       'nvidia-smi samples include context/driver memory and can miss transient peaks.',
                       'Kokkos startup scratch has an untracked 64 KiB deallocation on some runs.',
                       'Packet multiplicity also refines emission-time quadrature and RNG sampling.',
                       'Spatial quadrature changes source locations; spectrum rows stay fixed.',
                       'Mesh refinement also changes adaptive field steps and total emitted packets.',
                       'Two seeds give a reproducibility range, not a statistical confidence interval.',
                       'The prescribed-source slab is not a validated cone/ignition reproduction.']}
    (out/'memory_convergence.json').write_text(json.dumps(result,indent=2)+'\n')
    scalar_rows=[]
    for r in memory_only+rows:
        d={k:v for k,v in r.items() if k not in ('memory','physical_checks','directory')}
        for key in ['cuda_peak_bytes','packet_buffers_at_cuda_peak_bytes','source_templates_at_cuda_peak_bytes',
                    'mesh_and_field_at_cuda_peak_bytes','host_max_rss_bytes']:
            for m in r['memory']['ranks']: d[f'{key}_rank{m["rank"]}']=m[key]
        d['nvidia_peak_bytes']=max(r['memory']['nvidia_sampled_peak_bytes_by_gpu'].values())
        scalar_rows.append(d)
    for name,items in [('memory_convergence.csv',scalar_rows),('successive_refinements.csv',comparisons),
                       ('seed_comparisons.csv',seed_comparisons)]:
        if not items: continue
        keys=list(dict.fromkeys(k for r in items for k in r))
        with (out/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(items)
    print(json.dumps({'physical_runs':len(rows),'mesh_memory_probes':len(memory_only),
                      'successive_comparisons':len(comparisons),'seed_comparisons':len(seed_comparisons)},indent=2))


if __name__=='__main__':main()
