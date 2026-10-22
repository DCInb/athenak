#!/usr/bin/env python3
"""Temperature/pressure/current audit of the cone-source slab adaptation.

The cited papers publish pressure/current/field maps, not a thermal-temperature
reference for this slab. This script does not digitize pressure as temperature.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from fast_electron_bai_benchmark import Checkpoint
from fast_electron_cone_benchmark import KEV_ERG, NE_PER_GRAM, PRESSURE_UNIT, SOURCES

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tst'))
from test_suite.nr.test_nr_fast_electron_hybrid_cpu import binary_blocks

BUNIT = math.sqrt(4*math.pi*PRESSURE_UNIT)
JUNIT = 2.99792458e10*BUNIT/(4*math.pi*1e-4)*3.33564095198152e-6
LABELS = ['Mono, collimated', 'PIC, collimated', 'PIC, divergent']


def fields_checkpoint(path):
    cp = Checkpoint(path)
    n = 2*int(cp.ind[1])
    fields = {k: np.zeros((n,n,n)) for k in
              ['rho','ui','ue','bx','by','bz','jx','jy','jz','dep','ohmic']}
    for gid, loc in enumerate(cp.logical):
        u,b = cp.fluid(gid)
        u = u[(slice(None),*cp.active)]
        d = cp.cell_data[gid][(slice(None),*cp.active)]
        bc = [0.5*(np.take(a,range(a.shape[2-k]-1),axis=2-k)+
                   np.take(a,range(1,a.shape[2-k]),axis=2-k))[cp.active]
              for k,a in enumerate(b)]
        m = n//2
        i,j,k = loc[:3]*m
        sl = np.s_[k:k+m,j:j+m,i:i+m]
        for key,value in zip(fields, [u[0],u[5],u[6],*bc,*d[5:8],d[1],d[4]]):
            fields[key][sl] = value
    return convert(fields, cp.time, path)


def fields_binary(fluid, fast):
    blocks = binary_blocks(fluid)
    n = 2*blocks[0][3]['dens'].shape[-1]
    fields = {k:np.zeros((n,n,n)) for k in
              ['rho','ui','ue','bx','by','bz','jx','jy','jz','dep','ohmic']}
    for (loc,_,_,u), (floc,_,_,d) in zip(blocks,binary_blocks(fast)):
        np.testing.assert_array_equal(loc,floc)
        m = n//2
        i,j,k = loc[:3]*m
        sl = np.s_[k:k+m,j:j+m,i:i+m]
        values = [u[key] for key in ['dens','eion_d','eele_d','bcc1','bcc2','bcc3']]
        values += [d[key] for key in ['fe_j1','fe_j2','fe_j3','fe_energy','fe_ohmic']]
        for key,value in zip(fields,values): fields[key][sl] = value
    header = fluid.read_bytes()[:500].decode(errors='ignore')
    import re
    time = float(re.search(r'time\s*=\s*([-+\d.eE]+)',header)[1])
    fields, record = convert(fields,time,fluid)
    record['fast_fields_path'] = str(fast.resolve().relative_to(ROOT))
    record['fast_fields_sha256'] = hashlib.sha256(fast.read_bytes()).hexdigest()
    return fields, record


def convert(f,time,path):
    path = path.resolve()
    f['Te'] = (2/3)*f['ue']*PRESSURE_UNIT/(f['rho']*NE_PER_GRAM*KEV_ERG)
    f['Ti'] = (2/3)*f['ui']*PRESSURE_UNIT/(f['rho']*NE_PER_GRAM*KEV_ERG)
    f['Pi'] = (2/3)*f['ui']*PRESSURE_UNIT/1e15  # Gbar
    f['B'] = np.sqrt(sum(f[k]**2 for k in ['bx','by','bz']))*BUNIT/1e7  # kT
    f['J'] = np.sqrt(sum(f[k]**2 for k in ['jx','jy','jz']))*JUNIT  # A/m^2
    f['ion_adiabatic_ratio'] = f['Ti']/(f['rho']/450)**(2/3)
    for value in f.values(): assert np.isfinite(value).all()
    for key in ['rho','Te','Ti','Pi']: assert np.min(f[key]) > 0
    record = {'time_ps':time, 'path':str(path.relative_to(ROOT)),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'ranges':{key:[float(np.min(f[key])),float(np.max(f[key]))]
                        for key in ['rho','Te','Ti','Pi','B','J','ion_adiabatic_ratio']}}
    return f,record


def savefig(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def plot_final(cases,out):
    fig,axes = plt.subplots(3,4,figsize=(13,9),layout='constrained')
    specs = [('Te','Electron temperature [keV]',1,6,'inferno'),
             ('Ti','Ion temperature [keV]',0.8,1.15,'viridis'),
             ('Pi','Ion pressure [Gbar]',140,200,'viridis'),
             ('B','Magnetic magnitude [kT]',0,10,'magma')]
    for row,((f,_),label) in enumerate(zip(cases,LABELS)):
        for col,(key,title,lo,hi,cmap) in enumerate(specs):
            n=f[key].shape[0]
            # Average the two cells bracketing z=0. Identical physical plane in all cases.
            a=f[key][n//2-1:n//2+1].mean(axis=0)
            im=axes[row,col].imshow(a,origin='lower',extent=[0,64,-32,32],
                                   cmap=cmap,vmin=lo,vmax=hi,aspect='equal')
            if row==0: axes[row,col].set_title(title,fontsize=10)
            if col==0: axes[row,col].set_ylabel(label+'\ny [µm]')
            axes[row,col].set_xlabel('x [µm]')
            fig.colorbar(im,ax=axes[row,col],shrink=.75)
    fig.suptitle('Archived slab benchmark at 1.5 ps: 16³ cells, 1 ps pulse, 479.755 J\n'
                 'Paper thermal-temperature reference is unavailable; these are not validated target predictions.',fontsize=12)
    savefig(fig,out,'temperature_pressure_fields')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for (f,_),label in zip(cases,LABELS):
        n=f['Te'].shape[0];x=(np.arange(n)+.5)*64/n
        for ax,key in zip(axes,['Te','Ti','Pi']):
            ax.plot(x,f[key][n//2-1:n//2+1,n//2-1:n//2+1].mean(axis=(0,1)),label=label)
    for ax,label in zip(axes,['Te [keV]','Ti [keV]','Pi [Gbar]']):
        ax.set(xlabel='x [µm]',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[0].axhline(.1,color='gray',ls=':',label='Paper initial temperature')
    axes[1].axhline(.1,color='gray',ls=':')
    fig.suptitle('Central four-cell averages; dotted line = paper initial T = 0.1 keV (not its final temperature)')
    savefig(fig,out,'axial_temperature_pressure')


def plot_midpulse(cases,out):
    fig,axes=plt.subplots(3,2,figsize=(9,10),layout='constrained')
    for row,((f,rec),label) in enumerate(zip(cases,LABELS)):
        n=f['Te'].shape[0]
        for col,(key,title,lo,hi) in enumerate([
                ('Pi','Ion pressure [Gbar]',1.5,1500),('J','Fast current magnitude [A/m²]',5e15,1e18)]):
            a=f[key][n//2-1:n//2+1].mean(axis=0)
            im=axes[row,col].imshow(np.maximum(a,lo*1e-10),origin='lower',extent=[0,64,-32,32],
                                   cmap='viridis',norm=LogNorm(lo,hi),aspect='equal')
            axes[row,col].set(xlabel='x [µm]',ylabel='y [µm]',title=f'{label}, {rec["time_ps"]:g} ps\n{title}')
            fig.colorbar(im,ax=axes[row,col],shrink=.8,extend='both')
    fig.suptitle('Slab snapshots at mid-pulse; color limits match Strozzi (2012), Fig. 17\n'
                 'Its geometry, 10 ps time and 158/317 kJ pulses differ; no quantitative error metric is justified.',fontsize=12)
    savefig(fig,out,'midpulse_pressure_current')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--midpulse',type=Path)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cases=[];fine=[]
    for name in SOURCES:
        cases.append(fields_checkpoint(sorted((ROOT/'build-fast-electron-cone/final-complete'/
                                               (name+'-science')/'rst').glob('*.rst'))[-1]))
        fine.append(fields_checkpoint(sorted((ROOT/'build-fast-electron-cone/refined-final'/
                                              ('refined-'+name)/'rst').glob('*.rst'))[-1]))
        for tag,(fields,_) in [('coarse',cases[-1]),('refined',fine[-1])]:
            np.savez_compressed(out/(name+'-'+tag+'-fields.npz'),**fields)
    plot_final(cases,out)
    report={'verdict':'Not validated as a reproduction of either cited paper; numerically consistent slab adaptation.',
            'references':['https://arxiv.org/abs/1205.1594','https://arxiv.org/abs/1111.5089'],
            'temperature_reference':'Neither paper supplies a corresponding thermal-temperature map. Fig. 17 is ion pressure, not Te or Ti.',
            'initial_temperature_keV':{'benchmark':1.0,'paper':0.1},
            'initial_central_ion_pressure_Gbar':{'benchmark':450*NE_PER_GRAM*KEV_ERG/1e15,
                                               'paper_ideal_DT_estimate':450*NE_PER_GRAM*KEV_ERG*.1/1e15},
            'source_radial_profile':{'benchmark':'exp[-(r/18um)^8]',
                                     'long_paper':'exp[-0.5*(r/18um)^8]',
                                     'paper_to_benchmark_effective_area_ratio':2**.25},
            'unit_conversions':{'pressure_unit_dyn_cm2':PRESSURE_UNIT,'B_unit_gauss':BUNIT,
                                'current_code_to_A_m2':JUNIT},
            'array_units':{'rho':'g/cm^3','Te':'keV','Ti':'keV','Pi':'Gbar',
                           'B':'kT','J':'A/m^2','ui,ue,dep,ohmic':'code energy density',
                           'bx,by,bz,jx,jy,jz':'code fields','array_order':'z,y,x'},
            'coarse':{n:r for n,(_,r) in zip(SOURCES,cases)},
            'joint_mesh_source_refinement':{n:r for n,(_,r) in zip(SOURCES,fine)}}
    if args.midpulse:
        middle=[]
        for name in SOURCES:
            directory=args.midpulse/name/'bin'
            middle.append(fields_binary(sorted(directory.glob('*.fluid.*.bin'))[1],
                                        sorted(directory.glob('*.fast.*.bin'))[1]))
            assert abs(middle[-1][1]['time_ps']-.5)<1e-12
            np.savez_compressed(out/(name+'-midpulse-fields.npz'),**middle[-1][0])
        report['midpulse']={n:r for n,(_,r) in zip(SOURCES,middle)}
        plot_midpulse(middle,out)
    (out/'physics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
