#!/usr/bin/env python3
"""Check full-pulse, one-factor-at-a-time cone transport refinements.

Two resolutions measure sensitivity, not an asymptotic error bound. The field
timestep also controls source emission cadence, so that comparison includes
temporal source quadrature. This script does not claim ignition validation.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_fast_electron_scaling_fix import physical_checks
from fast_electron_cone_benchmark import ENERGY_UNIT, Setup, SOURCES
from fast_electron_cone_physics_audit import fields_checkpoint, LABELS

VARIANTS = {
    'mesh32': {'nx': 32},
    'field-time': {'field_cfl': .125},
    'fluid-time': {'hydro_cfl': .015},
    'spatial': {'radial_order': 16, 'spatial_azimuths': 16},
}


def final_state(directory):
    path = sorted((directory/'rst').glob('*.rst'))[-1]
    fields, record = fields_checkpoint(path)
    record.update(physical_checks(path))
    assert record['time_ps'] == 1.5
    maximum = 0.
    for line in (directory/'run.log').read_text().splitlines():
        if line.startswith('# fast_electrons '):
            row = {k: float(v) for k, v in re.findall(
                r'([a-z_]+)=([-+eE.\d]+)', line)}
            if row['injected_erg']:
                maximum = max(maximum, abs(row['residual_erg'])/row['injected_erg'])
    assert maximum < 1e-9
    record['max_relative_ledger_error_all_steps'] = maximum
    return fields, record


def restrict_average(cube, n):
    factor = cube.shape[0]//n
    assert cube.shape == (n*factor,)*3 and factor > 0
    return cube.reshape(n, factor, n, factor, n, factor).mean(axis=(1, 3, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Completed A/B benchmark directory with science.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs, baseline, output = args.runs.resolve(), args.baseline.resolve(), args.output.resolve()
    metadata = json.loads((runs/'metadata.json').read_text())
    reference_metadata = json.loads((baseline/'metadata.json').read_text())
    assert metadata['exe_sha256'] == reference_metadata['executables']['after']['sha256']
    records = [json.loads(line) for line in (runs/'results.jsonl').read_text().splitlines()]
    assert len(records) == len(SOURCES)*len(VARIANTS)
    assert {(r['name'], r['label']) for r in records} == {
        (name, variant) for name in SOURCES for variant in VARIANTS}
    reference_science = json.loads((baseline/'science.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    rows, validation, references = [], [], {}
    for name in SOURCES:
        reference = next(r for r in reference_science if r['name'] == name and r['ranks'] == 2)
        reference_fields, reference_record = final_state(baseline/reference['label'])
        references[name] = dict(reference_record, science=reference['science'])
        injected = reference['ledger']['injected_erg']
        for variant, changes in VARIANTS.items():
            r = next(r for r in records if (r['name'], r['label']) == (name, variant))
            expected = dict(asdict(Setup()), **changes)
            assert r['setup'] == expected, 'The comparison must change only its declared controls'
            assert r['time'] == 1.5
            # Compare at the common physical endpoint, including any surviving
            # population in the ledger; do not require or force thermalization.
            assert 0 <= r['ledger']['active_erg'] <= r['ledger']['injected_erg']
            assert r['ledger']['packets'] >= 0
            np.testing.assert_allclose(r['ledger']['injected_erg'], injected, rtol=1e-10)
            directory = runs/f'{variant}-{name}'
            fields, check = final_state(directory)
            check.update(name=name, variant=variant)
            validation.append(check)
            stats = reference['science']
            deposition = restrict_average(fields['dep'], Setup().nx)
            l1 = np.abs(deposition-reference_fields['dep']).sum()
            l1 *= (64/Setup().nx)**3*ENERGY_UNIT/injected
            row = {'source': name, 'variant': variant, **changes,
                   'whole_box_percent': 100*r['mesh_deposited_fraction'],
                   'central_cylinder_percent': 100*r['hotspot_fraction'],
                   'whole_box_delta_pp': 100*(r['mesh_deposited_fraction']-stats['mesh_deposited_fraction']),
                   'central_cylinder_delta_pp': 100*(r['hotspot_fraction']-stats['hotspot_fraction']),
                   'peak_Te_keV': r['max_Te_keV'], 'peak_B_kT': r['max_B_kT'],
                   'peak_Te_change_percent': 100*(r['max_Te_keV']/stats['max_Te_keV']-1),
                   'peak_B_change_percent': 100*(r['max_B_kT']/stats['max_B_kT']-1),
                   'minimum_Ti_keV': check['ranges']['Ti'][0],
                   'minimum_ion_adiabatic_ratio': check['ranges']['ion_adiabatic_ratio'][0],
                   'restricted_deposition_L1_over_injected': l1,
                   'remaining_packets': int(r['ledger']['packets']),
                   'active_energy_fraction': r['ledger']['active_erg']/injected,
                   'emitted_packets': r['total_emitted_macro_packets']}
            rows.append(row)
    result = {'metadata': metadata, 'baseline_metadata': reference_metadata,
              'references': references, 'rows': rows, 'validation': validation,
              'convergence_established': False,
              'limitations': ['Two levels per control do not establish an asymptotic error bound.',
                              'Field timestep changes also refine emission-time quadrature.',
                              'Spatial source quadrature changes sample locations and packet RNG streams.',
                              'Prescribed-source DT slab; no laser generation, cone, or ignition validation.']}
    (output/'full_pulse_sensitivity.json').write_text(json.dumps(result, indent=2)+'\n')
    keys = list(dict.fromkeys(key for r in rows for key in r))
    with (output/'full_pulse_sensitivity.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    variables = [('central_cylinder_delta_pp', 'Central deposition change [percentage points]'),
                 ('peak_Te_change_percent', 'Peak electron temperature change [%]'),
                 ('peak_B_change_percent', 'Peak magnetic field change [%]')]
    colors = ['#1f77b4', '#d37824', '#36826d']
    for ax, (key, label) in zip(axes, variables):
        for i, name in enumerate(SOURCES):
            values = [next(r[key] for r in rows if r['source'] == name and r['variant'] == v)
                      for v in VARIANTS]
            ax.bar(np.arange(4)+(i-1)*.24, values, width=.24, color=colors[i], label=LABELS[i])
        ax.axhline(0, color='black', linewidth=.6)
        ax.set_xticks(np.arange(4), ['Mesh\n16³ → 32³', 'Field step\nCFL ÷ 2',
                                     'Fluid step\nCFL ÷ 2', 'Source points\n64 → 256'])
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Full 1 ps pulse; transport to 1.5 ps; one control changed at a time\n'
                 'Same executable, two GPUs, equal injected energy; sensitivity is not an error bound', fontsize=11)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'full_pulse_sensitivity.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'completed': len(rows), 'max_ledger_error': max(
        r['max_relative_ledger_error_all_steps'] for r in validation)}, indent=2))


if __name__ == '__main__':
    main()
