#!/usr/bin/env python3
"""Compare active double-precision collision restart states after a controlled replay."""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np


def read_restart(path):
    with path.open('rb') as stream:
        parameters, section = {}, None
        while True:
            raw = stream.readline()
            assert raw, f'Missing restart parameter terminator: {path}'
            line = raw.decode('ascii').split('#', 1)[0].strip()
            if line == '<par_end>':
                break
            if line.startswith('<') and line.endswith('>'):
                section = line[1:-1]
                parameters.setdefault(section, {})
            elif '=' in line:
                name, value = line.split('=', 1)
                parameters[section][name.strip()] = value.strip()
        blocks, root_level = struct.unpack('=ii', stream.read(8))
        geometry = struct.unpack('=9d', stream.read(72))
        mesh_indices = struct.unpack('=19i', stream.read(76))
        block_indices = struct.unpack('=19i', stream.read(76))
        time, dt = struct.unpack('=dd', stream.read(16))
        cycle, = struct.unpack('=i', stream.read(4))
        locations = np.frombuffer(stream.read(16*blocks), dtype=np.int32).reshape(blocks, 4)
        stream.read(4*blocks)  # Measured load-balancing costs are not physical state.
        bytes_per_block, = struct.unpack('=Q', stream.read(8))
        offset = stream.tell()
        stream.seek(offset+blocks*bytes_per_block)
        ages = np.frombuffer(stream.read(4*blocks), dtype=np.int32)
    assert len(ages) == blocks, f'Missing refinement ages: {path}'
    ng, nx, ny, nz = block_indices[:4]
    scalars = int(parameters['hydro']['nscalars'])
    groups = int(parameters['thermal_radiation']['n_groups'])
    labels = ['density', 'momentum_x1', 'momentum_x2', 'momentum_x3', 'total_gas_energy']
    labels += [f'passive_scalar_{i}' for i in range(scalars)]
    labels += ['ion_energy', 'electron_energy'] + [f'radiation_{i:02d}' for i in range(groups)]
    shape = (blocks, len(labels), nz+2*ng, ny+2*ng, nx+2*ng)
    assert bytes_per_block == int(np.prod(shape[1:]))*8, f'Unexpected restart format: {path}'
    fields = np.memmap(path, mode='r', dtype=np.float64, offset=offset, shape=shape)
    dimensions = sum(n > 1 for n in (nx, ny, nz))
    root_volume = math.prod(geometry[6:9])
    assert root_volume > 0
    volumes = root_volume*2.0**(-dimensions*(locations[:, 3]-root_level))
    return dict(path=str(path.resolve()), blocks=blocks, root_level=root_level,
                geometry=geometry, mesh_indices=mesh_indices, block_indices=block_indices,
                time=time, dt=dt, cycle=cycle, locations=locations, ages=ages, labels=labels,
                fields=fields, volumes=volumes, scalars=scalars,
                active=(slice(None), slice(ng, ng+nz), slice(ng, ng+ny), slice(ng, ng+nx)),
                cells=nx*ny*nz)


def compare(reference, candidate, tolerance, *, time_tolerance=0.0):
    """Compare states; optional time tolerance handles roundoff in free-running CFL steps.

    Exact time equality remains the default for controlled restart regressions.
    Field, timestep, topology and refinement-age gates are independent of this option.
    """
    if not math.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError('time_tolerance must be finite and nonnegative')
    mismatches = [key for key in ('blocks', 'root_level', 'geometry',
                                  'block_indices', 'cycle', 'labels')
                  if reference[key] != candidate[key]]
    if not math.isclose(reference['time'], candidate['time'], rel_tol=time_tolerance,
                        abs_tol=0):
        mismatches.append('time')
    # Mesh initializes only ng/nx1/nx2/nx3/is/ie/js/je/ks/ke for the global root
    # grid (mesh.cpp). Its unused coarse entries are still serialized as raw
    # RegionIndcs bytes and can differ between independent fresh starts. All
    # nineteen per-block entries are initialized and remain checked above.
    if reference['mesh_indices'][:10] != candidate['mesh_indices'][:10]:
        mismatches.append('mesh_indices')
    for key in ('locations', 'ages'):
        if not np.array_equal(reference[key], candidate[key]):
            mismatches.append(key)
    if not math.isclose(reference['dt'], candidate['dt'], rel_tol=tolerance, abs_tol=0):
        mismatches.append('dt')
    report = dict(reference=reference['path'], candidate=candidate['path'],
                  metadata_mismatches=mismatches, cycle=reference['cycle'],
                  time=reference['time'], reference_dt=reference['dt'], candidate_dt=candidate['dt'],
                  candidate_time=candidate['time'],
                  time_exactly_equal=reference['time'] == candidate['time'],
                  time_relative_difference=abs(candidate['time']-reference['time'])/
                      max(abs(reference['time']), abs(candidate['time']), 1e-300),
                  time_relative_tolerance=time_tolerance,
                  leaf_blocks=reference['blocks'])
    if mismatches:
        return dict(report, passed=False)
    count = len(reference['labels'])
    ref_sq, diff_sq = np.zeros(count), np.zeros(count)
    ref_integral, new_integral, abs_integral = np.zeros(count), np.zeros(count), np.zeros(count)
    ref_max, max_difference = np.zeros(count), np.zeros(count)
    ref_min, new_min = np.full(count, np.inf), np.full(count, np.inf)
    different = np.zeros(count, dtype=np.int64)
    active_values, total_volume = 0, 0.0
    finite = True
    for block, volume in enumerate(reference['volumes']):
        before = np.ascontiguousarray(reference['fields'][block][reference['active']]).reshape(count, -1)
        after = np.ascontiguousarray(candidate['fields'][block][candidate['active']]).reshape(count, -1)
        finite &= bool(np.isfinite(before).all() and np.isfinite(after).all())
        delta = after-before
        ref_sq += volume*np.einsum('ij,ij->i', before, before)
        diff_sq += volume*np.einsum('ij,ij->i', delta, delta)
        ref_integral += volume*before.sum(axis=1)
        new_integral += volume*after.sum(axis=1)
        abs_integral += volume*np.abs(before).sum(axis=1)
        ref_max = np.maximum(ref_max, np.abs(before).max(axis=1))
        max_difference = np.maximum(max_difference, np.abs(delta).max(axis=1))
        ref_min = np.minimum(ref_min, before.min(axis=1))
        new_min = np.minimum(new_min, after.min(axis=1))
        different += (before.view(np.uint64) != after.view(np.uint64)).sum(axis=1)
        active_values += before.size
        total_volume += volume*reference['cells']
    # The allowed solution difference is five times the production true-residual
    # tolerance. Report maxima/integrals too, so local errors cannot hide in L2.
    scale_floor = 1e-30
    relative_l2 = np.sqrt(diff_sq/np.maximum(ref_sq, total_volume*scale_floor**2))
    scaled_max = max_difference/np.maximum(ref_max, scale_floor)
    integral_error = np.abs(new_integral-ref_integral)/np.maximum(abs_integral,
                                                               total_volume*scale_floor)
    thermal_start = 5+reference['scalars']
    positive = bool(new_min[0] > 0 and new_min[4] >= 0 and
                    np.all(new_min[thermal_start:] >= 0))
    per_field = []
    for i, label in enumerate(reference['labels']):
        per_field.append(dict(field=label, volume_relative_l2=float(relative_l2[i]),
                              max_absolute_difference=float(max_difference[i]),
                              max_difference_over_reference_max=float(scaled_max[i]),
                              integral_difference_over_reference_absolute_integral=float(integral_error[i]),
                              reference_min=float(ref_min[i]), candidate_min=float(new_min[i]),
                              reference_integral=float(ref_integral[i]),
                              candidate_integral=float(new_integral[i]),
                              bitwise_different_values=int(different[i])))
    energy_indices = [4]+list(range(thermal_start+2, count))
    total_energy_error = abs(new_integral[energy_indices].sum()-ref_integral[energy_indices].sum())
    total_energy_error /= max(abs_integral[energy_indices].sum(), total_volume*scale_floor)
    passed = finite and positive and bool(np.all(relative_l2 <= tolerance) and
                                           np.all(scaled_max <= 10*tolerance) and
                                           np.all(integral_error <= tolerance))
    return dict(report, passed=passed, all_active_values_finite=finite,
                density_and_thermal_energies_nonnegative=positive,
                compared_active_float64_values=active_values,
                bitwise_identical=bool(np.all(different == 0)),
                thresholds=dict(volume_relative_l2=tolerance,
                                max_difference_over_reference_max=10*tolerance,
                                integral_difference_over_reference_absolute_integral=tolerance),
                max_volume_relative_l2=float(relative_l2.max()),
                max_scaled_difference=float(scaled_max.max()),
                total_gas_plus_radiation_energy_relative_difference=float(total_energy_error),
                fields=per_field,
                exclusions=['ghost zones', 'load-balancing timing costs', 'parameter text',
                            'unused global-root coarse index entries'],
                scope='Final-state agreement between the same-step replays; not a separate PDE residual test.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--tolerance', type=float, default=5e-7)
    parser.add_argument('--time-tolerance', type=float, default=0.0,
                        help='Relative timestamp tolerance; default requires exact equality')
    args = parser.parse_args()
    result = compare(read_restart(args.reference), read_restart(args.candidate), args.tolerance,
                     time_tolerance=args.time_tolerance)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'fields'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
