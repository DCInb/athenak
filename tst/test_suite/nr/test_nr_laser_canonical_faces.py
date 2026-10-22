"""Nearly tangent rays must cross the exact faces used for block ownership."""
from pathlib import Path
import math
import os
import re
import subprocess

import pytest

INPUT = Path(__file__).resolve().parents[3]/'inputs/mhd/two_temperature_laser.athinput'


def run_ray(executable, directory, axis=0, sign=1, model='straight', ranks=1):
    directory.mkdir(parents=True, exist_ok=True)
    flags = ['time/integrator=rk1', 'time/tlim=1e-7', 'time/nlim=1',
             'laser/beam0_nrays=1', 'laser/absorption_coefficient=0.25',
             'laser/max_segments_per_launch=64', 'laser/max_transport_iterations=4',
             'laser/max_mpi_waves=4', 'laser/gpu_aware_mpi=false',
             'laser/model='+model, 'laser/report_diagnostics=true']
    # For the block ending at zero, fused xmin+12*dx is -1.7347e-18, while
    # its stored xmax is exactly zero. The normal probe is smaller than this
    # mismatch. This is the recorded coordinate of DCI production ray 82183.
    for d in range(3):
        normal = d == axis
        n = 336 if normal else 12
        flags += [f'mesh/nx{d+1}={n}', f'meshblock/nx{d+1}=12',
                  f'mesh/x{d+1}min={-0.6 if normal else -0.5}',
                  f'mesh/x{d+1}max={0.6 if normal else 0.5}',
                  f'laser/beam0_origin_x{d+1}={-sign*1.7347234759768071e-18 if normal else 0}',
                  f'laser/beam0_direction_x{d+1}={sign*9.058459752109943e-6 if normal else int(d==(axis+1)%3)}']
    flags += [f'output{n}/dt=-1' for n in range(1, 5)]
    command = ['mpirun', '-n', str(ranks), str(executable), '-i', str(INPUT), *flags]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=90)
    raw = result.stdout+result.stderr
    (directory/'run.log').write_text(raw)
    rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in raw.splitlines() if line.startswith('laser:')]
    return result, rows


@pytest.mark.parametrize('ranks', (1, 2))
@pytest.mark.parametrize('model', ('straight', 'refractive'))
@pytest.mark.parametrize('axis', (0, 1, 2))
@pytest.mark.parametrize('sign', (1, -1))
def test_canonical_face(tmp_path, ranks, model, axis, sign):
    executable = os.environ.get('ATHENA_LASER_TEST_EXE')
    if not executable:
        pytest.skip('Set ATHENA_LASER_TEST_EXE to a builtin AthenaK executable or wrapper')
    result, rows = run_ray(executable, tmp_path, axis, sign, model, ranks)
    assert result.returncode == 0, (tmp_path/'run.log').read_text()
    assert rows
    distance = 0.5*math.hypot(1.0, 9.058459752109943e-6)
    for row in rows:
        assert int(row['remaining_rays']) == 0
        assert int(row['reflected']) == 0
        assert math.isclose(float(row['path']), distance, rel_tol=1e-10, abs_tol=1e-12)
        incoming = float(row['launched'])
        assert math.isclose(float(row['escaped']), incoming*math.exp(-0.25*distance),
                            rel_tol=1e-10, abs_tol=1e-12)
        assert abs(float(row['residual'])) <= 1e-12
