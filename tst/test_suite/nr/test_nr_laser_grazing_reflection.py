"""A face reflection must stay on its specular ray and preserve absorbing path."""
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis/python"))
from bin_convert import read_binary

INPUT = REPO / "inputs/mhd/two_temperature_laser_reflection.athinput"
# DCI ray 34049: the reflected x component is about -1.87e-6. A normal
# displacement off the face creates an artificial offset_x/abs(nx) return path.
INCIDENT = np.array([-0.6427464363680306, -0.5416996485766798, -0.5416996485766798])
NORMALS = [
    np.array([-0.9214692693730957, 0.3826506290396875, 0.06687960598377324]),
    np.array([-0.9214692694591265, 0.38265062879591083, 0.0668796061932008]),
]


def run_reflected_ray(executable, directory, axis, sign, ranks, normal_index):
    directory.mkdir()
    incident = INCIDENT.copy()
    normal = NORMALS[normal_index].copy()
    origin = np.array([0.0, 0.05, 0.025])
    bounds = np.array([[-0.05, 0.05], [-0.3, 0.3], [-0.3, 0.3]])
    for values in (incident, normal, origin, bounds):
        values[[0, axis]] = values[[axis, 0]]
    incident[axis] *= sign
    normal[axis] *= sign
    incident /= np.linalg.norm(incident)
    normal /= np.linalg.norm(normal)
    reflected = incident - 2*np.dot(incident, normal)*normal
    reflected /= np.linalg.norm(reflected)
    distance = min(((bounds[d, 1] if n > 0 else bounds[d, 0])-origin[d])/n
                   for d, n in enumerate(reflected))
    nc = (9.1093837015e-28*math.pi*(2.99792458e10)**2 /
          (4.803204712570263e-10)**2)
    gradient = 0.1*nc/1e13*normal
    # The launch density is below nc but above the oblique turning density,
    # exercising an immediate face reflection in a resolved, nonzero gradient.
    rho0 = (0.5*nc/1e13 - gradient[0]*(origin[0]-bounds[0, 0]) -
            gradient[1]*origin[1] - gradient[2]*origin[2])
    coefficient = 0.7
    flags = [
        "job/basename=grazing", "time/nlim=1", "time/tlim=1e-12",
        "time/integrator=rk1", "laser/beam0_nrays=1",
        "laser/reflection_hysteresis_fraction=0.01",
        "laser/reflection_offset_fraction=1e-5",
        "laser/max_segments_per_launch=16", "laser/max_transport_iterations=8",
        "laser/max_mpi_waves=8", f"laser/absorption_coefficient={coefficient}",
        "laser/gpu_aware_mpi=false", f"problem/rho0={rho0:.17g}",
        f"problem/density_gradient={gradient[0]:.17g}",
        f"problem/density_gradient_x2={gradient[1]:.17g}",
        f"problem/density_gradient_x3={gradient[2]:.17g}",
        "output1/dt=1e-12", "output1/double_precision_binary=true",
    ]
    for d in range(3):
        flags += [
            f"mesh/nx{d+1}=24", f"meshblock/nx{d+1}=12",
            f"mesh/x{d+1}min={bounds[d, 0]:.17g}",
            f"mesh/x{d+1}max={bounds[d, 1]:.17g}",
            f"laser/beam0_origin_x{d+1}={origin[d]:.17g}",
            f"laser/beam0_direction_x{d+1}={incident[d]:.17g}",
        ]
    blocks = {}
    for flag in flags:
        name, value = flag.split("=", 1)
        block, key = name.split("/", 1)
        blocks.setdefault(block, []).append(f"{key} = {value}")
    deck = directory / "input.athinput"
    deck.write_text(INPUT.read_text() + "".join(
        f"\n<{block}>\n" + "\n".join(lines) + "\n" for block, lines in blocks.items()))
    command = ["mpirun", "-np", str(ranks), str(executable), "-i", str(deck)]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
    log = result.stdout+result.stderr
    (directory / "run.log").write_text(log)
    assert result.returncode == 0, log
    lines = [line for line in log.splitlines() if line.startswith("laser: launched=")]
    assert lines, log
    row = dict(re.findall(r"(\w+)=([^\s]+)", lines[-1]))
    assert int(row["reflected"]) == 1
    assert int(row["max_reflections"]) == 1
    assert int(row["remaining_rays"]) == 0
    assert float(row["remaining"]) == 0
    assert float(row["residual"]) < 1e-12
    assert float(row["path"]) == pytest.approx(distance, rel=1e-11, abs=1e-12)
    escaped = math.exp(-coefficient*distance)
    assert float(row["escaped"]) == pytest.approx(escaped, rel=1e-11, abs=1e-12)
    assert float(row["deposited"]) == pytest.approx(1-escaped, rel=1e-11, abs=1e-12)
    data = read_binary(str(sorted((directory / "bin").glob("grazing*.bin"))[-1]))
    q = np.asarray(data["mb_data"]["laser_q"])
    geometry = np.asarray(data["mb_geometry"])
    total, wrong_side = 0.0, 0.0
    for m, g in enumerate(geometry):
        widths = (g[1::2]-g[::2])/np.array(q.shape[1:][::-1])
        volume = np.prod(widths)
        total += q[m].sum()*volume
        n = q.shape[3-axis]
        centers = g[2*axis] + (np.arange(n)+0.5)*widths[axis]
        indices = np.nonzero(centers*reflected[axis] < 0)[0]
        wrong_side += np.take(q[m], indices, axis=2-axis).sum()*volume
    assert total == pytest.approx(1-escaped, rel=1e-11, abs=1e-12)
    assert wrong_side == 0.0, f"Off-trajectory cells absorbed {wrong_side:.17g}"
    return q


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("sign", (-1, 1))
def test_grazing_reflection(tmp_path, ranks, axis, sign):
    executable = os.environ.get("ATHENA_LASER_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_LASER_TEST_EXE to a builtin AthenaK executable or wrapper")
    fields = [run_reflected_ray(executable, tmp_path / f"normal-{n}",
                                axis, sign, ranks, n) for n in range(2)]
    # Tiny density-gradient changes must produce comparably small changes in
    # cell heating, without an offset-induced excursion into the other half-domain.
    assert np.max(np.abs(fields[1]-fields[0]))/np.max(fields[0]) < 1e-7
