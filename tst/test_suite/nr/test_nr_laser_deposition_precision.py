"""Overlapping ray sums must retain the precision of analytic slab absorption."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis/python"))
from bin_convert import read_binary


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("model", ("straight", "refractive"))
@pytest.mark.parametrize("integrator", ("rk1", "rk2"))
def test_overlapping_ray_precision(tmp_path, ranks, model, integrator):
    executable = os.environ.get("ATHENA_LASER_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_LASER_TEST_EXE to a builtin executable or wrapper")
    # All 50,000 rays take the same path through a uniform slab. The physical
    # answer is independent of their summation order and the MPI partition.
    dt = 1e-12
    flags = [
        "job/basename=precision", "time/nlim=1", f"time/tlim={dt}",
        f"time/integrator={integrator}", f"laser/model={model}",
        "laser/beam0_nrays=50000", "laser/beam0_radius=0",
        "output1/dt=-1", "output2/dt=-1", "output3/dt=-1",
        f"output4/dt={dt}",
    ]
    deck = tmp_path / "input.athinput"
    deck.write_text((REPO / "inputs/mhd/two_temperature_laser.athinput").read_text()
                    + "\n<output4>\ndouble_precision_binary = true\n")
    command = ["mpirun", "-np", str(ranks), executable, "-i", str(deck), *flags]
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2)+"\n")
    result = subprocess.run(command, cwd=tmp_path, text=True,
                            capture_output=True, timeout=120)
    (tmp_path / "run.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    data = read_binary(str(sorted((tmp_path / "bin").glob("precision*.bin"))[-1]))
    for m, geometry in enumerate(data["mb_geometry"]):
        q = np.asarray(data["mb_data"]["laser_q"][m]).ravel()
        energy = np.asarray(data["mb_data"]["laser_energy"][m]).ravel()
        dx = (geometry[1]-geometry[0])/q.size
        left = geometry[0] + dx*np.arange(q.size)
        expected = np.exp(-left)*(-np.expm1(-dx))/dx
        # Bounds ray propagation plus final double rounding, without an error
        # allowance proportional to the number of overlapping rays.
        np.testing.assert_allclose(q, expected, rtol=2e-14, atol=0)
        np.testing.assert_allclose(energy, dt*expected, rtol=2e-14, atol=0)
