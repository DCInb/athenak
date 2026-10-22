"""Rounding-scale density differences must not create a reflection normal."""

import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr.test_nr_laser_grazing_reflection import INPUT, read_binary


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("sense", (-1, 1))
def test_rounding_scale_gradient(tmp_path, monkeypatch, ranks, axis, sense):
    executable = os.environ.get("ATHENA_LASER_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_LASER_TEST_EXE to a built_in_pgens binary")
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    critical = 9.1093837015e-28 * math.pi * (2.99792458e10)**2 / (4.803204712570263e-10)**2
    density = 0.5 * critical / 1e13
    origin = np.roll([-0.49, 0.2 * sense, 0.025], axis)
    direction = np.roll([0.8, 0.6 * sense, 0.0], axis)
    fields = []
    for label, perturbation in (("flat", 0.0), ("rounding", 4 * np.finfo(float).eps)):
        directory = tmp_path / label
        directory.mkdir()
        gradient = np.roll([0.0, sense * perturbation * density * 24, 0.0], axis)
        blocks = {
            "time": {"nlim": 1, "tlim": 1e-20},
            "mesh": {}, "meshblock": {},
            "problem": {"rho0": density, "density_gradient": gradient[0],
                        "density_gradient_x2": gradient[1],
                        "density_gradient_x3": gradient[2]},
            "laser": {"beam0_nrays": 1, "absorption_coefficient": 0.7,
                      "gpu_aware_mpi": "false"},
            "output1": {"dt": 1e-20, "double_precision_binary": "true"},
        }
        for d in range(3):
            blocks["mesh"].update({f"nx{d+1}": 24, f"x{d+1}min": -0.5,
                                   f"x{d+1}max": 0.5})
            blocks["meshblock"][f"nx{d+1}"] = 12
            blocks["laser"].update({f"beam0_origin_x{d+1}": origin[d],
                                    f"beam0_direction_x{d+1}": direction[d]})
        deck = directory / "input.athinput"
        deck.write_text(INPUT.read_text() + "".join(
            f"\n<{block}>\n" + "".join(f"{key} = {value}\n" for key, value in values.items())
            for block, values in blocks.items()))
        result = subprocess.run(
            ["mpirun", "-np", str(ranks), str(Path(executable).resolve()), "-i", str(deck)],
            cwd=directory, capture_output=True, text=True, timeout=60)
        log = result.stdout + result.stderr
        (directory / "run.log").write_text(log)
        assert result.returncode == 0, log
        rows = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                for line in re.findall(r"laser: [^\n]*", log)]
        assert rows
        row = rows[-1]
        # An underdense uniform plasma has no reflection normal. The straight
        # ray exits after (0.5-0.2)/0.6=0.5, independently of density rounding.
        assert int(row["reflected"]) == 0
        assert int(row["remaining_rays"]) == 0
        assert float(row["path"]) == pytest.approx(0.5, abs=1e-12)
        assert float(row["escaped"]) == pytest.approx(math.exp(-0.35), abs=1e-12)
        assert float(row["deposited"]) == pytest.approx(1 - math.exp(-0.35), abs=1e-12)
        assert float(row["residual"]) < 1e-12
        output = read_binary(str(sorted((directory / "bin").glob("*.bin"))[-1]))
        q = np.asarray(output["mb_data"]["laser_q"])
        geometry = np.asarray(output["mb_geometry"])
        volume = np.prod(geometry[:, 1::2] - geometry[:, ::2], axis=1) / 12**3
        assert np.sum(q * volume[:, None, None, None]) == pytest.approx(
            1 - math.exp(-0.35), abs=1e-12)
        fields.append(q)
    np.testing.assert_allclose(fields[0], fields[1], rtol=1e-13, atol=0)
