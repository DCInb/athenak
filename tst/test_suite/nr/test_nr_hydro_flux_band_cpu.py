"""Check compact hydro flux indexing with real transport and coarse/fine exchanges.

Use a bounds-checked built_in_pgens executable via ATHENA_FLUX_BAND_TEST_EXE.
ATHENA_FLUX_BAND_MPI_LAUNCH may select an MPI launcher, including on CUDA builds.
"""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis/python"))
from bin_convert import read_binary  # noqa: E402


def run_case(directory, dimensions, ghosts, mode, refinement="static"):
    directory.mkdir(parents=True)
    source = (REPO / "inputs/tests/thermal_radiation_amr.athinput").read_text()
    # Separate dual energy from explicit conduction/viscosity, which the solver
    # intentionally does not support together. FOFC tests ordinary hydro.
    source = source.replace("transport_integrator = implicit",
                            "transport_integrator = explicit")
    source = source.replace("c_light = 1000.0", "c_light = 1.0")
    if mode == "fofc":
        source = source.replace("two_temperature = true", "two_temperature = false")
        source = source.replace("enabled = true", "enabled = false")
    diffusion = "" if mode == "2t" else "alpha_iso = 0.01\nnu_iso = 0.01"
    extra = f"""
<units>
length_cgs = 1.0
mass_cgs = 1.0
time_cgs = 1.0
mu = 1.0
<mesh>
nghost = {ghosts}
nx1 = 16
nx2 = {16 if dimensions > 1 else 1}
nx3 = {16 if dimensions > 2 else 1}
<meshblock>
nx1 = 8
nx2 = {8 if dimensions > 1 else 1}
nx3 = {8 if dimensions > 2 else 1}
<mesh_refinement>
refinement = {refinement}
communication_buffers = dynamic
<hydro>
dual_energy = {"true" if mode == "2t" else "false"}
fofc = {"true" if mode == "fofc" else "false"}
{diffusion}
<problem>
dl = 1.0
dr = 0.8
pl = 1.0
pr = 0.8
ul = 0.2
ur = 0.2
vl = 0.1
vr = 0.1
wl = 0.05
wr = 0.05
"""
    path = directory / "case.athinput"
    path.write_text(source + extra)
    exe = Path(os.environ.get("ATHENA_FLUX_BAND_TEST_EXE", "./athena")).resolve()
    launch = shlex.split(os.environ.get("ATHENA_FLUX_BAND_MPI_LAUNCH", ""))
    result = subprocess.run(
        [*launch, str(exe), "-i", str(path)], cwd=directory,
        text=True, capture_output=True, timeout=180)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = [read_binary(str(p)) for p in sorted((directory / "bin").glob("*.bin"))]
    # The driver may also write a terminal dump of the last completed cycle.
    assert {output["cycle"] for output in outputs} == {0, 1, 2, 3}
    assert outputs[-1]["cycle"] == 3
    return outputs


def integral(output, name):
    values = np.asarray(output["mb_data"][name])
    bounds = np.asarray(output["mb_geometry"])
    volume = np.prod(bounds[:, 1::2] - bounds[:, ::2], axis=1)
    volume /= np.prod(values.shape[1:])
    return np.sum(values * volume[:, None, None, None])


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("mode", ["2t", "diffusion", "fofc"])
def test_ghost_width_independence_and_conservation(tmp_path, dimensions, mode):
    # Four/six ghosts also satisfy PLM+FOFC and the even-ghost AMR requirement.
    widths = (4, 6) if mode == "fofc" else (2, 4)
    narrow = run_case(tmp_path / "narrow", dimensions, widths[0], mode)
    wide = run_case(tmp_path / "wide", dimensions, widths[1], mode)
    assert len(narrow) == len(wide)
    assert set(np.asarray(narrow[0]["mb_logical"])[:, 3]) == {0, 1}
    for left, right in zip(narrow, wide):
        np.testing.assert_array_equal(left["mb_logical"], right["mb_logical"])
        assert left["time"] == right["time"]
        for name in left["mb_data"]:
            a = np.asarray(left["mb_data"][name])
            b = np.asarray(right["mb_data"][name])
            assert np.all(np.isfinite(a)), name
            np.testing.assert_allclose(a, b, rtol=3e-13, atol=3e-14, err_msg=name)
    conserved = ["dens", "mom1", "mom2", "mom3", "ener"]
    if mode != "fofc":
        conserved += ["erad00_d", "erad01_d"]
    for name in conserved:
        np.testing.assert_allclose(
            integral(narrow[-1], name), integral(narrow[0], name),
            rtol=3e-12, atol=3e-14, err_msg=name)
    assert not np.array_equal(narrow[0]["mb_data"]["dens"],
                              narrow[-1]["mb_data"]["dens"])


def test_dynamic_amr_with_dual_energy(tmp_path):
    outputs = run_case(tmp_path / "adaptive", 3, 4, "2t", "adaptive")
    assert len(outputs[0]["mb_logical"]) != len(outputs[-1]["mb_logical"])
    for name in ("dens", "mom1", "mom2", "mom3", "ener", "erad00_d", "erad01_d"):
        np.testing.assert_allclose(
            integral(outputs[-1], name), integral(outputs[0], name),
            rtol=3e-12, atol=3e-14, err_msg=name)
