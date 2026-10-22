"""Cell AMG must solve the original diffusion equation across material jumps."""

import os
from pathlib import Path

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr.thermal_radiation_amr_utils import GROUPS, moments, run_case
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, composite_diffusion, leaf_geometry, manufactured_run,
)


@pytest.fixture
def cell_amg_executable(monkeypatch):
    executable = os.environ.get("ATHENA_AMG_TEST_EXE")
    if not executable:
        pytest.skip("Requires an AMG-enabled binary in ATHENA_AMG_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", str(Path(executable).resolve()))
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("boundary", ("periodic", "neumann", "dirichlet", "vacuum"))
def test_cell_amg_dense_material_jump(tmp_path, cell_amg_executable, ranks, boundary):
    flags = ["mesh/nx1=36", "meshblock/nx1=18",
             "refined_region1/x1min=-0.5", "refined_region1/x1max=0.5",
             "problem/dr=1000000", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/c_light=300000000",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-13",
             "thermal_radiation/implicit_max_iterations=100"]
    if boundary != "periodic":
        flags += ["mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow"]
        for side in ("inner", "outer"):
            flags += [f"thermal_radiation/implicit_x1_{side}_boundary={boundary}",
                      f"thermal_radiation/implicit_x1_{side}_value=0.1"]
    outputs = run_case(tmp_path / "jump", 1, "cell-amg", ranks=ranks,
                       extra_flags=flags)
    initial, final = outputs[0], outputs[-1]
    centers = np.concatenate([
        bounds[0]+(np.arange(18)+0.5)*(bounds[1]-bounds[0])/18
        for bounds in initial["mb_geometry"]])
    order = np.argsort(centers)
    dx = np.diff(np.sort(centers))[0]
    density = np.asarray(initial["mb_data"]["dens"]).ravel()[order]
    dt = final["time"]-initial["time"]
    for group, opacity in zip(GROUPS, (100.0, 35.0)):
        rhs = np.asarray(initial["mb_data"][group]).ravel()[order].copy()
        coefficient = 3.0e8/(3*density*opacity)
        matrix = np.eye(len(rhs))
        for i in range(len(rhs)):
            for j in (i-1, i+1):
                if boundary != "periodic" and j in (-1, len(rhs)):
                    if boundary == "dirichlet":
                        rate = 2*dt*coefficient[i]/dx**2
                        matrix[i, i] += rate
                        rhs[i] += rate*0.1
                    elif boundary == "vacuum":
                        matrix[i, i] += dt*min(coefficient[i]/dx, 1.5e8)/dx
                    continue
                j %= len(rhs)
                rate = dt*0.5*(coefficient[i]+coefficient[j])/dx**2
                matrix[i, i] += rate
                matrix[i, j] -= rate
        actual = np.asarray(final["mb_data"][group]).ravel()[order]
        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
        assert np.min(actual) >= 0
        if boundary in ("periodic", "neumann"):
            np.testing.assert_allclose(moments(final, group)[0],
                                       moments(initial, group)[0], rtol=2e-9)


@pytest.mark.parametrize("dimensions,variable", ((2, True), (3, False)))
@pytest.mark.parametrize("ranks", (1, 2))
def test_cell_amg_multidimensional_equation(tmp_path, cell_amg_executable,
                                          dimensions, variable, ranks):
    # Refining the whole root domain gives one uniform level, including periodic
    # wraps and remote MeshBlock faces. All spatial directions enter the RHS.
    output = manufactured_run(
        tmp_path / "manufactured", n=16, block_size=8, dimensions=dimensions,
        variable_coefficient=variable, ranks=ranks,
        extra_flags=("refined_region1/x1min=-0.5",
                     "thermal_radiation/implicit_preconditioner=cell-amg",
                     "thermal_radiation/implicit_amr_order=1",
                     "thermal_radiation/implicit_max_iterations=100"))
    owner, xyz, widths = leaf_geometry(output, 16, dimensions)
    _, rhs = _manufactured(*xyz, widths, dimensions, 0.01, variable)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    coefficient = (1+0.25*np.cos(2*np.pi*xyz[1]) if variable else np.ones_like(actual))
    residual = actual+0.01*composite_diffusion(
        actual, owner, widths, coefficient=coefficient)-rhs
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-10)
    np.testing.assert_allclose(np.dot(widths**dimensions, actual),
                               np.dot(widths**dimensions, rhs), rtol=0, atol=2e-11)


@pytest.mark.parametrize("ranks", (1, 2))
def test_cell_amg_equilibrium_skips_setup(tmp_path, cell_amg_executable, monkeypatch, ranks):
    deck = tmp_path / "equilibrium.athinput"
    deck.write_text(utils.INPUT.read_text()+"\n<thermal_radiation>\nsolver_timers = true\n")
    monkeypatch.setattr(utils, "INPUT", deck)
    directory = tmp_path / "equilibrium"
    outputs = run_case(directory, 1, "cell-amg", ranks=ranks, extra_flags=(
        "refined_region1/x1min=-0.5", "refined_region1/x1max=0.5",
        "thermal_radiation/initial_radiation_temperature_right=1",
        "problem/dr=1000000", "time/nlim=1", "time/tlim=0.0001"))
    log = (directory / "run.log").read_text()
    assert "# radiation cell AMG:" not in log
    assert "max_iterations=0" in log
    for group in GROUPS:
        np.testing.assert_array_equal(outputs[-1]["mb_data"][group],
                                      outputs[0]["mb_data"][group])
