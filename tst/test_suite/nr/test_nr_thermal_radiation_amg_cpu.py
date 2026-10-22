"""Independent equation and topology checks for the optional AMR AMG correction."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, composite_diffusion, leaf_geometry, manufactured_run,
)
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_1d_reference, assert_same_solution, moments,
    ordered_fields, run_case,
)


@pytest.fixture
def amg_executable(monkeypatch):
    executable = os.environ.get("ATHENA_AMG_TEST_EXE")
    if not executable:
        pytest.skip("Requires an AMG-enabled binary in ATHENA_AMG_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    return str(Path(executable).resolve())


def _check_solve(directory):
    log = (directory / "run.log").read_text()
    assert "AMR fallback" not in log
    assert "FATAL ERROR" not in log


@pytest.mark.parametrize("ranks", (1, 2))
def test_amg_periodic_dense_reference(tmp_path, amg_executable, ranks):
    # Periodicity leaves diffusion's constant mode in its nullspace; the positive
    # volume mass restores SPD. Compare to an independently assembled cell solve.
    directory = tmp_path / "periodic"
    outputs = run_case(directory, 1, "amr-amg", ranks=ranks, extra_flags=(
        "mesh/nx1=36", "meshblock/nx1=18", "refined_region1/x1max=0.5",
        "thermal_radiation/implicit_amr_order=1",
        "thermal_radiation/implicit_tolerance=1e-12"))
    assert_1d_reference(outputs, "periodic")
    for group in GROUPS:
        np.testing.assert_allclose(moments(outputs[-1], group)[0],
                                   moments(outputs[0], group)[0], rtol=2e-10)
    _check_solve(directory)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("dimensions,variable", ((2, True), (3, False)))
def test_amg_manufactured_equation(tmp_path, amg_executable, dimensions, variable, ranks):
    # Exercise the high-order outer equation with a genuinely coupled 3^d-cell
    # intermediate graph. The residual below does not call the solver's operator.
    directory = tmp_path / "manufactured"
    output = manufactured_run(directory, n=36, block_size=18,
                              dimensions=dimensions, variable_coefficient=variable,
                              ranks=ranks, extra_flags=(
                                  "thermal_radiation/implicit_preconditioner=amr-amg",
                                  "thermal_radiation/implicit_amr_inner_iterations=80",
                                  "thermal_radiation/implicit_gmres_restart=3"))
    owner, xyz, widths = leaf_geometry(output, 36, dimensions)
    _, rhs = _manufactured(*xyz, widths, dimensions, 0.01, variable)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    coefficient = (1.0 + 0.25 * np.cos(2.0 * np.pi * xyz[1])
                   if variable else np.ones_like(actual))
    residual = actual + 0.01 * composite_diffusion(
        actual, owner, widths, coefficient=coefficient) - rhs
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-10)
    np.testing.assert_allclose(np.dot(widths**dimensions, actual),
                               np.dot(widths**dimensions, rhs), rtol=0, atol=2e-11)
    _check_solve(directory / "solve")


@pytest.mark.parametrize("ranks", (1, 2))
def test_amg_same_count_regrid(tmp_path, amg_executable, monkeypatch, ranks):
    # Swap which half is refined without changing the leaf count. Every hierarchy
    # and coefficient cache must follow topology, not just the allocation size.
    input_file = tmp_path / "regrid.athinput"
    input_file.write_text(INPUT.read_text() +
                          "\n<thermal_radiation>\nsolver_timers = true\n")
    monkeypatch.setattr(utils, "INPUT", input_file)
    flags = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "mesh_refinement/refinement=adaptive", "mesh_refinement/refinement_interval=2",
             "mesh_refinement/max_nmb_per_rank=32", "amr_criterion0/value_max=1.5",
             "problem/dl=2", "problem/dr=1", "problem/xshock=0",
             "thermal_radiation/implicit_amr_inner_iterations=80",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "reference", 2, "amr-aggregate", extra_flags=flags)
    candidate = run_case(tmp_path / "amg", 2, "amr-amg", ranks=ranks, extra_flags=flags)
    assert_same_solution(reference, candidate)
    assert all(len(output["mb_logical"]) == 10 for output in candidate)
    assert not np.array_equal(ordered_fields(candidate[0])[0],
                              ordered_fields(candidate[-1])[0])
    before_regrid = next(output for output in candidate if output["cycle"] == 2)
    np.testing.assert_array_equal(ordered_fields(before_regrid)[0],
                                   ordered_fields(candidate[0])[0])
    for group in GROUPS:
        np.testing.assert_allclose(moments(candidate[-1], group)[0],
                                   moments(candidate[0], group)[0], rtol=2e-10)
    _check_solve(tmp_path / "amg")


@pytest.mark.parametrize("ranks", (1, 2))
def test_amg_periodic_self_neighbor(tmp_path, amg_executable, ranks):
    # A single x2 root block has periodic faces with the same block gid but
    # different aggregate endpoints. Dropping these edges changes the equation.
    flags = ("mesh/nx1=36", "mesh/nx2=18", "meshblock/nx1=18", "meshblock/nx2=18",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "time/nlim=1", "time/tlim=0.0001", "problem/shock_dir=2",
             "problem/xshock=-0.125", "problem/dr=100",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "reference", 2, "amr-aggregate", extra_flags=flags)
    candidate = run_case(tmp_path / "amg", 2, "amr-amg", ranks=ranks, extra_flags=flags)
    assert_same_solution(reference, candidate)
    for group in GROUPS:
        np.testing.assert_allclose(moments(candidate[-1], group)[0],
                                   moments(candidate[0], group)[0], rtol=2e-10)
    _check_solve(tmp_path / "amg")


def test_amg_partition_agreement(tmp_path, amg_executable):
    flags = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "thermal_radiation/implicit_tolerance=1e-12")
    serial = run_case(tmp_path / "serial", 2, "amr-amg", ranks=1, extra_flags=flags)
    parallel = run_case(tmp_path / "parallel", 2, "amr-amg", ranks=2, extra_flags=flags)
    assert_same_solution(serial, parallel)
    _check_solve(tmp_path / "parallel")


def test_amg_missing_backend_rejected(tmp_path):
    executable = os.environ.get("ATHENA_NO_AMG_TEST_EXE")
    if not executable:
        pytest.skip("Requires a build without HYPRE in ATHENA_NO_AMG_TEST_EXE")
    result = subprocess.run([str(Path(executable).resolve()), "-i", str(INPUT),
                             "mesh/nx1=36", "meshblock/nx1=18",
                             "thermal_radiation/implicit_preconditioner=amr-amg",
                             "time/nlim=0"], cwd=tmp_path, capture_output=True,
                            text=True, timeout=60, check=False)
    (tmp_path / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode != 0
    log = result.stdout + result.stderr
    assert "amr-amg" in log and "HYPRE" in log
