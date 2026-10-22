"""Distributed AMG: independent equations, boundaries, regridding and failures."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_suite.nr.test_nr_thermal_radiation_amg_cpu import amg_executable  # noqa: F401
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import composite_diffusion
from test_suite.nr.test_nr_thermal_radiation_amr_aggregate_mpicpu import aggregate_leaf_geometry
from test_suite.nr.test_nr_thermal_radiation_amr_preconditioner_mpicpu import monotone_diffusion
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, assert_1d_reference, assert_same_solution, assert_transport,
    boundary_flags, ordered_fields, run_case,
)


BLOCK18 = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
           "refined_region1/x1max=0.5", "refined_region1/x2max=0.5")


@pytest.mark.parametrize("amr_order,inner_iterations", ((1, 0), (2, 0), (2, 20)))
def test_amg_stiff_true_equation(tmp_path, amg_executable, amr_order, inner_iterations):
    # Unequal leaf volumes and a million-fold coefficient jump straddle ownership.
    flags = (*BLOCK18, "problem/dr=1000000", "problem/xshock=-0.125",
             "thermal_radiation/initial_radiation_x1=-0.125",
             "thermal_radiation/c_light=300000000", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/implicit_max_iterations=100",
             f"thermal_radiation/implicit_tolerance={1e-13 if amr_order == 1 else 2e-14}",
             f"thermal_radiation/implicit_amr_order={amr_order}",
             f"thermal_radiation/implicit_amr_inner_iterations={inner_iterations}")
    results = []
    for ranks in (1, 2):
        directory = tmp_path / f"ranks{ranks}"
        outputs = run_case(directory, 2, "amr-amg", ranks=ranks, extra_flags=flags)
        initial, final = outputs[0], outputs[-1]
        assert final["cycle"] == 1 and np.isclose(final["time"], 1e-4)
        assert len(final["mb_logical"]) == 10
        assert set(np.asarray(final["mb_logical"])[:, 3]) == {0, 1}
        assert "AMR fallback" not in (directory / "run.log").read_text()
        owner, widths = aggregate_leaf_geometry(final)
        weights = widths**2
        density = np.asarray(initial["mb_data"]["dens"]).ravel()
        for group, opacity in zip(GROUPS, (100.0, 35.0)):
            before = np.asarray(initial["mb_data"][group]).ravel()
            after = np.asarray(final["mb_data"][group]).ravel()
            assert np.all(np.isfinite(after)) and np.all(after >= 0)
            coefficient = 3e8/(3*density*opacity)
            action = (monotone_diffusion(after, owner, widths, coefficient)
                      if amr_order == 1 else composite_diffusion(
                          after, owner, widths, coefficient=coefficient))
            residual = after+1e-4*action-before
            relative = np.sqrt(np.dot(weights, residual**2)/np.dot(weights, before**2))
            assert relative < 1e-7
            np.testing.assert_allclose(np.dot(weights, after), np.dot(weights, before),
                                       rtol=2e-9, atol=2e-12)
        results.append(ordered_fields(final))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    for group in GROUPS:
        np.testing.assert_allclose(results[0][1][group], results[1][1][group],
                                   rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_amg_physical_boundary_reference(tmp_path, amg_executable, boundary):
    flags = ("mesh/nx1=36", "meshblock/nx1=18",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12", *boundary_flags(boundary))
    outputs = run_case(tmp_path / "boundary", 1, "amr-amg", ranks=2, extra_flags=flags)
    assert_1d_reference(outputs, boundary)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
@pytest.mark.parametrize("ranks", (2, 4))
def test_amg_3d_physical_faces(tmp_path, amg_executable, boundary, ranks):
    flags = [*BLOCK18, "mesh/nx3=36", "meshblock/nx3=18",
             "refined_region1/x3max=0.5", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12"]
    for axis in (1, 2, 3):
        for side, end in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x{axis}_bc=outflow",
                      f"thermal_radiation/implicit_x{axis}_{end}_boundary={boundary}",
                      f"thermal_radiation/implicit_x{axis}_{end}_value=0.1"]
    reference = run_case(tmp_path / "jacobi", 3, "jacobi", ranks=1, extra_flags=flags)
    candidate = run_case(tmp_path / "amg", 3, "amr-amg", ranks=ranks, extra_flags=flags)
    assert_same_solution(reference, candidate)


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_amg_regrid(tmp_path, amg_executable, adaptive):
    flags = (*BLOCK18, "mesh_refinement/max_nmb_per_rank=32",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", ranks=1,
                         adaptive=adaptive, extra_flags=flags)
    candidate = run_case(tmp_path / "amg", 2, "amr-amg", ranks=2,
                         adaptive=adaptive, extra_flags=flags)
    assert_transport(candidate, adaptive)
    assert_same_solution(reference, candidate)


@pytest.mark.parametrize("temperature", (1.0, 1e-5))
def test_amg_constant_mode(tmp_path, amg_executable, temperature):
    flags = (*BLOCK18, "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/c_light=300000000",
             f"thermal_radiation/initial_radiation_temperature={temperature}",
             f"thermal_radiation/initial_radiation_temperature_right={temperature}")
    outputs = run_case(tmp_path / "constant", 2, "amr-amg", ranks=2, extra_flags=flags)
    for group in GROUPS:
        np.testing.assert_array_equal(outputs[-1]["mb_data"][group],
                                      outputs[0]["mb_data"][group])


@pytest.mark.parametrize("mode", ("valid", "bad-csr", "memory-cap", "null-setup", "null-apply",
                                  "coarse-oracle"))
def test_amg_collective_adapter_failure(tmp_path, monkeypatch, mode, ranks=2):
    executable = os.environ.get("ATHENA_AMG_MPI_DRIVER")
    if not executable:
        pytest.skip("Requires thermal_radiation_hypre_mpi_driver in ATHENA_AMG_MPI_DRIVER")
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    # A rank-local validation failure must either return on all ranks or abort
    # the MPI job. The timeout catches a peer stranded in a backend collective.
    result = subprocess.run(["mpirun", "-np", str(ranks), str(Path(executable).resolve()), mode],
                            cwd=tmp_path, capture_output=True, text=True,
                            timeout=45, check=False)
    log = result.stdout + result.stderr
    (tmp_path / "run.log").write_text(log)
    if mode == "null-apply":
        assert result.returncode != 0, log
        assert "null-apply entered" in log, log
        assert "FATAL AMG ERROR" in log, log
        assert "unexpectedly returned" not in log, log
    else:
        assert result.returncode == 0, log
        assert f"PASS mode={mode}" in log, log
