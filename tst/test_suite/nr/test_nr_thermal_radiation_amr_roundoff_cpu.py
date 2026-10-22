"""Roundoff in reconstructed AMR fluxes with a large common radiation offset."""

import re

import numpy as np
import pytest

from test_suite.nr import test_nr_thermal_radiation_amr_accuracy_cpu as accuracy
from test_suite.nr.test_nr_thermal_radiation_cell_amg import cell_amg_executable


@pytest.mark.parametrize("background", (0.03, 1e-20))
def test_stiff_constant_amr_field(tmp_path, monkeypatch, background):
    # Expanded signed sums can manufacture a 1--2 ulp subface jump from these
    # constants. With dt*K/h^2 ~ 1e12 that exceeds the requested true residual.
    # A half-plane interface has constant coarse donors, so the difference form
    # should preserve this particular equilibrium without a Krylov iteration.
    def constant(x, y, z, width, dimensions, dt, variable_coefficient=False):
        field = np.full_like(x, background)
        return field, field

    monkeypatch.setattr(accuracy, "_manufactured", constant)
    output = accuracy.manufactured_run(
        tmp_path / "constant", dimensions=3, dt=1/1024,
        extra_flags=(f"thermal_radiation/c_light={3*2**40}",
                     "thermal_radiation/implicit_max_iterations=30",
                     "thermal_radiation/implicit_amr_inner_iterations=0"))
    values = np.asarray(output["mb_data"]["erad00_d"])
    np.testing.assert_array_equal(values, np.full_like(values, background))
    log = (tmp_path / "constant/solve/run.log").read_text()
    assert re.findall(r"^# implicit thermal radiation:.* max_iterations=(\d+)",
                      log, re.MULTILINE) == ["0"]


def test_near_constant_amr_field(tmp_path, monkeypatch):
    # Retain a resolved perturbation of the common offset. This exercises normal,
    # transverse, and mixed transverse differences with the unchanged PDE.
    original = accuracy._manufactured
    background, amplitude, coefficient, dt = 0.03, 3e-10, 1024.0, 1/1024

    def near_constant(x, y, z, width, dimensions, step, variable_coefficient=False):
        target, rhs = original(x, y, z, width, dimensions, step*coefficient)
        return background+amplitude*(target-1), background+amplitude*(rhs-1)

    monkeypatch.setattr(accuracy, "_manufactured", near_constant)
    output = accuracy.manufactured_run(
        tmp_path / "near_constant", dimensions=3, dt=dt,
        extra_flags=(f"thermal_radiation/c_light={3*coefficient}",
                     "thermal_radiation/implicit_amr_inner_iterations=20"))
    owner, xyz, widths = accuracy.leaf_geometry(output, 16, 3)
    target, rhs = near_constant(*xyz, widths, 3, dt)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    residual = actual+dt*coefficient*accuracy.composite_diffusion(
        actual, owner, widths)-rhs
    volumes = widths**3
    relative = np.sqrt(np.dot(volumes, residual**2)/np.dot(volumes, rhs**2))
    assert relative < 2e-11
    np.testing.assert_allclose(actual, target, rtol=0, atol=5e-11*background)
    np.testing.assert_allclose(np.dot(volumes, actual), np.dot(volumes, rhs),
                               rtol=0, atol=2e-12*background)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("inner", (0, 20))
def test_stiff_amr_backward_error(tmp_path, monkeypatch, cell_amg_executable,
                                 ranks, inner):
    # The resolved perturbation of b diffuses to less than an ulp of the common
    # offset. A rounded solution cannot pass a b-relative residual check, even
    # though it solves the equation to componentwise backward precision.
    background, amplitude, coefficient, dt = 0.03, 3e-10, 2.0**30, 1/1024
    original = accuracy._manufactured

    def stiff_rhs(x, y, z, width, dimensions, step, variable_coefficient=False):
        mode, _ = original(x, y, z, width, dimensions, 0.0)
        rhs = background+amplitude*(mode-1.0)
        return np.full_like(rhs, background), rhs

    monkeypatch.setattr(accuracy, "_manufactured", stiff_rhs)
    output = accuracy.manufactured_run(
        tmp_path / "stiff", n=8, block_size=4, dt=dt, ranks=ranks,
        extra_flags=(f"thermal_radiation/c_light={3*coefficient}",
                     "thermal_radiation/implicit_preconditioner=cell-amg",
                     "thermal_radiation/implicit_max_iterations=80",
                     f"thermal_radiation/implicit_amr_inner_iterations={inner}"))
    owner, xyz, widths = accuracy.leaf_geometry(output, 8, 2)
    _, rhs = stiff_rhs(*xyz, widths, 2, dt)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    volumes = widths**2

    # Independent composite matrix, with its constant mode fixed by the exact
    # volume integral. Scaling the diffusion equation and constraining the mean
    # avoids an ill-conditioned double-precision dense reference solve.
    count = actual.size
    basis = np.eye(count)
    diffusion = np.column_stack([
        accuracy.composite_diffusion(vector, owner, widths) for vector in basis])
    matrix = np.zeros((count+1, count+1))
    matrix[:count, :count] = diffusion+basis/(dt*coefficient)
    matrix[:count, count] = 1.0
    matrix[count, :count] = volumes
    forcing = np.append(rhs/(dt*coefficient), np.dot(volumes, rhs))
    expected = np.linalg.solve(matrix, forcing)[:count]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-12*background)
    np.testing.assert_allclose(np.dot(volumes, actual), np.dot(volumes, rhs),
                               rtol=0, atol=2e-12*background)

    log = (tmp_path / "stiff/solve/run.log").read_text()
    reports = [dict(re.findall(r"(\w+)=([^\s]+)", line))
               for line in log.splitlines()
               if line.startswith("# implicit radiation AMR GMRES finished:")]
    assert len(reports) == 1 and reports[0]["status"] == "converged"
    record = reports[0]
    assert 0 < int(record["iterations"]) < 80
    assert record["projected_converged"] == "1"
    assert float(record["true_relative_residual"]) > 16e-12
    assert float(record["backward_error"]) <= 1e-12
    assert "# implicit radiation AMR fallback:" not in log
