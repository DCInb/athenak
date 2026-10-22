"""MPI partition independence of quadratic AMR radiation and positivity retries."""

import re

import numpy as np
import pytest

from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, _run, composite_diffusion, leaf_geometry, manufactured_run,
)
from test_suite.nr.thermal_radiation_amr_utils import (
    INPUT, assert_same_solution, assert_transport, ordered_fields, read_binary,
)


@pytest.mark.parametrize("dimensions", (2, 3))
def test_mpi_quadratic_composite_operator(tmp_path, dimensions):
    serial = manufactured_run(tmp_path / "serial", dimensions=dimensions)
    parallel = manufactured_run(tmp_path / "parallel", dimensions=dimensions, ranks=2)
    logical, fields = ordered_fields(serial)
    mpi_logical, mpi_fields = ordered_fields(parallel)
    np.testing.assert_array_equal(logical, mpi_logical)
    np.testing.assert_allclose(fields["erad00_d"], mpi_fields["erad00_d"],
                               rtol=0, atol=3e-11)
    owner, xyz, widths = leaf_geometry(parallel, 16, dimensions)
    _, rhs = _manufactured(*xyz, widths, dimensions, 0.01)
    actual = np.asarray(parallel["mb_data"]["erad00_d"]).ravel()
    residual = actual + 0.01 * composite_diffusion(actual, owner, widths) - rhs
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)


def test_collective_positive_retry(tmp_path):
    dt = 1e-5
    serial = manufactured_run(tmp_path / "serial", dt=dt, impulse=True)
    parallel = manufactured_run(tmp_path / "parallel", dt=dt, impulse=True, ranks=2)
    cold = manufactured_run(tmp_path / "cold", dt=dt, impulse=True,
                            extra_flags=("thermal_radiation/implicit_amr_order=1",))
    logical, fields = ordered_fields(serial)
    mpi_logical, mpi_fields = ordered_fields(parallel)
    np.testing.assert_array_equal(logical, mpi_logical)
    np.testing.assert_allclose(fields["erad00_d"], mpi_fields["erad00_d"],
                               rtol=0, atol=3e-11)
    log = (tmp_path / "parallel" / "solve" / "run.log").read_text()
    assert "AMR fallback: group=0 reason=negative_energy" in log
    assert np.all(mpi_fields["erad00_d"] >= 0)
    for directory in ("serial", "parallel"):
        log = (tmp_path / directory / "solve" / "run.log").read_text()
        match = re.search(r"AMR initial guess: group=0 selected=warm "
                          r"residual_ratio=([^\s]+)", log)
        assert match, log
        assert 0.0 <= float(match[1]) <= 1.0
    cold_logical, cold_fields = ordered_fields(cold)
    np.testing.assert_array_equal(cold_logical, mpi_logical)
    np.testing.assert_allclose(cold_fields["erad00_d"], mpi_fields["erad00_d"],
                               rtol=0, atol=3e-11)
    # Check the same low-order equation and cold-start normalization after the
    # signed high-order iterate has been used only as the initial CG guess.
    owner, xyz, widths = leaf_geometry(parallel, 16, 2)
    rhs = np.full(widths.shape, 1e-12)
    rhs[np.isclose(xyz[0], -0.5 / 16) & np.isclose(xyz[1], 0.5 / 16)] = 1.0
    actual = np.asarray(parallel["mb_data"]["erad00_d"]).ravel()
    residual = actual + dt * composite_diffusion(actual, owner, widths, 1) - rhs
    cold_residual = dt * composite_diffusion(rhs, owner, widths, 1)
    volumes = widths**2
    scale = max(np.dot(volumes, rhs**2), np.dot(volumes, cold_residual**2))
    assert np.sqrt(np.dot(volumes, residual**2) / scale) <= 16e-12
    np.testing.assert_allclose(np.dot(volumes, actual), np.dot(volumes, rhs),
                               rtol=0, atol=2e-12)


@pytest.mark.parametrize("adaptive", (False, True))
def test_mhd_high_order_transport(tmp_path, adaptive):
    text = INPUT.read_text().replace("<hydro>", "<mhd>")
    text = text.replace("rsolver = hllc", "rsolver = hlld")
    text = text.replace("hydro_u", "mhd_u")
    text = text.replace("<problem>\n", "<problem>\n"
                        "bxl = 0.001\nbyl = 0.0\nbzl = 0.0\n"
                        "bxr = 0.001\nbyr = 0.0\nbzr = 0.0\n")
    input_file = tmp_path / "mhd.athinput"
    input_file.write_text(text)
    outputs = []
    for ranks in (1, 2):
        directory = tmp_path / f"rank{ranks}"
        flags = ["mesh/nx2=16", "meshblock/nx2=8"]
        if adaptive:
            flags.append("mesh_refinement/refinement=adaptive")
        _run(directory, ["-i", str(input_file), *flags], ranks)
        dumps = [read_binary(str(path)) for path in sorted((directory / "bin").glob("*.bin"))]
        assert_transport(dumps, "derefine" if adaptive else None)
        outputs.append(dumps)
    assert_same_solution(*outputs)
