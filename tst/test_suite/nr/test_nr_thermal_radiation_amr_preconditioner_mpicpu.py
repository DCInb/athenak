"""Stiff diffusion checks for the volume-weighted AMR block-coarse correction."""

import numpy as np
import pytest

from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, _run, composite_diffusion, leaf_geometry, manufactured_run,
)
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, ordered_fields, run_case,
)


def monotone_diffusion(values, owner, widths, coefficient):
    """Independent resistances and face-area sums for the monotone AMR operator."""
    area = 1.0 / owner.shape[0]
    result = np.zeros_like(values)
    for axis in range(2):
        adjacent = np.roll(owner, -1, axis=axis)
        mask = owner != adjacent
        left, right = owner[mask], adjacent[mask]
        conductance = (coefficient[left] + coefficient[right]) / (
            widths[left] + widths[right])
        unequal = widths[left] != widths[right]
        conductance[unequal] = 2.0 / (
            widths[left[unequal]] / coefficient[left[unequal]] +
            widths[right[unequal]] / coefficient[right[unequal]])
        flux = area * conductance * (values[left] - values[right])
        np.add.at(result, left, flux / widths[left]**2)
        np.add.at(result, right, -flux / widths[right]**2)
    return result


@pytest.mark.parametrize("amr_order,inner_iterations", ((1, 0), (2, 0), (2, 20)))
def test_stiff_amr_block_coarse_mpi(tmp_path, amr_order, inner_iterations):
    # A stationary material contact gives a million-fold coefficient jump.
    # dt*K/h^2 reaches 10^6, so the block-constant error needs a coarse solve.
    # Put the material/radiation jump inside the coarse region so a block
    # contains both coefficient scales. Every refined cell lies on one side
    # of x1=0; this also permits an
    # independent reconstruction of the signed quadratic subface fluxes.
    flags = ("mesh/nx1=32", "mesh/nx2=32", "problem/dr=1000000",
             "problem/xshock=-0.125", "thermal_radiation/initial_radiation_x1=-0.125",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "thermal_radiation/c_light=300000000", "time/nlim=1",
             "time/tlim=0.0001", "thermal_radiation/implicit_max_iterations=90",
             "thermal_radiation/implicit_tolerance=1e-13",
             f"thermal_radiation/implicit_amr_inner_iterations={inner_iterations}",
             f"thermal_radiation/implicit_amr_order={amr_order}")
    runs = []
    for ranks in (1, 2):
        directory = tmp_path / f"ranks{ranks}"
        outputs = run_case(directory, dimensions=2, ranks=ranks, extra_flags=flags)
        initial, final = outputs[0], outputs[-1]
        assert final["cycle"] == 1 and np.isclose(final["time"], 1e-4)
        assert set(np.asarray(final["mb_logical"])[:, 3]) == {0, 1}
        np.testing.assert_array_equal(initial["mb_logical"], final["mb_logical"])
        log = (directory / "run.log").read_text()
        assert "AMR fallback" not in log
        owner, _, widths = leaf_geometry(final, 32, 2)
        weights = widths**2
        density = np.asarray(initial["mb_data"]["dens"]).ravel()
        for group, opacity in zip(GROUPS, (100.0, 35.0)):
            before = np.asarray(initial["mb_data"][group]).ravel()
            after = np.asarray(final["mb_data"][group]).ravel()
            assert np.all(np.isfinite(after)) and np.all(after >= 0.0)
            coefficient = 3e8 / (3.0 * density * opacity)
            if amr_order == 1:
                diffusion = monotone_diffusion(after, owner, widths, coefficient)
            else:
                diffusion = composite_diffusion(after, owner, widths,
                                                coefficient=coefficient)
            residual = after + 1e-4 * diffusion - before
            rhs_norm = np.sqrt(np.dot(weights, before**2))
            # Check the actual equation relative to its RHS, independently of
            # the Krylov solver's larger initial-residual normalization.
            assert np.sqrt(np.dot(weights, residual**2)) / rhs_norm < 1e-7
            np.testing.assert_allclose(np.dot(weights, after), np.dot(weights, before),
                                       rtol=2e-9, atol=2e-12)
        runs.append(ordered_fields(final))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    for group in GROUPS:
        np.testing.assert_allclose(runs[0][1][group], runs[1][1][group],
                                   rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("dimensions,variable_coefficient", ((2, True), (3, False)))
def test_flexible_amr_manufactured_mpi(tmp_path, dimensions, variable_coefficient):
    # Three Arnoldi vectors exercise restarted flexible updates. The saved
    # inner-PCG corrections must solve the original reconstructed operator,
    # including its transverse coupling and volume-weighted MPI reductions.
    runs = []
    for ranks in (1, 2):
        directory = tmp_path / f"ranks{ranks}"
        output = manufactured_run(
            directory, dimensions=dimensions, variable_coefficient=variable_coefficient,
            extra_flags=("thermal_radiation/implicit_amr_inner_iterations=20",
                         "thermal_radiation/implicit_gmres_restart=3"), ranks=ranks)
        assert set(np.asarray(output["mb_logical"])[:, 3]) == {0, 1}
        assert "AMR fallback" not in (directory / "solve/run.log").read_text()
        owner, xyz, widths = leaf_geometry(output, 16, dimensions)
        _, rhs = _manufactured(*xyz, widths, dimensions, 0.01, variable_coefficient)
        actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0.0)
        coefficient = (1.0 + 0.25 * np.cos(2.0 * np.pi * xyz[1])
                       if variable_coefficient else np.ones_like(actual))
        residual = actual + 0.01 * composite_diffusion(
            actual, owner, widths, coefficient=coefficient) - rhs
        np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)
        weights = widths**dimensions
        np.testing.assert_allclose(np.dot(weights, actual), np.dot(weights, rhs),
                                   rtol=0, atol=2e-11)
        runs.append(ordered_fields(output))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_allclose(runs[0][1]["erad00_d"], runs[1][1]["erad00_d"],
                               rtol=0, atol=3e-10)


@pytest.mark.parametrize("ranks", (1, 2))
def test_positive_amr_projection_refines_solution(tmp_path, ranks):
    # A positive sparse RHS has a nearly empty exterior. Variable coefficients
    # permit small negative CG errors there; clipping them can increase the true
    # residual. The same frozen equation must then be solved more accurately.
    input_file = tmp_path / "positive.athinput"
    input_file.write_text(INPUT.read_text() +
                          "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    dt = 1e-6
    flags = ("mesh/nx1=16", "mesh/nx2=16", "meshblock/nx2=8",
             "refined_region1/x1min=0", "refined_region1/x1max=0.5",
             "refined_region1/x2min=-0.5", "refined_region1/x2max=0.5",
             "refined_region1/x3min=-0.5", "refined_region1/x3max=0.5",
             "time/nlim=0", f"time/tlim={dt}", f"time/initial_dt={dt}",
             "time/align_outputs=false", "thermal_radiation/n_groups=1",
             "thermal_radiation/group_bound_1=100", "thermal_radiation/c_light=3",
             "thermal_radiation/kappa_transport_0=1",
             "thermal_radiation/initial_profile=uniform",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-7")
    geometry = _run(tmp_path / "source", ["-i", str(input_file), *flags])
    checkpoint = sorted((tmp_path / "source/rst").glob("*.rst"))[-1]
    payload = bytearray(checkpoint.read_bytes())
    shape = (geometry["n_mbs"], 8, 1, 12, 12)
    per_block = int(np.prod(shape[1:])) * 8
    start = len(payload) - shape[0] * per_block
    assert np.frombuffer(payload, dtype=np.uint64, count=1, offset=start-8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    random = np.random.default_rng(0)
    state[:, 0] = 10**random.uniform(-4, 4, size=state[:, 0].shape)
    state[:, 7] = 0.0
    for _ in range(4):
        block = random.integers(shape[0])
        j, i = random.integers(2, 10), random.integers(2, 10)
        state[block, 7, 0, j, i] = 10**random.uniform(-6, 0)
    rhs = state[:, 7, 0, 2:10, 2:10].copy().ravel()
    density = state[:, 0, 0, 2:10, 2:10].copy().ravel()
    injected = tmp_path / "positive.rst"
    injected.write_bytes(payload)
    output = _run(tmp_path / "solve", ["-r", str(injected), "time/nlim=1",
                  "thermal_radiation/implicit_residual_check_interval=1"], ranks)
    log = (tmp_path / "solve/run.log").read_text()
    assert "positivity refinement: group=0 attempt=1" in log
    assert output["cycle"] == 1 and np.isclose(output["time"], dt)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0.0)
    owner, _, widths = leaf_geometry(output, 16, 2)
    weights = widths**2
    coefficient = 1.0 / density
    residual = actual + dt * monotone_diffusion(actual, owner, widths, coefficient) - rhs
    initial_residual = dt * monotone_diffusion(rhs, owner, widths, coefficient)
    scale = max(np.dot(weights, rhs**2), np.dot(weights, initial_residual**2))
    assert np.sqrt(np.dot(weights, residual**2) / scale) <= 16.0e-7
    np.testing.assert_allclose(np.dot(weights, actual), np.dot(weights, rhs),
                               rtol=2e-6, atol=1e-15)
