"""Optional global aggregate correction; set ATHENA_CHOLMOD_TEST_EXE to enable."""

import os

import numpy as np
import pytest

from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _run, composite_diffusion,
)
from test_suite.nr.test_nr_thermal_radiation_amr_aggregate_mpicpu import (
    aggregate_leaf_geometry,
)
from test_suite.nr.test_nr_thermal_radiation_amr_preconditioner_mpicpu import (
    monotone_diffusion,
)
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_1d_reference, assert_same_solution, assert_transport,
    boundary_flags, moments, ordered_fields, run_case,
)


@pytest.fixture(autouse=True)
def cholmod_executable(monkeypatch):
    executable = os.environ.get("ATHENA_CHOLMOD_TEST_EXE")
    if not executable:
        pytest.skip("Requires an Athena_ENABLE_CHOLMOD build in ATHENA_CHOLMOD_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)


BLOCK18 = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
           "refined_region1/x1max=0.5", "refined_region1/x2max=0.5")


def test_global_aggregate_1d_reference(tmp_path):
    flags = ("mesh/nx1=36", "meshblock/nx1=18",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12",
             *boundary_flags("vacuum"))
    outputs = run_case(tmp_path / "aggregate", 1, "amr-aggregate", ranks=2,
                       extra_flags=flags)
    assert_1d_reference(outputs, "vacuum")


@pytest.mark.parametrize("amr_order,inner_iterations", ((1, 0), (2, 0), (2, 20)))
def test_global_aggregate_stiff_mpi(tmp_path, amr_order, inner_iterations):
    # Ten leaves have unequal cell volumes and a million-fold coefficient jump.
    # Tangential child parity matters when mapping fine faces into the four
    # factor-nine aggregates in a coarse 18-square block. The equation oracle
    # reconstructs those interfaces independently from the output geometry.
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
        outputs = run_case(directory, 2, "amr-aggregate", ranks=ranks, extra_flags=flags)
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
            if amr_order == 1:
                action = monotone_diffusion(after, owner, widths, coefficient)
            else:
                action = composite_diffusion(
                    after, owner, widths, coefficient=coefficient)
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
    if amr_order == 1:
        reference = run_case(tmp_path / "jacobi", 2, "jacobi", extra_flags=(
            *flags, "thermal_radiation/implicit_max_iterations=1000"))
        topology, fields = ordered_fields(reference[-1])
        np.testing.assert_array_equal(topology, results[0][0])
        for group in GROUPS:
            np.testing.assert_allclose(fields[group], results[0][1][group],
                                       rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_global_aggregate_physical_faces(tmp_path, boundary):
    flags = [*BLOCK18, "mesh/nx3=36", "meshblock/nx3=18",
             "refined_region1/x3max=0.5", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12"]
    for axis in (1, 2, 3):
        for side, end in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x{axis}_bc=outflow",
                      f"thermal_radiation/implicit_x{axis}_{end}_boundary={boundary}",
                      f"thermal_radiation/implicit_x{axis}_{end}_value=0.1"]
    reference = run_case(tmp_path / "jacobi", 3, "jacobi", extra_flags=flags)
    candidate = run_case(tmp_path / "aggregate", 3, "amr-aggregate", ranks=2,
                         extra_flags=flags)
    assert_same_solution(reference, candidate)


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_global_aggregate_regrid(tmp_path, adaptive):
    # Rebuild symbolic structure and volume weights after refinement or derefinement,
    # then numerically refactor the two different group opacities at every step.
    flags = (*BLOCK18, "mesh_refinement/max_nmb_per_rank=32",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", adaptive=adaptive,
                         extra_flags=flags)
    candidate = run_case(tmp_path / "aggregate", 2, "amr-aggregate", ranks=2,
                         adaptive=adaptive, extra_flags=flags)
    assert_transport(candidate, adaptive)
    assert_same_solution(reference, candidate)


def test_global_aggregate_periodic_self_face(tmp_path):
    # Only one root block spans x2. Its periodic self-neighbor joins distinct
    # aggregates and must be counted once, even though both block gids agree.
    flags = (*BLOCK18, "mesh/nx2=18", "time/nlim=1", "time/tlim=0.0001",
             "problem/shock_dir=2", "problem/xshock=-0.125", "problem/dr=100",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", extra_flags=flags)
    candidate = run_case(tmp_path / "aggregate", 2, "amr-aggregate", ranks=2,
                         extra_flags=flags)
    assert_same_solution(reference, candidate)
    for group in GROUPS:
        np.testing.assert_allclose(moments(candidate[-1], group)[0],
                                   moments(candidate[0], group)[0], rtol=2e-10)


@pytest.mark.parametrize("temperature", (1.0, 1e-5))
def test_global_aggregate_constant(tmp_path, temperature):
    flags = (*BLOCK18, "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/c_light=300000000",
             f"thermal_radiation/initial_radiation_temperature={temperature}",
             f"thermal_radiation/initial_radiation_temperature_right={temperature}")
    outputs = run_case(tmp_path / "constant", 2, "amr-aggregate", ranks=2,
                       extra_flags=flags)
    for group in GROUPS:
        np.testing.assert_array_equal(outputs[-1]["mb_data"][group],
                                      outputs[0]["mb_data"][group])


def test_global_aggregate_positivity_fallback(tmp_path):
    # An impulse in the coarse cell next to refinement makes the quadratic
    # solve negative. The monotone retry changes the frozen face conductances
    # while retaining the same graph pattern, requiring numerical refactoring.
    dt, ghosts, cells = 1e-5, 2, 18
    input_file = tmp_path / "impulse.athinput"
    input_file.write_text(INPUT.read_text() +
                          "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    flags = (*BLOCK18, "time/nlim=0", f"time/initial_dt={dt}", f"time/tlim={dt}",
             "time/align_outputs=false", "thermal_radiation/n_groups=1",
             "thermal_radiation/group_bound_1=100", "thermal_radiation/c_light=3",
             "thermal_radiation/kappa_transport_0=1",
             "thermal_radiation/implicit_preconditioner=amr-aggregate",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    source = tmp_path / "source"
    initial = _run(source, ["-i", str(input_file), *flags])
    checkpoint = sorted((source / "rst").glob("*.rst"))[-1]
    payload = bytearray(checkpoint.read_bytes())
    extent = cells+2*ghosts
    shape = (initial["n_mbs"], 8, 1, extent, extent)
    per_block = int(np.prod(shape[1:])*8)
    start = len(payload)-shape[0]*per_block
    assert np.frombuffer(payload, dtype=np.uint64, count=1,
                         offset=start-8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    rhs = []
    for block, bounds in enumerate(initial["mb_geometry"]):
        width = (bounds[1]-bounds[0])/cells
        x = bounds[0]+(np.arange(extent)-ghosts+0.5)*width
        y = bounds[2]+(np.arange(extent)-ghosts+0.5)*width
        yy, xx = np.meshgrid(y, x, indexing="ij")
        values = np.full((extent, extent), 1e-12)
        values[np.isclose(xx, -0.5/36) & np.isclose(yy, 0.5/36)] = 1
        state[block, 7, 0] = values
        rhs.extend(values[ghosts:-ghosts, ghosts:-ghosts].ravel())
    injected = tmp_path / "impulse.rst"
    injected.write_bytes(payload)
    rhs = np.asarray(rhs)
    assert np.count_nonzero(rhs == 1) == 1
    results = []
    for order, ranks in ((1, 1), (2, 2)):
        directory = tmp_path / f"order{order}"
        output = _run(directory, ["-r", str(injected), "time/nlim=1",
                                  f"thermal_radiation/implicit_amr_order={order}"], ranks)
        owner, widths = aggregate_leaf_geometry(output)
        actual = np.asarray(output["mb_data"][GROUPS[0]]).ravel()
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
        action = monotone_diffusion(actual, owner, widths, np.ones_like(actual))
        np.testing.assert_allclose(actual+dt*action, rhs, rtol=0, atol=3e-10)
        np.testing.assert_allclose(np.dot(widths**2, actual), np.dot(widths**2, rhs),
                                   rtol=0, atol=2e-12)
        results.append(ordered_fields(output)[1][GROUPS[0]])
        log = (directory / "run.log").read_text()
        if order == 2:
            assert "AMR fallback: group=0 reason=negative_energy" in log
        else:
            assert "AMR fallback" not in log
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=3e-10)
