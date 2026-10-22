"""Native Kokkos multigrid against independent radiation/conduction equations.

ATHENA_NATIVE_TEST_EXE selects a built_in_pgens CPU or GPU executable; no HYPRE
or CHOLMOD is needed. ATHENA_NATIVE_MG_DRIVER selects the standalone oracle.
"""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr import test_nr_conduction_amr_cpu as conduction
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_1d_reference, assert_same_solution, assert_transport, run_case,
)


@pytest.fixture
def executable(monkeypatch):
    exe = os.environ.get("ATHENA_NATIVE_TEST_EXE")
    if not exe:
        pytest.skip("Set ATHENA_NATIVE_TEST_EXE to a built_in_pgens binary")
    for name in ("ATHENA_AMR_TEST_EXE", "ATHENA_CONDUCTION_TEST_EXE"):
        monkeypatch.setenv(name, str(Path(exe).resolve()))
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")


@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_algebraic_oracle(ranks):
    driver = os.environ.get("ATHENA_NATIVE_MG_DRIVER")
    if not driver:
        pytest.skip("Set ATHENA_NATIVE_MG_DRIVER to native_multigrid_driver")
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", str(ranks),
                             str(Path(driver).resolve())],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(re.findall(r"^native-mg cells=.* residual=", result.stdout, re.M)) == 30
    assert result.stdout.count("stiff-root cells=") == 6
    assert result.stdout.count("stiff-difference cells=") == 6
    assert result.stdout.count("stiff-schur g=") == 6
    assert result.stdout.count("weak-schur g=") == 3
    assert result.stdout.count("isolation stars=") == 4
    assert result.stdout.count("mass-reuse mass=") == 3


def test_invalid_mass_rejected_collectively():
    driver = os.environ.get("ATHENA_NATIVE_MG_DRIVER")
    if not driver:
        pytest.skip("Set ATHENA_NATIVE_MG_DRIVER to native_multigrid_driver")
    result = subprocess.run(["mpirun", "--oversubscribe", "-np", "2",
                             str(Path(driver).resolve()), "--bad-mass"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "nonpositive mass" in result.stdout + result.stderr


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("dimensions,order", ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2)))
def test_radiation_equation(tmp_path, executable, dimensions, order, ranks):
    flags = [f"thermal_radiation/implicit_amr_order={order}",
             "thermal_radiation/implicit_tolerance=1e-12"]
    reference = run_case(tmp_path / "jacobi", dimensions, "jacobi", ranks=ranks,
                         extra_flags=flags)
    native = run_case(tmp_path / "native", dimensions, "native-mg", ranks=ranks,
                      extra_flags=flags)
    assert_transport(native)
    assert_same_solution(reference, native)
    if dimensions == 1:
        assert_1d_reference(native)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("refinement", ("none", "static"))
@pytest.mark.parametrize("boundary", ("periodic", "neumann", "dirichlet", "vacuum"))
def test_radiation_boundaries(tmp_path, executable, ranks, refinement, boundary):
    flags = [f"mesh_refinement/refinement={refinement}",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12", "problem/dr=4"]
    if boundary != "periodic":
        for side, word in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x1_bc=outflow",
                      f"thermal_radiation/implicit_x1_{word}_boundary={boundary}",
                      f"thermal_radiation/implicit_x1_{word}_value=0.1"]
    reference = run_case(tmp_path / "jacobi", 1, "jacobi", ranks=ranks, extra_flags=flags)
    native = run_case(tmp_path / "native", 1, "native-mg", ranks=ranks, extra_flags=flags)
    assert_same_solution(reference, native)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_radiation_regrid(tmp_path, executable, ranks, adaptive):
    native = run_case(tmp_path / "native", 2, "native-mg", ranks=ranks,
                      adaptive=adaptive, extra_flags=("thermal_radiation/implicit_amr_order=1",))
    assert_transport(native, adaptive)


def test_radiation_equilibrium(tmp_path, executable):
    directory = tmp_path / "equilibrium"
    outputs = run_case(directory, 2, "native-mg", ranks=2, extra_flags=(
        "thermal_radiation/initial_radiation_temperature_right=1",))
    assert "max_iterations=0" in (directory / "run.log").read_text()
    for group in GROUPS:
        np.testing.assert_array_equal(outputs[-1]["mb_data"][group],
                                      outputs[0]["mb_data"][group])


@pytest.mark.parametrize("backend", ("jacobi", "native-mg"))
@pytest.mark.parametrize("limiter", ("harmonic", "larsen", "minmax", "levermore"))
def test_frozen_fld_flat_regions(tmp_path, executable, backend, limiter):
    # A step contains exactly flat plateaus and resolved gradients. The frozen
    # implicit operator must use the same continuous FLD closure in both. An
    # explicit-stability cap applied only on the plateaus changes this equation
    # and makes it discontinuous under arbitrarily small perturbations of E.
    outputs = run_case(tmp_path / "fld", 1, backend, ranks=2, extra_flags=(
        "mesh_refinement/refinement=none", "time/nlim=1", "time/tlim=0.0001",
        "thermal_radiation/kappa_transport_0=1",
        "thermal_radiation/kappa_transport_1=1",
        f"thermal_radiation/flux_limiter={limiter}",
        "thermal_radiation/implicit_tolerance=1e-12"))
    initial, final = outputs[0], outputs[-1]
    geometry = np.asarray(initial["mb_geometry"])
    cells = np.asarray(initial["mb_data"][GROUPS[0]]).shape[-1]
    centers = (geometry[:, 0, None]+(np.arange(cells)+0.5)*
               (geometry[:, 1]-geometry[:, 0])[:, None]/cells).ravel()
    order = np.argsort(centers)
    dx = 1.0/len(order)
    dt = final["time"]-initial["time"]
    for group in GROUPS:
        rhs = np.asarray(initial["mb_data"][group]).ravel()[order]
        gradient = np.abs(np.roll(rhs, -1)-np.roll(rhs, 1))/(2*dx)
        q = gradient/rhs  # sigma=alpha=1; energies are above the floor.
        if limiter == "harmonic":
            diffusion = 1/(3+q)
        elif limiter == "larsen":
            diffusion = 1/np.hypot(3, q)
        elif limiter == "minmax":
            diffusion = 1/np.maximum(3, q)
        else:
            diffusion = (2+q)/(6+3*q+q*q)
        matrix = np.eye(len(rhs))
        for i in range(len(rhs)):
            for j in ((i-1) % len(rhs), (i+1) % len(rhs)):
                weight = dt*1000*0.5*(diffusion[i]+diffusion[j])/dx**2
                matrix[i, i] += weight
                matrix[i, j] -= weight
        expected = np.linalg.solve(matrix, rhs)
        actual = np.asarray(final["mb_data"][group]).ravel()[order]
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(np.sum(actual), np.sum(rhs), rtol=2e-12)


@pytest.mark.parametrize("backend", ("jacobi", "native-mg"))
@pytest.mark.parametrize("limiter", ("harmonic", "larsen"))
@pytest.mark.parametrize("amplitude", (1e-160, 1e-250))
def test_fld_faint_gradient(tmp_path, executable, monkeypatch, backend, limiter, amplitude):
    # Rescaling E and its regularization floor leaves the FLD operator unchanged.
    # A direct sum of squared gradients spuriously makes these bands uniform.
    deck = tmp_path / "faint.athinput"
    deck.write_text(INPUT.read_text()+"\n<thermal_radiation>\nenergy_floor = 1e-30\n")
    monkeypatch.setattr("test_suite.nr.thermal_radiation_amr_utils.INPUT", deck)
    flags = ["mesh_refinement/refinement=none", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/kappa_transport_0=1",
             "thermal_radiation/kappa_transport_1=1",
             f"thermal_radiation/flux_limiter={limiter}",
             "thermal_radiation/implicit_tolerance=1e-12"]
    reference = run_case(tmp_path / "unit", 1, backend, ranks=2, extra_flags=flags)
    faint = run_case(tmp_path / "faint", 1, backend, ranks=2, extra_flags=[
        *flags, f"thermal_radiation/arad={amplitude}",
        f"thermal_radiation/energy_floor={amplitude*1e-30}"])
    for group in GROUPS:
        np.testing.assert_allclose(np.asarray(faint[-1]["mb_data"][group])/amplitude,
                                   reference[-1]["mb_data"][group], rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("backend", ("jacobi", "native-mg"))
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("amplitude", (1e-160, 1e-250))
@pytest.mark.parametrize("boundary", ("periodic", "dirichlet"))
def test_radiation_tiny_energy(tmp_path, executable, backend, order, amplitude, boundary):
    # The frozen linear equation is homogeneous in radiation amplitude. Compare
    # after rescaling: an absolute floor would silently accept an unsolved band.
    flags = [f"thermal_radiation/implicit_amr_order={order}",
             "thermal_radiation/implicit_tolerance=1e-12"]
    boundary_flags = []
    if boundary == "dirichlet":
        for side, word in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x1_bc=outflow",
                      f"thermal_radiation/implicit_x1_{word}_boundary=dirichlet"]
            boundary_flags.append(f"thermal_radiation/implicit_x1_{word}_value=0.1")
    reference = run_case(tmp_path / "unit", 2, backend, ranks=2,
                         extra_flags=flags+boundary_flags)
    faint_boundary = [flag.replace("=0.1", f"={0.1*amplitude}") for flag in boundary_flags]
    faint = run_case(tmp_path / "faint", 2, backend, ranks=2,
                     extra_flags=[*flags, *faint_boundary, f"thermal_radiation/arad={amplitude}"])
    for left, right in zip(reference, faint):
        assert left["cycle"] == right["cycle"]
        for group in GROUPS:
            np.testing.assert_allclose(np.asarray(right["mb_data"][group])/amplitude,
                                       left["mb_data"][group], rtol=2e-9, atol=2e-11)


def test_adaptive_gmres_restart(tmp_path, executable, monkeypatch):
    deck = tmp_path / "adaptive.athinput"
    deck.write_text(INPUT.read_text()+
                    "\n<thermal_radiation>\nimplicit_gmres_restart_max = 30\n")
    monkeypatch.setattr("test_suite.nr.thermal_radiation_amr_utils.INPUT", deck)
    flags = ["thermal_radiation/implicit_tolerance=1e-12",
             "thermal_radiation/kappa_transport_0=1",
             "thermal_radiation/kappa_transport_1=0.35"]
    reference = run_case(tmp_path / "reference", 2, "jacobi", ranks=2, extra_flags=flags)
    native = run_case(tmp_path / "native", 2, "native-mg", ranks=2,
                      extra_flags=[*flags, "thermal_radiation/implicit_gmres_restart=2"])
    assert_same_solution(reference, native)
    lengths = re.findall(r"GMRES restart growth: from=\d+ to=(\d+)",
                         (tmp_path / "native/run.log").read_text())
    assert lengths and max(map(int, lengths)) <= 30


@pytest.mark.parametrize("dimensions", (2, 3))
@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("orthogonalization", ("mgs2", "cgs2"))
@pytest.mark.parametrize("inner_iterations", (0, 3))
@pytest.mark.parametrize("metric", ("volume", "preconditioner"))
def test_native_preconditioner_metric(tmp_path, executable, monkeypatch,
                                      dimensions, ranks, orthogonalization, inner_iterations,
                                      metric):
    from test_suite.nr import test_nr_thermal_radiation_amr_accuracy_cpu as accuracy

    deck = tmp_path / "metric.athinput"
    deck.write_text(accuracy.INPUT.read_text()+"""
<thermal_radiation>
implicit_gmres_orthogonalization = mgs2
implicit_gmres_restart_max = 5
solver_timers = true
""")
    monkeypatch.setattr(accuracy, "INPUT", deck)
    output = accuracy.manufactured_run(
        tmp_path / "metric", n=8, block_size=4, dimensions=dimensions, ranks=ranks,
        variable_coefficient=dimensions == 2, extra_flags=(
            "thermal_radiation/implicit_preconditioner=native-mg",
            f"thermal_radiation/implicit_gmres_metric={metric}",
            f"thermal_radiation/implicit_gmres_orthogonalization={orthogonalization}",
            "thermal_radiation/implicit_gmres_restart=5",
            "thermal_radiation/implicit_gmres_restart_max=5",
            f"thermal_radiation/implicit_amr_inner_iterations={inner_iterations}",
            "thermal_radiation/implicit_report=true",
            "thermal_radiation/solver_timers=true"))
    owner, xyz, widths = accuracy.leaf_geometry(output, 8, dimensions)
    _, rhs = accuracy._manufactured(*xyz, widths, dimensions, 0.01, dimensions == 2)
    values = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    coefficient = (1+0.25*np.cos(2*np.pi*xyz[1]) if dimensions == 2
                   else np.ones_like(values))
    residual = values+0.01*accuracy.composite_diffusion(
        values, owner, widths, coefficient=coefficient)-rhs
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-10)
    np.testing.assert_allclose(np.dot(widths**dimensions, values),
                               np.dot(widths**dimensions, rhs), rtol=0, atol=2e-11)
    log = (tmp_path / "metric/solve/run.log").read_text()
    cached = int(inner_iterations == 0 and metric == "preconditioner")
    finished = re.search(r"GMRES finished: iterations=(\d+) restarts=(\d+).*"
                         rf"cached_metric={cached}.*status=converged", log)
    assert finished, log
    assert "stored_corrections=1" in finished[0]
    recycled = re.search(r"recycled_columns=(\d+)", finished[0])
    assert recycled, log
    # The independent equation above must still hold when the Arnoldi space
    # contains actual corrections carried across multiple restarts.
    assert (int(recycled[1]) > 0) == (inner_iterations == 0 and metric == "volume")
    iterations, restarts = map(int, finished.groups())
    calls = re.search(r"solver timer: rank=0 .* phase=preconditioner calls=(\d+)", log)
    assert calls, log
    if inner_iterations == 0:
        assert restarts > 1  # Exercise replacement of the cached basis at a restart.
        # One MG application per Arnoldi vector plus restart initialization;
        # the previous implementation used four per vector for this equation.
        assert int(calls[1]) <= iterations + restarts + 2
    else:
        # Truncated inner PCG is nonlinear in its RHS. Its stored corrections
        # must not be used as if they were the fixed metric's B*v vectors.
        assert int(calls[1]) > iterations + restarts


@pytest.mark.parametrize("ranks", (1, 2, 4))
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("amplitude", (1.0, 1.e-160))
def test_native_group_batch(tmp_path, executable, monkeypatch, ranks, order, amplitude):
    from test_suite.nr.thermal_radiation_amr_utils import ordered_fields

    deck = tmp_path / "five_groups.athinput"
    deck.write_text(INPUT.read_text()+"""
<thermal_radiation>
n_groups = 5
implicit_group_batch_size = 1
implicit_amg_reuse = false
group_bound_2 = 2
group_bound_3 = 4
group_bound_4 = 8
group_bound_5 = 100
kappa_transport_2 = 10
kappa_transport_3 = 350
kappa_transport_4 = 50
""")
    monkeypatch.setattr("test_suite.nr.thermal_radiation_amr_utils.INPUT", deck)
    flags = [f"thermal_radiation/implicit_amr_order={order}",
             f"thermal_radiation/arad={amplitude}"]
    scalar = run_case(tmp_path / "scalar", 2, "native-mg", ranks=ranks, extra_flags=flags)
    batched = run_case(tmp_path / "batch", 2, "native-mg", ranks=ranks,
                       extra_flags=[*flags, "thermal_radiation/implicit_group_batch_size=4"])
    assert "size=4 " in (tmp_path / "batch/run.log").read_text()
    for a, b in zip(scalar, batched):
        _, left = ordered_fields(a)
        _, right = ordered_fields(b)
        for name in left:
            scale = amplitude if name.startswith("erad") else 1.0
            np.testing.assert_allclose(right[name]/scale, left[name]/scale,
                                       rtol=2e-9, atol=2e-11)


def test_genuine_high_order_undershoot_still_retries(tmp_path, executable):
    from test_suite.nr import test_nr_thermal_radiation_amr_accuracy_cpu as accuracy

    output = accuracy.manufactured_run(
        tmp_path / "impulse", n=16, dimensions=2, dt=1.e-5, impulse=True, ranks=2,
        extra_flags=("thermal_radiation/implicit_preconditioner=native-mg",))
    log = (tmp_path / "impulse/solve/run.log").read_text()
    assert "# implicit radiation AMR fallback:" in log
    values = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    assert np.min(values) >= 0
    owner, xyz, widths = accuracy.leaf_geometry(output, 16, 2)
    rhs = np.full_like(values, 1.e-12)
    rhs[np.isclose(xyz[0], -0.5/16) & np.isclose(xyz[1], 0.5/16)] = 1
    residual = values+1.e-5*accuracy.composite_diffusion(values, owner, widths, order=1)-rhs
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3.e-10)


@pytest.mark.parametrize("ranks", (2, 4))
def test_native_batched_positivity_resumes(tmp_path, executable, monkeypatch, ranks):
    from test_suite.nr import test_nr_thermal_radiation_group_batch_mpicpu as batching
    from test_suite.nr import test_nr_thermal_radiation_amr_cholmod_mpicpu as runner

    deck = tmp_path / "batch.athinput"
    deck.write_text(INPUT.read_text()+
                    "\n<thermal_radiation>\nimplicit_group_batch_size = 2\n")
    original = runner._run

    def native_run(directory, flags, ranks=1):
        flags = [flag.replace("implicit_preconditioner=amr-amg",
                              "implicit_preconditioner=native-mg").replace(
                                  "implicit_amr_inner_iterations=20",
                                  "implicit_amr_inner_iterations=0") for flag in flags]
        return original(directory, flags, ranks)

    monkeypatch.setattr(runner, "_run", native_run)
    # Independent composite equation, positivity, conservation, and an unchanged
    # neighboring group all remain mandatory after the deferred fallback.
    batching.test_batched_positivity_retry(tmp_path, executable, deck, ranks)
    log = (tmp_path / "batch/run.log").read_text()
    solved = [int(n) for n in re.findall(
        r"# implicit radiation AMR GMRES finished: iterations=(\d+)", log)]
    # Only the impulse group needs a nontrivial high-order solve. Deferring its
    # monotone retry must not solve that same high-order equation a second time.
    assert sum(n > 0 for n in solved) == 1, solved
    assert "# implicit radiation AMR resume: group=0 order=1" in log


@pytest.mark.parametrize("backend", ("jacobi", "native-mg"))
def test_stiff_three_dimensional_equilibrium(tmp_path, executable, backend):
    directory = tmp_path / "constant"
    outputs = run_case(directory, 3, backend, ranks=2, extra_flags=(
        "thermal_radiation/initial_radiation_temperature_right=1",
        "thermal_radiation/c_light=1e14",
        "thermal_radiation/implicit_max_iterations=100"))
    assert "max_iterations=0" in (directory / "run.log").read_text()
    for group in GROUPS:
        np.testing.assert_array_equal(outputs[-1]["mb_data"][group],
                                      outputs[0]["mb_data"][group])


@pytest.fixture
def native_conduction(monkeypatch, executable):
    original = conduction.run

    def native_run(directory, dimensions=1, flags=(), **kwargs):
        kwargs["extra"] = (kwargs.get("extra", "") +
                           "\n<hydro>\nconduction_preconditioner = native-mg\n")
        return original(directory, dimensions, flags, **kwargs)

    monkeypatch.setattr(conduction, "run", native_run)


@pytest.mark.parametrize("refinement", ("none", "static"))
@pytest.mark.parametrize("density", (1.0, 4.0))
@pytest.mark.parametrize("boundary", ("periodic", "neumann", "dirichlet"))
def test_conduction_dense(tmp_path, native_conduction, boundary, density, refinement):
    conduction.test_dense_reference(tmp_path, boundary, density, refinement)


@pytest.mark.parametrize("dimensions", (2, 3))
@pytest.mark.parametrize("ranks", (1, 2))
def test_conduction_composite(tmp_path, native_conduction, dimensions, ranks, monkeypatch):
    original = conduction.run

    def distributed(*args, **kwargs):
        return original(*args, ranks=ranks, **kwargs)

    monkeypatch.setattr(conduction, "run", distributed)
    conduction.test_composite_equation(tmp_path, dimensions, dimensions, 2)


@pytest.mark.parametrize("change", ("refine", "derefine"))
def test_conduction_regrid_restart(tmp_path, native_conduction, change):
    conduction.test_restart_after_regrid(tmp_path, change)


@pytest.mark.parametrize("dimensions", (1, 2))
def test_conduction_tabular_spitzer(tmp_path, native_conduction, dimensions):
    conduction.test_tabular_spitzer(tmp_path, dimensions)


def test_conduction_three_levels(tmp_path, native_conduction):
    conduction.test_three_levels(tmp_path, 2)


def test_conduction_equilibrium(tmp_path, native_conduction):
    conduction.test_isothermal_equilibrium(tmp_path)


@pytest.mark.parametrize("ranks", (1, 2))
def test_radiation_and_conduction_together(tmp_path, executable, ranks):
    radiation = re.search(r"<thermal_radiation>\n.*?(?=\n<problem>)",
                          INPUT.read_text(), re.S).group()
    outputs = []
    for backend in ("jacobi", "native-mg"):
        extra = (radiation.replace("implicit_preconditioner = block-coarse",
                                   f"implicit_preconditioner = {backend}") +
                 f"\n<hydro>\nconduction_preconditioner = {backend}\n")
        outputs.append(conduction.run(tmp_path / backend, dimensions=2, ranks=ranks,
                                      extra=extra))
    for reference, candidate in zip(*outputs):
        assert reference["time"] == candidate["time"]
        for name in reference["mb_data"]:
            np.testing.assert_allclose(conduction.canonical(candidate, name),
                                       conduction.canonical(reference, name),
                                       rtol=2e-9, atol=3e-11, err_msg=name)
