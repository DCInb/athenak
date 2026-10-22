"""Conservation and convergence with cold mixed-material EOS endpoints."""
import re

import numpy as np
import pytest

from test_suite.nr.test_nr_conduction_amr_cpu import REPO, field, integral, run


def mixed_endpoint_settings():
    # A recorded DCI damping cell: its EOS floor is 6.679753382762798e-5,
    # with an Au mass fraction of 0.9999638168266429. Roundoff corrections
    # of about -23 ulps used to throttle unrelated cells to a 0.175 step.
    tables = REPO / "DCI_integrate/material_tables"
    return f"""
<mesh>
nx1 = 64
<meshblock>
nx1 = 16
<mesh_refinement>
refinement = none
<units>
length_cgs = 0.1
mass_cgs = 0.0011
time_cgs = 1e-9
mu = 1.4444444444444444
<time>
evolution = dynamic
nlim = 1
tlim = 0.0032
<hydro>
nscalars = 2
rsolver = hllc
alpha_iso = 0.0
alpha_spitzer = true
conduction_flux_limiter = larsen
conduction_flux_limit_coefficient = 0.06
conduction_nonlinear_tolerance = 1.e-8
conduction_max_nonlinear_iterations = 12
<materials>
nmaterials = 2
scalar_index = 0
material0_abar = 196.96655
material0_zbar = 79
material0_eos_table_file = {tables / 'au.2t_eos'}
material1_abar = 4.002602
material1_zbar = 2
material1_eos_table_file = {tables / 'he.2t_eos'}
eos_table_interpolation = geometric
eos_table_bounds = flash-extrapolate
<problem>
dl = 0.24761998266902963
dr = 0.24761998266902963
yl = 0.9999638168266429
yr = 0.9999638168266429
pl = 0.0024761998266902963
pr = 2.4761998266902963e-13
"""


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("refinement", ["none", "static"])
@pytest.mark.parametrize("backend", ["jacobi", "native-mg"])
@pytest.mark.parametrize("linear_tolerance", [1.e-10, 1.e-4])
@pytest.mark.parametrize("stiff", [False, True])
def test_mixed_endpoint_newton_convergence(tmp_path, ranks, refinement, backend,
                                         linear_tolerance, stiff):
    settings = (mixed_endpoint_settings() +
                f"\n<hydro>\nconduction_preconditioner = {backend}\n" +
                f"conduction_linear_tolerance = {linear_tolerance}\n")
    if stiff:
        # Exercise multigrid above the Jacobi condition-bound threshold.
        # A multiplier of 10000 is now solved by Jacobi; 100000 retains the
        # mixed-endpoint multigrid coverage and the same physical checks.
        settings += "conduction_spitzer_multiplier = 100000\n"
    flags = [f"mesh_refinement/refinement={refinement}"]
    outputs = run(tmp_path / "solve", extra=settings, flags=flags, ranks=ranks)
    control = run(tmp_path / "control", extra=settings.replace(
        "alpha_spitzer = true", "alpha_spitzer = false\nconduction_integrator = explicit"),
        flags=flags, ranks=ranks)
    final, reference = outputs[-1], control[-1]
    assert final["cycle"] == reference["cycle"] == 1
    assert final["time"] == reference["time"]
    for name in ("eele_d", "eion_d", "ener"):
        assert np.all(np.isfinite(field(final, name)))
        assert np.min(field(final, name)) > 0
        np.testing.assert_allclose(integral(final, name), integral(reference, name),
                                   rtol=2.e-11, atol=0)
    np.testing.assert_allclose(field(final, "eion_d"), field(reference, "eion_d"),
                               rtol=2.e-11, atol=0)
    log = (tmp_path / "solve/run.log").read_text()
    match = re.search(r"# implicit conduction:.*relative_residual=(\S+)", log)
    assert match and float(match[1]) <= 1.e-8
    if backend == "native-mg" and refinement == "static" and linear_tolerance == 1.e-4:
        # The old aggregation injected outward corrections into nearly
        # decoupled cold cells. Correct coarse isolation can eliminate them;
        # requiring that diagnostic would reject a more accurate correction.
        # Check that the intended preconditioner is exercised while retaining
        # all equation, positivity and conservation checks above.
        if stiff:
            assert re.search(r"native_newton_solves=[1-9]", log)
        else:
            assert re.search(r"native_newton_solves=0 jacobi_newton_solves=[1-9]", log)
    assert np.max(np.abs(field(final, "eele_d")-field(reference, "eele_d"))) > 0
