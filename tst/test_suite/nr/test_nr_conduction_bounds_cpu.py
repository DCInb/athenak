"""Cold tabular conduction must ignore corrections that round to no change.

Uses the He/Au tables staged for DCI_integrate run-1561715. Set
ATHENA_CONDUCTION_TEST_EXE to a built_in_pgens CPU executable.
"""

import re

import numpy as np
import pytest

from test_suite.nr.test_nr_conduction_amr_cpu import REPO, field, integral, run


def cold_tabular_settings(material="he", duration=1e-4, refinement="none"):
    abar, density = {
        "he": (4.002602, 9.090909090909091e-6),
        "au": (196.96655, 17.454545454545453),
    }[material]
    table = REPO / "DCI_integrate/material_tables" / f"{material}.2t_eos"
    return f"""
<mesh>
nx1 = 128
<meshblock>
nx1 = 32
<mesh_refinement>
refinement = {refinement}
<units>
length_cgs = 0.1
mass_cgs = 0.0011
time_cgs = 1e-9
mu = 1.4444444444444444
<time>
evolution = dynamic
nlim = 1
tlim = {duration}
<hydro>
rsolver = hllc
nscalars = 1
conduction_max_nonlinear_iterations = 80
alpha_iso = 0.0
alpha_spitzer = true
conduction_flux_limiter = larsen
conduction_flux_limit_coefficient = 0.06
<materials>
nmaterials = 1
scalar_index = 0
material0_abar = {abar}
material0_zbar = 1
material0_eos_table_file = {table}
eos_table_interpolation = geometric
eos_table_bounds = flash-extrapolate
<problem>
dl = {density}
dr = {density}
pl = {density*0.01}
pr = {density*1e-12}
"""


@pytest.mark.parametrize("material,duration", [
    ("he", 1e-4), ("he", 1e-3), ("au", 1e-4),
])
@pytest.mark.parametrize("refinement", ["none", "static"])
def test_cold_tabular_roundoff(tmp_path, material, duration, refinement):
    extra = cold_tabular_settings(material, duration, refinement)
    outputs = run(tmp_path / "solve", extra=extra)
    control = run(tmp_path / "control", extra=extra.replace(
        "alpha_spitzer = true",
        "alpha_spitzer = false\nconduction_integrator = explicit"))
    final, reference = outputs[-1], control[-1]
    assert final["cycle"] == reference["cycle"] == 1
    assert final["time"] == reference["time"]
    for name in ("eele_d", "eion_d", "ener"):
        assert np.all(np.isfinite(field(final, name)))
        assert np.min(field(final, name)) > 0
        np.testing.assert_allclose(
            integral(final, name), integral(reference, name), rtol=2e-11, atol=0)
    np.testing.assert_allclose(field(final, "eion_d"), field(reference, "eion_d"),
                               rtol=2e-11, atol=0)
    assert np.max(np.abs(field(final, "eele_d")-field(reference, "eele_d"))) > 0
    log = (tmp_path / "solve/run.log").read_text()
    match = re.search(r"# implicit conduction:.*relative_residual=(\S+)", log)
    assert match and float(match[1]) <= 1e-11


def test_real_temperature_bound_still_rejected(tmp_path):
    # The cold right boundary draws heat from cells already at their EOS floor.
    # This is a real outward step, not a sub-ulp PCG tail.
    flags = ["mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
             "hydro/conduction_x1_outer_boundary=dirichlet",
             "hydro/conduction_x1_outer_value=0"]
    with pytest.raises(AssertionError, match="EOS-table temperature bound"):
        run(tmp_path / "solve", extra=cold_tabular_settings(), flags=flags)
