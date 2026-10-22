"""Choose cheap diagonal PCG without sacrificing the composite heat equation."""

import os
from pathlib import Path
import re

import numpy as np
import pytest

from test_suite.nr import test_nr_conduction_amr_cpu as conduction


@pytest.mark.parametrize("ranks", (1, 2, 4))
@pytest.mark.parametrize("conductivity", (0.5, 5.0))
def test_conduction_condition_selection(tmp_path, monkeypatch, ranks, conductivity):
    executable = os.environ.get("ATHENA_NATIVE_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_NATIVE_TEST_EXE to a built_in_pgens binary")
    monkeypatch.setenv("ATHENA_CONDUCTION_TEST_EXE", str(Path(executable).resolve()))
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    flags = ["mesh/nx1=32", "time/nlim=1", "time/tlim=0.001",
             "time/initial_dt=0.001", f"hydro/alpha_iso={conductivity}",
             "hydro/conduction_nonlinear_tolerance=1e-12"]
    outputs = {}
    for backend in ("jacobi", "native-mg"):
        outputs[backend] = conduction.run(
            tmp_path / backend, ranks=ranks, flags=flags,
            extra=f"\n<hydro>\nconduction_preconditioner={backend}\n")
    initial, final = outputs["native-mg"][0], outputs["native-mg"][-1]
    assert final["cycle"] == 1 and final["time"] == 0.001
    e0 = conduction.field(initial, "eele_d").ravel()
    e1 = conduction.field(final, "eele_d").ravel()
    rho = conduction.field(final, "dens").ravel()
    laplacian = conduction.leaf_laplacian(
        final, e1 / (0.75 * rho), conductivity * rho)
    np.testing.assert_allclose(e1 - e0, 0.001 * laplacian, rtol=0, atol=3e-11)
    for field in ("eele_d", "eion_d", "ener"):
        np.testing.assert_allclose(conduction.integral(final, field),
                                   conduction.integral(initial, field),
                                   rtol=2e-12, atol=2e-14)
        assert np.min(conduction.field(final, field)) > 0
        np.testing.assert_allclose(conduction.canonical(final, field),
                                   conduction.canonical(outputs["jacobi"][-1], field),
                                   rtol=3e-11, atol=3e-13)
    log = (tmp_path / "native-mg/run.log").read_text()
    native, diagonal = map(int, re.search(
        r"native_newton_solves=(\d+) jacobi_newton_solves=(\d+)", log).groups())
    if conductivity == 0.5:
        # eta=5.4613 on the finest leaves: the hierarchy was more expensive
        # than Jacobi in DCI_3D at this conditioning. Selecting the same diagonal
        # path must give exactly the reference fields, including faint cells.
        assert native == 0 and diagonal > 0
        for field in final["mb_data"]:
            np.testing.assert_array_equal(
                conduction.canonical(final, field),
                conduction.canonical(outputs["jacobi"][-1], field))
    else:
        # Do not implement the optimization by disabling multigrid entirely.
        assert native > 0 and diagonal == 0
