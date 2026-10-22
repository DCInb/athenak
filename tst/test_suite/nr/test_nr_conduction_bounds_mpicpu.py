"""Cold EOS-bound corrections must not stall the global MPI Newton reduction."""

import numpy as np
import pytest

from test_suite.nr.test_nr_conduction_amr_cpu import canonical, run
from test_suite.nr.test_nr_conduction_bounds_cpu import cold_tabular_settings


@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("refinement", ["none", "static"])
def test_cold_tabular_mpi(tmp_path, ranks, refinement):
    extra = cold_tabular_settings(refinement=refinement)
    serial = run(tmp_path / "serial", extra=extra)
    parallel = run(tmp_path / "parallel", extra=extra, ranks=ranks)
    assert parallel[-1]["cycle"] == 1
    assert serial[-1]["time"] == parallel[-1]["time"]
    for name in ("dens", "ener", "eele_d", "eion_d"):
        np.testing.assert_allclose(canonical(serial[-1], name),
                                   canonical(parallel[-1], name),
                                   rtol=3e-10, atol=0)
