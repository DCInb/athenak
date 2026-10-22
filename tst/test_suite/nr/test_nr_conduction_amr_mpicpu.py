"""MPI decomposition and coarse/fine flux exchange for electron conduction."""

import numpy as np
import pytest

from test_suite.nr.test_nr_conduction_amr_cpu import (
    INPUT, assert_closed, canonical, run, tabular_spitzer_settings,
)


@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("change", [None, "refine", "derefine"])
def test_mpi_leaf_transport(tmp_path, ranks, change):
    flags = ["mesh/nghost=4"]
    if change:
        flags += ["mesh_refinement/refinement=adaptive",
                  f"amr_criterion0/value_max={0.5 if change == 'refine' else 2.0}"]
    serial = run(tmp_path / "serial", 3, flags)
    parallel = run(tmp_path / "parallel", 3, flags, ranks=ranks)
    assert_closed(parallel, changed=change is not None)
    assert len(serial) == len(parallel)
    for left, right in zip(serial, parallel):
        for name in ("dens", "ener", "eele_d", "eion_d"):
            np.testing.assert_allclose(canonical(left, name), canonical(right, name),
                                       rtol=3e-10, atol=3e-12)


def test_mpi_tabular_spitzer(tmp_path):
    extra = tabular_spitzer_settings(tmp_path)
    serial = run(tmp_path / "serial", 3, extra=extra)
    parallel = run(tmp_path / "parallel", 3, ranks=2, extra=extra)
    assert parallel[-1]["cycle"] == 1
    for name in ("ener", "eele_d", "eion_d"):
        np.testing.assert_allclose(canonical(serial[-1], name),
                                   canonical(parallel[-1], name),
                                   rtol=3e-10, atol=3e-12)


@pytest.mark.parametrize("change", ["refine", "derefine"])
def test_dynamic_buffers_mhd(tmp_path, change):
    """Exercise both cell- and face-centered migration with exact-sized buffers."""
    source = INPUT.read_text().replace("<hydro>", "<mhd>").replace("hydro_u", "mhd_u")
    source += """
<mhd>
dual_energy = false
<problem>
bxl = 0.1
bxr = 0.1
byl = 0.2
byr = -0.2
bzl = 0.0
bzr = 0.0
"""
    flags = ["mesh/nghost=4", "mesh_refinement/refinement=adaptive",
             f"amr_criterion0/value_max={0.5 if change == 'refine' else 2.0}"]
    fixed = run(tmp_path / "fixed", 3, flags + [
        "mesh_refinement/communication_buffers=fixed"], ranks=4, input_text=source)
    dynamic = run(tmp_path / "dynamic", 3, flags, ranks=4, input_text=source)
    assert_closed(dynamic, changed=True)
    for left, right in zip(fixed, dynamic):
        np.testing.assert_array_equal(left["mb_logical"], right["mb_logical"])
        for name in left["mb_data"]:
            np.testing.assert_allclose(canonical(left, name), canonical(right, name),
                                       rtol=3e-11, atol=3e-13)
