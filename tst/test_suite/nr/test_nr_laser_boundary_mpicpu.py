"""Grazing boundary crossings must also hand off to the neighboring MPI rank."""

import pytest

from test_suite.nr.test_nr_laser_boundary_cpu import check_boundary_crossing


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x1", "x2", "x3"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_grazing_boundary_mpi(axis, sign):
    check_boundary_crossing(axis, sign, coefficient=0.7, mpi=True)
