"""MPI+GPU variant of the Biermann decomposition regressions."""

from pathlib import Path

import pytest

from test_suite.nr import test_nr_biermann_subcycle_mpicpu as mpi_regression


def test_run():
    cache = Path("../CMakeCache.txt")
    if not cache.exists() or "Athena_ENABLE_MPI:BOOL=ON" not in cache.read_text(
            encoding="ascii"):
        pytest.skip("Biermann MPI+GPU regression requires an MPI-enabled GPU build")
    mpi_regression.test_run()
