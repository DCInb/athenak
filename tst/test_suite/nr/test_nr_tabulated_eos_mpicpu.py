"""MPI broadcast regression tests for the portable tabulated EOS."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils
from test_suite.nr.test_nr_tabulated_eos_cpu import write_tablereader_table


input_file = "../../../inputs/hydro/tabulated_eos.athinput"
native_table_file = "../../../inputs/hydro/gamma_law_eos_table.dat"
binary_table_file = "gamma_law_eos_tablereader_mpi.dat"


def run_table_case(basename, ranks, filename):
    flags = [
        f"job/basename={basename}",
        f"hydro/table_file={filename}",
        "time/tlim=0.01",
        "output1/dt=0.01",
        "output2/dt=-1.0",
    ]
    assert testutils.mpi_run(input_file, flags=flags, threads=ranks), (
        f"{basename} failed with {ranks} MPI ranks.")
    return athena_read.tab(f"tab/{basename}.hydro_w.00001.tab")


def compare_fields(candidate, reference):
    assert candidate.keys() == reference.keys()
    for field in reference:
        assert np.all(np.isfinite(candidate[field]))
        assert np.allclose(candidate[field], reference[field],
                           rtol=2.0e-10, atol=2.0e-12), field


def test_run():
    try:
        write_tablereader_table(binary_table_file)
        for label, filename in (
                ("native", native_table_file),
                ("binary", binary_table_file)):
            reference = run_table_case(f"eos_mpi_{label}_1", 1, filename)
            parallel = run_table_case(f"eos_mpi_{label}_2", 2, filename)
            compare_fields(parallel, reference)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
