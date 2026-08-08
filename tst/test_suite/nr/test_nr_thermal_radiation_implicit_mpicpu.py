"""MPI decomposition regression for implicit multilevel radiation transport."""

import numpy as np

import athena_read
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/hydro/mgfld_diffusion.athinput"
FINAL_TIME = 5.0e-4


def run_case(basename, ranks):
    """Advance a step placed on the one/two-rank ownership boundary."""
    flags = [
        f"job/basename={basename}",
        # Four x1 blocks divide two-per-rank.  The discontinuity at x1=0 is
        # therefore on the rank boundary in the two-rank run.  The periodic
        # direction has even global extent, as required by two-color smoothing.
        "mesh/nx1=36", "mesh/nx2=9", "mesh/nx3=9",
        "meshblock/nx1=9", "meshblock/nx2=9", "meshblock/nx3=9",
        "mesh/ix1_bc=periodic", "mesh/ox1_bc=periodic",
        "mesh/ix2_bc=outflow", "mesh/ox2_bc=outflow",
        "mesh/ix3_bc=outflow", "mesh/ox3_bc=outflow",
        "thermal_radiation/initial_profile=step",
        "thermal_radiation/initial_radiation_temperature=1.0",
        "thermal_radiation/initial_radiation_temperature_right=0.5",
        "thermal_radiation/initial_radiation_x1=0.0",
        "thermal_radiation/kappa_transport_0=100.0",
        "thermal_radiation/kappa_transport_1=100.0",
        "thermal_radiation/flux_limiter=none",
        "thermal_radiation/transport_integrator=implicit",
        "thermal_radiation/implicit_preconditioner=block-coarse",
        "thermal_radiation/source_cfl=0",
        "problem/pl=1.0e-8", "problem/pr=1.0e-8",
        "time/nlim=1", f"time/tlim={FINAL_TIME}",
        f"output1/dt={FINAL_TIME}", "output1/data_format=%24.17e",
    ]
    assert testutils.mpi_run(
        INPUT_FILE, flags=flags, threads=ranks, timeout=120.0), (
            f"{basename} failed with {ranks} MPI ranks")
    return (
        sorted_table(f"tab/{basename}.hydro_3t.00000.tab"),
        sorted_table(f"tab/{basename}.hydro_3t.00001.tab"),
    )


def sorted_table(path):
    """Make the comparison independent of MeshBlock/output ordering."""
    table = athena_read.tab(path)
    order = np.argsort(np.asarray(table["x1v"]))
    return {
        field: (np.asarray(values)[order]
                if np.asarray(values).shape == order.shape else np.asarray(values))
        for field, values in table.items()
    }


def test_run():
    try:
        initial_one, final_one = run_case("fld_implicit_mpi_1", 1)
        initial_two, final_two = run_case("fld_implicit_mpi_2", 2)

        for group in ("erad00", "erad01"):
            assert np.array_equal(initial_two[group], initial_one[group])
            assert np.unique(initial_one[group]).size == 2
            assert np.all(np.isfinite(final_one[group]))
            assert np.all(np.isfinite(final_two[group]))
            assert np.min(final_one[group]) >= 0.0
            assert np.min(final_two[group]) >= 0.0
            assert not np.array_equal(final_one[group], initial_one[group])
            assert np.var(final_one[group]) < np.var(initial_one[group])
            assert np.allclose(
                final_two[group], final_one[group], rtol=3.0e-9, atol=3.0e-11), group
            assert np.isclose(
                np.sum(final_one[group]), np.sum(initial_one[group]),
                rtol=5.0e-11, atol=5.0e-11)
            assert np.isclose(
                np.sum(final_two[group]), np.sum(initial_two[group]),
                rtol=5.0e-11, atol=5.0e-11)
    finally:
        testutils.cleanup()
