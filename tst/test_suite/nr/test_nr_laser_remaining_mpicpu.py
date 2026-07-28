"""MPI reduction regression for terminal laser remainder diagnostics."""

import os

import pytest

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_remaining_cpu import (
    appended_log,
    assert_remaining_partition,
    common_flags,
    laser_diagnostics,
)


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"


def test_run():
    try:
        # Force rank-distributed transport to terminate at max_mpi_waves. The
        # six-slot MPI reductions must preserve the aggregate and split buckets.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        with pytest.raises(RuntimeError, match="Failed to execute"):
            testutils.mpi_run(
                input_file,
                flags=common_flags("laser_remainder_mpi_wave") + [
                    "laser/absorption_coefficient=0.0",
                    "laser/max_segments_per_launch=1",
                    "laser/max_transport_iterations=1",
                    "laser/max_mpi_waves=1",
                    "laser/report_diagnostics=false",
                    "output2/dt=-1.0",
                    "output3/dt=-1.0",
                    "output4/dt=-1.0",
                ],
                threads=2,
                timeout=30.0,
            )
        log = appended_log(log_offset)
        diagnostics = laser_diagnostics(log)
        assert_remaining_partition(diagnostics, 1.0, 0.0)
        assert diagnostics["remaining_fraction"] == 1.0
        assert diagnostics["wave_remaining_rays"] == 1.0
        assert "Laser transport failed" in log
    finally:
        testutils.cleanup()
