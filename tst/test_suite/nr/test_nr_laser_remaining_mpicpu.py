"""MPI reduction regression for terminal laser remainder diagnostics."""

import os

import pytest

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_cpu import critical_density_cgs
from test_suite.nr.test_nr_laser_remaining_cpu import (
    appended_log,
    assert_remaining_partition,
    common_flags,
    laser_diagnostics,
    reflection_deck_with_hysteresis,
)


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"


def test_run():
    try:
        # Put the critical surface exactly on the x=0.5 rank boundary. The
        # reflected ray must be assigned directly to its forward-side rank,
        # without a zero-distance MPI exchange loop.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        hysteresis_input = reflection_deck_with_hysteresis(
            "laser_hysteresis_mpi_reflection.athinput"
        )
        rho_turn = critical_density_cgs(1.0) / 1.0e13
        boundary_gradient = (rho_turn - 0.5) / 0.5
        assert testutils.mpi_run(
            hysteresis_input,
            flags=common_flags("laser_hysteresis_mpi_reflection") + [
                f"problem/density_gradient={boundary_gradient:.17g}",
                "laser/report_diagnostics=true",
            ],
            threads=2,
            timeout=30.0,
        )
        reflected = laser_diagnostics(appended_log(log_offset))
        assert_remaining_partition(reflected, 0.0, 0.0)
        assert reflected["remaining_fraction"] == 0.0
        assert reflected["max_reflections"] == 1
        assert reflected["reflection_rearms"] == 1
        assert reflected["transfers"] == 0.0

        # Move the turn just onto rank 1. The reflected ray now migrates while
        # disarmed and must carry its saved turning density to rank 0 before rearm.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        assert testutils.mpi_run(
            hysteresis_input,
            flags=common_flags("laser_hysteresis_mpi_migration") + [
                "problem/density_gradient=1.218",
                "laser/report_diagnostics=true",
            ],
            threads=2,
            timeout=30.0,
        )
        migrated = laser_diagnostics(appended_log(log_offset))
        assert_remaining_partition(migrated, 0.0, 0.0)
        assert migrated["remaining_fraction"] == 0.0
        assert migrated["max_reflections"] == 1
        assert migrated["reflection_rearms"] == 1
        assert migrated["transfers"] >= 2.0

        # Force rank-distributed transport to terminate at max_mpi_waves. The
        # reductions must preserve the aggregate and split buckets.
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
        assert diagnostics["waves"] == 1
        assert diagnostics["iterations"] == 1
        assert log.count("laser_remaining_ray:") == 1
        assert "Laser transport failed" in log
    finally:
        testutils.cleanup()
        try:
            os.remove("laser_hysteresis_mpi_reflection.athinput")
        except FileNotFoundError:
            pass
