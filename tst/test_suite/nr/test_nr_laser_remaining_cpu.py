"""Terminal-remainder diagnostics for two-temperature laser transport."""

import os

import numpy as np
import pytest

import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"
reflection_input = "../../../inputs/mhd/two_temperature_laser_reflection.athinput"


def appended_log(offset):
    with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
        stream.seek(offset)
        return stream.read()


def laser_diagnostics(log):
    lines = [line for line in log.splitlines()
             if line.startswith("laser: launched=")]
    assert lines, "laser accounting line is missing"
    fields = {}
    for token in lines[-1].split()[1:]:
        key, value = token.split("=", 1)
        fields[key] = float(value)
    return fields


def assert_remaining_partition(fields, wave_power, reflection_power):
    assert np.isclose(fields["remaining"],
                      fields["wave_remaining"] +
                      fields["reflection_remaining"])
    assert np.isclose(fields["wave_remaining"], wave_power)
    assert np.isclose(fields["reflection_remaining"], reflection_power)
    assert fields["remaining_rays"] == fields["active"]
    assert fields["active"] == (
        fields["wave_remaining_rays"] +
        fields["reflection_remaining_rays"])


def common_flags(basename):
    return [
        f"job/basename={basename}",
        "time/integrator=rk1",
        "time/nlim=1",
        "time/tlim=1.0",
        "output1/dt=-1.0",
    ]


def test_run():
    try:
        # A completed ray retains the legacy total fields and reports zero in both
        # terminal-cause buckets.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        assert testutils.run(input_file, flags=common_flags("laser_remainder_zero") + [
            "laser/absorption_coefficient=0.0",
            "laser/report_diagnostics=true",
            "output2/dt=-1.0",
            "output3/dt=-1.0",
            "output4/dt=-1.0",
        ])
        completed = laser_diagnostics(appended_log(log_offset))
        assert completed["remaining"] == 0.0
        assert completed["remaining_fraction"] == 0.0
        assert_remaining_partition(completed, 0.0, 0.0)

        # Exhausting the global wave budget is a fatal incomplete transport, with
        # all terminal power and ray count attributed to the wave-cap bucket.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        with pytest.raises(RuntimeError, match="Failed to execute"):
            testutils.run(
                input_file,
                flags=common_flags("laser_remainder_wave") + [
                    "laser/absorption_coefficient=0.0",
                    "laser/max_segments_per_launch=1",
                    "laser/max_transport_iterations=1",
                    "laser/max_mpi_waves=1",
                    "laser/report_diagnostics=false",
                    "output2/dt=-1.0",
                    "output3/dt=-1.0",
                    "output4/dt=-1.0",
                ],
                timeout=30.0,
            )
        wave_log = appended_log(log_offset)
        wave = laser_diagnostics(wave_log)
        assert_remaining_partition(wave, 1.0, 0.0)
        assert wave["remaining_fraction"] == 1.0
        assert wave["wave_remaining_rays"] == 1.0
        assert "Laser transport failed" in wave_log

        # A zero reflection allowance reaches the critical surface but may not turn;
        # it is fatal and attributed only to the reflection-cap bucket.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        with pytest.raises(RuntimeError, match="Failed to execute"):
            testutils.run(
                reflection_input,
                flags=common_flags("laser_remainder_reflection") + [
                    "laser/max_reflections_per_ray=0",
                    "laser/report_diagnostics=false",
                ],
                timeout=30.0,
            )
        reflection_log = appended_log(log_offset)
        reflection = laser_diagnostics(reflection_log)
        assert_remaining_partition(reflection, 0.0, 1.0)
        assert reflection["remaining_fraction"] == 1.0
        assert reflection["reflection_remaining_rays"] == 1.0
        assert "Laser transport failed" in reflection_log
    finally:
        testutils.cleanup()
