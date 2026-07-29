"""Terminal-remainder diagnostics for two-temperature laser transport."""

import os

import numpy as np
import pytest

import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"
reflection_input = "../../../inputs/mhd/two_temperature_laser_reflection.athinput"
integer_diagnostic_fields = (
    "max_reflections",
    "suppressed_turns",
    "reflection_rearms",
)


def appended_log(offset):
    with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
        stream.seek(offset)
        return stream.read()


def reflection_deck_with_hysteresis(filename, fraction=0.01):
    with open(reflection_input, encoding="utf-8") as stream:
        text = stream.read()
    marker = "<laser>\n"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        marker + f"reflection_hysteresis_fraction = {fraction}\n",
        1,
    )
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(text)
    return filename


def laser_diagnostics(log):
    lines = [line for line in log.splitlines()
             if line.startswith("laser: launched=")]
    assert lines, "laser accounting line is missing"
    fields = {}
    for token in lines[-1].split()[1:]:
        key, text = token.split("=", 1)
        if key in integer_diagnostic_fields:
            value = int(text)
            assert value >= 0, f"negative laser diagnostic {key}={text}"
            fields[key] = value
        else:
            fields[key] = float(text)
    for key in integer_diagnostic_fields:
        assert key in fields, f"laser accounting line is missing {key}"
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

        # A planar critical surface turns exactly once with hysteresis enabled,
        # then rearms only after returning to the underdense side.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        hysteresis_input = reflection_deck_with_hysteresis(
            "laser_hysteresis_reflection.athinput"
        )
        assert testutils.run(
            hysteresis_input,
            flags=common_flags("laser_hysteresis_reflection") + [
                "laser/report_diagnostics=true",
            ],
        )
        reflected = laser_diagnostics(appended_log(log_offset))
        assert reflected["remaining"] == 0.0
        assert reflected["remaining_fraction"] == 0.0
        assert_remaining_partition(reflected, 0.0, 0.0)
        assert reflected["max_reflections"] == 1

        # Two overdense walls enclose a uniform underdense gap. After reflecting
        # from the right wall, the ray must rearm while crossing that gap and
        # reach the left wall, where the one-reflection cap terminates it.
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        two_wall_input = reflection_deck_with_hysteresis(
            "laser_hysteresis_two_wall.athinput"
        )
        with pytest.raises(RuntimeError, match="Failed to execute"):
            testutils.run(
                two_wall_input,
                flags=common_flags("laser_hysteresis_two_wall") + [
                    "laser/beam0_origin_x1=0.5",
                    "laser/beam0_direction_x1=1.0",
                    "laser/max_reflections_per_ray=1",
                    "laser/report_diagnostics=false",
                    "problem/density_profile=two_wall",
                    "problem/density_gradient=5.0",
                ],
                timeout=30.0,
            )
        two_wall_log = appended_log(log_offset)
        two_wall = laser_diagnostics(two_wall_log)
        assert_remaining_partition(two_wall, 0.0, 1.0)
        assert two_wall["remaining_fraction"] == 1.0
        assert two_wall["reflection_remaining_rays"] == 1.0
        assert two_wall["max_reflections"] == 1
        assert two_wall["reflection_rearms"] == 1
        assert "Laser transport failed" in two_wall_log

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
        for filename in (
            "laser_hysteresis_reflection.athinput",
            "laser_hysteresis_two_wall.athinput",
        ):
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
