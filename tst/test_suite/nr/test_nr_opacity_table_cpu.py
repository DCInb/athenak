"""Input-validation regression tests for tabulated thermal-radiation opacities."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/hydro/two_temperature_opacity_table.athinput"
opacity_table = "../../../inputs/hydro/two_temperature_opacity_table.dat"
zero_safe_table = "../../../inputs/hydro/two_temperature_opacity_zero.dat"
per_kind_input = Path("opacity_per_kind_scales.athinput")


def opacity_command(basename, extra_flags, case_input=input_file):
    """Return a minimal initialization-only opacity-table command."""
    return [
        "./athena", "-i", str(case_input),
        f"job/basename={basename}",
        f"thermal_radiation/opacity_table_file={opacity_table}",
        "time/nlim=0",
        "output1/dt=-1.0",
    ] + extra_flags


def write_per_kind_input():
    """Create a test-only input exposing the three optional override keys."""
    source = Path(input_file).read_text(encoding="ascii")
    common = "opacity_value_scale = 1.0\n"
    overrides = (
        "opacity_transport_scale = 1.0\n"
        "opacity_absorption_scale = 1.0\n"
        "opacity_emission_scale = 1.0\n"
    )
    assert source.count(common) == 1
    per_kind_input.write_text(
        source.replace(common, common+overrides, 1), encoding="ascii")


def run_scale_case(basename, scale):
    """Run one relaxation step using only the common opacity scale."""
    flags = [
        f"job/basename={basename}",
        f"thermal_radiation/opacity_table_file={opacity_table}",
        f"thermal_radiation/opacity_value_scale={scale}",
    ]
    assert testutils.run(input_file, flags=flags), f"{basename} failed."
    return athena_read.tab(f"tab/{basename}.hydro_3t.00001.tab")


def run_zero_safe_case(basename, coordinate_interpolation):
    flags = [
        f"job/basename={basename}",
        f"thermal_radiation/opacity_table_file={zero_safe_table}",
        "thermal_radiation/opacity_interpolation=geometric",
        ("thermal_radiation/opacity_coordinate_interpolation="
         f"{coordinate_interpolation}"),
    ]
    assert testutils.run(input_file, flags=flags), f"{basename} failed."
    return athena_read.tab(f"tab/{basename}.hydro_3t.00001.tab")


def test_run():
    try:
        write_per_kind_input()
        assert testutils.run_command(opacity_command("opacity_scale_valid", []))

        scale_parameters = (
            "opacity_density_scale",
            "opacity_temperature_scale",
            "opacity_group_bound_scale",
            "opacity_value_scale",
            "opacity_transport_scale",
            "opacity_absorption_scale",
            "opacity_emission_scale",
        )
        per_kind_parameters = set(scale_parameters[4:])
        for value in ("nan", "inf"):
            for parameter in scale_parameters:
                case_input = (per_kind_input if parameter in per_kind_parameters
                              else input_file)
                command = opacity_command(
                    f"opacity_{parameter}_{value}",
                    [f"thermal_radiation/{parameter}={value}"],
                    case_input,
                )
                assert not testutils.run_command(command), (
                    f"{parameter} unexpectedly accepted {value}.")

        positive_parameters = scale_parameters[:5]
        for value in ("0.0", "-1.0"):
            for parameter in positive_parameters:
                case_input = (per_kind_input if parameter in per_kind_parameters
                              else input_file)
                command = opacity_command(
                    f"opacity_{parameter}_{value}",
                    [f"thermal_radiation/{parameter}={value}"],
                    case_input,
                )
                assert not testutils.run_command(command), (
                    f"{parameter} unexpectedly accepted {value}.")

        source_parameters = scale_parameters[5:]
        for parameter in source_parameters:
            valid_zero = opacity_command(
                f"opacity_{parameter}_zero",
                [f"thermal_radiation/{parameter}=0.0"],
                per_kind_input,
            )
            assert testutils.run_command(valid_zero), (
                f"{parameter} unexpectedly rejected zero.")
            invalid_negative = opacity_command(
                f"opacity_{parameter}_negative",
                [f"thermal_radiation/{parameter}=-1.0"],
                per_kind_input,
            )
            assert not testutils.run_command(invalid_negative), (
                f"{parameter} unexpectedly accepted a negative value.")

        baseline = run_scale_case("opacity_common_scale_1", 1.0)
        scaled = run_scale_case("opacity_common_scale_2", 2.0)
        assert not np.allclose(
            scaled["erad00"], baseline["erad00"], rtol=1.0e-12, atol=1.0e-14)
        for result in (baseline, scaled):
            total = result["eion"] + result["eele"] + result["erad"]
            assert np.allclose(total, total[0], rtol=2.0e-12, atol=2.0e-13)

        # Strict legacy log(value) interpolation still rejects a zero table entry.
        strict_log = opacity_command(
            "opacity_zero_strict_log",
            [f"thermal_radiation/opacity_table_file={zero_safe_table}",
             "thermal_radiation/opacity_interpolation=log"],
        )
        assert not testutils.run_command(strict_log)

        # Zero-safe geometric interpolation falls back to linear only for stencils
        # containing a zero.  Log-coordinate and linear-coordinate lookup should both
        # remain finite/conservative and should differ at the off-center test state.
        zero_linear = run_zero_safe_case("opacity_zero_linear_coord", "linear")
        zero_log = run_zero_safe_case("opacity_zero_log_coord", "log")
        for result in (zero_linear, zero_log):
            for values in result.values():
                assert np.all(np.isfinite(values))
            total = result["eion"] + result["eele"] + result["erad"]
            assert np.allclose(total, total[0], rtol=2.0e-12, atol=2.0e-13)
        assert not np.allclose(zero_linear["erad00"], zero_log["erad00"],
                               rtol=1.0e-12, atol=1.0e-14)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        per_kind_input.unlink(missing_ok=True)
        testutils.cleanup()
