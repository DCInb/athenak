"""Regression tests for the FLASH-style flux-form Biermann battery."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_temperature_biermann.athinput"
T_FINAL = 1.0e-4
COEFFICIENT = 0.1
DENSITY_AMPLITUDE = 0.2
PRESSURE_AMPLITUDE = 0.2
Y_SLICE = 0.0078125


def run_case(basename, temperature_ratio, extra_flags=None):
    Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
    flags = [
        f"job/basename={basename}",
        f"mhd/initial_electron_temperature_ratio={temperature_ratio}",
    ]
    if extra_flags:
        flags.extend(extra_flags)
    assert testutils.run(input_file, flags=flags), f"{basename} run failed."


def analytic_b3_rate(x1, temperature_ratio, y_slice=Y_SLICE):
    wave_number = 2.0 * np.pi
    density = np.exp(DENSITY_AMPLITUDE * np.sin(wave_number * y_slice))
    pressure = np.exp(PRESSURE_AMPLITUDE * np.sin(wave_number * x1))
    electron_pressure_over_fraction = (
        2.0 * temperature_ratio / (1.0 + temperature_ratio))
    return (COEFFICIENT * electron_pressure_over_fraction * (pressure / density)
            * PRESSURE_AMPLITUDE * DENSITY_AMPLITUDE * wave_number**2
            * np.cos(wave_number * x1) * np.cos(wave_number * y_slice))


def load_profile(basename):
    field = athena_read.tab(f"tab/{basename}.biermann.00001.tab")
    two_temp = athena_read.tab(
        f"tab/{basename}.two_temperature.00001.tab")
    order = np.argsort(field["x1v"])
    sorted_field = dict(field)
    for key, value in field.items():
        if np.ndim(value) == 1 and len(value) == len(order):
            sorted_field[key] = value[order]
    return sorted_field, two_temp


def assert_energy_conserved(basename):
    history = athena_read.hst(f"{basename}.mhd.hst")
    assert np.allclose(history["tot-E"], history["tot-E"][0],
                       rtol=2.0e-12, atol=2.0e-12)


def test_run():
    try:
        rates = {}
        for basename, temperature_ratio in (("biermann_equal", 1.0),
                                            ("biermann_cold", 0.25)):
            run_case(basename, temperature_ratio)
            field, two_temp = load_profile(basename)
            numerical_rate = field["bcc3"] / T_FINAL
            exact_rate = analytic_b3_rate(field["x1v"], temperature_ratio)
            relative_l2 = (np.linalg.norm(numerical_rate - exact_rate)
                           / np.linalg.norm(exact_rate))
            assert relative_l2 < 1.5e-2
            assert np.max(np.abs(field["bcc1"])) < 2.0e-12
            assert np.max(np.abs(field["bcc2"])) < 2.0e-12
            assert np.all(two_temp["eion"] > 0.0)
            assert np.all(two_temp["eele"] > 0.0)
            assert np.all(two_temp["tion"] > 0.0)
            assert np.all(two_temp["tele"] > 0.0)
            assert_energy_conserved(basename)
            rates[temperature_ratio] = numerical_rate

        # For f_e=1/2, p_e/f_e scales as 2(Te/Ti)/(1+Te/Ti).
        amplitude_ratio = (np.linalg.norm(rates[0.25])
                           / np.linalg.norm(rates[1.0]))
        assert np.isclose(amplitude_ratio, 0.4, rtol=7.0e-3, atol=0.0)

        # The dedicated Strang/SSPRK2 path must retain the same early-time analytic
        # magnetic response, even when this weak case needs only one step per half.
        run_case("biermann_equal_subcycle", 1.0, [
            "mhd/biermann_subcycle=true",
        ])
        subcycle_field, subcycle_two_temp = load_profile(
            "biermann_equal_subcycle")
        subcycle_rate = subcycle_field["bcc3"]/T_FINAL
        exact_rate = analytic_b3_rate(subcycle_field["x1v"], 1.0)
        assert (np.linalg.norm(subcycle_rate-exact_rate)
                / np.linalg.norm(exact_rate)) < 1.5e-2
        assert np.all(subcycle_two_temp["eion"] > 0.0)
        assert np.all(subcycle_two_temp["eele"] > 0.0)
        assert_energy_conserved("biermann_equal_subcycle")

        # A large coefficient makes the FLASH thermal-magnetic speed, rather than
        # ideal-MHD waves, set the initial step.
        run_case("biermann_cfl", 1.0, [
            "mhd/biermann_coefficient=100.0",
            "time/nlim=1", "time/tlim=1.0",
        ])
        cfl_history = athena_read.hst("biermann_cfl.mhd.hst")
        assert cfl_history["dt"][0] < 5.0e-5

        # Keep the retained legacy shock treatment live and observable.  A steep
        # pressure profile plus compression activates the directional mask; the
        # dedicated subcycle deliberately uses its endpoint cochain instead, so this
        # compatibility check belongs on the stage-coupled path.
        legacy_mask_fields = {}
        for basename, suppress in (("biermann_legacy_masked", "true"),
                                   ("biermann_legacy_unmasked", "false")):
            run_case(basename, 1.0, [
                "mesh/nx1=32", "mesh/nx2=32",
                "meshblock/nx1=16", "meshblock/nx2=16",
                "mesh/x1min=-0.25", "mesh/x1max=0.75",
                "problem/pressure_amplitude=2.0",
                "problem/compression_rate_x1=4.0",
                "problem/compression_rate_x2=0.0",
                "problem/compression_rate_x3=0.0",
                "mhd/biermann_coefficient=5.0",
                "mhd/biermann_subcycle=false",
                f"mhd/biermann_shock_suppression={suppress}",
                "mhd/biermann_shock_threshold=0.1",
                "mhd/biermann_shock_compression_threshold=0.02",
                "time/tlim=0.001", "time/align_outputs=true",
                "output1/dt=0.001", "output2/dt=0.001",
                "output2/data_format=%24.17e", "output3/dt=0.001",
                "output3/data_format=%24.17e", "output4/dt=-1",
            ])
            field, two_temp = load_profile(basename)
            legacy_mask_fields[suppress] = np.asarray(field["bcc3"])
            for key in ("eion", "eele", "tion", "tele"):
                assert np.all(np.isfinite(two_temp[key]))
                assert np.all(two_temp[key] > 0.0)
            assert_energy_conserved(basename)
        masked_norm = np.linalg.norm(legacy_mask_fields["true"])
        unmasked_norm = np.linalg.norm(legacy_mask_fields["false"])
        assert masked_norm > 0.0
        assert masked_norm < 0.25*unmasked_norm

        # Exercise the full 3D arithmetic flux-CT path on a small mesh.
        run_case("biermann_3d", 1.0, [
            "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=16",
            "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
            "output2/slice_x2=0.03125", "output2/slice_x3=0.03125",
            "output3/slice_x2=0.03125", "output3/slice_x3=0.03125",
        ])
        field3d = athena_read.tab("tab/biermann_3d.biermann.00001.tab")
        assert np.max(np.abs(field3d["bcc3"])) > 1.0e-7
        assert np.max(np.abs(field3d["bcc1"])) < 2.0e-12
        assert np.max(np.abs(field3d["bcc2"])) < 2.0e-12
        assert_energy_conserved("biermann_3d")
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
