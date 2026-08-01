"""Tabular-material regression for cached Biermann thermodynamics."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


base_input = Path("../../../inputs/mhd/two_temperature_biermann.athinput")
material_input = Path("tabular_biermann.athinput")
ch_table = Path("tabular_biermann_ch.2t_eos")
he_table = Path("tabular_biermann_he.2t_eos")
T_FINAL = 1.0e-4
COEFFICIENT = 0.1
DENSITY_AMPLITUDE = 0.2
PRESSURE_AMPLITUDE = 0.2
Y_SLICE = 0.0078125
ELECTRON_PRESSURE_FRACTION = 0.8
CH_IONIZATION = 0.25
CH_ABAR = 6.5
MINIMUM_ELECTRON_FRACTION = 1.0e-12
DEFAULT_CFL = 0.3
SUBCYCLE_COEFFICIENT = 5.0
SUBCYCLE_CFL = 0.15
FINE_LEGACY_CFL = 0.0375


def write_table(path, abar, ionization):
    """Write an exact gamma-law table with Pe=0.8*rho*T and fixed Zbar."""
    path.write_text(
        "\n".join([
            "athenak_two_temperature_eos 1",
            "dimensions 2 2",
            f"abar {abar}",
            "density",
            "1.0 4.0",
            "temperature",
            "0.5 2.0",
            "ion_pressure",
            "0.05 0.2",
            "0.2 0.8",
            "electron_pressure",
            "0.2 0.8",
            "0.8 3.2",
            "ion_specific_internal_energy",
            "0.15 0.6",
            "0.15 0.6",
            "electron_specific_internal_energy",
            "0.6 2.4",
            "0.6 2.4",
            "mean_ionization",
            f"{ionization} {ionization}",
            f"{ionization} {ionization}",
            "end",
            "",
        ]), encoding="ascii")


def prepare_case(ch_ionization=CH_IONIZATION, he_ionization=0.5):
    write_table(ch_table, CH_ABAR, ch_ionization)
    write_table(he_table, 4.0, he_ionization)
    text = base_input.read_text(encoding="ascii")
    text = text.replace("<mhd>\n", "<mhd>\nnscalars = 1\n", 1)
    text = text.replace("rsolver = hlle", "rsolver = llf", 1)
    text = text.replace(
        "biermann_coefficient = 0.1\n",
        "biermann_coefficient = 0.1\n"
        f"biermann_minimum_electron_fraction = {MINIMUM_ELECTRON_FRACTION}\n",
        1)
    text = text.replace(
        "<problem>\n", "<problem>\nmaterial0_fraction = 1.0\n", 1)
    text += f"""

<units>
length_cgs = 1.0
mass_cgs = 2.0
time_cgs = 1.0
mu = 1.0

<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = {CH_ABAR}
material0_zbar = 3.5
material0_zeff = 5.285714285714286
material1_name = He
material1_abar = 4.0
material1_zbar = 2.0
material1_zeff = 2.0
material0_eos_table_file = {ch_table.resolve()}
material1_eos_table_file = {he_table.resolve()}
eos_table_bounds = clamp
eos_table_interpolation = geometric
eos_table_density_to_cgs = 2.0
eos_table_temperature_to_kelvin = 1.0
eos_table_pressure_from_cgs = 1.0
eos_table_specific_energy_from_cgs = 1.0
"""
    material_input.write_text(text, encoding="ascii")


def run_case(basename, coefficient=COEFFICIENT, extra_flags=None):
    Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
    flags = [
        f"job/basename={basename}",
        f"mhd/biermann_coefficient={coefficient}",
        "problem/material0_fraction=1.0",
    ]
    if extra_flags:
        flags.extend(extra_flags)
    assert testutils.run(str(material_input), flags=flags), f"{basename} failed."


def analytic_b3_rate(x1, y_slice=Y_SLICE, coefficient=COEFFICIENT):
    wave_number = 2.0*np.pi
    density = np.exp(DENSITY_AMPLITUDE*np.sin(wave_number*y_slice))
    pressure = np.exp(PRESSURE_AMPLITUDE*np.sin(wave_number*x1))
    electron_density_factor = CH_IONIZATION/CH_ABAR
    pressure_over_density_factor = (
        ELECTRON_PRESSURE_FRACTION/electron_density_factor)
    return (coefficient*pressure_over_density_factor*(pressure/density)
            * PRESSURE_AMPLITUDE*DENSITY_AMPLITUDE*wave_number**2
            * np.cos(wave_number*x1)*np.cos(wave_number*y_slice))


def expected_biermann_dt(coefficient):
    resolution = 64
    dx = 1.0/resolution
    centers = -0.5+(np.arange(resolution)+0.5)*dx
    x1, x2 = np.meshgrid(centers, centers, indexing="xy")
    density = np.exp(DENSITY_AMPLITUDE*np.sin(2.0*np.pi*x2))
    pressure = np.exp(PRESSURE_AMPLITUDE*np.sin(2.0*np.pi*x1))
    electron_density = density*CH_IONIZATION/CH_ABAR
    # NewTimeStep uses the larger adjacent one-sided slope so that a Nyquist
    # checkerboard cannot disappear from the explicit stability bound.
    log_ne = np.log(electron_density)
    dln2 = np.maximum(
        np.abs(np.roll(log_ne, -1, axis=0)-log_ne),
        np.abs(log_ne-np.roll(log_ne, 1, axis=0)))/dx
    pe = ELECTRON_PRESSURE_FRACTION*pressure
    gm1 = 2.0/3.0
    vtm = (coefficient*np.sqrt(gm1*pe)/electron_density*np.abs(dln2))
    return DEFAULT_CFL*dx/np.max(vtm)


def relative_l2_norm(left, right):
    denominator = np.linalg.norm(right)
    assert denominator > 0.0
    return np.linalg.norm(left-right)/denominator


def test_run():
    try:
        prepare_case()
        run_case("tabular_biermann")
        field = athena_read.tab("tab/tabular_biermann.biermann.00001.tab")
        two_temp = athena_read.tab(
            "tab/tabular_biermann.two_temperature.00001.tab")
        order = np.argsort(field["x1v"])
        numerical_rate = np.atleast_1d(field["bcc3"])[order]/T_FINAL
        exact_rate = analytic_b3_rate(np.atleast_1d(field["x1v"])[order])
        relative_l2 = (np.linalg.norm(numerical_rate-exact_rate)
                       / np.linalg.norm(exact_rate))
        assert relative_l2 < 1.5e-2
        assert np.max(np.abs(field["bcc1"])) < 2.0e-12
        assert np.max(np.abs(field["bcc2"])) < 2.0e-12
        assert np.all(np.isfinite(numerical_rate))
        assert np.all(two_temp["eion"] > 0.0)
        assert np.all(two_temp["eele"] > 0.0)
        history = athena_read.hst("tabular_biermann.mhd.hst")
        assert np.allclose(history["tot-E"], history["tot-E"][0],
                           rtol=2.0e-12, atol=2.0e-12)

        # A large C_B isolates the cached Pe/ne thermal-magnetic timestep.
        cfl_coefficient = 100.0
        run_case("tabular_biermann_cfl", cfl_coefficient, [
            "time/nlim=1", "time/tlim=1.0",
        ])
        cfl_history = athena_read.hst("tabular_biermann_cfl.mhd.hst")
        assert np.isfinite(cfl_history["dt"][0])
        assert np.isclose(cfl_history["dt"][0],
                          expected_biermann_dt(cfl_coefficient),
                          rtol=3.0e-10, atol=0.0)

        # Exercise the multirate path with a battery limit well below the macro
        # interval, then compare it with a finely stepped legacy calculation.  The
        # exact gamma-law table makes the component/total closure and cache relations
        # independently testable after every Biermann stage has completed.
        common_subcycle_flags = [
            "mhd/biermann_shock_suppression=false",
        ]
        run_case("tabular_biermann_subcycle", SUBCYCLE_COEFFICIENT, [
            *common_subcycle_flags,
            "mhd/biermann_subcycle=true",
            f"mhd/biermann_subcycle_cfl={SUBCYCLE_CFL}",
        ])
        run_case("tabular_biermann_fine_legacy", SUBCYCLE_COEFFICIENT, [
            *common_subcycle_flags,
            "mhd/biermann_subcycle=false",
            f"time/cfl_number={FINE_LEGACY_CFL}",
        ])

        subcycle_field = athena_read.tab(
            "tab/tabular_biermann_subcycle.biermann.00001.tab")
        subcycle_two_temp = athena_read.tab(
            "tab/tabular_biermann_subcycle.two_temperature.00001.tab")
        legacy_field = athena_read.tab(
            "tab/tabular_biermann_fine_legacy.biermann.00001.tab")
        legacy_two_temp = athena_read.tab(
            "tab/tabular_biermann_fine_legacy.two_temperature.00001.tab")
        subcycle_history = athena_read.hst(
            "tabular_biermann_subcycle.mhd.hst")
        legacy_history = athena_read.hst(
            "tabular_biermann_fine_legacy.mhd.hst")

        ordinary_battery_dt = expected_biermann_dt(SUBCYCLE_COEFFICIENT)
        assert ordinary_battery_dt < 0.5*T_FINAL
        assert np.isclose(subcycle_history["dt"][0], T_FINAL,
                          rtol=2.0e-13, atol=0.0)
        assert len(subcycle_history["time"])-1 == 1
        assert np.isclose(legacy_history["dt"][0],
                          ordinary_battery_dt*FINE_LEGACY_CFL/DEFAULT_CFL,
                          rtol=3.0e-10, atol=0.0)
        assert np.linalg.norm(subcycle_field["bcc3"]) > 1.0e-2

        for field in (subcycle_field, legacy_field):
            for key in ("dens", "eint"):
                assert np.all(np.isfinite(field[key]))
                assert np.all(field[key] > 0.0)
            assert np.all(field["s_00"] == 1.0)
        for two_temp in (subcycle_two_temp, legacy_two_temp):
            for key in ("eion", "eele", "tion", "tele"):
                assert np.all(np.isfinite(two_temp[key]))
                assert np.all(two_temp[key] > 0.0)
            assert np.count_nonzero(two_temp["eos_flags"]) == 0

        # eint is an energy density; the tabular component energies are specific.
        for field, two_temp in (
                (subcycle_field, subcycle_two_temp),
                (legacy_field, legacy_two_temp)):
            component_energy_density = field["dens"]*(
                two_temp["eion"]+two_temp["eele"])
            assert np.allclose(field["eint"], component_energy_density,
                               rtol=3.0e-11, atol=3.0e-12)
            assert np.allclose(two_temp["tion"], two_temp["eion"]/0.3,
                               rtol=3.0e-12, atol=3.0e-12)
            assert np.allclose(two_temp["tele"], two_temp["eele"]/1.2,
                               rtol=3.0e-12, atol=3.0e-12)

        for history in (subcycle_history, legacy_history):
            assert np.all(np.isfinite(history["tot-E"]))
            assert np.allclose(history["tot-E"], history["tot-E"][0],
                               rtol=2.0e-12, atol=2.0e-12)
        assert relative_l2_norm(subcycle_field["bcc3"],
                                legacy_field["bcc3"]) < 5.0e-4
        assert relative_l2_norm(subcycle_field["eint"],
                                legacy_field["eint"]) < 1.0e-5
        for key in ("eion", "eele", "tion", "tele"):
            assert relative_l2_norm(subcycle_two_temp[key],
                                    legacy_two_temp[key]) < 3.0e-5

        # A table can retain a numerical electron-pressure floor while its physical
        # ionization is effectively zero.  q_e=1e-200 would underflow q_e**2 and make
        # the old vTM expression pathological; the regularized plasma activation must
        # suppress both the battery and its timestep constraint.
        prepare_case(ch_ionization=CH_ABAR*1.0e-200)
        run_case("tabular_biermann_neutral", cfl_coefficient)
        neutral_field = athena_read.tab(
            "tab/tabular_biermann_neutral.biermann.00001.tab")
        assert np.count_nonzero(neutral_field["bcc1"]) == 0
        assert np.count_nonzero(neutral_field["bcc2"]) == 0
        assert np.count_nonzero(neutral_field["bcc3"]) == 0

        one_step = ["time/nlim=1", "time/tlim=1.0"]
        run_case("tabular_biermann_neutral_cfl", cfl_coefficient, one_step)
        run_case("tabular_biermann_neutral_control", 0.0, one_step)
        run_case("tabular_biermann_neutral_subcycle", cfl_coefficient, [
            *one_step,
            "mhd/biermann_subcycle=true",
        ])
        neutral_history = athena_read.hst(
            "tabular_biermann_neutral_cfl.mhd.hst")
        control_history = athena_read.hst(
            "tabular_biermann_neutral_control.mhd.hst")
        neutral_subcycle_history = athena_read.hst(
            "tabular_biermann_neutral_subcycle.mhd.hst")
        neutral_subcycle_field = athena_read.tab(
            "tab/tabular_biermann_neutral_subcycle.biermann.00001.tab")
        assert np.isfinite(neutral_history["dt"][0])
        assert neutral_history["dt"][0] == control_history["dt"][0]
        assert neutral_subcycle_history["dt"][0] == control_history["dt"][0]
        for key in ("bcc1", "bcc2", "bcc3"):
            assert np.count_nonzero(neutral_subcycle_field[key]) == 0

        # Resolve the complete neutral-activation transition.  With a spatially
        # constant q_e, the generated field scales exactly as S(q_e)/q_e.  At
        # q_e=1.25*q_min the cubic smoothstep is S(0.25)=0.15625, while at
        # q_e=3*q_min it is fully active, giving a field-norm ratio of 0.375.
        activation_coefficient = COEFFICIENT*MINIMUM_ELECTRON_FRACTION
        activation_fields = {}
        for label, electron_fraction in (
                ("below", 0.75*MINIMUM_ELECTRON_FRACTION),
                ("ramp", 1.25*MINIMUM_ELECTRON_FRACTION),
                ("full", 3.0*MINIMUM_ELECTRON_FRACTION)):
            prepare_case(ch_ionization=CH_ABAR*electron_fraction)
            basename = f"tabular_biermann_activation_{label}"
            run_case(basename, activation_coefficient, [
                "mhd/biermann_subcycle=true",
                "mhd/biermann_shock_suppression=false",
            ])
            activation_fields[label] = athena_read.tab(
                f"tab/{basename}.biermann.00001.tab")["bcc3"]

        assert np.count_nonzero(activation_fields["below"]) == 0
        ramp_norm = np.linalg.norm(activation_fields["ramp"])
        full_norm = np.linalg.norm(activation_fields["full"])
        assert ramp_norm > 0.0 and full_norm > 0.0
        assert np.isclose(ramp_norm/full_norm, 0.375,
                          rtol=2.0e-10, atol=0.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        for path in (material_input, ch_table, he_table):
            path.unlink(missing_ok=True)
        testutils.cleanup()
