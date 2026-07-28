"""Tabular-material inverse-bremsstrahlung regression test."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


base_input = Path("../../../inputs/mhd/two_temperature_laser.athinput")
material_input = Path("tabular_material_laser.athinput")
ch_table = Path("tabular_laser_ch.2t_eos")
he_table = Path("tabular_laser_he.2t_eos")
source_dt = 1.0e-6
atomic_mass = 1.660538921e-24
table_density_to_cgs = 1.0e-3
table_temperature_to_kelvin = 1.0e5


def write_table(path, abar, ion_pressure, electron_pressure,
                ion_energy, electron_energy, ionization):
    """Write a two-density, two-temperature native 2T EOS fixture."""
    path.write_text(
        "\n".join([
            "athenak_two_temperature_eos 1",
            "dimensions 2 2",
            f"abar {abar}",
            "density",
            "5.0e-4 2.0e-3",
            "temperature",
            "1.0e6 1.0e7",
            "ion_pressure",
            f"{0.5*ion_pressure} {5.0*ion_pressure}",
            f"{2.0*ion_pressure} {20.0*ion_pressure}",
            "electron_pressure",
            f"{0.5*electron_pressure} {5.0*electron_pressure}",
            f"{2.0*electron_pressure} {20.0*electron_pressure}",
            "ion_specific_internal_energy",
            f"{ion_energy} {10.0*ion_energy}",
            f"{ion_energy} {10.0*ion_energy}",
            "electron_specific_internal_energy",
            f"{electron_energy} {10.0*electron_energy}",
            f"{electron_energy} {10.0*electron_energy}",
            "mean_ionization",
            f"{ionization} {10.0*ionization}",
            f"{ionization} {10.0*ionization}",
            "end",
            "",
        ]), encoding="ascii")


def prepare_case():
    # At rho=1 and the table floor, both generated states have Te=1.0e6 K;
    # their low-temperature mean ionizations are 0.2 (CH) and 0.4 (He).
    write_table(ch_table, 6.5, 10.0, 20.0, 30.0, 40.0, 0.2)
    write_table(he_table, 4.0, 5.0, 15.0, 40.0, 100.0, 0.4)

    text = base_input.read_text(encoding="ascii")
    text = text.replace("<mhd>\n", "<mhd>\nnscalars = 1\n", 1)
    text = text.replace("rsolver = hlle", "rsolver = llf", 1)
    text = text.replace("<problem>\n", "<problem>\nyl = 1.0\nyr = 1.0\n", 1)
    text += f"""

<units>
length_cgs = 10.0
mass_cgs = 1.0
time_cgs = 1.0
mu = 1.0

<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
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
eos_table_density_to_cgs = {table_density_to_cgs}
eos_table_temperature_to_kelvin = {table_temperature_to_kelvin}
eos_table_pressure_from_cgs = 1.0
eos_table_specific_energy_from_cgs = 1.0
"""
    material_input.write_text(text, encoding="ascii")


def inverse_bremsstrahlung(ne, te, zeff, wavelength):
    electron_charge = 4.803204712570263e-10
    electron_mass = 9.1093837015e-28
    boltzmann = 1.380649e-16
    light_speed = 2.99792458e10
    critical_density = (electron_mass * np.pi * light_speed**2
                        / (electron_charge**2 * wavelength**2))
    density_ratio = min(ne / critical_density, 1.0 - 1.0e-12)
    coulomb_argument = (
        3.0 / (2.0 * zeff * electron_charge**3)
        * np.sqrt(boltzmann**3 * te**3 / (np.pi * ne)))
    coulomb_log = max(np.log(max(coulomb_argument, 1.0)), 1.0)
    collision_frequency = (
        4.0 / 3.0 * np.sqrt(2.0 * np.pi / electron_mass)
        * ne * zeff * electron_charge**4 * coulomb_log
        / (boltzmann * te)**1.5)
    group_speed = light_speed * np.sqrt(max(1.0 - density_ratio, 1.0e-12))
    return density_ratio * collision_frequency / group_speed


def run_material(basename, y0, beam_zeff):
    length_scale = 10.0
    wavelength_code = 1.0e-5
    flags = [
        f"job/basename={basename}",
        "time/integrator=rk1", f"time/tlim={source_dt}",
        "output1/dt=-1.0", f"output2/dt={source_dt}",
        f"output3/dt={source_dt}", "output4/dt=-1.0",
        "laser/absorption_model=inverse_bremsstrahlung",
        f"laser/length_scale_cgs={length_scale}",
        # Deliberately conflict with the table scales: tabular optics must ignore these.
        "laser/density_scale_cgs=9.0",
        "laser/temperature_scale_cgs=9.0e9",
        f"laser/beam0_wavelength={wavelength_code}",
        f"laser/beam0_zeff={beam_zeff}",
        f"problem/yl={y0}", f"problem/yr={y0}",
    ]
    assert testutils.run(str(material_input), flags=flags), f"{basename} failed."
    data = athena_read.tab(f"tab/{basename}.laser.00001.tab")
    order = np.argsort(data["x1v"])
    return ({name: np.atleast_1d(values)[order]
             for name, values in data.items() if np.ndim(values) > 0},
            length_scale, wavelength_code)


def test_run():
    try:
        prepare_case()
        for name, y0, abar, zbar_table, zeff, temperature in (
                ("tabular_laser_ch", 1.0, 6.5, 0.2,
                 (5.285714285714286/3.5)*0.2, 10.0),
                ("tabular_laser_he", 0.0, 4.0, 0.4, 0.4, 10.0)):
            laser, length, wavelength = run_material(name, y0, 9.0)
            ne = table_density_to_cgs*zbar_table/(abar*atomic_mass)
            te = temperature*table_temperature_to_kelvin
            coefficient = inverse_bremsstrahlung(
                ne, te, zeff, wavelength*length)*length
            dx = laser["x1v"][1]-laser["x1v"][0]
            left_edge = laser["x1v"]-0.5*dx
            expected = (np.exp(-coefficient*left_edge)
                        * -np.expm1(-coefficient*dx)/dx)
            assert np.allclose(laser["laser_q"], expected,
                               rtol=3.0e-11, atol=3.0e-13)
            assert np.allclose(laser["laser_tau"], coefficient*dx,
                               rtol=3.0e-11, atol=3.0e-13)

        # A beam input cannot override the material cache's dynamic Zeff.
        reference, *_ = run_material("tabular_laser_ch_ref", 1.0, 1.0)
        changed = athena_read.tab("tab/tabular_laser_ch.laser.00001.tab")
        order = np.argsort(changed["x1v"])
        assert np.array_equal(reference["laser_q"],
                              np.atleast_1d(changed["laser_q"])[order])
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        for path in (material_input, ch_table, he_table):
            path.unlink(missing_ok=True)
        testutils.cleanup()
