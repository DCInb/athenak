"""Material-aware inverse-bremsstrahlung regression test."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


base_input = Path("../../../inputs/mhd/two_temperature_laser.athinput")
material_input = Path("two_material_laser.athinput")
source_dt = 1.0e-6


def write_material_input():
    text = base_input.read_text(encoding="ascii")
    text = text.replace("<mhd>\n", "<mhd>\nnscalars = 1\n", 1)
    text = text.replace(
        "temperature_scale_cgs = 1.0\n",
        "temperature_scale_cgs = 1.0\n"
        "temperature_mean_molecular_weight = 1.0\n", 1)
    text = text.replace("<problem>\n", "<problem>\nyl = 1.0\nyr = 1.0\n", 1)
    text += """

<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
material0_zbar = 3.5
material0_zeff = 5.285714285714286
material1_name = DT
material1_abar = 2.5
material1_zbar = 1.0
material1_zeff = 1.0
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
    length_scale = 1.0e-2
    density_scale = 1.0e-3
    temperature_scale = 1.0e6
    wavelength_code = 1.0e-2
    flags = [
        f"job/basename={basename}",
        "time/integrator=rk1", f"time/tlim={source_dt}",
        "output1/dt=-1.0", f"output2/dt={source_dt}",
        f"output3/dt={source_dt}", "output4/dt=-1.0",
        "laser/absorption_model=inverse_bremsstrahlung",
        f"laser/length_scale_cgs={length_scale}",
        f"laser/density_scale_cgs={density_scale}",
        f"laser/temperature_scale_cgs={temperature_scale}",
        "laser/temperature_mean_molecular_weight=1.0",
        f"laser/beam0_wavelength={wavelength_code}",
        f"laser/beam0_zeff={beam_zeff}",
        f"problem/yl={y0}", f"problem/yr={y0}",
    ]
    assert testutils.run(str(material_input), flags=flags), f"{basename} failed."
    data = athena_read.tab(f"tab/{basename}.laser.00001.tab")
    order = np.argsort(data["x1v"])
    return ({name: np.atleast_1d(values)[order]
             for name, values in data.items() if np.ndim(values) > 0},
            length_scale, density_scale, temperature_scale, wavelength_code)


def test_run():
    try:
        write_material_input()
        atomic_mass = 1.660538921e-24
        for name, y0, abar, zbar, zeff in (
                ("material_laser_ch", 1.0, 6.5, 3.5, 37.0 / 7.0),
                ("material_laser_dt", 0.0, 2.5, 1.0, 1.0)):
            laser, length, density, temperature, wavelength = run_material(
                name, y0, 9.0)
            mean_particle_mass = abar / (1.0 + zbar)
            ne = density * zbar / abar / atomic_mass
            te = temperature * mean_particle_mass
            coefficient = inverse_bremsstrahlung(
                ne, te, zeff, wavelength * length) * length
            dx = laser["x1v"][1] - laser["x1v"][0]
            left_edge = laser["x1v"] - 0.5 * dx
            expected = (np.exp(-coefficient * left_edge)
                        * -np.expm1(-coefficient * dx) / dx)
            assert np.allclose(laser["laser_q"], expected,
                               rtol=3.0e-11, atol=3.0e-13)
            assert np.allclose(laser["laser_tau"], coefficient * dx,
                               rtol=3.0e-11, atol=3.0e-13)

        # A per-beam Zeff change cannot affect a material-aware coefficient.
        ch_reference, *_ = run_material("material_laser_ch_ref", 1.0, 1.0)
        ch_zeff9 = athena_read.tab("tab/material_laser_ch.laser.00001.tab")
        order = np.argsort(ch_zeff9["x1v"])
        assert np.array_equal(ch_reference["laser_q"],
                              np.atleast_1d(ch_zeff9["laser_q"])[order])
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        material_input.unlink(missing_ok=True)
        testutils.cleanup()
