"""Analytic regression tests for Hamiltonian 2T laser refraction."""

import os

import numpy as np

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_cpu import read_laser_binary


input_file = "../../../inputs/mhd/two_temperature_laser_refraction.athinput"
ELECTRON_NUMBER_SCALE = 1.0e13


def critical_density_cgs(wavelength=1.0):
    electron_charge = 4.803204712570263e-10
    electron_mass = 9.1093837015e-28
    light_speed = 2.99792458e10
    return (electron_mass * np.pi * light_speed**2
            / (electron_charge**2 * wavelength**2))


def run_case(basename, flags):
    log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
    common = [
        f"job/basename={basename}",
        "time/tlim=1.0e-8",
        "output1/dt=1.0e-8",
    ]
    assert testutils.run(input_file, flags=common + flags), (
        f"{basename} refractive laser run failed.")
    data = read_laser_binary(
        f"bin/{basename}.laser_refractive.00001.bin")
    result = {
        name: sum(np.asarray(block).sum()
                  for block in data["mb_data"][name])
        for name in (
            "laser_path", "laser_dir1", "laser_dir2",
            "laser_dispersion_error", "laser_x1_moment",
            "laser_x2_moment")
    }
    with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
        stream.seek(log_offset)
        lines = [line for line in stream
                 if line.startswith("laser: launched=")]
    assert lines, "refractive laser accounting line is missing"
    diagnostics = dict(token.split("=", 1) for token in lines[-1].split()[1:])
    result["iterations"] = int(diagnostics["iterations"])
    result["waves"] = int(diagnostics["waves"])
    return result


def integrate_reference(y, dydx, points=200001):
    x = np.linspace(0.0, 1.0, points)
    arc_factor = np.sqrt(1.0 + dydx(x)**2)
    return (np.trapezoid(arc_factor, x),
            np.trapezoid(y(x) * arc_factor, x))


def test_run():
    try:
        critical_density = critical_density_cgs()
        number_to_critical = ELECTRON_NUMBER_SCALE / critical_density
        y0 = 0.19921875

        # Uniform density: refraction must reduce exactly to a straight ray.
        uniform = run_case("laser_refract_uniform", [])
        assert 0 < uniform["iterations"] < 256*uniform["waves"]
        assert abs(uniform["laser_path"] - 1.0) < 2.0e-12
        assert abs(uniform["laser_dir1"] - 1.0) < 2.0e-12
        assert abs(uniform["laser_dir2"]) < 2.0e-12
        assert abs(uniform["laser_x2_moment"] / uniform["laser_path"] - y0) < 2.0e-12
        assert uniform["laser_dispersion_error"] < 1.0e-11

        # A constant transverse density gradient has a parabolic analytic path.
        gradient = 0.1
        grad_mu = number_to_critical * gradient
        mu_initial = number_to_critical * (0.5 + gradient * y0)
        qx = np.sqrt(1.0 - mu_initial)
        y_gradient = lambda x: y0 - grad_mu * x**2 / (4.0 * qx**2)
        dydx_gradient = lambda x: -grad_mu * x / (2.0 * qx**2)
        path_ref, y_moment_ref = integrate_reference(
            y_gradient, dydx_gradient)
        bent = run_case("laser_refract_gradient", [
            f"problem/density_gradient_x2={gradient}",
        ])
        assert np.isclose(bent["laser_path"], path_ref, rtol=2.0e-5)
        assert np.isclose(bent["laser_x2_moment"], y_moment_ref,
                          rtol=2.0e-5, atol=2.0e-7)
        assert np.isclose(bent["laser_dir2"], y_gradient(1.0) - y0,
                          rtol=2.0e-5, atol=2.0e-7)

        # Opposite launches through a quadratic density lens must be mirror images.
        curvature = 2.0
        lens_flags = [f"problem/density_curvature_x2={curvature}"]
        plus = run_case("laser_refract_lens_plus", lens_flags)
        minus = run_case("laser_refract_lens_minus", lens_flags + [
            f"laser/beam0_origin_x2={-y0}",
        ])
        assert np.isclose(plus["laser_path"], minus["laser_path"],
                          rtol=2.0e-12, atol=2.0e-12)
        assert np.isclose(plus["laser_x2_moment"],
                          -minus["laser_x2_moment"],
                          rtol=2.0e-7, atol=1.0e-8)
        assert np.isclose(plus["laser_dir2"], -minus["laser_dir2"],
                          rtol=2.0e-7, atol=1.0e-8)

        # The KDK trajectory converges at second order as the cell step is reduced.
        lens_mu = number_to_critical * curvature
        qx_lens = np.sqrt(
            1.0 - number_to_critical * (0.5 + curvature * y0**2))
        frequency = np.sqrt(lens_mu) / qx_lens
        y_lens = lambda x: y0 * np.cos(frequency * x)
        dydx_lens = lambda x: -y0 * frequency * np.sin(frequency * x)
        path_lens_ref, y_lens_moment_ref = integrate_reference(
            y_lens, dydx_lens)
        fractions = np.asarray((1.0, 0.5, 0.25, 0.125))
        trajectories = []
        dispersion = []
        for fraction in fractions:
            result = run_case(f"laser_refract_step_{fraction}", lens_flags + [
                f"laser/refractive_cell_fraction={fraction}",
                "laser/refractive_curvature_fraction=1.0",
            ])
            assert np.isclose(result["laser_path"], path_lens_ref, rtol=2.0e-4)
            assert np.isclose(result["laser_x2_moment"], y_lens_moment_ref,
                              rtol=2.0e-3, atol=2.0e-4)
            trajectories.append(result["laser_x2_moment"])
            dispersion.append(result["laser_dispersion_error"] /
                              result["laser_path"])
        # Self-convergence removes the fixed grid-reconstruction error while retaining
        # the KDK trajectory error. Binary output is intentionally single precision.
        errors = np.abs(np.diff(trajectories))
        observed_order = np.polyfit(
            np.log(fractions[:-1]), np.log(errors), 1)[0]
        assert observed_order > 1.7
        assert errors[-1] < errors[0] / 8.0
        assert max(dispersion) < 5.0e-3
    finally:
        testutils.cleanup()
