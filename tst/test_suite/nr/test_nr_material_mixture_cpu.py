"""Regression tests for passive-scalar material closure and Spitzer exchange."""

import math

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_material_relax.athinput"


def sorted_tab(filename):
    data = athena_read.tab(filename)
    order = np.argsort(data["x1v"])
    return {name: np.atleast_1d(values)[order]
            for name, values in data.items() if np.ndim(values) > 0}


def expected_temperatures(fe, ratio, time, exchange_time):
    fi = 1.0 - fe
    tion0 = 1.0 / (fi + fe * ratio)
    tele0 = ratio * tion0
    decay = math.exp(-(1.0 + fe / fi) * time / exchange_time)
    delta = (tion0 - tele0) * decay
    return 1.0 + fe * delta, 1.0 - fi * delta


def spitzer_time_code(y0, tion, tele):
    electron_mass = 9.1093837015e-28
    electron_charge = 4.803204712570263e-10
    atomic_mass = 1.660538921e-24
    boltzmann = 1.3806488e-16
    if y0 == 1.0:
        abar, zbar, zeff = 6.5, 3.5, 37.0 / 7.0
    else:
        abar, zbar, zeff = 2.5, 1.0, 1.0
    ion_weight = 1.0 / abar
    electron_weight = zbar / abar
    mean_particle_mass = 1.0 / (ion_weight + electron_weight)
    kelvin_per_code = (1.0e8 ** 2 * mean_particle_mass * atomic_mass
                       / boltzmann)
    tion_kelvin = tion * kelvin_per_code
    tele_kelvin = tele * kelvin_per_code
    ne = electron_weight / atomic_mass
    ion_mass = abar * atomic_mass
    thermal_speed_squared = (boltzmann * tele_kelvin / electron_mass
                             + boltzmann * tion_kelvin / ion_mass)
    tau_seconds = (
        3.0 * electron_mass * ion_mass * thermal_speed_squared ** 1.5
        / (8.0 * math.sqrt(2.0 * math.pi) * ne * zeff
           * electron_charge ** 4 * 10.0))
    return tau_seconds / 1.0e-9


def test_run():
    try:
        assert testutils.run(input_file), "Two-material constant exchange failed."
        initial = sorted_tab("tab/two_material_relax.mhd_2t.00000.tab")
        final = sorted_tab("tab/two_material_relax.mhd_2t.00001.tab")
        scalar = sorted_tab("tab/two_material_relax.mhd_w_s.00001.tab")

        left = initial["x1v"] < 0.0
        right = ~left
        for mask, fe, exchange_time in ((left, 3.5 / 4.5, 0.2),
                                        (right, 0.5, 0.4)):
            tion0, tele0 = expected_temperatures(fe, 0.25, 0.0,
                                                 exchange_time)
            tion, tele = expected_temperatures(fe, 0.25, 0.05,
                                               exchange_time)
            assert np.allclose(initial["tion"][mask], tion0,
                               rtol=2.0e-12, atol=2.0e-12)
            assert np.allclose(initial["tele"][mask], tele0,
                               rtol=2.0e-12, atol=2.0e-12)
            assert np.allclose(final["tion"][mask], tion,
                               rtol=2.0e-10, atol=2.0e-10)
            assert np.allclose(final["tele"][mask], tele,
                               rtol=2.0e-10, atol=2.0e-10)
        assert np.allclose(final["eion"] + final["eele"], 1.5,
                           rtol=2.0e-12, atol=2.0e-12)
        assert np.all((scalar["s_00"] >= 0.0) & (scalar["s_00"] <= 1.0))

        # One first-order step freezes the state-dependent Spitzer coefficient at
        # the initial state, for which the exponential update is analytic.
        dt = 1.0e-3
        flags = [
            "job/basename=two_material_spitzer",
            "mhd/t_ei_model=spitzer",
            "mhd/t_ei_coulomb_log=10.0",
            "time/nlim=1", "time/tlim=1.0", f"time/initial_dt={dt}",
            f"output1/dt={dt}", f"output2/dt={dt}",
        ]
        assert testutils.run(input_file, flags=flags), (
            "Two-material Spitzer exchange failed.")
        spitzer_initial = sorted_tab(
            "tab/two_material_spitzer.mhd_2t.00000.tab")
        spitzer_final = sorted_tab(
            "tab/two_material_spitzer.mhd_2t.00001.tab")
        left = spitzer_initial["x1v"] < 0.0
        right = ~left
        for mask, y0, fe in ((left, 1.0, 3.5 / 4.5),
                             (right, 0.0, 0.5)):
            tion0 = spitzer_initial["tion"][mask][0]
            tele0 = spitzer_initial["tele"][mask][0]
            tau = spitzer_time_code(y0, tion0, tele0)
            fi = 1.0 - fe
            decay = math.exp(-(1.0 + fe / fi) * dt / tau)
            delta = (tion0 - tele0) * decay
            tion = 1.0 + fe * delta
            tele = 1.0 - fi * delta
            assert np.allclose(spitzer_final["tion"][mask], tion,
                               rtol=3.0e-10, atol=3.0e-10)
            assert np.allclose(spitzer_final["tele"][mask], tele,
                               rtol=3.0e-10, atol=3.0e-10)
        assert np.allclose(
            spitzer_final["eion"] + spitzer_final["eele"], 1.5,
            rtol=2.0e-12, atol=2.0e-12)

        # Primitive out-of-range input is converted to conservative rho*Y and clamped.
        clamp_flags = [
            "job/basename=two_material_clamp", "problem/yl=1.2",
            "problem/yr=-0.2", "time/nlim=1", "time/tlim=1.0e-6",
            "time/initial_dt=1.0e-6", "output1/dt=-1.0",
            "output2/dt=1.0e-6",
        ]
        assert testutils.run(input_file, flags=clamp_flags), (
            "Two-material scalar-clamp run failed.")
        clamped = sorted_tab("tab/two_material_clamp.mhd_w_s.00001.tab")
        assert np.all((clamped["s_00"] >= 0.0) & (clamped["s_00"] <= 1.0))
        assert np.allclose(clamped["s_00"][clamped["x1v"] < 0.0], 1.0,
                           rtol=0.0, atol=2.0e-14)
        assert np.allclose(clamped["s_00"][clamped["x1v"] > 0.0], 0.0,
                           rtol=0.0, atol=2.0e-14)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
