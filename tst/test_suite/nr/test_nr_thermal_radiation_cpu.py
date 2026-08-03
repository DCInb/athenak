"""Regression tests for 2T multigroup thermal radiation on CPUs."""

import math

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


relax_input = "../../../inputs/hydro/two_temperature_mgfld.athinput"
diffusion_input = "../../../inputs/hydro/mgfld_diffusion.athinput"
table_diffusion_input = (
    "../../../inputs/hydro/mgfld_opacity_table_diffusion.athinput")
opacity_input = "../../../inputs/hydro/two_temperature_opacity_table.athinput"
opacity_table = "../../../inputs/hydro/two_temperature_opacity_table.dat"


def planck_integral(x):
    """Match the cancellation-safe device implementation used by the solver."""
    infinity = 6.4939394022668291491
    if x <= 0.0:
        return 0.0
    if x >= 50.0:
        return infinity
    if x < 0.5:
        x2 = x*x
        x3 = x2*x
        return (x3/3.0 - x3*x/8.0 + x3*x2/60.0
                - x3*x2*x2/5040.0 + x3*x2*x2*x2/272160.0
                - x3*x2*x2*x2*x2/13305600.0)

    tail = 0.0
    for n in range(1, 65):
        invn = 1.0/n
        invn2 = invn*invn
        tail += math.exp(-n*x)*(x*x*x*invn + 3.0*x*x*invn2
                                + 6.0*x*invn2*invn + 6.0*invn2*invn2)
    return min(max(infinity-tail, 0.0), infinity)


def planck_group_fraction(lower, upper, temperature):
    infinity = 6.4939394022668291491
    return ((planck_integral(upper/temperature)
             - planck_integral(lower/temperature))/infinity)


def test_run():
    try:
        assert testutils.run(relax_input), "Matter-radiation relaxation run failed."
        initial = athena_read.tab(
            "tab/two_temperature_mgfld.hydro_3t.00000.tab")
        final = athena_read.tab(
            "tab/two_temperature_mgfld.hydro_3t.00001.tab")

        energy_initial = initial["eion"] + initial["eele"] + initial["erad"]
        energy_final = final["eion"] + final["eele"] + final["erad"]
        assert np.allclose(energy_final, energy_initial, rtol=2.0e-11, atol=2.0e-11)
        assert np.all(final["tele"] < initial["tele"])
        assert np.all(final["erad"] > initial["erad"])
        assert np.allclose(final["erad"],
                           final["erad00"] + final["erad01"] + final["erad02"],
                           rtol=2.0e-11, atol=2.0e-11)
        assert np.all(final["erad00"] >= 0.0)
        assert np.all(final["erad01"] >= 0.0)
        assert np.all(final["erad02"] >= 0.0)

        assert testutils.run(diffusion_input), "Multigroup FLD transport run failed."
        initial = athena_read.tab("tab/mgfld_diffusion.hydro_3t.00000.tab")
        final = athena_read.tab("tab/mgfld_diffusion.hydro_3t.00001.tab")

        # Periodic conservative diffusion retains group-integrated energy while reducing
        # the variance of the radiation-energy step.
        assert np.isclose(np.sum(final["erad"]), np.sum(initial["erad"]),
                          rtol=2.0e-11, atol=2.0e-11)
        assert np.var(final["erad"]) < np.var(initial["erad"])
        assert np.all(final["erad00"] >= 0.0)
        assert np.all(final["erad01"] >= 0.0)
        assert np.allclose(final["eion"], initial["eion"], rtol=0.0, atol=2.0e-12)
        assert np.allclose(final["eele"], initial["eele"], rtol=0.0, atol=2.0e-12)

        table_diffusion_flags = [
            f"thermal_radiation/opacity_table_file={opacity_table}",
        ]
        assert testutils.run(table_diffusion_input, flags=table_diffusion_flags), (
            "Tabulated transport-opacity run failed.")
        initial = athena_read.tab(
            "tab/mgfld_opacity_table_diffusion.hydro_3t.00000.tab")
        final = athena_read.tab(
            "tab/mgfld_opacity_table_diffusion.hydro_3t.00001.tab")
        assert np.isclose(np.sum(final["erad"]), np.sum(initial["erad"]),
                          rtol=2.0e-11, atol=2.0e-11)
        assert np.var(final["erad"]) < np.var(initial["erad"])

        opacity_cases = {
            "linear": ([1.25, 2.50], [1.875, 3.125]),
            "log": ([0.60, 1.20], [0.900, 1.500]),
        }
        for interpolation, (absorption, emission) in opacity_cases.items():
            basename = f"opacity_{interpolation}"
            flags = [
                f"job/basename={basename}",
                f"thermal_radiation/opacity_interpolation={interpolation}",
                f"thermal_radiation/opacity_table_file={opacity_table}",
            ]
            assert testutils.run(opacity_input, flags=flags), (
                f"{interpolation} opacity-table run failed.")
            initial = athena_read.tab(f"tab/{basename}.hydro_3t.00000.tab")
            final = athena_read.tab(f"tab/{basename}.hydro_3t.00001.tab")

            density = 2.0
            dt = 0.01
            arad = 0.1
            tele = initial["tele"]
            blackbody = arad*tele**4
            bounds = [0.0, 1.0, 100.0]
            for group in range(2):
                fraction = planck_group_fraction(
                    bounds[group], bounds[group+1], tele[0])
                old = density*initial[f"erad0{group}"]
                siga = density*absorption[group]
                sige = density*emission[group]
                expected = (old + dt*sige*blackbody*fraction)/(1.0 + dt*siga)
                assert np.allclose(final[f"erad0{group}"], expected/density,
                                   rtol=3.0e-12, atol=3.0e-13)

            total_initial = initial["eion"] + initial["eele"] + initial["erad"]
            total_final = final["eion"] + final["eele"] + final["erad"]
            assert np.allclose(total_final, total_initial,
                               rtol=3.0e-12, atol=3.0e-13)
            assert np.all(final["tele"] < initial["tele"])

        # Force emission to exceed the available electron energy.  This
        # exercises the positive-change limiter with non-unit density and
        # unequal group updates, so cached energy density cannot be confused
        # with the final primitive specific energy.
        basename = "opacity_emission_limited"
        flags = [
            f"job/basename={basename}",
            "thermal_radiation/opacity_interpolation=linear",
            f"thermal_radiation/opacity_table_file={opacity_table}",
            "thermal_radiation/arad=100.0",
            "thermal_radiation/initial_radiation_temperature=0.0",
        ]
        assert testutils.run(opacity_input, flags=flags), (
            "Emission-limited opacity-table run failed.")
        initial = athena_read.tab(f"tab/{basename}.hydro_3t.00000.tab")
        final = athena_read.tab(f"tab/{basename}.hydro_3t.00001.tab")

        density = 2.0
        dt = 0.01
        tele = initial["tele"]
        blackbody = 100.0*tele**4
        bounds = [0.0, 1.0, 100.0]
        absorption = [1.25, 2.50]
        emission = [1.875, 3.125]
        raw_updates = []
        old_groups = []
        for group in range(2):
            old = density*initial[f"erad0{group}"]
            fraction = planck_group_fraction(
                bounds[group], bounds[group+1], tele[0])
            raw = ((old + dt*density*emission[group]*blackbody*fraction)
                   /(1.0 + dt*density*absorption[group]))
            old_groups.append(old)
            raw_updates.append(raw)

        positive = sum(raw-old for raw, old in zip(raw_updates, old_groups))
        available = density*initial["eele"]
        emission_scale = available/positive
        assert np.all(positive > available)
        assert np.all((emission_scale > 0.0) & (emission_scale < 1.0))
        for group, (raw, old) in enumerate(zip(raw_updates, old_groups)):
            expected = (old + emission_scale*(raw-old))/density
            assert np.allclose(final[f"erad0{group}"], expected,
                               rtol=3.0e-12, atol=3.0e-13)
        assert not np.allclose(final["erad01"], raw_updates[1]/density,
                               rtol=3.0e-12, atol=3.0e-13)
        assert np.allclose(final["eele"], 0.0, rtol=0.0, atol=3.0e-13)
        assert np.allclose(final["erad"]-initial["erad"],
                           initial["eele"]-final["eele"],
                           rtol=3.0e-12, atol=3.0e-13)
        total_initial = initial["eion"] + initial["eele"] + initial["erad"]
        total_final = final["eion"] + final["eele"] + final["erad"]
        assert np.allclose(total_final, total_initial,
                           rtol=3.0e-12, atol=3.0e-13)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
