"""Conservation, weak exchange, and material-normalization regressions."""

from pathlib import Path
import re

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


@pytest.mark.parametrize("absorbing_group", [False, True])
def test_radiation_background_preserves_electrons(absorbing_group):
    """An uncoupled bright group cannot remove the material's internal energy."""
    try:
        flags = [
            "job/basename=hedp_bright_background", "time/nlim=1",
            "time/tlim=0.01", "output1/dt=0.01",
            "thermal_radiation/n_groups=1",
            "thermal_radiation/arad=1.0",
            "thermal_radiation/initial_radiation_temperature=1.0e5",
            "thermal_radiation/group_bound_1=1.0e8",
            "thermal_radiation/kappa_absorption_0=0.0",
            "thermal_radiation/kappa_emission_0=0.0",
            "thermal_radiation/source_cfl=0.0",
            "thermal_radiation/source_report=true",
            "output1/data_format=%24.17e",
        ]
        if absorbing_group:
            flags += [
                "thermal_radiation/n_groups=2",
                "thermal_radiation/arad=1.0e20",
                "thermal_radiation/initial_radiation_temperature=1.0",
                "thermal_radiation/group_bound_1=1.0e-6",
                "thermal_radiation/group_bound_2=1.0e3",
                "thermal_radiation/kappa_absorption_0=1.0",
                "thermal_radiation/kappa_absorption_1=0.0",
                "thermal_radiation/kappa_emission_1=0.0",
            ]
        testutils.run("../../../inputs/hydro/two_temperature_mgfld.athinput",
                      flags=flags)
        initial = athena_read.tab(
            "tab/hedp_bright_background.hydro_3t.00000.tab")
        final = athena_read.tab(
            "tab/hedp_bright_background.hydro_3t.00001.tab")
        if absorbing_group:
            final_path = Path("tab/hedp_bright_background.hydro_3t.00001.tab")
            dt = float(re.search(r"time=([^\s]+)", final_path.read_text()).group(1))
            absorbed = initial["erad00"]*dt/(1.0+dt)
            np.testing.assert_allclose(final["eele"], initial["eele"]+absorbed,
                                       rtol=2.0e-11, atol=0.0)
            np.testing.assert_allclose(final["erad00"], initial["erad00"]/(1.0+dt),
                                       rtol=2.0e-11, atol=0.0)
            fields = ("eion", "erad01")
        else:
            fields = ("eele", "eion", "tele", "erad")
        for field in fields:
            np.testing.assert_allclose(final[field], initial[field],
                                       rtol=2.0e-13, atol=0.0)
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("materials", [False, True])
def test_weak_exchange_cold_electrons(materials):
    """Resolve electron heating far below an ulp of the dominant ion energy."""
    deck = Path("hedp_weak_material.athinput")
    try:
        source = Path("../../../inputs/hydro/two_temperature_relax.athinput")
        if materials:
            text = source.read_text().replace("<hydro>\n", "<hydro>\nnscalars = 1\n", 1)
            text += """
<materials>
nmaterials = 1
scalar_index = 0
material0_name = test
material0_abar = 2.0
material0_zbar = 1.0
material0_zeff = 1.0
"""
            deck.write_text(text)
            source = deck
        testutils.run(str(source), flags=[
            "job/basename=hedp_weak_exchange",
            "hydro/initial_electron_temperature_ratio=1.0e-20",
            "hydro/t_ei=1.0e20", "time/tlim=0.1",
            "output1/data_format=%24.17e",
        ])
        initial = athena_read.tab("tab/hedp_weak_exchange.hydro_2t.00000.tab")
        final = athena_read.tab("tab/hedp_weak_exchange.hydro_2t.00001.tab")
        fraction = -np.expm1(-2.0*0.1/1.0e20)
        expected = (initial["eele"] +
                    0.5*(initial["eion"]-initial["eele"])*fraction)
        np.testing.assert_allclose(final["eele"], expected, rtol=2.0e-12, atol=0.0)
        np.testing.assert_allclose(final["eele"]+final["eion"],
                                   initial["eele"]+initial["eion"],
                                   rtol=2.0e-13, atol=0.0)
    finally:
        deck.unlink(missing_ok=True)
        testutils.cleanup()


def test_radiation_emission_cannot_loosen_convergence():
    """A huge trial emissivity is not an available energy reservoir."""
    deck = Path("hedp_stiff_emission.athinput")
    try:
        text = Path("../../../inputs/hydro/two_temperature_mgfld.athinput").read_text()
        deck.write_text(text.replace(
            "<thermal_radiation>\n",
            "<thermal_radiation>\nsource_nonlinear_tolerance = 1.0e-13\n", 1))
        testutils.run(str(deck), flags=[
            "job/basename=hedp_stiff_emission", "time/nlim=1", "time/tlim=0.01",
            "thermal_radiation/n_groups=1", "thermal_radiation/arad=1.0e20",
            "thermal_radiation/initial_radiation_temperature=0.0",
            "thermal_radiation/group_bound_1=1.0e8",
            "thermal_radiation/kappa_absorption_0=0.0",
            "thermal_radiation/kappa_emission_0=1.0e5",
            "thermal_radiation/source_cfl=0.0",
            "output1/dt=0.01", "output1/data_format=%24.17e",
        ])
        initial = athena_read.tab("tab/hedp_stiff_emission.hydro_3t.00000.tab")
        final = athena_read.tab("tab/hedp_stiff_emission.hydro_3t.00001.tab")
        energy = initial["eele"][0]
        capacity = energy/initial["tele"][0]
        low, high = 0.0, initial["tele"][0]
        for _ in range(120):
            trial = 0.5*(low+high)
            if capacity*trial + 1.0e23*trial**4 > energy:
                high = trial
            else:
                low = trial
        np.testing.assert_allclose(final["eele"], capacity*0.5*(low+high),
                                   rtol=1.0e-6, atol=0.0)
        np.testing.assert_allclose(final["eele"]+final["erad"], initial["eele"],
                                   rtol=2.0e-13, atol=0.0)
    finally:
        deck.unlink(missing_ok=True)
        testutils.cleanup()


def test_biermann_material_timestep_normalization():
    """At fixed Pe and Z, the thermal-magnetic speed is proportional to A."""
    deck = Path("hedp_biermann_material.athinput")
    try:
        text = Path("../../../inputs/mhd/two_temperature_biermann.athinput").read_text()
        text = text.replace("<mhd>\n", "<mhd>\nnscalars = 1\n", 1)
        text = text.replace("rsolver = hlle", "rsolver = llf", 1)
        text += """
<materials>
nmaterials = 1
scalar_index = 0
material0_name = test
material0_abar = 2.0
material0_zbar = 1.0
material0_zeff = 1.0
"""
        deck.write_text(text)
        timesteps = []
        for mass in (2.0, 32.0):
            basename = f"hedp_biermann_mass_{int(mass)}"
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            testutils.run(str(deck), flags=[
                f"job/basename={basename}", f"materials/material0_abar={mass}",
                "mhd/biermann_coefficient=100.0",
                "time/nlim=0", "time/tlim=1.0",
                "output2/dt=-1", "output3/dt=-1",
            ])
            history = athena_read.hst(f"{basename}.mhd.hst")
            timesteps.append(history["dt"][0])
        np.testing.assert_allclose(timesteps[0]/timesteps[1], 16.0,
                                   rtol=2.0e-11, atol=0.0)
    finally:
        deck.unlink(missing_ok=True)
        testutils.cleanup()
