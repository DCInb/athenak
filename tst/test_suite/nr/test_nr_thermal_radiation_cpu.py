"""Regression tests for 2T multigroup thermal radiation on CPUs."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


relax_input = "../../../inputs/hydro/two_temperature_mgfld.athinput"
diffusion_input = "../../../inputs/hydro/mgfld_diffusion.athinput"


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
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
