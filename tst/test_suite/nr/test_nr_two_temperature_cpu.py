"""Regression test for exact, conservative ion/electron heat exchange."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/hydro/two_temperature_relax.athinput"
dual_energy_input = "../../../inputs/mhd/two_temperature_dual_energy.athinput"


def test_run():
    try:
        assert testutils.run(input_file), "Two-temperature relaxation run failed."
        data = athena_read.tab("tab/two_temperature.hydro_2t.00001.tab")

        # The normalized heat capacities are fi=fe=1/2, so the equilibrium
        # temperature is one and Delta T decays as exp[-2*t/t_ei].
        delta_t0 = 1.0 / 0.55 - 0.1 / 0.55
        delta_t = delta_t0 * np.exp(-2.0 * 0.1 / 0.2)
        tion_exact = 1.0 + 0.5 * delta_t
        tele_exact = 1.0 - 0.5 * delta_t

        assert np.allclose(data["tion"], tion_exact, rtol=2.0e-11, atol=2.0e-11)
        assert np.allclose(data["tele"], tele_exact, rtol=2.0e-11, atol=2.0e-11)
        assert np.allclose(data["eion"] + data["eele"], 1.5,
                           rtol=2.0e-12, atol=2.0e-12)

        assert testutils.run(dual_energy_input), "2T dual-energy MHD run failed."
        initial = athena_read.tab(
            "tab/two_temperature_dual_energy.mhd_2t.00000.tab")
        final = athena_read.tab(
            "tab/two_temperature_dual_energy.mhd_2t.00001.tab")

        # B^2/2=5e17 has a double-precision spacing far larger than the gas energy 1.5.
        # The conservative subtraction therefore cannot recover pressure; the 2T
        # auxiliary sum must retain both the initial and evolved material energy.
        assert np.allclose(initial["eion"] + initial["eele"], 1.5,
                           rtol=2.0e-12, atol=2.0e-12)
        assert np.allclose(final["eion"] + final["eele"], 1.5,
                           rtol=2.0e-11, atol=2.0e-11)
        assert np.allclose(final["eion"], initial["eion"],
                           rtol=2.0e-11, atol=2.0e-11)
        assert np.allclose(final["eele"], initial["eele"],
                           rtol=2.0e-11, atol=2.0e-11)
        assert np.all(final["tion"] > 0.0)
        assert np.all(final["tele"] > 0.0)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
