"""GPU smoke regression for production mixed-material radiation coupling."""

from pathlib import Path

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils
from test_suite.nr.mixed_thermal_radiation_utils import (
    NGROUPS,
    prepare_case,
    run_mixed_transport_probe,
)


input_file = Path("mixed_thermal_radiation_gpu.athinput")
material0_table = Path("mixed_opacity_gpu_ch.dat")
material1_table = Path("mixed_opacity_gpu_he.dat")


def test_run():
    try:
        prepare_case(input_file, material0_table, material1_table)
        flags = [
            "job/basename=mixed_radiation_gpu",
            "problem/yl=0.25", "problem/yr=0.25",
        ]
        assert testutils.run(str(input_file), flags=flags), (
            "Mixed-material radiation GPU run failed.")
        initial = athena_read.tab(
            "tab/mixed_radiation_gpu.mhd_3t.00000.tab")
        final = athena_read.tab(
            "tab/mixed_radiation_gpu.mhd_3t.00001.tab")
        for group in range(NGROUPS):
            field = f"erad{group:02d}"
            assert np.all(np.isfinite(final[field]))
            assert np.all(final[field] >= 0.0)
        initial_total = initial["eion"]+initial["eele"]+initial["erad"]
        final_total = final["eion"]+final["eele"]+final["erad"]
        assert np.allclose(final_total, initial_total,
                           rtol=4.0e-13, atol=4.0e-13)
        assert np.all(final["eele"] >= 0.0)
        assert np.all(np.isfinite(final["tele"]))

        for output in run_mixed_transport_probe(
                input_file, "mixed_radiation_flux_3d_gpu"):
            output.unlink()
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        input_file.unlink(missing_ok=True)
        material0_table.unlink(missing_ok=True)
        material1_table.unlink(missing_ok=True)
        testutils.cleanup()
