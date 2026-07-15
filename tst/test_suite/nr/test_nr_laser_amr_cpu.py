"""SMR/AMR traversal regressions for the two-temperature laser module."""

import glob
import shutil

import numpy as np
import pytest

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_cpu import (
    critical_density_cgs,
    read_laser_binary,
)


input_file = "../../../inputs/mhd/two_temperature_laser_amr.athinput"


def run_case(basename, flags, cycles=1):
    common = [
        f"job/basename={basename}",
        f"time/nlim={cycles}",
        "time/tlim=1.0",
        "output1/dt=-1.0",
        "output1/dcycle=1",
    ]
    assert testutils.run(input_file, flags=common + flags), (
        f"{basename} AMR laser run failed.")
    outputs = sorted(glob.glob(f"bin/{basename}.laser_amr.*.bin"))
    assert outputs
    return read_laser_binary(outputs[-1])


def integrated_diagnostics(output):
    """Integrate one-dimensional block data with each block's refinement level."""
    root_blocks = output["Nx1"] // output["nx1_mb"]
    totals = {
        name: 0.0
        for name in ("laser_q", "laser_energy", "laser_tau", "laser_path")
    }
    for block, logical in enumerate(output["mb_logical"]):
        refinement = 2.0**int(logical[3])
        dx = 1.0 / (root_blocks * refinement * output["nx1_mb"])
        totals["laser_q"] += np.sum(output["mb_data"]["laser_q"][block]) * dx
        totals["laser_energy"] += (
            np.sum(output["mb_data"]["laser_energy"][block]) * dx)
        totals["laser_tau"] += np.sum(output["mb_data"]["laser_tau"][block])
        totals["laser_path"] += np.sum(output["mb_data"]["laser_path"][block])
    return totals


def test_run():
    try:
        # Transparent and constant-opacity paths cross coarse-to-fine and fine-to-coarse
        # interfaces without changing their geometric or conservative result.
        transparent = integrated_diagnostics(run_case(
            "laser_smr_transparent", ["laser/absorption_coefficient=0.0"]))
        assert np.isclose(transparent["laser_path"], 1.0,
                          rtol=2.0e-6, atol=2.0e-7)
        assert transparent["laser_q"] == 0.0

        coefficient = 2.0
        expected_deposition = -np.expm1(-coefficient)
        layouts = ((0.125, 0.375), (0.25, 0.75), (0.625, 0.875))
        for index, (left, right) in enumerate(layouts):
            output = run_case(f"laser_smr_layout_{index}", [
                f"refined_region1/x1min={left}",
                f"refined_region1/x1max={right}",
                f"laser/absorption_coefficient={coefficient}",
            ])
            totals = integrated_diagnostics(output)
            assert np.isclose(totals["laser_path"], 1.0,
                              rtol=2.0e-6, atol=2.0e-7)
            assert np.isclose(totals["laser_tau"], coefficient,
                              rtol=2.0e-6, atol=2.0e-7)
            assert np.isclose(totals["laser_q"], expected_deposition,
                              rtol=2.0e-6, atol=2.0e-7)
            assert np.isclose(
                totals["laser_energy"], expected_deposition * output["time"],
                rtol=3.0e-6, atol=3.0e-10)

        # Put the critical surface just to either side of, and exactly on, a refinement
        # interface. Linear profiles have an analytic round-trip length of 2*x_turn.
        density_to_electrons = 1.0e13
        rho0 = 0.5
        rho_turn = critical_density_cgs(1.0) / density_to_electrons
        for index, turn_x in enumerate((0.249, 0.25, 0.251, 0.75)):
            density_gradient = (rho_turn-rho0) / turn_x
            output = run_case(f"laser_smr_reflect_{index}", [
                "laser/critical_reflection=true",
                "laser/absorption_coefficient=0.0",
                f"problem/density_gradient={density_gradient}",
            ])
            totals = integrated_diagnostics(output)
            assert np.isclose(totals["laser_path"], 2.0 * turn_x,
                              rtol=3.0e-5, atol=3.0e-6)

        # Trigger adaptive refinement after the first cycle. The ray map refreshes to
        # eight refined blocks, while power, path, optical depth, and cumulative energy
        # remain equal to the analytic uniform-medium solution.
        adaptive_output = run_case("laser_dynamic_amr", [
            "mesh_refinement/refinement=adaptive",
            f"laser/absorption_coefficient={coefficient}",
        ], cycles=3)
        adaptive = integrated_diagnostics(adaptive_output)
        assert len(adaptive_output["mb_logical"]) == 8
        assert np.isclose(adaptive["laser_path"], 1.0,
                          rtol=2.0e-6, atol=2.0e-7)
        assert np.isclose(adaptive["laser_tau"], coefficient,
                          rtol=2.0e-6, atol=2.0e-7)
        assert np.isclose(adaptive["laser_q"], expected_deposition,
                          rtol=2.0e-6, atol=2.0e-7)
        assert np.isclose(
            adaptive["laser_energy"],
            expected_deposition * adaptive_output["time"],
            rtol=3.0e-6, atol=3.0e-9)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
        shutil.rmtree("bin", ignore_errors=True)
