"""Restart and dynamic-AMR qualification for Biermann subcycling."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
UNINTERRUPTED = "biermann_subcycle_uninterrupted"
RESTARTED = "biermann_subcycle_restarted"
ADAPTIVE = "biermann_subcycle_dynamic_amr"
ADAPTIVE_UNINTERRUPTED = "biermann_subcycle_amr_uninterrupted"
ADAPTIVE_RESTARTED = "biermann_subcycle_amr_restarted"


def common_flags(basename):
    return [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=8", "meshblock/nx2=8",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/tlim=1.0", "time/ndiag=1",
        "output1/dt=1.0e-20",
        "output2/dt=-1", "output3/dt=-1",
        "output4/dt=-1", "output4/dcycle=-1",
        "output5/dt=-1", "output5/dcycle=4",
        "output6/dt=-1", "output6/dcycle=4",
        "output7/dt=-1", "output7/dcycle=-1",
    ]


def latest_binary(basename, output_id):
    paths = sorted(Path("bin").glob(f"{basename}.{output_id}.*.bin"))
    assert paths, (basename, output_id)
    return bin_convert.read_binary(paths[-1])


def assert_same_binary(left, right):
    """Restarting at a macro-step boundary must reproduce the complete state."""
    assert left["cycle"] == right["cycle"]
    assert left["time"] == right["time"]
    assert left["var_names"] == right["var_names"]
    for key in ("mb_index", "mb_logical", "mb_geometry"):
        assert np.array_equal(left[key], right[key]), key
    assert left["mb_data"].keys() == right["mb_data"].keys()
    for key in left["mb_data"]:
        left_values = np.asarray(left["mb_data"][key])
        right_values = np.asarray(right["mb_data"][key])
        scale = max(np.linalg.norm(left_values), np.linalg.norm(right_values), 1.0)
        error = np.linalg.norm(left_values-right_values)/scale
        assert error < 5.0e-14, (key, error,
                                np.max(np.abs(left_values-right_values)))


def assert_history_conserved(basename, tolerance=3.0e-12):
    history = athena_read.hst(f"{basename}.mhd.hst")
    assert np.all(np.isfinite(history["tot-E"]))
    assert np.allclose(history["tot-E"], history["tot-E"][0],
                       rtol=tolerance, atol=tolerance)
    return history


def run_restart_equivalence():
    assert testutils.run(
        INPUT_FILE,
        flags=[*common_flags(UNINTERRUPTED), "time/nlim=4"],
        timeout=120.0,
    )

    split_flags = common_flags(RESTARTED)
    split_flags[split_flags.index("output7/dcycle=-1")] = "output7/dcycle=2"
    assert testutils.run(
        INPUT_FILE,
        flags=[*split_flags, "time/nlim=2"],
        timeout=120.0,
    )
    restarts = sorted(Path("rst").glob(f"{RESTARTED}.*.rst"))
    assert restarts
    assert testutils.run_command([
        "./athena", "-r", str(restarts[-1]),
        "time/nlim=4", "time/tlim=1.0", "time/ndiag=1",
    ], timeout=120.0)

    uninterrupted_field = latest_binary(UNINTERRUPTED, "biermann_full")
    restarted_field = latest_binary(RESTARTED, "biermann_full")
    uninterrupted_thermal = latest_binary(
        UNINTERRUPTED, "two_temperature_full")
    restarted_thermal = latest_binary(RESTARTED, "two_temperature_full")
    assert uninterrupted_field["cycle"] == 4
    assert restarted_field["cycle"] == 4
    assert_same_binary(uninterrupted_field, restarted_field)
    assert_same_binary(uninterrupted_thermal, restarted_thermal)
    assert_history_conserved(UNINTERRUPTED)
    assert_history_conserved(RESTARTED)


def run_dynamic_amr():
    flags = [*adaptive_flags(ADAPTIVE), "time/nlim=3"]
    assert testutils.run(INPUT_FILE, flags=flags, timeout=120.0)

    divb_paths = sorted(Path("bin").glob(f"{ADAPTIVE}.divb.*.bin"))
    assert len(divb_paths) >= 2
    initial = bin_convert.read_binary(divb_paths[0])
    final_divb = bin_convert.read_binary(divb_paths[-1])
    assert initial["cycle"] == 0
    assert set(np.asarray(initial["mb_logical"])[:, 3]) == {0, 1}
    final_levels = set(np.asarray(final_divb["mb_logical"])[:, 3])
    assert final_levels == {0, 1}, final_levels
    assert final_divb["n_mbs"] > initial["n_mbs"]

    divb = np.concatenate([
        np.asarray(block).ravel()
        for block in final_divb["mb_data"]["divb"]
    ])
    assert np.all(np.isfinite(divb))
    assert np.max(np.abs(divb)) < 5.0e-13

    field = latest_binary(ADAPTIVE, "biermann_full")
    thermal = latest_binary(ADAPTIVE, "two_temperature_full")
    assert field["cycle"] == thermal["cycle"] == 3
    assert np.array_equal(field["mb_logical"], thermal["mb_logical"])
    for key in ("dens", "eint", "bcc1", "bcc2", "bcc3"):
        values = np.asarray(field["mb_data"][key])
        assert np.all(np.isfinite(values)), key
    for key in ("eion", "eele", "tion", "tele"):
        values = np.asarray(thermal["mb_data"][key])
        assert np.all(np.isfinite(values)), key
        assert np.all(values > 0.0), key

    density = np.asarray(field["mb_data"]["dens"])
    internal = np.asarray(field["mb_data"]["eint"])
    eion = np.asarray(thermal["mb_data"]["eion"])
    eele = np.asarray(thermal["mb_data"]["eele"])
    assert np.all(density > 0.0)
    assert np.all(internal > 0.0)
    assert np.allclose(density*(eion+eele), internal,
                       rtol=3.0e-11, atol=3.0e-12)
    assert np.max(np.abs(field["mb_data"]["bcc3"])) > 1.0e-3
    history = assert_history_conserved(ADAPTIVE, tolerance=1.0e-11)
    assert len(history["time"])-1 == 3


def adaptive_flags(basename):
    flags = common_flags(basename)
    replacements = {
        "output4/dcycle=-1": "output4/dcycle=1",
        "output5/dcycle=4": "output5/dcycle=3",
        "output6/dcycle=4": "output6/dcycle=3",
    }
    flags = [replacements.get(flag, flag) for flag in flags]
    flags.extend([
        "mesh_refinement/refinement=adaptive",
        # Seed one corner block.  The location criterion adds a disjoint central
        # fine region after cycle one, which is the topology change checkpointed below.
        "refined_region1/x1min=-0.5", "refined_region1/x1max=-0.49",
        "refined_region1/x2min=-0.5", "refined_region1/x2max=-0.49",
    ])
    return flags


def run_adaptive_restart_equivalence():
    """Restart after cycle-one refinement and reproduce the adaptive trajectory."""
    assert testutils.run(
        INPUT_FILE,
        flags=[*adaptive_flags(ADAPTIVE_UNINTERRUPTED), "time/nlim=3"],
        timeout=120.0,
    )

    split_flags = adaptive_flags(ADAPTIVE_RESTARTED)
    split_flags[split_flags.index("output7/dcycle=-1")] = "output7/dcycle=1"
    assert testutils.run(
        INPUT_FILE,
        flags=[*split_flags, "time/nlim=1"],
        timeout=120.0,
    )
    restarts = sorted(Path("rst").glob(f"{ADAPTIVE_RESTARTED}.*.rst"))
    assert restarts
    assert testutils.run_command([
        "./athena", "-r", str(restarts[-1]),
        "time/nlim=3", "time/tlim=1.0", "time/ndiag=1",
    ], timeout=120.0)

    uninterrupted_field = latest_binary(
        ADAPTIVE_UNINTERRUPTED, "biermann_full")
    restarted_field = latest_binary(ADAPTIVE_RESTARTED, "biermann_full")
    uninterrupted_thermal = latest_binary(
        ADAPTIVE_UNINTERRUPTED, "two_temperature_full")
    restarted_thermal = latest_binary(
        ADAPTIVE_RESTARTED, "two_temperature_full")
    assert uninterrupted_field["cycle"] == restarted_field["cycle"] == 3
    assert set(np.asarray(uninterrupted_field["mb_logical"])[:, 3]) == {0, 1}
    assert_same_binary(uninterrupted_field, restarted_field)
    assert_same_binary(uninterrupted_thermal, restarted_thermal)
    assert_history_conserved(ADAPTIVE_UNINTERRUPTED, tolerance=1.0e-11)
    assert_history_conserved(ADAPTIVE_RESTARTED, tolerance=1.0e-11)


def test_run():
    basenames = (UNINTERRUPTED, RESTARTED, ADAPTIVE,
                 ADAPTIVE_UNINTERRUPTED, ADAPTIVE_RESTARTED)
    try:
        for basename in basenames:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for directory in (Path("bin"), Path("rst")):
                for path in directory.glob(f"{basename}.*"):
                    path.unlink(missing_ok=True)
        run_restart_equivalence()
        run_dynamic_amr()
        run_adaptive_restart_equivalence()
    finally:
        testutils.cleanup()
        for basename in basenames:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for directory in (Path("bin"), Path("rst")):
                for path in directory.glob(f"{basename}.*"):
                    path.unlink(missing_ok=True)
