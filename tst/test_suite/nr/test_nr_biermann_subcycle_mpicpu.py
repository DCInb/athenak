"""MPI decomposition regressions for complete Biermann subcycling."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
FINAL_TIME = 5.0e-3
SMR_FINAL_TIME = 1.0e-8
SMR_RESOLUTION = 16
SMR_BLOCK_SIZE = 8
AMR_CYCLES = 3


def sorted_table(path):
    table = athena_read.tab(path)
    order = np.argsort(np.asarray(table["x1v"]))
    return {
        key: (np.asarray(value)[order]
              if np.asarray(value).shape == order.shape else np.asarray(value))
        for key, value in table.items()
    }


def run_case(basename, ranks):
    flags = [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=8", "meshblock/nx2=8",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/align_outputs=true",
        f"time/tlim={FINAL_TIME}",
        f"output1/dt={FINAL_TIME}",
        f"output2/dt={FINAL_TIME}", "output2/data_format=%24.17e",
        f"output3/dt={FINAL_TIME}", "output3/data_format=%24.17e",
        f"output4/dt={FINAL_TIME}",
    ]
    assert testutils.mpi_run(
        INPUT_FILE, flags=flags, threads=ranks, timeout=120.0), (
            f"{basename} failed with {ranks} MPI ranks")
    return (
        sorted_table(f"tab/{basename}.biermann.00001.tab"),
        sorted_table(f"tab/{basename}.two_temperature.00001.tab"),
        athena_read.hst(f"{basename}.mhd.hst"),
        bin_convert.read_binary(f"bin/{basename}.divb.00001.bin"),
    )


def run_smr_case(basename, ranks):
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={SMR_RESOLUTION}",
        f"mesh/nx2={SMR_RESOLUTION}",
        f"mesh/nx3={SMR_RESOLUTION}",
        f"meshblock/nx1={SMR_BLOCK_SIZE}",
        f"meshblock/nx2={SMR_BLOCK_SIZE}",
        f"meshblock/nx3={SMR_BLOCK_SIZE}",
        "mesh_refinement/refinement=static",
        # Refine only the lower root block.  Its eight children meet the other
        # seven root blocks at a true-3D coarse/fine corner split across ranks.
        "refined_region1/x1min=-0.5",
        "refined_region1/x1max=-0.49",
        "refined_region1/x2min=-0.5",
        "refined_region1/x2max=-0.49",
        "refined_region1/x3min=-0.5",
        "refined_region1/x3max=-0.49",
        "problem/pressure_amplitude=0.20",
        "problem/pressure_x2_amplitude=0.20",
        "problem/pressure_x3_amplitude=0.05",
        "problem/density_amplitude=0.10",
        "problem/density_x3_amplitude=0.20",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/align_outputs=true",
        f"time/tlim={SMR_FINAL_TIME}",
        f"output1/dt={SMR_FINAL_TIME}",
        "output2/dt=-1", "output3/dt=-1",
        "output4/variable=mhd_divb", "output4/id=divb",
        "output4/double_precision_binary=true",
        f"output4/dt={SMR_FINAL_TIME}",
        "output5/variable=mhd_u_bcc", "output5/id=conserved",
        "output5/double_precision_binary=true",
        f"output5/dt={SMR_FINAL_TIME}",
        "output6/variable=mhd_2t", "output6/id=two_temperature_full",
        "output6/double_precision_binary=true",
        f"output6/dt={SMR_FINAL_TIME}",
        "output7/dt=-1",
    ]
    assert testutils.mpi_run(
        INPUT_FILE, flags=flags, threads=ranks, timeout=120.0), (
            f"{basename} failed with {ranks} MPI ranks")
    return {
        "conserved": bin_convert.read_binary(
            f"bin/{basename}.conserved.00001.bin"),
        "two_temperature": bin_convert.read_binary(
            f"bin/{basename}.two_temperature_full.00001.bin"),
        "divb": bin_convert.read_binary(f"bin/{basename}.divb.00001.bin"),
        "history": athena_read.hst(f"{basename}.mhd.hst"),
    }


def run_amr_case(basename, ranks):
    flags = [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=8", "meshblock/nx2=8",
        "mesh_refinement/refinement=adaptive",
        # Start multilevel in one corner, then let the location criterion add a
        # disjoint central fine region after the first completed macro cycle.
        "refined_region1/x1min=-0.5", "refined_region1/x1max=-0.49",
        "refined_region1/x2min=-0.5", "refined_region1/x2max=-0.49",
        "mhd/biermann_shock_suppression=false",
        "mhd/biermann_coefficient=5.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/tlim=1.0", f"time/nlim={AMR_CYCLES}", "time/ndiag=1",
        "output1/dt=1.0e-20",
        "output2/dt=-1", "output3/dt=-1",
        "output4/variable=mhd_divb", "output4/id=divb",
        "output4/double_precision_binary=true",
        "output4/dt=-1", "output4/dcycle=1",
        "output5/variable=mhd_u_bcc", "output5/id=conserved",
        "output5/double_precision_binary=true",
        "output5/dt=-1", f"output5/dcycle={AMR_CYCLES}",
        "output6/variable=mhd_2t", "output6/id=two_temperature_full",
        "output6/double_precision_binary=true",
        "output6/dt=-1", f"output6/dcycle={AMR_CYCLES}",
        "output7/dt=-1", "output7/dcycle=-1",
        "output8/dt=-1", "output8/dcycle=-1",
    ]
    assert testutils.mpi_run(
        INPUT_FILE, flags=flags, threads=ranks, timeout=120.0), (
            f"{basename} failed with {ranks} MPI ranks")

    divb_paths = sorted(Path("bin").glob(f"{basename}.divb.*.bin"))
    # Initialization and every completed cycle are required.  Finalize may add a
    # second terminal-cycle dump after the scheduled dcycle output.
    assert len(divb_paths) >= AMR_CYCLES + 1
    return {
        "initial_divb": bin_convert.read_binary(divb_paths[0]),
        "conserved": bin_convert.read_binary(
            f"bin/{basename}.conserved.00001.bin"),
        "two_temperature": bin_convert.read_binary(
            f"bin/{basename}.two_temperature_full.00001.bin"),
        "divb": bin_convert.read_binary(divb_paths[-1]),
        "history": athena_read.hst(f"{basename}.mhd.hst"),
    }


def assert_invariants(result):
    field, two_temperature, history, divb_output = result
    for key in ("dens", "eint"):
        assert np.all(np.isfinite(field[key]))
        assert np.all(field[key] > 0.0)
    for key in ("eion", "eele", "tion", "tele"):
        assert np.all(np.isfinite(two_temperature[key]))
        assert np.all(two_temperature[key] > 0.0)
    assert np.allclose(
        field["dens"]*(two_temperature["eion"]+two_temperature["eele"]),
        field["eint"], rtol=3.0e-11, atol=3.0e-12)
    assert np.allclose(history["tot-E"], history["tot-E"][0],
                       rtol=3.0e-12, atol=3.0e-12)
    divb = np.concatenate([
        np.asarray(block).ravel() for block in divb_output["mb_data"]["divb"]
    ])
    assert divb.size == 32*32
    assert np.max(np.abs(divb)) < 5.0e-13


def canonical_order(output):
    logical = np.asarray(output["mb_logical"])
    return sorted(range(len(logical)), key=lambda block: tuple(logical[block]))


def canonical_data(output, variable):
    return np.stack([
        np.asarray(output["mb_data"][variable][block])
        for block in canonical_order(output)
    ])


def assert_corner_topology(output):
    assert (output["Nx1"], output["Nx2"], output["Nx3"]) == (
        SMR_RESOLUTION,)*3
    assert (output["nx1_mb"], output["nx2_mb"], output["nx3_mb"]) == (
        SMR_BLOCK_SIZE,)*3

    logical = np.asarray(output["mb_logical"])
    coarse = {tuple(location[:3]) for location in logical
              if location[3] == 0}
    fine = {tuple(location[:3]) for location in logical
            if location[3] == 1}
    root = {(i, j, k) for k in range(2) for j in range(2)
            for i in range(2)}
    assert coarse == root-{(0, 0, 0)}, coarse
    assert fine == root, fine
    assert output["n_mbs"] == 15


def assert_smr_invariants(result):
    conserved = result["conserved"]
    two_temperature = result["two_temperature"]
    divergence = result["divb"]
    history = result["history"]

    for output in (conserved, two_temperature, divergence):
        assert output["cycle"] == 1
        assert_corner_topology(output)

    assert set(("dens", "mom1", "mom2", "mom3", "ener",
                "eion_d", "eele_d", "bcc1", "bcc2", "bcc3")) <= set(
                    conserved["var_names"])
    for key in conserved["var_names"]:
        assert np.all(np.isfinite(canonical_data(conserved, key))), key
    for key in ("dens", "ener", "eion_d", "eele_d"):
        assert np.all(canonical_data(conserved, key) > 0.0), key

    for key in ("eion", "eele", "tion", "tele"):
        values = canonical_data(two_temperature, key)
        assert np.all(np.isfinite(values)), key
        assert np.all(values > 0.0), key
    density = canonical_data(conserved, "dens")
    for conserved_key, specific_key in (("eion_d", "eion"),
                                         ("eele_d", "eele")):
        assert np.allclose(
            canonical_data(conserved, conserved_key),
            density*canonical_data(two_temperature, specific_key),
            rtol=3.0e-11, atol=3.0e-12), conserved_key

    magnetic_maxima = np.array([
        np.max(np.abs(canonical_data(conserved, key)))
        for key in ("bcc1", "bcc2", "bcc3")
    ])
    assert np.all(magnetic_maxima > 1.0e-9), magnetic_maxima

    divb = canonical_data(divergence, "divb")
    assert np.all(np.isfinite(divb))
    assert np.max(np.abs(divb)) < 5.0e-13

    assert len(history["time"]) == 2
    for key, values in history.items():
        assert np.all(np.isfinite(values)), key
    for key in ("mass", "1-mom", "2-mom", "3-mom", "tot-E"):
        assert np.allclose(history[key], history[key][0],
                           rtol=3.0e-12, atol=3.0e-12), key
    assert np.all(np.array([
        history[key][-1] for key in ("1-ME", "2-ME", "3-ME")
    ]) > 0.0)


def assert_amr_invariants(result):
    initial = result["initial_divb"]
    conserved = result["conserved"]
    two_temperature = result["two_temperature"]
    divergence = result["divb"]
    history = result["history"]

    assert initial["cycle"] == 0
    assert set(np.asarray(initial["mb_logical"])[:, 3]) == {0, 1}
    for output in (conserved, two_temperature, divergence):
        assert output["cycle"] == AMR_CYCLES
        assert set(np.asarray(output["mb_logical"])[:, 3]) == {0, 1}
        assert output["n_mbs"] > initial["n_mbs"]

    required = {"dens", "mom1", "mom2", "mom3", "ener",
                "eion_d", "eele_d", "bcc1", "bcc2", "bcc3"}
    assert required <= set(conserved["var_names"])
    for key in conserved["var_names"]:
        assert np.all(np.isfinite(canonical_data(conserved, key))), key
    for key in ("dens", "ener", "eion_d", "eele_d"):
        assert np.all(canonical_data(conserved, key) > 0.0), key

    density = canonical_data(conserved, "dens")
    for conserved_key, specific_key in (("eion_d", "eion"),
                                         ("eele_d", "eele")):
        specific = canonical_data(two_temperature, specific_key)
        assert np.all(np.isfinite(specific)), specific_key
        assert np.all(specific > 0.0), specific_key
        assert np.allclose(canonical_data(conserved, conserved_key),
                           density*specific,
                           rtol=3.0e-11, atol=3.0e-12), conserved_key
    for key in ("tion", "tele"):
        values = canonical_data(two_temperature, key)
        assert np.all(np.isfinite(values)), key
        assert np.all(values > 0.0), key

    assert np.max(np.abs(canonical_data(conserved, "bcc3"))) > 1.0e-3
    divb = canonical_data(divergence, "divb")
    assert np.all(np.isfinite(divb))
    assert np.max(np.abs(divb)) < 5.0e-13

    assert len(history["time"]) == AMR_CYCLES + 1
    for key, values in history.items():
        assert np.all(np.isfinite(values)), key
    for key in ("mass", "1-mom", "2-mom", "3-mom", "tot-E"):
        assert np.allclose(history[key], history[key][0],
                           rtol=1.0e-11, atol=1.0e-11), key


def assert_binary_decomposition_equal(left, right):
    metadata = ("Nx1", "Nx2", "Nx3", "nx1_mb", "nx2_mb", "nx3_mb",
                "n_mbs", "cycle", "var_names")
    for key in metadata:
        assert left[key] == right[key], key

    left_order = canonical_order(left)
    right_order = canonical_order(right)
    assert np.array_equal(
        np.asarray(left["mb_logical"])[left_order],
        np.asarray(right["mb_logical"])[right_order])
    assert np.array_equal(
        np.asarray(left["mb_geometry"])[left_order],
        np.asarray(right["mb_geometry"])[right_order])
    for key in left["var_names"]:
        assert np.array_equal(
            canonical_data(left, key), canonical_data(right, key)), key


def test_run():
    basenames = ("biermann_subcycle_mpi_1", "biermann_subcycle_mpi_2")
    smr_basenames = ("biermann_subcycle_smr_mpi_1",
                     "biermann_subcycle_smr_mpi_2")
    amr_basenames = ("biermann_subcycle_amr_mpi_1",
                     "biermann_subcycle_amr_mpi_2")
    try:
        for basename in (*basenames, *smr_basenames, *amr_basenames):
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for directory, suffix in ((Path("bin"), "bin"),
                                      (Path("tab"), "tab")):
                for path in directory.glob(f"{basename}.*.{suffix}"):
                    path.unlink(missing_ok=True)
        serial_decomposition = run_case(basenames[0], 1)
        parallel_decomposition = run_case(basenames[1], 2)
        assert_invariants(serial_decomposition)
        assert_invariants(parallel_decomposition)

        # Cell fields must not depend on whether block boundaries cross MPI ranks.
        # History reductions can differ in their last bits because MPI changes summation
        # order, so gate the actual solution tables exactly and histories numerically.
        for left, right in zip(serial_decomposition[:2],
                               parallel_decomposition[:2]):
            assert left.keys() == right.keys()
            for key in left:
                assert np.array_equal(left[key], right[key]), key
        left_history = serial_decomposition[2]
        right_history = parallel_decomposition[2]
        assert left_history.keys() == right_history.keys()
        for key in left_history:
            assert np.allclose(left_history[key], right_history[key],
                               rtol=3.0e-13, atol=3.0e-15), key

        serial_smr = run_smr_case(smr_basenames[0], 1)
        parallel_smr = run_smr_case(smr_basenames[1], 2)
        assert_smr_invariants(serial_smr)
        assert_smr_invariants(parallel_smr)
        for output_name in ("conserved", "two_temperature", "divb"):
            assert_binary_decomposition_equal(
                serial_smr[output_name], parallel_smr[output_name])
        assert serial_smr["history"].keys() == parallel_smr["history"].keys()
        for key in serial_smr["history"]:
            assert np.allclose(
                serial_smr["history"][key], parallel_smr["history"][key],
                rtol=3.0e-13, atol=3.0e-15), key

        serial_amr = run_amr_case(amr_basenames[0], 1)
        parallel_amr = run_amr_case(amr_basenames[1], 2)
        assert_amr_invariants(serial_amr)
        assert_amr_invariants(parallel_amr)
        for output_name in ("conserved", "two_temperature", "divb"):
            assert_binary_decomposition_equal(
                serial_amr[output_name], parallel_amr[output_name])
        assert serial_amr["history"].keys() == parallel_amr["history"].keys()
        for key in serial_amr["history"]:
            assert np.allclose(
                serial_amr["history"][key], parallel_amr["history"][key],
                rtol=3.0e-13, atol=3.0e-15), key
    finally:
        testutils.cleanup()
        for basename in (*basenames, *smr_basenames, *amr_basenames):
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for path in Path("bin").glob(f"{basename}.*.bin"):
                path.unlink(missing_ok=True)
