"""Cell AMG on conservative mixed-level graphs, including MPI and regrids."""

import os
from pathlib import Path
import re

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr.test_nr_thermal_radiation_cell_amg import cell_amg_executable
from test_suite.nr.test_nr_thermal_radiation_cell_amg_batch import (
    input_deck, compare_groups, records,
)
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, composite_diffusion, leaf_geometry, manufactured_run,
)
from test_suite.nr.test_nr_thermal_radiation_amr_restart_mpicpu import execute


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("boundary", ("periodic", "neumann", "dirichlet", "vacuum"))
def test_cell_amr_dense_equation(tmp_path, cell_amg_executable, ranks, boundary):
    flags = ["thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12", "problem/dr=10000"]
    if boundary != "periodic":
        flags += utils.boundary_flags(boundary)
    out = utils.run_case(tmp_path / "mixed", 1, "cell-amg", ranks=ranks,
                         extra_flags=flags)
    utils.assert_1d_reference(out, boundary)
    for group in utils.GROUPS:
        assert np.asarray(out[-1]["mb_data"][group]).min() >= 0


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("dimensions,normal,order,variable,inner", (
    (2, 0, 2, True, 0), (3, 0, 1, False, 0),
    (3, 1, 2, False, 0), (3, 2, 2, False, 20),
))
def test_cell_amr_composite_equation(tmp_path, cell_amg_executable, ranks,
                                    dimensions, normal, order, variable, inner):
    output = manufactured_run(
        tmp_path / "mixed", dimensions=dimensions, normal=normal, ranks=ranks,
        variable_coefficient=variable,
        extra_flags=("thermal_radiation/implicit_preconditioner=cell-amg",
                     f"thermal_radiation/implicit_amr_order={order}",
                     f"thermal_radiation/implicit_amr_inner_iterations={inner}"))
    assert set(np.asarray(output["mb_logical"])[:, 3]) == {0, 1}
    owner, xyz, widths = leaf_geometry(output, 16, dimensions)
    _, rhs = _manufactured(*xyz, widths, dimensions, 0.01, variable)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    coefficient = 1+0.25*np.cos(2*np.pi*xyz[1]) if variable else np.ones_like(actual)
    residual = actual+0.01*composite_diffusion(
        actual, owner, widths, order=order, coefficient=coefficient)-rhs
    np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-10)
    np.testing.assert_allclose(np.dot(widths**dimensions, actual),
                               np.dot(widths**dimensions, rhs), rtol=0, atol=2e-11)
    assert actual.min() >= 0


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_cell_amr_adapt_and_restart(tmp_path, cell_amg_executable, adaptive):
    deck = tmp_path / "adaptive.athinput"
    deck.write_text(utils.INPUT.read_text()+"\n<thermal_radiation>\n"
                    "implicit_preconditioner = cell-amg\nsolver_timers = true\n"
                    "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    flags = ["mesh/nx2=16", "meshblock/nx2=8", "mesh_refinement/refinement=adaptive"]
    if adaptive == "refine":
        flags.append("amr_criterion0/value_max=0.5")
    candidate = execute(tmp_path / "candidate", 2, ["-i", str(deck), *flags])
    reference = execute(tmp_path / "reference", 2, ["-i", str(deck), *flags,
                        "thermal_radiation/implicit_preconditioner=jacobi"])
    utils.assert_transport(candidate, adaptive)
    utils.assert_same_solution(reference, candidate)
    checkpoint = sorted((tmp_path / "candidate/rst").glob("*.rst"))[1]
    resumed = execute(tmp_path / "resumed", 1, ["-r", str(checkpoint), "time/nlim=3"])
    utils.assert_same_solution([candidate[-1]], [resumed[-1]])
    log = (tmp_path / "candidate/run.log").read_text()
    assert "volume_weighted=1" in log and "volume_weighted=0" in log


@pytest.mark.parametrize("halo", ("packed", "host"))
@pytest.mark.parametrize("reuse", (False, True))
def test_cell_amr_batch_same_count_regrid(tmp_path, cell_amg_executable, monkeypatch,
                                       halo, reuse):
    if reuse and not os.environ.get("ATHENA_AMG_REUSE_TEST_EXE"):
        pytest.skip("Requires a reuse-enabled builtin executable")
    if reuse:
        monkeypatch.setenv("ATHENA_AMR_TEST_EXE",
                           str(Path(os.environ["ATHENA_AMG_REUSE_TEST_EXE"]).resolve()))
    deck = input_deck(tmp_path, 5)
    # Move refinement from the right half to the left half at cycle 2 without
    # changing global block count. A count-only cache would retain wrong edges.
    deck.write_text(deck.read_text()+"\n<refined_region1>\nlevel = 1\n"
        "x1min = 0\nx1max = 0.5\nx2min = -0.5\nx2max = 0.5\n"
        "x3min = -0.5\nx3max = 0.5\n"
        "\n<output1>\ndt = -1\ndcycle = 1\n"
        "\n<thermal_radiation>\nimplicit_amr_order = 2\nimplicit_amg_reuse = false\n")
    flags = ["-i", str(deck), "mesh/nx1=16", "mesh/nx2=16",
             "meshblock/nx1=8", "meshblock/nx2=8", "time/nlim=3", "time/tlim=0.0007",
             "mesh_refinement/refinement=adaptive", "mesh_refinement/refinement_interval=2",
             "mesh_refinement/max_nmb_per_rank=64", "amr_criterion0/value_max=1.5",
             "problem/dl=2", "problem/dr=1", "problem/xshock=0",
             f"thermal_radiation/implicit_halo_exchange={halo}"]
    scalar = execute(tmp_path / "scalar", 4,
                     [*flags, "thermal_radiation/implicit_group_batch_size=1"])
    if reuse:
        flags.append("thermal_radiation/implicit_amg_reuse=true")
    batch = execute(tmp_path / "batch", 4, flags)
    assert len(scalar) == len(batch)
    assert {out["cycle"] for out in batch} == {0, 1, 2, 3}
    assert all(out["n_mbs"] == 10 for out in batch)
    assert not np.array_equal(utils.ordered_fields(batch[0])[0],
                              utils.ordered_fields(batch[-1])[0])
    for left, right in zip(scalar, batch):
        compare_groups(left, right, 5)
        for group in range(5):
            name = f"erad{group:02d}_d"
            np.testing.assert_allclose(utils.moments(right, name)[0],
                                       utils.moments(batch[0], name)[0], rtol=2e-10)
    report = records(tmp_path / "batch")
    assert len(report) == 12 and all(row["size"] == "4" for row in report)
    cache = [dict(re.findall(r"(\w+)=([^\s]+)", line))
             for line in (tmp_path / "batch/run.log").read_text().splitlines()
             if line.startswith("# radiation cell AMG:")]
    assert any(int(row["remote_interface_faces"]) > 0 for row in cache)
    assert sum(int(row["topology_rebuilt"]) for row in cache) >= 8
    if reuse:
        from test_suite.nr.test_nr_thermal_radiation_amg_reuse import reuse_records
        transfers = reuse_records(tmp_path / "batch")
        assert any(row["reused"] == "1" for row in transfers)
        assert all(row["reused"] == "0" for row in transfers if row["cycle"] == "2")
    checkpoint = sorted((tmp_path / "batch/rst").glob("*.rst"))[1]
    resumed = execute(tmp_path / "resumed", 2, ["-r", str(checkpoint), "time/nlim=3"])
    compare_groups(batch[-1], resumed[-1], 5)
