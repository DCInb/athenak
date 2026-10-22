"""Uniform static cell-AMG partitions preserve the equation and restart ownership."""

import re

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import INPUT, GROUPS, ordered_fields
from test_suite.nr.test_nr_thermal_radiation_cell_amg import cell_amg_executable
from test_suite.nr.test_nr_thermal_radiation_amr_restart_mpicpu import (
    execute, assert_full_state,
)


def uniform_input(directory):
    text = re.sub(r"\n<refined_region1>.*?(?=\n<)", "", INPUT.read_text(), flags=re.S)
    path = directory / "uniform.athinput"
    path.write_text(text + "\n<mesh>\nmeshblock_order = z_order\n"
                    "meshblock_tiles_x2 = 2\nmeshblock_tiles_x3 = 2\n"
                    "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    return path


FLAGS = ["mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32",
         "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
         "thermal_radiation/implicit_preconditioner=cell-amg",
         "thermal_radiation/implicit_amr_order=1", "time/tlim=0.0007"]
TILED = ["mesh/meshblock_order=x1_tiled", "mesh/meshblock_tiles_x2=2",
         "mesh/meshblock_tiles_x3=2"]


def seed_checkpoint(directory, input_file, order):
    initial = execute(directory, 4, ["-i", str(input_file), *FLAGS, *order,
                                     "time/nlim=0"])[0]
    checkpoint = sorted((directory / "rst").glob("*.rst"))[-1]
    payload = bytearray(checkpoint.read_bytes())
    extent = 12  # Eight active cells plus two ghosts on each face.
    shape = (initial["n_mbs"], 9, extent, extent, extent)
    per_block = int(np.prod(shape[1:]) * 8)
    start = len(payload) - int(np.prod(shape) * 8)
    assert np.frombuffer(payload, dtype=np.uint64, count=1,
                         offset=start-8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    for block, bounds in enumerate(initial["mb_geometry"]):
        h = (bounds[1]-bounds[0])/8
        axes = [bounds[2*a] + (np.arange(extent)-1.5)*h for a in range(3)]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        state[block, 0] = 1.0 + 0.25*np.cos(2*np.pi*y)
        # Variation in every direction exercises halos across tiled rank faces.
        state[block, 7] = 1.0 + 0.1*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        state[block, 8] = 0.7 + 0.1*np.sin(2*np.pi*y)*np.cos(2*np.pi*z)
    injected = directory / "seed.rst"
    injected.write_bytes(payload)
    return injected


def test_cell_amg_tiled_equation_and_restart(tmp_path, cell_amg_executable):
    deck = uniform_input(tmp_path)
    baseline = seed_checkpoint(tmp_path / "z_seed", deck, [])
    tiled = seed_checkpoint(tmp_path / "tile_seed", deck, TILED)
    reference = execute(tmp_path / "z_solve", 4,
                        ["-r", str(baseline), "time/nlim=3"])
    candidate = execute(tmp_path / "tile_solve", 4,
                        ["-r", str(tiled), "time/nlim=3"])
    assert_full_state(reference[-1], candidate[-1])
    assert set(np.asarray(candidate[-1]["mb_logical"])[:, 3]) == {0}
    _, initial_fields = ordered_fields(candidate[0])
    _, final_fields = ordered_fields(candidate[-1])
    for group in GROUPS:
        assert np.min(final_fields[group]) >= 0
        np.testing.assert_allclose(np.sum(final_fields[group]),
                                   np.sum(initial_fields[group]), rtol=2e-10)
    checkpoint = sorted((tmp_path / "tile_solve/rst").glob("*.rst"))[1]
    resumed = execute(tmp_path / "resumed", 4,
                      ["-r", str(checkpoint), "time/nlim=3"])
    assert_full_state(candidate[-1], resumed[-1])
    # Both order changes must fail before reading fields under the wrong GIDs.
    for name, checkpoint, order in (("z_to_tile", baseline, TILED),
                                     ("tile_to_z", tiled, ["mesh/meshblock_order=z_order"])):
        execute(tmp_path / name, 4, ["-r", str(checkpoint), *order],
                expected_error="restart MeshBlock order does not match")


@pytest.mark.parametrize("adaptive", [False, True])
def test_cell_amg_tiled_rejects_refinement(tmp_path, cell_amg_executable, adaptive):
    deck = tmp_path / "refined.athinput"
    deck.write_text(INPUT.read_text() + "\n<mesh>\nmeshblock_order = z_order\n"
                    "meshblock_tiles_x2 = 2\nmeshblock_tiles_x3 = 2\n")
    flags = [*FLAGS, *TILED, "time/nlim=0"]
    if adaptive: flags.append("mesh_refinement/refinement=adaptive")
    execute(tmp_path / "rejected", 4, ["-i", str(deck), *flags],
            expected_error="x1 ordering requires a uniform")
