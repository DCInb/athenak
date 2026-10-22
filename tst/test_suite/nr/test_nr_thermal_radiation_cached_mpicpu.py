"""Check cached Galerkin weights, team restriction and the local-face fast path.

Compare complete volumes against the unchanged point-Jacobi operator solve and
against two-rank transport. Unequal opacities and growing dt exercise cache
refreshes between both groups and timesteps. No existing run files are modified.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


def ordered_fields(path):
    data = read_binary(str(path))
    logical = data["mb_logical"]
    order = np.lexsort(logical.T[::-1])
    return data, {key: np.asarray(value)[order] for key, value in data["mb_data"].items()}


@pytest.mark.parametrize("shape,block,boundary", [
    ((36, 1, 1), (18, 1, 1), "periodic"),
    ((36, 18, 1), (18, 18, 1), "neumann"),
    ((36, 18, 18), (18, 18, 18), "periodic"),
    ((36, 18, 18), (18, 18, 18), "neumann"),
    ((36, 18, 18), (18, 18, 18), "vacuum"),
    ((36, 18, 18), (18, 18, 18), "dirichlet"),
])
def test_cached_multilevel(tmp_path, shape, block, boundary):
    executable = Path(os.environ.get("ATHENA_SOLVER_TEST_EXE", "./athena")).resolve()
    run_env = os.environ.copy()
    # Slurm may advertise one MPI slot for a multi-core, one-task GPU allocation.
    # These two CPU ranks fit within the test job's eight allocated cores.
    run_env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    outputs = []
    for ranks, preconditioner in ((1, "jacobi"), (1, "block-coarse"), (2, "block-coarse")):
        directory = tmp_path / (preconditioner + "-" + str(ranks))
        directory.mkdir()
        flags = ["thermal_radiation/implicit_preconditioner=" + preconditioner]
        for axis in range(1, 4):
            flags += ["mesh/nx{}={}".format(axis, shape[axis-1]),
                      "meshblock/nx{}={}".format(axis, block[axis-1])]
            for side, end in (("i", "inner"), ("o", "outer")):
                fluid_bc = "periodic" if boundary == "periodic" else "outflow"
                radiation_bc = "neumann" if boundary == "periodic" else boundary
                flags += ["mesh/{}x{}_bc={}".format(side, axis, fluid_bc),
                          "thermal_radiation/implicit_x{}_{}_boundary={}".format(
                              axis, end, radiation_bc)]
        command = ["mpirun", "-np", str(ranks), str(executable), "-i",
                   str(REPO / "inputs/tests/thermal_radiation_multilevel.athinput")] + flags
        with (directory / "run.log").open("w") as log:
            subprocess.run(command, cwd=str(directory), stdout=log,
                           stderr=subprocess.STDOUT, env=run_env, timeout=120, check=True)
        initial, initial_fields = ordered_fields(next((directory / "bin").glob("*.00000.bin")))
        final, final_fields = ordered_fields(next((directory / "bin").glob("*.00001.bin")))
        assert final["cycle"] == 3
        assert np.isclose(final["time"], 7.0e-4, rtol=1.0e-12, atol=1.0e-15)
        for group in ("erad00", "erad01"):
            a, b = initial_fields[group], final_fields[group]
            assert np.all(np.isfinite(b)) and np.min(b) >= 0.0
            assert not np.array_equal(a, b)
            # Native volume dumps store float32 even for double-precision solves.
            if boundary in ("periodic", "neumann"):
                assert np.isclose(a.sum(dtype=float), b.sum(dtype=float), rtol=1.0e-7)
            elif boundary == "vacuum":
                assert b.sum(dtype=float) < a.sum(dtype=float)
        outputs.append(final_fields)
    for candidate in outputs[1:]:
        for field, reference in outputs[0].items():
            assert np.allclose(candidate[field], reference, rtol=3.0e-7, atol=3.0e-9), field
