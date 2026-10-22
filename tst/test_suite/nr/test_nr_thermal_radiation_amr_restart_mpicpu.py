"""Restart implicit transport from an AMR checkpoint on different MPI ranks."""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_same_solution, assert_transport, ordered_fields,
    read_binary,
)


def execute(directory, ranks, arguments, expected_error=None):
    """Keep checkpoint replays and their complete-volume outputs isolated."""
    executable = Path(os.environ.get("ATHENA_AMR_TEST_EXE", "./athena")).resolve()
    directory.mkdir()
    command = ["mpirun", "-np", str(ranks), str(executable), *arguments]
    result = subprocess.run(command, cwd=directory, capture_output=True,
                            text=True, timeout=120, check=False)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    if expected_error is not None:
        assert result.returncode != 0
        assert expected_error in result.stdout + result.stderr
        return []
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted((directory / "bin").glob("*.bin"))
    assert paths, result.stdout
    return [read_binary(str(path)) for path in paths]


def restart_input(directory, interval=2, per_rank=False):
    path = directory / "restart.athinput"
    path.write_text(INPUT.read_text() + f"""

<output2>
file_type = rst
dt = -1
dcycle = {interval}
single_file_per_rank = {str(per_rank).lower()}
""")
    return path


def assert_full_state(reference, candidate):
    assert_same_solution([reference], [candidate])
    _, reference_fields = ordered_fields(reference)
    _, resumed_fields = ordered_fields(candidate)
    assert reference_fields.keys() == resumed_fields.keys()
    for field, values in reference_fields.items():
        tolerance = 2.0e-11 if field in GROUPS else 2.0e-14
        np.testing.assert_allclose(resumed_fields[field], values,
                                   rtol=2.0e-9, atol=tolerance, err_msg=field)


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
@pytest.mark.parametrize("checkpoint_cycle", (1, 2))
@pytest.mark.parametrize("align_outputs", (False, True))
def test_amr_checkpoint_restart(tmp_path, adaptive, checkpoint_cycle, align_outputs):
    input_file = restart_input(tmp_path, checkpoint_cycle)
    flags = ["mesh/nx2=16", "meshblock/nx2=8",
             f"time/align_outputs={str(align_outputs).lower()}",
             "mesh_refinement/refinement=adaptive"]
    if adaptive == "refine":
        flags.append("amr_criterion0/value_max=0.5")
    uninterrupted = execute(tmp_path / "uninterrupted", 1,
                            ["-i", str(input_file), *flags])
    assert_transport(uninterrupted, adaptive)

    # The scheduled restart is distinct from the terminal checkpoint.
    # Selecting the final file would conceal output-before-regrid ordering errors.
    checkpoints = sorted((tmp_path / "uninterrupted/rst").glob("*.rst"))
    assert len(checkpoints) == 2 + 3 // checkpoint_cycle
    checkpoint = checkpoints[1]
    snapshot = execute(tmp_path / "checkpoint_state", 1,
                       ["-r", str(checkpoint),
                        f"time/nlim={checkpoint_cycle}"])[-1]
    assert snapshot["cycle"] == checkpoint_cycle
    checkpoint_time = (2**checkpoint_cycle - 1) * 1.0e-4
    assert np.isclose(snapshot["time"], checkpoint_time, rtol=0.0, atol=2.0e-15)
    expected_levels = ({0, 1} if checkpoint_cycle == 1
                       else {1} if adaptive == "refine" else {0})
    assert set(np.asarray(snapshot["mb_logical"])[:, 3]) == expected_levels
    changed = not np.array_equal(ordered_fields(snapshot)[0],
                                 ordered_fields(uninterrupted[0])[0])
    assert changed == (checkpoint_cycle == 2)

    for ranks in (1, 2):
        resumed = execute(tmp_path / f"resumed_{ranks}", ranks,
                          ["-r", str(checkpoint), "time/nlim=3"])
        final = resumed[-1]
        assert_transport([uninterrupted[0], snapshot, *resumed], adaptive)
        assert_full_state(uninterrupted[-1], final)


def test_amr_restart_footer_compatibility(tmp_path):
    input_file = restart_input(tmp_path)
    execute(tmp_path / "source", 1, ["-i", str(input_file),
                                     "mesh_refinement/refinement=adaptive"])
    checkpoint = sorted((tmp_path / "source/rst").glob("*.rst"))[1]
    reference = execute(tmp_path / "reference", 1,
                        ["-r", str(checkpoint), "time/nlim=2"])[-1]

    # Older files have the unchanged mesh/fluid payload, without a version key
    # or refinement-age footer. They must still load their physical state exactly.
    payload = checkpoint.read_bytes()
    header_end = payload.index(b"<par_end>") + 10
    header, count = re.subn(rb"(?m)^restart_age_version\s*=.*\n", b"",
                            payload[:header_end])
    assert count == 1
    block_count = np.frombuffer(payload, dtype=np.intc, count=1,
                                offset=header_end)[0]
    footer_size = int(block_count) * np.dtype(np.intc).itemsize
    legacy = tmp_path / "legacy.rst"
    legacy.write_bytes(header + payload[header_end:-footer_size])
    loaded = execute(tmp_path / "legacy", 2,
                     ["-r", str(legacy), "time/nlim=2"])[-1]
    assert_full_state(reference, loaded)

    truncated = tmp_path / "truncated.rst"
    truncated.write_bytes(payload[:-np.dtype(np.intc).itemsize])
    execute(tmp_path / "truncated", 2,
            ["-r", str(truncated), "time/nlim=2"],
            expected_error="Incomplete AMR refinement ages in restart")


def test_amr_per_rank_checkpoint(tmp_path):
    input_file = restart_input(tmp_path, interval=1, per_rank=True)
    baseline = execute(tmp_path / "baseline", 2,
                       ["-i", str(input_file), "mesh/nx2=16", "meshblock/nx2=8",
                        "mesh_refinement/refinement=adaptive"])
    assert_transport(baseline, "derefine")
    checkpoint = sorted((tmp_path / "baseline/rst/rank_00000000").glob("*.rst"))[1]
    resumed = execute(tmp_path / "resumed", 2,
                      ["-r", str(checkpoint), "time/nlim=3"])
    assert_transport([baseline[0], *resumed], "derefine")
    assert_full_state(baseline[-1], resumed[-1])
