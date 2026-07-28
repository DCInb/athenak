"""Regression tests for aligned output scheduling at the time limit."""

import glob
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

import test_suite.testutils as testutils


INPUT_FILE = "inputs/output_alignment.athinput"


@pytest.mark.parametrize("walltime", ("bad", "1:60:00", "-1:00:00"))
def test_invalid_walltime_is_rejected(walltime):
    result = subprocess.run(
        ["./athena", "-i", INPUT_FILE, "-t", walltime],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    message = result.stdout+result.stderr
    assert "-t" in message
    assert "requires" in message or "must be followed" in message


@pytest.mark.parametrize(
    ("tlim", "expected_count", "penultimate_time"),
    ((10.0, 101, 9.9), (10.0001, 102, 10.0)),
)
def test_aligned_terminal_output(tlim, expected_count, penultimate_time):
    """Write the terminal state once without suppressing a distinct cadence point."""
    basename = f"output_alignment_{tlim:g}".replace(".", "p")
    try:
        testutils.run(
            INPUT_FILE,
            [
                f"job/basename={basename}",
                f"time/tlim={tlim:.10g}",
                "output3/dt=20.0",
            ],
        )

        paths = sorted(glob.glob(f"tab/{basename}.hydro_w.*.tab"))
        assert len(paths) == expected_count
        assert paths[-1].endswith(f".{expected_count - 1:05d}.tab")

        times = []
        for path in paths[-2:]:
            with open(path, encoding="ascii") as stream:
                match = re.search(r"time=(\S+)", stream.readline())
            assert match is not None
            times.append(float(match.group(1)))
        assert times == pytest.approx((penultimate_time, tlim))

        if tlim == 10.0:
            with open(paths[-1], encoding="ascii") as stream:
                terminal_header = stream.readline()
            cycle_match = re.search(r"cycle=(\d+)", terminal_header)
            assert cycle_match is not None
            assert int(cycle_match.group(1)) == 200

            history = np.loadtxt(f"{basename}.hydro.hst", comments="#", ndmin=2)
            assert len(np.unique(history[:, 0])) == 201
            assert np.min(history[:, 1]) > 1.0e-2
    finally:
        testutils.cleanup()
        for path in glob.glob(f"{basename}*.hst"):
            os.remove(path)
        shutil.rmtree("rst", ignore_errors=True)


def test_aligned_restart_recovers_physics_timestep():
    """Do not apply the event-remainder growth limiter after a restart."""
    basename = "output_alignment_restart"
    try:
        assert testutils.run(
            INPUT_FILE,
            [f"job/basename={basename}", "time/tlim=0.3"],
        )
        restart = f"rst/{basename}.00001.rst"
        assert os.path.isfile(restart)

        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        command = [
            "./athena", "-r", restart,
            "time/tlim=0.5",
            "time/ndiag=1",
        ]
        assert testutils.run_command(command), "Aligned-output restart failed."
        with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
            stream.seek(log_offset)
            restart_log = stream.read()

        diagnostics = re.findall(
            r"cycle=(\d+) time=(\S+) dt=(\S+)", restart_log)
        assert diagnostics
        first_time = float(diagnostics[0][1])
        first_dt = float(diagnostics[0][2])
        assert first_time == pytest.approx(0.3)
        assert first_dt > 5.0e-2

        paths = sorted(glob.glob(f"tab/{basename}.hydro_w.*.tab"))
        assert len(paths) == 6
        times = []
        for path in paths:
            with open(path, encoding="ascii") as stream:
                match = re.search(r"time=(\S+)", stream.readline())
            assert match is not None
            times.append(float(match.group(1)))
        assert times == pytest.approx(np.arange(6)*0.1)
    finally:
        testutils.cleanup()
        for path in glob.glob(f"{basename}*.hst"):
            os.remove(path)
        shutil.rmtree("rst", ignore_errors=True)


def test_walltime_checkpoint_is_atomic_and_preserves_restart_schedule():
    """An immediate walltime stop writes only a resumable rolling checkpoint."""
    basename = "output_alignment_walltime"
    try:
        log_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        command = [
            "./athena", "-i", INPUT_FILE, "-t", "00:00:00",
            f"job/basename={basename}",
            "time/ndiag=1",
        ]
        assert testutils.run_command(command), "Immediate walltime checkpoint failed."
        with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
            stream.seek(log_offset)
            walltime_log = stream.read()
        assert "Terminating on wall clock limit" in walltime_log

        checkpoint = f"rst/{basename}.walltime.rst"
        assert os.path.isfile(checkpoint)
        assert not glob.glob("rst/*.part")
        # Initialize writes the ordinary t=0 outputs.  Walltime Finalize must not add a
        # terminal table or consume the numbered restart schedule.
        assert glob.glob(f"tab/{basename}.hydro_w.*.tab") == [
            f"tab/{basename}.hydro_w.00000.tab"
        ]
        assert os.path.isfile(f"rst/{basename}.00000.rst")

        parse_offset = os.path.getsize(testutils.LOG_FILE_PATH)
        assert testutils.run_command(["./athena", "-r", checkpoint, "-n"])
        with open(testutils.LOG_FILE_PATH, encoding="utf-8") as stream:
            stream.seek(parse_offset)
            parameter_dump = stream.read()
        output3 = parameter_dump.split("<output3>", 1)[1].split("<par_end>", 1)[0]
        file_number = re.search(r"(?m)^file_number\s*=\s*(\S+)", output3)
        assert file_number is not None
        assert int(file_number.group(1)) == 1
        last_time = re.search(r"(?m)^last_time\s*=\s*(\S+)", output3)
        assert last_time is not None
        assert float(last_time.group(1)) == pytest.approx(0.0)

        # A real one-cycle reload validates the payload, not only its text header.
        assert testutils.run_command([
            "./athena", "-r", checkpoint,
            "time/nlim=1", "time/tlim=10.0", "time/ndiag=1",
        ]), "Walltime checkpoint could not advance a complete cycle."
    finally:
        testutils.cleanup()
        for path in glob.glob(f"{basename}*.hst"):
            os.remove(path)
        shutil.rmtree("rst", ignore_errors=True)
