"""Regression tests for aligned output scheduling at the time limit."""

import glob
import re

import pytest

import test_suite.testutils as testutils


INPUT_FILE = "inputs/output_alignment.athinput"


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
            [f"job/basename={basename}", f"time/tlim={tlim:.10g}"],
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
    finally:
        testutils.cleanup()
