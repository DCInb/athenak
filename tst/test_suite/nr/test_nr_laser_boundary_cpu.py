"""Grazing rays must cross rounded block faces and retain analytic attenuation."""

import math
import os
from pathlib import Path

import pytest

import test_suite.testutils as testutils
from test_suite.nr.test_nr_laser_remaining_cpu import appended_log, laser_diagnostics


INPUT = (Path(__file__).resolve().parents[3] /
         "inputs/mhd/two_temperature_laser.athinput")


def check_boundary_crossing(axis, sign, coefficient, mpi=False):
    # Terminal state of run-1557536's ray 82183. Its finest block has 12 cells
    # along [-0.042857142857142816, 0]. On CUDA the fused last-face expression
    # rounds to this negative y, while the block ownership boundary is exactly 0.
    # The old direction-scaled probe remains negative and tracing stalls forever.
    origin = [0.480033602132903914, -sign * 1.73472347597680709e-18,
              0.318930498762795833]
    direction = [-0.541761057969048054, sign * 9.05845975210994347e-6,
                 0.840532543085752426]
    origin[axis], origin[1] = origin[1], origin[axis]
    direction[axis], direction[1] = direction[1], direction[axis]
    norm = math.sqrt(sum(value * value for value in direction))
    direction = [value / norm for value in direction]
    distance = min(((0.6 if n > 0 else -0.6) - x) / n
                   for x, n in zip(origin, direction))
    escaped = math.exp(-coefficient * distance)
    flags = [
        "job/basename=laser_grazing_boundary",
        "time/integrator=rk1", "time/nlim=1", "time/tlim=1.0e-8",
        "laser/beam0_nrays=1", f"laser/absorption_coefficient={coefficient}",
        "laser/max_segments_per_launch=16", "laser/max_transport_iterations=8",
        "laser/max_mpi_waves=4", "laser/report_diagnostics=true",
    ]
    for d in range(3):
        coordinate = f"x{d + 1}"
        flags += [
            f"mesh/n{coordinate}={336 if d == axis else 12}",
            f"mesh/{coordinate}min=-0.6", f"mesh/{coordinate}max=0.6",
            f"meshblock/n{coordinate}=12",
            f"laser/beam0_origin_{coordinate}={origin[d]:.17g}",
            f"laser/beam0_direction_{coordinate}={direction[d]:.17g}",
        ]
    flags += [f"output{n}/dt=-1.0" for n in range(1, 5)]
    offset = os.path.getsize(testutils.LOG_FILE_PATH)
    try:
        if mpi:
            assert testutils.mpi_run(str(INPUT), flags=flags, threads=2, timeout=30)
        else:
            assert testutils.run(str(INPUT), flags=flags, timeout=30)
        log = appended_log(offset)
        diagnostics = laser_diagnostics(log)
        assert diagnostics["remaining_rays"] == 0
        assert diagnostics["remaining"] == 0
        assert diagnostics["iterations"] < 32
        assert diagnostics["residual"] < 1.0e-12
        assert diagnostics["path"] == pytest.approx(distance, rel=1.0e-12)
        assert diagnostics["escaped"] == pytest.approx(escaped, rel=1.0e-12)
        assert diagnostics["deposited"] == pytest.approx(1.0 - escaped, abs=1.0e-12)
        assert "laser_remaining_ray:" not in log
        if mpi:
            assert diagnostics["transfers"] >= 1
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x1", "x2", "x3"])
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
@pytest.mark.parametrize("coefficient", [0.0, 0.7], ids=["transparent", "absorbing"])
def test_grazing_boundary(axis, sign, coefficient):
    check_boundary_crossing(axis, sign, coefficient)
