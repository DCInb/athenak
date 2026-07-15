"""Regression tests for device DDA laser-ray transport and 2T deposition."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"


def reference_cell_path(origin, direction, lower, upper):
    """Return the ray length inside one Cartesian cell."""
    enter = 0.0
    leave = np.inf
    for x0, ray_dir, cell_lo, cell_hi in zip(
            origin, direction, lower, upper):
        if abs(ray_dir) < 1.0e-15:
            if x0 < cell_lo or x0 >= cell_hi:
                return 0.0
            continue
        first = (cell_lo - x0) / ray_dir
        second = (cell_hi - x0) / ray_dir
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
    return max(leave - enter, 0.0)


def reference_2d_paths(origin_y, direction, resolution):
    ray_dir = np.asarray(direction, dtype=float)
    ray_dir /= np.linalg.norm(ray_dir)
    result = np.zeros((resolution, resolution))
    dx = 1.0 / resolution
    dy = 1.0 / resolution
    for j in range(resolution):
        for i in range(resolution):
            result[j, i] = reference_cell_path(
                (0.0, origin_y), ray_dir,
                (i * dx, -0.5 + j * dy),
                ((i + 1) * dx, -0.5 + (j + 1) * dy))
    return result


def run_transparent_case(basename, flags):
    common = [
        f"job/basename={basename}",
        "laser/absorption_coefficient=0.0",
        "time/tlim=1.0e-7",
        "output1/dt=-1.0",
        "output2/dt=-1.0",
        "output3/dt=1.0e-7",
    ]
    assert testutils.run(input_file, flags=common + flags), (
        f"{basename} laser transport run failed.")
    data = athena_read.tab(f"tab/{basename}.laser.00001.tab")
    order = np.argsort(data["x1v"])
    fields = ("x1v", "laser_q", "laser_energy", "laser_ray_count",
              "laser_tau", "laser_path")
    return {name: np.atleast_1d(data[name])[order] for name in fields}


def test_run():
    try:
        resolution = 8
        cases = (
            (-0.31, (1.0, 0.37)),
            (0.22, (1.0, -0.65)),
            (-0.47, (1.0, 1.60)),
        )
        for case_index, (origin_y, direction) in enumerate(cases):
            expected = reference_2d_paths(origin_y, direction, resolution)
            measured = np.zeros_like(expected)
            counts = np.zeros_like(expected)
            for j in range(resolution):
                slice_y = -0.5 + (j + 0.5) / resolution
                basename = f"laser_dda_{case_index}_{j}"
                data = run_transparent_case(basename, [
                    f"mesh/nx1={resolution}",
                    f"mesh/nx2={resolution}",
                    "meshblock/nx1=4",
                    "meshblock/nx2=4",
                    f"laser/beam0_origin_x2={origin_y}",
                    f"laser/beam0_direction_x1={direction[0]}",
                    f"laser/beam0_direction_x2={direction[1]}",
                    f"output3/slice_x2={slice_y}",
                ])
                measured[j, :] = data["laser_path"]
                counts[j, :] = data["laser_ray_count"]
                assert np.count_nonzero(data["laser_q"]) == 0
                assert np.count_nonzero(data["laser_energy"]) == 0
                assert np.count_nonzero(data["laser_tau"]) == 0

            # The formatted-table input uses 12 digits after the decimal point.
            assert np.allclose(measured, expected, rtol=2.0e-11,
                               atol=2.0e-13)
            assert np.array_equal(counts, (expected > 0.0).astype(float))

        # Exercise three-dimensional indexing and transfers between eight
        # same-rank MeshBlocks with an exactly axis-aligned ray.
        data = run_transparent_case("laser_dda_3d", [
            "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8",
            "meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4",
            "laser/beam0_origin_x2=0.0",
            "laser/beam0_origin_x3=0.0",
            "laser/beam0_direction_x1=1.0",
            "laser/beam0_direction_x2=0.0",
            "laser/beam0_direction_x3=0.0",
        ])
        assert np.allclose(data["laser_path"], 1.0 / 8.0,
                           rtol=2.0e-11, atol=2.0e-13)
        assert np.array_equal(data["laser_ray_count"], np.ones(8))
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
