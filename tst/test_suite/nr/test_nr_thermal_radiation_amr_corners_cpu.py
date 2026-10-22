"""Nested leaf trees, wide halos, and surface energy balance for implicit FLD."""

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, assert_same_solution, assert_transport, moments, ordered_fields, run_case,
)


@pytest.mark.parametrize("dimensions", (1, 3))
def test_wide_amr_halos(tmp_path, dimensions):
    reference = run_case(tmp_path / "ng2", dimensions=dimensions)
    candidate = run_case(tmp_path / "ng4", dimensions=dimensions,
                         extra_flags=("mesh/nghost=4",))
    assert_transport(candidate)
    assert_same_solution(reference, candidate)


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_three_dimensional_topology_change(tmp_path, adaptive):
    outputs = run_case(tmp_path / adaptive, dimensions=3, adaptive=adaptive)
    assert_transport(outputs, adaptive)


@pytest.mark.parametrize("dimensions", (1, 3))
def test_three_leaf_levels(tmp_path, dimensions):
    outputs = run_case(
        tmp_path / "nested", dimensions=dimensions,
        extra_flags=("mesh/nghost=4", "mesh_refinement/num_levels=3",
                     "refined_region1/level=2",
                     f"thermal_radiation/implicit_amr_order={1 if dimensions == 1 else 2}"))
    initial = outputs[0]
    topology, _ = ordered_fields(initial)
    assert set(topology[:, 3]) == {0, 1, 2}
    assert outputs[-1]["cycle"] == 3
    for group in GROUPS:
        energy0, square0 = moments(initial, group)
        low = np.min(initial["mb_data"][group])
        high = np.max(initial["mb_data"][group])
        for output in outputs[1:]:
            np.testing.assert_array_equal(ordered_fields(output)[0], topology)
            values = np.asarray(output["mb_data"][group])
            assert np.all(np.isfinite(values))
            assert np.min(values) >= low - 2.0e-10
            assert np.max(values) <= high + 2.0e-10
            energy, square = moments(output, group)
            np.testing.assert_allclose(energy, energy0, rtol=2.0e-10,
                                       atol=2.0e-12)
            assert square < square0
    if dimensions == 1:
        _assert_nested_backward_euler(outputs)


def _assert_nested_backward_euler(outputs):
    """Solve diffusion independently on the three cell widths of the leaf grid."""
    initial = outputs[0]
    centers, widths = [], []
    for geometry, block in zip(initial["mb_geometry"],
                               initial["mb_data"][GROUPS[0]]):
        count = np.asarray(block).size
        width = (geometry[1] - geometry[0]) / count
        centers.extend(geometry[0] + (np.arange(count) + 0.5) * width)
        widths.extend([width] * count)
    order = np.argsort(centers)
    widths = np.asarray(widths)[order]
    assert np.unique(widths).size == 3
    count = len(widths)
    for group, opacity in zip(GROUPS, (100.0, 35.0)):
        expected = np.asarray(initial["mb_data"][group]).ravel()[order]
        diffusion = 1000.0 / (3.0 * opacity)
        for previous, output in zip(outputs, outputs[1:]):
            dt = output["time"] - previous["time"]
            matrix = np.eye(count)
            for cell in range(count):
                for neighbor in ((cell - 1) % count, (cell + 1) % count):
                    separation = 0.5 * (widths[cell] + widths[neighbor])
                    weight = dt * diffusion / (widths[cell] * separation)
                    matrix[cell, cell] += weight
                    matrix[cell, neighbor] -= weight
            expected = np.linalg.solve(matrix, expected)
            actual = np.asarray(output["mb_data"][group]).ravel()[order]
            np.testing.assert_allclose(actual, expected, rtol=3.0e-9,
                                       atol=3.0e-11, err_msg=group)


@pytest.mark.parametrize("dimensions,block", ((2, 8), (3, 8), (2, 4)))
def test_amr_surface_energy_balance(tmp_path, dimensions, block):
    # Different boundary types and values on different faces exercise corners and
    # transverse transport. Refined leaves touch both lower and upper boundaries.
    boundaries = ("dirichlet", "vacuum", "neumann", "dirichlet",
                  "vacuum", "neumann")
    flags = ["mesh/nghost=4", "refined_region1/x1max=0.5",
             "refined_region1/x2max=0.5", "refined_region1/x3max=0.5"]
    flags += [f"meshblock/nx{direction+1}={block}" for direction in range(dimensions)]
    for direction in range(dimensions):
        for side, end in enumerate(("inner", "outer")):
            face = 2 * direction + side
            axis = direction + 1
            prefix = "i" if side == 0 else "o"
            flags += [f"mesh/{prefix}x{axis}_bc=outflow",
                      f"thermal_radiation/implicit_x{axis}_{end}_boundary="
                      f"{boundaries[face]}",
                      f"thermal_radiation/implicit_x{axis}_{end}_value="
                      f"{0.1 + 0.02 * face}"]
    outputs = run_case(tmp_path / "surface", dimensions=dimensions,
                       extra_flags=flags)
    assert set(np.asarray(outputs[0]["mb_logical"])[:, 3]) == {0, 1}
    for group, opacity in zip(GROUPS, (100.0, 35.0)):
        diffusion = 1000.0 / (3.0 * opacity)
        for previous, output in zip(outputs, outputs[1:]):
            dt = output["time"] - previous["time"]
            power = 0.0
            for geometry, block in zip(output["mb_geometry"],
                                       output["mb_data"][group]):
                values = np.asarray(block)
                assert np.all(np.isfinite(values)) and np.min(values) >= 0.0
                widths = ((np.asarray(geometry)[1::2]
                           - np.asarray(geometry)[::2]) / values.shape[::-1])
                for direction in range(dimensions):
                    area = np.prod(np.delete(widths, direction))
                    conductance = diffusion / widths[direction]
                    for side, coordinate in enumerate((-0.5, 0.5)):
                        face = 2 * direction + side
                        if not np.isclose(geometry[face], coordinate):
                            continue
                        plane = np.take(values, 0 if side == 0 else -1,
                                        axis=2 - direction)
                        if boundaries[face] == "dirichlet":
                            flux = 2.0 * conductance * (plane - (0.1 + 0.02 * face))
                        elif boundaries[face] == "vacuum":
                            flux = min(conductance, 500.0) * plane
                        else:
                            continue
                        power += area * np.sum(flux)
            lost_energy = moments(previous, group)[0] - moments(output, group)[0]
            np.testing.assert_allclose(lost_energy, dt * power, rtol=2.0e-9,
                                       atol=3.0e-11, err_msg=group)
