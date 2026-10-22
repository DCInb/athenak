"""Full-volume checks for conservative implicit radiation transport on AMR."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402

INPUT = REPO / "inputs/tests/thermal_radiation_amr.athinput"
GROUPS = ("erad00_d", "erad01_d")


def run_case(directory, dimensions=1, preconditioner="block-coarse",
             adaptive=None, ranks=None, extra_flags=()):
    """Run in an isolated directory and return every complete leaf-volume dump."""
    executable = Path(os.environ.get("ATHENA_AMR_TEST_EXE", "./athena")).resolve()
    directory.mkdir()
    flags = [f"thermal_radiation/implicit_preconditioner={preconditioner}"]
    for axis in (2, 3):
        flags += [f"mesh/nx{axis}={16 if axis <= dimensions else 1}",
                  f"meshblock/nx{axis}={8 if axis <= dimensions else 1}"]
    if adaptive is not None:
        flags.append("mesh_refinement/refinement=adaptive")
        if adaptive == "refine":
            flags.append("amr_criterion0/value_max=0.5")
        else:
            assert adaptive == "derefine"
    command = [str(executable), "-i", str(INPUT), *flags, *extra_flags]
    if ranks is not None:
        command = ["mpirun", "-np", str(ranks), *command]
    result = subprocess.run(command, cwd=directory, capture_output=True,
                            text=True, timeout=120, check=False)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted((directory / "bin").glob("*.bin"))
    assert len(paths) >= 2, result.stdout
    return [read_binary(str(path)) for path in paths]


def ordered_fields(output):
    """Canonical leaf ordering independent of rank ownership or output order."""
    logical = np.asarray(output["mb_logical"])
    order = np.lexsort(logical.T[::-1])
    return logical[order], {
        name: np.asarray(values, dtype=float)[order]
        for name, values in output["mb_data"].items()
    }


def cell_volumes(output):
    geometry = np.asarray(output["mb_geometry"])
    block_volumes = np.prod(geometry[:, 1::2] - geometry[:, ::2], axis=1)
    cells = np.asarray(output["mb_data"][GROUPS[0]]).shape[1:]
    return block_volumes / np.prod(cells)


def moments(output, group):
    """Volume-weighted first and second moments over the composite leaf mesh."""
    values = np.asarray(output["mb_data"][group], dtype=float)
    weights = cell_volumes(output)[:, None, None, None]
    return (np.sum(values * weights), np.sum(values * values * weights))


def assert_transport(outputs, adaptive=None):
    initial, final = outputs[0], outputs[-1]
    assert final["cycle"] == 3
    assert np.isclose(final["time"], 7.0e-4, rtol=0.0, atol=2.0e-15)
    assert set(np.asarray(initial["mb_logical"])[:, 3]) == {0, 1}
    initial_topology, _ = ordered_fields(initial)
    final_topology, _ = ordered_fields(final)
    if adaptive is None:
        assert np.array_equal(final_topology, initial_topology)
    else:
        assert not np.array_equal(final_topology, initial_topology)
        expected = {1} if adaptive == "refine" else {0}
        assert set(final_topology[:, 3]) == expected
        # Confirm at least one transport step follows the topology change.
        assert any(not np.array_equal(ordered_fields(out)[0], initial_topology)
                   for out in outputs[1:-1])
    for group in GROUPS:
        initial_energy, initial_square = moments(initial, group)
        low = np.min(initial["mb_data"][group])
        high = np.max(initial["mb_data"][group])
        for output in outputs[1:]:
            values = np.asarray(output["mb_data"][group])
            assert np.all(np.isfinite(values)), group
            assert np.min(values) >= low - 2.0e-10, group
            assert np.max(values) <= high + 2.0e-10, group
            energy, square = moments(output, group)
            assert np.isclose(energy, initial_energy, rtol=2.0e-10,
                              atol=2.0e-12), (group, initial_energy, energy)
            assert square < initial_square, (group, initial_square, square)
        assert moments(final, group)[1] < 0.99 * initial_square, group


def assert_same_solution(reference, candidate):
    assert len(reference) == len(candidate)
    for left, right in zip(reference, candidate):
        assert left["cycle"] == right["cycle"]
        assert np.isclose(left["time"], right["time"], rtol=0.0, atol=2.0e-15)
        left_topology, left_fields = ordered_fields(left)
        right_topology, right_fields = ordered_fields(right)
        assert np.array_equal(left_topology, right_topology)
        for field in GROUPS:
            np.testing.assert_allclose(right_fields[field], left_fields[field],
                                       rtol=2.0e-9, atol=2.0e-11,
                                       err_msg=field)


def boundary_flags(boundary):
    flags = ["refined_region1/x1max=0.5"]
    value = 0.0 if boundary == "vacuum" else 0.1
    for side, end in (("i", "inner"), ("o", "outer")):
        flags += [f"mesh/{side}x1_bc=outflow",
                  f"thermal_radiation/implicit_x1_{end}_boundary={boundary}",
                  f"thermal_radiation/implicit_x1_{end}_value={value}"]
    return flags


def assert_boundary_transport(outputs, boundary):
    """Check prescribed-boundary positivity, loss, and closed-wall conservation."""
    if boundary == "neumann":
        assert_transport(outputs)
    initial = outputs[0]
    for group in GROUPS:
        high = np.max(initial["mb_data"][group])
        if boundary == "dirichlet":
            high = max(high, 0.1)
        last_energy = moments(initial, group)[0]
        last_time = initial["time"]
        for output in outputs[1:]:
            values = np.asarray(output["mb_data"][group])
            assert np.all(np.isfinite(values)) and np.min(values) >= 0.0
            assert np.max(values) <= high + 2.0e-10
            energy = moments(output, group)[0]
            if boundary == "vacuum" and output["time"] > last_time:
                assert energy < last_energy
            last_energy = energy
            last_time = output["time"]


def assert_1d_reference(outputs, boundary="periodic"):
    """Independent dense backward-Euler solve on the actual nonuniform cells."""
    initial = outputs[0]
    centers = []
    widths = []
    for geometry, block in zip(initial["mb_geometry"],
                               initial["mb_data"][GROUPS[0]]):
        count = np.asarray(block).size
        dx = (geometry[1] - geometry[0]) / count
        centers.extend(geometry[0] + (np.arange(count) + 0.5) * dx)
        widths.extend([dx] * count)
    order = np.argsort(centers)
    widths = np.asarray(widths)[order]
    assert np.unique(widths).size == 2
    count = len(widths)
    density = np.asarray(initial["mb_data"]["dens"]).ravel()[order]
    for group, opacity in zip(GROUPS, (100.0, 35.0)):
        expected = np.asarray(initial["mb_data"][group]).ravel()[order]
        diffusivity = 1000.0 / (3.0 * density * opacity)
        for previous, output in zip(outputs, outputs[1:]):
            dt = output["time"] - previous["time"]
            matrix = np.eye(count)
            rhs = expected.copy()
            for cell in range(count):
                for neighbor in (cell - 1, cell + 1):
                    if boundary != "periodic" and neighbor in (-1, count):
                        conductance = diffusivity[cell] / widths[cell]
                        if boundary == "dirichlet":
                            coefficient = 2.0 * dt * conductance / widths[cell]
                            matrix[cell, cell] += coefficient
                            rhs[cell] += coefficient * 0.1
                        elif boundary == "vacuum":
                            matrix[cell, cell] += (
                                dt * min(conductance, 500.0) / widths[cell])
                        else:
                            assert boundary == "neumann"
                        continue
                    neighbor %= count
                    if widths[cell] == widths[neighbor]:
                        conductance = (
                            0.5 * (diffusivity[cell] + diffusivity[neighbor])
                            / widths[cell])
                    else:
                        resistance = (0.5 * widths[cell] / diffusivity[cell]
                                      + 0.5 * widths[neighbor] / diffusivity[neighbor])
                        conductance = 1.0 / resistance
                    coefficient = dt * conductance / widths[cell]
                    matrix[cell, cell] += coefficient
                    matrix[cell, neighbor] -= coefficient
            expected = np.linalg.solve(matrix, rhs)
            actual = np.asarray(output["mb_data"][group]).ravel()[order]
            np.testing.assert_allclose(actual, expected, rtol=3.0e-9,
                                       atol=3.0e-11, err_msg=group)
