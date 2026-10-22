"""Spatial accuracy and independent composite flux checks for quadratic AMR FLD.

Smooth cell averages are injected into a disposable static checkpoint.  The
manufactured right-hand side gives an exact continuum backward-Euler solution,
so these tests measure spatial error without contamination by time truncation.
"""

import itertools
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import INPUT, read_binary


def _run(directory, arguments, ranks=None):
    directory.mkdir(parents=True)
    executable = Path(os.environ.get("ATHENA_AMR_TEST_EXE", "./athena")).resolve()
    command = [str(executable), *arguments]
    if ranks is not None:
        command = ["mpirun", "-np", str(ranks), *command]
    result = subprocess.run(command, cwd=directory, capture_output=True,
                            text=True, timeout=180, check=False)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted((directory / "bin").glob("*.bin"))
    assert paths, result.stdout
    return read_binary(str(paths[-1]))


def _manufactured(x, y, z, width, dimensions, dt, variable_coefficient=False):
    mode = 0.1 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    if dimensions == 3:
        mode *= np.sin(2 * np.pi * z)
    mode *= np.sinc(width)**dimensions
    target = 1.0 + mode
    rhs = 1.0 + (1.0 + dt * dimensions * 4 * np.pi**2) * mode
    if variable_coefficient:
        assert dimensions == 2
        # K=1+a*cos(2*pi*y): -div(K*grad(u)) has an additional
        # 6*a*pi^2*0.1*cos(2*pi*x)*sin(4*pi*y) mode.
        rhs += (dt * 6 * 0.25 * np.pi**2 * 0.1
                * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)
                * np.sinc(width) * np.sinc(2 * width))
    return target, rhs


def manufactured_run(directory, n=16, dimensions=2, dt=0.01,
                     variable_coefficient=False, extra_flags=(), ranks=None,
                     impulse=False, normal=0, ghosts=2, block_size=8):
    """Run one step from manufactured cell averages, preserving the native layout."""
    directory.mkdir()
    input_file = directory / "manufactured.athinput"
    text = INPUT.read_text()
    for key, value in (("implicit_amr_order", 2), ("implicit_gmres_restart", 30)):
        if f"{key} =" not in text:
            text = text.replace("<thermal_radiation>\n",
                                f"<thermal_radiation>\n{key} = {value}\n")
    input_file.write_text(text + "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    flags = [f"mesh/nx1={n}", f"mesh/nx2={n}",
             f"meshblock/nx1={block_size}", f"meshblock/nx2={block_size}",
             f"mesh/nghost={ghosts}",
             f"mesh/nx3={n if dimensions == 3 else 1}",
             f"meshblock/nx3={block_size if dimensions == 3 else 1}",
             "refined_region1/x1min=-0.5", "refined_region1/x1max=0.5",
             "refined_region1/x2min=-0.5", "refined_region1/x2max=0.5",
             "refined_region1/x3min=-0.5", "refined_region1/x3max=0.5",
             f"refined_region1/x{normal+1}min=0",
             "mesh_refinement/max_nmb_per_rank=1024", "time/nlim=0",
             f"time/initial_dt={dt}", f"time/tlim={dt}",
             "time/align_outputs=false", "thermal_radiation/n_groups=1",
             "thermal_radiation/group_bound_1=100", "thermal_radiation/c_light=3",
             "thermal_radiation/kappa_transport_0=1",
             "thermal_radiation/initial_profile=uniform",
             "thermal_radiation/implicit_tolerance=1e-12",
             *extra_flags]
    source = directory / "source"
    geometry = _run(source, ["-i", str(input_file), *flags])
    checkpoint = sorted((source / "rst").glob("*.rst"))[-1]
    payload = bytearray(checkpoint.read_bytes())
    extent = block_size + 2 * ghosts
    shape = (geometry["n_mbs"], 8, extent if dimensions == 3 else 1, extent, extent)
    per_block = int(np.prod(shape[1:]) * np.dtype(np.float64).itemsize)
    start = len(payload) - int(shape[0] * per_block)
    # Eight conserved components (rho, momentum3, E, ion, electron, radiation).
    # Static checkpoints have no adaptive-age footer.  Refuse another format.
    assert np.frombuffer(payload, dtype=np.uint64, count=1,
                         offset=start - 8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    for block, bounds in enumerate(geometry["mb_geometry"]):
        h = (bounds[1] - bounds[0]) / block_size
        x = bounds[0] + (np.arange(extent) - ghosts + 0.5) * h
        y = bounds[2] + (np.arange(extent) - ghosts + 0.5) * h
        z = (bounds[4] + (np.arange(extent) - ghosts + 0.5) * h
             if dimensions == 3 else np.array([0.0]))
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        _, rhs = _manufactured(xx, yy, zz, h, dimensions, dt,
                               variable_coefficient)
        if impulse:
            rhs = np.full_like(rhs, 1.0e-12)
            selected = (np.isclose(xx, -0.5 / n)
                        & np.isclose(yy, 0.5 / n))
            rhs[selected] = 1.0
        state[block, 7] = rhs
        if variable_coefficient:
            # Constant gas pressure and zero velocity make this density profile
            # stationary. With chat=3 and kappa=1 the frozen K equals 1/rho.
            state[block, 0] = 1.0 / (1.0 + 0.25 * np.cos(2 * np.pi * yy))
    injected = directory / "manufactured.rst"
    injected.write_bytes(payload)
    output = _run(directory / "solve",
                  ["-r", str(injected), "time/nlim=1"], ranks)
    assert output["cycle"] == 1
    np.testing.assert_allclose(output["time"], dt, rtol=0, atol=1e-15)
    return output


def leaf_geometry(output, n, dimensions):
    """Map leaf cells onto a finest-level ownership raster, independent of blocks."""
    block_size = np.asarray(output["mb_data"]["erad00_d"]).shape[-1]
    cells_per_block = block_size**dimensions
    count = output["n_mbs"] * cells_per_block
    owner = np.full((2 * n,) * dimensions, -1, dtype=int)
    coordinates = np.empty((3, count))
    widths = np.empty(count)
    for block, bounds in enumerate(output["mb_geometry"]):
        h = (bounds[1] - bounds[0]) / block_size
        begin, end = block * cells_per_block, (block + 1) * cells_per_block
        widths[begin:end] = h
        axes = [bounds[2 * axis] + (np.arange(block_size) + 0.5) * h
                for axis in range(dimensions)]
        grids = np.meshgrid(*axes[::-1], indexing="ij")[::-1]
        coordinates[:, begin:end] = 0.0
        for axis, grid in enumerate(grids):
            coordinates[axis, begin:end] = grid.ravel()
        ids = np.arange(begin, end).reshape((block_size,) * dimensions)
        scale = int(round(2 * n * h))
        for axis in range(dimensions):
            ids = np.repeat(ids, scale, axis=axis)
        start = [int(round((bounds[2 * axis] + 0.5) * 2 * n))
                 for axis in range(dimensions)][::-1]
        slices = tuple(slice(first, first + block_size * scale) for first in start)
        owner[slices] = ids
    assert np.all(owner >= 0)
    return owner, coordinates, widths


def _subface_trace(values, owner, cpos, fpos, axis, positive=False):
    length = owner.shape[0]
    tangent_axes = [a for a in range(owner.ndim) if a != axis]
    trace = np.zeros(cpos.shape[1])
    minimum = values[owner[tuple(cpos)]].copy()
    maximum = minimum.copy()
    for offsets in itertools.product((-1, 0, 1), repeat=owner.ndim - 1):
        donor = cpos.copy()
        weight = np.ones_like(trace)
        for tangent, offset in zip(tangent_axes, offsets):
            donor[tangent] = (donor[tangent] + 2 * offset) % length
            if offset:
                weight *= offset * (2 * (fpos[tangent] % 2) - 1) / 8
        value = values[owner[tuple(donor)]]
        trace += weight * value
        minimum = np.minimum(minimum, value)
        maximum = np.maximum(maximum, value)
    return np.clip(trace, minimum, maximum) if positive else trace


def composite_diffusion(values, owner, widths, order=2, coefficient=None):
    """Independent face-area sum on a periodic half-plane composite leaf mesh."""
    dimensions = owner.ndim
    length = owner.shape[0]
    area = (1.0 / length)**(dimensions - 1)
    result = np.zeros_like(values)
    if coefficient is None:
        coefficient = np.ones_like(values)
    grid = np.indices(owner.shape)
    for axis in range(dimensions):
        adjacent = np.roll(owner, -1, axis=axis)
        mask = owner != adjacent
        a, b = owner[mask], adjacent[mask]
        flux = 2.0 * (values[a] - values[b]) / (widths[a] + widths[b])
        flux *= 0.5 * (coefficient[a] + coefficient[b])
        unequal = widths[a] != widths[b]
        if order == 2 and np.any(unequal):
            # Donors one coarse cell away transversely stay on the coarse half-plane.
            pos_a = grid[:, mask][:, unequal]
            pos_b = pos_a.copy()
            pos_b[axis] = (pos_b[axis] + 1) % length
            coarse_left = widths[a[unequal]] > widths[b[unequal]]
            cpos = np.where(coarse_left, pos_a, pos_b)
            fpos = np.where(coarse_left, pos_b, pos_a)
            f0 = owner[tuple(fpos)]
            second = fpos.copy()
            second[axis] = (second[axis] + np.where(coarse_left, 1, -1)) % length
            f1 = owner[tuple(second)]
            trace = _subface_trace(values, owner, cpos, fpos, axis)
            kc = _subface_trace(coefficient, owner, cpos, fpos, axis, True)
            resistance = 1 / kc + 0.5 / coefficient[f0] + 0.5 / coefficient[f1]
            flux[unequal] = (np.where(coarse_left, 1, -1) * length / resistance
                             * (trace - 0.5 * (values[f0] + values[f1])))
        np.add.at(result, a, area * flux / widths[a]**dimensions)
        np.add.at(result, b, -area * flux / widths[b]**dimensions)
    return result


def frozen_lp_coefficient(values, owner, xyz, widths, correct_interface=True):
    """Compute LP coefficients from leaf averages without any MeshBlock halo data."""
    dimensions, length = owner.ndim, owner.shape[0]
    origin = np.rint((xyz[:dimensions][::-1] + 0.5 - 0.5 * widths) * length).astype(int)
    scale = np.rint(widths * length).astype(int)
    gradient_squared = np.zeros_like(values)
    for axis in range(dimensions):
        adjacent = []
        for side in (-1, 1):
            average = np.empty_like(values)
            for ratio in (1, 2):
                mask = scale == ratio
                first = origin[:, mask].copy()
                first[axis] += side * ratio
                average[mask] = 0
                for offset in itertools.product(range(ratio), repeat=dimensions):
                    position = (first + np.array(offset)[:, None]) % length
                    average[mask] += values[owner[tuple(position)]] / ratio**dimensions
            adjacent.append(average)
        derivative = (adjacent[1] - adjacent[0]) / (2 * widths)
        if correct_interface:
            for side in (-1, 1):
                neighbor = origin.copy()
                neighbor[axis] = (neighbor[axis] + side) % length
                mask = (scale == 1) & (widths[owner[tuple(neighbor)]] > widths)
                if not np.any(mask):
                    continue
                cpos, fpos = neighbor[:, mask], origin[:, mask]
                trace = _subface_trace(values, owner, cpos, fpos, axis)
                second = fpos.copy()
                second[axis] = (second[axis] - side) % length
                f1 = values[owner[tuple(second)]]
                derivative[mask] = (-side * (-2 * trace - 3 * values[mask] + 5 * f1)
                                    / (8 * widths[mask]))
        gradient_squared += derivative**2
    reduced_gradient = np.sqrt(gradient_squared) / values
    return 3 * (2 + reduced_gradient) / (6 + 3 * reduced_gradient + reduced_gradient**2)


@pytest.mark.parametrize("variable_coefficient", (False, True))
def test_second_order_spatial_convergence(tmp_path, variable_coefficient):
    errors = []
    for n in (16, 32, 64):
        output = manufactured_run(tmp_path / f"n{n}", n=n,
                                  variable_coefficient=variable_coefficient)
        _, xyz, widths = leaf_geometry(output, n, 2)
        target, rhs = _manufactured(*xyz, widths, 2, 0.01, variable_coefficient)
        actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
        volumes = widths**2
        errors.append(np.sqrt(np.sum(volumes * (actual - target)**2)))
        np.testing.assert_allclose(np.dot(volumes, actual), np.dot(volumes, rhs),
                                   rtol=0, atol=2e-11)
        log = (tmp_path / f"n{n}" / "solve" / "run.log").read_text()
        assert "AMR fallback" not in log
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(orders > 1.8), (errors, orders)


@pytest.mark.parametrize(
    "dimensions,normal,ghosts", ((2, 0, 2), (3, 0, 2), (3, 1, 2), (3, 2, 4)))
def test_independent_composite_flux_residual(tmp_path, dimensions, normal, ghosts):
    output = manufactured_run(tmp_path / "mixed", dimensions=dimensions,
                              normal=normal, ghosts=ghosts)
    owner, xyz, widths = leaf_geometry(output, 16, dimensions)
    _, rhs = _manufactured(*xyz, widths, dimensions, 0.01)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    residual = actual + 0.01 * composite_diffusion(actual, owner, widths) - rhs
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)


def test_short_gmres_restarts(tmp_path):
    output = manufactured_run(
        tmp_path / "restart3",
        extra_flags=("thermal_radiation/implicit_gmres_restart=3",))
    owner, xyz, widths = leaf_geometry(output, 16, 2)
    _, rhs = _manufactured(*xyz, widths, 2, 0.01)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    residual = actual + 0.01 * composite_diffusion(actual, owner, widths) - rhs
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)


def test_flux_limiter_uses_composite_cell_gradient(tmp_path):
    output = manufactured_run(
        tmp_path / "lp",
        extra_flags=("thermal_radiation/flux_limiter=levermore-pomraning",))
    owner, xyz, widths = leaf_geometry(output, 16, 2)
    _, rhs = _manufactured(*xyz, widths, 2, 0.01)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    coefficient = frozen_lp_coefficient(rhs, owner, xyz, widths)
    residual = actual + 0.01 * composite_diffusion(
        actual, owner, widths, coefficient=coefficient) - rhs
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)
    old_coefficient = frozen_lp_coefficient(rhs, owner, xyz, widths, False)
    old_residual = actual + 0.01 * composite_diffusion(
        actual, owner, widths, coefficient=old_coefficient) - rhs
    assert np.max(np.abs(old_residual)) > 1e-5


def test_negative_high_order_solution_retries_conservative_operator(tmp_path):
    dt = 1.0e-5
    output = manufactured_run(tmp_path / "impulse", dt=dt, impulse=True)
    owner, xyz, widths = leaf_geometry(output, 16, 2)
    initial = np.full(widths.shape, 1.0e-12)
    initial[np.isclose(xyz[0], -0.5 / 16) & np.isclose(xyz[1], 0.5 / 16)] = 1
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
    assert np.all(actual >= 0)
    log = (tmp_path / "impulse" / "solve" / "run.log").read_text()
    assert "AMR fallback: group=0 reason=negative_energy" in log
    residual = actual + dt * composite_diffusion(actual, owner, widths, 1) - initial
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=3e-10)
    np.testing.assert_allclose(np.dot(widths**2, actual), np.dot(widths**2, initial),
                               rtol=0, atol=2e-12)
