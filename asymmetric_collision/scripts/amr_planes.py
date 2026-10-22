"""Extract collision slices from AthenaK leaf blocks without interpolation."""

import numpy as np


def load_amr_plane(fluid, thermal, plane, scales):
    """Rasterize leaf cells exactly, repeating coarse values on the finest grid.

    Select the cell containing zero separately in each intersecting leaf block,
    using the same half-open bounds and CellCenterIndex formula as AthenaK output.
    Native cell centers are retained for velocity arrows and volume comparisons.
    Fluid/thermal metadata must first be checked by plot_dumps.read_pair.
    """
    if plane not in ("xy", "xz"):
        raise ValueError(f"Unsupported plane: {plane}")
    normal_axis, horizontal_axis = (2, 1) if plane == "xy" else (1, 2)
    root = np.array([fluid[f"Nx{i}"] for i in (1, 2, 3)], dtype=int)
    block_shape = np.array([fluid[f"nx{i}_mb"] for i in (1, 2, 3)], dtype=int)
    if np.any(root <= 1) or np.any(block_shape <= 1):
        raise ValueError("Collision AMR slices require a three-dimensional mesh")
    low = np.array([fluid[f"x{i}min"] for i in (1, 2, 3)])
    high = np.array([fluid[f"x{i}max"] for i in (1, 2, 3)])
    logical = np.asarray(fluid["mb_logical"])
    indices = np.asarray(fluid["mb_index"])
    geometry = np.asarray(fluid["mb_geometry"])
    if logical.shape != (fluid["n_mbs"], 4) or np.any(logical[:, 3] < 0):
        raise ValueError("Invalid AMR logical locations")

    selected = []
    for block, bounds in enumerate(geometry):
        lower, upper = bounds[2 * normal_axis:2 * normal_axis + 2]
        if not lower <= 0.0 < upper:
            continue
        normal_index = int(((0.0 - lower) / (upper - lower)) * block_shape[normal_axis])
        begin, end = indices[block, 2 * normal_axis:2 * normal_axis + 2]
        if not begin <= normal_index <= end:
            raise ValueError("Saved slice does not contain the requested zero-plane cell")
        selected.append((block, normal_index))
    if not selected:
        raise ValueError("No AMR blocks intersect the zero plane")

    maximum_level = max(int(logical[block, 3]) for block, _ in selected)
    effective_grid = root * 2 ** maximum_level
    shape = (int(effective_grid[0]), int(effective_grid[horizontal_axis]))
    specs = {
        "density": (fluid, "dens", scales["density_gcc"]),
        "Ti": (thermal, "tion", scales["temperature_ev"]),
        "Te": (thermal, "tele", scales["temperature_ev"]),
        "v1": (fluid, "velx", scales["velocity_kms"]),
        "v2": (fluid, "vely", scales["velocity_kms"]),
        "v3": (fluid, "velz", scales["velocity_kms"]),
        "upper_tag": (fluid, "s_02", 1.0),
        "lower_tag": (fluid, "s_03", 1.0),
    }
    data = {name: np.full(shape, np.nan) for name in specs}
    for name in ("cell_x", "cell_y", "normal_coordinate", "cell_level"):
        data[name] = np.full(shape, np.nan)
    coverage = np.zeros(shape, dtype=np.uint16)
    block_counts, native_cells, widths, slice_blocks = {}, 0, [], []

    for block, normal_index in selected:
        level = int(logical[block, 3])
        repeat = 2 ** (maximum_level - level)
        cell_width = (high - low) / (root * 2 ** level)
        bounds = geometry[block].reshape(3, 2)
        expected_low = low + logical[block, :3] * block_shape * cell_width
        if (not np.allclose(bounds[:, 0], expected_low, rtol=0, atol=1e-12)
                or not np.allclose(bounds[:, 1] - bounds[:, 0],
                                   block_shape * cell_width, rtol=1e-10, atol=1e-12)):
            raise ValueError("AMR block geometry disagrees with its logical location")
        starts, ends = indices[block, ::2], indices[block, 1::2] + 1
        if np.any(starts < 0) or np.any(ends > block_shape):
            raise ValueError("AMR collision plotting expects active cells without ghosts")
        start = (logical[block, :3] * block_shape + starts) * repeat
        end = (logical[block, :3] * block_shape + ends) * repeat
        if np.any(start < 0) or np.any(end > effective_grid):
            raise ValueError("AMR cells lie outside the declared domain")
        region = (slice(start[0], end[0]),
                  slice(start[horizontal_axis], end[horizontal_axis]))
        if np.any(coverage[region]):
            raise ValueError("Overlapping AMR leaf cells in the requested plane")
        coverage[region] += 1
        local_normal = normal_index - starts[normal_axis]
        expected_shape = tuple((ends - starts)[::-1])
        for name, (raw, variable, scale) in specs.items():
            values = np.asarray(raw["mb_data"][variable][block], dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(f"Unexpected AMR block shape for {variable}: {values.shape}")
            values = (values[local_normal, :, :] if plane == "xy"
                      else values[:, local_normal, :]).T
            data[name][region] = values.repeat(repeat, axis=0).repeat(repeat, axis=1) * scale

        # Arrows originate at actual native centers, even where coarse cells repeat.
        centers = {}
        for axis in (0, horizontal_axis):
            positions = logical[block, axis] * block_shape[axis] + np.arange(starts[axis], ends[axis])
            centers[axis] = (low[axis] + (positions + 0.5) * cell_width[axis]) * scales["length_um"]
        data["cell_x"][region] = np.repeat(centers[horizontal_axis], repeat)[None, :]
        data["cell_y"][region] = np.repeat(centers[0], repeat)[:, None]
        normal_width = (bounds[normal_axis, 1] - bounds[normal_axis, 0]) / block_shape[normal_axis]
        normal_center = bounds[normal_axis, 0] + (normal_index + 0.5) * normal_width
        data["normal_coordinate"][region] = normal_center * scales["length_um"]
        data["cell_level"][region] = level
        block_counts[str(level)] = block_counts.get(str(level), 0) + 1
        native_cells += int((ends[0] - starts[0]) * (ends[horizontal_axis] - starts[horizontal_axis]))
        widths.append(cell_width * scales["length_um"])
        slice_blocks.append({
            "level": level,
            "bounds_um": (bounds[[horizontal_axis, 0], :].ravel()
                          * scales["length_um"]).tolist(),
        })

    if np.any(coverage != 1):
        raise ValueError(f"Incomplete AMR plane: {np.count_nonzero(coverage != 1)} uncovered pixels")
    if any(np.any(~np.isfinite(values)) for values in data.values()):
        raise ValueError("AMR plane contains non-finite values")
    data["speed"] = np.sqrt(sum(data[name] ** 2 for name in ("v1", "v2", "v3")))
    data["horizontal_velocity"] = data["v2" if plane == "xy" else "v3"]
    edges = [np.linspace(low[i], high[i], effective_grid[i] + 1) * scales["length_um"]
             for i in (horizontal_axis, 0)]
    normal_range = [float(data["normal_coordinate"].min()), float(data["normal_coordinate"].max())]
    normal = normal_range[0] if np.isclose(*normal_range, rtol=0, atol=1e-10) else None
    widths = np.asarray(widths)
    mesh = {
        "type": "amr", "root_grid": root.tolist(),
        "effective_grid_at_finest_slice_level": effective_grid.tolist(),
        "slice_levels": sorted(int(level) for level in block_counts),
        "slice_blocks_by_level": block_counts, "native_slice_cells": native_cells,
        "slice_blocks": slice_blocks,
        "slice_block_bounds_order": ["horizontal_min", "horizontal_max", "x1_min", "x1_max"],
        "display_shape": list(shape), "requested_normal_um": 0.0,
        "normal_cell_center_range_um": normal_range,
        "cell_width_range_um": {f"x{i + 1}": [float(widths[:, i].min()), float(widths[:, i].max())]
                                for i in range(3)},
        "rasterization": "piecewise constant leaf cells; coarse values repeated without interpolation",
        "coverage_complete_without_overlap": True,
    }
    return data, edges[0], edges[1], normal, mesh
