"""Exact straight rays through Cartesian AMR leaf cells; no uniform regridding."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from radiation_transfer import formal_step


class LeafMesh:
    def __init__(self, raw):
        self.low = np.array([raw[f"x{i}min"] for i in (1, 2, 3)])
        self.high = np.array([raw[f"x{i}max"] for i in (1, 2, 3)])
        self.shape = np.array([raw[f"nx{i}_mb"] for i in (1, 2, 3)], dtype=int)
        root = np.array([raw[f"Nx{i}"] for i in (1, 2, 3)], dtype=int)
        if (np.any(~np.isfinite(self.low)) or np.any(~np.isfinite(self.high))
                or np.any(self.high <= self.low) or np.any(root <= 1)
                or np.any(self.shape < 1) or np.any(root % self.shape)):
            raise ValueError("Expected a three-dimensional Cartesian mesh of complete blocks")
        n = raw["n_mbs"]
        self.bounds = np.array(raw["mb_geometry"], dtype=float).reshape(n, 3, 2)
        logical = np.asarray(raw["mb_logical"])
        expected = np.column_stack((np.zeros(3, dtype=int), self.shape - 1)).ravel()
        if not np.all(np.asarray(raw["mb_index"]) == expected):
            raise ValueError("Ray tracing requires full-volume active cells, without slices or ghosts")
        if (not n or logical.shape != (n, 4) or np.any(logical != logical.astype(int))
                or np.any(logical < 0) or np.any(logical[:, 3] > 30)):
            raise ValueError("Invalid AMR logical locations")
        self.levels = logical[:, 3].astype(int)
        locations = set()
        root_blocks = root // self.shape
        for b, (x, y, z, level) in enumerate(logical):
            key = (int(level), int(x), int(y), int(z))
            if key in locations:
                raise ValueError("Overlapping AMR leaves (duplicate block)")
            locations.add(key)
            count = root_blocks * 2**int(level)
            position = np.array([x, y, z])
            if np.any(position >= count):
                raise ValueError("AMR block lies outside the domain")
            width = (self.high - self.low) / count
            expected_low = self.low + position * width
            expected_bounds = np.column_stack((expected_low, self.low + (position + 1) * width))
            # Location data can be float32. Canonical logical geometry below makes
            # shared faces identical even when the saved endpoints are rounded.
            tolerance = 2e-7 * np.max(self.high - self.low)
            if not np.allclose(self.bounds[b], expected_bounds, rtol=0, atol=tolerance):
                raise ValueError("AMR geometry disagrees with its logical location")
            self.bounds[b] = expected_bounds
        for level, x, y, z in locations:
            for parent in range(level):
                divisor = 2**(level - parent)
                if (parent, x // divisor, y // divisor, z // divisor) in locations:
                    raise ValueError("Overlapping AMR leaves (parent and child)")
        maximum = int(self.levels.max())
        volume = sum(8**(maximum - int(level)) for level in self.levels)
        if volume != int(np.prod(root_blocks)) * 8**maximum:
            raise ValueError("Incomplete AMR volume; missing leaves or rank files")


@dataclass
class Camera:
    origins: np.ndarray
    direction: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    center: np.ndarray
    u_edges: np.ndarray
    v_edges: np.ndarray


def make_camera(mesh, pixels=(256, 256), observer=(0, 0, 1), up=(1, 0, 0),
                center=None, extent=None):
    """Orthographic camera: observer is the vector FROM plasma TO the observer.

    Coordinates/edges are in code length. Default +x3 view has +x2 right,
    +x1 up. For an axial view, choose --up 0 0 1 (up cannot parallel observer).
    """
    nx, ny = pixels
    if min(nx, ny) < 1 or int(nx) != nx or int(ny) != ny:
        raise ValueError("Image dimensions must be positive integers")
    direction, vertical = np.asarray(observer, dtype=float), np.asarray(up, dtype=float)
    if (direction.shape != (3,) or vertical.shape != (3,)
            or np.any(~np.isfinite(direction)) or np.any(~np.isfinite(vertical))
            or np.linalg.norm(direction) == 0 or np.linalg.norm(vertical) == 0):
        raise ValueError("Camera observer/up vectors must be finite and nonzero")
    direction = direction / np.linalg.norm(direction)
    # Remove numerical zeros so axis-aligned views have exact parallel slabs.
    direction[np.abs(direction) < 1e-14] = 0
    direction /= np.linalg.norm(direction)
    vertical = vertical / np.linalg.norm(vertical)
    horizontal = np.cross(direction, vertical)
    if np.linalg.norm(horizontal) < 1e-12:
        raise ValueError("Camera up must not be parallel to observer; set --up")
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(horizontal, direction)
    center = (0.5 * (mesh.low + mesh.high) if center is None
              else np.asarray(center, dtype=float))
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        raise ValueError("Camera center must have three finite coordinates")
    if extent is None:
        corners = np.array(list(product(*zip(mesh.low, mesh.high)))) - center
        u, v = corners @ horizontal, corners @ vertical
        extent = [u.min(), u.max(), v.min(), v.max()]
    if (len(extent) != 4 or np.any(~np.isfinite(extent))
            or extent[0] >= extent[1] or extent[2] >= extent[3]):
        raise ValueError("Camera extent must be UMIN UMAX VMIN VMAX with positive widths")
    ue, ve = np.linspace(*extent[:2], int(nx) + 1), np.linspace(*extent[2:], int(ny) + 1)
    u, v = np.meshgrid(0.5 * (ue[1:] + ue[:-1]), 0.5 * (ve[1:] + ve[:-1]))
    origins = center + u.ravel()[:, None] * horizontal + v.ravel()[:, None] * vertical
    return Camera(origins, direction, horizontal, vertical, center, ue, ve)


def intersect_box(origins, direction, bounds):
    """Ray slab intervals. Parallel faces use half-open [lower,upper) ownership."""
    near, far = np.full(len(origins), -np.inf), np.full(len(origins), np.inf)
    for axis in range(3):
        low, high = bounds[axis]
        if direction[axis] == 0:
            outside = (origins[:, axis] < low) | (origins[:, axis] >= high)
            near[outside], far[outside] = np.inf, -np.inf
        else:
            a = (low - origins[:, axis]) / direction[axis]
            b = (high - origins[:, axis]) / direction[axis]
            near, far = np.maximum(near, np.minimum(a, b)), np.minimum(far, np.maximum(a, b))
    return near, far


def trace_block(origins, direction, bounds, shape, entry, exit_, alpha, emissivity):
    """Traverse native cell faces from far side to observer using vectorized DDA."""
    width = (bounds[:, 1] - bounds[:, 0]) / shape
    position = (origins + entry[:, None] * direction - bounds[:, 0]) / width
    rounded = np.rint(position)
    position = np.where(np.abs(position - rounded) < 64 * np.finfo(float).eps * shape,
                        rounded, position)
    position = np.nextafter(position, position + np.sign(direction))
    indices = np.clip(np.floor(position).astype(int), 0, shape - 1)
    intensity, depth = np.zeros((len(origins), alpha.shape[-1])), np.zeros((len(origins), alpha.shape[-1]))
    path = np.zeros(len(origins))
    current = entry.copy()
    active = np.arange(len(origins))
    tolerance = 64 * np.finfo(float).eps * max(np.max(np.abs(bounds)), np.max(width))
    for _ in range(int(shape.sum()) + 3):
        if not len(active):
            break
        ijk = indices[active]
        if np.any(ijk < 0) or np.any(ijk >= shape):
            raise ValueError("Ray left a block before reaching its exit face")
        faces = np.full((len(active), 3), np.inf)
        for axis in range(3):
            if direction[axis] != 0:
                boundary = bounds[axis, 0] + (ijk[:, axis] + (direction[axis] > 0)) * width[axis]
                faces[:, axis] = (boundary - origins[active, axis]) / direction[axis]
        end = np.minimum(faces.min(axis=1), exit_[active])
        ds = np.maximum(end - current[active], 0)
        if np.any(end < current[active] - tolerance):
            raise ValueError("Ray traversal made a negative step")
        cell = (ijk[:, 2] * shape[1] + ijk[:, 1]) * shape[0] + ijk[:, 0]
        intensity[active] = formal_step(intensity[active], alpha[cell], emissivity[cell], ds[:, None])
        depth[active] += alpha[cell] * ds[:, None]
        path[active] += ds
        crossed = faces <= end[:, None] + tolerance
        indices[active] += crossed * np.sign(direction).astype(int)
        current[active] = end
        active = active[end < exit_[active] - tolerance]
    if len(active) or not np.allclose(path, exit_ - entry, rtol=1e-12, atol=tolerance):
        raise ValueError("Ray traversal did not cover its complete block chord")
    return intensity, depth


def trace_image(mesh, camera, coefficient_function, background, progress=None):
    """Integrate each block, then compose its transfer operator in per-ray order.

    Memory scales with image rays/block intersections, plus one block's plasma
    coefficients. The volume is never expanded to the finest AMR cell grid.
    """
    background = np.atleast_1d(np.asarray(background, dtype=float))
    if background.ndim != 1 or np.any(~np.isfinite(background)) or np.any(background < 0):
        raise ValueError("Background group intensities must be finite and non-negative")
    rays, starts, stops, emissions, depths = [], [], [], [], []
    for block, bounds in enumerate(mesh.bounds):
        entry, exit_ = intersect_box(camera.origins, camera.direction, bounds)
        selected = np.flatnonzero(exit_ > entry)
        if len(selected):
            alpha, emissivity = coefficient_function(block)
            expected = (int(np.prod(mesh.shape)), len(background))
            if (alpha.shape != expected or emissivity.shape != expected
                    or np.any(~np.isfinite(alpha)) or np.any(alpha < 0)
                    or np.any(~np.isfinite(emissivity)) or np.any(emissivity < 0)):
                raise ValueError("Invalid block emission/absorption coefficients")
            emission, depth = trace_block(camera.origins[selected], camera.direction, bounds,
                                           mesh.shape, entry[selected], exit_[selected], alpha, emissivity)
            rays.append(selected)
            starts.append(entry[selected])
            stops.append(exit_[selected])
            emissions.append(emission)
            depths.append(depth)
        if progress is not None:
            progress(block + 1, len(mesh.bounds))
    count = len(camera.origins)
    intensity = np.broadcast_to(background, (count, len(background))).copy()
    tau, length = np.zeros_like(intensity), np.zeros(count)
    crossings = np.zeros(count, dtype=int)
    if rays:
        rays, starts, stops = np.concatenate(rays), np.concatenate(starts), np.concatenate(stops)
        emission, depth = np.concatenate(emissions), np.concatenate(depths)
        order = np.lexsort((starts, rays))
        rays, starts, stops, emission, depth = (values[order] for values in
                                               (rays, starts, stops, emission, depth))
        crossings = np.bincount(rays, minlength=count)
        offsets = np.cumsum(crossings) - crossings
        last_exit = np.full(count, -np.inf)
        tolerance = 1e-11 * np.max(mesh.high - mesh.low)
        for step in range(int(crossings.max())):
            selected = np.flatnonzero(crossings > step)
            hit = offsets[selected] + step
            if np.any(starts[hit] < last_exit[selected] - tolerance):
                raise ValueError("Overlapping ray segments across AMR leaves")
            if step and np.any(np.abs(starts[hit] - last_exit[selected]) > tolerance):
                raise ValueError("Uncovered ray segment between AMR leaves")
            intensity[selected] = intensity[selected] * np.exp(-depth[hit]) + emission[hit]
            tau[selected] += depth[hit]
            length[selected] += stops[hit] - starts[hit]
            last_exit[selected] = stops[hit]
    domain_entry, domain_exit = intersect_box(camera.origins, camera.direction,
                                             np.column_stack((mesh.low, mesh.high)))
    expected_length = np.maximum(domain_exit - domain_entry, 0)
    if not np.allclose(length, expected_length, rtol=1e-11, atol=1e-12 * np.max(mesh.high - mesh.low)):
        raise ValueError("Rays did not cover the full domain chord")
    if np.any(~np.isfinite(intensity)) or np.any(~np.isfinite(tau)):
        raise ValueError("Non-finite radiation image")
    shape = (len(camera.v_edges) - 1, len(camera.u_edges) - 1)
    return {"intensity": intensity.T.reshape((-1, *shape)),
            "optical_depth": tau.T.reshape((-1, *shape)),
            "path_length": length.reshape(shape), "block_crossings": crossings.reshape(shape)}
