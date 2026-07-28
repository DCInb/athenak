#!/usr/bin/env python3
"""Render DCI laser path/count maps with path-weighted ray directions.

AthenaK's ``laser`` output is a cell rasterization of ray segments, not a set of
individual ray polylines.  This tool plots that rasterization in an x1-x2 (xy) or
x1-x3 (xz) plane and overlays the projected, path-weighted mean segment direction.
It accepts both the inexpensive DCI plane outputs and collective full-volume files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import math
from pathlib import Path
import re
import sys
import types
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
VIS_PYTHON = REPO / "vis" / "python"
if str(VIS_PYTHON) not in sys.path:
    sys.path.insert(0, str(VIS_PYTHON))
try:
    from bin_convert import read_binary  # noqa: E402
except ModuleNotFoundError as exc:
    # The lean ``athena`` plotting environment omits h5py.  AthenaK .bin reading does
    # not use it, but the shared converter imports it for optional athdf routines.
    if exc.name != "h5py":
        raise
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    from bin_convert import read_binary  # type: ignore[no-redef]  # noqa: E402


DEFAULT_OUTPUT_DIR = CASE_DIR / "plots" / "laser_rays"
PLANE_AXES = {
    "xy": (2, 0, 1, "x_1", "x_2"),
    "xz": (1, 0, 2, "x_1", "x_3"),
}
AXIS_KEYS = ("x1", "x2", "x3")
DEFAULT_SHELL = (0.8, 1.0, 50.0)


@dataclass(frozen=True)
class LaserPlane:
    """A uniform DCI laser plane in physical plotting units."""

    source: Path
    plane: str
    quantity: str
    x_edges_mm: np.ndarray
    y_edges_mm: np.ndarray
    values: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    direction_coherence: np.ndarray
    time_ns: float
    cycle: int
    normal_location_mm: float
    shell_inner_mm: float
    shell_outer_mm: float
    shell_half_angle_deg: float


@dataclass(frozen=True)
class RenderOptions:
    norm: str = "log"
    vmin: float | None = None
    vmax: float | None = None
    cmap: str = "magma"
    vectors: str = "quiver"
    min_coherence: float = 0.20
    quiver_spacing_mm: float = 0.10
    quiver_length_mm: float = 0.11
    stream_density: float = 1.25
    stream_max_resolution: int = 256
    shell_outline: bool = False
    dpi: int = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+",
        help=(
            "laser .bin files, shell-style globs, or run/bin directories; a directory "
            "prefers its laser_xy/laser_xz slices and falls back to full laser dumps"
        ),
    )
    parser.add_argument("--plane", choices=tuple(PLANE_AXES), default="xy")
    parser.add_argument(
        "--location", type=float, default=0.0,
        help="requested coordinate normal to the plane in AthenaK code units",
    )
    parser.add_argument(
        "--quantity", choices=("path", "ray-count"), default="path",
        help="background raster: summed ray-segment path or segment count per cell",
    )
    parser.add_argument(
        "--vectors", choices=("quiver", "stream", "none"), default="quiver",
        help="overlay for path-weighted projected segment direction",
    )
    parser.add_argument("--norm", choices=("log", "linear"), default="log")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--min-coherence", type=float, default=0.20)
    parser.add_argument("--quiver-spacing-mm", type=float, default=0.10)
    parser.add_argument("--quiver-length-mm", type=float, default=0.11)
    parser.add_argument("--stream-density", type=float, default=1.25)
    parser.add_argument("--stream-max-resolution", type=int, default=256)
    parser.add_argument(
        "--shell-outline", action=argparse.BooleanOptionalAction, default=False,
        help="overlay the runtime CH cap (0.8--1.0 mm and 50 degrees in the DCI decks)",
    )
    parser.add_argument("--latest", action="store_true", help="render only the last file")
    parser.add_argument("--output", type=Path, help="PNG path; requires one selected input")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def input_header(lines: Iterable[str]) -> dict[str, dict[str, str]]:
    """Parse the parameter dump embedded in an AthenaK binary header."""
    blocks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            blocks[current][key.strip()] = value.strip()
    return blocks


def header_number(
    blocks: dict[str, dict[str, str]], block: str, key: str,
    default: float | None = None,
) -> float:
    try:
        return float(blocks[block][key].split()[0])
    except (KeyError, ValueError, IndexError) as exc:
        if default is not None:
            return default
        raise RuntimeError(f"Binary header is missing numeric <{block}>/{key}") from exc


def _axis_counts(raw: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(raw[f"Nx{axis}"]) for axis in (1, 2, 3))


def _block_counts(raw: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(raw[f"nx{axis}_mb"]) for axis in (1, 2, 3))


def _block_ranges(
    raw: dict[str, Any], block: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    logical = np.asarray(raw["mb_logical"])[block]
    indices = np.asarray(raw["mb_index"])[block]
    block_counts = _block_counts(raw)
    global_counts = _axis_counts(raw)
    result = []
    for axis in range(3):
        if global_counts[axis] == 1:
            # bin_convert subtracts nghost from the stored zero index even on a
            # degenerate dimension; normalize that harmless 2D convention here.
            start = end = 0
        else:
            start = int(logical[axis])*block_counts[axis]+int(indices[2*axis])
            end = int(logical[axis])*block_counts[axis]+int(indices[2*axis+1])
        result.append((start, end))
    return tuple(result)  # type: ignore[return-value]


def _axis_centers(raw: dict[str, Any], axis: int) -> np.ndarray:
    key = AXIS_KEYS[axis]
    count = _axis_counts(raw)[axis]
    edges = np.linspace(
        float(raw[f"{key}min"]), float(raw[f"{key}max"]), count+1,
        dtype=np.float64,
    )
    return 0.5*(edges[:-1]+edges[1:])


def _selected_normal_index(raw: dict[str, Any], axis: int, location: float) -> int:
    available: set[int] = set()
    for block in range(int(raw["n_mbs"])):
        low, high = _block_ranges(raw, block)[axis]
        available.update(range(low, high+1))
    if not available:
        raise RuntimeError("Laser dump contains no cells along the requested normal axis")
    centers = _axis_centers(raw, axis)
    valid = sorted(index for index in available if 0 <= index < centers.size)
    if not valid:
        raise RuntimeError("Laser dump cell indices are outside its declared mesh")
    distances = np.abs(centers[valid]-location)
    minimum = float(np.min(distances))
    tolerance = 64.0*np.finfo(float).eps*max(abs(location), 1.0)
    # AthenaK's CellCenterIndex selects the upper cell when a requested coordinate is
    # exactly on a face (including zero on the even DCI meshes), hence max on a tie.
    tied = [index for index, distance in zip(valid, distances)
            if abs(float(distance)-minimum) <= tolerance]
    return max(tied)


def assemble_fields(
    raw: dict[str, Any], plane: str, location: float, names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Assemble selected fields from a collective uniform dump or plane dump."""
    if plane not in PLANE_AXES:
        raise ValueError(f"Unsupported plane: {plane}")
    missing = sorted(set(names)-set(raw["var_names"]))
    if missing:
        raise RuntimeError(f"Laser dump is missing fields: {missing}")
    logical = np.asarray(raw["mb_logical"])
    if logical.ndim != 2 or logical.shape[1] < 4 or np.any(logical[:, 3] != 0):
        raise RuntimeError("DCI laser plotting currently requires a uniform level-0 mesh")

    normal_axis, horizontal_axis, vertical_axis, _, _ = PLANE_AXES[plane]
    target = _selected_normal_index(raw, normal_axis, location)
    counts = _axis_counts(raw)
    shape = (counts[vertical_axis], counts[horizontal_axis])
    fields = {name: np.full(shape, np.nan, dtype=np.float64) for name in names}
    coverage = np.zeros(shape, dtype=np.uint8)

    for block in range(int(raw["n_mbs"])):
        ranges = _block_ranges(raw, block)
        normal_low, normal_high = ranges[normal_axis]
        if not normal_low <= target <= normal_high:
            continue
        horizontal_low, horizontal_high = ranges[horizontal_axis]
        vertical_low, vertical_high = ranges[vertical_axis]
        expected_shape = (
            vertical_high-vertical_low+1,
            horizontal_high-horizontal_low+1,
        )
        local_normal = target-normal_low
        for name in names:
            block_values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            if plane == "xy":
                values = block_values[local_normal, :, :]
            else:
                values = block_values[:, local_normal, :]
            if values.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected {name} block plane shape {values.shape}; "
                    f"expected {expected_shape}"
                )
            fields[name][
                vertical_low:vertical_high+1,
                horizontal_low:horizontal_high+1,
            ] = values
        coverage[
            vertical_low:vertical_high+1,
            horizontal_low:horizontal_high+1,
        ] += 1

    missing_cells = int(np.count_nonzero(coverage == 0))
    duplicate_cells = int(np.count_nonzero(coverage > 1))
    if missing_cells or duplicate_cells:
        raise RuntimeError(
            "Requested plane is not a complete collective DCI output: "
            f"missing_cells={missing_cells}, duplicate_cells={duplicate_cells}. "
            "Use the single collective file, not one bin/rank_* shard."
        )
    if any(np.any(~np.isfinite(values)) for values in fields.values()):
        raise RuntimeError("Laser plane contains non-finite diagnostic values")

    horizontal_key = AXIS_KEYS[horizontal_axis]
    vertical_key = AXIS_KEYS[vertical_axis]
    horizontal_edges = np.linspace(
        float(raw[f"{horizontal_key}min"]), float(raw[f"{horizontal_key}max"]),
        counts[horizontal_axis]+1, dtype=np.float64,
    )
    vertical_edges = np.linspace(
        float(raw[f"{vertical_key}min"]), float(raw[f"{vertical_key}max"]),
        counts[vertical_axis]+1, dtype=np.float64,
    )
    actual_location = float(_axis_centers(raw, normal_axis)[target])
    return fields, horizontal_edges, vertical_edges, actual_location


def load_laser_plane(
    path: Path, plane: str = "xy", location: float = 0.0,
    quantity: str = "path",
) -> LaserPlane:
    """Read one AthenaK laser output and extract the requested DCI plane."""
    path = path.expanduser().resolve()
    raw = read_binary(str(path))
    required = (
        "laser_path", "laser_ray_count", "laser_dir1", "laser_dir2", "laser_dir3",
    )
    fields, x_edges, y_edges, actual_location = assemble_fields(
        raw, plane, location, required,
    )
    blocks = input_header(raw["header"])
    length_mm = 10.0*header_number(blocks, "units", "length_cgs")
    time_ns = float(raw["time"])*header_number(blocks, "units", "time_cgs")/1.0e-9

    path_sum = fields["laser_path"]
    if np.any(path_sum < 0.0) or np.any(fields["laser_ray_count"] < 0.0):
        raise RuntimeError("Laser path and ray-count diagnostics must be non-negative")
    if quantity == "path":
        values = path_sum*length_mm
    elif quantity == "ray-count":
        values = fields["laser_ray_count"]
    else:
        raise ValueError(f"Unsupported quantity: {quantity}")

    means = []
    for name in ("laser_dir1", "laser_dir2", "laser_dir3"):
        means.append(np.divide(
            fields[name], path_sum, out=np.zeros_like(path_sum), where=path_sum > 0.0,
        ))
    normal_axis, horizontal_axis, vertical_axis, _, _ = PLANE_AXES[plane]
    del normal_axis
    direction_x = means[horizontal_axis]
    direction_y = means[vertical_axis]
    coherence = np.sqrt(sum(component*component for component in means))

    inner = header_number(blocks, "problem", "inner_radius", DEFAULT_SHELL[0])
    outer = header_number(blocks, "problem", "outer_radius", DEFAULT_SHELL[1])
    half_angle = header_number(
        blocks, "problem", "opening_half_angle_deg", DEFAULT_SHELL[2],
    )
    return LaserPlane(
        source=path,
        plane=plane,
        quantity=quantity,
        x_edges_mm=x_edges*length_mm,
        y_edges_mm=y_edges*length_mm,
        values=values,
        direction_x=direction_x,
        direction_y=direction_y,
        direction_coherence=coherence,
        time_ns=time_ns,
        cycle=int(raw["cycle"]),
        normal_location_mm=actual_location*length_mm,
        shell_inner_mm=inner*length_mm,
        shell_outer_mm=outer*length_mm,
        shell_half_angle_deg=half_angle,
    )


def shell_cap_outline(
    inner_radius: float, outer_radius: float, half_angle_deg: float,
    normal_location: float = 0.0, samples: int = 361,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shell/cap boundary curves in a plane at fixed transverse location."""
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("Shell radii must satisfy 0 < inner < outer")
    if not (0.0 < half_angle_deg < 90.0):
        raise ValueError("Shell half-angle must lie between 0 and 90 degrees")
    alpha = math.radians(half_angle_deg)
    fixed = abs(normal_location)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for radius in (inner_radius, outer_radius):
        maximum_transverse = radius*math.sin(alpha)
        if fixed <= maximum_transverse:
            qmax = math.sqrt(max(maximum_transverse**2-fixed**2, 0.0))
            q = np.linspace(-qmax, qmax, samples, dtype=np.float64)
            x = -np.sqrt(np.maximum(radius**2-fixed**2-q*q, 0.0))
            segments.append((x, q))
    sine = math.sin(alpha)
    minimum_rim_radius = max(inner_radius, fixed/sine)
    if minimum_rim_radius <= outer_radius:
        radii = np.linspace(minimum_rim_radius, outer_radius, samples, dtype=np.float64)
        x = -radii*math.cos(alpha)
        q = np.sqrt(np.maximum((radii*sine)**2-fixed**2, 0.0))
        segments.append((x, q))
        segments.append((x, -q))
    return segments


def _normalization(values: np.ndarray, options: RenderOptions):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("Laser plane has no finite values")
    if options.norm == "linear":
        vmin = 0.0 if options.vmin is None else options.vmin
        vmax = float(np.max(finite)) if options.vmax is None else options.vmax
        if options.vmin is None and options.vmax is None and vmax == vmin:
            vmax = vmin+1.0
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            raise RuntimeError(f"Invalid linear color range: vmin={vmin}, vmax={vmax}")
        return Normalize(vmin=vmin, vmax=vmax), np.ma.masked_invalid(values)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        if options.vmin is not None or options.vmax is not None:
            raise RuntimeError(
                "Explicit logarithmic limits cannot be applied to a zero-ray frame"
            )
        # Quiet initialization/post-pulse frames are valid diagnostics.  A fixed linear
        # 0--1 placeholder keeps batch rendering continuous without inventing ray paths.
        return Normalize(vmin=0.0, vmax=1.0), np.ma.masked_invalid(values)
    vmin = float(np.min(positive)) if options.vmin is None else options.vmin
    vmax = float(np.max(positive)) if options.vmax is None else options.vmax
    if vmax == vmin:
        vmin = 0.5*vmax
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
        raise RuntimeError(f"Invalid logarithmic color range: vmin={vmin}, vmax={vmax}")
    masked = np.ma.masked_where(~np.isfinite(values) | (values <= 0.0), values)
    return LogNorm(vmin=vmin, vmax=vmax), masked


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5*(edges[:-1]+edges[1:])


def _normalized_projected_directions(
    plane: LaserPlane, minimum_coherence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = np.hypot(plane.direction_x, plane.direction_y)
    valid = (
        (plane.values > 0.0)
        & np.isfinite(projected)
        & np.isfinite(plane.direction_coherence)
        & (plane.direction_coherence >= minimum_coherence)
        & (projected >= minimum_coherence)
    )
    u = np.divide(
        plane.direction_x, projected, out=np.zeros_like(projected), where=valid,
    )
    v = np.divide(
        plane.direction_y, projected, out=np.zeros_like(projected), where=valid,
    )
    return u, v, valid


def render_laser_plane(
    plane: LaserPlane, output: Path, options: RenderOptions = RenderOptions(),
) -> Path:
    """Render one deterministic, headless PNG."""
    if not (0.0 <= options.min_coherence <= 1.0):
        raise ValueError("min_coherence must lie in [0, 1]")
    if (options.dpi <= 0 or options.stream_max_resolution <= 0
            or options.stream_density <= 0.0
            or options.quiver_spacing_mm <= 0.0
            or options.quiver_length_mm <= 0.0):
        raise ValueError("resolution, vector spacing/length, and density must be positive")
    norm, mapped = _normalization(plane.values, options)
    x = _centers(plane.x_edges_mm)
    y = _centers(plane.y_edges_mm)
    u, v, valid = _normalized_projected_directions(plane, options.min_coherence)

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "text.usetex": False,
        "axes.unicode_minus": True,
        "figure.dpi": options.dpi,
        "savefig.dpi": options.dpi,
    }):
        fig, ax = plt.subplots(figsize=(11.2, 6.8), constrained_layout=True)
        ax.set_facecolor("#111318")
        image = ax.pcolormesh(
            plane.x_edges_mm, plane.y_edges_mm, mapped,
            shading="flat", cmap=options.cmap, norm=norm, rasterized=True,
        )
        if not np.any(plane.values > 0.0):
            ax.text(
                0.5, 0.5, "No active ray segments", transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=13,
                bbox={"facecolor": "#111318", "edgecolor": "#aaaaaa", "alpha": 0.82},
                zorder=8,
            )
        if options.vectors == "quiver" and np.any(valid):
            dx = float(np.min(np.diff(plane.x_edges_mm)))
            dy = float(np.min(np.diff(plane.y_edges_mm)))
            spacing = max(options.quiver_spacing_mm, min(dx, dy))
            stride = max(1, int(math.ceil(spacing/min(dx, dy))))
            sampled = np.zeros_like(valid)
            sampled[::stride, ::stride] = True
            selected = valid & sampled
            yy, xx = np.meshgrid(y, x, indexing="ij")
            ax.quiver(
                xx[selected], yy[selected],
                options.quiver_length_mm*u[selected],
                options.quiver_length_mm*v[selected],
                color="#38d9ff", edgecolor="#07151a", linewidth=0.28,
                angles="xy", scale_units="xy", scale=1.0, width=0.0035,
                headwidth=3.8, headlength=4.8, headaxislength=4.2,
                pivot="mid", zorder=5,
            )
        elif options.vectors == "stream" and np.any(valid):
            stride = max(
                1, int(math.ceil(max(len(x), len(y))/options.stream_max_resolution)),
            )
            stream_valid = valid[::stride, ::stride]
            stream_u = np.ma.masked_where(~stream_valid, u[::stride, ::stride])
            stream_v = np.ma.masked_where(~stream_valid, v[::stride, ::stride])
            ax.streamplot(
                x[::stride], y[::stride], stream_u, stream_v,
                density=options.stream_density, color="#38d9ff",
                linewidth=0.65, arrowsize=0.85, minlength=0.04,
                integration_direction="both", zorder=5,
            )
        elif options.vectors not in ("none", "quiver", "stream"):
            raise ValueError(f"Unsupported vector overlay: {options.vectors}")

        legend_handles = []
        if options.vectors != "none":
            legend_handles.append(Line2D(
                [0], [0], color="#38d9ff", linewidth=1.6,
                label="path-weighted projected direction",
            ))
        if options.shell_outline:
            segments = shell_cap_outline(
                plane.shell_inner_mm, plane.shell_outer_mm,
                plane.shell_half_angle_deg, plane.normal_location_mm,
            )
            for index, (outline_x, outline_y) in enumerate(segments):
                ax.plot(
                    outline_x, outline_y, color="#8df0a8", linewidth=1.25,
                    linestyle="--", alpha=0.95, zorder=6,
                    label="nominal CH cap" if index == 0 else None,
                )
            if segments:
                legend_handles.append(Line2D(
                    [0], [0], color="#8df0a8", linestyle="--", linewidth=1.4,
                    label=(
                        f"CH cap {plane.shell_inner_mm:g}--{plane.shell_outer_mm:g} mm, "
                        f"{plane.shell_half_angle_deg:g} deg half-angle"
                    ),
                ))

        _, _, _, x_label, y_label = PLANE_AXES[plane.plane]
        normal_label = "x_3" if plane.plane == "xy" else "x_2"
        ax.set_xlabel(rf"${x_label}$ [mm]")
        ax.set_ylabel(rf"${y_label}$ [mm]")
        ax.set_xlim(float(plane.x_edges_mm[0]), float(plane.x_edges_mm[-1]))
        ax.set_ylim(float(plane.y_edges_mm[0]), float(plane.y_edges_mm[-1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"DCI laser-ray raster at t = {plane.time_ns:.8g} ns, cycle {plane.cycle}\n"
            rf"{plane.plane} plane at ${normal_label}={plane.normal_location_mm:+.5g}$ mm",
            fontsize=14,
        )
        if plane.quantity == "path":
            colorbar_label = r"Rasterized ray path $\sum ds$ per cell [mm]"
        else:
            colorbar_label = "Ray segments crossing cell"
        fig.colorbar(image, ax=ax, pad=0.025, label=colorbar_label)
        if legend_handles:
            ax.legend(
                handles=legend_handles, loc="upper right", framealpha=0.84,
                facecolor="#111318", edgecolor="#aaaaaa", labelcolor="white",
                fontsize=8.5,
            )
        fig.text(
            0.5, 0.006,
            "Cell-accumulated ray segments are shown; lines/arrows are mean directions, "
            "not reconstructed individual ray polylines.",
            ha="center", va="bottom", fontsize=8.5, color="#40464d",
        )
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output, facecolor="white",
            metadata={"Software": "DCI_3D/plot_laser_rays.py"},
        )
        plt.close(fig)
    return output


def _is_plane_file(path: Path, plane: str) -> bool:
    return re.search(rf"\.laser_{plane}\.\d{{5}}\.bin$", path.name) is not None


def _is_volume_file(path: Path) -> bool:
    return re.search(r"\.laser\.\d{5}\.bin$", path.name) is not None


def _directory_laser_files(directory: Path, plane: str) -> list[Path]:
    bin_dir = directory / "bin" if (directory / "bin").is_dir() else directory
    sliced = sorted(path for path in bin_dir.glob(f"*.laser_{plane}.*.bin")
                    if _is_plane_file(path, plane))
    if sliced:
        return sliced
    return sorted(path for path in bin_dir.glob("*.laser.*.bin")
                  if _is_volume_file(path))


def resolve_inputs(specifications: Iterable[str], plane: str) -> list[Path]:
    """Resolve files deterministically, preferring requested plane dumps per directory."""
    found: list[Path] = []
    for specification in specifications:
        expanded = str(Path(specification).expanduser())
        if glob.has_magic(expanded):
            matches = [Path(value) for value in sorted(glob.glob(expanded))]
            if not matches:
                raise FileNotFoundError(f"Input pattern matched no files: {specification}")
            for match in matches:
                found.extend(
                    _directory_laser_files(match, plane) if match.is_dir() else [match]
                )
            continue
        path = Path(expanded)
        if path.is_dir():
            files = _directory_laser_files(path, plane)
            if not files:
                raise FileNotFoundError(
                    f"No collective laser_{plane} or laser volume dumps below {path}"
                )
            found.extend(files)
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"Laser input does not exist: {path}")
    unique = {path.expanduser().resolve(): None for path in found}
    return sorted(unique)


def default_output_name(path: Path, plane: str, quantity: str) -> str:
    stem = path.name[:-4] if path.name.endswith(".bin") else path.name
    return f"{stem}.{plane}.{quantity}.rays.png"


def main() -> int:
    args = parse_args()
    try:
        files = resolve_inputs(args.inputs, args.plane)
        if args.latest:
            files = files[-1:]
        if args.output is not None and len(files) != 1:
            raise RuntimeError("--output requires exactly one selected input (use --latest)")
        options = RenderOptions(
            norm=args.norm,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            vectors=args.vectors,
            min_coherence=args.min_coherence,
            quiver_spacing_mm=args.quiver_spacing_mm,
            quiver_length_mm=args.quiver_length_mm,
            stream_density=args.stream_density,
            stream_max_resolution=args.stream_max_resolution,
            shell_outline=args.shell_outline,
            dpi=args.dpi,
        )
        for path in files:
            output = (
                args.output if args.output is not None
                else args.output_dir/default_output_name(path, args.plane, args.quantity)
            )
            if output.exists() and not args.overwrite:
                print(f"SKIP {output} (use --overwrite)")
                continue
            plane = load_laser_plane(path, args.plane, args.location, args.quantity)
            rendered = render_laser_plane(plane, output, options)
            visible = int(np.count_nonzero(plane.values > 0.0))
            print(
                f"{rendered}: t={plane.time_ns:.8g} ns cycle={plane.cycle} "
                f"visible_cells={visible}"
            )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Cannot plot DCI laser rays: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
