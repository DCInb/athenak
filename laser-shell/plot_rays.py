#!/usr/bin/env python3
"""Visualize configured launch rays and rasterized paths in the laser-shell run.

AthenaK's laser dumps contain cell-accumulated segment diagnostics rather than
individual ray histories.  This script therefore plots ``sum(ds)`` in the two
cells adjacent to z=0 and overlays the path-weighted mean segment direction. It
also reproduces the exact launch aperture and Gaussian ray-power weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


FRAME_DIR = CASE_DIR / "plots" / "laser_rays"
GIF_PATH = CASE_DIR / "plots" / "laser_rays.gif"
LAUNCH_PATH = CASE_DIR / "plots" / "laser_rays_launch.png"
INPUT_PATH = CASE_DIR / "laser_shell.athinput"
STATUS_PATH = CASE_DIR / "run_status.json"
DEFAULT_RUN_DIR = Path("/home/mengqi/data/athenak-2t/laser-shell/run")
DEFAULT_TIMES_NS = (2.5, 5.0)

# For fully ionized equimolar CH at 1.053 um (the case's 1-omega laser).
CRITICAL_CH_DENSITY_G_CC = 3.10067e-3
PATH_VMIN_MM = 1.0e-3
PATH_VMAX_MM = 4.0e-1
ARROW_SPACING_MM = 0.10
MIN_DIRECTION_COHERENCE = 0.20


@dataclass
class Frame:
    dump: str
    time_ns: float
    x1_mm: np.ndarray
    x2_mm: np.ndarray
    path_mm: np.ndarray
    direction_x1: np.ndarray
    direction_x2: np.ndarray
    density_g_cc: np.ndarray
    slice_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the configured launch rays and simulated ray-path diagnostics."
    )
    parser.add_argument(
        "--mode", choices=("both", "launch", "traced"), default="both",
        help="render launch geometry, traced diagnostics, or both (default: both)",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="AthenaK run directory; defaults to run_status.json, NFS output, then local run",
    )
    parser.add_argument(
        "--times", type=float, nargs="+", default=list(DEFAULT_TIMES_NS),
        metavar="NS", help="physical times to select for traced frames",
    )
    parser.add_argument(
        "--time-tolerance-ns", type=float, default=1.0e-6,
        help="maximum allowed difference between requested and stored frame time",
    )
    parser.add_argument(
        "--allow-config-mismatch", action="store_true",
        help="plot dumps whose runtime beam geometry differs from the current input",
    )
    return parser.parse_args()


def input_value(name: str) -> float:
    text = INPUT_PATH.read_text()
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    matches = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*({number})", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one numeric {name}, found {len(matches)}")
    return float(matches[0])


def input_string(name: str) -> str:
    text = INPUT_PATH.read_text()
    matches = re.findall(
        rf"(?m)^\s*{re.escape(name)}\s*=\s*([^\s#]+)", text
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}, found {len(matches)}")
    return matches[0]


def binary_metadata(path: Path) -> dict[str, object]:
    """Read the time and runtime input from an Athena binary prefix only."""
    with path.open("rb") as stream:
        code_header = stream.readline().split()
        if not code_header or code_header[0] != b"Athena":
            raise TypeError(f"{path} is not an Athena binary dump")
        if code_header[-1].split(b"=")[-1] != b"1.1":
            raise TypeError(f"Unsupported Athena binary version in {path}")
        parameter_count = int(stream.readline().split(b"=")[-1])
        parameters: dict[str, str] = {}
        for _ in range(parameter_count - 1):
            key, value = stream.readline().decode("utf-8").split("=", 1)
            parameters[key.strip()] = value.strip()
        stream.readline()  # variable count
        stream.readline()  # variable names
        header_size = int(stream.readline().split(b"=")[-1])
        header_lines = [
            line.decode("utf-8").split("#", 1)[0].strip()
            for line in stream.read(header_size).split(b"\n")
        ]

    blocks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in header_lines:
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            blocks[current][key.strip()] = value.strip()
    return {"time": float(parameters["time"]), "input_blocks": blocks}


def resolve_run_dir(requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())
    else:
        if STATUS_PATH.is_file():
            status = json.loads(STATUS_PATH.read_text())
            value = status.get("run_dir")
            if isinstance(value, str) and value:
                status_path = Path(value).expanduser()
                candidates.append(
                    status_path if status_path.is_absolute() else CASE_DIR / status_path
                )
        candidates.extend((DEFAULT_RUN_DIR, CASE_DIR / "run"))

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if any((candidate / "bin").glob("laser_shell.laser.*.bin")):
            return candidate
    raise FileNotFoundError(
        "No laser dumps found under: " + ", ".join(str(path) for path in checked)
    )


def runtime_geometry_mismatches(path: Path) -> list[str]:
    metadata = binary_metadata(path)
    blocks = metadata["input_blocks"]
    assert isinstance(blocks, dict)
    comparisons = (
        ("laser", "beam0_origin_x1"),
        ("laser", "beam0_origin_x2"),
        ("laser", "beam0_origin_x3"),
        ("laser", "beam0_direction_x1"),
        ("laser", "beam0_direction_x2"),
        ("laser", "beam0_direction_x3"),
        ("laser", "beam0_radius"),
        ("laser", "beam0_nrays"),
        ("laser", "beam0_power"),
        ("laser", "beam0_wavelength"),
        ("laser", "beam0_zeff"),
        ("laser", "beam0_start_time"),
        ("laser", "beam0_end_time"),
        ("laser", "electron_number_per_gram"),
        ("laser", "inverse_bremsstrahlung_coulomb_log"),
        ("laser", "inverse_bremsstrahlung_temperature_floor"),
        ("laser", "max_reflections_per_ray"),
        ("problem", "inner_radius"),
        ("problem", "outer_radius"),
        ("problem", "opening_half_angle_deg"),
        ("problem", "ambient_density"),
        ("problem", "solid_density"),
        ("problem", "temperature"),
        ("units", "length_cgs"),
        ("units", "mass_cgs"),
        ("units", "time_cgs"),
    )
    mismatches = []
    for block, key in comparisons:
        try:
            stored = float(blocks[block][key].split()[0])
        except (KeyError, TypeError, ValueError):
            mismatches.append(f"missing runtime <{block}>/{key}")
            continue
        configured = input_value(key)
        if not math.isclose(stored, configured, rel_tol=0.0, abs_tol=1.0e-12):
            mismatches.append(
                f"<{block}>/{key}: dump={stored:g}, input={configured:g}"
            )
    for block, key in (
        ("laser", "model"),
        ("laser", "beam0_profile"),
        ("laser", "unit_system"),
        ("laser", "absorption_model"),
        ("laser", "critical_reflection"),
        ("laser", "oblique_turning"),
    ):
        try:
            stored = blocks[block][key].split()[0]
        except (KeyError, TypeError):
            mismatches.append(f"missing runtime <{block}>/{key}")
            continue
        configured = input_string(key)
        if stored != configured:
            mismatches.append(
                f"<{block}>/{key}: dump={stored}, input={configured}"
            )
    return mismatches


def select_dumps(
    bin_dir: Path, requested_times_ns: list[float], tolerance_ns: float
) -> list[str]:
    if tolerance_ns < 0.0:
        raise ValueError("--time-tolerance-ns must be non-negative")
    time_scale_ns = input_value("time_cgs") / 1.0e-9
    candidates = []
    for path in sorted(bin_dir.glob("laser_shell.laser.*.bin")):
        match = re.search(r"\.(\d{5})\.bin$", path.name)
        if match is None:
            continue
        time_ns = float(binary_metadata(path)["time"]) * time_scale_ns
        candidates.append((match.group(1), time_ns))
    if not candidates:
        raise FileNotFoundError(f"No laser dumps in {bin_dir}")

    selected = []
    for requested_time in requested_times_ns:
        dump, stored_time = min(candidates, key=lambda item: abs(item[1]-requested_time))
        error = abs(stored_time-requested_time)
        if error > tolerance_ns:
            available = ", ".join(f"{time:g}" for _, time in candidates)
            raise RuntimeError(
                f"No dump at {requested_time:g} ns within {tolerance_ns:g} ns; "
                f"nearest is {stored_time:g} ns. Available times: {available}"
            )
        if dump not in selected:
            selected.append(dump)
    return selected


def assemble_midplane(path: Path, names: tuple[str, ...]):
    """Extract only the central x1-x2 planes from uniform MeshBlocks."""
    raw = read_binary(str(path))
    missing = set(names).difference(raw["var_names"])
    if missing:
        raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
    levels = np.asarray(raw["mb_logical"])[:, 3]
    if np.any(levels != 0):
        raise RuntimeError("plot_rays.py currently expects the case's uniform level-0 mesh")

    nx = (int(raw["Nx1"]), int(raw["Nx2"]), int(raw["Nx3"]))
    mb_nx = (
        int(raw["nx1_out_mb"]),
        int(raw["nx2_out_mb"]),
        int(raw["nx3_out_mb"]),
    )
    indices = midplane_indices(nx[2])
    fields = {name: np.zeros((nx[1], nx[0]), dtype=np.float64) for name in names}
    for block, logical in enumerate(raw["mb_logical"]):
        i0 = int(logical[0]) * mb_nx[0]
        j0 = int(logical[1]) * mb_nx[1]
        k0 = int(logical[2]) * mb_nx[2]
        local_indices = [index-k0 for index in indices if k0 <= index < k0+mb_nx[2]]
        if not local_indices:
            continue
        for name in names:
            values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            nz, ny, nx_local = values.shape
            for local_index in local_indices:
                fields[name][j0:j0+ny, i0:i0+nx_local] += (
                    values[local_index] / len(indices)
                )

    extent = (
        float(raw["x1min"]), float(raw["x1max"]),
        float(raw["x2min"]), float(raw["x2max"]),
        float(raw["x3min"]), float(raw["x3max"]),
    )
    return fields, extent, float(raw["time"]), indices, nx[2]


def midplane_indices(nz: int) -> tuple[int, ...]:
    if nz % 2:
        return (nz // 2,)
    return (nz // 2 - 1, nz // 2)


def load_frame(bin_dir: Path, dump: str) -> Frame:
    laser_path = bin_dir / f"laser_shell.laser.{dump}.bin"
    fluid_path = bin_dir / f"laser_shell.fluid.{dump}.bin"
    if not laser_path.is_file() or not fluid_path.is_file():
        raise FileNotFoundError(f"Missing paired laser/fluid dump {dump} in {bin_dir}")

    laser, extent, time, indices, nz = assemble_midplane(
        laser_path, ("laser_path", "laser_dir1", "laser_dir2")
    )
    fluid, fluid_extent, fluid_time, fluid_indices, fluid_nz = assemble_midplane(
        fluid_path, ("dens",)
    )
    if not np.allclose(extent, fluid_extent) or not np.isclose(time, fluid_time):
        raise RuntimeError(f"Laser and fluid dump {dump} do not describe the same state")
    if indices != fluid_indices or nz != fluid_nz:
        raise RuntimeError(f"Laser and fluid dump {dump} have different x3 meshes")

    ny, nx = laser["laser_path"].shape
    path = laser["laser_path"]
    dir1_integral = laser["laser_dir1"]
    dir2_integral = laser["laser_dir2"]
    density = fluid["dens"]
    if not np.any(path > 0.0):
        raise RuntimeError(f"Laser dump {dump} has no active ray paths")

    direction_x1 = np.divide(
        dir1_integral, path, out=np.zeros_like(path), where=path > 0.0
    )
    direction_x2 = np.divide(
        dir2_integral, path, out=np.zeros_like(path), where=path > 0.0
    )

    length_mm = input_value("length_cgs") * 10.0
    density_scale = input_value("mass_cgs") / input_value("length_cgs")**3
    dx1 = (extent[1] - extent[0]) / nx
    dx2 = (extent[3] - extent[2]) / ny
    dx3 = (extent[5] - extent[4]) / nz
    x1 = (extent[0] + (np.arange(nx) + 0.5) * dx1) * length_mm
    x2 = (extent[2] + (np.arange(ny) + 0.5) * dx2) * length_mm
    z_centers = (extent[4] + (np.asarray(indices) + 0.5) * dx3) * length_mm
    if len(indices) == 1:
        slice_description = f"z = {z_centers[0]:+.4f} mm"
    else:
        slice_description = (
            f"average of z = {z_centers[0]:+.4f} and {z_centers[1]:+.4f} mm"
        )

    time_ns = time * input_value("time_cgs") / 1.0e-9
    return Frame(
        dump=dump,
        time_ns=time_ns,
        x1_mm=x1,
        x2_mm=x2,
        path_mm=path * length_mm,
        direction_x1=direction_x1,
        direction_x2=direction_x2,
        density_g_cc=density * density_scale,
        slice_description=slice_description,
    )


def configured_launch_rays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce BuildInitialRays for the configured three-dimensional beam."""
    nrays = int(input_value("beam0_nrays"))
    radius = input_value("beam0_radius")
    origin = np.asarray(
        [input_value(f"beam0_origin_x{axis}") for axis in (1, 2, 3)],
        dtype=float,
    )
    direction = np.asarray(
        [input_value(f"beam0_direction_x{axis}") for axis in (1, 2, 3)],
        dtype=float,
    )
    direction /= np.linalg.norm(direction)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(direction[2]) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    basis_u = np.cross(direction, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(direction, basis_u)

    ray_index = np.arange(nrays, dtype=float)
    sample_radius = radius*np.sqrt((ray_index+0.5)/nrays)
    angle = math.pi*(3.0-math.sqrt(5.0))*ray_index
    offsets_u = sample_radius*np.cos(angle)
    offsets_v = sample_radius*np.sin(angle)
    positions = (
        origin[np.newaxis, :]
        + offsets_u[:, np.newaxis]*basis_u[np.newaxis, :]
        + offsets_v[:, np.newaxis]*basis_v[np.newaxis, :]
    )
    if input_string("beam0_profile") == "gaussian" and radius > 0.0:
        weights = np.exp(-2.0*(sample_radius/radius)**2)
    else:
        weights = np.ones(nrays)
    powers = input_value("beam0_power")*weights/np.sum(weights)
    return positions, direction, powers


def render_launch_geometry() -> Path:
    """Plot exact launch positions and representative paths to the nominal shell."""
    positions, direction, powers = configured_launch_rays()
    inner_radius = input_value("inner_radius")
    outer_radius = input_value("outer_radius")
    half_angle = math.radians(input_value("opening_half_angle_deg"))
    aperture_radius = input_value("beam0_radius")
    origin_x1 = input_value("beam0_origin_x1")
    length_mm = input_value("length_cgs")*10.0

    if not np.allclose(direction, (-1.0, 0.0, 0.0), atol=1.0e-12):
        raise RuntimeError("Launch-geometry renderer currently expects the case's -x1 beam")
    transverse_radius2 = positions[:, 1]**2 + positions[:, 2]**2
    if np.any(transverse_radius2 >= inner_radius**2):
        raise RuntimeError("A configured ray misses the nominal inner sphere")
    endpoints = positions.copy()
    endpoints[:, 0] = -np.sqrt(inner_radius**2-transverse_radius2)

    fig = plt.figure(figsize=(16.0, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0))
    ax = fig.add_subplot(grid[0], projection="3d")
    aperture_ax = fig.add_subplot(grid[1])

    theta = np.linspace(0.0, half_angle, 28)
    phi = np.linspace(0.0, 2.0*math.pi, 96)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    for radius, alpha in ((outer_radius, 0.30), (inner_radius, 0.18)):
        shell_x1 = -radius*np.cos(theta_mesh)
        shell_x2 = radius*np.sin(theta_mesh)*np.cos(phi_mesh)
        shell_x3 = radius*np.sin(theta_mesh)*np.sin(phi_mesh)
        ax.plot_surface(
            shell_x1*length_mm, shell_x2*length_mm, shell_x3*length_mm,
            color="#7b8791", edgecolor="none", alpha=alpha, shade=True,
            rcount=32, ccount=64,
        )
    rim_radius, rim_phi = np.meshgrid(
        np.linspace(inner_radius, outer_radius, 8), phi
    )
    ax.plot_surface(
        -rim_radius*np.cos(half_angle)*length_mm,
        rim_radius*np.sin(half_angle)*np.cos(rim_phi)*length_mm,
        rim_radius*np.sin(half_angle)*np.sin(rim_phi)*length_mm,
        color="#63717b", edgecolor="none", alpha=0.45, shade=True,
    )

    ray_count = min(640, len(positions))
    selected = np.linspace(0, len(positions)-1, ray_count, dtype=int)
    segments = np.stack((positions[selected], endpoints[selected]), axis=1)*length_mm
    relative_power = powers/np.max(powers)
    cmap = plt.get_cmap("inferno")
    colors = cmap(relative_power[selected])
    colors[:, 3] = 0.22 + 0.68*relative_power[selected]
    collection = Line3DCollection(segments, colors=colors, linewidths=0.75)
    ax.add_collection3d(collection)

    ring = np.linspace(0.0, 2.0*math.pi, 240)
    ax.plot(
        np.full_like(ring, origin_x1)*length_mm,
        aperture_radius*np.cos(ring)*length_mm,
        aperture_radius*np.sin(ring)*length_mm,
        color="#d04a35", linewidth=1.8,
    )
    ax.quiver(
        origin_x1*length_mm, 0.0, 0.0, -0.42*length_mm, 0.0, 0.0,
        color="#d04a35", linewidth=2.2, arrow_length_ratio=0.18,
    )
    ax.set_xlim(-1.08*length_mm, 1.58*length_mm)
    ax.set_ylim(-0.72*length_mm, 0.72*length_mm)
    ax.set_zlim(-0.72*length_mm, 0.72*length_mm)
    ax.set_box_aspect((2.66, 1.44, 1.44))
    ax.view_init(elev=21, azim=-57)
    ax.set_xlabel(r"$x_1$ [mm]", labelpad=9)
    ax.set_ylabel(r"$x_2$ [mm]", labelpad=9)
    ax.set_zlabel(r"$x_3$ [mm]", labelpad=9)
    ax.set_title("Parallel rays from the right to the open shell", pad=16, fontsize=14)
    ax.legend(
        handles=(
            Line2D((0,), (0,), color="#d04a35", linewidth=2.0,
                   label="source aperture and propagation direction"),
            Line2D((0,), (0,), color=cmap(0.72), linewidth=2.0,
                   label=f"{ray_count:,} representative rays"),
            Patch(facecolor="#7b8791", alpha=0.45, label="nominal CH shell"),
        ),
        loc="upper left", framealpha=0.94, fontsize=9,
    )

    transverse_x2 = positions[:, 1]*length_mm
    transverse_x3 = positions[:, 2]*length_mm
    scatter = aperture_ax.scatter(
        transverse_x2, transverse_x3, c=relative_power, cmap=cmap,
        vmin=0.0, vmax=1.0, s=2.2, linewidths=0.0, rasterized=True,
    )
    opening_radius = inner_radius*math.sin(half_angle)*length_mm
    aperture_circle = plt.Circle(
        (0.0, 0.0), aperture_radius*length_mm, fill=False,
        color="#d04a35", linewidth=1.6,
    )
    opening_circle = plt.Circle(
        (0.0, 0.0), opening_radius, fill=False,
        color="#38434a", linewidth=1.4, linestyle="--",
    )
    aperture_ax.add_patch(opening_circle)
    aperture_ax.add_patch(aperture_circle)
    aperture_ax.set_aspect("equal")
    limit = 1.12*opening_radius
    aperture_ax.set_xlim(-limit, limit)
    aperture_ax.set_ylim(-limit, limit)
    aperture_ax.set_xlabel(r"$x_2$ [mm]")
    aperture_ax.set_ylabel(r"$x_3$ [mm]")
    aperture_ax.set_title("Equal-area aperture samples with Gaussian power", fontsize=14)
    aperture_ax.grid(color="#d8dde1", linewidth=0.6, alpha=0.7)
    aperture_ax.legend(
        handles=(
            Line2D((0,), (0,), color="#d04a35", label="0.320 mm beam aperture"),
            Line2D((0,), (0,), color="#38434a", linestyle="--",
                   label=f"{opening_radius:.3f} mm projected inner opening"),
        ),
        loc="upper right", framealpha=0.94, fontsize=9,
    )
    colorbar = fig.colorbar(scatter, ax=aperture_ax, pad=0.03, shrink=0.88)
    colorbar.set_label(r"Relative ray power $P/P_{\rm max}$")

    pulse_power_tw = input_value("beam0_power")*1.0e-19
    pulse_duration_ns = input_value("beam0_end_time")-input_value("beam0_start_time")
    fig.suptitle(
        f"Configured laser rays: {len(positions):,} rays, {pulse_power_tw:g} TW, "
        f"{pulse_duration_ns:g} ns square pulse",
        fontsize=17,
    )
    fig.text(
        0.5, 0.008,
        "Lines show configured straight launch paths to first contact with the nominal inner surface; "
        "plasma reflection and absorption require a traced simulation frame.",
        ha="center", va="bottom", fontsize=9.5, color="#3f474d",
    )
    LAUNCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LAUNCH_PATH, dpi=180, facecolor="white")
    plt.close(fig)
    return LAUNCH_PATH


def render(frame: Frame) -> Path:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    time_tag = f"{frame.time_ns:010.6f}".replace(".", "p")
    output = FRAME_DIR / f"laser_shell.laser.t{time_tag}ns.rays.png"

    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    ax.set_facecolor("#111318")
    masked_path = np.ma.masked_less(frame.path_mm, PATH_VMIN_MM)
    image = ax.pcolormesh(
        frame.x1_mm,
        frame.x2_mm,
        masked_path,
        shading="nearest",
        cmap="magma",
        norm=LogNorm(vmin=PATH_VMIN_MM, vmax=PATH_VMAX_MM),
        rasterized=True,
    )

    contour_levels = (CRITICAL_CH_DENSITY_G_CC, 1.0e-2)
    contour_colors = ("#f2f2f2", "#66ff99")
    ax.contour(
        frame.x1_mm,
        frame.x2_mm,
        frame.density_g_cc,
        levels=contour_levels,
        colors=contour_colors,
        linewidths=(1.25, 1.15),
        linestyles=("--", "-"),
        alpha=0.95,
    )

    coherence = np.hypot(frame.direction_x1, frame.direction_x2)
    arrow_mask = (
        (frame.path_mm >= PATH_VMIN_MM)
        & (coherence >= MIN_DIRECTION_COHERENCE)
    )
    arrow_x1 = np.divide(
        frame.direction_x1,
        coherence,
        out=np.zeros_like(coherence),
        where=arrow_mask,
    )
    arrow_x2 = np.divide(
        frame.direction_x2,
        coherence,
        out=np.zeros_like(coherence),
        where=arrow_mask,
    )
    dx1_mm = float(np.min(np.diff(frame.x1_mm)))
    dx2_mm = float(np.min(np.diff(frame.x2_mm)))
    arrow_stride = max(
        1, math.ceil(ARROW_SPACING_MM/min(abs(dx1_mm), abs(dx2_mm)))
    )
    sampled = np.zeros_like(arrow_mask)
    sampled[::arrow_stride, ::arrow_stride] = True
    arrow_mask &= sampled
    yy, xx = np.meshgrid(frame.x2_mm, frame.x1_mm, indexing="ij")
    quiver = ax.quiver(
        xx[arrow_mask],
        yy[arrow_mask],
        arrow_x1[arrow_mask],
        arrow_x2[arrow_mask],
        color="#38d9ff",
        edgecolor="#07151a",
        linewidth=0.30,
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.0040,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.2,
        pivot="mid",
        zorder=5,
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, extend="min")
    colorbar.set_label(r"Rasterized ray path $\sum ds$ per cell [mm]", fontsize=12)
    ax.set_xlabel(r"$x_1$ [mm]", fontsize=13)
    ax.set_ylabel(r"$x_2$ [mm]", fontsize=13)
    ax.set_xlim(frame.x1_mm[0], frame.x1_mm[-1])
    ax.set_ylim(frame.x2_mm[0], frame.x2_mm[-1])
    ax.set_aspect("equal")
    ax.set_title(
        f"Laser-ray paths at t = {frame.time_ns:.6g} ns\n"
        f"near-midplane ({frame.slice_description})",
        fontsize=15,
        pad=12,
    )
    legend = [
        Line2D(
            [0], [0], color="#38d9ff", marker=r"$\rightarrow$", markersize=18,
            linestyle="None", label="path-weighted mean direction",
        ),
        Line2D(
            [0], [0], color=contour_colors[0], linestyle="--", linewidth=1.5,
            label=rf"CH critical density ({CRITICAL_CH_DENSITY_G_CC:.4f} g cm$^{{-3}}$)",
        ),
        Line2D(
            [0], [0], color=contour_colors[1], linestyle="-", linewidth=1.5,
            label=r"CH density = 0.010 g cm$^{-3}$",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        framealpha=0.85,
        facecolor="#111318",
        edgecolor="#aaaaaa",
        labelcolor="white",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.006,
        "Ray-segment occupancy is unweighted by power; the dump does not store individual polylines or Gaussian intensity. "
        "Arrows with directional coherence < 0.20 are omitted.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def write_gif(frames: list[Path]) -> None:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(frame).convert("RGB") for frame in frames]
    try:
        images[0].save(
            GIF_PATH,
            save_all=True,
            append_images=images[1:],
            duration=1100,
            loop=0,
            optimize=True,
        )
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = parse_args()
    if args.mode in ("both", "launch"):
        launch_path = render_launch_geometry()
        print(launch_path)

    if args.mode in ("both", "traced"):
        try:
            run_dir = resolve_run_dir(args.run_dir)
            bin_dir = run_dir / "bin"
            laser_paths = sorted(bin_dir.glob("laser_shell.laser.*.bin"))
            mismatches = runtime_geometry_mismatches(laser_paths[0])
            if mismatches and not args.allow_config_mismatch:
                details = "; ".join(mismatches)
                raise RuntimeError(
                    f"Run geometry differs from {INPUT_PATH}: {details}. "
                    "Use matching outputs or pass --allow-config-mismatch."
                )
            dumps = select_dumps(bin_dir, args.times, args.time_tolerance_ns)
            frames = [load_frame(bin_dir, dump) for dump in dumps]
            rendered = [render(frame) for frame in frames]
            write_gif(rendered)
            for frame, path in zip(frames, rendered):
                occupied = int(np.count_nonzero(frame.path_mm >= PATH_VMIN_MM))
                print(
                    f"{path}: t={frame.time_ns:.6g} ns, "
                    f"{occupied} visible midplane cells"
                )
            print(GIF_PATH)
        except (FileNotFoundError, RuntimeError) as error:
            if args.mode == "traced":
                raise
            print(f"Skipping traced frames: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
