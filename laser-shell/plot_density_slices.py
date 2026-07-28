#!/usr/bin/env python3
"""Render fixed-scale physical-density midplane slices from laser-shell dumps.

The production fluid dumps are several gigabytes each.  This reader seeks directly
to the two cell planes adjacent to x3=0 instead of loading whole MeshBlocks, averages
those planes, and then renders every frame with one shared logarithmic color scale.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


CASE_DIR = Path(__file__).resolve().parent
STATUS_PATH = CASE_DIR / "run_status.json"
DEFAULT_RUN_DIR = Path("/home/mengqi/data/athenak-2t/laser-shell/run")
DEFAULT_OUTPUT_DIR = CASE_DIR / "plots" / "density_slices_xy_log"
DEFAULT_GIF_PATH = CASE_DIR / "plots" / "density_slices_xy_log.gif"
DEFAULT_OVERLAY_OUTPUT_DIR = CASE_DIR / "plots" / "density_laser_overlay_xy_log"
DEFAULT_OVERLAY_GIF_PATH = CASE_DIR / "plots" / "density_laser_overlay_xy_log.gif"
PATH_VMIN_MM = 1.0e-3
PATH_VMAX_MM = 4.0e-1
ARROW_SPACING_MM = 0.10
MIN_DIRECTION_COHERENCE = 0.20
RAY_COLOR = "#26e6ff"


@dataclass(frozen=True)
class Frame:
    path: Path
    time_ns: float
    density_g_cc: np.ndarray
    extent_mm: tuple[float, float, float, float]
    slice_description: str
    rays: RayOverlay | None = None


@dataclass(frozen=True)
class RayOverlay:
    path_mm: np.ndarray
    direction_x1: np.ndarray
    direction_x2: np.ndarray


@dataclass(frozen=True)
class DumpPrefix:
    data_start: int
    data_bytes: int
    time_ns: float
    location_size: int
    variable_size: int
    num_variables: int
    variable_names: tuple[str, ...]
    shape: tuple[int, int]
    nx3: int
    central_indices: tuple[int, int]
    density_scale: float
    length_scale_mm: float
    extent_mm: tuple[float, float, float, float]
    slice_description: str


@dataclass(frozen=True)
class PlaneSample:
    relative_offset: int
    i0: int
    j0: int
    nx: int
    ny: int
    nz: int
    block_index: int
    local_k: int


@dataclass(frozen=True)
class SliceLayout:
    data_bytes: int
    signature: tuple[int, ...]
    samples: tuple[PlaneSample, ...]
    num_blocks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fixed-scale x1-x2 density slices from laser-shell fluid dumps."
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="run directory; defaults to run_status.json, production data, then local run",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="PNG output directory; defaults depend on whether rays are overlaid",
    )
    parser.add_argument(
        "--gif", type=Path,
        help="GIF output path; defaults depend on whether rays are overlaid",
    )
    parser.add_argument(
        "--max-index", type=int,
        help="highest five-digit dump index to include",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="concurrent sparse readers (default: 4)",
    )
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap")
    parser.add_argument(
        "--overlay-rays", action="store_true",
        help="overlay rasterized laser paths and path-weighted mean directions",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution")
    parser.add_argument(
        "--duration-ms", type=int, default=120,
        help="GIF duration per frame in milliseconds",
    )
    return parser.parse_args()


def resolve_run_dir(requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())
    else:
        if STATUS_PATH.is_file():
            status = json.loads(STATUS_PATH.read_text())
            value = status.get("run_dir")
            if isinstance(value, str) and value:
                path = Path(value).expanduser()
                candidates.append(path if path.is_absolute() else CASE_DIR / path)
        candidates.extend((DEFAULT_RUN_DIR, CASE_DIR / "run"))

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if any((candidate / "bin").glob("laser_shell.fluid.*.bin")):
            return candidate
    raise FileNotFoundError(
        "No fluid dumps found under: " + ", ".join(str(path) for path in checked)
    )


def dump_index(path: Path) -> int:
    match = re.search(r"\.(\d{5})\.bin$", path.name)
    if match is None:
        raise RuntimeError(f"Cannot parse dump index from {path}")
    return int(match.group(1))


def parse_runtime_header(text: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            blocks[current][key.strip()] = value.strip()
    return blocks


def header_value(
    blocks: dict[str, dict[str, str]], block: str, key: str
) -> str:
    try:
        return blocks[block][key].split()[0]
    except KeyError as exc:
        raise RuntimeError(f"Missing runtime <{block}>/{key}") from exc


def read_dump_prefix(stream, path: Path) -> DumpPrefix:
    file_size = path.stat().st_size
    if stream.readline() != b"Athena binary output version=1.1\n":
        raise RuntimeError(f"Unrecognized Athena binary dump: {path}")
    preheader_size = int(stream.readline().split(b"=")[-1])
    if preheader_size != 5:
        raise RuntimeError(f"Unsupported preheader size {preheader_size} in {path}")
    time = float(stream.readline().split(b"=")[-1])
    stream.readline()  # cycle
    location_size = int(stream.readline().split(b"=")[-1])
    variable_size = int(stream.readline().split(b"=")[-1])
    num_variables = int(stream.readline().split(b"=")[-1])
    variable_names = [item.decode("ascii") for item in stream.readline().split()[1:]]
    header_offset = int(stream.readline().split(b"=")[-1])

    if location_size not in (4, 8) or variable_size not in (4, 8):
        raise RuntimeError(f"Unsupported scalar sizes in {path}")
    if len(variable_names) != num_variables:
        raise RuntimeError(f"Variable-count mismatch in {path}")
    runtime_text = stream.read(header_offset).decode("ascii")
    blocks = parse_runtime_header(runtime_text)
    nx1 = int(header_value(blocks, "mesh", "nx1"))
    nx2 = int(header_value(blocks, "mesh", "nx2"))
    nx3 = int(header_value(blocks, "mesh", "nx3"))
    x1min = float(header_value(blocks, "mesh", "x1min"))
    x1max = float(header_value(blocks, "mesh", "x1max"))
    x2min = float(header_value(blocks, "mesh", "x2min"))
    x2max = float(header_value(blocks, "mesh", "x2max"))
    x3min = float(header_value(blocks, "mesh", "x3min"))
    x3max = float(header_value(blocks, "mesh", "x3max"))
    length_cgs = float(header_value(blocks, "units", "length_cgs"))
    mass_cgs = float(header_value(blocks, "units", "mass_cgs"))
    time_cgs = float(header_value(blocks, "units", "time_cgs"))
    if nx3 < 2 or nx3 % 2:
        raise RuntimeError("Expected the case's even three-dimensional x3 mesh")

    central_indices = (nx3 // 2 - 1, nx3 // 2)
    length_scale_mm = length_cgs * 10.0
    dz_mm = (x3max - x3min) / nx3 * length_scale_mm
    z0_mm = (x3min + (central_indices[0] + 0.5) * (x3max - x3min) / nx3) * length_scale_mm
    z1_mm = (x3min + (central_indices[1] + 0.5) * (x3max - x3min) / nx3) * length_scale_mm
    if not math.isclose(z0_mm, -z1_mm, rel_tol=0.0, abs_tol=1.0e-12 * max(1.0, dz_mm)):
        raise RuntimeError("The two central x3 planes do not bracket zero symmetrically")
    return DumpPrefix(
        data_start=stream.tell(),
        data_bytes=file_size - stream.tell(),
        time_ns=time * time_cgs / 1.0e-9,
        location_size=location_size,
        variable_size=variable_size,
        num_variables=num_variables,
        variable_names=tuple(variable_names),
        shape=(nx2, nx1),
        nx3=nx3,
        central_indices=central_indices,
        density_scale=mass_cgs / length_cgs**3,
        length_scale_mm=length_scale_mm,
        extent_mm=(
            x1min * length_scale_mm,
            x1max * length_scale_mm,
            x2min * length_scale_mm,
            x2max * length_scale_mm,
        ),
        slice_description=f"average of x3 = {z0_mm:+.4f} and {z1_mm:+.4f} mm",
    )


def layout_signature(prefix: DumpPrefix) -> tuple[int, ...]:
    return (
        prefix.location_size,
        prefix.variable_size,
        prefix.num_variables,
        prefix.shape[0],
        prefix.shape[1],
        prefix.nx3,
        *prefix.central_indices,
    )


def field_index(prefix: DumpPrefix, name: str) -> int:
    try:
        return prefix.variable_names.index(name)
    except ValueError as exc:
        available = ", ".join(prefix.variable_names)
        raise RuntimeError(f'Variable "{name}" not found; options are {{{available}}}') from exc


def build_slice_layout(path: Path) -> SliceLayout:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = read_dump_prefix(stream, path)
        density_index = field_index(prefix, "dens")
        blocks = parse_runtime_header_from_path(path)
        nghost = int(header_value(blocks, "mesh", "nghost"))
        location_dtype = np.dtype("<f4" if prefix.location_size == 4 else "<f8")
        coverage = np.zeros(prefix.shape, dtype=np.uint8)
        samples: list[PlaneSample] = []
        block_index = 0
        while stream.tell() < file_size:
            block_header = stream.read(24)
            if not block_header:
                break
            if len(block_header) != 24:
                raise RuntimeError(f"Truncated MeshBlock header in {path}")
            indices = np.frombuffer(block_header, dtype="<i4").astype(np.int64) - nghost
            logical_raw = stream.read(16)
            geometry_raw = stream.read(6 * prefix.location_size)
            if len(logical_raw) != 16 or len(geometry_raw) != 6 * prefix.location_size:
                raise RuntimeError(f"Truncated MeshBlock metadata in {path}")
            logical = np.frombuffer(logical_raw, dtype="<i4")
            # Parse geometry to validate the expected scalar format, even though the
            # uniform-grid global extent comes from the runtime header.
            np.frombuffer(geometry_raw, dtype=location_dtype)
            if int(logical[3]) != 0:
                raise RuntimeError("Sparse density renderer expects the uniform level-0 mesh")

            block_nx1 = int(indices[1] - indices[0] + 1)
            block_nx2 = int(indices[3] - indices[2] + 1)
            block_nx3 = int(indices[5] - indices[4] + 1)
            if min(block_nx1, block_nx2, block_nx3) <= 0:
                raise RuntimeError(f"Invalid MeshBlock dimensions in {path}")
            cells_per_block = block_nx1 * block_nx2 * block_nx3
            variable_bytes = cells_per_block * prefix.variable_size
            cell_data_start = stream.tell()
            i0 = int(logical[0]) * block_nx1
            j0 = int(logical[1]) * block_nx2
            k0 = int(logical[2]) * block_nx3

            for global_k in prefix.central_indices:
                if not (k0 <= global_k < k0 + block_nx3):
                    continue
                local_k = global_k - k0
                plane_offset = (
                    cell_data_start
                    + density_index * variable_bytes
                    + local_k * block_nx1 * block_nx2 * prefix.variable_size
                )
                samples.append(
                    PlaneSample(
                        relative_offset=plane_offset - prefix.data_start,
                        i0=i0,
                        j0=j0,
                        nx=block_nx1,
                        ny=block_nx2,
                        nz=block_nx3,
                        block_index=block_index,
                        local_k=local_k,
                    )
                )
                coverage[j0:j0 + block_nx2, i0:i0 + block_nx1] += 1

            stream.seek(cell_data_start + prefix.num_variables * variable_bytes)
            block_index += 1

        if stream.tell() != file_size:
            raise RuntimeError(f"Binary record alignment failed in {path}")
        if not np.all(coverage == len(prefix.central_indices)):
            raise RuntimeError(f"Incomplete or overlapping midplane coverage in {path}")
    return SliceLayout(
        data_bytes=prefix.data_bytes,
        signature=layout_signature(prefix) + (density_index,),
        samples=tuple(samples),
        num_blocks=block_index,
    )


def parse_runtime_header_from_path(path: Path) -> dict[str, dict[str, str]]:
    with path.open("rb") as stream:
        for _ in range(8):
            stream.readline()
        header_offset = int(stream.readline().split(b"=")[-1])
        return parse_runtime_header(stream.read(header_offset).decode("ascii"))


def read_midplane_density(path: Path, layout: SliceLayout | None = None) -> Frame:
    if layout is None:
        layout = build_slice_layout(path)
    with path.open("rb") as stream:
        prefix = read_dump_prefix(stream, path)
        signature = layout_signature(prefix) + (field_index(prefix, "dens"),)
        if prefix.data_bytes != layout.data_bytes or signature != layout.signature:
            raise RuntimeError(f"MeshBlock layout differs from the reference dump: {path}")
        variable_dtype = np.dtype("<f4" if prefix.variable_size == 4 else "<f8")
        density = np.zeros(prefix.shape, dtype=np.float64)
        for sample in layout.samples:
            stream.seek(prefix.data_start + sample.relative_offset)
            raw = stream.read(sample.nx * sample.ny * prefix.variable_size)
            if len(raw) != sample.nx * sample.ny * prefix.variable_size:
                raise RuntimeError(f"Truncated density plane in {path}")
            plane = np.frombuffer(raw, dtype=variable_dtype).reshape(sample.ny, sample.nx)
            density[
                sample.j0:sample.j0 + sample.ny,
                sample.i0:sample.i0 + sample.nx,
            ] += plane.astype(np.float64, copy=False) / len(prefix.central_indices)

    return Frame(
        path=path,
        time_ns=prefix.time_ns,
        density_g_cc=density * prefix.density_scale,
        extent_mm=prefix.extent_mm,
        slice_description=prefix.slice_description,
    )


def paired_record_size(prefix: DumpPrefix, layout: SliceLayout) -> int:
    if prefix.data_bytes % layout.num_blocks:
        raise RuntimeError("Laser dump data do not contain the reference MeshBlock count")
    record_size = prefix.data_bytes // layout.num_blocks
    sample = layout.samples[0]
    cells_per_block = sample.nx * sample.ny * sample.nz
    expected = (
        24 + 16 + 6 * prefix.location_size
        + prefix.num_variables * cells_per_block * prefix.variable_size
    )
    if record_size != expected:
        raise RuntimeError(
            f"Laser MeshBlock record size {record_size} does not match expected {expected}"
        )
    return record_size


def validate_paired_layout(path: Path, layout: SliceLayout) -> None:
    with path.open("rb") as stream:
        prefix = read_dump_prefix(stream, path)
        record_size = paired_record_size(prefix, layout)
        checked_blocks: set[int] = set()
        for sample in layout.samples:
            if sample.block_index in checked_blocks:
                continue
            checked_blocks.add(sample.block_index)
            logical_offset = prefix.data_start + sample.block_index * record_size + 24
            stream.seek(logical_offset)
            raw = stream.read(16)
            if len(raw) != 16:
                raise RuntimeError(f"Truncated laser MeshBlock metadata in {path}")
            logical = np.frombuffer(raw, dtype="<i4")
            global_k = int(logical[2]) * sample.nz + sample.local_k
            if (
                int(logical[0]) * sample.nx != sample.i0
                or int(logical[1]) * sample.ny != sample.j0
                or global_k not in prefix.central_indices
                or int(logical[3]) != 0
            ):
                raise RuntimeError(f"Paired/fluid MeshBlock ordering differs in {path}")


def read_midplane_fields(
    path: Path,
    layout: SliceLayout,
    names: tuple[str, ...],
) -> tuple[DumpPrefix, tuple[np.ndarray, ...]]:
    with path.open("rb") as stream:
        prefix = read_dump_prefix(stream, path)
        if (
            prefix.location_size != layout.signature[0]
            or prefix.variable_size != layout.signature[1]
            or prefix.shape != (layout.signature[3], layout.signature[4])
            or prefix.nx3 != layout.signature[5]
            or prefix.central_indices != (layout.signature[6], layout.signature[7])
        ):
            raise RuntimeError(f"Paired dump is not co-spatial with density layout: {path}")
        record_size = paired_record_size(prefix, layout)
        indices = tuple(field_index(prefix, name) for name in names)
        variable_dtype = np.dtype("<f4" if prefix.variable_size == 4 else "<f8")
        fields = tuple(np.zeros(prefix.shape, dtype=np.float64) for _ in indices)

        for sample in layout.samples:
            cells_per_block = sample.nx * sample.ny * sample.nz
            variable_bytes = cells_per_block * prefix.variable_size
            record_start = prefix.data_start + sample.block_index * record_size
            cell_data_start = record_start + 24 + 16 + 6 * prefix.location_size
            plane_cells = sample.nx * sample.ny
            for field, variable_index in zip(fields, indices):
                offset = (
                    cell_data_start
                    + variable_index * variable_bytes
                    + sample.local_k * plane_cells * prefix.variable_size
                )
                stream.seek(offset)
                raw = stream.read(plane_cells * prefix.variable_size)
                if len(raw) != plane_cells * prefix.variable_size:
                    raise RuntimeError(f"Truncated field plane in {path}")
                plane = np.frombuffer(raw, dtype=variable_dtype).reshape(sample.ny, sample.nx)
                field[
                    sample.j0:sample.j0 + sample.ny,
                    sample.i0:sample.i0 + sample.nx,
                ] += plane.astype(np.float64, copy=False) / len(prefix.central_indices)
    return prefix, fields


def read_midplane_rays(path: Path, layout: SliceLayout) -> tuple[float, RayOverlay]:
    prefix, fields = read_midplane_fields(
        path, layout, ("laser_path", "laser_dir1", "laser_dir2")
    )
    path_code, direction1_integral, direction2_integral = fields
    direction_x1 = np.divide(
        direction1_integral,
        path_code,
        out=np.zeros_like(path_code),
        where=path_code > 0.0,
    )
    direction_x2 = np.divide(
        direction2_integral,
        path_code,
        out=np.zeros_like(path_code),
        where=path_code > 0.0,
    )
    return (
        prefix.time_ns,
        RayOverlay(
            path_mm=path_code * prefix.length_scale_mm,
            direction_x1=direction_x1,
            direction_x2=direction_x2,
        ),
    )


def render_frame(
    frame: Frame,
    output_dir: Path,
    norm: LogNorm,
    cmap: str,
    dpi: int,
) -> Path:
    tag = "dens_rays" if frame.rays is not None else "dens"
    output = output_dir / f"{frame.path.stem}.{tag}.png"
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    image = ax.imshow(
        frame.density_g_cc,
        origin="lower",
        extent=frame.extent_mm,
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
        rasterized=True,
    )
    if frame.rays is not None:
        render_ray_overlay(ax, frame)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label(r"Mass density $\rho$ [g cm$^{-3}$]", fontsize=12)
    ax.set_xlabel(r"$x_1$ [mm]", fontsize=13)
    ax.set_ylabel(r"$x_2$ [mm]", fontsize=13)
    title_quantity = "CH mass density with laser-ray paths" if frame.rays is not None else "CH mass density"
    ax.set_title(
        f"{title_quantity} at t = {frame.time_ns:.6g} ns\n"
        f"near-midplane ({frame.slice_description})",
        fontsize=15,
        pad=12,
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output


def render_ray_overlay(ax, frame: Frame) -> None:
    assert frame.rays is not None
    path = frame.rays.path_mm
    visible = np.isfinite(path) & (path >= PATH_VMIN_MM)
    if not np.any(visible):
        return

    log_strength = np.zeros_like(path)
    log_strength[visible] = np.clip(
        (
            np.log10(path[visible]) - math.log10(PATH_VMIN_MM)
        ) / (
            math.log10(PATH_VMAX_MM) - math.log10(PATH_VMIN_MM)
        ),
        0.0,
        1.0,
    )
    rgba = np.zeros(path.shape + (4,), dtype=np.float64)
    rgba[..., :3] = to_rgb(RAY_COLOR)
    rgba[..., 3][visible] = 0.45 + 0.50 * log_strength[visible]
    ax.imshow(
        rgba,
        origin="lower",
        extent=frame.extent_mm,
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
        zorder=3,
    )

    coherence = np.hypot(frame.rays.direction_x1, frame.rays.direction_x2)
    arrow_mask = visible & (coherence >= MIN_DIRECTION_COHERENCE)
    arrow_x1 = np.divide(
        frame.rays.direction_x1,
        coherence,
        out=np.zeros_like(coherence),
        where=arrow_mask,
    )
    arrow_x2 = np.divide(
        frame.rays.direction_x2,
        coherence,
        out=np.zeros_like(coherence),
        where=arrow_mask,
    )
    ny, nx = path.shape
    x1min, x1max, x2min, x2max = frame.extent_mm
    dx1 = (x1max - x1min) / nx
    dx2 = (x2max - x2min) / ny
    x1 = x1min + (np.arange(nx) + 0.5) * dx1
    x2 = x2min + (np.arange(ny) + 0.5) * dx2
    stride = max(1, math.ceil(ARROW_SPACING_MM / min(abs(dx1), abs(dx2))))
    sampled = np.zeros_like(arrow_mask)
    sampled[::stride, ::stride] = True
    arrow_mask &= sampled
    yy, xx = np.meshgrid(x2, x1, indexing="ij")
    ax.quiver(
        xx[arrow_mask],
        yy[arrow_mask],
        arrow_x1[arrow_mask],
        arrow_x2[arrow_mask],
        color=RAY_COLOR,
        edgecolor="#06171c",
        linewidth=0.35,
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
    ax.legend(
        handles=(
            Line2D(
                [0], [0], color=RAY_COLOR, linewidth=3.0,
                label=r"rasterized ray path $\sum ds$",
            ),
            Line2D(
                [0], [0], color=RAY_COLOR, marker=r"$\rightarrow$", markersize=17,
                linestyle="None", label="path-weighted mean direction",
            ),
        ),
        loc="upper right",
        framealpha=0.88,
        facecolor="#111318",
        edgecolor="#aaaaaa",
        labelcolor="white",
        fontsize=9,
    )


def write_gif(paths: list[Path], output: Path, duration_ms: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths]
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.dpi < 1 or args.duration_ms < 1:
        raise ValueError("--dpi and --duration-ms must be positive")
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_OVERLAY_OUTPUT_DIR if args.overlay_rays else DEFAULT_OUTPUT_DIR
        )
    if args.gif is None:
        args.gif = DEFAULT_OVERLAY_GIF_PATH if args.overlay_rays else DEFAULT_GIF_PATH

    run_dir = resolve_run_dir(args.run_dir)
    paths = sorted((run_dir / "bin").glob("laser_shell.fluid.*.bin"))
    if args.max_index is not None:
        paths = [path for path in paths if dump_index(path) <= args.max_index]
    if not paths:
        raise FileNotFoundError(f"No selected fluid dumps in {run_dir / 'bin'}")

    layout = build_slice_layout(paths[0])
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        frames = list(executor.map(partial(read_midplane_density, layout=layout), paths))
    reference_extent = frames[0].extent_mm
    reference_shape = frames[0].density_g_cc.shape
    for frame in frames[1:]:
        if frame.extent_mm != reference_extent or frame.density_g_cc.shape != reference_shape:
            raise RuntimeError("Selected density dumps are not co-spatial")

    if args.overlay_rays:
        laser_paths = [
            run_dir / "bin" / f"laser_shell.laser.{dump_index(path):05d}.bin"
            for path in paths
        ]
        missing = [path for path in laser_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing paired laser dumps: " + ", ".join(str(path) for path in missing)
            )
        validate_paired_layout(laser_paths[0], layout)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            ray_frames = list(
                executor.map(partial(read_midplane_rays, layout=layout), laser_paths)
            )
        combined = []
        for frame, (ray_time_ns, rays) in zip(frames, ray_frames):
            if not math.isclose(frame.time_ns, ray_time_ns, rel_tol=0.0, abs_tol=1.0e-9):
                raise RuntimeError(
                    f"Fluid/laser time mismatch at dump {dump_index(frame.path):05d}"
                )
            combined.append(replace(frame, rays=rays))
        frames = combined

    positive_min = min(
        float(np.min(frame.density_g_cc[frame.density_g_cc > 0.0]))
        for frame in frames
    )
    density_max = max(float(np.max(frame.density_g_cc)) for frame in frames)
    norm = LogNorm(vmin=positive_min, vmax=density_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "limits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        header = ["file", "time_ns", "density_min_g_cc", "density_max_g_cc"]
        if args.overlay_rays:
            header.append("visible_ray_cells")
        writer.writerow(header)
        for frame in frames:
            positive = frame.density_g_cc[frame.density_g_cc > 0.0]
            row = [
                frame.path.name,
                f"{frame.time_ns:.16g}",
                f"{float(np.min(positive)):.16g}",
                f"{float(np.max(frame.density_g_cc)):.16g}",
            ]
            if args.overlay_rays:
                assert frame.rays is not None
                row.append(str(int(np.count_nonzero(frame.rays.path_mm >= PATH_VMIN_MM))))
            writer.writerow(row)

    rendered = []
    for number, frame in enumerate(frames, start=1):
        output = render_frame(frame, args.output_dir, norm, args.cmap, args.dpi)
        rendered.append(output)
        print(f"[{number}/{len(frames)}] {output}", flush=True)
    write_gif(rendered, args.gif, args.duration_ms)
    print(
        f"shared density limits: {positive_min:.8g} to {density_max:.8g} g cm^-3",
        flush=True,
    )
    print(args.gif, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
