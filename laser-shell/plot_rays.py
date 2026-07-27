#!/usr/bin/env python3
"""Visualize rasterized laser-ray paths in the laser-shell run.

AthenaK's laser dumps contain cell-accumulated segment diagnostics rather than
individual ray histories.  This script therefore plots ``sum(ds)`` in the two
cells adjacent to z=0 and overlays the path-weighted mean segment direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


BIN_DIR = CASE_DIR / "run" / "bin"
FRAME_DIR = CASE_DIR / "plots" / "laser_rays"
GIF_PATH = CASE_DIR / "plots" / "laser_rays.gif"
INPUT_PATH = CASE_DIR / "laser_shell.athinput"
ACTIVE_DUMPS = ("00001", "00002")

# For fully ionized equimolar CH at 1.053 um (the case's 1-omega laser).
CRITICAL_CH_DENSITY_G_CC = 3.10067e-3
PATH_VMIN_MM = 1.0e-3
PATH_VMAX_MM = 4.0e-1
ARROW_STRIDE = 4
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


def input_value(name: str) -> float:
    text = INPUT_PATH.read_text()
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    matches = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*({number})", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one numeric {name}, found {len(matches)}")
    return float(matches[0])


def assemble_selected(path: Path, names: tuple[str, ...]):
    """Assemble selected fields from this case's uniform MeshBlocks."""
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
    fields = {
        name: np.zeros((nx[2], nx[1], nx[0]), dtype=np.float64)
        for name in names
    }
    for block, logical in enumerate(raw["mb_logical"]):
        i0 = int(logical[0]) * mb_nx[0]
        j0 = int(logical[1]) * mb_nx[1]
        k0 = int(logical[2]) * mb_nx[2]
        for name in names:
            values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            nz, ny, nx_local = values.shape
            fields[name][k0:k0+nz, j0:j0+ny, i0:i0+nx_local] = values

    extent = (
        float(raw["x1min"]), float(raw["x1max"]),
        float(raw["x2min"]), float(raw["x2max"]),
        float(raw["x3min"]), float(raw["x3max"]),
    )
    return fields, extent, float(raw["time"])


def midplane_indices(nz: int) -> tuple[int, ...]:
    if nz % 2:
        return (nz // 2,)
    return (nz // 2 - 1, nz // 2)


def load_frame(dump: str) -> Frame:
    laser_path = BIN_DIR / f"laser_shell.laser.{dump}.bin"
    fluid_path = BIN_DIR / f"laser_shell.fluid.{dump}.bin"
    if not laser_path.is_file() or not fluid_path.is_file():
        raise FileNotFoundError(f"Missing paired laser/fluid dump {dump} in {BIN_DIR}")

    laser, extent, time = assemble_selected(
        laser_path, ("laser_path", "laser_dir1", "laser_dir2")
    )
    fluid, fluid_extent, fluid_time = assemble_selected(fluid_path, ("dens",))
    if not np.allclose(extent, fluid_extent) or not np.isclose(time, fluid_time):
        raise RuntimeError(f"Laser and fluid dump {dump} do not describe the same state")

    nz, ny, nx = laser["laser_path"].shape
    indices = midplane_indices(nz)
    path = np.mean(laser["laser_path"][list(indices)], axis=0)
    dir1_integral = np.mean(laser["laser_dir1"][list(indices)], axis=0)
    dir2_integral = np.mean(laser["laser_dir2"][list(indices)], axis=0)
    density = np.mean(fluid["dens"][list(indices)], axis=0)
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


def render(frame: Frame) -> Path:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    output = FRAME_DIR / f"laser_shell.laser.{frame.dump}.rays.png"

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
    sampled = np.zeros_like(arrow_mask)
    sampled[::ARROW_STRIDE, ::ARROW_STRIDE] = True
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
        f"Laser-ray paths at t = {frame.time_ns:.1f} ns\n"
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
        "Ray-segment occupancy is unweighted by power; it is not stored individual polylines or Gaussian intensity. "
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
    frames = [load_frame(dump) for dump in ACTIVE_DUMPS]
    rendered = [render(frame) for frame in frames]
    write_gif(rendered)
    for frame, path in zip(frames, rendered):
        occupied = int(np.count_nonzero(frame.path_mm >= PATH_VMIN_MM))
        print(f"{path}: t={frame.time_ns:.1f} ns, {occupied} visible cells")
    print(GIF_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
