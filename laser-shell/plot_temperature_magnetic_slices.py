#!/usr/bin/env python3
"""Plot laser-shell ion/electron temperatures and diagnose the zero magnetic field."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import plot_density_slices as sparse


CASE_DIR = Path(__file__).resolve().parent
ION_DIR = CASE_DIR / "plots" / "ion_temperature_slices_xy_log"
ELECTRON_DIR = CASE_DIR / "plots" / "electron_temperature_slices_xy_log"
ION_GIF = CASE_DIR / "plots" / "ion_temperature_slices_xy_log.gif"
ELECTRON_GIF = CASE_DIR / "plots" / "electron_temperature_slices_xy_log.gif"
LIMITS_PATH = CASE_DIR / "plots" / "temperature_slices_xy_log_limits.csv"
MAGNETIC_PATH = CASE_DIR / "plots" / "magnetic_field_zero_xy.png"
ATOMIC_MASS_UNIT_CGS = 1.660538921e-24
BOLTZMANN_CGS = 1.3806488e-16


@dataclass(frozen=True)
class TemperatureFrame:
    path: Path
    time_ns: float
    ion_kelvin: np.ndarray
    electron_kelvin: np.ndarray
    extent_mm: tuple[float, float, float, float]
    slice_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot physical ion/electron temperature slices for laser-shell."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-index", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--duration-ms", type=int, default=120)
    return parser.parse_args()


def temperature_scale_kelvin(reference: Path) -> float:
    blocks = sparse.parse_runtime_header_from_path(reference)
    length_cgs = float(sparse.header_value(blocks, "units", "length_cgs"))
    time_cgs = float(sparse.header_value(blocks, "units", "time_cgs"))
    mu = float(sparse.header_value(blocks, "units", "mu"))
    velocity_cgs = length_cgs / time_cgs
    return velocity_cgs**2 * mu * ATOMIC_MASS_UNIT_CGS / BOLTZMANN_CGS


def read_temperature_frame(
    path: Path,
    layout: sparse.SliceLayout,
    scale_kelvin: float,
) -> TemperatureFrame:
    prefix, fields = sparse.read_midplane_fields(path, layout, ("tion", "tele"))
    ion, electron = fields
    return TemperatureFrame(
        path=path,
        time_ns=prefix.time_ns,
        ion_kelvin=ion * scale_kelvin,
        electron_kelvin=electron * scale_kelvin,
        extent_mm=prefix.extent_mm,
        slice_description=prefix.slice_description,
    )


def render_temperature(
    frame: TemperatureFrame,
    field: np.ndarray,
    species: str,
    tag: str,
    output_dir: Path,
    norm: LogNorm,
    cmap: str,
    dpi: int,
) -> Path:
    output = output_dir / f"{frame.path.stem}.{tag}.png"
    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    image = ax.imshow(
        field,
        origin="lower",
        extent=frame.extent_mm,
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
        rasterized=True,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label(f"{species} temperature [K]", fontsize=12)
    ax.set_xlabel(r"$x_1$ [mm]", fontsize=13)
    ax.set_ylabel(r"$x_2$ [mm]", fontsize=13)
    ax.set_title(
        f"{species} temperature at t = {frame.time_ns:.6g} ns\n"
        f"near-midplane ({frame.slice_description})",
        fontsize=15,
        pad=12,
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output


def render_zero_magnetic_field(
    path: Path,
    layout: sparse.SliceLayout,
    dpi: int,
) -> tuple[Path, float]:
    prefix, fields = sparse.read_midplane_fields(
        path, layout, ("bcc1", "bcc2", "bcc3")
    )
    bmag = np.sqrt(fields[0] ** 2 + fields[1] ** 2 + fields[2] ** 2)
    maximum = float(np.max(bmag))
    if maximum != 0.0:
        raise RuntimeError(
            "The magnetic field is nonzero; a fixed-scale magnetic sequence is required"
        )

    fig, ax = plt.subplots(figsize=(12.5, 7.4), constrained_layout=True)
    ax.imshow(
        bmag,
        origin="lower",
        extent=prefix.extent_mm,
        interpolation="nearest",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        rasterized=True,
    )
    ax.set_xlabel(r"$x_1$ [mm]", fontsize=13)
    ax.set_ylabel(r"$x_2$ [mm]", fontsize=13)
    ax.set_title(
        f"Magnetic-field magnitude at t = {prefix.time_ns:.6g} ns\n"
        f"near-midplane ({prefix.slice_description})",
        fontsize=15,
        pad=12,
    )
    ax.text(
        0.5,
        0.5,
        r"$|B| = 0$ throughout the slice",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=18,
    )
    fig.savefig(MAGNETIC_PATH, dpi=dpi, facecolor="white")
    plt.close(fig)
    return MAGNETIC_PATH, maximum


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.dpi < 1 or args.duration_ms < 1:
        raise ValueError("--jobs, --dpi, and --duration-ms must be positive")

    run_dir = sparse.resolve_run_dir(args.run_dir)
    bin_dir = run_dir / "bin"
    fluid_paths = sorted(bin_dir.glob("laser_shell.fluid.*.bin"))
    temperature_paths = sorted(bin_dir.glob("laser_shell.two_temperature.*.bin"))
    if args.max_index is not None:
        fluid_paths = [path for path in fluid_paths if sparse.dump_index(path) <= args.max_index]
        temperature_paths = [
            path for path in temperature_paths if sparse.dump_index(path) <= args.max_index
        ]
    fluid_by_index = {sparse.dump_index(path): path for path in fluid_paths}
    temperature_by_index = {
        sparse.dump_index(path): path for path in temperature_paths
    }
    indices = sorted(fluid_by_index.keys() & temperature_by_index.keys())
    if not indices:
        raise FileNotFoundError("No paired fluid/two-temperature dumps were selected")
    fluid_paths = [fluid_by_index[index] for index in indices]
    temperature_paths = [temperature_by_index[index] for index in indices]

    layout = sparse.build_slice_layout(fluid_paths[0])
    sparse.validate_paired_layout(temperature_paths[0], layout)
    scale_kelvin = temperature_scale_kelvin(temperature_paths[0])
    reader = partial(
        read_temperature_frame,
        layout=layout,
        scale_kelvin=scale_kelvin,
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        frames = list(executor.map(reader, temperature_paths))

    positive_min = min(
        float(np.min(field[field > 0.0]))
        for frame in frames
        for field in (frame.ion_kelvin, frame.electron_kelvin)
    )
    temperature_max = max(
        float(np.max(field))
        for frame in frames
        for field in (frame.ion_kelvin, frame.electron_kelvin)
    )
    norm = LogNorm(vmin=positive_min, vmax=temperature_max)
    ION_DIR.mkdir(parents=True, exist_ok=True)
    ELECTRON_DIR.mkdir(parents=True, exist_ok=True)

    ion_paths: list[Path] = []
    electron_paths: list[Path] = []
    for number, frame in enumerate(frames, start=1):
        ion_paths.append(
            render_temperature(
                frame, frame.ion_kelvin, "Ion", "tion_K", ION_DIR,
                norm, args.cmap, args.dpi,
            )
        )
        electron_paths.append(
            render_temperature(
                frame, frame.electron_kelvin, "Electron", "tele_K", ELECTRON_DIR,
                norm, args.cmap, args.dpi,
            )
        )
        print(f"[{number}/{len(frames)}] {frame.time_ns:.6g} ns", flush=True)

    sparse.write_gif(ion_paths, ION_GIF, args.duration_ms)
    sparse.write_gif(electron_paths, ELECTRON_GIF, args.duration_ms)
    with LIMITS_PATH.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("file", "time_ns", "tion_min_K", "tion_max_K", "tele_min_K", "tele_max_K")
        )
        for frame in frames:
            writer.writerow(
                (
                    frame.path.name,
                    f"{frame.time_ns:.16g}",
                    f"{float(np.min(frame.ion_kelvin)):.16g}",
                    f"{float(np.max(frame.ion_kelvin)):.16g}",
                    f"{float(np.min(frame.electron_kelvin)):.16g}",
                    f"{float(np.max(frame.electron_kelvin)):.16g}",
                )
            )

    magnetic_path, magnetic_max = render_zero_magnetic_field(
        fluid_paths[-1], layout, args.dpi
    )
    print(f"shared temperature limits: {positive_min:.8g} to {temperature_max:.8g} K")
    print(f"temperature conversion: 1 code unit = {scale_kelvin:.12g} K")
    print(f"magnetic maximum: {magnetic_max:.8g}")
    print(ION_GIF)
    print(ELECTRON_GIF)
    print(magnetic_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
