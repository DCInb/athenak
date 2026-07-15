#!/usr/bin/env python3
"""Visualize FLASH laser-tube ray tracks and compare with the analytic solution."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from analyze_benchmark import read_laser_binary


BENCH_DIR = Path(__file__).resolve().parent
DOMAIN_X1 = (0.0, 2.0 * math.pi)
DOMAIN_X2 = (-5.0, 5.0)
DOMAIN_X3 = (-5.0, 5.0)
INITIAL_RADIUS = 3.0


def analytic_radius(x1: np.ndarray) -> np.ndarray:
    """Kaiser quadratic-tube radial trajectory in the AthenaK coordinates."""
    return INITIAL_RADIUS * np.cos(x1 / 4.0)


def extract_radial_track(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce all eight rays to a path-weighted radial trajectory per x1 plane."""
    path = np.asarray(fields["laser_path"], dtype=np.float64)
    occupied = path > 0.0

    x1_local = np.full(path.shape, np.nan, dtype=np.float64)
    x2_local = np.full(path.shape, np.nan, dtype=np.float64)
    x3_local = np.full(path.shape, np.nan, dtype=np.float64)
    np.divide(fields["laser_x1_moment"], path, out=x1_local, where=occupied)
    np.divide(fields["laser_x2_moment"], path, out=x2_local, where=occupied)
    np.divide(fields["laser_x3_moment"], path, out=x3_local, where=occupied)
    radius_local = np.hypot(x2_local, x3_local)

    plane_path = path.sum(axis=(0, 1))
    valid = plane_path > 0.0
    x1 = fields["laser_x1_moment"].sum(axis=(0, 1))[valid] / plane_path[valid]

    weighted_radius = np.where(occupied, path * radius_local, 0.0)
    radius = weighted_radius.sum(axis=(0, 1))[valid] / plane_path[valid]

    plane_indices = np.flatnonzero(valid)
    radius_std = np.empty(radius.shape, dtype=np.float64)
    for output_index, i in enumerate(plane_indices):
        weights = path[:, :, i]
        local_valid = weights > 0.0
        residual = radius_local[:, :, i][local_valid] - radius[output_index]
        radius_std[output_index] = math.sqrt(
            float(np.sum(weights[local_valid] * residual * residual) / plane_path[i])
        )

    exact = analytic_radius(x1)
    error = np.abs(radius - exact)
    return {
        "plane_index": plane_indices,
        "x1": x1,
        "radius": radius,
        "radius_exact": exact,
        "abs_error": error,
        "radius_std": radius_std,
        "plane_path": plane_path[valid],
    }


def path_weighted_error(track: dict[str, np.ndarray]) -> float:
    return float(
        np.sum(track["plane_path"] * track["abs_error"])
        / np.sum(track["plane_path"])
    )


def plot_projection(
    ax: plt.Axes,
    projection: np.ndarray,
    transverse_domain: tuple[float, float],
    initial_coordinates: list[float],
    norm: LogNorm,
    transverse_label: str,
    title: str,
) -> None:
    masked = np.ma.masked_less_equal(projection, 0.0)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=(DOMAIN_X1[0], DOMAIN_X1[1], transverse_domain[0], transverse_domain[1]),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=norm,
    )

    x1_exact = np.linspace(DOMAIN_X1[0], DOMAIN_X1[1], 800)
    for curve_index, initial_coordinate in enumerate(initial_coordinates):
        ax.plot(
            x1_exact,
            initial_coordinate * np.cos(x1_exact / 4.0),
            color="#55e6ff",
            linestyle="--",
            linewidth=1.2,
            alpha=0.95,
            label="analytic rays" if curve_index == 0 else None,
        )
    ax.plot(2.0 * math.pi, 0.0, marker="o", markersize=4.5, color="white",
            markeredgecolor="black", markeredgewidth=0.5, clip_on=False,
            label=r"analytic focus $x_1=2\pi$")
    ax.set_xlim(DOMAIN_X1)
    ax.set_ylim(-3.45, 3.45)
    ax.set_xlabel(r"$x_1$ [cm]")
    ax.set_ylabel(f"${transverse_label}$ [cm]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("projected ray path length [cm]")


def write_comparison_csv(
    output_path: Path,
    tracks: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    fieldnames = [
        "resolution",
        "x1_plane_index",
        "x1_cm",
        "simulated_radius_cm",
        "analytic_radius_cm",
        "absolute_error_cm",
        "ray_to_ray_radius_std_cm",
        "path_length_in_plane_cm",
    ]
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for resolution, track in tracks:
            for row_index in range(track["x1"].size):
                writer.writerow({
                    "resolution": resolution,
                    "x1_plane_index": int(track["plane_index"][row_index]),
                    "x1_cm": f"{track['x1'][row_index]:.16e}",
                    "simulated_radius_cm": f"{track['radius'][row_index]:.16e}",
                    "analytic_radius_cm": f"{track['radius_exact'][row_index]:.16e}",
                    "absolute_error_cm": f"{track['abs_error'][row_index]:.16e}",
                    "ray_to_ray_radius_std_cm": f"{track['radius_std'][row_index]:.16e}",
                    "path_length_in_plane_cm": f"{track['plane_path'][row_index]:.16e}",
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r32",
        type=Path,
        default=BENCH_DIR / "runs/r32_gpu1/bin/flash_laser_tube_r32.flash_tube.00001.bin",
        help="32^3 AthenaK laser diagnostic binary",
    )
    parser.add_argument(
        "--r64",
        type=Path,
        default=BENCH_DIR / "runs/r64_gpu1/bin/flash_laser_tube_r64.flash_tube.00001.bin",
        help="64^3 AthenaK laser diagnostic binary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_DIR / "plots",
        help="directory for PNG, PDF, and CSV outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields32, nx32 = read_laser_binary(args.r32)
    fields64, nx64 = read_laser_binary(args.r64)
    if nx32 != (32, 32, 32) or nx64 != (64, 64, 64):
        raise RuntimeError(f"Unexpected resolutions: r32={nx32}, r64={nx64}")

    track32 = extract_radial_track(fields32)
    track64 = extract_radial_track(fields64)
    error32 = path_weighted_error(track32)
    error64 = path_weighted_error(track64)

    projection_x2 = fields64["laser_path"].sum(axis=0)
    projection_x3 = fields64["laser_path"].sum(axis=1)
    positive_values = np.concatenate(
        (projection_x2[projection_x2 > 0.0], projection_x3[projection_x3 > 0.0])
    )
    vmax = float(np.max(positive_values))
    vmin = max(float(np.min(positive_values)), vmax * 1.0e-4)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), constrained_layout=True)
    unique_launch_coordinates = [
        -INITIAL_RADIUS,
        -INITIAL_RADIUS / math.sqrt(2.0),
        0.0,
        INITIAL_RADIUS / math.sqrt(2.0),
        INITIAL_RADIUS,
    ]
    plot_projection(
        axes[0, 0],
        projection_x2,
        DOMAIN_X2,
        unique_launch_coordinates,
        norm,
        "x_2",
        r"64$^3$ path projection onto $x_1$--$x_2$",
    )
    plot_projection(
        axes[0, 1],
        projection_x3,
        DOMAIN_X3,
        unique_launch_coordinates,
        norm,
        "x_3",
        r"64$^3$ path projection onto $x_1$--$x_3$",
    )

    x1_exact = np.linspace(DOMAIN_X1[0], DOMAIN_X1[1], 800)
    radial_ax = axes[1, 0]
    radial_ax.plot(x1_exact, analytic_radius(x1_exact), color="black", linewidth=2.0,
                   label=r"analytic: $r=3\cos(x_1/4)$")
    radial_ax.plot(track32["x1"], track32["radius"], color="#d97706", linewidth=1.3,
                   marker="o", markersize=2.8,
                   label=rf"32$^3$ simulation (MAE={error32:.3e} cm)")
    radial_ax.plot(track64["x1"], track64["radius"], color="#2563eb", linewidth=1.4,
                   marker="o", markersize=2.2,
                   label=rf"64$^3$ simulation (MAE={error64:.3e} cm)")
    radial_ax.set_xlim(DOMAIN_X1)
    radial_ax.set_ylim(0.0, 3.15)
    radial_ax.set_xlabel(r"$x_1$ [cm]")
    radial_ax.set_ylabel(r"path-weighted radius $r$ [cm]")
    radial_ax.set_title("Radial trajectory")
    radial_ax.grid(alpha=0.25)
    radial_ax.legend(fontsize=8.5)

    residual_ax = axes[1, 1]
    residual_ax.semilogy(track32["x1"], np.maximum(track32["abs_error"], 1.0e-14),
                         color="#d97706", linewidth=1.3, marker="o", markersize=2.8,
                         label=rf"32$^3$ (MAE={error32:.3e} cm)")
    residual_ax.semilogy(track64["x1"], np.maximum(track64["abs_error"], 1.0e-14),
                         color="#2563eb", linewidth=1.4, marker="o", markersize=2.2,
                         label=rf"64$^3$ (MAE={error64:.3e} cm)")
    residual_ax.set_xlim(DOMAIN_X1)
    residual_ax.set_xlabel(r"$x_1$ [cm]")
    residual_ax.set_ylabel(r"$|r_{\rm sim}-r_{\rm analytic}|$ [cm]")
    residual_ax.set_title("Trajectory residual")
    residual_ax.grid(alpha=0.25, which="both")
    residual_ax.legend(fontsize=8.5)

    fig.suptitle(
        "AthenaK FLASH quadratic laser-tube benchmark: simulated and analytic ray tracks",
        fontsize=14,
    )
    png_path = args.output_dir / "laser_track_vs_analytic.png"
    pdf_path = args.output_dir / "laser_track_vs_analytic.pdf"
    csv_path = args.output_dir / "track_comparison.csv"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_comparison_csv(csv_path, [(32, track32), (64, track64)])

    print(f"32^3 path-weighted radial MAE: {error32:.8e} cm")
    print(f"64^3 path-weighted radial MAE: {error64:.8e} cm")
    print(f"error ratio (64/32): {error64/error32:.8f}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
