#!/usr/bin/env python3
"""Plot laser tracks and power deposition on the FLASH tube SMR grid."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from analyze_smr_benchmark import (
    DOMAIN_X1_MAX,
    INTERFACE_X1,
    N_RAYS,
    block_cell_sizes,
    read_multilevel_binary,
)


BENCH_DIR = Path(__file__).resolve().parent
UNIFORM_DIR = BENCH_DIR.parent / "flash-laser-tube"
sys.path.insert(0, str(UNIFORM_DIR))
from plot_deposition import (  # noqa: E402
    analytic_deposition_rate,
    analytic_remaining_power,
)


INITIAL_RADIUS = 3.0
TRANSVERSE_LIMIT = 3.45


def analytic_radius(x1: np.ndarray) -> np.ndarray:
    return INITIAL_RADIUS * np.cos(x1 / 4.0)


def block_projection(dataset: dict, field_name: str, integrate_x3: bool) -> list[dict]:
    """Project native leaf blocks onto x1-x2 while retaining their SMR resolution."""
    groups: dict[tuple[float, ...], dict] = {}
    for block in dataset["blocks"]:
        x1min, x1max, x2min, x2max, _, _ = block["bounds"]
        level = block["logical"][3]
        key = tuple(round(value, 14) for value in (x1min, x1max, x2min, x2max))
        dx1, dx2, dx3 = block_cell_sizes(block)
        values = np.asarray(block["fields"][field_name], dtype=np.float64).sum(axis=0)
        if integrate_x3:
            values = values * dx3
        if key not in groups:
            ny, nx = values.shape
            groups[key] = {
                "level": level,
                "x_edges": np.linspace(x1min, x1max, nx + 1),
                "y_edges": np.linspace(x2min, x2max, ny + 1),
                "values": np.zeros(values.shape, dtype=np.float64),
                "dx1": dx1,
                "dx2": dx2,
            }
        groups[key]["values"] += values
    return list(groups.values())


def projection_norm(patches: list[dict], dynamic_range: float) -> LogNorm:
    positive = np.concatenate(
        [patch["values"][patch["values"] > 0.0] for patch in patches]
    )
    vmax = float(np.max(positive))
    vmin = max(float(np.min(positive)), vmax * dynamic_range)
    return LogNorm(vmin=vmin, vmax=vmax)


def overlay_analytic_rays(ax: plt.Axes, color: str = "#4df0ff") -> None:
    x1 = np.linspace(0.0, DOMAIN_X1_MAX, 900)
    launches = [
        -INITIAL_RADIUS,
        -INITIAL_RADIUS / math.sqrt(2.0),
        0.0,
        INITIAL_RADIUS / math.sqrt(2.0),
        INITIAL_RADIUS,
    ]
    for index, launch in enumerate(launches):
        ax.plot(
            x1,
            launch * np.cos(x1 / 4.0),
            color=color,
            linestyle="--",
            linewidth=1.15,
            alpha=0.95,
            label="analytic rays" if index == 0 else None,
        )


def plot_projection(
    ax: plt.Axes,
    patches: list[dict],
    norm: LogNorm,
    cmap: str,
    colorbar_label: str,
    title: str,
) -> None:
    image = None
    for patch in patches:
        image = ax.pcolormesh(
            patch["x_edges"],
            patch["y_edges"],
            np.ma.masked_less_equal(patch["values"], 0.0),
            shading="flat",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
    overlay_analytic_rays(ax)
    ax.axvline(
        INTERFACE_X1,
        color="black",
        linestyle=":",
        linewidth=1.4,
        label=r"SMR interface $x_1=\pi$",
    )
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5}
    ax.text(0.25 * DOMAIN_X1_MAX, 3.25, "level 0", color="black",
            ha="center", va="center", fontsize=8.5, bbox=label_box)
    ax.text(0.75 * DOMAIN_X1_MAX, 3.25, "level 1", color="black",
            ha="center", va="center", fontsize=8.5, bbox=label_box)
    ax.set_xlim(0.0, DOMAIN_X1_MAX)
    ax.set_ylim(-TRANSVERSE_LIMIT, TRANSVERSE_LIMIT)
    ax.set_xlabel(r"$x_1$ [cm]")
    ax.set_ylabel(r"$x_2$ [cm]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.85)
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label(colorbar_label)


def axial_bins(dataset: dict) -> tuple[list[tuple[float, float, int]], dict]:
    keys = {}
    for block in dataset["blocks"]:
        x1min, x1max, _, _, _, _ = block["bounds"]
        level = block["logical"][3]
        nx = block["fields"]["laser_path"].shape[2]
        dx1 = (x1max - x1min) / nx
        for i in range(nx):
            left = x1min + i * dx1
            right = left + dx1
            key = (round(left, 14), round(right, 14), level)
            keys[key] = (left, right, level)
    bins = sorted(keys.values(), key=lambda item: item[0])
    return bins, {
        (round(left, 14), round(right, 14), level): index
        for index, (left, right, level) in enumerate(bins)
    }


def extract_axial_diagnostics(dataset: dict) -> dict[str, np.ndarray]:
    bins, index_by_key = axial_bins(dataset)
    count = len(bins)
    path_weight = np.zeros(count)
    x1_moment = np.zeros(count)
    radius_moment = np.zeros(count)
    bin_deposition = np.zeros(count)

    for block in dataset["blocks"]:
        fields = block["fields"]
        path = np.asarray(fields["laser_path"], dtype=np.float64)
        occupied = path > 0.0
        x2_local = np.zeros(path.shape, dtype=np.float64)
        x3_local = np.zeros(path.shape, dtype=np.float64)
        np.divide(fields["laser_x2_moment"], path, out=x2_local, where=occupied)
        np.divide(fields["laser_x3_moment"], path, out=x3_local, where=occupied)
        radius_local = np.hypot(x2_local, x3_local)

        x1min, x1max, _, _, _, _ = block["bounds"]
        level = block["logical"][3]
        dx1, dx2, dx3 = block_cell_sizes(block)
        for i in range(path.shape[2]):
            left = x1min + i * dx1
            right = left + dx1
            key = (round(left, 14), round(right, 14), level)
            output_index = index_by_key[key]
            weights = path[:, :, i]
            path_weight[output_index] += float(np.sum(weights))
            x1_moment[output_index] += float(
                np.sum(fields["laser_x1_moment"][:, :, i], dtype=np.float64)
            )
            radius_moment[output_index] += float(
                np.sum(weights * radius_local[:, :, i], dtype=np.float64)
            )
            bin_deposition[output_index] += float(
                np.sum(fields["laser_q"][:, :, i], dtype=np.float64)
                * dx1
                * dx2
                * dx3
                / N_RAYS
            )

    left = np.array([entry[0] for entry in bins])
    right = np.array([entry[1] for entry in bins])
    level = np.array([entry[2] for entry in bins], dtype=int)
    x1 = x1_moment / path_weight
    radius = radius_moment / path_weight
    radius_exact = analytic_radius(x1)
    rate = bin_deposition / (right - left)
    exact_power_left = analytic_remaining_power(left)
    exact_power_right = analytic_remaining_power(right)
    exact_bin_deposition = exact_power_left - exact_power_right
    exact_bin_average_rate = exact_bin_deposition / (right - left)
    edges = np.concatenate(([left[0]], right))
    if not np.allclose(right[:-1], left[1:], rtol=0.0, atol=2.0e-13):
        raise RuntimeError("SMR axial bins are not contiguous")

    return {
        "left": left,
        "right": right,
        "edges": edges,
        "level": level,
        "x1": x1,
        "path_weight": path_weight,
        "radius": radius,
        "radius_exact": radius_exact,
        "radius_error": np.abs(radius - radius_exact),
        "bin_deposition": bin_deposition,
        "rate": rate,
        "exact_bin_deposition": exact_bin_deposition,
        "exact_bin_average_rate": exact_bin_average_rate,
    }


def write_csv(path: Path, diagnostics: dict[str, np.ndarray]) -> None:
    fieldnames = [
        "level",
        "x1_left_cm",
        "x1_track_moment_cm",
        "x1_right_cm",
        "simulated_radius_cm",
        "analytic_radius_cm",
        "radius_absolute_error_cm",
        "ray_path_length_in_axial_bin_cm",
        "simulated_bin_deposition_erg_per_s_per_ray",
        "analytic_bin_deposition_erg_per_s_per_ray",
        "simulated_deposition_rate_erg_per_s_per_cm_per_ray",
        "analytic_bin_average_rate_erg_per_s_per_cm_per_ray",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(diagnostics["left"].size):
            writer.writerow({
                "level": int(diagnostics["level"][i]),
                "x1_left_cm": f"{diagnostics['left'][i]:.16e}",
                "x1_track_moment_cm": f"{diagnostics['x1'][i]:.16e}",
                "x1_right_cm": f"{diagnostics['right'][i]:.16e}",
                "simulated_radius_cm": f"{diagnostics['radius'][i]:.16e}",
                "analytic_radius_cm": f"{diagnostics['radius_exact'][i]:.16e}",
                "radius_absolute_error_cm": f"{diagnostics['radius_error'][i]:.16e}",
                "ray_path_length_in_axial_bin_cm": f"{diagnostics['path_weight'][i]:.16e}",
                "simulated_bin_deposition_erg_per_s_per_ray":
                    f"{diagnostics['bin_deposition'][i]:.16e}",
                "analytic_bin_deposition_erg_per_s_per_ray":
                    f"{diagnostics['exact_bin_deposition'][i]:.16e}",
                "simulated_deposition_rate_erg_per_s_per_cm_per_ray":
                    f"{diagnostics['rate'][i]:.16e}",
                "analytic_bin_average_rate_erg_per_s_per_cm_per_ray":
                    f"{diagnostics['exact_bin_average_rate'][i]:.16e}",
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=BENCH_DIR
        / "runs/smr_gpu1/bin/flash_laser_tube_smr.flash_tube_smr.00001.bin",
        help="AthenaK SMR laser binary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_DIR / "plots",
        help="directory for plot and CSV outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = read_multilevel_binary(args.input)

    path_patches = block_projection(dataset, "laser_path", integrate_x3=False)
    deposition_patches = block_projection(dataset, "laser_q", integrate_x3=True)
    path_norm = projection_norm(path_patches, 1.0e-4)
    deposition_norm = projection_norm(deposition_patches, 1.0e-5)
    diagnostics = extract_axial_diagnostics(dataset)

    track_mae = float(
        np.sum(diagnostics["path_weight"] * diagnostics["radius_error"])
        / np.sum(diagnostics["path_weight"])
    )
    deposition_profile_l1 = float(
        np.sum(
            np.abs(
                diagnostics["bin_deposition"]
                - diagnostics["exact_bin_deposition"]
            )
        )
        / np.sum(diagnostics["exact_bin_deposition"])
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    plot_projection(
        axes[0, 0],
        path_patches,
        path_norm,
        "magma",
        "projected ray path length [cm]",
        r"SMR ray paths projected onto $x_1$--$x_2$",
    )
    plot_projection(
        axes[0, 1],
        deposition_patches,
        deposition_norm,
        "viridis",
        r"$\int q_{\rm laser}\,dx_3$ [erg s$^{-1}$ cm$^{-2}$]",
        r"SMR deposited power projected onto $x_1$--$x_2$",
    )

    x1_dense = np.linspace(0.0, DOMAIN_X1_MAX, 1000)
    track_ax = axes[1, 0]
    track_ax.axvspan(INTERFACE_X1, DOMAIN_X1_MAX, color="#dbeafe", alpha=0.5,
                     label="level 1 refined slab")
    track_ax.plot(x1_dense, analytic_radius(x1_dense), color="black", linewidth=2.0,
                  label=r"analytic: $r=3\cos(x_1/4)$")
    track_ax.plot(diagnostics["x1"], diagnostics["radius"], color="#2563eb",
                  linewidth=1.4, marker="o", markersize=2.6,
                  label=rf"SMR simulation (MAE={track_mae:.3e} cm)")
    track_ax.axvline(INTERFACE_X1, color="#475569", linestyle=":", linewidth=1.3)
    track_ax.set_xlim(0.0, DOMAIN_X1_MAX)
    track_ax.set_ylim(0.0, 3.15)
    track_ax.set_xlabel(r"propagation distance $x_1$ [cm]")
    track_ax.set_ylabel(r"path-weighted radius $r$ [cm]")
    track_ax.set_title("Radial laser trajectory")
    track_ax.grid(alpha=0.25)
    track_ax.legend(fontsize=8.4)

    deposition_ax = axes[1, 1]
    deposition_ax.axvspan(
        INTERFACE_X1,
        DOMAIN_X1_MAX,
        color="#dbeafe",
        alpha=0.5,
        label="level 1 refined slab",
    )
    deposition_ax.plot(
        x1_dense,
        analytic_deposition_rate(x1_dense),
        color="black",
        linewidth=2.0,
        label="analytic solution",
    )
    deposition_ax.stairs(
        diagnostics["rate"],
        diagnostics["edges"],
        baseline=None,
        color="#dc2626",
        linewidth=1.35,
        label=rf"SMR simulation (profile L1={deposition_profile_l1:.3e})",
    )
    deposition_ax.axvline(
        INTERFACE_X1, color="#475569", linestyle=":", linewidth=1.3
    )
    deposition_ax.set_xlim(0.0, DOMAIN_X1_MAX)
    deposition_ax.set_xlabel(r"propagation distance $x_1$ [cm]")
    deposition_ax.set_ylabel(
        r"$dP_{\rm dep}/dx_1$ [erg s$^{-1}$ cm$^{-1}$ ray$^{-1}$]"
    )
    deposition_ax.set_title("Laser power deposition versus distance")
    deposition_ax.grid(alpha=0.25)
    deposition_ax.legend(fontsize=8.4)

    fig.suptitle(
        "AthenaK FLASH quadratic laser tube on a static-refinement grid",
        fontsize=14,
    )
    png_path = args.output_dir / "smr_laser_track_and_deposition.png"
    pdf_path = args.output_dir / "smr_laser_track_and_deposition.pdf"
    csv_path = args.output_dir / "smr_track_deposition_comparison.csv"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_csv(csv_path, diagnostics)

    simulated_deposition = float(np.sum(diagnostics["bin_deposition"]))
    exact_deposition = float(np.sum(diagnostics["exact_bin_deposition"]))
    print(f"track path-weighted MAE: {track_mae:.8e} cm")
    print(f"deposition profile L1: {deposition_profile_l1:.8e}")
    print(f"simulated deposited power: {simulated_deposition:.12f} erg/s/ray")
    print(f"analytic deposited power: {exact_deposition:.12f} erg/s/ray")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
