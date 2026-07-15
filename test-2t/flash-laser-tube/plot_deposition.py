#!/usr/bin/env python3
"""Plot laser power deposition versus distance for the FLASH tube benchmark."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_benchmark import CODATA_EXIT, N_RAYS, read_laser_binary


BENCH_DIR = Path(__file__).resolve().parent
X1_MAX = 2.0 * math.pi
TRANSVERSE_WIDTH = 10.0
INITIAL_POWER_PER_RAY = 1.0

# CODATA cgs constants, matching src/laser/laser_physics.hpp.
ELECTRON_CHARGE_CGS = 4.803204712570263e-10
ELECTRON_MASS_CGS = 9.1093837015e-28
BOLTZMANN_CGS = 1.380649e-16
LIGHT_SPEED_CGS = 2.99792458e10
WAVELENGTH_CGS = 1.0e-4
TEMPERATURE_CENTER_K = 1.160451812155008e8


def critical_density(wavelength: float) -> float:
    return (
        ELECTRON_MASS_CGS
        * math.pi
        * LIGHT_SPEED_CGS**2
        / (ELECTRON_CHARGE_CGS**2 * wavelength**2)
    )


def collision_frequency_center() -> float:
    """Electron-ion collision frequency at ne/nc=1/2, Te=10 keV, Z=lnLambda=1."""
    ne = 0.5 * critical_density(WAVELENGTH_CGS)
    return (
        (4.0 / 3.0)
        * math.sqrt(2.0 * math.pi / ELECTRON_MASS_CGS)
        * ne
        * ELECTRON_CHARGE_CGS**4
        / (BOLTZMANN_CGS * TEMPERATURE_CENTER_K) ** 1.5
    )


# The launch density ratio is 0.5 + 0.02*3^2 = 0.68, so the conserved axial
# Hamiltonian wave-vector component is q1=sqrt(1-0.68)=sqrt(0.32).
ATTENUATION_SCALE = collision_frequency_center() / (
    LIGHT_SPEED_CGS * math.sqrt(0.32)
)


def analytic_density_ratio(x1: np.ndarray) -> np.ndarray:
    """ne/nc along r=3*cos(x1/4)."""
    return 0.5 + 0.18 * np.cos(x1 / 4.0) ** 2


def analytic_optical_depth(x1: np.ndarray) -> np.ndarray:
    """Closed-form integral of inverse-bremsstrahlung attenuation along a ray.

    Te proportional to ne^(2/3) makes nu_ei constant.  Multiplication by the
    refracted path element cancels sqrt(1-ne/nc) in the group-speed denominator,
    leaving d(tau)/dx1 = ATTENUATION_SCALE*(0.59+0.09*cos(x1/2)).
    """
    return ATTENUATION_SCALE * (0.59 * x1 + 0.18 * np.sin(x1 / 2.0))


def analytic_remaining_power(x1: np.ndarray) -> np.ndarray:
    return INITIAL_POWER_PER_RAY * np.exp(-analytic_optical_depth(x1))


def analytic_deposition_rate(x1: np.ndarray) -> np.ndarray:
    return (
        ATTENUATION_SCALE
        * analytic_density_ratio(x1)
        * analytic_remaining_power(x1)
    )


def extract_deposition(fields: dict[str, np.ndarray], nx: tuple[int, int, int]):
    nx1, nx2, nx3 = nx
    dx1 = X1_MAX / nx1
    dx2 = TRANSVERSE_WIDTH / nx2
    dx3 = TRANSVERSE_WIDTH / nx3
    cell_volume = dx1 * dx2 * dx3

    laser_q = np.asarray(fields["laser_q"], dtype=np.float64)
    if np.any(laser_q < 0.0):
        raise RuntimeError("laser_q contains negative deposited-power density")

    edges = np.linspace(0.0, X1_MAX, nx1 + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    simulated_bin_deposition = (
        laser_q.sum(axis=(0, 1)) * cell_volume / N_RAYS
    )

    exact_power_edges = analytic_remaining_power(edges)
    exact_bin_deposition = exact_power_edges[:-1] - exact_power_edges[1:]
    simulated_cumulative = np.concatenate(
        ([0.0], np.cumsum(simulated_bin_deposition))
    )
    exact_cumulative = INITIAL_POWER_PER_RAY - exact_power_edges
    simulated_remaining = INITIAL_POWER_PER_RAY - simulated_cumulative

    return {
        "edges": edges,
        "centers": centers,
        "dx1": dx1,
        "simulated_bin_deposition": simulated_bin_deposition,
        "exact_bin_deposition": exact_bin_deposition,
        "simulated_rate": simulated_bin_deposition / dx1,
        "exact_bin_average_rate": exact_bin_deposition / dx1,
        "simulated_cumulative": simulated_cumulative,
        "exact_cumulative": exact_cumulative,
        "simulated_remaining": simulated_remaining,
        "exact_remaining": exact_power_edges,
    }


def profile_l1(track: dict[str, np.ndarray]) -> float:
    return float(
        np.sum(
            np.abs(
                track["simulated_bin_deposition"]
                - track["exact_bin_deposition"]
            )
        )
        / track["exact_cumulative"][-1]
    )


def write_csv(
    output_path: Path,
    tracks: list[tuple[int, dict[str, np.ndarray]]],
) -> None:
    fieldnames = [
        "resolution",
        "bin_index",
        "x1_left_cm",
        "x1_center_cm",
        "x1_right_cm",
        "simulated_bin_deposition_erg_per_s_per_ray",
        "analytic_bin_deposition_erg_per_s_per_ray",
        "simulated_deposition_rate_erg_per_s_per_cm_per_ray",
        "analytic_bin_average_rate_erg_per_s_per_cm_per_ray",
        "simulated_cumulative_deposition_erg_per_s_per_ray",
        "analytic_cumulative_deposition_erg_per_s_per_ray",
        "simulated_remaining_power_erg_per_s_per_ray",
        "analytic_remaining_power_erg_per_s_per_ray",
        "remaining_power_residual_erg_per_s_per_ray",
    ]
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for resolution, track in tracks:
            for i, center in enumerate(track["centers"]):
                writer.writerow({
                    "resolution": resolution,
                    "bin_index": i,
                    "x1_left_cm": f"{track['edges'][i]:.16e}",
                    "x1_center_cm": f"{center:.16e}",
                    "x1_right_cm": f"{track['edges'][i + 1]:.16e}",
                    "simulated_bin_deposition_erg_per_s_per_ray":
                        f"{track['simulated_bin_deposition'][i]:.16e}",
                    "analytic_bin_deposition_erg_per_s_per_ray":
                        f"{track['exact_bin_deposition'][i]:.16e}",
                    "simulated_deposition_rate_erg_per_s_per_cm_per_ray":
                        f"{track['simulated_rate'][i]:.16e}",
                    "analytic_bin_average_rate_erg_per_s_per_cm_per_ray":
                        f"{track['exact_bin_average_rate'][i]:.16e}",
                    "simulated_cumulative_deposition_erg_per_s_per_ray":
                        f"{track['simulated_cumulative'][i + 1]:.16e}",
                    "analytic_cumulative_deposition_erg_per_s_per_ray":
                        f"{track['exact_cumulative'][i + 1]:.16e}",
                    "simulated_remaining_power_erg_per_s_per_ray":
                        f"{track['simulated_remaining'][i + 1]:.16e}",
                    "analytic_remaining_power_erg_per_s_per_ray":
                        f"{track['exact_remaining'][i + 1]:.16e}",
                    "remaining_power_residual_erg_per_s_per_ray":
                        f"{track['simulated_remaining'][i + 1] - track['exact_remaining'][i + 1]:.16e}",
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

    track32 = extract_deposition(fields32, nx32)
    track64 = extract_deposition(fields64, nx64)
    exact_exit = float(analytic_remaining_power(np.array([X1_MAX]))[0])
    if not math.isclose(exact_exit, CODATA_EXIT, rel_tol=2.0e-15, abs_tol=0.0):
        raise RuntimeError(
            f"Analytic exit power {exact_exit:.16e} does not match {CODATA_EXIT:.16e}"
        )

    x_dense = np.linspace(0.0, X1_MAX, 1000)
    power_dense = analytic_remaining_power(x_dense)
    cumulative_dense = INITIAL_POWER_PER_RAY - power_dense
    rate_dense = analytic_deposition_rate(x_dense)

    color32 = "#d97706"
    color64 = "#2563eb"
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)

    rate_ax = axes[0, 0]
    rate_ax.plot(x_dense, rate_dense, color="black", linewidth=2.0,
                 label="analytic solution")
    rate_ax.stairs(track32["simulated_rate"], track32["edges"], color=color32,
                   linewidth=1.25, label=r"32$^3$ simulation")
    rate_ax.stairs(track64["simulated_rate"], track64["edges"], color=color64,
                   linewidth=1.25, label=r"64$^3$ simulation")
    rate_ax.set_ylabel(r"$dP_{\rm dep}/dx_1$ [erg s$^{-1}$ cm$^{-1}$ ray$^{-1}$]")
    rate_ax.set_title("Local power-deposition rate")
    rate_ax.legend(fontsize=8.5)

    cumulative_ax = axes[0, 1]
    cumulative_ax.plot(x_dense, cumulative_dense, color="black", linewidth=2.0,
                       label="analytic solution")
    cumulative_ax.plot(track32["edges"], track32["simulated_cumulative"],
                       drawstyle="steps-post", color=color32, linewidth=1.25,
                       label=rf"32$^3$ (profile L1={profile_l1(track32):.3e})")
    cumulative_ax.plot(track64["edges"], track64["simulated_cumulative"],
                       drawstyle="steps-post", color=color64, linewidth=1.25,
                       label=rf"64$^3$ (profile L1={profile_l1(track64):.3e})")
    cumulative_ax.set_ylabel(r"cumulative $P_{\rm dep}$ [erg s$^{-1}$ ray$^{-1}$]")
    cumulative_ax.set_title("Cumulative deposited power")
    cumulative_ax.legend(fontsize=8.5)

    remaining_ax = axes[1, 0]
    remaining_ax.plot(x_dense, power_dense, color="black", linewidth=2.0,
                      label="analytic solution")
    remaining_ax.plot(track32["edges"], track32["simulated_remaining"],
                      drawstyle="steps-post", color=color32, linewidth=1.25,
                      label=rf"32$^3$: $P_{{exit}}$={track32['simulated_remaining'][-1]:.9f}")
    remaining_ax.plot(track64["edges"], track64["simulated_remaining"],
                      drawstyle="steps-post", color=color64, linewidth=1.25,
                      label=rf"64$^3$: $P_{{exit}}$={track64['simulated_remaining'][-1]:.9f}")
    remaining_ax.scatter([X1_MAX], [exact_exit], color="black", marker="o", s=25,
                         zorder=5, label=rf"analytic $P_{{exit}}$={exact_exit:.9f}")
    remaining_ax.set_ylabel(r"remaining $P$ [erg s$^{-1}$ ray$^{-1}$]")
    remaining_ax.set_title("Remaining laser power")
    remaining_ax.legend(fontsize=8.2)

    residual_ax = axes[1, 1]
    residual32 = track32["simulated_remaining"] - track32["exact_remaining"]
    residual64 = track64["simulated_remaining"] - track64["exact_remaining"]
    residual_ax.axhline(0.0, color="black", linewidth=1.0)
    residual_ax.plot(track32["edges"], residual32, drawstyle="steps-post",
                     color=color32, linewidth=1.3, label=r"32$^3$")
    residual_ax.plot(track64["edges"], residual64, drawstyle="steps-post",
                     color=color64, linewidth=1.3, label=r"64$^3$")
    residual_ax.set_ylabel(r"$P_{\rm sim}-P_{\rm analytic}$ [erg s$^{-1}$ ray$^{-1}$]")
    residual_ax.set_title("Remaining-power residual")
    residual_ax.legend(fontsize=8.5)

    for ax in axes.flat:
        ax.set_xlim(0.0, X1_MAX)
        ax.set_xlabel(r"propagation distance $x_1$ [cm]")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "AthenaK FLASH quadratic laser tube: power deposition versus distance",
        fontsize=14,
    )

    png_path = args.output_dir / "laser_power_deposition_vs_analytic.png"
    pdf_path = args.output_dir / "laser_power_deposition_vs_analytic.pdf"
    csv_path = args.output_dir / "deposition_comparison.csv"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    write_csv(csv_path, [(32, track32), (64, track64)])

    exact_deposited = INITIAL_POWER_PER_RAY - exact_exit
    print(f"analytic exit power: {exact_exit:.12f} erg/s/ray")
    print(f"analytic deposited power: {exact_deposited:.12f} erg/s/ray")
    for resolution, track in ((32, track32), (64, track64)):
        deposited = float(track["simulated_cumulative"][-1])
        exit_power = float(track["simulated_remaining"][-1])
        print(
            f"{resolution}^3 deposited={deposited:.12f} exit={exit_power:.12f} "
            f"profile_L1={profile_l1(track):.8e} "
            f"exit_residual={exit_power-exact_exit:+.8e}"
        )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
