#!/usr/bin/env python3
"""Standalone plots for prescribed cone-source transport and measured timings."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from fast_electron_cone_benchmark import SOURCES, TP_MEV, POWER_ERG_S, spectrum

LABELS = ["1.5 MeV, collimated control", "PIC spectrum, collimated control",
          "PIC spectrum, divergent"]
COLORS = ["#2266aa", "#df8632", "#188b70"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--refined", type=Path,
                        help="Completed full-refinement suite for sensitivity plots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup = json.loads((args.results/"metadata.json").read_text())["setup"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    eps = np.geomspace(.12, 6.37, 500)
    axes[0].plot(eps*TP_MEV, spectrum(eps)/(TP_MEV*quad(spectrum, .12, 6.37)[0]),
                 color=COLORS[2], label="PIC-derived fit")
    axes[0].axvline(1.5, color=COLORS[0], label="Monoenergetic control")
    axes[0].set(xlabel="Kinetic energy (MeV)", ylabel="Number PDF (MeV⁻¹)",
                yscale="log", title="Energy spectrum")
    axes[0].legend(fontsize=8)
    theta = np.linspace(0, np.pi/2, 600)
    for delta, label, c in [(10, "Collimated: mean 6.91°", COLORS[1]),
                             (90, "Divergent: mean 52.17°", COLORS[2])]:
        d = np.deg2rad(delta)
        f = lambda t: np.sin(t)*np.exp(-(t/d)**4)
        axes[1].plot(np.rad2deg(theta), f(theta)/quad(f, 0, np.pi/2)[0]*np.pi/180,
                     label=label, color=c)
    axes[1].set(xlabel="Polar angle from beam axis (degrees)",
                ylabel="Number PDF (degree⁻¹)", title="Angular spectrum with sin θ")
    axes[1].legend(fontsize=8)
    r = np.linspace(0, 30, 400)
    axes[2].plot(r, np.exp(-(r/18)**8), color="#654e9c")
    axes[2].axvline(18, ls=":", color="gray")
    axes[2].set(xlabel="Radius at source plane (µm)", ylabel="Relative intensity",
                title="Common finite beam spot")
    fig.suptitle("Reconstructed sources for cone-guided fast-ignition transport")
    for ext in ("png", "pdf"):
        fig.savefig(args.output/f"source_profiles.{ext}", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    summaries = list(csv.DictReader((args.results/"summary.csv").open()))
    for i, name in enumerate(SOURCES):
        data = np.load(args.results/f"{name}-science/fields.npz")
        axes[0].plot(data["x_um"], data["deposition_J_per_um"],
                     label=LABELS[i], color=COLORS[i])
        a = json.loads((args.results/f"{name}-science/analysis.json").read_text())
        axes[1].bar(i-.16, a["mesh_deposited_fraction"]*100, width=.3,
                    color=COLORS[i], label="Whole box" if i == 0 else None)
        axes[1].bar(i+.16, a["hotspot_fraction"]*100, width=.3,
                    color=COLORS[i], alpha=.5, hatch="//",
                    label="r < 18 µm, x < 26.67 µm" if i == 0 else None)
        for rank, offset, alpha in [(1, -.16, 1), (2, .16, .5)]:
            row = next((r for r in summaries if r["name"] == name and int(r["ranks"]) == rank), None)
            if row:
                med, lo, hi = (float(row[k]) for k in ("median_s", "min_s", "max_s"))
                axes[2].bar(i+offset, med, width=.3, color=COLORS[i], alpha=alpha,
                            yerr=[[med-lo], [hi-med]], capsize=3,
                            label=f"{rank} GPU" if i == 0 else None)
    axes[0].set(xlabel="Depth in DT (µm)", ylabel="Deposited energy (J / µm)",
                title="Axial deposition")
    axes[0].legend(fontsize=7)
    for ax in axes[1:]:
        ax.set_xticks(range(3), ["Mono\ncollimated", "PIC\ncollimated", "PIC\ndivergent"])
        ax.legend(fontsize=8)
    axes[1].set(ylabel="Fraction of injected electron energy (%)", title="Energy coupling")
    axes[2].set(ylabel="Evolution time (s)", title="Median and range of timings")
    joules = POWER_ERG_S*setup["power_scale"]*setup["duration_ps"]*1e-19
    fig.suptitle(f"Uniform DT slab • {setup['nx']}³ cells • {joules:.0f} J / "
                 f"{setup['duration_ps']:g} ps • final time {setup['tlim_ps']:g} ps")
    for ext in ("png", "pdf"):
        fig.savefig(args.output/f"transport_benchmark.{ext}", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    arrays = [np.load(args.results/f"{s}-science/fields.npz") for s in SOURCES]
    maps = [a["electron_temperature_keV"][len(a["x_um"])//2] for a in arrays]
    vmax = max(a.max() for a in maps)
    for ax, a, label in zip(axes, maps, LABELS):
        im = ax.imshow(a, origin="lower", extent=[0, 64, -32, 32], aspect="equal",
                       vmin=1, vmax=vmax, cmap="magma")
        ax.set(xlabel="Depth in DT (µm)", ylabel="Transverse y (µm)", title=label)
    fig.colorbar(im, ax=axes, label="Electron temperature (keV)", shrink=.8)
    fig.suptitle(f"Final electron temperature • {setup['nx']}³ cells • central z cell slice")
    for ext in ("png", "pdf"):
        fig.savefig(args.output/f"temperature_maps.{ext}", dpi=190)
    plt.close(fig)

    if args.refined:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
        for i, name in enumerate(SOURCES):
            for offset, alpha, style, path, label in [
                    (-.16, 1, "-", args.results/f"{name}-science", "16³ / 64 source points"),
                    (.16, .5, "--", args.refined/f"refined-{name}", "32³ / 128 source points")]:
                data = np.load(path/"fields.npz")
                record = json.loads((path/"analysis.json").read_text())
                axes[0].plot(data["x_um"], data["deposition_J_per_um"], style,
                             color=COLORS[i], label=f"{LABELS[i]} ({label[:3]})")
                for ax, key, factor in [(axes[1], "hotspot_fraction", 100),
                                        (axes[2], "max_B_kT", 1)]:
                    ax.bar(i+offset, factor*record[key], width=.3, alpha=alpha,
                           color=COLORS[i], label=label if i == 0 else None)
        axes[0].set(xlabel="Depth in DT (µm)", ylabel="Deposited energy (J / µm)",
                    title="Axial deposition: solid coarse, dashed refined")
        axes[0].legend(fontsize=6)
        for ax in axes[1:]:
            ax.set_xticks(range(3), ["Mono\ncollimated", "PIC\ncollimated", "PIC\ndivergent"])
            ax.legend(fontsize=7)
        axes[1].set(ylabel="Fraction of injected energy (%)", title="Central-cylinder deposition")
        axes[2].set(ylabel="Peak cell-centered |B| (kT)", title="Magnetic-field sensitivity")
        fig.suptitle("Full-pulse mesh + spatial-source refinement • quantitative convergence not established")
        for ext in ("png", "pdf"):
            fig.savefig(args.output/f"resolution_sensitivity.{ext}", dpi=190)
        plt.close(fig)


if __name__ == "__main__":
    main()
