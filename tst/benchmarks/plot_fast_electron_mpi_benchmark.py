#!/usr/bin/env python3
"""Plot paired GPU timings and strong scaling, retaining measured timing ranges."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for directory in args.input
            for row in csv.DictReader((directory/"summary.csv").open())]
    names = list(dict.fromkeys(r["case"] for r in rows))
    lookup = {(r["case"], r["variant"], int(r["ranks"])): r for r in rows}
    if len(lookup) != len(rows):
        parser.error("inputs contain duplicate case/variant/rank configurations")
    labels = {"gyration": "Gyration\n32,768", "bell": "Bell\n128",
              "oscillation": "Backreaction\n32,768"}
    for name in names:
        if name not in labels:
            n = int(float(lookup[name, "after", 1]["packets"]))
            labels[name] = f"Gyration\n{n:,}"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    x, width = np.arange(len(names)), .19
    modes = [("before", 1, "Before, 1 GPU", "#9bb5c1"),
             ("before", 2, "Before, 2 GPUs", "#447587"),
             ("after", 1, "After, 1 GPU", "#e6b282"),
             ("after", 2, "After, 2 GPUs", "#ba651d")]
    for n, (variant, ranks, label, color) in enumerate(modes):
        selected = [lookup[name, variant, ranks] for name in names]
        ys = np.array([float(r["median_s"]) for r in selected])
        errors = [ys-np.array([float(r["min_s"]) for r in selected]),
                  np.array([float(r["max_s"]) for r in selected])-ys]
        axes[0].bar(x+(n-1.5)*width, ys, width, label=label, color=color,
                    yerr=errors, capsize=2)
    axes[0].set(yscale="log", ylabel="Evolution time (s, log scale)",
                title="Median evolution time and measured range")
    axes[0].legend(ncol=2, fontsize=9)
    for n, (variant, color) in enumerate([("before", "#447587"), ("after", "#ba651d")]):
        speed = [float(lookup[name, variant, 1]["median_s"])/
                 float(lookup[name, variant, 2]["median_s"]) for name in names]
        bars = axes[1].bar(x+(n-.5)*.32, speed, .32, label=variant.capitalize(), color=color)
        axes[1].bar_label(bars, fmt="%.2fx", padding=4, fontsize=9)
    axes[1].axhline(1, color="#555555", lw=1, ls="--")
    axes[1].axhline(2, color="#777777", lw=1, ls=":")
    axes[1].set(ylim=(0, 2.25), ylabel="1-GPU time / 2-GPU time",
                title="Strong scaling: above 1 means two GPUs are faster")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [labels[n] for n in names])
        ax.set_xlabel("Case and particle count")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Fast-electron MPI optimization · A100 PCIe 40 GB\n"
                 "Identical physical time and timestep controls; startup and checkpoint writes excluded")
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output/f"mpi_scaling.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
