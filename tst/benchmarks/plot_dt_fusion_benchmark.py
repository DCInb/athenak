#!/usr/bin/env python3
"""Plot timing medians and observed min/max ranges from the D-T benchmark CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output path without suffix")
    args = parser.parse_args()
    with args.summary.open() as stream:
        rows = list(csv.DictReader(stream))

    def select(**keys):
        result = [r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items())]
        return sorted(result, key=lambda r: (int(r["cells"]), int(r["samples"])))

    def error(rows):
        values = np.array([float(r["median_s"]) for r in rows])
        low = np.array([float(r["min_s"]) for r in rows])
        high = np.array([float(r["max_s"]) for r in rows])
        return values, np.array([values-low, high-values])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), layout="constrained")
    colors = {"cpu": "#345995", "gpu": "#008577"}
    for backend, label in (("cpu", "One CPU core"), ("gpu", "One A100")):
        selected = select(backend=backend, mode="full", samples=8, ranks=1, block=8, stretch=1)
        values, spread = error(selected)
        axes[0].errorbar([int(r["cells"]) for r in selected], values, yerr=spread,
                         marker="o", color=colors[backend], capsize=4, label=label)
    axes[0].set(xscale="log", yscale="log", xlabel="Mesh cells", ylabel="Evolution time (s)",
                title="CPU and GPU throughput")
    axes[0].set_xticks([8**3, 16**3, 32**3], ["8³", "16³", "32³"])
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=.2, which="both")

    selected = select(backend="gpu", n=32, mode="full", ranks=1, block=8, stretch=1)
    values, spread = error(selected)
    axes[1].errorbar([int(r["samples"]) for r in selected], values, yerr=spread,
                     marker="o", color=colors["gpu"], capsize=4)
    axes[1].set(xlabel="Packets / cell / birth batch", ylabel="Evolution time (s)",
                title="Particle sampling cost · 32³ cells", xticks=[2, 8, 32], ylim=(0, None))
    axes[1].grid(axis="y", alpha=.2)

    single = select(backend="gpu", n=32, mode="full", samples=8, ranks=1, block=8, stretch=1)[0]
    strong = select(backend="gpu", n=32, mode="full", samples=8, ranks=2, block=8, stretch=1)[0]
    weak = select(backend="gpu", n=32, mode="full", samples=8, ranks=2, block=8, stretch=2)[0]
    values, spread = error([single, strong, weak])
    axes[2].bar([0, 1, 2], values, yerr=spread, capsize=4,
                 color=[colors["gpu"], "#dc8b35", "#dc8b35"])
    axes[2].set(xticks=[0, 1, 2], xticklabels=["1 GPU\n32³", "2 GPUs\n32³", "2 GPUs\n64×32×32"],
                ylabel="Evolution time (s)", title="MPI scaling · 8 packets / cell")
    for i, value in enumerate(values):
        axes[2].text(i, value, f"{value:.2f} s", ha="center", va="bottom")
    axes[2].set_ylim(0, max(values+spread[1])*1.22)
    axes[2].grid(axis="y", alpha=.2)
    axes[2].set_axisbelow(True)
    fig.suptitle("D–T burn and nonlocal alpha transport: 6 steps, 0.63 ps\n"
                 "Median of 3 runs; whiskers show observed min–max; startup excluded", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
