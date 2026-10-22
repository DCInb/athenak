#!/usr/bin/env python3
"""Compare the D-T baseline with device residency and reduced halo exchanges."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select(rows, n=32, ranks=1, stretch=1, block=8):
    keys = dict(backend="gpu", n=n, mode="full", samples=8, ranks=ranks,
                stretch=stretch, block=block)
    return next(r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without suffix")
    args = parser.parse_args()
    with args.baseline.open() as stream:
        baseline = list(csv.DictReader(stream))
    with args.optimized.open() as stream:
        optimized = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    colors = ("#dc8b35", "#008577")
    x = np.arange(3)
    for offset, rows, label, color in zip((-.18, .18), (baseline, optimized),
                                         ("Baseline", "Optimized"), colors):
        chosen = [select(rows), select(rows, ranks=2), select(rows, ranks=2, stretch=2)]
        values = np.array([float(r["median_s"]) for r in chosen])
        low = np.array([float(r["min_s"]) for r in chosen])
        high = np.array([float(r["max_s"]) for r in chosen])
        axes[0].bar(x+offset, values, .34, yerr=[values-low, high-values],
                    capsize=3, color=color, label=label)
        for position, value, top in zip(x+offset, values, high):
            axes[0].text(position, top*1.15, f"{value:.3f}", ha="center", fontsize=9)
    axes[0].set(yscale="log", ylabel="Evolution time (s, log scale)",
                title="Same physics, sampling and physical duration",
                xticks=x, xticklabels=["1 GPU\n32³", "2 GPUs\n32³", "2 GPUs\n64×32×32"],
                ylim=(.03, 25))
    axes[0].legend(frameon=False)

    labels, strong, weak = [], [], []
    for n, block in ((32, 8), (32, 16), (64, 8), (64, 16), (64, 32)):
        one = float(select(optimized, n=n, block=block)["median_s"])
        two = float(select(optimized, n=n, ranks=2, block=block)["median_s"])
        double = float(select(optimized, n=n, ranks=2, stretch=2, block=block)["median_s"])
        labels.append(f"{n}³ mesh*\n{block}³ blocks")
        strong.append(100*one/(2*two))
        weak.append(100*one/double)
    x = np.arange(len(labels))
    axes[1].bar(x-.18, strong, .34, color="#345995", label="Strong scaling")
    axes[1].bar(x+.18, weak, .34, color=colors[1], label="Weak scaling")
    for offset, values in ((-.18, strong), (.18, weak)):
        for position, value in zip(x+offset, values):
            axes[1].text(position, value+2, f"{value:.0f}%", ha="center", fontsize=9)
    axes[1].axhline(100, color="gray", linestyle="--", linewidth=1)
    axes[1].set(ylabel="Two-GPU efficiency (%)", title="Optimized scaling on one node",
                xticks=x, xticklabels=labels, ylim=(0, max(115, max(weak)*1.15)))
    axes[1].legend(frameon=False, loc="upper left", ncol=2)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("D–T burn and nonlocal alpha transport · NVIDIA A100\n"
                 "6 steps to 0.63 ps, 8 packets/cell/birth; medians of 3 runs", fontsize=12)
    fig.supxlabel("*One-GPU reference size: strong scaling keeps the total mesh; weak scaling doubles it.\n"
                  "Left whiskers show min–max. Evolution excludes startup; sampling convergence is separate.",
                  fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=160)


if __name__ == "__main__":
    main()
