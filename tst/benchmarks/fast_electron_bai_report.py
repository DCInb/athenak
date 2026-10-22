#!/usr/bin/env python3
"""Plot measured Sun--Bai benchmark trajectories, convergence, and timing ranges."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fast_electron_bai_benchmark import Case, analyze, gyration_exact, oscillation_exact, write_csv


def read_result(path):
    result = json.loads((path/"complete.json").read_text())
    # Recompute derived diagnostics from retained checkpoints with the current
    # analysis implementation, without rerunning or altering simulation states.
    for row in result["accuracy"]:
        directory = path/f"{row['case']}-L{row['level']}"
        case = Case(**json.loads((directory/"case.json").read_text()))
        row.update(analyze(case, directory/"initial.rst", directory/"accuracy"))
    return result


def compare_trajectories(reference, other, name, level):
    config_path = f"{name}-L{level}/case.json"
    reference_case = Case(**json.loads((reference/config_path).read_text()))
    other_case = Case(**json.loads((other/config_path).read_text()))
    for key in ("nx", "mb", "length", "omega", "loading", "c", "ppc", "field_cfl", "tlim"):
        if getattr(reference_case, key) != getattr(other_case, key):
            raise ValueError(f"Mismatched setup {key} for {name}: {other}")
    suffix = f"{name}-L{level}/accuracy/trajectory.json"
    left = json.loads((reference/suffix).read_text())
    right = json.loads((other/suffix).read_text())
    if len(left) != len(right):
        raise ValueError(f"Different output samples for {name}: {other}")
    checks = {}
    for key in ("time", "particle_p", "gas_velocity", "displacement", "mode", "energy", "kinetic_energy"):
        a = np.array([row[key] for row in left])
        b = np.array([row[key] for row in right])
        scale = max(1., float(np.max(np.abs(a))))
        error = float(np.max(np.abs(a-b)))/scale
        checks[key] = error
        if error > 1e-9:
            raise ValueError(f"Backend mismatch in {name}/{key}: {other}: {error}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--mpi", type=Path)
    parser.add_argument("--bell-control", type=Path)
    parser.add_argument("--bell-doubled", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.bell_control) != bool(args.bell_doubled):
        parser.error("Supply both --bell-control and --bell-doubled")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = read_result(args.gpu)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    names = ["gyration", "bell", "oscillation"]
    order_rows = []
    for col, name in enumerate(names):
        metrics = sorted([r for r in gpu["accuracy"] if r["case"] == name], key=lambda r: r["level"])
        best = metrics[-1]
        directory = args.gpu/f"{name}-L{best['level']}"
        case = Case(**json.loads((directory/"case.json").read_text()))
        rows = json.loads((directory/"accuracy/trajectory.json").read_text())
        ts = np.array([r["time"] for r in rows])
        dense_t = np.linspace(0, case.tlim, 1000)
        ax = axes[0, col]
        if name == "gyration":
            x, _ = gyration_exact(dense_t, case.c)
            observed = np.array([r["displacement"] for r in rows])
            for d, color in enumerate(["#247BA0", "#D06B35"]):
                ax.plot(dense_t, x[:, d], color=color, label=["y analytic", "z analytic"][d])
                ax.plot(ts, observed[:, d+1], "o", ms=3, color=color,
                        label=["y solver", "z solver"][d])
            ax.set(ylabel="Displacement", xlabel=r"$\Omega t$", title="Gyration: transrelativistic orbit")
            measures = [("max_position_error_over_rg", "Position / gyroradius"),
                        ("max_momentum_rel_error", "Momentum / initial momentum")]
        elif name == "bell":
            mode = np.array([complex(*r["mode"]) for r in rows])
            ax.semilogy(dense_t, np.exp(best["analytic_growth"]*dense_t), color="#247BA0", label="Analytic growth")
            ax.semilogy(ts, np.abs(mode/mode[0]), "o", ms=4, color="#D06B35", label="Solver")
            ax.set(ylabel=r"$|B_{y,k}(t)/B_{y,k}(0)|$", xlabel="Time", title="Bell: fastest-growing mode")
            measures = [("growth_rel_error", "Growth-rate error"),
                        ("frequency_rel_error", "Phase-frequency error")]
        else:
            exact = -case.loading*oscillation_exact(dense_t, case)
            gas = np.array([r["gas_velocity"] for r in rows])
            ax.plot(dense_t, exact[:, 1], color="#247BA0", label="Electron-only analytic solution")
            ax.plot(ts, gas[:, 1], "o", ms=4, color="#D06B35", label="Solver")
            ax.set(ylabel=r"Gas $u_y$", xlabel=r"$\Omega t$", title="Backreaction: one electron species")
            measures = [("max_gas_velocity_rel_error", "Gas-velocity error"),
                        ("max_kinetic_energy_rel_error", "Combined kinetic-energy error"),
                        ("frequency_rel_error", "Frequency error")]
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax = axes[1, col]
        factors = 2.**np.array([r["level"] for r in metrics])
        for key, label in measures:
            errors = np.array([r[key] for r in metrics])
            ax.loglog(factors, errors, "o-", label=label)
            orders = np.log(errors[:-1]/errors[1:])/np.log(factors[1:]/factors[:-1])
            for i, order in enumerate(orders):
                order_rows.append({"case": name, "metric": key, "from_level": metrics[i]["level"],
                                   "to_level": metrics[i+1]["level"], "order": float(order)})
        ax.set_xticks(factors, [str(int(f)) for f in factors])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set(ylabel="Relative error", xlabel="Spatial refinement factor" if name == "bell"
               else "Field timestep refinement factor")
        ax.legend(fontsize=8)
        ax.grid(which="both", alpha=.2)
    fig.suptitle("Current fast-electron solver: three tests adapted from Sun & Bai (2023)\n"
                 "Top: finest measured solution. Bottom: convergence; mesh output excluded from separate timing runs.", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output/f"accuracy.{suffix}")
    plt.close(fig)
    write_csv(args.output/"convergence.csv", order_rows)

    series = [("1 A100, current", gpu)]
    for label, path in [("1 CPU core, current", args.cpu),
                        ("1 A100, original", args.baseline), ("2 A100, current", args.mpi)]:
        if path:
            series.append((label, read_result(path)))
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    width = .8/len(series)
    comparisons = []
    accuracy_rows = []
    backend_checks = {}
    reference = {r["case"]: r for r in gpu["performance"]}
    for j, (label, result) in enumerate(series):
        accuracy_rows.extend({"backend": label, **row} for row in result["accuracy"])
        data = {r["case"]: r for r in result["performance"]}
        for name in names:
            for key in ("level", "cells", "packets"):
                if data[name][key] != reference[name][key]:
                    raise ValueError(f"Mismatched {key} in {label}/{name}")
        medians = np.array([data[n]["median_loop_s"] for n in names])
        low = medians-np.array([data[n]["min_loop_s"] for n in names])
        high = np.array([data[n]["max_loop_s"] for n in names])-medians
        positions = np.arange(3)+(j-(len(series)-1)/2)*width
        bars = ax.bar(positions, medians, width, label=label, yerr=[low, high], capsize=3)
        ax.bar_label(bars, [f"{v:.3g}s" for v in medians], padding=3, fontsize=8)
        for name in names:
            comparisons.append({"backend": label, **data[name]})
    ax.set_xticks(np.arange(3), [f"{title}\n{reference[name]['cells']:,} cells, "
                                f"{reference[name]['packets']:,} packets"
                                for name, title in zip(names, ["Gyration", "Bell", "Electron backreaction"])])
    ax.set_yscale("log")
    ax.set(ylabel="Evolution time (s), lower is better",
           title="Identical workloads per case; medians and full repeat ranges\n"
                 "Initialization and mesh output excluded; these are small validation workloads.")
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.margins(y=.12)
    ax.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output/f"performance.{suffix}")
    plt.close(fig)
    write_csv(args.output/"comparison.csv", comparisons)
    write_csv(args.output/"accuracy_metrics.csv", accuracy_rows)
    for label, path in [("cpu", args.cpu), ("baseline", args.baseline), ("mpi", args.mpi)]:
        if path:
            backend_checks[label] = {name: compare_trajectories(args.gpu, path, name,
                                                               reference[name]["level"])
                                     for name in names}
    (args.output/"backend_checks.json").write_text(json.dumps(backend_checks, indent=2))
    if args.bell_control:
        control = read_result(args.bell_control)["accuracy"][0]
        doubled = read_result(args.bell_doubled)["accuracy"][0]
        if (control["case"] != "bell" or doubled["case"] != "bell"
                or control["cells"] != doubled["cells"]
                or control["packets"] != doubled["packets"]
                or control["field_cfl"]/control["c"] != doubled["field_cfl"]/doubled["c"]):
            raise ValueError("Bell light-speed check requires matching populations, grids, and substeps")
        sensitivity = {key+"_relative_change": abs(doubled[key]/control[key]-1)
                       for key in ("growth", "frequency")}
        sensitivity.update(control=control, doubled=doubled)
        (args.output/"light_speed_check.json").write_text(json.dumps(sensitivity, indent=2))
    (args.output/"analysis_metadata.json").write_text(json.dumps({
        "argv": sys.argv,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(__file__).with_name("fast_electron_bai_benchmark.py"))}
    }, indent=2))


if __name__ == "__main__":
    main()
