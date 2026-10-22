#!/usr/bin/env python3
"""Paired one/two-GPU timing using identical Sun--Bai restart fixtures.

Preserve the reference executable and its matching source before rebuilding.
Run on an allocated two-GPU node. Warmups and output writes are excluded from
timing statistics; alternating run order limits drift. Optional accuracy runs
compare full active checkpoint states and trajectories with archived fixtures.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics

import numpy as np

from fast_electron_bai_benchmark import (
    ROOT, Case, Checkpoint, analyze, capture, run, seed_particles, write_csv,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_accuracy(reference, candidate):
    checks = {}

    def compare(key, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.all(np.isfinite(b)):
            raise ValueError(f"Invalid {key} shape or values")
        error = float(np.max(np.abs(a-b))) / max(1., float(np.max(np.abs(a))))
        checks[key] = max(checks.get(key, 0.), error)
        if error > 1e-9:
            raise ValueError(f"Physics mismatch {key}: {error}")

    a = json.loads((reference/"trajectory.json").read_text())
    b = json.loads((candidate/"trajectory.json").read_text())
    if len(a) != len(b):
        raise ValueError("Different trajectory sample counts")
    for key in a[0]:
        if all(r[key] is None for r in a+b):
            continue
        compare("trajectory/"+key, [r[key] for r in a], [r[key] for r in b])
    # Averaged trajectories alone could conceal a local field/deposition error.
    left = Checkpoint(sorted((reference/"rst").glob("*.rst"))[-1])
    right = Checkpoint(sorted((candidate/"rst").glob("*.rst"))[-1])
    np.testing.assert_array_equal(left.logical, right.logical)
    assert left.cycle == right.cycle
    compare("endpoint/time", left.time, right.time)
    for gid in range(left.nmb):
        lu, lb = left.fluid(gid)
        ru, rb = right.fluid(gid)
        # Compare each component with its own scale, excluding stale ghosts.
        for v in range(7):
            compare(f"fluid/{v}", lu[v][left.active], ru[v][right.active])
        for v in range(11):
            compare(f"diagnostics/{v}", left.cell_data[gid][v][left.active],
                    right.cell_data[gid][v][right.active])
        for d in range(3):
            face = list(left.active)
            axis = 2-d
            face[axis] = slice(face[axis].start, face[axis].stop+1)
            compare(f"magnetic/{d}", lb[d][tuple(face)], rb[d][tuple(face)])
    lp, rp = left.all_packets(), right.all_packets()
    np.testing.assert_array_equal(lp["rng"], rp["rng"])
    for key in ("x", "n", "energy", "weight"):
        compare("particles/"+key, lp[key], rp[key])
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--select", default="gyration|bell|oscillation")
    parser.add_argument("--launcher", default="mpiexec --oversubscribe -n {ranks}")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--dense-ppc", type=int, default=0,
                        help="Also time gyration with this many particles/cell (a cube); "
                             "raise the packet capacity for this fixture if necessary")
    args = parser.parse_args()
    if args.repeats < 1 or args.dense_ppc < 0:
        parser.error("repeats must be positive and dense-ppc nonnegative")
    for key in ("before", "after", "before_source", "fixtures", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    variants = {"before": args.before, "after": args.after}
    metadata = {
        "host": platform.node(), "utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job": os.getenv("SLURM_JOB_ID"), "argv": os.sys.argv,
        "gpu": capture(["nvidia-smi", "-q"]), "cpu": capture(["lscpu"]),
        "mpi": capture(["mpiexec", "--version"]),
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {k: os.getenv(k) for k in
                        ("CUDA_VISIBLE_DEVICES", "UCX_TLS", "OMPI_MCA_pml", "OMP_NUM_THREADS")},
        "executables": {k: {"path": str(v), "sha256": digest(v)} for k, v in variants.items()},
        "source_sha256": {label: {str(p.relative_to(root)): digest(p)
                                  for p in sorted((root/"src/fast_electrons").glob("*.*"))}
                          for label, root in [("before", args.before_source), ("after", ROOT)]},
        "benchmark_sha256": digest(Path(__file__)),
    }
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (args.output/"benchmark.py").write_bytes(Path(__file__).read_bytes())
    workloads = []
    for name in ("gyration", "bell", "oscillation"):
        if re.search(args.select, name):
            directory = args.fixtures/f"{name}-L1"
            case = Case(**json.loads((directory/"case.json").read_text()))
            workloads.append((name, case, directory/"initial.rst"))
    if args.dense_ppc:
        case = replace(Case("gyration", level=1), ppc=args.dense_ppc)
        cube = round(args.dense_ppc**(1/3))
        if cube**3 != args.dense_ppc or math.prod(case.nx)*case.ppc > 2147483647:
            parser.error("dense-ppc must be a cube fitting signed 32-bit particle indexing")
        seed = args.output/"dense-initial.rst"
        seed_particles(args.fixtures/"gyration-L1/initial.rst", seed, case)
        workloads.append((f"gyration-ppc{args.dense_ppc}", case, seed))
    if not workloads:
        parser.error("no workloads selected")
    records, accuracy, checks = [], [], {}
    modes = [(variant, ranks) for ranks in (1, 2) for variant in variants]
    for name, case, seed in workloads:
        (args.output/f"{name}-case.json").write_text(json.dumps(
            {**asdict(case), "seed": str(seed), "seed_sha256": digest(seed)}, indent=2))
        for repeat in range(-1, args.repeats):
            order = modes if repeat % 2 == 0 else modes[::-1]
            for variant, ranks in order:
                launch = shlex.split(args.launcher.format(ranks=ranks)) if ranks > 1 else []
                directory = args.output/name/f"{variant}-{ranks}"/(
                    "warmup" if repeat < 0 else f"repeat-{repeat}")
                result = run(variants[variant], directory,
                             ["-r", seed, "time/nlim=-1", "output1/dt=-1",
                              f"fast_electrons/max_packets={max(2000000, math.prod(case.nx)*case.ppc)}"],
                             args.timeout, launch)
                ledger = result.pop("ledger")
                if (not math.isclose(result["time"], case.tlim, rel_tol=2e-6)
                        or result["cycles"] <= 0
                        or ledger["packets"] != math.prod(case.nx)*case.ppc
                        or abs(ledger["residual_erg"]/ledger["injected_erg"]) > 1e-8):
                    raise ValueError(f"Incomplete or unconserved run: {directory}")
                row = {"case": name, "variant": variant, "ranks": ranks, "repeat": repeat,
                       "cells": math.prod(case.nx), "packets": ledger["packets"],
                       **result, "ledger": ledger}
                records.append(row)
                (args.output/"measurements.json").write_text(json.dumps(records, indent=2)+"\n")
                print(json.dumps(row), flush=True)
        if args.accuracy and name in ("gyration", "bell", "oscillation"):
            for ranks in (1, 2):
                launch = shlex.split(args.launcher.format(ranks=ranks)) if ranks > 1 else []
                directory = args.output/name/f"after-{ranks}"/"accuracy"
                run(args.after, directory, ["-r", seed, "time/nlim=-1",
                    "output1/file_number=0", "output1/last_time=0"], args.timeout, launch)
                metric = {"ranks": ranks, **analyze(case, seed, directory)}
                accuracy.append(metric)
                checks[f"{name}-{ranks}"] = compare_accuracy(
                    args.fixtures/f"{name}-L1/accuracy", directory)
                (args.output/"accuracy_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
                write_csv(args.output/"accuracy.csv", accuracy)
                print(json.dumps({"accuracy": metric}), flush=True)
    summary, comparison = [], []
    for name, case, _ in workloads:
        selected = [r for r in records if r["case"] == name and r["repeat"] >= 0]
        first = selected[0]
        ledger_error = max(abs(r["ledger"][k]-first["ledger"][k]) /
                           first["ledger"]["injected_erg"] for r in selected for k in
                           ("injected_erg", "deposited_erg", "active_erg", "escaped_erg"))
        if ledger_error > 1e-9 or any(r["cycles"] != first["cycles"] for r in selected):
            raise ValueError(f"Mismatched integration/ledger: {name}")
        medians = {}
        for variant, ranks in modes:
            rows = [r for r in selected if (r["variant"], r["ranks"]) == (variant, ranks)]
            times = [r["loop_s"] for r in rows]
            medians[variant, ranks] = statistics.median(times)
            summary.append({"case": name, "variant": variant, "ranks": ranks,
                            "packets": first["packets"], "repeats": len(times),
                            "median_s": statistics.median(times), "min_s": min(times),
                            "max_s": max(times), "ledger_difference": ledger_error})
        comparison.append({"case": name, "before_1gpu_s": medians["before", 1],
                           "before_2gpu_s": medians["before", 2],
                           "after_1gpu_s": medians["after", 1],
                           "after_2gpu_s": medians["after", 2],
                           "two_gpu_improvement": medians["before", 2]/medians["after", 2],
                           "after_strong_scaling": medians["after", 1]/medians["after", 2]})
    write_csv(args.output/"summary.csv", summary)
    write_csv(args.output/"comparison.csv", comparison)
    (args.output/"complete.json").write_text(json.dumps(comparison, indent=2)+"\n")


if __name__ == "__main__":
    main()
