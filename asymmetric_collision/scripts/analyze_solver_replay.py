#!/usr/bin/env python3
"""Analyze the short Nsight replay without touching the production run.

Uses only the standard library. Kernel lambda numbers below refer to the current
thermal_radiation.cpp implementation; assertions cross-check them against logs.
Times are trace measurements, not profiler-overhead-corrected production times.
"""
import argparse
import bisect
import collections
import csv
import json
import re
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    args = parser.parse_args()
    root = args.replay_dir.resolve()
    db = sqlite3.connect((root / "solver_trace.sqlite").as_uri() + "?mode=ro", uri=True)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = list(db.execute(
        "SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
    kernel_ids = {row[2] for row in kernels}

    def kernel_id(method, instance=None):
        candidates = [i for i in kernel_ids if method + "(" in names[i]]
        if instance is not None:
            token = "::[lambda(int, int, int, int) (instance {})]".format(instance)
            candidates = [i for i in candidates if token in names[i]]
        assert len(candidates) == 1, (method, instance, candidates)
        return candidates[0]

    cycle_re = re.compile(r"^elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)", re.M)
    linear_re = re.compile(
        r"# implicit thermal radiation: groups=(\d+) max_iterations=(\d+) "
        r"max_residual_replacements=(\d+) max_relative_residual=(\S+) "
        r"max_componentwise_backward_error=(\S+)")
    source_re = re.compile(
        r"# nonlinear thermal radiation source: max_iterations=(\d+) "
        r"fallback_cells=(\d+) max_relative_residual=(\S+)")
    timings = {}
    for mode in ("quiet", "reports", "profiled"):
        log = (root / mode / "run.log").read_text()
        cycles = [[float(e), int(c), float(t), float(dt)]
                  for e, c, t, dt in cycle_re.findall(log)]
        assert len(cycles) >= 2 and "Terminating on cycle limit" in log
        seconds = [b[0] - a[0] for a, b in zip(cycles, cycles[1:])]
        timings[mode] = dict(cycles=cycles, step_seconds=seconds,
                             total_seconds=sum(seconds),
                             mean_seconds=sum(seconds) / len(seconds))
    log = (root / "profiled" / "run.log").read_text()
    linear = [dict(groups=int(g), max_iterations=int(it), replacements=int(rep),
                   relative_residual=float(r), backward_error=float(b))
              for g, it, rep, r, b in linear_re.findall(log)]
    sources = [dict(max_iterations=int(it), fallback_cells=int(fb), residual=float(r))
               for it, fb, r in source_re.findall(log)]
    steps = len(linear)
    assert steps == len(sources) == len(timings["profiled"]["step_seconds"])
    first_cycle = timings["profiled"]["cycles"][0][1]
    ngroups = linear[0]["groups"]
    copy_id = kernel_id("ThermalRadiation::SolveImplicitTransport", 1)
    update_id = kernel_id("ThermalRadiation::SolveImplicitTransport", 5)
    commit_id = kernel_id("ThermalRadiation::SolveImplicitTransport", 14)
    dt_id = kernel_id("Hydro::NewTimeStep")
    copies = [r for r in kernels if r[2] == copy_id]
    commits = [r for r in kernels if r[2] == commit_id]
    updates = [r[0] for r in kernels if r[2] == update_id]
    dt_kernels = [r for r in kernels if r[2] == dt_id]
    assert len(copies) == len(commits) == steps * ngroups
    assert len(dt_kernels) == steps + 1
    evolution_start, evolution_end = dt_kernels[0][1], dt_kernels[-1][1]
    evolution_kernels = [r for r in kernels if evolution_start <= r[0] < evolution_end]
    group_rows, transport = [], []
    for step in range(steps):
        first, last = step * ngroups, (step + 1) * ngroups - 1
        low, high = copies[first][0], commits[last][1]
        selected = [r for r in kernels if low <= r[0] and r[1] <= high]
        transport.append(dict(cycle=first_cycle + step, span_seconds=(high-low)/1e9,
                              kernel_seconds=sum(b-a for a, b, _ in selected)/1e9,
                              kernel_count=len(selected)))
        counts = []
        for group in range(ngroups):
            index = first + group
            a, b = copies[index][0], commits[index][1]
            count = bisect.bisect_right(updates, b) - bisect.bisect_left(updates, a)
            counts.append(count)
            group_rows.append(dict(cycle=first_cycle + step, group=group,
                                   pcg_iterations=count, span_seconds=(b-a)/1e9))
        assert max(counts) == linear[step]["max_iterations"]
        transport[-1].update(pcg_iterations=sum(counts),
                             mean_group_iterations=sum(counts)/ngroups)

    category = collections.defaultdict(lambda: [0, 0])
    for a, b, kid in evolution_kernels:
        match = re.search(
            r"((?:ThermalRadiation|TwoTemperature|Hydro|MeshBoundaryValuesCC|"
            r"MeshBoundaryValues)::[^<(]+)", names[kid])
        method = match.group(0) if match else "other"
        category[method][0] += 1
        category[method][1] += b-a
    category_rows = [dict(method=name, kernel_count=n, kernel_seconds=ns/1e9)
                     for name, (n, ns) in sorted(category.items(), key=lambda x: -x[1][1])]
    api = [dict(name=name, calls=n, seconds=sec) for name, n, sec in db.execute(
        "SELECT s.value,count(*),sum(r.end-r.start)/1e9 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "WHERE r.start>=? AND r.end<=? GROUP BY s.value "
        "ORDER BY sum(r.end-r.start) DESC", (evolution_start, evolution_end))]
    summary = dict(
        timings=timings, linear=linear, sources=sources, transport=transport,
        evolution_trace_seconds=(evolution_end-evolution_start)/1e9,
        evolution_kernel_seconds=sum(b-a for a, b, _ in evolution_kernels)/1e9,
        evolution_kernel_count=len(evolution_kernels), kernel_categories=category_rows,
        cuda_api=api, profiler_overhead_fraction=(timings["profiled"]["total_seconds"] /
                                                timings["quiet"]["total_seconds"] - 1),
        notes=["Transport spans: first group copy kernel to final group commit kernel.",
               "Evolution excludes initialization, bounded by timestep kernels.",
               "CUDA API durations overlap GPU work; do not add them to kernel times.",
               "Three-step checkpoint sample, not a full-run convergence guarantee."])
    with (root / "solver_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    for filename, rows in (("solver_groups.csv", group_rows),
                           ("solver_kernel_categories.csv", category_rows)):
        with (root / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in (
        "evolution_trace_seconds", "evolution_kernel_seconds", "evolution_kernel_count",
        "profiler_overhead_fraction", "transport")}, indent=2))
    print("Saved solver_summary.json, solver_groups.csv, solver_kernel_categories.csv in", root)


if __name__ == "__main__":
    main()
