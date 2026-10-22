#!/usr/bin/env python3
"""Summarize the isolated old/new GPU runs and evolution-only Nsight traces."""
import argparse
import json
from pathlib import Path
import re
import sqlite3


def trace_summary(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    dt = list(db.execute(
        "SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id "
        "WHERE s.value LIKE '%Hydro::NewTimeStep(%' ORDER BY k.start"))
    assert len(dt) == 4
    start, end = dt[0][1], dt[-1][1]
    rows = list(db.execute(
        "SELECT s.value,count(*),sum(k.end-k.start)/1e9,"
        "min(k.gridX*k.gridY*k.gridZ),max(k.gridX*k.gridY*k.gridZ),"
        "min(k.blockX*k.blockY*k.blockZ),max(k.blockX*k.blockY*k.blockZ) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
        "WHERE k.start>=? AND k.end<=? GROUP BY k.demangledName", (start, end)))
    methods = {}
    restriction = []
    pcg_iterations = 0
    for name, count, seconds, grid_min, grid_max, threads_min, threads_max in rows:
        match = re.search(
            r"((?:ThermalRadiation|TwoTemperature|Hydro|MeshBoundaryValuesCC|"
            r"MeshBoundaryValues)::[^<(]+)", name)
        method = match.group(0) if match else "other"
        stats = methods.setdefault(method, dict(kernels=0, gpu_seconds=0.0))
        stats["kernels"] += count
        stats["gpu_seconds"] += seconds
        # The old serial restriction is 4-index lambda 8; the new team restriction
        # is the only (TeamMember,int,int) lambda in this method. Other identifiers
        # are unchanged in SolveImplicitTransport itself.
        if method == "ThermalRadiation::ApplyImplicitPreconditioner" and (
                "[lambda(int, int, int, int) (instance 8)]" in name or
                "[lambda(Kokkos::Impl::CudaTeamMember, int, int)" in name):
            restriction.append(dict(gpu_seconds=seconds, kernels=count,
                                    cuda_blocks=[grid_min, grid_max],
                                    threads_per_block=[threads_min, threads_max],
                                    team_kernel="CudaTeamMember, int, int" in name))
        if method == "ThermalRadiation::SolveImplicitTransport" and (
                "[lambda(int, int, int, int) (instance 5)]" in name):
            pcg_iterations += count
    api = [dict(name=name, calls=count, seconds=seconds)
           for name, count, seconds in db.execute(
               "SELECT s.value,count(*),sum(r.end-r.start)/1e9 "
               "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
               "WHERE r.start>=? AND r.end<=? GROUP BY s.value "
               "ORDER BY sum(r.end-r.start) DESC", (start, end))]
    return dict(evolution_seconds=(end-start)/1e9,
                kernel_seconds=sum(row[2] for row in rows),
                kernel_count=sum(row[1] for row in rows),
                pcg_iterations=pcg_iterations, methods=methods,
                restriction_candidates=restriction, cuda_api=api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_dir", type=Path)
    args = parser.parse_args()
    result = {}
    for variant in ("baseline", "optimized"):
        repeats = []
        for repeat in (1, 2):
            text = (args.test_dir / (variant+"-"+str(repeat)) / "run.log").read_text()
            elapsed = [float(x) for x in re.findall(r"^elapsed=(\S+)", text, re.M)]
            assert len(elapsed) == 4 and "Terminating on cycle limit" in text
            repeats.append((elapsed[-1]-elapsed[0])/3)
        result[variant] = dict(seconds_per_step=repeats,
                               mean_seconds_per_step=sum(repeats)/len(repeats),
                               trace=trace_summary(args.test_dir / (variant+"_trace.sqlite")))
    result["speedup"] = (result["baseline"]["mean_seconds_per_step"] /
                         result["optimized"]["mean_seconds_per_step"])
    result["notes"] = ["API times include waits for GPU execution and are not additive.",
                       "Trace intervals exclude restart initialization/finalization.",
                       "restriction_candidates include old lambda 8; in the optimized "
                       "trace select the team_kernel=true entry (lambda numbering shifted)."]
    with (args.test_dir / "performance_comparison.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    for variant in ("baseline", "optimized"):
        trace = result[variant]["trace"]
        print(variant, "seconds/step", result[variant]["seconds_per_step"],
              "kernel count", trace["kernel_count"], "GPU seconds", trace["kernel_seconds"])
        print("restriction candidates", trace["restriction_candidates"])
    print("Unprofiled speedup:", result["speedup"])


if __name__ == "__main__":
    main()
