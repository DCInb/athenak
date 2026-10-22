#!/usr/bin/env python3
"""Archive completed cone-source measurements without large native checkpoints.

The supplied source snapshot must match the executable provenance recorded in the
timing metadata. Failed refinement logs are retained separately and never counted
as completed measurements. Run after the validation suites have finished.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from fast_electron_cone_benchmark import SOURCES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-suite", default="final-complete")
    parser.add_argument("--mesh-suite", default="mesh32-pinned")
    args = parser.parse_args()
    root, output = args.runs.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    timing_root = root/args.timing_suite
    metadata = json.loads((timing_root/"metadata.json").read_text())
    for name, expected in metadata["source_hashes"].items():
        source = args.source/Path(name).name
        if digest(source) != expected:
            raise ValueError(f"Source differs from timing executable provenance: {source}")
        copy_file(source, output/"measured-source"/source.name)
    for name, expected in metadata["profile_hashes"].items():
        if digest(args.profiles/name) != expected:
            raise ValueError(f"Profile differs from timing input: {name}")
    for p in args.profiles.iterdir():
        if p.is_file():
            copy_file(p, output/"profiles"/p.name)
    copy_file(args.profiles.parent/"references.bib", output/"references.bib")

    suites = ["base-complete", "oracles", "short", "no-fields", "refined",
              "refined-final"]
    if args.timing_suite not in suites:
        suites.append(args.timing_suite)
    if (root/args.mesh_suite/"results.jsonl").exists():
        suites.append(args.mesh_suite)
    completed, failed, max_residual = [], [], 0.0
    for suite in suites:
        source = root/suite
        timings = source/"measurements.jsonl"
        successful_timings = set()
        if timings.exists():
            for line in timings.read_text().splitlines():
                r = json.loads(line)
                successful_timings.add(f"{r['name']}-r{r['ranks']}-{r['repeat']}")
        for p in (source/"fine-profiles").glob("*"):
            if p.is_file():
                copy_file(p, output/suite/"fine-profiles"/p.name)
        for pattern in ("*.json", "*.jsonl", "*.csv", "*.athinput"):
            for p in source.glob(pattern):
                copy_file(p, output/suite/p.name)
        for directory in sorted(source.iterdir()):
            if not directory.is_dir() or not (directory/"run.log").exists():
                continue
            log = (directory/"run.log").read_text()
            is_complete = ((directory/"analysis.json").exists() or
                           directory.name in successful_timings)
            status = completed if is_complete else failed
            status.append(str(directory.relative_to(root)))
            for name in ("run.log", "command.json", "analysis.json", "fields.npz"):
                if (directory/name).exists():
                    copy_file(directory/name, output/suite/directory.name/name)
            if is_complete:
                for line in log.splitlines():
                    if line.startswith("# fast_electrons "):
                        row = {k: float(v) for k, v in re.findall(
                            r"([a-z_]+)=([-+eE.\d]+)", line)}
                        if row["injected_erg"]:
                            max_residual = max(max_residual,
                                abs(row["residual_erg"])/row["injected_erg"])
        if (root/f"{suite}.log").exists():
            copy_file(root/f"{suite}.log", output/"suite-logs"/f"{suite}.log")
    for name in ("build-gpu.log", "build-cpu.log", "regression-gpu.log",
                 "regression-cpu.log", "quadrature-tests.log", "source-tests-gpu.log",
                 "gpu-placement.csv", "gpu-placement-final.csv", "memory-cgroup.txt",
                 "memory-cgroup-final.txt", "memory-cgroup-final-progress.txt",
                 "source-reproduction.json",
                 "regression-gpu-host-copy.log", "regression-cpu-host-copy.log",
                 "build-gpu-host-copy.log", "build-cpu-host-copy.log"):
        if (root/name).exists():
            copy_file(root/name, output/"checks"/name)

    for p in (root/"before-host-copy").glob("*.*"):
        copy_file(p, output/"baseline-source"/p.name)
    timing = list(csv.DictReader((timing_root/"summary.csv").open()))
    baseline_timing = list(csv.DictReader((root/"base-complete/summary.csv").open()))
    physics = []
    for name in SOURCES:
        record = json.loads((timing_root/f"{name}-science/analysis.json").read_text())
        ts = {int(r["ranks"]): float(r["median_s"]) for r in timing if r["name"] == name}
        physics.append({"source": name, "whole_box_percent": 100*record["mesh_deposited_fraction"],
                        "central_cylinder_percent": 100*record["hotspot_fraction"],
                        "max_Te_keV": record["max_Te_keV"], "max_B_kT": record["max_B_kT"],
                        "one_gpu_seconds": ts[1], "two_gpu_seconds": ts[2],
                        "two_gpu_speedup": ts[1]/ts[2],
                        "emitted_macro_packets": record["total_emitted_macro_packets"]})
    with (output/"benchmark.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=physics[0])
        writer.writeheader()
        writer.writerows(physics)
    comparison = []
    baseline_metadata = json.loads((root/"base-complete/metadata.json").read_text())
    for r in timing:
        before = next(b for b in baseline_timing
                      if b["name"] == r["name"] and b["ranks"] == r["ranks"])
        comparison.append({"source": r["name"], "ranks": int(r["ranks"]),
                           "before_host_copy_median_s": float(before["median_s"]),
                           "after_host_copy_median_s": float(r["median_s"]),
                           "same_host": baseline_metadata["hostname"] == metadata["hostname"],
                           "speedup": float(before["median_s"])/float(r["median_s"])})
    with (output/"host_memory_change_timings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison[0])
        writer.writeheader()
        writer.writerows(comparison)

    short = [json.loads(line) for line in (root/"short/results.jsonl").read_text().splitlines()]
    base = {r["name"]: r for r in short if r["label"] == "coarse"}
    sensitivity = []
    for r in short:
        if r["label"] == "coarse":
            continue
        b = base[r["name"]]
        sensitivity.append({"variant": r["label"], "source": r["name"],
                            "whole_box_delta_percentage_points":
                                100*(r["mesh_deposited_fraction"]-b["mesh_deposited_fraction"]),
                            "central_cylinder_delta_percentage_points":
                                100*(r["hotspot_fraction"]-b["hotspot_fraction"])})
    with (output/"short_pulse_sensitivity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sensitivity[0])
        writer.writeheader()
        writer.writerows(sensitivity)
    refined = []
    for name in SOURCES:
        path = root/f"refined-final/refined-{name}/analysis.json"
        if path.exists():
            r = json.loads(path.read_text())
            c = json.loads((timing_root/f"{name}-science/analysis.json").read_text())
            refined.append({"source": name, "nx": r["setup"]["nx"],
                            "radial_order": r["setup"]["radial_order"],
                            "spatial_azimuths": r["setup"]["spatial_azimuths"],
                            "whole_box_percent": 100*r["mesh_deposited_fraction"],
                            "central_cylinder_percent": 100*r["hotspot_fraction"],
                            "central_cylinder_delta_percentage_points":
                                100*(r["hotspot_fraction"]-c["hotspot_fraction"]),
                            "max_Te_keV": r["max_Te_keV"], "max_B_kT": r["max_B_kT"]})
    if refined:
        with (output/"full_pulse_refinement.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=refined[0])
            writer.writeheader()
            writer.writerows(refined)
    counts = {}
    for name in ("regression-cpu-host-copy", "regression-gpu-host-copy", "quadrature-tests"):
        log = (root/f"{name}.log").read_text()
        counts[name] = int(re.search(r"(\d+) passed", log)[1])
        if re.search(r"\d+ (?:failed|error)", log):
            raise ValueError(f"Unsuccessful regression: {name}")
    measurements = [json.loads(line) for line in
                    (timing_root/"measurements.jsonl").read_text().splitlines()]
    repeats = {f"{name}/r{rank}": sum(r["name"] == name and r["ranks"] == rank
                                    for r in measurements)
               for name in SOURCES for rank in (1, 2)}
    checks = {"timing_executable_sha256": metadata["exe_sha256"],
              "completed_run_count": len(completed), "completed_runs": completed,
              "incomplete_runs_excluded": failed,
              "max_relative_packet_ledger_residual_all_completed_macro_steps": max_residual,
              "timing_runs": len(measurements), "timing_repeats_per_case_and_rank": repeats,
              "cpu_regression_passed": counts["regression-cpu-host-copy"],
              "gpu_regression_passed": counts["regression-gpu-host-copy"],
              "source_quadrature_tests_passed": counts["quadrature-tests"],
              "full_pulse_refinement_complete": len(refined) == len(SOURCES),
              "native_checkpoints_included": False,
              "limitations": ["prescribed source transport, not laser PIC or fusion burn",
                              "16 cubed timing mesh is not spatially converged",
                              "see incomplete_runs_excluded for failed or interrupted trials",
                              "baseline and current executable results are stored separately"]}
    (output/"validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    for path in output.rglob("*"):
        if path.is_file() and path.name == "SHA256SUMS":
            path.unlink()
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output/"SHA256SUMS").write_text("".join(
        f"{digest(p)}  {p.relative_to(output)}\n" for p in files))
    print(json.dumps({"files": len(files), "completed": len(completed),
                      "incomplete": failed, "max_relative_residual": max_residual,
                      "bytes": sum(p.stat().st_size for p in files)}, indent=2))


if __name__ == "__main__":
    main()
