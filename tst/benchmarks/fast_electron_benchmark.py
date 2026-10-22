#!/usr/bin/env python3
"""Measure hybrid fast-electron evolution, separately from initialization and profiling.

Use an allocated compute node. Each run checks the source energy, completed physical
time, cycle count, and conservative energy ledger. No mesh output is timed.
"""

import argparse
import csv
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NUMBER = r"[-+0-9.eE]+"


@dataclass(frozen=True)
class Case:
    name: str
    ndim: int = 2
    samples: int = 128
    refined: bool = False
    ranks: int = 1
    origin: float = 0.1


CASES = [Case("small-1d", 1, 1), Case("beam-1d", 1, 2048),
         Case("small-2d", 2, 1), Case("beam-2d", 2, 2048),
         Case("beam-3d", 3, 128), Case("amr-2d", 2, 128, True),
         Case("mpi-2d", 2, 2048, False, 2),
         Case("mpi-migration", 2, 2048, False, 2, 0.49)]


def input_text(case):
    text = (ROOT / "inputs/particles/fast_electrons_hybrid.athinput").read_text()
    text = text.split("<output1>")[0]
    edits = {
        "mesh": {"nx2": 16 if case.ndim > 1 else 1,
                 "nx3": 16 if case.ndim > 2 else 1},
        "meshblock": {"nx2": 8 if case.ndim > 1 else 1,
                      "nx3": 8 if case.ndim > 2 else 1},
        "time": {"tlim": 0.004, "initial_dt": 0.001, "nlim": 3},
        "fast_electrons": {"packets_per_bin": case.samples, "power_erg_s": 1e19,
                           "end_time": 0.002, "origin_x1": case.origin,
                           "report_diagnostics": "true"},
    }
    lines, block = [], None
    for line in text.splitlines():
        match = re.match(r"<([^>]+)>", line)
        if match:
            block = match[1]
        key = line.split("=", 1)[0].strip()
        if "=" in line and key in edits.get(block, {}):
            line = f"{key} = {edits[block][key]}"
        lines.append(line)
    if case.refined:
        lines.append("""
<mesh_refinement>
refinement = static
num_levels = 2
max_nmb_per_rank = 128
<refined_region1>
level = 1
x1min = 0.0
x1max = 0.25
x2min = 0.0
x2max = 0.25
""")
    return "\n".join(lines) + "\n"


def capture(command):
    try:
        return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def run(case, repeat, args):
    directory = args.output / case.name / str(repeat)
    directory.mkdir(parents=True)
    path = directory / "input.athinput"
    path.write_text(input_text(case))
    command = [str(args.exe), "-i", str(path)]
    if case.ranks > 1:
        command = shlex.split(args.launcher.format(ranks=case.ranks)) + command
    elif hasattr(os, "sched_getaffinity"):
        command = ["taskset", "-c", str(min(os.sched_getaffinity(0))), *command]
    if args.profile:
        command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                   "--cpuctxsw=none", "--force-overwrite=true", "--export=sqlite",
                   "-o", str(directory / "trace"), *command]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    start = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise RuntimeError(f"Benchmark timeout: {directory}")
    wall = time.perf_counter() - start
    output = (directory / "run.log").read_text()
    if code:
        raise RuntimeError(f"Benchmark failed ({code}): {directory}\n{output[-3000:]}")
    loop = float(re.search(r"cpu time used\s*=\s*(" + NUMBER + r")", output)[1])
    end = re.findall(r"^time=(" + NUMBER + r") cycle=(\d+)$", output, re.M)[-1]
    if int(end[1]) != 3 or not math.isclose(float(end[0]), 0.004, rel_tol=2e-6):
        raise RuntimeError(f"Incomplete evolution: {end}")
    diagnostics = [dict((k, float(v)) for k, v in re.findall(
        r"([a-z_]+)=(" + NUMBER + r")", line)) for line in output.splitlines()
        if line.startswith("# fast_electrons ")]
    if len(diagnostics) != 3:
        raise RuntimeError(f"Unexpected diagnostic count: {len(diagnostics)}")
    final = diagnostics[-1]
    if not math.isclose(final["injected_erg"], 2e7, rel_tol=1e-10):
        raise RuntimeError(f"Incorrect injected energy: {final}")
    if max(abs(d["residual_erg"]) / d["injected_erg"] for d in diagnostics) > 1e-10:
        raise RuntimeError(f"Energy conservation failed: {diagnostics}")
    result = {**asdict(case), "repeat": repeat, "loop_s": loop, "wall_s": wall,
              "diagnostics": final, "profiled": args.profile, "variant": args.variant}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT,
                        help="source tree matching the executable, for recorded hashes")
    parser.add_argument("--reference-exe", type=Path,
                        help="alternate reference/candidate runs for a paired comparison")
    parser.add_argument("--reference-source-dir", type=Path,
                        help="source tree matching the reference executable")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--select", default=".", help="regular expression selecting cases")
    parser.add_argument("--launcher", default="mpiexec --oversubscribe --bind-to core -n {ranks}")
    parser.add_argument("--profile", action="store_true", help="one separate Nsight run per case")
    args = parser.parse_args()
    args.exe, args.output = args.exe.resolve(), args.output.resolve()
    args.source_dir = args.source_dir.resolve()
    args.variant = "candidate"
    if args.reference_exe and (not args.reference_source_dir or args.profile):
        parser.error("paired timing needs --reference-source-dir and excludes --profile")
    cases = [case for case in CASES if re.search(args.select, case.name)]
    if not cases or args.repeats < 1:
        parser.error("select at least one case and use positive repeats")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metadata.json").exists():
        parser.error("output already contains a benchmark; use a new directory")
    metadata = {
        "hostname": platform.node(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": capture(["lscpu"]), "gpu": capture(["nvidia-smi", "-q"]),
        "compiler": capture(["g++", "--version"]), "cuda": capture(["nvcc", "--version"]),
        "mpi": capture(["mpiexec", "--version"]), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "executable": str(args.exe), "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(args.source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((args.source_dir / "src/fast_electrons").glob("*.*"))},
        "repeats": args.repeats, "profiled": args.profile,
    }
    output = args.output
    variants = [args]
    if args.reference_exe:
        reference = copy.copy(args)
        reference.exe = args.reference_exe.resolve()
        reference.source_dir = args.reference_source_dir.resolve()
        reference.variant = "reference"
        reference.output = output / "reference"
        args.output = output / "candidate"
        variants.append(reference)
        metadata["reference"] = {
            "executable": str(reference.exe),
            "exe_sha256": hashlib.sha256(reference.exe.read_bytes()).hexdigest(),
            "source_sha256": {
                str(p.relative_to(reference.source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((reference.source_dir / "src/fast_electrons").glob("*.*"))},
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if not args.profile:
        for variant in variants:
            run(Case("warmup", 1, 1), "warmup", variant)
    records = []
    with (output / "measurements.jsonl").open("w") as stream:
        for repeat in range(1 if args.profile else args.repeats):
            shift = repeat % len(cases)
            for case in cases[shift:] + cases[:shift]:
                order = variants if repeat % 2 == 0 else variants[::-1]
                for variant in order:
                    record = run(case, repeat, variant)
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
    summary = []
    for case in cases:
        for variant in variants:
            times = [r["loop_s"] for r in records
                     if r["name"] == case.name and r["variant"] == variant.variant]
            summary.append({**asdict(case), "variant": variant.variant,
                            "median_s": statistics.median(times),
                            "min_s": min(times), "max_s": max(times), "profiled": args.profile})
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    if args.reference_exe:
        comparison = []
        for case in cases:
            rows = {r["variant"]: r for r in summary if r["name"] == case.name}
            ledgers = [r["diagnostics"] for r in records if r["name"] == case.name]
            first = ledgers[0]
            error = max(abs(d[field]-first[field]) / first["injected_erg"]
                        for d in ledgers for field in
                        ("injected_erg", "deposited_erg", "escaped_erg", "active_erg"))
            if error > 1e-10 or any(d["packets"] != first["packets"] for d in ledgers):
                raise RuntimeError(f"Reference/candidate physics mismatch: {case.name}")
            comparison.append({"name": case.name,
                "reference_s": rows["reference"]["median_s"],
                "candidate_s": rows["candidate"]["median_s"],
                "speedup": rows["reference"]["median_s"] / rows["candidate"]["median_s"],
                "ledger_difference_over_injected": error, "packets": first["packets"]})
        with (output / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)


if __name__ == "__main__":
    main()
