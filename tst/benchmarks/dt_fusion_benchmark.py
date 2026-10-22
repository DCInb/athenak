#!/usr/bin/env python3
"""Reproducible D-T throughput/scaling benchmark; leaves the solver unchanged.

Run on an allocated compute node. The primary measurement is AthenaK's evolution
timer (a wall-clock timer despite the printed 'cpu time used' label). Process wall
time includes initialization. Profiling is separate from the timed repetitions.
"""
import argparse
import csv
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
CYCLES = 6
END_TIME = 0.00063  # ns: initial 1e-5, then at most 2x growth, six cycles


@dataclass(frozen=True)
class Case:
    backend: str
    n: int
    mode: str = "full"
    samples: int = 8
    ranks: int = 1
    block: int = 8
    stretch: int = 1

    @property
    def cells(self):
        return self.n**3 * self.stretch

    @property
    def name(self):
        return (f"{self.backend}-{self.n*self.stretch}x{self.n}x{self.n}"
                f"-{self.mode}-p{self.samples}-r{self.ranks}-b{self.block}")


def input_text(case):
    # Reuse the tested physical state, excluding all mesh/file output blocks.
    text = (ROOT / "inputs/hydro/dt_fusion.athinput").read_text().split("<output1>")[0]
    edits = {
        "mesh": {"nx1": case.n*case.stretch, "nx2": case.n, "nx3": case.n,
                 "x1max": 0.02*case.stretch,
                 **{f"{side}x{d}_bc": "periodic" for side in ("i", "o")
                    for d in (1, 2, 3)}},
        "meshblock": {f"nx{d}": case.block for d in (1, 2, 3)},
        "time": {"tlim": END_TIME, "nlim": CYCLES, "initial_dt": 1e-5},
        "dt_fusion": {"enabled": str(case.mode != "hydro").lower(),
                      "rate_multiplier": 0 if case.mode == "idle" else 1,
                      "collisions": str(case.mode != "transparent").lower(),
                      "scattering": str(case.mode != "no_scatter").lower(),
                      "packets_per_cell": case.samples, "periodic_transport": "true"},
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
        if match and block == "dt_fusion":
            # A per-rank cap large enough even for the complete global population.
            lines.append(f"max_packets = {2*CYCLES*case.cells*case.samples+1024}")
    return "\n".join(lines) + "\n"


def capture(command):
    try:
        p = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=20)
        return p.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def execute(command, directory, timeout):
    start = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        p = subprocess.Popen(command, cwd=directory, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise RuntimeError(f"Benchmark timeout: {directory}")
    elapsed = time.perf_counter() - start
    output = (directory / "run.log").read_text()
    if code:
        raise RuntimeError(f"Benchmark failed ({code}): {directory}\n{output[-3000:]}")
    return elapsed, output


def run(case, repeat, args, profiling=False):
    directory = args.output / case.name / str(repeat)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "input.athinput"
    path.write_text(input_text(case))
    exe = args.cpu_exe if case.backend == "cpu" else args.gpu_exe
    command = [str(exe), "-i", str(path)]
    if case.ranks > 1:
        command = shlex.split(args.launcher.format(ranks=case.ranks)) + command
    elif hasattr(os, "sched_getaffinity"):
        # Both the serial CPU and single-GPU host thread use the same allocated core.
        command = ["taskset", "-c", str(min(os.sched_getaffinity(0))), *command]
    if profiling:
        command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                   "--cpuctxsw=none", "--force-overwrite=true", "--export=sqlite",
                   "-o", str(directory / "trace"), *command]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    wall, output = execute(command, directory, args.timeout)
    loop = float(re.search(r"cpu time used\s*=\s*("+NUMBER+r")", output)[1])
    end = re.findall(r"^time=("+NUMBER+r") cycle=(\d+)$", output, re.M)[-1]
    assert int(end[1]) == CYCLES and math.isclose(float(end[0]), END_TIME, rel_tol=2e-6), end
    diagnostics = [dict((key, float(value)) for key, value in re.findall(
        r"([a-z_]+)=("+NUMBER+r")", line)) for line in output.splitlines()
        if line.startswith("# dt_fusion ")]
    d = diagnostics[-1] if diagnostics else {}
    if case.mode not in ("idle", "hydro"):
        assert len(diagnostics) == 2*CYCLES, "Unexpected source subcycling"
        assert d["reactions"] > 0
        assert math.isclose(d["packets"], 2*CYCLES*case.cells*case.samples), d
        assert abs(d["residual_erg"]) / d["alpha_born_erg"] < 1e-9
    result = {**asdict(case), "name": case.name, "cells": case.cells,
              "repeat": repeat, "loop_s": loop, "wall_s": wall,
              "zone_cycles_s": case.cells*CYCLES/loop, "diagnostics": d,
              "directory": str(directory.relative_to(args.output))}
    print(json.dumps({k: result[k] for k in ("name", "repeat", "loop_s", "wall_s")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-exe", type=Path, required=True)
    parser.add_argument("--gpu-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=ROOT,
                        help="source tree matching these executables, for recorded hashes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--launcher", default="mpiexec --oversubscribe --bind-to core --report-bindings -n {ranks}")
    parser.add_argument("--pilot", action="store_true", help="one 16^3 and 32^3 case per backend")
    parser.add_argument("--profile", action="store_true", help="separate Nsight Systems run")
    parser.add_argument("--profile-only", action="store_true", help="profile without timing repetitions")
    parser.add_argument("--profile-ranks", type=int, default=1, choices=(1, 2))
    parser.add_argument("--select", help="regular expression selecting benchmark case names")
    parser.add_argument("--scaling", action="store_true", help="also measure 64^3 cells/GPU")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cpu_exe = args.cpu_exe.resolve()
    args.gpu_exe = args.gpu_exe.resolve()
    args.source_dir = args.source_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "measurements.jsonl").exists():
        parser.error("output already contains measurements; use a new directory")
    metadata = {
        "hostname": platform.node(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": capture(["lscpu"]), "gpu": capture(["nvidia-smi", "-q"]),
        "cuda": capture(["nvcc", "--version"]), "mpi": capture(["mpiexec", "--version"]),
        "nsys": capture(["nsys", "--version"]),
        "affinity": sorted(os.sched_getaffinity(0)), "cycles": CYCLES,
        "physical_time_ns": END_TIME, "repeats": args.repeats,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ucx_tls": os.environ.get("UCX_TLS"),
        "source_sha256": {str(p.relative_to(args.source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((args.source_dir / "src/nuclear").glob("*.*"))},
        "executables": {str(exe): {"sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                                  "cmake_cache": (exe.parents[1]/"CMakeCache.txt").read_text()}
                        for exe in (args.cpu_exe, args.gpu_exe)},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.profile_only:
        run(Case("gpu", 32, ranks=args.profile_ranks), "profile", args, profiling=True)
        return
    cases = [Case(backend, n, mode) for n in (8, 16, 32)
             for mode in ("hydro", "full") for backend in ("cpu", "gpu")]
    cases += [Case("gpu", 32, mode) for mode in ("idle", "transparent", "no_scatter")]
    cases += [Case("gpu", 32, samples=n) for n in (2, 32)]
    cases += [Case("gpu", 32, block=16), Case("gpu", 32, ranks=2),
              Case("gpu", 32, ranks=2, stretch=2)]
    if args.scaling:
        cases += [Case("gpu", 64), Case("gpu", 64, ranks=2),
                  Case("gpu", 64, ranks=2, stretch=2),
                  Case("gpu", 32, ranks=2, block=16),
                  Case("gpu", 32, ranks=2, stretch=2, block=16),
                  Case("gpu", 64, block=16), Case("gpu", 64, ranks=2, block=16),
                  Case("gpu", 64, ranks=2, stretch=2, block=16),
                  Case("gpu", 64, block=32), Case("gpu", 64, ranks=2, block=32),
                  Case("gpu", 64, ranks=2, stretch=2, block=32)]
    if args.pilot:
        cases = [Case(backend, n) for n in (16, 32) for backend in ("cpu", "gpu")]
    if args.select:
        cases = [case for case in cases if re.search(args.select, case.name)]
    if not cases:
        parser.error("no benchmark cases selected")
    # Warm each backend; these runs do not enter statistics.
    for backend in ("cpu", "gpu"):
        run(Case(backend, 8), "warmup", args)
    records = []
    with (args.output / "measurements.jsonl").open("w") as out:
        # Rotate the order between repetitions to reduce fixed-order bias.
        for repeat in range(args.repeats):
            order = cases[repeat:] + cases[:repeat]
            for case in order:
                record = run(case, repeat, args)
                records.append(record)
                out.write(json.dumps(record) + "\n")
                out.flush()
    summary = []
    for case in cases:
        rows = [r for r in records if r["name"] == case.name]
        times = [r["loop_s"] for r in rows]
        summary.append({**asdict(case), "cells": case.cells, "cycles": CYCLES,
                        "median_s": statistics.median(times), "min_s": min(times),
                        "max_s": max(times), "stdev_s": statistics.stdev(times) if len(times)>1 else 0,
                        "wall_median_s": statistics.median(r["wall_s"] for r in rows),
                        "zone_cycles_s": case.cells*CYCLES/statistics.median(times),
                        "final_packets": rows[0]["diagnostics"].get("packets", 0)})
    with (args.output / "summary.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    if args.profile:
        run(Case("gpu", 32, ranks=args.profile_ranks), "profile", args, profiling=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
