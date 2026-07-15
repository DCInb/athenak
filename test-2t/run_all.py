#!/usr/bin/env python3
"""Run the five 2T feature cases sequentially on 1, 2, and 4 GPUs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys
import time


TEST_DIR = Path(__file__).resolve().parent
REPO = TEST_DIR.parent
BUILD_BINARY = TEST_DIR / "build" / "src" / "athena"
HELPER = Path("/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh")

FEATURES = (
    ("two_temperature", TEST_DIR / "inputs" / "two_temperature_relax_gpu.athinput",
     "tests/shock_tube"),
    ("biermann_battery", REPO / "inputs" / "mhd" /
     "two_temperature_biermann.athinput", "tests/biermann_battery"),
    ("dual_energy", REPO / "inputs" / "mhd" /
     "two_temperature_dual_energy.athinput", "tests/shock_tube"),
    ("thermal_radiation", TEST_DIR / "inputs" /
     "two_temperature_mgfld_gpu.athinput", "tests/shock_tube"),
    ("laser", TEST_DIR / "inputs" /
     "two_temperature_laser_gpu.athinput", "tests/shock_tube"),
)
GPU_COUNTS = (1, 2, 4)


def write_environment(selected_devices: list[str]) -> None:
    lines = [
        f"repository={REPO}",
        f"binary={BUILD_BINARY}",
        f"selected_physical_gpus={','.join(selected_devices)}",
    ]
    commands = (
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--short"],
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,compute_mode",
         "--format=csv,noheader"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                                check=False)
        lines.append("")
        lines.append("command=" + " ".join(command))
        lines.append(result.stdout.rstrip())
        if result.stderr:
            lines.append("stderr=" + result.stderr.rstrip())
    (TEST_DIR / "environment.txt").write_text("\n".join(lines) + "\n")


def prepare_run_dir(feature: str, ranks: int) -> Path:
    run_dir = TEST_DIR / "runs" / feature / f"gpu{ranks}"
    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    link = src_dir / "athena"
    if link.exists() or link.is_symlink():
        if link.resolve() != BUILD_BINARY.resolve():
            raise RuntimeError(f"Unexpected existing binary link: {link}")
    else:
        link.symlink_to(BUILD_BINARY)
    generated = [path for path in run_dir.iterdir() if path.name != "src"]
    if generated:
        raise RuntimeError(
            f"Refusing to overwrite existing run outputs in {run_dir}: {generated}")
    return run_dir


def main() -> int:
    if not BUILD_BINARY.is_file() or not os.access(BUILD_BINARY, os.X_OK):
        raise RuntimeError(f"AthenaK binary is missing or not executable: {BUILD_BINARY}")
    if not HELPER.is_file() or not os.access(HELPER, os.X_OK):
        raise RuntimeError(f"AthenaK helper is missing or not executable: {HELPER}")

    selected_devices = [entry.strip() for entry in
                        os.environ.get("ATHENAK_TEST_GPUS", "2,3,4,5").split(",")
                        if entry.strip()]
    if len(selected_devices) < max(GPU_COUNTS):
        raise RuntimeError("ATHENAK_TEST_GPUS must list at least four physical GPU IDs")

    (TEST_DIR / "logs").mkdir(parents=True, exist_ok=True)
    write_environment(selected_devices)
    rows: list[dict[str, str | int | float]] = []

    for feature, input_file, helper_problem in FEATURES:
        for ranks in GPU_COUNTS:
            run_dir = prepare_run_dir(feature, ranks)
            visible = selected_devices[:ranks]
            log_path = TEST_DIR / "logs" / f"{feature}_gpu{ranks}.log"
            command = [
                str(HELPER), "run",
                "--problem", helper_problem,
                "--repo", str(REPO),
                "--build-dir", str(run_dir),
                "--input", str(input_file),
                "--ranks", str(ranks),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
            start = time.monotonic()
            print(f"START feature={feature} gpus={ranks} visible={env['CUDA_VISIBLE_DEVICES']}",
                  flush=True)
            with log_path.open("w") as log:
                log.write("command=" + " ".join(command) + "\n")
                log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
                log.flush()
                completed = subprocess.run(command, cwd=REPO, env=env, text=True,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           check=False)
            elapsed = time.monotonic() - start
            rows.append({
                "feature": feature,
                "gpus": ranks,
                "visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "input": str(input_file),
                "run_dir": str(run_dir),
                "log": str(log_path),
                "exit_code": completed.returncode,
                "elapsed_seconds": f"{elapsed:.6f}",
            })
            print(f"DONE feature={feature} gpus={ranks} exit={completed.returncode} "
                  f"elapsed={elapsed:.3f}s", flush=True)

    status_file = TEST_DIR / "run_status.csv"
    with status_file.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["exit_code"] != 0]
    print(f"Completed {len(rows)} sequential jobs; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise
