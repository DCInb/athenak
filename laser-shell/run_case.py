#!/usr/bin/env python3
"""Build and run the exact 5 ns laser pulse, then coast the shell to 10 ns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
INPUT = CASE_DIR / "laser_shell.athinput"
BUILD_DIR = CASE_DIR / "build"
BINARY = BUILD_DIR / "src" / "athena"
RUN_DIR = CASE_DIR / "run"
LOG_DIR = CASE_DIR / "logs"
HELPER = Path(
    "/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh"
)
ENV_SCRIPT = Path("/home/mengqi/Research/bashrc_athenaK")
PROBLEM = "../../laser-shell/laser_shell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build before running")
    parser.add_argument("--clean", action="store_true", help="remove prior run outputs")
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("ATHENAK_TEST_GPUS", "0"),
        help="comma-separated CUDA device IDs",
    )
    return parser.parse_args()


def clean_outputs() -> None:
    for path in (
        RUN_DIR,
        LOG_DIR,
        CASE_DIR / "plots",
        CASE_DIR / "diagnostics.md",
        CASE_DIR / "results.json",
        CASE_DIR / "run_status.json",
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def build(jobs: int) -> None:
    command = [
        str(HELPER),
        "build",
        "--problem",
        PROBLEM,
        "--repo",
        str(REPO),
        "--build-dir",
        str(BUILD_DIR),
        "--jobs",
        str(jobs),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def prepare_run_dir() -> None:
    source_dir = RUN_DIR / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    link = source_dir / "athena"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(BINARY)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    start = time.monotonic()
    with log_path.open("w") as log:
        log.write("command=" + shlex.join(command) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode, time.monotonic() - start


def run_restart(
    restart: Path, ranks: int, log_path: Path, env: dict[str, str]
) -> tuple[int, float, list[str]]:
    relative_restart = restart.relative_to(RUN_DIR)
    mpi_command = [
        "mpirun",
        "-n",
        str(ranks),
        "./src/athena",
        "-r",
        str(relative_restart),
        "time/tlim=10.0",
        "laser/beam0_end_time=4.999999999999",
        "output5/dt=-1.0",
    ]
    shell_command = (
        f"source {shlex.quote(str(ENV_SCRIPT))}; "
        f"cd {shlex.quote(str(RUN_DIR))}; exec {shlex.join(mpi_command)}"
    )
    command = ["bash", "-lc", shell_command]
    start = time.monotonic()
    with log_path.open("w") as log:
        log.write("command=" + shell_command + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode, time.monotonic() - start, mpi_command


def main() -> int:
    args = parse_args()
    devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(devices) < args.ranks:
        raise RuntimeError(
            f"Need at least {args.ranks} GPU IDs, received {len(devices)}"
        )
    if args.clean:
        clean_outputs()
    if args.build:
        build(args.jobs)
    if not BINARY.is_file():
        raise RuntimeError(f"Missing {BINARY}; rerun with --build")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {RUN_DIR}; use --clean")

    prepare_run_dir()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[: args.ranks])

    phase1_command = [
        str(HELPER),
        "run",
        "--problem",
        PROBLEM,
        "--repo",
        str(REPO),
        "--build-dir",
        str(RUN_DIR),
        "--input",
        str(INPUT),
        "--ranks",
        str(args.ranks),
    ]
    print("START phase=laser-on interval=0..5 ns", flush=True)
    phase1_code, phase1_elapsed = run_logged(
        phase1_command, LOG_DIR / "phase1_laser_on.log", env
    )
    print(
        f"DONE phase=laser-on exit={phase1_code} elapsed={phase1_elapsed:.3f}s",
        flush=True,
    )

    restart_paths = sorted(RUN_DIR.rglob("*.rst"))
    history_path = RUN_DIR / "laser_shell.user.hst"
    if phase1_code != 0 or not restart_paths or not history_path.is_file():
        status = {
            "phase1_exit_code": phase1_code,
            "phase1_elapsed_seconds": phase1_elapsed,
            "restart_found": bool(restart_paths),
            "history_found": history_path.is_file(),
        }
        (CASE_DIR / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        return 1

    phase1_history = RUN_DIR / "phase1_laser_shell.user.hst"
    shutil.copy2(history_path, phase1_history)
    restart = restart_paths[-1]

    print("START phase=laser-off interval=5..10 ns", flush=True)
    phase2_code, phase2_elapsed, phase2_command = run_restart(
        restart, args.ranks, LOG_DIR / "phase2_laser_off.log", env
    )
    print(
        f"DONE phase=laser-off exit={phase2_code} elapsed={phase2_elapsed:.3f}s",
        flush=True,
    )

    status = {
        "ranks": args.ranks,
        "visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "phase1_exit_code": phase1_code,
        "phase1_elapsed_seconds": phase1_elapsed,
        "phase2_exit_code": phase2_code,
        "phase2_elapsed_seconds": phase2_elapsed,
        "restart": str(restart),
        "phase1_command": phase1_command,
        "phase2_command": phase2_command,
    }
    (CASE_DIR / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return 0 if phase2_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
