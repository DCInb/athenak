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
import threading
import time


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
INPUT = CASE_DIR / "laser_shell.athinput"
BUILD_DIR = CASE_DIR / "build"
BINARY = BUILD_DIR / "src" / "athena"
DEFAULT_RUN_DIR = Path(
    os.environ.get(
        "ATHENAK_LASER_SHELL_RUN_DIR",
        "/home/mengqi/data/athenak-2t/laser-shell/run",
    )
)
LOG_DIR = CASE_DIR / "logs"
HELPER = Path(
    "/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh"
)
ENV_SCRIPT = Path("/home/mengqi/Research/bashrc_athenaK")
PROBLEM = "../../laser-shell/laser_shell"
RUN_SENTINEL = ".athenak_laser_shell_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build before running")
    parser.add_argument("--clean", action="store_true", help="remove prior run outputs")
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("ATHENAK_TEST_GPUS", "0,1,2,3,4,5,6,7"),
        help="comma-separated CUDA device IDs",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="production output directory (defaults to the large data filesystem)",
    )
    return parser.parse_args()


def validate_run_dir(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    if resolved_run_dir.name != "run" or resolved_run_dir.parent.name != "laser-shell":
        raise RuntimeError(
            f"Run directory must end in laser-shell/run, received {resolved_run_dir}"
        )
    sentinel = resolved_run_dir / RUN_SENTINEL
    if (
        resolved_run_dir.is_dir()
        and any(resolved_run_dir.iterdir())
        and not sentinel.is_file()
    ):
        raise RuntimeError(
            f"Refusing nonempty run directory without {RUN_SENTINEL}: "
            f"{resolved_run_dir}"
        )
    return resolved_run_dir


def clean_outputs(run_dir: Path) -> None:
    resolved_run_dir = validate_run_dir(run_dir)
    for path in (
        resolved_run_dir,
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


def prepare_run_dir(run_dir: Path) -> None:
    run_dir = validate_run_dir(run_dir)
    source_dir = run_dir / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_SENTINEL).write_text(
        "Owned by laser-shell/run_case.py; safe for --clean.\n"
    )
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


def write_status(status: dict[str, object]) -> None:
    (CASE_DIR / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")


def mpi_shell_command(run_dir: Path, mpi_command: list[str]) -> str:
    return (
        f"source {shlex.quote(str(ENV_SCRIPT))}; "
        f"cd {shlex.quote(str(run_dir))}; exec {shlex.join(mpi_command)}"
    )


def run_fresh(
    run_dir: Path, ranks: int, log_path: Path, env: dict[str, str]
) -> tuple[int, float, list[str]]:
    mpi_command = [
        "mpirun",
        "-n",
        str(ranks),
        "./src/athena",
        "--kokkos-map-device-id-by=mpi_rank",
        "-i",
        str(INPUT),
    ]
    command = ["bash", "-lc", mpi_shell_command(run_dir, mpi_command)]
    code, elapsed = run_logged(command, log_path, env)
    return code, elapsed, mpi_command


def run_restart(
    run_dir: Path, restart: Path, ranks: int, log_path: Path, env: dict[str, str]
) -> tuple[int, float, list[str]]:
    relative_restart = restart.relative_to(run_dir)
    mpi_command = [
        "mpirun",
        "-n",
        str(ranks),
        "./src/athena",
        "--kokkos-map-device-id-by=mpi_rank",
        "-r",
        str(relative_restart),
        "time/tlim=10.0",
        "laser/beam0_end_time=4.999999999999",
        "output5/dt=1.0",
    ]
    shell_command = mpi_shell_command(run_dir, mpi_command)
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


class GpuMemoryMonitor:
    """Sample physical GPU memory while AthenaK is alive."""

    def __init__(self, device_ids: list[str], interval: float = 0.5):
        self.device_ids = device_ids
        self.interval = interval
        self.total_mib = {device: 0 for device in device_ids}
        self.peak_used_mib = {device: 0 for device in device_ids}
        self.baseline_used_mib = {device: 0 for device in device_ids}
        self.model_names = {device: "" for device in device_ids}
        self.uuids = {device: "" for device in device_ids}
        self.baseline_compute_processes: list[dict[str, object]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        selected = set(self.device_ids)
        for line in result.stdout.splitlines():
            index, name, uuid, used, total = (
                item.strip() for item in line.split(",", 4)
            )
            if index not in selected:
                continue
            self.model_names[index] = name
            self.uuids[index] = uuid
            self.total_mib[index] = int(total)
            self.peak_used_mib[index] = max(self.peak_used_mib[index], int(used))

    def _compute_processes(self) -> list[dict[str, object]]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        uuid_to_device = {
            uuid: device for device, uuid in self.uuids.items() if uuid
        }
        processes = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 3 or parts[0] not in uuid_to_device:
                continue
            processes.append(
                {
                    "device_id": uuid_to_device[parts[0]],
                    "pid": int(parts[1]),
                    "used_mib": int(parts[2]),
                }
            )
        return processes

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.errors.append(str(error))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._sample()
        self.baseline_used_mib = dict(self.peak_used_mib)
        self.baseline_compute_processes = self._compute_processes()
        if self.baseline_compute_processes:
            raise RuntimeError(
                "Selected GPUs already have compute processes: "
                f"{self.baseline_compute_processes}"
            )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join()
        self._sample()
        peak_delta_mib = {
            device: max(
                self.peak_used_mib[device]-self.baseline_used_mib[device], 0
            )
            for device in self.device_ids
        }
        fractions = {
            device: peak_delta_mib[device] / self.total_mib[device]
            if self.total_mib[device] > 0 else 0.0
            for device in self.device_ids
        }
        return {
            "device_ids": self.device_ids,
            "model_names": self.model_names,
            "uuids": self.uuids,
            "total_mib": self.total_mib,
            "baseline_used_mib": self.baseline_used_mib,
            "baseline_compute_processes": self.baseline_compute_processes,
            "peak_used_mib": self.peak_used_mib,
            "peak_delta_mib": peak_delta_mib,
            "peak_fraction": fractions,
            "monitor_errors": self.errors,
        }


def main() -> int:
    args = parse_args()
    run_dir = validate_run_dir(args.run_dir)
    devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if args.ranks <= 0:
        raise RuntimeError(f"--ranks must be positive, received {args.ranks}")
    if len(set(devices)) != len(devices):
        raise RuntimeError(f"GPU IDs must be unique, received {devices}")
    if len(devices) < args.ranks:
        raise RuntimeError(
            f"Need at least {args.ranks} GPU IDs, received {len(devices)}"
        )
    if args.clean:
        clean_outputs(run_dir)
    if args.build:
        build(args.jobs)
    if not BINARY.is_file():
        raise RuntimeError(f"Missing {BINARY}; rerun with --build")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {run_dir}; use --clean")

    prepare_run_dir(run_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[: args.ranks])

    write_status({
        "state": "phase1_running",
        "ranks": args.ranks,
        "visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "run_dir": str(run_dir),
    })
    monitor = GpuMemoryMonitor(devices[: args.ranks])
    monitor.start()
    print("START phase=laser-on interval=0..5 ns", flush=True)
    phase1_code, phase1_elapsed, phase1_command = run_fresh(
        run_dir, args.ranks, LOG_DIR / "phase1_laser_on.log", env
    )
    gpu_memory = monitor.stop()
    print(
        f"DONE phase=laser-on exit={phase1_code} elapsed={phase1_elapsed:.3f}s",
        flush=True,
    )

    restart_paths = sorted(run_dir.rglob("*.rst"))
    history_path = run_dir / "laser_shell.user.hst"
    if phase1_code != 0 or not restart_paths or not history_path.is_file():
        status = {
            "state": "phase1_failed",
            "ranks": args.ranks,
            "visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "phase1_exit_code": phase1_code,
            "phase1_elapsed_seconds": phase1_elapsed,
            "run_dir": str(run_dir),
            "gpu_memory": gpu_memory,
            "restart_found": bool(restart_paths),
            "history_found": history_path.is_file(),
        }
        write_status(status)
        return 1

    phase1_history = run_dir / "phase1_laser_shell.user.hst"
    shutil.copy2(history_path, phase1_history)
    restart = restart_paths[-1]

    write_status({
        "state": "phase2_running",
        "ranks": args.ranks,
        "visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "run_dir": str(run_dir),
        "gpu_memory": gpu_memory,
        "phase1_exit_code": phase1_code,
        "phase1_elapsed_seconds": phase1_elapsed,
        "restart": str(restart),
        "phase1_command": phase1_command,
    })
    print("START phase=laser-off interval=5..10 ns", flush=True)
    phase2_code, phase2_elapsed, phase2_command = run_restart(
        run_dir, restart, args.ranks, LOG_DIR / "phase2_laser_off.log", env
    )
    print(
        f"DONE phase=laser-off exit={phase2_code} elapsed={phase2_elapsed:.3f}s",
        flush=True,
    )

    status = {
        "state": "complete" if phase2_code == 0 else "phase2_failed",
        "ranks": args.ranks,
        "visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "run_dir": str(run_dir),
        "gpu_memory": gpu_memory,
        "phase1_exit_code": phase1_code,
        "phase1_elapsed_seconds": phase1_elapsed,
        "phase2_exit_code": phase2_code,
        "phase2_elapsed_seconds": phase2_elapsed,
        "restart": str(restart),
        "phase1_command": phase1_command,
        "phase2_command": phase2_command,
    }
    write_status(status)
    return 0 if phase2_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
