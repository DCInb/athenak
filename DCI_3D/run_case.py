#!/usr/bin/env python3
"""Safely build and launch provisional DCI_3D validation or production runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
PRODUCTION_INPUT = CASE_DIR / "dci_3d.athinput"
CALIBRATION_INPUT = CASE_DIR / "dci_3d_calibration.athinput"
OPACITY_TABLE = CASE_DIR / "ch_surrogate.opacity"
TABLE_GENERATOR = CASE_DIR / "generate_reference_tables.py"
MATERIAL_TABLE_DIR = CASE_DIR / "material_tables"
MATERIAL_TABLE_NAMES = (
    "ch.2t_eos",
    "he.2t_eos",
    "ch_20g.opacity",
    "he_20g.opacity",
)
MATERIAL_TABLE_MANIFEST = MATERIAL_TABLE_DIR / "manifest.json"
BUILD_DIR = CASE_DIR / "build"
BINARY = BUILD_DIR / "src" / "athena"
HELPER = Path(
    "/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh"
)
ENV_SCRIPT = Path("/home/mengqi/Research/bashrc_athenaK")
PROBLEM = "../../DCI_3D/dci_3d"
RUN_SENTINEL = ".athenak_dci_3d_run"
OUTPUT_BLOCKS = range(1, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "calibrate", "production"),
        default="validate",
        help=(
            "validate=compact nlim=0, smoke=compact nlim=2, "
            "calibrate=production mesh nlim=2, production=5+5 ns"
        ),
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--regenerate-material-tables",
        action="store_true",
        help=(
            "recreate audited CH/He EOS and opacity tables from the local "
            "3d_zb.zip (the provisional deck does not consume them yet)"
        ),
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("ATHENAK_TEST_GPUS", "0,1,2,3,4,5,6,7"),
        help="comma-separated physical CUDA device indices",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--nlim",
        type=int,
        choices=(0, 2),
        help="override nlim for non-production validation",
    )
    parser.add_argument(
        "--allow-busy-gpus",
        action="store_true",
        help="permit a non-idle baseline (never recommended for memory acceptance)",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def material_tables_are_valid() -> bool:
    try:
        manifest = json.loads(MATERIAL_TABLE_MANIFEST.read_text(encoding="utf-8"))
        records = manifest["tables"]
        if not isinstance(records, list):
            return False
        expected = set(MATERIAL_TABLE_NAMES)
        actual: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                return False
            name = record.get("output")
            expected_hash = record.get("output_sha256")
            if not isinstance(name, str) or not isinstance(expected_hash, str):
                return False
            path = MATERIAL_TABLE_DIR / name
            if not path.is_file() or sha256_path(path) != expected_hash:
                return False
            actual.add(name)
        return actual == expected
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_material_tables(force: bool, dry_run: bool) -> None:
    if not force and material_tables_are_valid():
        return
    command = [sys.executable, str(TABLE_GENERATOR), "--force"]
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=REPO, check=True)
    if not material_tables_are_valid():
        raise RuntimeError("Generated DCI material-table manifest failed verification")


def default_run_dir(mode: str) -> Path:
    if mode == "production":
        return CASE_DIR / "run"
    return CASE_DIR / "runs" / mode


def device_ids(raw: str, ranks: int) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != ranks or len(set(devices)) != ranks:
        raise RuntimeError(
            f"Expected {ranks} unique GPU IDs, received {devices}"
        )
    if ranks != 8:
        raise RuntimeError("DCI_3D acceptance requires exactly eight MPI ranks/GPUs")
    if any(not item.isdigit() for item in devices):
        raise RuntimeError(f"GPU IDs must be numeric physical indices: {devices}")
    return devices


def clean_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    sentinel = run_dir / RUN_SENTINEL
    if not sentinel.is_file():
        raise RuntimeError(
            f"Refusing to clean directory without {RUN_SENTINEL}: {run_dir}"
        )
    shutil.rmtree(run_dir)


def prepare_run_dir(run_dir: Path, mode: str) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        sentinel = run_dir / RUN_SENTINEL
        if not sentinel.is_file():
            raise RuntimeError(
                f"Refusing nonempty run directory without {RUN_SENTINEL}: {run_dir}"
            )
        if mode == "production":
            extras = [path for path in run_dir.iterdir() if path != sentinel]
            if extras:
                raise RuntimeError(
                    "Production run directory is not empty; inspect it and use "
                    "--clean only if its sentinel confirms ownership"
                )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_SENTINEL).write_text(
        "Owned by DCI_3D/run_case.py; safe for this launcher's --clean.\n"
    )


def build(jobs: int, dry_run: bool) -> None:
    command = [
        str(HELPER),
        "build",
        "--problem",
        PROBLEM,
        "--repo",
        str(REPO),
        "--build-dir",
        str(BUILD_DIR.relative_to(REPO)),
        "--jobs",
        str(jobs),
    ]
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=REPO, check=True)


def query_gpu(devices: list[str]) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for device in devices:
        command = [
            "nvidia-smi",
            "-i",
            device,
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        row = subprocess.run(
            command, check=True, text=True, capture_output=True
        ).stdout.strip()
        name, total, used = [part.strip() for part in row.split(",")]
        result[device] = {
            "name": name,
            "total_mib": float(total),
            "used_mib": float(used),
        }
    return result


def gpu_processes(device: str) -> list[str]:
    command = [
        "nvidia-smi",
        "-i",
        device,
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(
        command, check=True, text=True, capture_output=True
    ).stdout.strip()
    return [line for line in output.splitlines() if line.strip()]


class GpuMemoryMonitor:
    def __init__(self, devices: list[str], baseline: dict[str, dict[str, object]]):
        self.devices = devices
        self.baseline = baseline
        self.peak = {
            device: float(baseline[device]["used_mib"]) for device in devices
        }
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                current = query_gpu(self.devices)
                for device in self.devices:
                    self.peak[device] = max(
                        self.peak[device], float(current[device]["used_mib"])
                    )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.errors.append(str(exc))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        self._thread.join()
        records: dict[str, object] = {}
        for device in self.devices:
            total = float(self.baseline[device]["total_mib"])
            initial = float(self.baseline[device]["used_mib"])
            delta = max(0.0, self.peak[device]-initial)
            records[device] = {
                **self.baseline[device],
                "baseline_used_mib": initial,
                "peak_used_mib": self.peak[device],
                "peak_delta_mib": delta,
                "peak_fraction": delta/total,
                "within_60_80_percent": 0.60 <= delta/total <= 0.80,
            }
        return {"devices": records, "errors": self.errors}


def disabled_output_overrides() -> list[str]:
    return [f"output{number}/dt=-1.0" for number in OUTPUT_BLOCKS]


def nonproduction_overrides(mode: str, nlim: int | None) -> list[str]:
    selected_nlim = {"validate": 0, "smoke": 2, "calibrate": 2}[mode]
    if nlim is not None:
        selected_nlim = nlim
    overrides = [f"time/nlim={selected_nlim}", "time/tlim=1.0e-3"]
    if mode in ("validate", "smoke"):
        # Two blocks per direction: one production-size block per MPI rank.
        overrides.extend(("mesh/nx1=100", "mesh/nx2=92", "mesh/nx3=92"))
    overrides.extend(disabled_output_overrides())
    return overrides


def mpi_command(input_path: Path, overrides: list[str]) -> list[str]:
    return [
        "mpirun",
        "-n",
        "8",
        str(BINARY),
        "--kokkos-map-device-id-by=mpi_rank",
        "-i",
        str(input_path),
        f"thermal_radiation/opacity_table_file={OPACITY_TABLE}",
        *overrides,
    ]


def shell_command(run_dir: Path, mpi: list[str]) -> list[str]:
    body = (
        f"source {shlex.quote(str(ENV_SCRIPT))}; "
        f"cd {shlex.quote(str(run_dir))}; exec {shlex.join(mpi)}"
    )
    return ["bash", "-lc", body]


def run_logged(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    monitor: GpuMemoryMonitor,
) -> tuple[int, float, dict[str, object]]:
    start = time.monotonic()
    monitor.start()
    try:
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
    finally:
        memory = monitor.stop()
    return result.returncode, time.monotonic()-start, memory


def memory_is_accepted(memory: dict[str, object]) -> bool:
    records = memory["devices"]
    assert isinstance(records, dict)
    return all(
        bool(record["within_60_80_percent"])
        for record in records.values()
        if isinstance(record, dict)
    )


def print_dry_run(run_dir: Path, command: list[str], devices: list[str]) -> None:
    env = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    print(f"mkdir -p {shlex.quote(str(run_dir))}")
    print(
        f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
        + shlex.join(shell_command(run_dir, command))
    )


def main() -> int:
    args = parse_args()
    devices = device_ids(args.gpus, args.ranks)
    run_dir = (args.run_dir or default_run_dir(args.mode)).expanduser().resolve()

    if args.mode == "production" and args.nlim is not None:
        raise RuntimeError("--nlim is only valid for non-production modes")
    if args.mode == "production" and not args.dry_run:
        raise RuntimeError(
            "Production is disabled: AP transport is stable in a compact smoke run, "
            "but convergence, restart, final 20-group material physics, full-mesh "
            "memory, and 5/10 ns endpoints remain unverified. See DCI_3D/README.md."
        )
    if args.clean and args.dry_run:
        raise RuntimeError("Combine neither --clean nor destructive actions with --dry-run")

    if args.regenerate_material_tables:
        prepare_material_tables(True, args.dry_run)
    if args.build:
        build(args.jobs, args.dry_run)
    if not args.dry_run and not BINARY.is_file():
        raise RuntimeError(f"AthenaK executable not found; use --build: {BINARY}")

    input_path = (
        PRODUCTION_INPUT if args.mode == "production" else CALIBRATION_INPUT
    )
    first_overrides = (
        []
        if args.mode == "production"
        else nonproduction_overrides(args.mode, args.nlim)
    )
    first_mpi = mpi_command(input_path, first_overrides)
    if args.dry_run:
        print_dry_run(run_dir, first_mpi, devices)
        if args.mode == "production":
            print("# Phase 2 restarts the 5 ns checkpoint with time/tlim=10.0.")
        return 0

    if args.clean:
        clean_run_dir(run_dir)
    prepare_run_dir(run_dir, args.mode)

    baseline = query_gpu(devices)
    processes = {device: gpu_processes(device) for device in devices}
    busy = {device: rows for device, rows in processes.items() if rows}
    if busy and not args.allow_busy_gpus:
        raise RuntimeError(f"Refusing GPUs with existing compute processes: {busy}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    status: dict[str, object] = {
        "mode": args.mode,
        "ranks": args.ranks,
        "gpus": devices,
        "run_dir": str(run_dir),
        "baseline": baseline,
        "baseline_processes": processes,
    }

    first_monitor = GpuMemoryMonitor(devices, baseline)
    first_code, first_elapsed, first_memory = run_logged(
        shell_command(run_dir, first_mpi),
        run_dir / "phase1.log",
        env,
        first_monitor,
    )
    status.update(
        phase1_exit_code=first_code,
        phase1_elapsed_seconds=first_elapsed,
        phase1_memory=first_memory,
        phase1_mpi_command=first_mpi,
    )
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if first_code != 0:
        return first_code

    if args.mode == "calibrate" and not memory_is_accepted(first_memory):
        print(
            "Calibration completed, but at least one GPU was outside the required "
            "60-80% peak-memory band. See run_status.json."
        )
        return 2

    if args.mode != "production":
        return 0

    restarts = sorted((run_dir / "rst").glob("dci_3d.*.rst"))
    if not restarts:
        raise RuntimeError("Phase 1 completed without a DCI_3D restart checkpoint")
    restart = restarts[-1]
    second_mpi = [
        "mpirun",
        "-n",
        "8",
        str(BINARY),
        "--kokkos-map-device-id-by=mpi_rank",
        "-r",
        str(restart.relative_to(run_dir)),
        "time/tlim=10.0",
        "laser/beam0_end_time=4.999999999999",
        f"thermal_radiation/opacity_table_file={OPACITY_TABLE}",
    ]
    second_baseline = query_gpu(devices)
    second_monitor = GpuMemoryMonitor(devices, second_baseline)
    second_code, second_elapsed, second_memory = run_logged(
        shell_command(run_dir, second_mpi),
        run_dir / "phase2.log",
        env,
        second_monitor,
    )
    status.update(
        phase2_exit_code=second_code,
        phase2_elapsed_seconds=second_elapsed,
        phase2_memory=second_memory,
        phase2_mpi_command=second_mpi,
        restart=str(restart),
    )
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return second_code


if __name__ == "__main__":
    raise SystemExit(main())
