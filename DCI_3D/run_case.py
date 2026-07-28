#!/usr/bin/env python3
"""Build, validate, and gate the reference-informed DCI_3D case."""

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
TABLE_GENERATOR = CASE_DIR / "generate_reference_tables.py"
MATERIAL_TABLE_DIR = CASE_DIR / "material_tables"
MATERIAL_TABLE_MANIFEST = MATERIAL_TABLE_DIR / "manifest.json"
ARCHIVE_SHA256 = "952708009c9e3bc00dc645e11c9c0f804614def9c70cc999b78c92f16c8a96cf"
EXPECTED_MATERIAL_TABLES = {
    "ch.2t_eos": {
        "kind": "two_temperature_eos",
        "material": "CH",
        "sha256": "b29624877c7c90ed1d8c385bef6a7882b106dd8202bf0398301e2dee09faa0d8",
    },
    "he.2t_eos": {
        "kind": "two_temperature_eos",
        "material": "He",
        "sha256": "aae12f2dde296992ad630094e5755f7f52baa0816c678771e075f4848a9d63d0",
    },
    "ch_20g.opacity": {
        "kind": "opacity",
        "material": "CH",
        "sha256": "47ee4b8ab3e7f249e4b7108ab5efbaabeee71bb7dd88cdea59f4b4c64738f94d",
    },
    "he_20g.opacity": {
        "kind": "opacity",
        "material": "He",
        "sha256": "1e0daba15df1a23f5f558663e867dda181886ea53b0dbad119beb6fa1215f420",
    },
}
TABLE_INPUT_OVERRIDES = {
    "materials/material0_eos_table_file": "material_tables/ch.2t_eos",
    "materials/material1_eos_table_file": "material_tables/he.2t_eos",
    "materials/material0_opacity_table_file": "material_tables/ch_20g.opacity",
    "materials/material1_opacity_table_file": "material_tables/he_20g.opacity",
}
BUILD_DIR = CASE_DIR / "build"
BINARY = BUILD_DIR / "src" / "athena"
HELPER = Path(
    "/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh"
)
ENV_SCRIPT = Path("/home/mengqi/Research/bashrc_athenaK")
PROBLEM = "../../DCI_3D/dci_3d"
RUN_SENTINEL = ".athenak_dci_3d_run"
OUTPUT_BLOCKS = range(1, 12)
DEFAULT_PRODUCTION_GATE = CASE_DIR / "production_gate.json"
PRODUCTION_GATE_SCHEMA = 1
REQUIRED_PRODUCTION_CHECKS = (
    "compact_20group_50step",
    "compact_output_and_restart",
    "finite_nonnegative_3t",
    "causal_timestep_no_collapse",
    "laser_and_boundary_energy_closure",
    "ch_mass_conservation",
    "restart_continuity",
    "resolution_or_opacity_sensitivity",
    "reduced_light_speed_sensitivity",
    "gpu_memory_60_80_all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "calibrate", "production"),
        default="validate",
        help=(
            "validate=compact nlim=0, smoke=compact 50-step plus restart, "
            "calibrate=production mesh nlim=2, production=5+5 ns"
        ),
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--regenerate-material-tables",
        action="store_true",
        help=(
            "recreate audited CH/He EOS and opacity tables from local 3d_zb.zip"
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
        help="non-negative cycle-limit override for non-production validation",
    )
    parser.add_argument(
        "--production-gate",
        type=Path,
        default=DEFAULT_PRODUCTION_GATE,
        help="evidence manifest required for an actual production launch",
    )
    parser.add_argument(
        "--radiation-c-light",
        type=float,
        choices=(10.0, 30.0, 299.792458),
        help=(
            "non-production RSLA sensitivity value in code units; default smoke "
            "cycle count scales to approximately the same physical interval"
        ),
    )
    parser.add_argument(
        "--compact-scale",
        type=int,
        choices=(1, 2),
        default=1,
        help="linear compact-mesh scale for validate/smoke resolution checks",
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
        if manifest.get("archive_sha256") != ARCHIVE_SHA256:
            return False
        records = manifest["tables"]
        if not isinstance(records, list):
            return False
        expected = set(EXPECTED_MATERIAL_TABLES)
        actual: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                return False
            name = record.get("output")
            if not isinstance(name, str) or name not in EXPECTED_MATERIAL_TABLES:
                return False
            specification = EXPECTED_MATERIAL_TABLES[name]
            expected_hash = specification["sha256"]
            if (
                record.get("output_sha256") != expected_hash
                or record.get("kind") != specification["kind"]
                or record.get("material") != specification["material"]
            ):
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


def table_overrides() -> list[str]:
    return [f"{key}={value}" for key, value in TABLE_INPUT_OVERRIDES.items()]


def stage_material_tables(run_dir: Path) -> None:
    """Copy verified tables into the owned run tree for portable restart/transfer."""
    destination = run_dir / "material_tables"
    destination.mkdir(parents=True, exist_ok=True)
    for name, specification in EXPECTED_MATERIAL_TABLES.items():
        source = MATERIAL_TABLE_DIR / name
        target = destination / name
        shutil.copy2(source, target)
        if sha256_path(target) != specification["sha256"]:
            raise RuntimeError(f"Staged material table failed verification: {target}")
    shutil.copy2(MATERIAL_TABLE_MANIFEST, destination / "manifest.json")


def gate_artifact_hashes() -> dict[str, str]:
    artifacts = {
        "athena_binary": BINARY,
        "dci_3d.cpp": CASE_DIR / "dci_3d.cpp",
        "dci_3d.athinput": PRODUCTION_INPUT,
        "dci_3d_calibration.athinput": CALIBRATION_INPUT,
        "run_case.py": Path(__file__).resolve(),
    }
    result = {name: sha256_path(path) for name, path in artifacts.items()}
    for name, specification in EXPECTED_MATERIAL_TABLES.items():
        result[f"material_tables/{name}"] = str(specification["sha256"])
    return result


def validate_production_gate(path: Path) -> dict[str, object]:
    """Require immutable evidence for every acceptance gate before production."""
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Production gate is missing or invalid: {path}: {exc}") from exc
    if gate.get("schema") != PRODUCTION_GATE_SCHEMA:
        raise RuntimeError(
            f"Production gate schema must be {PRODUCTION_GATE_SCHEMA}: {path}"
        )
    expected_artifacts = gate_artifact_hashes()
    if gate.get("artifacts") != expected_artifacts:
        raise RuntimeError(
            "Production gate artifact hashes do not match this binary, case, launcher, "
            "and material tables"
        )
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_PRODUCTION_CHECKS):
        raise RuntimeError("Production gate does not contain the exact required check set")
    for name in REQUIRED_PRODUCTION_CHECKS:
        record = checks[name]
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise RuntimeError(f"Production gate has not passed: {name}")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise RuntimeError(f"Production gate lacks immutable evidence: {name}")
        for item in evidence:
            if not isinstance(item, dict):
                raise RuntimeError(f"Malformed production evidence for {name}")
            evidence_name = item.get("path")
            evidence_hash = item.get("sha256")
            if not isinstance(evidence_name, str) or not isinstance(evidence_hash, str):
                raise RuntimeError(f"Malformed production evidence for {name}")
            evidence_path = Path(evidence_name).expanduser()
            if not evidence_path.is_absolute():
                evidence_path = path.parent / evidence_path
            if not evidence_path.is_file() or sha256_path(evidence_path) != evidence_hash:
                raise RuntimeError(
                    f"Production evidence is missing or changed for {name}: {evidence_path}"
                )
    return gate


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


def default_smoke_cycles(c_light: float | None, compact_scale: int) -> int:
    value = 10.0 if c_light is None else c_light
    return max(50, int(round(50.0*value/10.0*compact_scale)))


def nonproduction_overrides(
    mode: str,
    nlim: int | None,
    radiation_c_light: float | None,
    compact_scale: int,
) -> list[str]:
    selected_nlim = {
        "validate": 0,
        "smoke": default_smoke_cycles(radiation_c_light, compact_scale),
        "calibrate": 2,
    }[mode]
    if nlim is not None:
        selected_nlim = nlim
    overrides = [f"time/nlim={selected_nlim}", "time/tlim=1.0"]
    if radiation_c_light is not None:
        overrides.append(f"thermal_radiation/c_light={radiation_c_light:.17g}")
    if mode in ("validate", "smoke"):
        # Scale 1 is two blocks per direction; scale 2 supplies a matched resolution gate.
        overrides.extend(
            (
                f"mesh/nx1={100*compact_scale}",
                f"mesh/nx2={64*compact_scale}",
                f"mesh/nx3={64*compact_scale}",
            )
        )
    overrides.extend(disabled_output_overrides())
    if mode == "smoke" and nlim is None:
        # The default smoke is an acceptance-gate run: cross both diagnostic and restart
        # boundaries, then run_case restarts it for ten additional cycles.
        overrides.extend(
            (
                "output1/dt=1.0e-4",
                "output3/dt=5.0e-4",
                "output4/dt=5.0e-4",
                "output11/dt=5.0e-4",
            )
        )
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
        *table_overrides(),
        *overrides,
    ]


def restart_command(restart: Path, overrides: list[str]) -> list[str]:
    return [
        "mpirun",
        "-n",
        "8",
        str(BINARY),
        "--kokkos-map-device-id-by=mpi_rank",
        "-r",
        str(restart),
        *table_overrides(),
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
    records = memory.get("devices")
    errors = memory.get("errors")
    if not isinstance(records, dict) or len(records) != 8 or errors:
        return False
    return all(
        isinstance(record, dict) and bool(record.get("within_60_80_percent"))
        for record in records.values()
    )


def print_dry_run(run_dir: Path, command: list[str], devices: list[str]) -> None:
    env = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    print(f"mkdir -p {shlex.quote(str(run_dir))}")
    print(
        "cp -a "
        f"{shlex.quote(str(MATERIAL_TABLE_DIR) + '/.')} "
        f"{shlex.quote(str(run_dir / 'material_tables') + '/')}"
    )
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
    if args.mode == "production" and args.radiation_c_light is not None:
        raise RuntimeError("--radiation-c-light is only valid for non-production modes")
    if args.mode in ("production", "calibrate") and args.compact_scale != 1:
        raise RuntimeError("--compact-scale applies only to validate and smoke modes")
    if args.nlim is not None and args.nlim < 0:
        raise RuntimeError("--nlim must be non-negative")
    if args.clean and args.dry_run:
        raise RuntimeError("Combine neither --clean nor destructive actions with --dry-run")

    prepare_material_tables(args.regenerate_material_tables, args.dry_run)
    if args.build:
        build(args.jobs, args.dry_run)
    if not args.dry_run and not BINARY.is_file():
        raise RuntimeError(f"AthenaK executable not found; use --build: {BINARY}")

    production_gate: dict[str, object] | None = None
    production_gate_path = args.production_gate.expanduser().resolve()
    if args.mode == "production" and not args.dry_run:
        production_gate = validate_production_gate(production_gate_path)

    input_path = (
        PRODUCTION_INPUT if args.mode == "production" else CALIBRATION_INPUT
    )
    first_overrides = (
        []
        if args.mode == "production"
        else nonproduction_overrides(
            args.mode,
            args.nlim,
            args.radiation_c_light,
            args.compact_scale,
        )
    )
    first_mpi = mpi_command(input_path, first_overrides)
    if args.dry_run:
        print_dry_run(run_dir, first_mpi, devices)
        if args.mode == "production":
            print(
                "# Actual production additionally requires a hash-matched evidence "
                f"manifest: {production_gate_path}"
            )
            print("# Phase 2 restarts the exact 5 ns checkpoint with time/tlim=10.0.")
        elif args.mode == "smoke" and args.nlim is None:
            print("# Phase 2 restarts the compact checkpoint for 10 more RK2 cycles.")
        return 0

    if args.clean:
        clean_run_dir(run_dir)
    prepare_run_dir(run_dir, args.mode)
    stage_material_tables(run_dir)

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
        "radiation_c_light_override": args.radiation_c_light,
        "compact_scale": args.compact_scale,
        "run_dir": str(run_dir),
        "case_artifacts": gate_artifact_hashes(),
        "material_manifest_sha256": sha256_path(MATERIAL_TABLE_MANIFEST),
        "baseline": baseline,
        "baseline_processes": processes,
    }
    if production_gate is not None:
        status["production_gate"] = str(production_gate_path)
        status["production_gate_sha256"] = sha256_path(production_gate_path)

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

    run_smoke_restart = args.mode == "smoke" and args.nlim is None
    if args.mode != "production" and not run_smoke_restart:
        return 0

    restart_basename = "dci_3d" if args.mode == "production" else "dci_3d_calibration"
    restarts = sorted((run_dir / "rst").glob(f"{restart_basename}.*.rst"))
    if not restarts:
        raise RuntimeError(
            f"Phase 1 completed without a {restart_basename} restart checkpoint"
        )
    restart = restarts[-1]
    if args.mode == "production":
        second_overrides = ["time/tlim=10.0"]
    else:
        first_cycles = default_smoke_cycles(
            args.radiation_c_light, args.compact_scale
        )
        extra_cycles = max(
            10,
            int(
                round(
                    10.0
                    * (args.radiation_c_light or 10.0)
                    / 10.0
                    * args.compact_scale
                )
            ),
        )
        second_overrides = [
            f"time/nlim={first_cycles+extra_cycles}",
            "time/tlim=1.0",
            *disabled_output_overrides(),
            "output1/dt=1.0e-4",
        ]
        if args.radiation_c_light is not None:
            second_overrides.append(
                f"thermal_radiation/c_light={args.radiation_c_light:.17g}"
            )
    second_mpi = restart_command(restart.relative_to(run_dir), second_overrides)
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
