#!/usr/bin/env python3
"""Build, validate, and gate the reference-informed DCI_3D case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Any


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
RUN_LOCK = ".run_case.lock"
RUN_STATUS = "run_status.json"
STAGED_DIR = "staged"
GPU_LOCK_ROOT = Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/tmp/athenak-{os.getuid()}")
) / "dci_3d-gpu-locks"
OUTPUT_BLOCKS = range(1, 12)
DEFAULT_PRODUCTION_GATE = CASE_DIR / "production_gate.json"
PRODUCTION_GATE_SCHEMA = 8
PRODUCTION_RADIATION_C_LIGHT = 30.0
CALIBRATION_CYCLES = 22
PRODUCTION_STATUS_SCHEMA = 2
PRODUCTION_PHASE1_TARGET = 5.0
PRODUCTION_FINAL_TARGET = 10.0
RESTART_HEADER_LIMIT = 40 * 1024
RESTART_PARAMETER_END = b"<par_end>\n"
# DCI production is built with Athena_SINGLE_PRECISION=OFF.  The fixed portion follows
# RestartOutput::WriteOutputFile and Mesh::BuildTreeFromRestart exactly.
RESTART_FIXED_HEADER = struct.Struct("=ii9d19i19iddi")
RESTART_LOCATION_BYTES = 4 * 4
RESTART_COST_BYTES = 4
PRODUCTION_MESH_SHAPE = (500, 256, 256)
PRODUCTION_BLOCK_SHAPE = (50, 32, 32)
PRODUCTION_MESHBLOCKS = 640
PRODUCTION_ROOT_LEVEL = 4
PRODUCTION_RESTART_DATA_SIZE = 17_397_504
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
    "physical_light_speed_sensitivity",
    "gpu_memory_60_80_all",
)


def parse_wall_time(value: str) -> str:
    """Validate Athena's `-t <hours>:<minutes>:<seconds>` syntax."""
    match = re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9])", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "wall time must use H+:MM:SS with two-digit minutes and seconds"
        )
    hours, minutes, seconds = (int(part) for part in match.groups())
    if hours == 0 and minutes == 0 and seconds == 0:
        raise argparse.ArgumentTypeError("wall time must be positive")
    if hours * 3600 + minutes * 60 + seconds > 2**31 - 1:
        raise argparse.ArgumentTypeError("wall time exceeds Athena's signed-int seconds")
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "smoke", "calibrate", "production"),
        default="validate",
        help=(
            "validate=compact nlim=0, smoke=compact 50-step plus restart, "
            f"calibrate=production mesh nlim={CALIBRATION_CYCLES}, "
            "production=5+5 ns"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a segmented production run from its newest verified restart",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "validate and stage a fresh production run without launching Athena; "
            "start it later with --resume"
        ),
    )
    parser.add_argument(
        "--segment-wall-time",
        type=parse_wall_time,
        metavar="H+:MM:SS",
        help=(
            "gracefully stop each production segment with Athena -t and continue "
            "until exactly 5 then 10 ns; intermediate boundaries atomically replace "
            "one rolling restart without advancing scheduled outputs"
        ),
    )
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
            "cycle count scales from the c_hat=30 production baseline"
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
    parser.add_argument(
        "--laser-max-reflections",
        type=int,
        help="non-production reflection-cap convergence override",
    )
    parser.add_argument(
        "--laser-reflection-offset",
        type=float,
        help="non-production turning-offset convergence override in cell widths",
    )
    parser.add_argument(
        "--laser-reflection-hysteresis",
        type=float,
        help="non-production fractional underdense rearm-band override",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON status file without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class RunLock:
    """Advisory lock held for the lifetime of one launcher process."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.stream: Any | None = None

    def __enter__(self) -> "RunLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / RUN_LOCK
        self.stream = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"Another launcher owns the run lock: {path}") from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class GpuReservation:
    """Cross-run advisory locks held for every selected physical GPU."""

    def __init__(self, devices: list[str], run_dir: Path):
        self.devices = sorted(devices, key=int)
        self.run_dir = run_dir
        self.streams: list[Any] = []

    def __enter__(self) -> "GpuReservation":
        GPU_LOCK_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        GPU_LOCK_ROOT.chmod(0o700)
        try:
            for device in self.devices:
                path = GPU_LOCK_ROOT / f"gpu-{device}.lock"
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    stream.seek(0)
                    owner = stream.read().strip() or "unknown owner"
                    stream.close()
                    raise RuntimeError(
                        f"GPU {device} is reserved by another launcher: {owner}"
                    ) from exc
                stream.seek(0)
                stream.truncate()
                stream.write(
                    f"pid={os.getpid()} run_dir={self.run_dir} started={utc_now()}\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
                self.streams.append(stream)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for stream in reversed(self.streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        self.streams.clear()


class StatusHeartbeat:
    """Atomically publish launcher/child liveness while a segment is running."""

    def __init__(
        self,
        path: Path,
        status: dict[str, Any],
        segment: dict[str, Any],
        interval_seconds: float = 30.0,
    ):
        self.path = path
        self.status = status
        self.segment = segment
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def _publish(self, child_running: bool) -> None:
        now = utc_now()
        self.segment["heartbeat_at"] = now
        self.segment["child_running"] = child_running
        self.status["heartbeat_at"] = now
        self.status["launcher_pid"] = os.getpid()
        atomic_write_json(self.path, self.status)

    def _run(self) -> None:
        try:
            while not self.stop_event.wait(self.interval_seconds):
                self._publish(True)
        except BaseException as exc:
            self.error = exc

    def start(self, child_pid: int) -> None:
        self.segment["child_pid"] = child_pid
        self._publish(True)
        self.thread = threading.Thread(
            target=self._run,
            name=f"dci-status-heartbeat-{child_pid}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._publish(False)
        if self.error is not None:
            raise RuntimeError(f"Production status heartbeat failed: {self.error}")


class RestartValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RestartInfo:
    path: Path
    kind: str
    file_number: int | None
    meshblocks: int
    root_level: int
    mesh_shape: tuple[int, int, int]
    block_shape: tuple[int, int, int]
    data_size: int
    time: float
    dt: float
    cycle: int
    size: int
    sha256: str | None = None


def restart_parameter_value(parameter_text: str, block: str, key: str) -> str:
    section = re.search(
        rf"(?ms)^<{re.escape(block)}>\s*$\n(.*?)(?=^<|\Z)", parameter_text
    )
    if section is None:
        raise RestartValidationError(f"Restart parameter block is missing: <{block}>")
    for raw_line in section.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name == key:
            return value
    raise RestartValidationError(f"Restart parameter is missing: {block}/{key}")


def inspect_restart(
    path: Path, basename: str, expected_run_id: str | None = None
) -> RestartInfo:
    """Read and structurally validate one DCI double-precision restart."""
    numbered_match = re.fullmatch(
        rf"{re.escape(basename)}\.([0-9]{{5}})\.rst", path.name
    )
    walltime_name = f"{basename}.walltime.rst"
    if numbered_match is None and path.name != walltime_name:
        raise RestartValidationError(f"Unexpected restart filename: {path.name}")
    kind = "numbered" if numbered_match is not None else "walltime"
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(RESTART_HEADER_LIMIT)
    except OSError as exc:
        raise RestartValidationError(f"Cannot read restart {path}: {exc}") from exc

    parameter_end = prefix.find(RESTART_PARAMETER_END)
    if parameter_end < 0:
        raise RestartValidationError(f"Restart has no complete <par_end> header: {path}")
    header_end = parameter_end + len(RESTART_PARAMETER_END)
    fixed_end = header_end + RESTART_FIXED_HEADER.size
    if len(prefix) < fixed_end:
        raise RestartValidationError(f"Restart fixed header is truncated: {path}")
    try:
        parameter_text = prefix[:parameter_end].decode("ascii")
        values = RESTART_FIXED_HEADER.unpack_from(prefix, header_end)
    except (UnicodeDecodeError, struct.error) as exc:
        raise RestartValidationError(f"Restart header cannot be decoded: {path}") from exc
    if restart_parameter_value(parameter_text, "job", "basename") != basename:
        raise RestartValidationError(
            f"Restart parameter dump has the wrong basename: {path}"
        )
    if expected_run_id is not None:
        try:
            restart_run_id = restart_parameter_value(parameter_text, "job", "run_id")
        except RestartValidationError as exc:
            raise RestartValidationError(
                f"Restart has no production run identity: {path}"
            ) from exc
        if restart_run_id != expected_run_id:
            raise RestartValidationError(
                f"Restart belongs to a different production run: {path}"
            )
    if restart_parameter_value(parameter_text, "output11", "file_type") != "rst":
        raise RestartValidationError(f"Restart output11 is not a restart writer: {path}")
    try:
        next_file_number = int(
            restart_parameter_value(parameter_text, "output11", "file_number")
        )
        single_file_per_rank = int(
            restart_parameter_value(
                parameter_text, "output11", "single_file_per_rank"
            )
        )
    except ValueError as exc:
        raise RestartValidationError(f"Restart output11 metadata is invalid: {path}") from exc
    file_number = int(numbered_match.group(1)) if numbered_match is not None else None
    counter_is_valid = (
        next_file_number == file_number + 1
        if file_number is not None else next_file_number >= 0
    )
    if not counter_is_valid or single_file_per_rank != 0:
        raise RestartValidationError(f"Restart output11 counter/layout is invalid: {path}")

    meshblocks = int(values[0])
    root_level = int(values[1])
    mesh_indices = values[11:30]
    block_indices = values[30:49]
    restart_time = float(values[-3])
    restart_dt = float(values[-2])
    cycle = int(values[-1])
    if meshblocks <= 0 or root_level < 0:
        raise RestartValidationError(f"Restart mesh header is invalid: {path}")
    mesh_shape = tuple(int(mesh_indices[index]) for index in (1, 2, 3))
    block_shape = tuple(int(block_indices[index]) for index in (1, 2, 3))
    if any(value <= 0 for value in (*mesh_shape, *block_shape)):
        raise RestartValidationError(f"Restart mesh dimensions are invalid: {path}")
    if any(mesh % block != 0 for mesh, block in zip(mesh_shape, block_shape)):
        raise RestartValidationError(f"Restart MeshBlocks do not tile its mesh: {path}")
    expected_meshblocks = math.prod(
        mesh // block for mesh, block in zip(mesh_shape, block_shape)
    )
    if meshblocks != expected_meshblocks:
        raise RestartValidationError(
            f"Restart MeshBlock count is inconsistent: {path}"
        )
    expected_root_level = math.ceil(math.log2(max(
        mesh // block for mesh, block in zip(mesh_shape, block_shape)
    )))
    if root_level != expected_root_level:
        raise RestartValidationError(f"Restart root level is inconsistent: {path}")
    if (
        not math.isfinite(restart_time)
        or restart_time < 0.0
        or not math.isfinite(restart_dt)
        or restart_dt <= 0.0
        or cycle < 0
    ):
        raise RestartValidationError(f"Restart time/cycle header is invalid: {path}")

    data_size_offset = (
        fixed_end
        + meshblocks * (RESTART_LOCATION_BYTES + RESTART_COST_BYTES)
    )
    try:
        with path.open("rb") as stream:
            stream.seek(data_size_offset)
            raw_data_size = stream.read(8)
        if len(raw_data_size) != 8:
            raise RestartValidationError(f"Restart data-size field is truncated: {path}")
        data_size = struct.unpack("=Q", raw_data_size)[0]
    except OSError as exc:
        raise RestartValidationError(f"Cannot inspect restart payload: {path}") from exc
    expected_size = data_size_offset + 8 + meshblocks * data_size
    if data_size <= 0 or file_size != expected_size:
        raise RestartValidationError(
            f"Restart payload is incomplete: {path} has {file_size} bytes, "
            f"expected {expected_size}"
        )

    locations_offset = fixed_end
    try:
        with path.open("rb") as stream:
            stream.seek(locations_offset)
            raw_locations = stream.read(meshblocks * RESTART_LOCATION_BYTES)
            raw_costs = stream.read(meshblocks * RESTART_COST_BYTES)
    except OSError as exc:
        raise RestartValidationError(
            f"Cannot inspect restart locations/load-balance costs: {path}"
        ) from exc
    if len(raw_locations) != meshblocks * RESTART_LOCATION_BYTES:
        raise RestartValidationError(f"Restart logical-location list is truncated: {path}")
    if len(raw_costs) != meshblocks * RESTART_COST_BYTES:
        raise RestartValidationError(f"Restart load-balance cost list is truncated: {path}")
    costs = struct.unpack(f"={meshblocks}f", raw_costs)
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in costs):
        raise RestartValidationError(f"Restart load-balance costs are invalid: {path}")
    locations = {
        struct.unpack_from("=4i", raw_locations, index * RESTART_LOCATION_BYTES)
        for index in range(meshblocks)
    }
    blocks_per_axis = tuple(
        mesh // block for mesh, block in zip(mesh_shape, block_shape)
    )
    expected_locations = {
        (x1, x2, x3, root_level)
        for x3 in range(blocks_per_axis[2])
        for x2 in range(blocks_per_axis[1])
        for x1 in range(blocks_per_axis[0])
    }
    if locations != expected_locations:
        raise RestartValidationError(f"Restart logical-location set is invalid: {path}")

    return RestartInfo(
        path=path,
        kind=kind,
        file_number=file_number,
        meshblocks=meshblocks,
        root_level=root_level,
        mesh_shape=mesh_shape,
        block_shape=block_shape,
        data_size=data_size,
        time=restart_time,
        dt=restart_dt,
        cycle=cycle,
        size=file_size,
    )


def restart_record(info: RestartInfo, run_dir: Path) -> dict[str, Any]:
    try:
        relative = info.path.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Restart is outside its owned run tree: {info.path}") from exc
    return {
        "path": str(relative),
        "kind": info.kind,
        "file_number": info.file_number,
        "meshblocks": info.meshblocks,
        "root_level": info.root_level,
        "mesh_shape": list(info.mesh_shape),
        "block_shape": list(info.block_shape),
        "data_size": info.data_size,
        "time": info.time,
        "dt": info.dt,
        "cycle": info.cycle,
        "size": info.size,
        "sha256": info.sha256,
    }


def validate_production_restart_layout(info: RestartInfo) -> None:
    expected = (
        PRODUCTION_MESHBLOCKS,
        PRODUCTION_ROOT_LEVEL,
        PRODUCTION_MESH_SHAPE,
        PRODUCTION_BLOCK_SHAPE,
        PRODUCTION_RESTART_DATA_SIZE,
    )
    actual = (
        info.meshblocks,
        info.root_level,
        info.mesh_shape,
        info.block_shape,
        info.data_size,
    )
    if actual != expected:
        raise RestartValidationError(
            f"Restart does not match the gated production layout: {info.path}"
        )


def stable_sha256(path: Path, expected_size: int) -> str:
    before = path.stat()
    if before.st_size != expected_size:
        raise RestartValidationError(f"Restart changed size before hashing: {path}")
    digest = sha256_path(path)
    after = path.stat()
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
    ):
        raise RestartValidationError(f"Restart changed while hashing: {path}")
    return digest


def recorded_restart_hashes(status: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in status.get("segments", []):
        if not isinstance(segment, dict) or segment.get("state") == "superseded":
            continue
        restart = segment.get("restart")
        if not isinstance(restart, dict):
            continue
        path = restart.get("path")
        digest = restart.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            result[Path(path).name] = digest
    checkpoint = status.get("phase1_checkpoint")
    if isinstance(checkpoint, dict):
        path = checkpoint.get("path")
        digest = checkpoint.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            result[Path(path).name] = digest
    return result


def supersede_restart_lineage(
    status: dict[str, Any], selected: RestartInfo
) -> None:
    """Retire recorded checkpoints that are later than a rollback point."""
    selected_progress = (selected.time, selected.cycle)
    now = utc_now()
    for segment in status.get("segments", []):
        if not isinstance(segment, dict):
            continue
        record = segment.get("restart")
        if not isinstance(record, dict):
            continue
        try:
            progress = (float(record["time"]), int(record["cycle"]))
        except (KeyError, TypeError, ValueError):
            continue
        if progress > selected_progress:
            segment["state"] = "superseded"
            segment["superseded_at"] = now
            segment["superseded_by_rollback"] = {
                "time": selected.time,
                "cycle": selected.cycle,
            }
    if selected.time < PRODUCTION_PHASE1_TARGET:
        status.pop("phase1_checkpoint", None)


def select_valid_restart(
    run_dir: Path,
    basename: str,
    minimum_time: float = 0.0,
    expected_hashes: dict[str, str] | None = None,
    require_production_layout: bool = False,
    expected_run_id: str | None = None,
) -> tuple[RestartInfo, list[str]]:
    """Select by embedded time/cycle across numbered and rolling checkpoints."""
    expected_hashes = expected_hashes or {}
    restart_dir = run_dir / "rst"
    candidates = list(
        restart_dir.glob(f"{basename}.[0-9][0-9][0-9][0-9][0-9].rst")
    )
    walltime = restart_dir / f"{basename}.walltime.rst"
    if walltime.is_file() or walltime.is_symlink():
        candidates.append(walltime)
    rejected: list[str] = []
    valid: list[RestartInfo] = []
    for candidate in candidates:
        try:
            if candidate.is_symlink():
                raise RestartValidationError("restart must not be a symlink")
            try:
                candidate.resolve().relative_to(restart_dir.resolve())
            except ValueError as exc:
                raise RestartValidationError("restart escapes its owned directory") from exc
            info = inspect_restart(candidate, basename, expected_run_id)
            if require_production_layout:
                validate_production_restart_layout(info)
            if info.time < minimum_time:
                raise RestartValidationError(
                    f"restart time {info.time} precedes recorded time {minimum_time}"
                )
            valid.append(info)
        except (OSError, RestartValidationError) as exc:
            rejected.append(f"{candidate.name}: {exc}")
    valid.sort(
        key=lambda info: (
            info.time,
            info.cycle,
            1 if info.kind == "numbered" else 0,
            -1 if info.file_number is None else info.file_number,
        ),
        reverse=True,
    )
    for info in valid:
        try:
            digest = stable_sha256(info.path, info.size)
            expected = expected_hashes.get(info.path.name)
            if expected is not None and digest != expected:
                raise RestartValidationError("recorded SHA-256 does not match")
            return RestartInfo(**{**info.__dict__, "sha256": digest}), rejected
        except (OSError, RestartValidationError) as exc:
            rejected.append(f"{info.path.name}: {exc}")
    detail = "; ".join(rejected) if rejected else "no restart files found"
    raise RuntimeError(f"No valid {basename} restart in {restart_dir}: {detail}")


def production_phase_for_time(current_time: float) -> tuple[int, float] | None:
    """Map a verified restart time to the exact remaining production phase."""
    if current_time < 0.0 or current_time > PRODUCTION_FINAL_TARGET:
        raise RuntimeError(f"Production restart time is outside [0, 10] ns: {current_time}")
    # Never treat an undershoot as a completed phase.  Mesh::NewTimeStep clips the final
    # dt to tlim, and 5.0/10.0 are exactly representable, so target restarts must be exact.
    if current_time < PRODUCTION_PHASE1_TARGET:
        return (1, PRODUCTION_PHASE1_TARGET)
    if current_time < PRODUCTION_FINAL_TARGET:
        return (2, PRODUCTION_FINAL_TARGET)
    return None


def stage_production_binary(
    run_dir: Path, source: Path, expected_sha256: str
) -> Path:
    """Create a content-addressed, read-only executable inside the run tree."""
    if sha256_path(source) != expected_sha256:
        raise RuntimeError("Source Athena executable no longer matches the production gate")
    directory = run_dir / STAGED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"athena-{expected_sha256}"
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_path(destination) != expected_sha256
        ):
            raise RuntimeError(f"Staged Athena executable is corrupt: {destination}")
        destination.chmod(0o555)
        return destination
    temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copy2(source, temporary)
        temporary.chmod(0o555)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if sha256_path(temporary) != expected_sha256:
            raise RuntimeError("Staged Athena executable failed hash verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def stage_production_input(
    run_dir: Path, source: Path, expected_sha256: str, run_id: str
) -> Path:
    """Stage a gate-hashed deck with a unique identity preserved in restarts."""
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise RuntimeError("Production run identity must be 128-bit lowercase hex")
    try:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Cannot stage production deck: {source}: {exc}") from exc
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Source production deck no longer matches the production gate")
    if re.search(r"(?m)^\s*run_id\s*=", source_text):
        raise RuntimeError("Gate-hashed production deck must not define job/run_id")
    marker = "<job>\n"
    if source_text.count(marker) != 1:
        raise RuntimeError("Production deck must contain exactly one <job> block")
    staged_bytes = source_text.replace(
        marker, f"{marker}run_id = {run_id}\n", 1
    ).encode("utf-8")
    staged_sha256 = hashlib.sha256(staged_bytes).hexdigest()
    directory = run_dir / STAGED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"dci_3d-{staged_sha256}.athinput"
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_path(destination) != staged_sha256
        ):
            raise RuntimeError(f"Staged production deck is corrupt: {destination}")
        destination.chmod(0o444)
        return destination
    temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as stream:
            stream.write(staged_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        if sha256_path(temporary) != staged_sha256:
            raise RuntimeError("Staged production deck failed hash verification")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def production_input_run_id(path: Path) -> str:
    try:
        value = restart_parameter_value(
            path.read_text(encoding="utf-8"), "job", "run_id"
        )
    except (OSError, UnicodeDecodeError, RestartValidationError) as exc:
        raise RuntimeError(f"Staged production deck has no run identity: {path}") from exc
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise RuntimeError(f"Staged production deck has an invalid run identity: {path}")
    return value


def validate_staged_material_tables(
    run_dir: Path, expected_manifest_sha256: str | None = None
) -> None:
    for name, specification in EXPECTED_MATERIAL_TABLES.items():
        path = run_dir / "material_tables" / name
        if not path.is_file() or sha256_path(path) != specification["sha256"]:
            raise RuntimeError(f"Staged material table is missing or changed: {path}")
    staged_manifest = run_dir / "material_tables" / "manifest.json"
    expected_manifest = (
        sha256_path(MATERIAL_TABLE_MANIFEST)
        if expected_manifest_sha256 is None
        else expected_manifest_sha256
    )
    if not staged_manifest.is_file() or sha256_path(staged_manifest) != expected_manifest:
        raise RuntimeError(f"Staged material manifest is missing or changed: {staged_manifest}")


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
        target.chmod(0o444)
    staged_manifest = destination / "manifest.json"
    shutil.copy2(MATERIAL_TABLE_MANIFEST, staged_manifest)
    staged_manifest.chmod(0o444)


def gate_artifact_hashes(binary_path: Path = BINARY) -> dict[str, str]:
    artifacts = {
        "athena_binary": binary_path,
        "dci_3d.cpp": CASE_DIR / "dci_3d.cpp",
        "dci_3d.athinput": PRODUCTION_INPUT,
        "dci_3d_calibration.athinput": CALIBRATION_INPUT,
        "run_case.py": Path(__file__).resolve(),
        "verify_production_gate.py": CASE_DIR / "verify_production_gate.py",
    }
    result = {name: sha256_path(path) for name, path in artifacts.items()}
    for name, specification in EXPECTED_MATERIAL_TABLES.items():
        result[f"material_tables/{name}"] = str(specification["sha256"])
    return result


def validate_production_gate(
    path: Path, binary_path: Path = BINARY
) -> dict[str, object]:
    """Require immutable evidence for every acceptance gate before production."""
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Production gate is missing or invalid: {path}: {exc}") from exc
    if gate.get("schema") != PRODUCTION_GATE_SCHEMA:
        raise RuntimeError(
            f"Production gate schema must be {PRODUCTION_GATE_SCHEMA}: {path}"
        )
    expected_artifacts = gate_artifact_hashes(binary_path)
    if gate.get("artifacts") != expected_artifacts:
        raise RuntimeError(
            "Production gate artifact hashes do not match this binary, case, launcher, "
            "and material tables"
        )
    generator = gate.get("generator")
    verifier = CASE_DIR / "verify_production_gate.py"
    if not isinstance(generator, dict) or generator.get("sha256") != sha256_path(verifier):
        raise RuntimeError("Production gate was not generated by this verifier")
    sources = gate.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Production gate has no immutable source manifest")
    source_paths: dict[str, Path] = {}
    for name, record in sources.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Malformed production source: {name}")
        source_name = record.get("path")
        source_hash = record.get("sha256")
        if not isinstance(source_name, str) or not isinstance(source_hash, str):
            raise RuntimeError(f"Malformed production source: {name}")
        source_path = Path(source_name).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent/source_path
        if not source_path.is_file() or sha256_path(source_path) != source_hash:
            raise RuntimeError(f"Production source is missing or changed: {source_path}")
        source_paths[name] = source_path
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_PRODUCTION_CHECKS):
        raise RuntimeError("Production gate does not contain the exact required check set")
    for name in REQUIRED_PRODUCTION_CHECKS:
        record = checks[name]
        if not isinstance(record, dict) or record.get("passed") is not True:
            raise RuntimeError(f"Production gate has not passed: {name}")
        source_ids = record.get("source_ids")
        if (not isinstance(source_ids, list) or not source_ids or
                any(source_id not in source_paths for source_id in source_ids)):
            raise RuntimeError(f"Production gate lacks immutable evidence: {name}")
    sys.path.insert(0, str(CASE_DIR))
    import verify_production_gate
    recomputed = verify_production_gate.recompute_checks_from_gate(gate, path)
    if recomputed != checks:
        raise RuntimeError(
            "Production gate metrics no longer reproduce from their immutable evidence"
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
    owned_contents = [path for path in run_dir.iterdir() if path.name != RUN_LOCK]
    if not owned_contents:
        return
    sentinel = run_dir / RUN_SENTINEL
    if not sentinel.is_file():
        raise RuntimeError(
            f"Refusing to clean directory without {RUN_SENTINEL}: {run_dir}"
        )
    # The caller holds RUN_LOCK.  Preserve that inode while removing owned contents so a
    # second launcher cannot race into a newly-created directory during --clean.
    for path in run_dir.iterdir():
        if path.name == RUN_LOCK:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def prepare_run_dir(run_dir: Path, mode: str) -> None:
    contents = (
        [path for path in run_dir.iterdir() if path.name != RUN_LOCK]
        if run_dir.exists() else []
    )
    if contents:
        sentinel = run_dir / RUN_SENTINEL
        if not sentinel.is_file():
            raise RuntimeError(
                f"Refusing nonempty run directory without {RUN_SENTINEL}: {run_dir}"
            )
        if mode == "production":
            extras = [
                path for path in contents if path.name != RUN_SENTINEL
            ]
            if extras:
                raise RuntimeError(
                    "Production run directory is not empty; inspect it and use "
                    "--clean only if its sentinel confirms ownership"
                )
    run_dir.mkdir(parents=True, exist_ok=True)
    # TranFile provisions each isolated remote lineage with mode 0700.  Keep the
    # source root identical so the final archive checksum has no permission-only
    # delta, independent of the interactive shell's umask.
    if mode == "production":
        run_dir.chmod(0o700)
    (run_dir / RUN_SENTINEL).write_text(
        "Owned by DCI_3D/run_case.py; safe for this launcher's --clean.\n"
    )


def validate_owned_resume_tree(run_dir: Path) -> None:
    if not run_dir.is_dir() or not (run_dir / RUN_SENTINEL).is_file():
        raise RuntimeError(
            f"--resume requires an owned tree containing {RUN_SENTINEL}: {run_dir}"
        )
    if not (run_dir / RUN_STATUS).is_file():
        raise RuntimeError(f"--resume requires an atomic production status: {run_dir/RUN_STATUS}")


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


def production_output_cadences(path: Path = PRODUCTION_INPUT) -> dict[str, float]:
    """Read every production output cadence from the gate-hashed deck."""
    result: dict[str, float] = {}
    block: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        section = re.fullmatch(r"<(output[0-9]+)>", line)
        if section is not None:
            block = section.group(1)
            continue
        if line.startswith("<"):
            block = None
            continue
        if block is None or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "dt":
            result[block] = float(value)
    expected = {f"output{number}" for number in OUTPUT_BLOCKS}
    if set(result) != expected or any(value <= 0.0 for value in result.values()):
        raise RuntimeError("Production deck must define a positive cadence for outputs 1-11")
    return result


def default_smoke_cycles(c_light: float | None, compact_scale: int) -> int:
    value = PRODUCTION_RADIATION_C_LIGHT if c_light is None else c_light
    # c_hat=10 and 30 are not transport-CFL limited in the measured compact case,
    # while the physical-c comparison is.  Never shorten the 50-cycle baseline;
    # scale only faster-light comparisons and doubled spatial resolution upward.
    speed_scale = max(1.0, value/PRODUCTION_RADIATION_C_LIGHT)
    return int(round(50.0*speed_scale*compact_scale))


def nonproduction_overrides(
    mode: str,
    nlim: int | None,
    radiation_c_light: float | None,
    compact_scale: int,
    laser_max_reflections: int | None = None,
    laser_reflection_offset: float | None = None,
    laser_reflection_hysteresis: float | None = None,
) -> list[str]:
    selected_nlim = {
        "validate": 0,
        "smoke": default_smoke_cycles(radiation_c_light, compact_scale),
        "calibrate": CALIBRATION_CYCLES,
    }[mode]
    if nlim is not None:
        selected_nlim = nlim
    overrides = [
        f"time/nlim={selected_nlim}",
        "time/tlim=1.0",
        "problem/allow_laser_transport_variants=true",
    ]
    if laser_max_reflections is not None:
        overrides.append(
            f"laser/max_reflections_per_ray={laser_max_reflections}"
        )
    if laser_reflection_offset is not None:
        overrides.append(
            f"laser/reflection_offset_fraction={laser_reflection_offset:.17g}"
        )
    if laser_reflection_hysteresis is not None:
        overrides.append(
            "laser/reflection_hysteresis_fraction="
            f"{laser_reflection_hysteresis:.17g}"
        )
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
    if mode == "smoke":
        # Explicit-cycle diagnostic runs still need history evidence. The default smoke
        # additionally crosses volume/restart boundaries and launches phase two.
        overrides.append("output1/dt=1.0e-4")
        if nlim is None:
            overrides.extend(
                (
                "output3/dt=5.0e-4",
                "output4/dt=5.0e-4",
                "output8/dt=1.0",
                "output9/dt=1.0",
                "output10/dt=1.0",
                "output11/dt=5.0e-4",
                )
            )
    return overrides


def smoke_restart_overrides(
    radiation_c_light: float | None,
    compact_scale: int,
    laser_max_reflections: int | None = None,
    laser_reflection_offset: float | None = None,
    laser_reflection_hysteresis: float | None = None,
) -> list[str]:
    first_cycles = default_smoke_cycles(radiation_c_light, compact_scale)
    value = (
        PRODUCTION_RADIATION_C_LIGHT
        if radiation_c_light is None else radiation_c_light
    )
    extra_cycles = int(round(
        10.0*max(1.0, value/PRODUCTION_RADIATION_C_LIGHT)*compact_scale
    ))
    overrides = [
        f"time/nlim={first_cycles+extra_cycles}",
        "time/tlim=1.0",
        *disabled_output_overrides(),
        "output1/dt=1.0e-4",
        # Construct one 3T output object so Driver::Finalize writes a terminal,
        # post-restart full volume for cell-by-cell gate validation.
        "output9/dt=1.0",
    ]
    if radiation_c_light is not None:
        overrides.append(
            f"thermal_radiation/c_light={radiation_c_light:.17g}"
        )
    if laser_max_reflections is not None:
        overrides.append(
            f"laser/max_reflections_per_ray={laser_max_reflections}"
        )
    if laser_reflection_offset is not None:
        overrides.append(
            f"laser/reflection_offset_fraction={laser_reflection_offset:.17g}"
        )
    if laser_reflection_hysteresis is not None:
        overrides.append(
            "laser/reflection_hysteresis_fraction="
            f"{laser_reflection_hysteresis:.17g}"
        )
    return overrides


def athena_mpi_prefix(
    binary_path: Path = BINARY, wall_time: str | None = None
) -> list[str]:
    command = [
        "mpirun",
        "-n",
        "8",
        str(binary_path),
    ]
    if wall_time is not None:
        command.extend(("-t", wall_time))
    command.extend((
        "--kokkos-map-device-id-by=mpi_rank",
    ))
    return command


def mpi_command(
    input_path: Path,
    overrides: list[str],
    binary_path: Path = BINARY,
    wall_time: str | None = None,
) -> list[str]:
    return [
        *athena_mpi_prefix(binary_path, wall_time),
        "-i",
        str(input_path),
        *table_overrides(),
        *overrides,
    ]


def restart_command(
    restart: Path,
    overrides: list[str],
    binary_path: Path = BINARY,
    wall_time: str | None = None,
) -> list[str]:
    return [
        *athena_mpi_prefix(binary_path, wall_time),
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
    monitor: GpuMemoryMonitor | None,
    heartbeat: StatusHeartbeat | None = None,
) -> tuple[int, float, dict[str, object] | None]:
    start = time.monotonic()
    if monitor is not None:
        monitor.start()
    process: subprocess.Popen[str] | None = None
    heartbeat_started = False
    try:
        with log_path.open("w") as log:
            log.write("command=" + shlex.join(command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if heartbeat is not None:
                    heartbeat.start(process.pid)
                    heartbeat_started = True
                exit_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    finally:
        heartbeat_error: BaseException | None = None
        if heartbeat_started and heartbeat is not None:
            try:
                heartbeat.stop()
            except BaseException as exc:
                heartbeat_error = exc
        memory = monitor.stop() if monitor is not None else None
        if heartbeat_error is not None:
            raise heartbeat_error
    return exit_code, time.monotonic()-start, memory


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


def gpu_preflight(
    devices: list[str], allow_busy: bool
) -> tuple[dict[str, dict[str, float | str]], dict[str, list[str]]]:
    baseline = query_gpu(devices)
    processes = {device: gpu_processes(device) for device in devices}
    busy = {device: rows for device, rows in processes.items() if rows}
    if busy and not allow_busy:
        raise RuntimeError(f"Refusing GPUs with existing compute processes: {busy}")
    return baseline, processes


def load_production_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_STATUS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Production status is missing or invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status_schema") != PRODUCTION_STATUS_SCHEMA:
        raise RuntimeError(
            f"Production status schema must be {PRODUCTION_STATUS_SCHEMA}: {path}"
        )
    if payload.get("mode") != "production" or payload.get("run_dir") != str(run_dir):
        raise RuntimeError(f"Production status does not own this run tree: {path}")
    if not isinstance(payload.get("segments"), list):
        raise RuntimeError(f"Production status has no segment history: {path}")
    return payload


def is_prepared_without_restart(run_dir: Path, status: dict[str, Any]) -> bool:
    restart_dir = run_dir / "rst"
    has_restart = restart_dir.is_dir() and (
        any(restart_dir.glob("dci_3d.[0-9][0-9][0-9][0-9][0-9].rst"))
        or (restart_dir / "dci_3d.walltime.rst").exists()
    )
    return (
        status.get("state") == "prepared"
        and not status.get("segments")
        and float(status.get("current_time", 0.0)) == 0.0
        and int(status.get("current_cycle", 0)) == 0
        and not has_restart
    )


def status_restart_info(
    run_dir: Path,
    record: dict[str, Any],
    expected_time: float,
    expected_run_id: str,
) -> RestartInfo:
    path_value = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise RuntimeError("Recorded production checkpoint is malformed")
    path = (run_dir / path_value).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Recorded restart escapes the run tree: {path}") from exc
    info = inspect_restart(path, "dci_3d", expected_run_id)
    validate_production_restart_layout(info)
    if info.time != expected_time:
        raise RuntimeError(
            f"Recorded checkpoint is at {info.time}, expected exactly {expected_time}"
        )
    actual_digest = stable_sha256(path, info.size)
    if actual_digest != digest:
        raise RuntimeError(f"Recorded checkpoint hash changed: {path}")
    return RestartInfo(**{**info.__dict__, "sha256": actual_digest})


def staged_binary_from_status(run_dir: Path, status: dict[str, Any]) -> Path:
    path_value = status.get("staged_binary")
    expected = status.get("staged_binary_sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise RuntimeError("Production status has no staged binary provenance")
    unresolved = run_dir / path_value
    if unresolved.is_symlink():
        raise RuntimeError(f"Staged Athena executable must not be a symlink: {unresolved}")
    path = unresolved.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Staged executable escapes the run tree: {path}") from exc
    mode = path.stat().st_mode if path.is_file() else 0
    if (
        not path.is_file()
        or mode & 0o222
        or not mode & 0o111
        or sha256_path(path) != expected
    ):
        raise RuntimeError(f"Staged Athena executable is missing or changed: {path}")
    return path


def staged_input_from_status(run_dir: Path, status: dict[str, Any]) -> Path:
    path_value = status.get("staged_input")
    expected = status.get("staged_input_sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise RuntimeError("Production status has no staged input provenance")
    unresolved = run_dir / path_value
    if unresolved.is_symlink():
        raise RuntimeError(f"Staged production deck must not be a symlink: {unresolved}")
    path = unresolved.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Staged production deck escapes the run tree: {path}") from exc
    mode = path.stat().st_mode if path.is_file() else 0
    if not path.is_file() or mode & 0o222 or sha256_path(path) != expected:
        raise RuntimeError(f"Staged production deck is missing or changed: {path}")
    run_id = status.get("run_id")
    if not isinstance(run_id, str) or production_input_run_id(path) != run_id:
        raise RuntimeError(f"Staged production deck has the wrong run identity: {path}")
    return path


def initialize_production_status(
    args: argparse.Namespace,
    run_dir: Path,
    devices: list[str],
    gate: dict[str, object],
    gate_path: Path,
    staged_binary: Path,
    staged_input: Path,
    run_id: str,
) -> dict[str, Any]:
    if getattr(args, "prepare_only", False):
        baseline: dict[str, dict[str, float | str]] = {}
        processes: dict[str, list[str]] = {}
    else:
        baseline, processes = gpu_preflight(devices, args.allow_busy_gpus)
    now = utc_now()
    status: dict[str, Any] = {
        "status_schema": PRODUCTION_STATUS_SCHEMA,
        "mode": "production",
        "state": "prepared" if getattr(args, "prepare_only", False) else "initialized",
        "created_at": now,
        "updated_at": now,
        "ranks": args.ranks,
        "gpus": devices,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "segment_wall_time": args.segment_wall_time,
        "output_cadences": production_output_cadences(staged_input),
        "current_time": 0.0,
        "current_cycle": 0,
        "target_time": PRODUCTION_PHASE1_TARGET,
        "segments": [],
        "case_artifacts": gate_artifact_hashes(staged_binary),
        "material_manifest_sha256": sha256_path(MATERIAL_TABLE_MANIFEST),
        "production_gate": str(gate_path),
        "production_gate_sha256": sha256_path(gate_path),
        "staged_binary": str(staged_binary.relative_to(run_dir)),
        "staged_binary_sha256": sha256_path(staged_binary),
        "staged_input": str(staged_input.relative_to(run_dir)),
        "staged_input_sha256": sha256_path(staged_input),
        "baseline": baseline,
        "baseline_processes": processes,
    }
    if production_input_run_id(staged_input) != run_id:
        raise RuntimeError("Staged production deck does not contain this run identity")
    if gate.get("artifacts") != status["case_artifacts"]:
        raise RuntimeError("Staged production artifacts no longer match the gate")
    atomic_write_json(run_dir / RUN_STATUS, status)
    return status


def validate_resumed_production(
    args: argparse.Namespace,
    run_dir: Path,
    devices: list[str],
    gate_path: Path,
) -> tuple[dict[str, Any], Path, Path, RestartInfo | None, str | None]:
    validate_owned_resume_tree(run_dir)
    status = load_production_status(run_dir)
    if status.get("ranks") != args.ranks or status.get("gpus") != devices:
        raise RuntimeError("--resume must use the original eight ranks and GPU mapping")
    run_id = status.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise RuntimeError("Production status has no valid run identity")
    staged_binary = staged_binary_from_status(run_dir, status)
    staged_input = staged_input_from_status(run_dir, status)
    recorded_gate_hash = status.get("production_gate_sha256")
    if not gate_path.is_file() or sha256_path(gate_path) != recorded_gate_hash:
        raise RuntimeError("The production gate used to start this run is missing or changed")
    gate = validate_production_gate(gate_path, staged_binary)
    if status.get("case_artifacts") != gate_artifact_hashes(staged_binary):
        raise RuntimeError("Current production sources do not match the resumed run")
    if gate.get("artifacts") != status.get("case_artifacts"):
        raise RuntimeError("Resumed run artifacts do not match its production gate")
    manifest_hash = status.get("material_manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise RuntimeError("Production status has no material-manifest provenance")
    validate_staged_material_tables(run_dir, manifest_hash)
    if sha256_path(run_dir / "material_tables" / "manifest.json") != status.get(
        "material_manifest_sha256"
    ):
        raise RuntimeError("Staged material manifest does not match run provenance")
    if status.get("output_cadences") != production_output_cadences(staged_input):
        raise RuntimeError("Production output cadences changed since this run started")
    # Refuse an orphaned or foreign MPI job before inspecting files it may still write.
    gpu_preflight(devices, args.allow_busy_gpus)

    recorded_time = float(status.get("current_time", 0.0))
    recorded_cycle = int(status.get("current_cycle", 0))
    prepared_without_restart = is_prepared_without_restart(run_dir, status)
    if prepared_without_restart:
        restart = None
        rejected: list[str] = []
    else:
        restart, rejected = select_valid_restart(
            run_dir,
            "dci_3d",
            # A corrupted recorded checkpoint must be allowed to fall back to the prior
            # verified file, even though that intentionally rolls back simulated time.
            minimum_time=0.0,
            expected_hashes=recorded_restart_hashes(status),
            require_production_layout=True,
            expected_run_id=run_id,
        )
    if rejected:
        status.setdefault("restart_rejections", []).extend(rejected)
    if restart is not None and (restart.time, restart.cycle) < (
        recorded_time,
        recorded_cycle,
    ):
        supersede_restart_lineage(status, restart)
        status.setdefault("restart_rollbacks", []).append({
            "recorded_time": recorded_time,
            "recorded_cycle": recorded_cycle,
            "selected_time": restart.time,
            "selected_cycle": restart.cycle,
            "selected_restart": restart_record(restart, run_dir),
            "at": utc_now(),
        })
    for segment in status["segments"]:
        if isinstance(segment, dict) and segment.get("state") == "running":
            segment["state"] = "interrupted"
            segment["ended_at"] = utc_now()
    if restart is None:
        status.update(
            state="prepared",
            updated_at=utc_now(),
            resume_validated_at=utc_now(),
            current_time=0.0,
            current_cycle=0,
        )
        status.pop("last_restart", None)
    else:
        status.update(
            state="resumed",
            updated_at=utc_now(),
            current_time=restart.time,
            current_cycle=restart.cycle,
            last_restart=restart_record(restart, run_dir),
        )

        if restart.time == PRODUCTION_PHASE1_TARGET:
            if restart.kind != "numbered":
                raise RuntimeError("Exact 5 ns resume requires a numbered restart")
            status["phase1_checkpoint"] = restart_record(restart, run_dir)
        if restart.time > PRODUCTION_PHASE1_TARGET:
            checkpoint = status.get("phase1_checkpoint")
            if not isinstance(checkpoint, dict):
                raise RuntimeError("Phase-2 resume lacks the exact 5 ns checkpoint provenance")
            status_restart_info(
                run_dir, checkpoint, PRODUCTION_PHASE1_TARGET, run_id
            )

    wall_time = args.segment_wall_time
    if wall_time is None:
        stored_wall_time = status.get("segment_wall_time")
        wall_time = stored_wall_time if isinstance(stored_wall_time, str) else None
    else:
        status["segment_wall_time"] = wall_time
    atomic_write_json(run_dir / RUN_STATUS, status)
    return status, staged_binary, staged_input, restart, wall_time


def run_production_segments(
    args: argparse.Namespace,
    run_dir: Path,
    devices: list[str],
    status: dict[str, Any],
    staged_binary: Path,
    staged_input: Path,
    current_restart: RestartInfo | None,
    wall_time: str | None,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    current_time = 0.0 if current_restart is None else current_restart.time
    current_cycle = 0 if current_restart is None else current_restart.cycle
    run_id = status.get("run_id")
    if not isinstance(run_id, str):
        raise RuntimeError("Production status has no run identity")

    while True:
        phase_target = production_phase_for_time(current_time)
        if phase_target is None:
            status.update(
                state="complete",
                updated_at=utc_now(),
                current_time=PRODUCTION_FINAL_TARGET,
                target_time=PRODUCTION_FINAL_TARGET,
                active_segment=None,
            )
            atomic_write_json(run_dir / RUN_STATUS, status)
            return 0
        phase, target = phase_target
        if phase == 2:
            checkpoint = status.get("phase1_checkpoint")
            if not isinstance(checkpoint, dict):
                if current_time == PRODUCTION_PHASE1_TARGET and current_restart is not None:
                    checkpoint = restart_record(current_restart, run_dir)
                    status["phase1_checkpoint"] = checkpoint
                else:
                    raise RuntimeError(
                        "Phase 2 cannot start without an exact, verified 5 ns checkpoint"
                    )
            status_restart_info(
                run_dir, checkpoint, PRODUCTION_PHASE1_TARGET, run_id
            )

        if staged_binary_from_status(run_dir, status) != staged_binary.resolve():
            raise RuntimeError("Staged Athena executable changed between segments")
        if staged_input_from_status(run_dir, status) != staged_input.resolve():
            raise RuntimeError("Staged production deck changed between segments")
        manifest_hash = status.get("material_manifest_sha256")
        if not isinstance(manifest_hash, str):
            raise RuntimeError("Production status has no material-manifest provenance")
        validate_staged_material_tables(run_dir, manifest_hash)
        baseline, processes = gpu_preflight(devices, args.allow_busy_gpus)
        segment_index = len(status["segments"])
        log_path = run_dir / f"phase{phase}.segment{segment_index:05d}.log"
        overrides = [f"time/tlim={target:.1f}"]
        if current_restart is None:
            mpi = mpi_command(
                staged_input,
                overrides,
                binary_path=staged_binary,
                wall_time=wall_time,
            )
            restart_source = None
        else:
            relative_restart = current_restart.path.resolve().relative_to(run_dir.resolve())
            mpi = restart_command(
                relative_restart,
                overrides,
                binary_path=staged_binary,
                wall_time=wall_time,
            )
            restart_source = restart_record(current_restart, run_dir)

        segment: dict[str, Any] = {
            "index": segment_index,
            "phase": phase,
            "target_time": target,
            "start_time": current_time,
            "start_cycle": current_cycle,
            "restart_source": restart_source,
            "wall_time": wall_time,
            "log": str(log_path.relative_to(run_dir)),
            "mpi_command": mpi,
            "baseline": baseline,
            "baseline_processes": processes,
            "state": "running",
            "started_at": utc_now(),
        }
        status["segments"].append(segment)
        status.update(
            state="running",
            updated_at=utc_now(),
            active_segment=segment_index,
            current_time=current_time,
            current_cycle=current_cycle,
            target_time=target,
        )
        atomic_write_json(run_dir / RUN_STATUS, status)

        try:
            heartbeat = StatusHeartbeat(
                run_dir / RUN_STATUS, status, segment
            )
            exit_code, elapsed, memory = run_logged(
                shell_command(run_dir, mpi), log_path, env, None, heartbeat
            )
        except BaseException as exc:
            segment.update(
                state="interrupted",
                ended_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            status.update(
                state="interrupted", updated_at=utc_now(), active_segment=None
            )
            atomic_write_json(run_dir / RUN_STATUS, status)
            raise

        segment.update(
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            memory_monitor=memory,
            ended_at=utc_now(),
        )
        if exit_code != 0:
            segment["state"] = "failed"
            status.update(state="failed", updated_at=utc_now(), active_segment=None)
            atomic_write_json(run_dir / RUN_STATUS, status)
            return exit_code

        try:
            expected_hashes = recorded_restart_hashes(status)
            # The rolling checkpoint is intentionally atomically replaced by this
            # segment.  Its old recorded hash is verified on --resume, not after a
            # successful child that must produce new content at the same pathname.
            expected_hashes.pop("dci_3d.walltime.rst", None)
            next_restart, rejected = select_valid_restart(
                run_dir,
                "dci_3d",
                minimum_time=current_time,
                expected_hashes=expected_hashes,
                require_production_layout=True,
                expected_run_id=run_id,
            )
            if (
                current_restart is not None
                and current_restart.kind == "numbered"
                and next_restart.kind == "numbered"
                and next_restart.file_number is not None
                and current_restart.file_number is not None
                and next_restart.file_number <= current_restart.file_number
            ):
                raise RuntimeError("Athena did not advance its numbered restart counter")
            if next_restart.time <= current_time or next_restart.cycle <= current_cycle:
                raise RuntimeError(
                    "Production segment must advance both simulation time and cycle"
                )
            if next_restart.time > target:
                raise RuntimeError(
                    f"Production segment overshot {target} ns: {next_restart.time}"
                )
            if next_restart.time == target and next_restart.kind != "numbered":
                raise RuntimeError(
                    f"Exact {target:.0f} ns phase boundary requires a regular "
                    "numbered restart"
                )
        except Exception as exc:
            segment.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            status.update(state="failed", updated_at=utc_now(), active_segment=None)
            atomic_write_json(run_dir / RUN_STATUS, status)
            raise

        if rejected:
            segment["restart_rejections"] = rejected
        segment["restart"] = restart_record(next_restart, run_dir)
        segment["state"] = "completed"
        current_restart = next_restart
        current_time = next_restart.time
        current_cycle = next_restart.cycle
        status.update(
            state="segment_completed",
            updated_at=utc_now(),
            active_segment=None,
            current_time=current_time,
            current_cycle=current_cycle,
            last_restart=restart_record(next_restart, run_dir),
        )

        if current_time == target:
            if phase == 1:
                status["phase1_checkpoint"] = restart_record(next_restart, run_dir)
                status["state"] = "phase1_complete"
            else:
                status["state"] = "complete"
        elif wall_time is None:
            segment["state"] = "failed"
            segment["error"] = (
                "Athena exited before its target without --segment-wall-time"
            )
            status["state"] = "failed"
            atomic_write_json(run_dir / RUN_STATUS, status)
            raise RuntimeError(segment["error"])
        atomic_write_json(run_dir / RUN_STATUS, status)


def run_production(
    args: argparse.Namespace,
    run_dir: Path,
    devices: list[str],
    gate_path: Path,
    fresh_gate: dict[str, object] | None,
) -> int:
    if args.resume:
        status, staged_binary, staged_input, restart, wall_time = validate_resumed_production(
            args, run_dir, devices, gate_path
        )
    else:
        if fresh_gate is None:
            raise RuntimeError("Fresh production requires a validated gate")
        if args.clean:
            clean_run_dir(run_dir)
        prepare_run_dir(run_dir, "production")
        stage_material_tables(run_dir)
        expected_binary_hash = fresh_gate["artifacts"]["athena_binary"]
        if not isinstance(expected_binary_hash, str):
            raise RuntimeError("Production gate has no Athena executable hash")
        staged_binary = stage_production_binary(
            run_dir, BINARY, expected_binary_hash
        )
        artifacts = fresh_gate.get("artifacts")
        expected_input_hash = (
            artifacts.get("dci_3d.athinput") if isinstance(artifacts, dict) else None
        )
        if not isinstance(expected_input_hash, str):
            raise RuntimeError("Production gate has no production-deck hash")
        run_id = secrets.token_hex(16)
        staged_input = stage_production_input(
            run_dir, PRODUCTION_INPUT, expected_input_hash, run_id
        )
        status = initialize_production_status(
            args,
            run_dir,
            devices,
            fresh_gate,
            gate_path,
            staged_binary,
            staged_input,
            run_id,
        )
        if args.prepare_only:
            return 0
        restart = None
        wall_time = args.segment_wall_time
    return run_production_segments(
        args,
        run_dir,
        devices,
        status,
        staged_binary,
        staged_input,
        restart,
        wall_time,
    )


def refuse_unowned_nonempty_tree(run_dir: Path) -> None:
    contents = (
        [path for path in run_dir.iterdir() if path.name != RUN_LOCK]
        if run_dir.exists()
        else []
    )
    if contents and not (run_dir / RUN_SENTINEL).is_file():
        raise RuntimeError(
            f"Refusing nonempty directory without {RUN_SENTINEL}: {run_dir}"
        )


def print_production_resume_dry_run(
    args: argparse.Namespace, run_dir: Path, devices: list[str]
) -> None:
    validate_owned_resume_tree(run_dir)
    status = load_production_status(run_dir)
    staged_binary = staged_binary_from_status(run_dir, status)
    staged_input = staged_input_from_status(run_dir, status)
    run_id = status.get("run_id")
    if not isinstance(run_id, str):
        raise RuntimeError("Production status has no run identity")
    if is_prepared_without_restart(run_dir, status):
        restart = None
        rejected: list[str] = []
    else:
        restart, rejected = select_valid_restart(
            run_dir,
            "dci_3d",
            minimum_time=0.0,
            expected_hashes=recorded_restart_hashes(status),
            require_production_layout=True,
            expected_run_id=run_id,
        )
    for message in rejected:
        print(f"# Rejected restart: {message}")
    wall_time = args.segment_wall_time
    if wall_time is None and isinstance(status.get("segment_wall_time"), str):
        wall_time = status["segment_wall_time"]
    if restart is None:
        mpi = mpi_command(
            staged_input,
            [f"time/tlim={PRODUCTION_PHASE1_TARGET:.1f}"],
            binary_path=staged_binary,
            wall_time=wall_time,
        )
        print("# Start prepared production phase 1 from t=0; target is exactly 5.0 ns.")
        env = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
        print(
            f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
            + shlex.join(shell_command(run_dir, mpi))
        )
        return
    recorded_time = float(status.get("current_time", 0.0))
    if restart.time < recorded_time:
        print(
            f"# Fallback rolls back from recorded t={recorded_time:.17g} "
            f"to t={restart.time:.17g}."
        )
    phase_target = production_phase_for_time(restart.time)
    if phase_target is None:
        print("# Production already has a verified exact 10 ns restart.")
        return
    phase, target = phase_target
    if phase == 2 and restart.time > PRODUCTION_PHASE1_TARGET:
        checkpoint = status.get("phase1_checkpoint")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Phase-2 resume lacks exact 5 ns checkpoint provenance")
        status_restart_info(
            run_dir, checkpoint, PRODUCTION_PHASE1_TARGET, run_id
        )
    mpi = restart_command(
        restart.path.relative_to(run_dir),
        [f"time/tlim={target:.1f}"],
        binary_path=staged_binary,
        wall_time=wall_time,
    )
    print(
        f"# Resume phase {phase} from t={restart.time:.17g}, cycle={restart.cycle}; "
        f"target remains exactly {target:.1f} ns."
    )
    env = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    print(
        f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
        + shlex.join(shell_command(run_dir, mpi))
    )


def main() -> int:
    args = parse_args()
    devices = device_ids(args.gpus, args.ranks)
    run_dir = (args.run_dir or default_run_dir(args.mode)).expanduser().resolve()

    if args.mode == "production" and args.nlim is not None:
        raise RuntimeError("--nlim is only valid for non-production modes")
    if args.mode == "production" and args.radiation_c_light is not None:
        raise RuntimeError("--radiation-c-light is only valid for non-production modes")
    laser_variant_values = (
        args.laser_max_reflections,
        args.laser_reflection_offset,
        args.laser_reflection_hysteresis,
    )
    if args.mode == "production" and any(
        value is not None for value in laser_variant_values
    ):
        raise RuntimeError("laser transport overrides are only valid for non-production modes")
    if args.mode in ("production", "calibrate") and args.compact_scale != 1:
        raise RuntimeError("--compact-scale applies only to validate and smoke modes")
    if args.nlim is not None and args.nlim < 0:
        raise RuntimeError("--nlim must be non-negative")
    if args.laser_max_reflections is not None and args.laser_max_reflections <= 0:
        raise RuntimeError("--laser-max-reflections must be positive")
    if (
        args.laser_reflection_offset is not None
        and (
            not math.isfinite(args.laser_reflection_offset)
            or args.laser_reflection_offset <= 0.0
        )
    ):
        raise RuntimeError("--laser-reflection-offset must be finite and positive")
    if (
        args.laser_reflection_hysteresis is not None
        and (
            not math.isfinite(args.laser_reflection_hysteresis)
            or not 0.0 <= args.laser_reflection_hysteresis < 1.0
        )
    ):
        raise RuntimeError(
            "--laser-reflection-hysteresis must be finite and lie in [0,1)"
        )
    if args.resume and args.mode != "production":
        raise RuntimeError("--resume is only valid in production mode")
    if args.prepare_only and args.mode != "production":
        raise RuntimeError("--prepare-only is only valid in production mode")
    if args.prepare_only and args.resume:
        raise RuntimeError("--prepare-only and --resume are mutually exclusive")
    if args.segment_wall_time is not None and args.mode != "production":
        raise RuntimeError("--segment-wall-time is only valid in production mode")
    if args.resume and args.clean:
        raise RuntimeError("--resume and --clean are mutually exclusive")
    if args.resume and (args.build or args.regenerate_material_tables):
        raise RuntimeError("--resume uses staged artifacts; do not combine it with a rebuild")
    if args.clean and args.dry_run:
        raise RuntimeError("Combine neither --clean nor destructive actions with --dry-run")

    production_gate_path = args.production_gate.expanduser().resolve()
    if args.mode == "production" and args.dry_run and args.resume:
        print_production_resume_dry_run(args, run_dir, devices)
        return 0

    if not args.resume:
        prepare_material_tables(args.regenerate_material_tables, args.dry_run)
    if args.build:
        build(args.jobs, args.dry_run)
    if not args.dry_run and not args.resume and not BINARY.is_file():
        raise RuntimeError(f"AthenaK executable not found; use --build: {BINARY}")

    if args.mode == "production":
        if args.dry_run:
            digest = (
                sha256_path(BINARY) if BINARY.is_file() else "HASH_FROM_PRODUCTION_GATE"
            )
            staged = run_dir / STAGED_DIR / f"athena-{digest}"
            dry_run_id = "0" * 32
            input_text = PRODUCTION_INPUT.read_text(encoding="utf-8").replace(
                "<job>\n", f"<job>\nrun_id = {dry_run_id}\n", 1
            )
            input_digest = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            staged_input = (
                run_dir / STAGED_DIR / f"dci_3d-{input_digest}.athinput"
            )
            mpi = mpi_command(
                staged_input,
                [f"time/tlim={PRODUCTION_PHASE1_TARGET:.1f}"],
                binary_path=staged,
                wall_time=args.segment_wall_time,
            )
            print(
                "mkdir -p "
                f"{shlex.quote(str(run_dir / STAGED_DIR))} "
                f"{shlex.quote(str(run_dir / 'material_tables'))}"
            )
            print(
                "cp -a "
                f"{shlex.quote(str(MATERIAL_TABLE_DIR) + '/.')} "
                f"{shlex.quote(str(run_dir / 'material_tables') + '/')}"
            )
            print(f"cp --preserve=mode,timestamps {BINARY} {staged}")
            print(f"chmod 0555 {staged}")
            print(
                f"# Stage {PRODUCTION_INPUT} as {staged_input}, inject a fresh "
                "128-bit job/run_id, and chmod 0444."
            )
            if args.prepare_only:
                print("# Stop after atomically writing state=prepared; launch later with --resume.")
            else:
                print(
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(','.join(devices))} "
                    + shlex.join(shell_command(run_dir, mpi))
                )
            print(
                "# Actual production requires a hash-matched evidence manifest: "
                f"{production_gate_path}"
            )
            print("# Every segment preserves exact phase targets at 5.0 then 10.0 ns.")
            if args.segment_wall_time is not None:
                print(
                    "# Intermediate wall-time stops replace only "
                    "rst/dci_3d.walltime.rst; scheduled outputs are unchanged."
                )
            return 0

        fresh_gate: dict[str, object] | None = None
        if args.resume:
            validate_owned_resume_tree(run_dir)
        else:
            fresh_gate = validate_production_gate(production_gate_path)
            refuse_unowned_nonempty_tree(run_dir)
        with RunLock(run_dir):
            if args.prepare_only:
                return run_production(
                    args,
                    run_dir,
                    devices,
                    production_gate_path,
                    fresh_gate,
                )
            with GpuReservation(devices, run_dir):
                return run_production(
                    args,
                    run_dir,
                    devices,
                    production_gate_path,
                    fresh_gate,
                )

    input_path = CALIBRATION_INPUT
    first_overrides = nonproduction_overrides(
        args.mode,
        args.nlim,
        args.radiation_c_light,
        args.compact_scale,
        args.laser_max_reflections,
        args.laser_reflection_offset,
        args.laser_reflection_hysteresis,
    )
    first_mpi = mpi_command(input_path, first_overrides)
    if args.dry_run:
        print_dry_run(run_dir, first_mpi, devices)
        if args.mode == "smoke" and args.nlim is None:
            print("# Phase 2 restarts the compact checkpoint for 10 more RK2 cycles.")
        return 0

    refuse_unowned_nonempty_tree(run_dir)
    with RunLock(run_dir):
        if args.clean:
            clean_run_dir(run_dir)
        prepare_run_dir(run_dir, args.mode)
        stage_material_tables(run_dir)
        baseline, processes = gpu_preflight(devices, args.allow_busy_gpus)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        status: dict[str, Any] = {
            "mode": args.mode,
            "ranks": args.ranks,
            "gpus": devices,
            "radiation_c_light_override": args.radiation_c_light,
            "compact_scale": args.compact_scale,
            "laser_max_reflections_override": args.laser_max_reflections,
            "laser_reflection_offset_override": args.laser_reflection_offset,
            "laser_reflection_hysteresis_override": (
                args.laser_reflection_hysteresis
            ),
            "run_dir": str(run_dir),
            "case_artifacts": gate_artifact_hashes(),
            "material_manifest_sha256": sha256_path(MATERIAL_TABLE_MANIFEST),
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
        atomic_write_json(run_dir / RUN_STATUS, status)
        if first_code != 0:
            return first_code

        if args.mode == "calibrate" and (
            first_memory is None or not memory_is_accepted(first_memory)
        ):
            print(
                "Calibration completed, but at least one GPU was outside the required "
                "60-80% peak-memory band. See run_status.json."
            )
            return 2

        run_smoke_restart = args.mode == "smoke" and args.nlim is None
        if not run_smoke_restart:
            return 0

        restart_basename = "dci_3d_calibration"
        restarts = sorted((run_dir / "rst").glob(f"{restart_basename}.*.rst"))
        if not restarts:
            raise RuntimeError(
                f"Phase 1 completed without a {restart_basename} restart checkpoint"
            )
        restart = restarts[-1]
        second_overrides = smoke_restart_overrides(
            args.radiation_c_light,
            args.compact_scale,
            args.laser_max_reflections,
            args.laser_reflection_offset,
            args.laser_reflection_hysteresis,
        )
        second_mpi = restart_command(restart.relative_to(run_dir), second_overrides)
        second_baseline, _ = gpu_preflight(devices, args.allow_busy_gpus)
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
        atomic_write_json(run_dir / RUN_STATUS, status)
        return second_code


if __name__ == "__main__":
    raise SystemExit(main())
