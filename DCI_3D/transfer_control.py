#!/usr/bin/env python3
"""Safely provision, supervise, audit, and seal DCI_3D output transfers."""

from __future__ import annotations

import argparse
import base64
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (CASE_DIR / "run").resolve()
DEFAULT_GATE = (CASE_DIR / "production_gate.json").resolve()
DEFAULT_BASE_CONFIG = (CASE_DIR / "tranfile_config.json").resolve()
TRANFILE_WATCHER = Path("/home/mengqi/Research/TranFile/file_watcher.py")
LOCAL_SSH = Path("/usr/bin/ssh")
LOCAL_RSYNC = Path("/usr/bin/rsync")
REMOTE_BASE = PurePosixPath("/home/mengqi/data/DCI_3D")
IDENTITY_NAME = "TRANSFER_IDENTITY.json"
COMPLETE_NAME = "TRANSFER_COMPLETE.json"
STATUS_NAME = "run_status.json"
IDENTITY_SCHEMA = 1
COMPLETE_SCHEMA = 1
STATUS_SCHEMA = 1
TRANSFER_ID_RE = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-g[0-9a-f]{64}-r[0-9a-f]{32}$"
)
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAFE_SSH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
IGNORE_PATTERNS = (
    "*.part",
    "*.tmp",
    COMPLETE_NAME,
    ".tranfile-partial",
    "*/.tranfile-partial/*",
)


class TransferControlError(RuntimeError):
    """Raised when transfer provenance or a safety invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Any, mode: int = 0o600) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload), mode=mode)


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransferControlError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TransferControlError(f"{description} must be a JSON object: {path}")
    return payload


def require_regular_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TransferControlError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_absolute_path(raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise TransferControlError(f"{field} must be an absolute path")
    return Path(raw).resolve()


def is_ignored(relative_path: str, name: str) -> bool:
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative_path, pattern)
        for pattern in IGNORE_PATTERNS
    )


def validate_ssh_component(value: str, field: str) -> str:
    if (
        not value
        or value.startswith("-")
        or SAFE_SSH_COMPONENT_RE.fullmatch(value) is None
    ):
        raise TransferControlError(f"Unsafe {field}: {value!r}")
    return value


class CommandRunner:
    """Subprocess adapter kept injectable so tests never contact SSH or rsync."""

    def run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: int | float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if not args or any(not isinstance(item, str) or "\x00" in item for item in args):
            raise TransferControlError("Command argv is malformed")
        try:
            result = subprocess.run(
                args,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransferControlError(f"Command failed to execute: {args[0]}: {exc}") from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
            raise TransferControlError(
                f"Command failed ({result.returncode}): {args[0]}: {detail}"
            )
        return result


@dataclass(frozen=True)
class TransferContext:
    run_dir: Path
    identity_path: Path
    identity: dict[str, Any]
    config_path: Path
    config: dict[str, Any]


REMOTE_PROVISION_SCRIPT = r'''# DCI_REMOTE_PROVISION_V1
import base64, json, os, pathlib, sys
base = pathlib.Path(sys.argv[1])
transfer_id = sys.argv[2]
payload = base64.b64decode(sys.argv[3], validate=True)
destination = base / transfer_id
if destination.parent != base or not base.is_absolute() or not base.is_dir():
    raise SystemExit(20)
identity = destination / "TRANSFER_IDENTITY.json"
if destination.exists():
    if not destination.is_dir() or not identity.is_file() or identity.read_bytes() != payload:
        raise SystemExit(21)
    raise SystemExit(0)
os.mkdir(destination, 0o700)
temporary = destination / ".TRANSFER_IDENTITY.json.part"
try:
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, identity)
    os.chmod(identity, 0o444)
    directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except BaseException:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        os.rmdir(destination)
    except OSError:
        pass
    raise
'''


REMOTE_STATUS_SCRIPT = r'''# DCI_REMOTE_STATUS_V1
import hashlib, json, os, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
identity = root / "TRANSFER_IDENTITY.json"
if not root.is_absolute() or not root.is_dir() or not identity.is_file():
    raise SystemExit(30)
parts = []
for directory, names, files in os.walk(root, followlinks=False):
    base = pathlib.Path(directory)
    for name in files:
        path = base / name
        relative = path.relative_to(root).as_posix()
        if name.endswith(".part") or "/.tranfile-partial/" in f"/{relative}":
            item = {"path": relative}
            try:
                info = path.lstat()
                item.update(size=info.st_size, mtime_ns=info.st_mtime_ns)
            except OSError as exc:
                item["error"] = str(exc)
            parts.append(item)
processes = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        fields = (entry / "cmdline").read_bytes().split(b"\0")
        argv = [field.decode("utf-8", "replace") for field in fields if field]
    except OSError:
        continue
    if (any(pathlib.Path(value).name == "rsync" for value in argv)
            and any(str(root) in value for value in argv)):
        processes.append({"pid": int(entry.name), "argv": argv})
usage = shutil.disk_usage(root)
payload = {
    "identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
    "free_space": {"total_bytes": usage.total, "used_bytes": usage.used,
                   "free_bytes": usage.free},
    "part_files": sorted(parts, key=lambda item: item["path"]),
    "rsync_processes": sorted(processes, key=lambda item: item["pid"]),
}
print(json.dumps(payload, sort_keys=True))
'''


REMOTE_MANIFEST_SCRIPT = r'''# DCI_REMOTE_MANIFEST_V1
import base64, fnmatch, hashlib, json, os, pathlib, stat, sys
root = pathlib.Path(sys.argv[1])
patterns = json.loads(base64.b64decode(sys.argv[2], validate=True))
def ignored(relative, name):
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative, pattern)
               for pattern in patterns)
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
entries = []
for directory, names, files in os.walk(root, topdown=True, followlinks=False):
    base = pathlib.Path(directory)
    names[:] = [name for name in names
                if not ignored((base / name).relative_to(root).as_posix(), name)]
    for name in files:
        path = base / name
        relative = path.relative_to(root).as_posix()
        if ignored(relative, name):
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            entries.append({"path": relative, "type": "symlink",
                            "target": os.readlink(path)})
        elif stat.S_ISREG(info.st_mode):
            entries.append({"path": relative, "type": "file", "size": info.st_size,
                            "sha256": digest(path)})
        else:
            raise SystemExit(41)
print(json.dumps({"entries": sorted(entries, key=lambda item: item["path"])},
                 sort_keys=True))
'''


REMOTE_HASH_SCRIPT = r'''# DCI_REMOTE_HASH_V1
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit(50)
result = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
print(result.hexdigest())
'''


class TransferController:
    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        watcher_path: Path = TRANFILE_WATCHER,
        python_executable: Path | None = None,
        now_factory: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.runner = runner or CommandRunner()
        self.watcher_path = watcher_path.expanduser().resolve()
        self.python_executable = Path(
            python_executable or sys.executable
        ).expanduser().resolve()
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper

    @staticmethod
    def _load_production_status(run_dir: Path) -> dict[str, Any]:
        status_path = require_regular_file(run_dir / STATUS_NAME, "production status")
        status = read_json_object(status_path, "production status")
        if status.get("status_schema") != 2 or status.get("mode") != "production":
            raise TransferControlError("run_status.json is not a schema-2 production status")
        if status.get("run_dir") != str(run_dir):
            raise TransferControlError("Production status does not own the selected run directory")
        run_id = status.get("run_id")
        if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
            raise TransferControlError("Production status has no valid run_id")
        return status

    @staticmethod
    def _safe_remote_directory(raw: Any, transfer_id: str) -> PurePosixPath:
        if not isinstance(raw, str):
            raise TransferControlError("Identity remote_dir is missing")
        remote = PurePosixPath(raw)
        if (
            not remote.is_absolute()
            or remote.parent != REMOTE_BASE
            or remote.name != transfer_id
        ):
            raise TransferControlError("Remote directory escapes the fixed DCI_3D base")
        return remote

    @staticmethod
    def _state_paths(transfer_id: str) -> tuple[Path, Path, Path, Path]:
        state_root = (CASE_DIR / "tranfile" / "state" / transfer_id).resolve()
        log_root = (CASE_DIR / "tranfile" / "logs" / transfer_id).resolve()
        return (
            state_root / "config.json",
            state_root / "transfer_state.json",
            log_root / "file_watcher.log",
            state_root / "file_watcher.lock",
        )

    def _new_config(
        self,
        base: dict[str, Any],
        run_dir: Path,
        remote_dir: PurePosixPath,
        paths: tuple[Path, Path, Path, Path],
    ) -> dict[str, Any]:
        config_path, state_path, log_path, lock_path = paths
        del config_path
        remote_host = validate_ssh_component(str(base.get("remote_host", "")), "remote_host")
        remote_user = validate_ssh_component(str(base.get("remote_user", "")), "remote_user")
        ssh_key_raw = base.get("ssh_key_path")
        ssh_key: str | None = None
        if ssh_key_raw not in (None, ""):
            ssh_key_path = Path(str(ssh_key_raw)).expanduser().resolve()
            require_regular_file(ssh_key_path, "SSH key")
            ssh_key = str(ssh_key_path)
        return {
            "local_watch_dir": str(run_dir),
            "remote_host": remote_host,
            "remote_user": remote_user,
            "remote_dir": str(remote_dir),
            "ssh_key_path": ssh_key,
            "file_stable_wait_seconds": float(
                base.get("file_stable_wait_seconds", 15.0)
            ),
            "poll_interval_seconds": float(base.get("poll_interval_seconds", 10.0)),
            "log_file": str(log_path),
            "state_file": str(state_path),
            "lock_file": str(lock_path),
            "allowed_extensions": [],
            "ignore_patterns": list(IGNORE_PATTERNS),
            "retry_attempts": int(base.get("retry_attempts", 5)),
            # The controller creates exactly one identity-bound leaf.  The shared watcher
            # must never recreate or redirect it on a later service restart.
            "create_remote_dir": False,
            "recursive": True,
            "watch_mode": "polling",
            "transfer_method": "rsync",
            "rsync_compress": False,
            "rsync_io_timeout_seconds": int(
                base.get("rsync_io_timeout_seconds", 300)
            ),
            "transfer_wall_timeout_seconds": int(
                base.get("transfer_wall_timeout_seconds", 7200)
            ),
            "stable_checks_required": max(
                2, int(base.get("stable_checks_required", 4))
            ),
            "delete_after_transfer": False,
            "delete_extensions": [],
        }

    @staticmethod
    def _validate_config_safety(
        context_identity: dict[str, Any], config_path: Path, config: dict[str, Any]
    ) -> None:
        transfer_id = context_identity.get("transfer_id")
        if not isinstance(transfer_id, str) or TRANSFER_ID_RE.fullmatch(transfer_id) is None:
            raise TransferControlError("Transfer identity has an invalid transfer_id")
        expected_paths = TransferController._state_paths(transfer_id)
        expected_config, expected_state, expected_log, expected_lock = expected_paths
        if config_path != expected_config:
            raise TransferControlError("Run-specific config is outside the ignored state tree")
        path_expectations = {
            "local_watch_dir": Path(str(context_identity["local_run_dir"])).resolve(),
            "state_file": expected_state,
            "log_file": expected_log,
            "lock_file": expected_lock,
        }
        for field, expected in path_expectations.items():
            actual = require_absolute_path(config.get(field), f"config {field}")
            if actual != expected:
                raise TransferControlError(f"Config {field} changed from its identity")
        if config.get("remote_dir") != context_identity.get("remote_dir"):
            raise TransferControlError("Config remote_dir changed from its identity")
        for field in ("remote_host", "remote_user", "ssh_key_path"):
            if config.get(field) != context_identity.get(field):
                raise TransferControlError(f"Config {field} changed from its identity")
        if config.get("delete_after_transfer") is not False:
            raise TransferControlError("Local deletion must remain disabled")
        if config.get("delete_extensions") != []:
            raise TransferControlError("delete_extensions must remain empty")
        if config.get("create_remote_dir") is not False:
            raise TransferControlError("Watcher remote-directory creation must remain disabled")
        if config.get("recursive") is not True or config.get("transfer_method") != "rsync":
            raise TransferControlError("Transfer must remain recursive rsync")
        if config.get("allowed_extensions") != []:
            raise TransferControlError("All completed run files must remain eligible")
        patterns = config.get("ignore_patterns")
        if not isinstance(patterns, list) or set(patterns) != set(IGNORE_PATTERNS):
            raise TransferControlError("Temporary-file ignore policy changed")

    def prepare(
        self,
        run_dir: Path,
        gate_path: Path = DEFAULT_GATE,
        base_config_path: Path = DEFAULT_BASE_CONFIG,
    ) -> TransferContext:
        run_dir = run_dir.expanduser().resolve()
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise TransferControlError(f"Production run directory is unavailable: {run_dir}")
        identity_path = run_dir / IDENTITY_NAME
        if identity_path.exists():
            context = self.load_context(run_dir)
            self._provision_remote_identity(context)
            return context

        status = self._load_production_status(run_dir)
        gate_path = require_regular_file(gate_path.expanduser().resolve(), "production gate")
        base_config_path = require_regular_file(
            base_config_path.expanduser().resolve(), "TranFile base config"
        )
        require_regular_file(self.watcher_path, "shared TranFile watcher")
        require_regular_file(self.python_executable, "Python executable")
        gate_sha256 = sha256_path(gate_path)
        if status.get("production_gate_sha256") != gate_sha256:
            raise TransferControlError("Production status and selected gate hash disagree")
        run_id = str(status["run_id"])
        moment = self.now_factory().astimezone(timezone.utc)
        transfer_id = (
            f"{moment.strftime('%Y%m%dT%H%M%SZ')}-g{gate_sha256}-r{run_id}"
        )
        if TRANSFER_ID_RE.fullmatch(transfer_id) is None:
            raise TransferControlError("Generated transfer identity is malformed")
        remote_dir = REMOTE_BASE / transfer_id
        paths = self._state_paths(transfer_id)
        config_path, state_path, log_path, lock_path = paths
        base_config = read_json_object(base_config_path, "TranFile base config")
        if PurePosixPath(str(base_config.get("remote_dir"))) != REMOTE_BASE:
            raise TransferControlError(
                f"Base config remote_dir must be exactly {REMOTE_BASE}"
            )
        config = self._new_config(base_config, run_dir, remote_dir, paths)
        config_payload = canonical_json_bytes(config)
        created_at = moment.isoformat()
        identity: dict[str, Any] = {
            "identity_schema": IDENTITY_SCHEMA,
            "transfer_id": transfer_id,
            "created_at": created_at,
            "run_id": run_id,
            "local_run_dir": str(run_dir),
            "production_gate": str(gate_path),
            "production_gate_sha256": gate_sha256,
            "remote_base": str(REMOTE_BASE),
            "remote_dir": str(remote_dir),
            "remote_host": config["remote_host"],
            "remote_user": config["remote_user"],
            "ssh_key_path": config["ssh_key_path"],
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
            "state_file": str(state_path),
            "log_file": str(log_path),
            "lock_file": str(lock_path),
            "tranfile_watcher": str(self.watcher_path),
            "tranfile_watcher_sha256": sha256_path(self.watcher_path),
            "python_executable": str(self.python_executable),
        }
        self._validate_config_safety(identity, config_path, config)
        atomic_write_bytes(config_path, config_payload, mode=0o444)
        atomic_write_json(state_path, {"files": {}}, mode=0o600)
        atomic_write_bytes(log_path, b"", mode=0o600)
        atomic_write_bytes(lock_path, b"", mode=0o600)
        atomic_write_json(identity_path, identity, mode=0o444)
        context = TransferContext(run_dir, identity_path, identity, config_path, config)
        self._provision_remote_identity(context)
        return context

    def load_context(self, run_dir: Path) -> TransferContext:
        run_dir = run_dir.expanduser().resolve()
        identity_path = require_regular_file(
            run_dir / IDENTITY_NAME, "transfer identity"
        )
        identity = read_json_object(identity_path, "transfer identity")
        if identity.get("identity_schema") != IDENTITY_SCHEMA:
            raise TransferControlError("Unsupported transfer identity schema")
        transfer_id = identity.get("transfer_id")
        if not isinstance(transfer_id, str) or TRANSFER_ID_RE.fullmatch(transfer_id) is None:
            raise TransferControlError("Transfer identity has an invalid transfer_id")
        if identity.get("local_run_dir") != str(run_dir):
            raise TransferControlError("Transfer identity belongs to a different run tree")
        if identity.get("remote_base") != str(REMOTE_BASE):
            raise TransferControlError("Transfer identity has a different remote base")
        self._safe_remote_directory(identity.get("remote_dir"), transfer_id)
        validate_ssh_component(str(identity.get("remote_host", "")), "remote_host")
        validate_ssh_component(str(identity.get("remote_user", "")), "remote_user")

        status = self._load_production_status(run_dir)
        if status.get("run_id") != identity.get("run_id"):
            raise TransferControlError("Transfer and production run identities disagree")
        gate_path = require_regular_file(
            require_absolute_path(identity.get("production_gate"), "production_gate"),
            "production gate",
        )
        gate_digest = sha256_path(gate_path)
        if (
            gate_digest != identity.get("production_gate_sha256")
            or status.get("production_gate_sha256") != gate_digest
        ):
            raise TransferControlError("Production gate changed since transfer preparation")

        watcher = require_regular_file(
            require_absolute_path(identity.get("tranfile_watcher"), "tranfile_watcher"),
            "shared TranFile watcher",
        )
        if watcher != self.watcher_path or sha256_path(watcher) != identity.get(
            "tranfile_watcher_sha256"
        ):
            raise TransferControlError("Shared TranFile watcher changed since preparation")
        python = require_regular_file(
            require_absolute_path(identity.get("python_executable"), "python_executable"),
            "Python executable",
        )
        if python != self.python_executable:
            raise TransferControlError("Transfer must resume with its original Python executable")

        config_path = require_regular_file(
            require_absolute_path(identity.get("config_path"), "config_path"),
            "run-specific TranFile config",
        )
        if sha256_path(config_path) != identity.get("config_sha256"):
            raise TransferControlError("Run-specific TranFile config changed")
        if stat.S_IMODE(config_path.stat().st_mode) & 0o222:
            raise TransferControlError("Run-specific TranFile config must be read-only")
        config = read_json_object(config_path, "run-specific TranFile config")
        self._validate_config_safety(identity, config_path, config)
        for field in ("state_file", "log_file", "lock_file"):
            expected = require_absolute_path(identity.get(field), f"identity {field}")
            actual = require_absolute_path(config.get(field), f"config {field}")
            if expected != actual:
                raise TransferControlError(f"Identity and config disagree on {field}")
        return TransferContext(run_dir, identity_path, identity, config_path, config)

    @staticmethod
    def _ssh_base_args(config: dict[str, Any]) -> list[str]:
        args = [
            str(LOCAL_SSH),
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4",
        ]
        key = config.get("ssh_key_path")
        if isinstance(key, str) and key:
            args.extend(["-i", str(require_absolute_path(key, "ssh_key_path"))])
        return args

    @staticmethod
    def _ssh_target(config: dict[str, Any]) -> str:
        user = validate_ssh_component(str(config.get("remote_user", "")), "remote_user")
        host = validate_ssh_component(str(config.get("remote_host", "")), "remote_host")
        return f"{user}@{host}"

    @classmethod
    def _rsync_ssh_command(cls, config: dict[str, Any]) -> str:
        return shlex.join(cls._ssh_base_args(config))

    @classmethod
    def _remote_spec(cls, context: TransferContext, name: str = "") -> str:
        remote = cls._safe_remote_directory(
            context.identity["remote_dir"], context.identity["transfer_id"]
        )
        if name:
            if PurePosixPath(name).name != name or name in {".", ".."}:
                raise TransferControlError(f"Unsafe remote file name: {name!r}")
            remote = remote / name
        suffix = "/" if not name else ""
        return f"{cls._ssh_target(context.config)}:{remote}{suffix}"

    def _run_remote_python(
        self,
        context: TransferContext,
        script: str,
        arguments: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        if any(not argument or "\x00" in argument for argument in arguments):
            raise TransferControlError("Remote Python argument is malformed")
        command = [
            *self._ssh_base_args(context.config),
            self._ssh_target(context.config),
            "python3",
            "-",
            *arguments,
        ]
        return self.runner.run(command, input_text=script, timeout=timeout)

    def _provision_remote_identity(self, context: TransferContext) -> None:
        identity_payload = context.identity_path.read_bytes()
        encoded = base64.b64encode(identity_payload).decode("ascii")
        self._run_remote_python(
            context,
            REMOTE_PROVISION_SCRIPT,
            [str(REMOTE_BASE), context.identity["transfer_id"], encoded],
            timeout=30,
        )
        remote_digest = self._remote_file_sha256(context, IDENTITY_NAME)
        if remote_digest != hashlib.sha256(identity_payload).hexdigest():
            raise TransferControlError("Remote transfer identity does not match local identity")

    def _remote_file_sha256(self, context: TransferContext, name: str) -> str:
        remote = self._safe_remote_directory(
            context.identity["remote_dir"], context.identity["transfer_id"]
        ) / name
        result = self._run_remote_python(
            context, REMOTE_HASH_SCRIPT, [str(remote)], timeout=30
        )
        digest = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise TransferControlError("Remote SHA-256 response is malformed")
        return digest

    def _remote_status(self, context: TransferContext) -> dict[str, Any]:
        result = self._run_remote_python(
            context,
            REMOTE_STATUS_SCRIPT,
            [context.identity["remote_dir"]],
            timeout=60,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TransferControlError("Remote status response is malformed") from exc
        if not isinstance(payload, dict):
            raise TransferControlError("Remote status response is not an object")
        expected = sha256_path(context.identity_path)
        if payload.get("identity_sha256") != expected:
            raise TransferControlError("Remote transfer identity mismatch")
        return payload

    def watch(self, run_dir: Path) -> int:
        context = self.load_context(run_dir)
        self._remote_status(context)
        result = self.runner.run(
            [
                str(self.python_executable),
                str(self.watcher_path),
                "--config",
                str(context.config_path),
            ],
            check=False,
        )
        return result.returncode

    @staticmethod
    def _scan_local_tree(
        context: TransferContext,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        files: list[dict[str, Any]] = []
        parts: list[dict[str, Any]] = []
        hazards: list[dict[str, Any]] = []
        root = context.run_dir
        for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            kept_names = []
            for name in names:
                path = base / name
                relative = path.relative_to(root).as_posix()
                try:
                    info = path.lstat()
                except OSError as exc:
                    hazards.append({"path": relative, "error": str(exc)})
                    continue
                if stat.S_ISLNK(info.st_mode):
                    hazards.append({"path": relative, "type": "symlink_directory"})
                    continue
                if is_ignored(relative, name):
                    continue
                kept_names.append(name)
            names[:] = kept_names
            for name in filenames:
                path = base / name
                relative = path.relative_to(root).as_posix()
                try:
                    info = path.lstat()
                except OSError as exc:
                    hazards.append({"path": relative, "error": str(exc)})
                    continue
                is_part = name.endswith(".part") or "/.tranfile-partial/" in f"/{relative}"
                if is_part:
                    parts.append({
                        "path": relative,
                        "size": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                    })
                    continue
                if is_ignored(relative, name):
                    continue
                if stat.S_ISLNK(info.st_mode):
                    hazards.append({"path": relative, "type": "symlink"})
                    continue
                if not stat.S_ISREG(info.st_mode):
                    hazards.append({"path": relative, "type": "special_file"})
                    continue
                files.append({
                    "path": relative,
                    "absolute_path": str(path.resolve()),
                    "size": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                })
        return (
            sorted(files, key=lambda item: item["path"]),
            sorted(parts, key=lambda item: item["path"]),
            sorted(hazards, key=lambda item: item["path"]),
        )

    @staticmethod
    def _load_transfer_state(context: TransferContext) -> dict[str, Any]:
        path = require_regular_file(
            require_absolute_path(context.config["state_file"], "state_file"),
            "TranFile state",
        )
        state = read_json_object(path, "TranFile state")
        if not isinstance(state.get("files"), dict):
            raise TransferControlError("TranFile state has no files object")
        return state

    @staticmethod
    def _backlog(
        local_files: list[dict[str, Any]], state: dict[str, Any]
    ) -> dict[str, Any]:
        state_files = state["files"]
        pending: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for record in local_files:
            entry = state_files.get(record["absolute_path"])
            exact = (
                isinstance(entry, dict)
                and entry.get("size") == record["size"]
                and entry.get("mtime_ns") == record["mtime_ns"]
            )
            success = exact and entry.get("status") == "success"
            if success:
                continue
            reason = "unrecorded"
            if isinstance(entry, dict):
                reason = "failed" if exact and entry.get("status") == "failed" else "changed"
            item = {
                "path": record["path"],
                "size": record["size"],
                "mtime_ns": record["mtime_ns"],
                "reason": reason,
            }
            pending.append(item)
            if reason == "failed":
                item = dict(item)
                item["last_error"] = entry.get("last_error")
                item["last_attempt_at"] = entry.get("last_attempt_at")
                failures.append(item)
        oldest = min(pending, key=lambda item: item["mtime_ns"]) if pending else None
        historical_failures = sum(
            1 for entry in state_files.values()
            if isinstance(entry, dict) and entry.get("status") == "failed"
        )
        return {
            "count": len(pending),
            "bytes": sum(int(item["size"]) for item in pending),
            "oldest_pending": oldest,
            "pending": pending,
            "failures": {
                "exact_signature_count": len(failures),
                "historical_state_count": historical_failures,
                "exact_signature": failures,
            },
        }

    @staticmethod
    def _local_processes(context: TransferContext) -> dict[str, list[dict[str, Any]]]:
        watchers: list[dict[str, Any]] = []
        rsync: list[dict[str, Any]] = []
        config_token = str(context.config_path)
        watcher_token = str(Path(context.identity["tranfile_watcher"]))
        remote_token = str(context.identity["remote_dir"])
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "cmdline").read_bytes().split(b"\0")
                argv = [field.decode("utf-8", "replace") for field in fields if field]
            except OSError:
                continue
            record = {"pid": int(entry.name), "argv": argv}
            if watcher_token in argv and config_token in argv:
                watchers.append(record)
            if (
                any(Path(value).name == "rsync" for value in argv)
                and any(remote_token in value for value in argv)
            ):
                rsync.append(record)
        return {
            "watcher": sorted(watchers, key=lambda item: item["pid"]),
            "rsync": sorted(rsync, key=lambda item: item["pid"]),
        }

    @staticmethod
    def _watcher_lock_held(context: TransferContext) -> bool:
        lock_path = require_absolute_path(context.config["lock_file"], "lock_file")
        if not lock_path.exists():
            return False
        try:
            with lock_path.open("r", encoding="utf-8") as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                finally:
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        except OSError:
            return False
        return False

    def status(
        self,
        run_dir: Path,
        *,
        strict_remote: bool = False,
    ) -> dict[str, Any]:
        context = self.load_context(run_dir)
        local_files, local_parts, hazards = self._scan_local_tree(context)
        state = self._load_transfer_state(context)
        backlog = self._backlog(local_files, state)
        usage = shutil.disk_usage(context.run_dir)
        local_processes = self._local_processes(context)
        local_processes["watcher_lock_held"] = self._watcher_lock_held(context)  # type: ignore[assignment]
        remote: dict[str, Any]
        try:
            remote = self._remote_status(context)
            remote["identity_verified"] = True
        except TransferControlError as exc:
            if strict_remote:
                raise
            remote = {"identity_verified": False, "error": str(exc)}
        return {
            "status_schema": STATUS_SCHEMA,
            "checked_at": utc_now(),
            "transfer_id": context.identity["transfer_id"],
            "run_id": context.identity["run_id"],
            "local_run_dir": str(context.run_dir),
            "remote_dir": context.identity["remote_dir"],
            "eligible_file_count": len(local_files),
            "eligible_bytes": sum(int(item["size"]) for item in local_files),
            "backlog": backlog,
            "part_files": {
                "local": local_parts,
                "remote": remote.get("part_files", []),
            },
            "hazards": hazards,
            "free_space": {
                "local": {
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                },
                "remote": remote.get("free_space"),
            },
            "processes": {
                "local": local_processes,
                "remote_rsync": remote.get("rsync_processes", []),
            },
            "remote": remote,
        }

    @staticmethod
    def _require_quiescent(report: dict[str, Any], sample: str) -> None:
        if report["backlog"]["count"] != 0:
            raise TransferControlError(f"{sample}: transfer backlog is not zero")
        if report["hazards"]:
            raise TransferControlError(f"{sample}: run tree has symlink/special-file hazards")
        if report["part_files"]["local"] or report["part_files"]["remote"]:
            raise TransferControlError(f"{sample}: partial files still exist")
        processes = report["processes"]
        if processes["local"]["rsync"] or processes["remote_rsync"]:
            raise TransferControlError(f"{sample}: rsync is still active")
        if not report["remote"].get("identity_verified"):
            raise TransferControlError(f"{sample}: remote identity is not verified")

    @staticmethod
    def _production_is_complete(context: TransferContext) -> dict[str, Any]:
        status = TransferController._load_production_status(context.run_dir)
        if (
            status.get("state") != "complete"
            or float(status.get("current_time", -1.0)) != 10.0
            or status.get("active_segment") is not None
            or status.get("run_id") != context.identity.get("run_id")
        ):
            raise TransferControlError(
                "Production must be state=complete at exactly 10 ns with no active segment"
            )
        return status

    def _rsync_dry_run(self, context: TransferContext) -> list[str]:
        args = [
            str(LOCAL_RSYNC),
            "-a",
            "--checksum",
            "--dry-run",
            "--itemize-changes",
            "--omit-dir-times",
            "--out-format=%i\t%n%L",
        ]
        for pattern in IGNORE_PATTERNS:
            args.append(f"--exclude={pattern}")
        args.extend([
            "-e",
            self._rsync_ssh_command(context.config),
            f"{context.run_dir}/",
            self._remote_spec(context),
        ])
        result = self.runner.run(args, timeout=7200)
        return [line for line in result.stdout.splitlines() if line.strip()]

    @staticmethod
    def _local_manifest(context: TransferContext) -> dict[str, Any]:
        files, parts, hazards = TransferController._scan_local_tree(context)
        if parts or hazards:
            raise TransferControlError("Local tree changed while constructing its manifest")
        entries: list[dict[str, Any]] = []
        for record in files:
            path = Path(record["absolute_path"])
            before = path.stat()
            digest = sha256_path(path)
            after = path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_size != record["size"]
                or before.st_mtime_ns != record["mtime_ns"]
            ):
                raise TransferControlError(f"File changed while hashing: {path}")
            entries.append({
                "path": record["path"],
                "type": "file",
                "size": record["size"],
                "sha256": digest,
            })
        return {"entries": entries}

    def _remote_manifest(self, context: TransferContext) -> dict[str, Any]:
        encoded = base64.b64encode(canonical_json_bytes(list(IGNORE_PATTERNS))).decode(
            "ascii"
        )
        result = self._run_remote_python(
            context,
            REMOTE_MANIFEST_SCRIPT,
            [context.identity["remote_dir"], encoded],
            timeout=24 * 3600,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TransferControlError("Remote manifest response is malformed") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
            raise TransferControlError("Remote manifest response has no entries")
        return payload

    def _publish_complete(
        self, context: TransferContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        complete_path = context.run_dir / COMPLETE_NAME
        atomic_write_json(complete_path, payload, mode=0o444)
        args = [
            str(LOCAL_RSYNC),
            "-a",
            "--partial-dir=.tranfile-partial",
            "--chmod=F0444",
            "-e",
            self._rsync_ssh_command(context.config),
            str(complete_path),
            self._remote_spec(context, COMPLETE_NAME),
        ]
        self.runner.run(args, timeout=7200)
        digest = sha256_path(complete_path)
        if self._remote_file_sha256(context, COMPLETE_NAME) != digest:
            raise TransferControlError("Remote completion marker hash mismatch")
        return payload

    def seal(self, run_dir: Path, *, settle_seconds: float = 30.0) -> dict[str, Any]:
        if settle_seconds < 0.0:
            raise TransferControlError("settle_seconds must be nonnegative")
        context = self.load_context(run_dir)
        existing = context.run_dir / COMPLETE_NAME
        if existing.exists():
            marker = read_json_object(
                require_regular_file(existing, "transfer completion marker"),
                "transfer completion marker",
            )
            if (
                marker.get("complete_schema") != COMPLETE_SCHEMA
                or marker.get("transfer_id") != context.identity["transfer_id"]
                or self._remote_file_sha256(context, COMPLETE_NAME) != sha256_path(existing)
            ):
                raise TransferControlError("Existing transfer completion marker is invalid")
            return marker

        run_lock_path = context.run_dir / ".run_case.lock"
        if not run_lock_path.exists():
            atomic_write_bytes(run_lock_path, b"", mode=0o600)
        with run_lock_path.open("r+", encoding="utf-8") as lock_stream:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TransferControlError("Production run lock is still held") from exc
            self._production_is_complete(context)
            first = self.status(context.run_dir, strict_remote=True)
            self._require_quiescent(first, "first zero-backlog sample")
            self.sleeper(settle_seconds)
            second = self.status(context.run_dir, strict_remote=True)
            self._require_quiescent(second, "second zero-backlog sample")
            differences = self._rsync_dry_run(context)
            if differences:
                raise TransferControlError(
                    "Final rsync checksum dry-run is not empty: " + differences[0]
                )
            local_manifest = self._local_manifest(context)
            remote_manifest = self._remote_manifest(context)
            if local_manifest != remote_manifest:
                raise TransferControlError("Local and remote SHA-256 manifests differ")
            # Recheck process/partial state after the expensive hashes and before publishing.
            final = self.status(context.run_dir, strict_remote=True)
            self._require_quiescent(final, "post-manifest sample")
            manifest_bytes = canonical_json_bytes(local_manifest)
            marker = {
                "complete_schema": COMPLETE_SCHEMA,
                "completed_at": utc_now(),
                "transfer_id": context.identity["transfer_id"],
                "run_id": context.identity["run_id"],
                "local_run_dir": str(context.run_dir),
                "remote_dir": context.identity["remote_dir"],
                "production_gate_sha256": context.identity["production_gate_sha256"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "file_count": len(local_manifest["entries"]),
                "total_bytes": sum(
                    int(entry.get("size", 0)) for entry in local_manifest["entries"]
                ),
                "zero_backlog_samples": [first["checked_at"], second["checked_at"]],
                "final_rsync_checksum_dry_run_empty": True,
                "local_remote_manifests_equal": True,
            }
            result = self._publish_complete(context, marker)
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="provision immutable run transfer")
    prepare.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    prepare.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    subparsers.add_parser("watch", help="verify identity and run shared TranFile watcher")
    subparsers.add_parser("status", help="report exact-signature transfer status")
    seal = subparsers.add_parser("seal", help="prove and publish transfer completion")
    seal.add_argument("--settle-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controller = TransferController()
    try:
        if args.command == "prepare":
            context = controller.prepare(args.run_dir, args.gate, args.base_config)
            print(json.dumps(context.identity, indent=2, sort_keys=True))
            return 0
        if args.command == "watch":
            return controller.watch(args.run_dir)
        if args.command == "status":
            report = controller.status(args.run_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["remote"].get("identity_verified") else 2
        if args.command == "seal":
            marker = controller.seal(
                args.run_dir, settle_seconds=args.settle_seconds
            )
            print(json.dumps(marker, indent=2, sort_keys=True))
            return 0
        raise AssertionError(args.command)
    except TransferControlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
