"""Mocked safety and completion tests for the DCI transfer controller."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_control as tc


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.remote_files: dict[str, bytes] = {}
        self.dry_run_output = ""
        self.remote_parts: list[dict[str, object]] = []
        self.remote_rsync: list[dict[str, object]] = []
        self.manifest_factory = None
        self.watch_returncode = 0

    def run(self, args, *, input_text=None, timeout=None, check=True):
        del timeout, check
        assert isinstance(args, list)
        self.calls.append((list(args), input_text))
        stdout = ""
        returncode = 0
        if input_text and "DCI_REMOTE_PROVISION_V1" in input_text:
            base, transfer_id, encoded = args[-3:]
            self.remote_files[f"{base}/{transfer_id}/{tc.IDENTITY_NAME}"] = base64.b64decode(
                encoded
            )
        elif input_text and "DCI_REMOTE_STATUS_V1" in input_text:
            root = args[-1]
            identity = self.remote_files[f"{root}/{tc.IDENTITY_NAME}"]
            stdout = json.dumps({
                "identity_sha256": hashlib.sha256(identity).hexdigest(),
                "free_space": {"total_bytes": 1000, "used_bytes": 100,
                               "free_bytes": 900},
                "part_files": self.remote_parts,
                "rsync_processes": self.remote_rsync,
            })
        elif input_text and "DCI_REMOTE_MANIFEST_V1" in input_text:
            assert self.manifest_factory is not None
            stdout = json.dumps(self.manifest_factory())
        elif input_text and "DCI_REMOTE_HASH_V1" in input_text:
            stdout = hashlib.sha256(self.remote_files[args[-1]]).hexdigest() + "\n"
        elif args[0] == str(tc.LOCAL_RSYNC) and "--dry-run" in args:
            stdout = self.dry_run_output
        elif args[0] == str(tc.LOCAL_RSYNC):
            source = Path(args[-2])
            remote_name = args[-1].split(":", 1)[1]
            self.remote_files[remote_name] = source.read_bytes()
        elif len(args) >= 2 and Path(args[1]).name == "file_watcher.py":
            returncode = self.watch_returncode
        return subprocess.CompletedProcess(args, returncode, stdout, "")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    case = tmp_path / "DCI_3D"
    case.mkdir()
    monkeypatch.setattr(tc, "CASE_DIR", case)
    run = case / "run"
    run.mkdir(mode=0o700)
    gate = case / "production_gate.json"
    gate.write_text('{"schema": 4}\n', encoding="utf-8")
    gate_digest = tc.sha256_path(gate)
    run_id = "1" * 32
    write_json(run / tc.STATUS_NAME, {
        "status_schema": 2,
        "mode": "production",
        "state": "running",
        "run_dir": str(run.resolve()),
        "run_id": run_id,
        "production_gate_sha256": gate_digest,
        "current_time": 1.0,
        "target_time": 5.0,
        "active_segment": 0,
    })
    key = tmp_path / "id_ed25519"
    key.write_text("test key\n", encoding="utf-8")
    base = case / "tranfile_config.json"
    write_json(base, {
        "local_watch_dir": str(run),
        "remote_host": "example.test",
        "remote_user": "mengqi",
        "remote_dir": str(tc.REMOTE_BASE),
        "ssh_key_path": str(key),
        "delete_after_transfer": False,
        "delete_extensions": [],
    })
    watcher = tmp_path / "file_watcher.py"
    watcher.write_text("# shared watcher\n", encoding="utf-8")
    runner = FakeRunner()
    controller = tc.TransferController(
        runner=runner,
        watcher_path=watcher,
        python_executable=Path(sys.executable).resolve(),
        now_factory=lambda: datetime(2026, 7, 29, 1, 2, 3, tzinfo=timezone.utc),
        sleeper=lambda _: None,
    )
    context = controller.prepare(run, gate, base)
    monkeypatch.setattr(
        tc.TransferController,
        "_local_processes",
        staticmethod(lambda context: {"watcher": [], "rsync": []}),
    )
    return controller, context, runner


def mark_current_files(context: tc.TransferContext, *, failed: str | None = None) -> None:
    files, _, _ = tc.TransferController._scan_local_tree(context)
    entries = {}
    for record in files:
        entries[record["absolute_path"]] = {
            "status": "failed" if record["path"] == failed else "success",
            "size": record["size"],
            "mtime_ns": record["mtime_ns"],
            "last_error": "mock failure" if record["path"] == failed else None,
        }
    tc.atomic_write_json(Path(context.config["state_file"]), {"files": entries})


def complete_production(context: tc.TransferContext) -> None:
    status = json.loads((context.run_dir / tc.STATUS_NAME).read_text())
    status.update(state="complete", current_time=10.0, target_time=10.0,
                  active_segment=None, segments=[{"state": "completed", "exit_code": 0}])
    tc.atomic_write_json(context.run_dir / tc.STATUS_NAME, status)


def test_prepare_writes_absolute_non_deleting_identity_bound_config(prepared):
    _, context, runner = prepared
    assert tc.TRANSFER_ID_RE.fullmatch(context.identity["transfer_id"])
    assert context.identity["remote_dir"].startswith(f"{tc.REMOTE_BASE}/")
    assert context.config["delete_after_transfer"] is False
    assert context.config["delete_extensions"] == []
    assert context.config["create_remote_dir"] is False
    assert context.config["recursive"] is True
    for field in ("local_watch_dir", "state_file", "log_file", "lock_file"):
        assert Path(context.config[field]).is_absolute()
    assert context.config_path.is_file()
    assert context.identity_path.is_file()
    assert not (context.config_path.stat().st_mode & 0o222)
    assert all(isinstance(args, list) for args, _ in runner.calls)


def test_prepare_and_load_reject_nonprivate_run_root(prepared):
    controller, context, runner = prepared
    context.run_dir.chmod(0o755)
    calls_before = len(runner.calls)

    with pytest.raises(tc.TransferControlError, match="mode must be exactly 0700"):
        controller.prepare(
            context.run_dir,
            Path(context.identity["production_gate"]),
            tc.CASE_DIR / "tranfile_config.json",
        )
    with pytest.raises(tc.TransferControlError, match="mode must be exactly 0700"):
        controller.load_context(context.run_dir)

    assert len(runner.calls) == calls_before


def test_resume_rejects_delete_policy_or_identity_change(prepared):
    controller, context, _ = prepared
    unsafe = dict(context.config)
    unsafe["delete_after_transfer"] = True
    with pytest.raises(tc.TransferControlError, match="deletion"):
        controller._validate_config_safety(context.identity, context.config_path, unsafe)

    identity = json.loads(context.identity_path.read_text())
    identity["run_id"] = "2" * 32
    os.chmod(context.identity_path, 0o600)
    tc.atomic_write_json(context.identity_path, identity, mode=0o444)
    with pytest.raises(tc.TransferControlError, match="identities disagree"):
        controller.load_context(context.run_dir)


def test_status_uses_exact_size_mtime_signature_and_reports_failures(prepared):
    controller, context, _ = prepared
    first = context.run_dir / "first.bin"
    second = context.run_dir / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"012345")
    mark_current_files(context, failed="second.bin")
    first.write_bytes(b"abcd")

    report = controller.status(context.run_dir, strict_remote=True)
    assert report["backlog"]["count"] == 2
    assert report["backlog"]["bytes"] == 10
    assert report["backlog"]["failures"]["exact_signature_count"] == 1
    reasons = {item["path"]: item["reason"] for item in report["backlog"]["pending"]}
    assert reasons == {"first.bin": "changed", "second.bin": "failed"}
    assert report["free_space"]["local"]["free_bytes"] > 0
    assert report["free_space"]["remote"]["free_bytes"] == 900


def test_status_reports_local_and_remote_parts_and_processes(prepared):
    controller, context, runner = prepared
    (context.run_dir / "output.part").write_bytes(b"partial")
    runner.remote_parts = [{"path": ".tranfile-partial/output", "size": 7}]
    runner.remote_rsync = [{"pid": 99, "argv": ["rsync"]}]
    mark_current_files(context)
    report = controller.status(context.run_dir, strict_remote=True)
    assert report["part_files"]["local"][0]["path"] == "output.part"
    assert report["part_files"]["remote"] == runner.remote_parts
    assert report["processes"]["remote_rsync"] == runner.remote_rsync


def test_watch_uses_exact_safe_argv(prepared):
    controller, context, runner = prepared
    assert controller.watch(context.run_dir) == 0
    args, _ = runner.calls[-1]
    assert args == [str(controller.python_executable), str(controller.watcher_path),
                    "--config", str(context.config_path)]


def test_seal_requires_completed_production(prepared):
    controller, context, _ = prepared
    with pytest.raises(tc.TransferControlError, match="state=complete"):
        controller.seal(context.run_dir, settle_seconds=0)


def test_seal_requires_two_zero_backlogs_empty_dry_run_and_equal_manifests(prepared):
    controller, context, runner = prepared
    (context.run_dir / ".run_case.lock").touch()
    (context.run_dir / "payload.bin").write_bytes(b"payload")
    complete_production(context)
    mark_current_files(context)
    runner.manifest_factory = lambda: controller._local_manifest(context)

    marker = controller.seal(context.run_dir, settle_seconds=0)
    assert marker["final_rsync_checksum_dry_run_empty"] is True
    assert marker["local_remote_manifests_equal"] is True
    assert marker["file_count"] > 0
    assert (context.run_dir / tc.COMPLETE_NAME).is_file()
    assert runner.remote_files[
        f"{context.identity['remote_dir']}/{tc.COMPLETE_NAME}"
    ] == (context.run_dir / tc.COMPLETE_NAME).read_bytes()


def test_seal_refuses_nonempty_checksum_dry_run(prepared):
    controller, context, runner = prepared
    (context.run_dir / ".run_case.lock").touch()
    complete_production(context)
    mark_current_files(context)
    runner.dry_run_output = ">f.st......\tpayload.bin\n"
    with pytest.raises(tc.TransferControlError, match="not empty"):
        controller.seal(context.run_dir, settle_seconds=0)
