import argparse
import json
import os
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_case


def write_restart(
    path: Path,
    *,
    basename: str = "synthetic",
    time: float = 1.0,
    cycle: int = 10,
    payload_size: int = 32,
    next_file_number: int | None = None,
    cost: float = 1.0,
    run_id: str | None = None,
) -> None:
    numbered = path.name.split(".")[-2]
    number = int(numbered) if numbered.isdigit() else None
    if next_file_number is None:
        next_file_number = 0 if number is None else number + 1
    run_id_line = "" if run_id is None else f"run_id = {run_id}\n"
    parameters = (
        "#------------------------- PAR_DUMP -------------------------\n"
        "<job>\n"
        f"basename = {basename}\n"
        f"{run_id_line}"
        "<mesh>\n"
        "nghost = 2\n"
        "nx1 = 1\n"
        "nx2 = 1\n"
        "nx3 = 1\n"
        "<meshblock>\n"
        "nx1 = 1\n"
        "nx2 = 1\n"
        "nx3 = 1\n"
        "<output11>\n"
        "file_type = rst\n"
        f"file_number = {next_file_number}\n"
        "single_file_per_rank = 0\n"
        "<par_end>\n"
    ).encode("ascii")
    region = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    indices = (2, 1, 1, 1, *([0] * 15))
    fixed = run_case.RESTART_FIXED_HEADER.pack(
        1, 0, *region, *indices, *indices, time, 0.01, cycle
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        parameters
        + fixed
        + struct.pack("=4i", 0, 0, 0, 0)
        + struct.pack("=f", cost)
        + struct.pack("=Q", payload_size)
        + b"x" * payload_size
    )


def restart_info(
    path: Path, number: int | None, time: float, cycle: int
) -> run_case.RestartInfo:
    return run_case.RestartInfo(
        path=path,
        kind="walltime" if number is None else "numbered",
        file_number=number,
        meshblocks=run_case.PRODUCTION_MESHBLOCKS,
        root_level=run_case.PRODUCTION_ROOT_LEVEL,
        mesh_shape=run_case.PRODUCTION_MESH_SHAPE,
        block_shape=run_case.PRODUCTION_BLOCK_SHAPE,
        data_size=run_case.PRODUCTION_RESTART_DATA_SIZE,
        time=time,
        dt=1.0e-5,
        cycle=cycle,
        size=123,
        sha256=f"hash-{number}-{time}-{cycle}",
    )


def test_restart_parser_checks_exact_size_and_counter(tmp_path):
    restart = tmp_path / "synthetic.00003.rst"
    write_restart(restart, time=2.5, cycle=25)
    info = run_case.inspect_restart(restart, "synthetic")
    assert info.file_number == 3
    assert info.time == 2.5
    assert info.cycle == 25
    assert info.mesh_shape == (1, 1, 1)

    restart.write_bytes(restart.read_bytes()[:-1])
    with pytest.raises(run_case.RestartValidationError, match="incomplete"):
        run_case.inspect_restart(restart, "synthetic")


def test_restart_parser_accepts_walltime_and_rejects_invalid_cost(tmp_path):
    rolling = tmp_path / "synthetic.walltime.rst"
    write_restart(rolling, time=2.75, cycle=30, next_file_number=4)
    info = run_case.inspect_restart(rolling, "synthetic")
    assert info.kind == "walltime"
    assert info.file_number is None
    assert (info.time, info.cycle) == (2.75, 30)

    write_restart(rolling, time=2.75, cycle=30, cost=float("nan"))
    with pytest.raises(run_case.RestartValidationError, match="costs are invalid"):
        run_case.inspect_restart(rolling, "synthetic")


def test_restart_parser_enforces_production_run_identity(tmp_path):
    restart = tmp_path / "synthetic.00000.rst"
    run_id = "1" * 32
    write_restart(restart, run_id=run_id)
    run_case.inspect_restart(restart, "synthetic", run_id)
    with pytest.raises(run_case.RestartValidationError, match="different production run"):
        run_case.inspect_restart(restart, "synthetic", "2" * 32)


def test_restart_selection_falls_back_from_truncated_newest(tmp_path):
    restart_dir = tmp_path / "rst"
    older = restart_dir / "synthetic.00000.rst"
    newest = restart_dir / "synthetic.00001.rst"
    write_restart(older, time=1.0, cycle=10)
    write_restart(newest, time=2.0, cycle=20)
    newest.write_bytes(newest.read_bytes()[:-7])

    selected, rejected = run_case.select_valid_restart(tmp_path, "synthetic")
    assert selected.path == older
    assert selected.sha256 == run_case.sha256_path(older)
    assert rejected and rejected[0].startswith(newest.name)


def test_restart_selection_can_roll_back_from_corrupt_recorded_hash(tmp_path):
    restart_dir = tmp_path / "rst"
    older = restart_dir / "synthetic.00000.rst"
    newest = restart_dir / "synthetic.00001.rst"
    write_restart(older, time=1.0, cycle=10)
    write_restart(newest, time=2.0, cycle=20)
    expected = {
        older.name: run_case.sha256_path(older),
        newest.name: "0" * 64,
    }
    selected, rejected = run_case.select_valid_restart(
        tmp_path, "synthetic", expected_hashes=expected
    )
    assert selected.path == older
    assert any("SHA-256" in message for message in rejected)


def test_restart_selection_uses_embedded_progress_across_checkpoint_kinds(tmp_path):
    restart_dir = tmp_path / "rst"
    numbered = restart_dir / "synthetic.99999.rst"
    rolling = restart_dir / "synthetic.walltime.rst"
    write_restart(numbered, time=1.0, cycle=10)
    write_restart(rolling, time=2.0, cycle=20)

    selected, rejected = run_case.select_valid_restart(tmp_path, "synthetic")
    assert not rejected
    assert selected.path == rolling
    assert selected.kind == "walltime"


def test_resume_hash_rejects_replaced_rolling_checkpoint(tmp_path):
    restart_dir = tmp_path / "rst"
    rolling = restart_dir / "synthetic.walltime.rst"
    write_restart(rolling, time=1.0, cycle=10)
    first_hash = run_case.sha256_path(rolling)
    status = {
        "segments": [
            {"restart": {"path": "rst/synthetic.walltime.rst", "sha256": first_hash}}
        ]
    }

    replacement = restart_dir / ".synthetic.walltime.rst.part"
    write_restart(replacement, time=2.0, cycle=20)
    os.replace(replacement, rolling)
    with pytest.raises(RuntimeError, match="No valid") as error:
        run_case.select_valid_restart(
            tmp_path,
            "synthetic",
            expected_hashes=run_case.recorded_restart_hashes(status),
        )
    assert "recorded SHA-256 does not match" in str(error.value)

    status["segments"][0]["restart"]["sha256"] = run_case.sha256_path(rolling)
    selected, _ = run_case.select_valid_restart(
        tmp_path,
        "synthetic",
        expected_hashes=run_case.recorded_restart_hashes(status),
    )
    assert (selected.time, selected.cycle) == (2.0, 20)


def test_rollback_across_five_supersedes_future_hashes_and_checkpoint(tmp_path):
    selected = restart_info(
        tmp_path / "rst" / "dci_3d.walltime.rst", None, 4.0, 100
    )
    status = {
        "segments": [
            {
                "state": "completed",
                "restart": {
                    "path": "rst/dci_3d.walltime.rst",
                    "time": 4.0,
                    "cycle": 100,
                    "sha256": "at-four",
                },
            },
            {
                "state": "completed",
                "restart": {
                    "path": "rst/dci_3d.00000.rst",
                    "time": 5.0,
                    "cycle": 200,
                    "sha256": "old-five",
                },
            },
            {
                "state": "completed",
                "restart": {
                    "path": "rst/dci_3d.walltime.rst",
                    "time": 7.0,
                    "cycle": 300,
                    "sha256": "old-seven",
                },
            },
        ],
        "phase1_checkpoint": {
            "path": "rst/dci_3d.00000.rst",
            "time": 5.0,
            "cycle": 200,
            "sha256": "old-five",
        },
    }

    run_case.supersede_restart_lineage(status, selected)
    assert status["segments"][0]["state"] == "completed"
    assert [segment["state"] for segment in status["segments"][1:]] == [
        "superseded",
        "superseded",
    ]
    assert "phase1_checkpoint" not in status
    assert run_case.recorded_restart_hashes(status) == {
        "dci_3d.walltime.rst": "at-four"
    }


def test_production_phase_boundaries_never_accept_undershoot():
    assert run_case.production_phase_for_time(4.99999999995) == (1, 5.0)
    assert run_case.production_phase_for_time(5.0) == (2, 10.0)
    assert run_case.production_phase_for_time(9.99999999995) == (2, 10.0)
    assert run_case.production_phase_for_time(10.0) is None
    with pytest.raises(RuntimeError, match="outside"):
        run_case.production_phase_for_time(10.00000000001)


def test_wall_time_validation_and_command_uses_staged_binary(tmp_path):
    assert run_case.parse_wall_time("168:00:00") == "168:00:00"
    for value in ("1:2:3", "1:60:00", "0:00:00", "9999999:00:00"):
        with pytest.raises(argparse.ArgumentTypeError):
            run_case.parse_wall_time(value)

    staged = tmp_path / "staged" / "athena-hash"
    command = run_case.mpi_command(
        tmp_path / "case.athinput",
        [],
        binary_path=staged,
        wall_time="168:00:00",
    )
    assert command[3] == str(staged)
    assert command[4:6] == ["-t", "168:00:00"]
    assert str(run_case.BINARY) not in command


def test_laser_transport_cli_overrides_reach_both_smoke_commands(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case.py",
            "--mode",
            "smoke",
            "--laser-max-reflections",
            "32",
            "--laser-reflection-offset",
            "0.125",
            "--laser-reflection-hysteresis",
            "0.25",
        ],
    )
    args = run_case.parse_args()
    expected = {
        "laser/max_reflections_per_ray=32",
        "laser/reflection_offset_fraction=0.125",
        "laser/reflection_hysteresis_fraction=0.25",
    }

    initial = run_case.nonproduction_overrides(
        args.mode,
        args.nlim,
        args.radiation_c_light,
        args.compact_scale,
        args.laser_max_reflections,
        args.laser_reflection_offset,
        args.laser_reflection_hysteresis,
    )
    restart = run_case.smoke_restart_overrides(
        args.radiation_c_light,
        args.compact_scale,
        args.laser_max_reflections,
        args.laser_reflection_offset,
        args.laser_reflection_hysteresis,
    )

    assert expected <= set(initial)
    assert expected <= set(restart)
    assert "problem/allow_laser_transport_variants=true" in initial


def test_production_rejects_laser_transport_cli_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case.py",
            "--mode",
            "production",
            "--run-dir",
            str(tmp_path),
            "--laser-max-reflections",
            "32",
        ],
    )
    with pytest.raises(RuntimeError, match="only valid for non-production modes"):
        run_case.main()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--laser-max-reflections", "0", "must be positive"),
        ("--laser-reflection-offset", "nan", "finite and positive"),
        ("--laser-reflection-hysteresis", "1", r"lie in \[0,1\)"),
    ),
)
def test_invalid_laser_transport_cli_overrides_are_rejected(
    tmp_path, monkeypatch, option, value, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_case.py", "--run-dir", str(tmp_path), option, value],
    )
    with pytest.raises(RuntimeError, match=message):
        run_case.main()


def test_run_lock_rejects_second_launcher(tmp_path):
    with run_case.RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="Another launcher"):
            with run_case.RunLock(tmp_path):
                pass


def test_gpu_reservation_rejects_cross_run_race(tmp_path, monkeypatch):
    monkeypatch.setattr(run_case, "GPU_LOCK_ROOT", tmp_path / "gpu-locks")
    with run_case.GpuReservation(["0", "1"], tmp_path / "run-a"):
        with pytest.raises(RuntimeError, match="GPU 0 is reserved"):
            with run_case.GpuReservation(["0", "2"], tmp_path / "run-b"):
                pass


def test_staged_binary_is_content_addressed_and_read_only(tmp_path):
    source = tmp_path / "athena"
    source.write_bytes(b"immutable executable")
    source.chmod(0o755)
    digest = run_case.sha256_path(source)
    run_dir = tmp_path / "run"
    staged = run_case.stage_production_binary(run_dir, source, digest)
    assert staged.name == f"athena-{digest}"
    assert staged.read_bytes() == source.read_bytes()
    assert staged.stat().st_mode & 0o777 == 0o555
    assert run_case.stage_production_binary(run_dir, source, digest) == staged


def test_staged_input_is_content_addressed_and_read_only(tmp_path):
    source = tmp_path / "dci_3d.athinput"
    source.write_text("<job>\nbasename = dci_3d\n", encoding="utf-8")
    digest = run_case.sha256_path(source)
    run_dir = tmp_path / "run"
    run_id = "a" * 32
    staged = run_case.stage_production_input(run_dir, source, digest, run_id)
    assert staged.name.startswith("dci_3d-")
    assert "run_id = " + run_id in staged.read_text(encoding="utf-8")
    assert run_case.production_input_run_id(staged) == run_id
    assert staged.stat().st_mode & 0o777 == 0o444


def test_atomic_status_replacement(tmp_path):
    path = tmp_path / "run_status.json"
    run_case.atomic_write_json(path, {"state": "running", "segment": 2})
    assert json.loads(path.read_text()) == {"state": "running", "segment": 2}
    assert not list(tmp_path.glob(".*.tmp"))


def test_prepared_run_is_resumable_without_restart(tmp_path):
    status = {
        "state": "prepared",
        "segments": [],
        "current_time": 0.0,
        "current_cycle": 0,
    }
    assert run_case.is_prepared_without_restart(tmp_path, status)
    (tmp_path / "rst").mkdir()
    assert run_case.is_prepared_without_restart(tmp_path, status)
    write_restart(tmp_path / "rst" / "dci_3d.walltime.rst")
    assert not run_case.is_prepared_without_restart(tmp_path, status)


def test_validate_resume_starts_prepared_run_without_checkpoint(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / run_case.RUN_SENTINEL).write_text("owned\n", encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}\n", encoding="utf-8")
    staged_binary = run_dir / "staged" / "athena"
    staged_input = run_dir / "staged" / "input"
    manifest = run_dir / "material_tables" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    artifacts = {"athena_binary": "binary-hash"}
    status = {
        "status_schema": run_case.PRODUCTION_STATUS_SCHEMA,
        "mode": "production",
        "state": "prepared",
        "run_dir": str(run_dir),
        "run_id": "a" * 32,
        "ranks": 8,
        "gpus": [str(number) for number in range(8)],
        "segments": [],
        "current_time": 0.0,
        "current_cycle": 0,
        "case_artifacts": artifacts,
        "production_gate_sha256": run_case.sha256_path(gate_path),
        "material_manifest_sha256": run_case.sha256_path(manifest),
        "output_cadences": {},
    }
    run_case.atomic_write_json(run_dir / run_case.RUN_STATUS, status)
    monkeypatch.setattr(run_case, "staged_binary_from_status", lambda *args: staged_binary)
    monkeypatch.setattr(run_case, "staged_input_from_status", lambda *args: staged_input)
    monkeypatch.setattr(
        run_case, "validate_production_gate", lambda *args: {"artifacts": artifacts}
    )
    monkeypatch.setattr(run_case, "gate_artifact_hashes", lambda *args: artifacts)
    monkeypatch.setattr(run_case, "validate_staged_material_tables", lambda *args: None)
    monkeypatch.setattr(run_case, "production_output_cadences", lambda *args: {})
    monkeypatch.setattr(run_case, "gpu_preflight", lambda *args: ({}, {}))
    args = argparse.Namespace(
        ranks=8,
        segment_wall_time=None,
        allow_busy_gpus=False,
    )

    resumed, binary, input_path, restart, wall_time = (
        run_case.validate_resumed_production(
            args,
            run_dir,
            [str(number) for number in range(8)],
            gate_path,
        )
    )
    assert (binary, input_path, restart, wall_time) == (
        staged_binary,
        staged_input,
        None,
        None,
    )
    assert resumed["state"] == "prepared"


def test_segment_loop_reaches_exact_five_then_ten_without_gpu_monitor(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = run_dir / "staged" / "athena-hash"
    staged.parent.mkdir()
    staged.write_bytes(b"athena")
    staged.chmod(0o555)
    staged_input = run_dir / "staged" / "dci_3d-hash.athinput"
    staged_input.write_text(
        f"<job>\nrun_id = {'a' * 32}\nbasename = dci_3d\n",
        encoding="utf-8",
    )
    staged_input.chmod(0o444)
    sequence = [
        restart_info(run_dir / "rst" / "dci_3d.walltime.rst", None, 2.0, 100),
        restart_info(run_dir / "rst" / "dci_3d.00000.rst", 0, 5.0, 200),
        restart_info(run_dir / "rst" / "dci_3d.walltime.rst", None, 7.0, 300),
        restart_info(run_dir / "rst" / "dci_3d.00001.rst", 1, 10.0, 400),
    ]
    commands = []
    monitors = []

    monkeypatch.setattr(run_case, "gpu_preflight", lambda devices, busy: ({}, {}))
    monkeypatch.setattr(run_case, "validate_staged_material_tables", lambda *args: None)
    monkeypatch.setattr(
        run_case,
        "select_valid_restart",
        lambda *args, **kwargs: (sequence.pop(0), []),
    )
    monkeypatch.setattr(
        run_case,
        "status_restart_info",
        lambda run_dir, record, expected_time, expected_run_id: restart_info(
            run_dir / record["path"], record["file_number"], expected_time, 200
        ),
    )

    def fake_run_logged(command, log_path, env, monitor, heartbeat):
        commands.append(command)
        monitors.append(monitor)
        heartbeat.start(12345)
        heartbeat.stop()
        return 0, 1.0, None

    monkeypatch.setattr(run_case, "run_logged", fake_run_logged)
    status = {
        "status_schema": run_case.PRODUCTION_STATUS_SCHEMA,
        "mode": "production",
        "state": "initialized",
        "run_dir": str(run_dir),
        "run_id": "a" * 32,
        "segments": [],
        "staged_binary": str(staged.relative_to(run_dir)),
        "staged_binary_sha256": run_case.sha256_path(staged),
        "staged_input": str(staged_input.relative_to(run_dir)),
        "staged_input_sha256": run_case.sha256_path(staged_input),
        "material_manifest_sha256": "manifest-hash",
        "output_cadences": {f"output{number}": 0.5 for number in range(1, 12)},
    }
    args = argparse.Namespace(ranks=8, allow_busy_gpus=False)
    result = run_case.run_production_segments(
        args,
        run_dir,
        [str(number) for number in range(8)],
        status,
        staged,
        staged_input,
        None,
        "1:00:00",
    )
    assert result == 0
    assert [segment["phase"] for segment in status["segments"]] == [1, 1, 2, 2]
    assert [segment["target_time"] for segment in status["segments"]] == [
        5.0, 5.0, 10.0, 10.0
    ]
    assert status["current_time"] == 10.0
    assert status["state"] == "complete"
    assert [segment["restart"]["kind"] for segment in status["segments"]] == [
        "walltime",
        "numbered",
        "walltime",
        "numbered",
    ]
    assert all(monitor is None for monitor in monitors)
    assert all(str(staged) in command[-1] for command in commands)
    assert str(staged_input) in commands[0][-1]
    assert len(list(run_dir.glob("phase*.segment*.log"))) == 0


def test_exact_target_walltime_failure_is_persisted(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = run_dir / "staged" / "athena"
    staged_input = run_dir / "staged" / "input"
    staged.parent.mkdir()
    staged.touch()
    staged_input.touch()
    next_restart = restart_info(
        run_dir / "rst" / "dci_3d.walltime.rst", None, 5.0, 100
    )
    monkeypatch.setattr(run_case, "staged_binary_from_status", lambda *args: staged)
    monkeypatch.setattr(run_case, "staged_input_from_status", lambda *args: staged_input)
    monkeypatch.setattr(run_case, "validate_staged_material_tables", lambda *args: None)
    monkeypatch.setattr(run_case, "gpu_preflight", lambda *args: ({}, {}))
    monkeypatch.setattr(
        run_case,
        "run_logged",
        lambda *args: (0, 1.0, None),
    )
    monkeypatch.setattr(
        run_case,
        "select_valid_restart",
        lambda *args, **kwargs: (next_restart, []),
    )
    status = {
        "status_schema": run_case.PRODUCTION_STATUS_SCHEMA,
        "mode": "production",
        "state": "initialized",
        "run_dir": str(run_dir),
        "run_id": "a" * 32,
        "segments": [],
        "material_manifest_sha256": "manifest",
    }

    with pytest.raises(RuntimeError, match="numbered restart"):
        run_case.run_production_segments(
            argparse.Namespace(allow_busy_gpus=False),
            run_dir,
            [str(number) for number in range(8)],
            status,
            staged,
            staged_input,
            None,
            "1:00:00",
        )
    persisted = json.loads((run_dir / run_case.RUN_STATUS).read_text())
    assert persisted["state"] == "failed"
    assert persisted["segments"][0]["state"] == "failed"
    assert "numbered restart" in persisted["segments"][0]["error"]


def test_segment_requires_time_and_cycle_to_both_advance(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = run_dir / "staged" / "athena"
    staged_input = run_dir / "staged" / "input"
    staged.parent.mkdir()
    staged.touch()
    staged_input.touch()
    current = restart_info(run_dir / "rst" / "dci_3d.walltime.rst", None, 2.0, 100)
    stagnant = restart_info(run_dir / "rst" / "dci_3d.walltime.rst", None, 2.0, 101)
    monkeypatch.setattr(run_case, "staged_binary_from_status", lambda *args: staged)
    monkeypatch.setattr(run_case, "staged_input_from_status", lambda *args: staged_input)
    monkeypatch.setattr(run_case, "validate_staged_material_tables", lambda *args: None)
    monkeypatch.setattr(run_case, "gpu_preflight", lambda *args: ({}, {}))
    monkeypatch.setattr(run_case, "run_logged", lambda *args: (0, 1.0, None))
    monkeypatch.setattr(
        run_case,
        "select_valid_restart",
        lambda *args, **kwargs: (stagnant, []),
    )
    status = {
        "status_schema": run_case.PRODUCTION_STATUS_SCHEMA,
        "mode": "production",
        "state": "resumed",
        "run_dir": str(run_dir),
        "run_id": "a" * 32,
        "segments": [],
        "material_manifest_sha256": "manifest",
    }
    with pytest.raises(RuntimeError, match="advance both"):
        run_case.run_production_segments(
            argparse.Namespace(allow_busy_gpus=False),
            run_dir,
            [str(number) for number in range(8)],
            status,
            staged,
            staged_input,
            current,
            "1:00:00",
        )
