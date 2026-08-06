"""Deterministic unit tests for the DCI production-gate verifier."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_production_gate as gate
import run_case


def laser_diagnostic(
    *, launched: float = 1.0, remaining: float = 0.0,
    wave_remaining: float | None = None,
    reflection_remaining: float | None = None,
    residual: float = 0.0,
    waves: int = 9,
    max_reflections: int = 8,
    suppressed_turns: int = 3,
    reflection_rearms: int = 2,
) -> str:
    fields = [
        f"launched={launched:.17e}",
        f"deposited={launched-remaining:.17e}",
        "escaped=0.00000000000000000e+00",
        f"remaining={remaining:.17e}",
    ]
    if wave_remaining is not None or reflection_remaining is not None:
        assert wave_remaining is not None and reflection_remaining is not None
        fields.extend((
            f"wave_remaining={wave_remaining:.17e}",
            f"reflection_remaining={reflection_remaining:.17e}",
            f"wave_remaining_rays={int(wave_remaining > 0.0)}",
            f"reflection_remaining_rays={int(reflection_remaining > 0.0)}",
        ))
    fields.extend((
        f"residual={residual:.17e}",
        "active=0", "reflected=8",
        f"max_reflections={max_reflections}",
        f"suppressed_turns={suppressed_turns}",
        f"reflection_rearms={reflection_rearms}",
        "transfers=0", f"waves={waves}", "iterations=2304", "segments=1",
        "path=1.00000000000000000e+00",
        "dispersion=0.00000000000000000e+00",
    ))
    return "laser: " + " ".join(fields) + "\n"


def test_schema_and_required_checks_match_launcher():
    assert gate.SCHEMA == 8
    assert run_case.PRODUCTION_GATE_SCHEMA == gate.SCHEMA
    assert run_case.CALIBRATION_CYCLES == gate.CALIBRATION_CYCLES == 22
    assert "physical_light_speed_sensitivity" in gate.CHECK_NAMES
    assert tuple(run_case.REQUIRED_PRODUCTION_CHECKS) == gate.CHECK_NAMES


def write_history(path: Path, scale: float = 1.0,
                  final_time: float = 1.0e-4,
                  eos_bad_cells: float = 0.0,
                  eos_floor_cells: float = 0.0,
                  radiation_scale: float = 1.0,
                  centroid_scale: float = 1.0) -> None:
    names = ["time", "dt", "laser_Edep", "rad_Pesc", "CH_mass", "eion_E",
             "eele_E", "erad_E", "erad_soft", "erad_mid", "erad_hard",
             "chain_E", "laser_x", "eos_floor", "eos_bad"]
    chain_delta = (1.0e-3-0.5*2.0e-4*final_time)*scale
    rows = (
        (0.0, 1e-4, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, .3, .4, .3,
         3.0, 0.0, 0.0, 0.0),
        (final_time, 1e-4, 1e-3*scale, 2e-4*scale, 2.0,
         1.0+2e-4*scale, 1.0+4e-4*scale,
         1.0+1e-6*radiation_scale, .3, .4, .3+1e-6*radiation_scale,
         3.0+chain_delta, -5e-4*scale*centroid_scale,
         eos_floor_cells, eos_bad_cells),
    )
    header = "# " + " ".join(f"[{index}]={name}" for index, name in enumerate(names, 1))
    path.write_text(header+"\n"+"\n".join(
        " ".join(f"{value:.17e}" for value in row) for row in rows)+"\n")


def test_history_parser_and_sensitivity(tmp_path):
    first = tmp_path/"first.hst"
    second = tmp_path/"second.hst"
    write_history(first)
    write_history(second, 1.1)
    reference = gate.read_history(first)
    comparison = gate.read_history(second)
    assert reference["CH_mass"] == [2.0, 2.0]
    metrics = gate.history_sensitivity(reference, comparison)
    assert metrics["common_time"] == 1.0e-4
    assert 0.0 < metrics["laser_Edep"] < 0.1
    assert 0.0 < metrics["eion_E"] < 0.1
    assert 0.0 < metrics["eion_absolute_difference_over_deposited"] < 0.1
    assert metrics["laser_centroid"] <= 1.0e-12
    assert metrics["erad_E"] == 0.0
    assert metrics["erad_absolute_difference_over_deposited"] == 0.0


def test_sensitivity_policy_gates_matter_and_absolute_radiation_only():
    settings = {
        "sensitivity_relative_tolerance": 0.05,
        "radiation_deposited_relative_tolerance": 0.01,
    }
    metrics = {field: 0.01 for field in gate.SENSITIVITY_RELATIVE_FIELDS}
    metrics.update(
        common_time=1.0,
        erad_E=0.99,
        erad_absolute_difference_over_deposited=0.009,
    )
    assert gate.sensitivity_is_accepted(metrics, settings)

    for field in gate.SENSITIVITY_RELATIVE_FIELDS:
        failed = dict(metrics)
        failed[field] = 0.051
        assert not gate.sensitivity_is_accepted(failed, settings), field

    failed = dict(metrics)
    failed["erad_absolute_difference_over_deposited"] = 0.011
    assert not gate.sensitivity_is_accepted(failed, settings)


def test_cycle_parser(tmp_path):
    log = tmp_path/"phase.log"
    log.write_text(
        "cycle=0 time=0.000000e+00 dt=2.000000e-04\n"
        "cycle=50 time=1.000000e-02 dt=1.500000e-04\n")
    rows = gate.parse_cycle_log(log)
    assert rows[0] == {"cycle": 0, "time": 0.0, "dt": 2.0e-4}
    assert rows[-1]["cycle"] == 50


def test_laser_parser_reads_every_total_and_split_diagnostic(tmp_path):
    log = tmp_path/"phase.log"
    log.write_text(
        "unrelated output\n"
        + laser_diagnostic(remaining=1.0e-11)
        + "cycle=1 time=1e-4 dt=1e-4\n"
        + laser_diagnostic(
            remaining=5.0e-11,
            wave_remaining=2.0e-11,
            reflection_remaining=3.0e-11,
        )
    )
    rows = gate.parse_laser_diagnostics(log)
    assert len(rows) == 2
    assert rows[0]["line_number"] == 2
    assert rows[0]["remaining"] == 1.0e-11
    assert rows[0]["waves"] == 9
    assert isinstance(rows[0]["waves"], int)
    assert "wave_remaining" not in rows[0]
    assert rows[1]["line_number"] == 4
    assert rows[1]["wave_remaining"] == 2.0e-11
    assert rows[1]["reflection_remaining"] == 3.0e-11
    assert rows[1]["wave_remaining_rays"] == 1
    assert rows[1]["max_reflections"] == 8
    assert rows[1]["suppressed_turns"] == 3
    assert rows[1]["reflection_rearms"] == 2


def test_laser_parser_requires_reflection_behavior_counters(tmp_path):
    log = tmp_path/"phase.log"
    log.write_text(
        laser_diagnostic(
            wave_remaining=0.0,
            reflection_remaining=0.0,
        ).replace(" max_reflections=8", "")
    )
    passed, metrics = gate.laser_remainder_metrics({"phase": log}, 1.0e-10)
    assert not passed
    assert "max_reflections" in metrics["parse_errors"]["phase"]


def test_laser_parser_requires_positive_integral_transport_waves(tmp_path):
    log = tmp_path/"phase.log"
    diagnostic = laser_diagnostic(
        wave_remaining=0.0,
        reflection_remaining=0.0,
    )
    log.write_text(diagnostic.replace(" waves=9", ""))
    passed, metrics = gate.laser_remainder_metrics({"phase": log}, 1.0e-10)
    assert not passed
    assert "waves" in metrics["parse_errors"]["phase"]

    log.write_text(diagnostic.replace("waves=9", "waves=9.5"))
    passed, metrics = gate.laser_remainder_metrics({"phase": log}, 1.0e-10)
    assert not passed
    assert "Invalid laser count waves" in metrics["parse_errors"]["phase"]

    log.write_text(diagnostic.replace("waves=9", "waves=0"))
    passed, metrics = gate.laser_remainder_metrics({"phase": log}, 1.0e-10)
    assert not passed
    assert "Invalid laser count waves" in metrics["parse_errors"]["phase"]


def test_laser_remainder_metrics_gate_every_record_and_require_each_log(tmp_path):
    phase1 = tmp_path/"phase1.log"
    phase2 = tmp_path/"phase2.log"
    phase1.write_text(laser_diagnostic(
        remaining=1.0e-10,
        wave_remaining=1.0e-10,
        reflection_remaining=0.0,
    ))
    phase2.write_text(laser_diagnostic(
        remaining=1.0e-10,
        wave_remaining=4.0e-11,
        reflection_remaining=6.0e-11,
    ))
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert passed
    assert metrics["diagnostic_count"] == 2
    assert metrics["split_diagnostic_count"] == 2
    assert metrics["split_diagnostics_complete"]
    assert metrics["maximum_observed_transport_waves"] == 9
    assert metrics["maximum_total_remainder_fraction"] == 1.0e-10
    assert 0.999e-10 <= metrics["maximum_split_remainder_fraction"] <= 1.0e-10

    phase1.write_text(laser_diagnostic(remaining=0.0))
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert not passed
    assert not metrics["split_diagnostics_complete"]

    phase1.write_text(laser_diagnostic(
        remaining=0.0,
        wave_remaining=0.0,
        reflection_remaining=0.0,
        residual=2.0e-10,
    ))
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert not passed
    assert metrics["maximum_laser_conservation_residual"] == 2.0e-10

    # This is the final diagnostic from the old production run. Its 44.08%
    # remainder was included in conservation accounting and therefore masked by
    # the old deposited-energy-only closure check.
    phase1.write_text(
        laser_diagnostic(
            remaining=0.0,
            wave_remaining=0.0,
            reflection_remaining=0.0,
        )
        + "laser: launched=1.81818181818178609e-03 "
        "deposited=4.64369493175387910e-04 "
        "escaped=5.52403361617890502e-04 "
        "remaining=8.01408963388458183e-04 "
        "residual=2.71917904859369308e-14 active=2067 reflected=3929 "
        "max_reflections=64 suppressed_turns=0 reflection_rearms=0 "
        "transfers=10890 waves=1024 iterations=262144 segments=1965976 "
        "path=1.19796497351896141e+04 "
        "dispersion=0.00000000000000000e+00\n"
    )
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert not passed
    assert metrics["diagnostic_count"] == 3
    assert metrics["violating_diagnostic_count"] == 1
    assert 0.4407 < metrics["maximum_total_remainder_fraction"] < 0.4409
    assert metrics["worst_source_id"] == "smoke_phase1_log"
    assert metrics["worst_line_number"] == 2

    # A misleading zero aggregate cannot hide a nonzero split remainder.
    phase2.write_text(laser_diagnostic(
        remaining=0.0,
        wave_remaining=0.0,
        reflection_remaining=0.0,
    ))
    phase1.write_text(laser_diagnostic(
        remaining=0.0,
        wave_remaining=2.0e-10,
        reflection_remaining=0.0,
    ))
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert not passed
    assert metrics["maximum_total_remainder_fraction"] == 0.0
    assert metrics["maximum_split_remainder_fraction"] == 2.0e-10

    phase2.write_text("cycle=1 time=1e-4 dt=1e-4\n")
    passed, metrics = gate.laser_remainder_metrics(
        {"smoke_phase1_log": phase1, "smoke_phase2_log": phase2}, 1.0e-10
    )
    assert not passed
    assert not metrics["diagnostics_present"]
    assert metrics["diagnostic_counts_by_source"]["smoke_phase2_log"] == 0


def test_reset_aware_cumulative_delta():
    assert gate.reset_aware_cumulative_delta([0.0, 0.4, 1.0, 0.1, 0.3]) == 1.3
    history = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0],
        "laser_Edep": [0.0, 0.4, 1.0, 0.1, 0.3],
        "laser_x": [0.0, -0.2, -0.5, -0.05, -0.15],
    }
    assert gate.reset_aware_value_at(history, "laser_Edep", 1.5) == 0.7
    assert gate.reset_aware_value_at(history, "laser_Edep", 4.0) == 1.3
    assert gate.reset_aware_value_at(history, "laser_x", 4.0) == -0.65


def test_source_record_is_content_addressed(tmp_path):
    output = tmp_path/"production_gate.json"
    source = tmp_path/"evidence.json"
    source.write_text('{"passed": true}\n')
    record = gate.source_record(source, output)
    assert record["path"] == "evidence.json"
    assert record["sha256"] == gate.sha256_path(source)
    source.write_text('{"passed": false}\n')
    assert record["sha256"] != gate.sha256_path(source)


def test_deck_value(tmp_path):
    deck = tmp_path/"case.athinput"
    deck.write_text("<laser>\nbeam0_power = 2.0e19 # erg/s\n")
    assert gate.read_deck_value(deck, "beam0_power") == 2.0e19


def test_smoke_enables_checkpoint_full_volume_outputs():
    overrides = run_case.nonproduction_overrides("smoke", None, None, 1)
    for number in (8, 9, 10):
        assert f"output{number}/dt=1.0" in overrides


def test_explicit_cycle_smoke_keeps_history_only():
    overrides = run_case.nonproduction_overrides("smoke", 60, 299.792458, 1)
    assert "output1/dt=1.0e-4" in overrides
    assert "output9/dt=1.0" not in overrides


def test_smoke_restart_runbook_requests_terminal_3t_volume():
    overrides = run_case.smoke_restart_overrides(None, 1)
    assert "time/nlim=60" in overrides
    assert "output9/dt=1.0" in overrides


def test_full_volume_glob_excludes_plane_slices(tmp_path):
    sliced = tmp_path/"case.three_t_xy.00002.bin"
    volume = tmp_path/"case.three_t.00001.bin"
    sliced.touch()
    volume.touch()
    assert gate.latest_file(tmp_path, "*.three_t.*.bin") == volume


def test_discover_sources_finds_root_level_histories(tmp_path):
    smoke = tmp_path/"smoke"
    resolution = tmp_path/"resolution2"
    rsla = tmp_path/"rsla10"
    physical = tmp_path/"cphys650"
    calibration = tmp_path/"calibrate"
    for directory in (smoke, resolution, rsla, physical, calibration):
        directory.mkdir()

    for directory in (smoke, resolution, rsla):
        (directory/"run_status.json").touch()
        (directory/"phase1.log").touch()
        (directory/"phase2.log").touch()
        (directory/f"{directory.name}.user.hst").touch()
    (physical/"run_status.json").touch()
    (physical/"phase1.log").touch()
    (physical/"cphys.user.hst").touch()
    (calibration/"run_status.json").touch()
    (calibration/"phase1.log").touch()

    (smoke/"bin").mkdir()
    (smoke/"bin"/"case.fluid.00000.bin").touch()
    (smoke/"bin"/"case.three_t.00000.bin").touch()
    (smoke/"bin"/"case.laser.00000.bin").touch()
    (smoke/"rst").mkdir()
    (smoke/"rst"/"case.00000.rst").touch()
    (smoke/"material_tables").mkdir()
    (smoke/"material_tables"/"manifest.json").touch()

    sources = gate.discover_sources(
        smoke, resolution, rsla, physical, calibration
    )
    assert sources["smoke_history"].parent == smoke
    assert sources["resolution_history"].parent == resolution
    assert sources["rsla_history"].parent == rsla
    assert sources["physical_history"].parent == physical
    assert all(path.parent.name != "hst" for name, path in sources.items()
               if name.endswith("_history"))


def test_all_checks_are_derived_from_artifacts(tmp_path, monkeypatch):
    artifacts = {"athena_binary": "abc123"}

    def status(mode, scale=1, c_light=None):
        return {
            "mode": mode, "compact_scale": scale,
            "radiation_c_light_override": c_light,
            "phase1_exit_code": 0, "phase2_exit_code": 0,
            "case_artifacts": artifacts,
            "phase1_mpi_command": ["time/nlim=50"],
            "baseline_processes": {
                str(index): [] for index in range(gate.ACCEPTANCE_RANKS)
            },
        }

    smoke_status = status("smoke")
    resolution_status = status("smoke", scale=2)
    rsla_status = status("smoke", c_light=10.0)
    physical_status = status("smoke", c_light=299.792458)
    physical_status["phase1_mpi_command"] = ["time/nlim=650"]
    calibration_status = status("calibrate")
    calibration_status["phase1_mpi_command"] = ["time/nlim=22"]
    calibration_status["phase1_memory"] = {
        "errors": [],
        "devices": {
            str(index): {
                "name": "Tesla V100-SXM2-16GB", "peak_fraction": 0.7,
                "within_60_80_percent": True,
            } for index in range(gate.ACCEPTANCE_RANKS)
        },
    }

    sources = {}
    for name, value in (
        ("smoke_status", smoke_status),
        ("resolution_status", resolution_status),
        ("rsla_status", rsla_status),
        ("physical_status", physical_status),
        ("calibration_status", calibration_status),
    ):
        path = tmp_path/f"{name}.json"
        path.write_text(json.dumps(value))
        sources[name] = path

    for name, rows in (
        ("smoke_phase1_log", ((0, 0.0, 2e-4), (50, .008, 2e-4))),
        ("smoke_phase2_log", ((50, .008, 2e-4), (60, .01, 2e-4))),
        ("resolution_phase1_log", ((0, 0.0, 1e-4), (100, .008, 1e-4))),
        ("resolution_phase2_log", ((100, .008, 1e-4), (120, .01, 1e-4))),
        ("rsla_phase1_log", ((0, 0.0, 2e-4), (50, .008, 2e-4))),
        ("rsla_phase2_log", ((50, .008, 2e-4), (60, .01, 2e-4))),
        ("physical_phase1_log", ((0, 0.0, 2e-5), (650, .012, 2e-5))),
        ("calibration_phase1_log", ((0, 0.0, 2e-4), (22, 4.4e-3, 2e-4))),
    ):
        path = tmp_path/f"{name}.log"
        text = "".join(
            f"cycle={cycle} time={time:.8e} dt={dt:.8e}\n"
            for cycle, time, dt in rows)
        if name in gate.LASER_DIAGNOSTIC_SOURCE_IDS:
            diagnostic = laser_diagnostic(
                remaining=5.0e-11,
                wave_remaining=2.0e-11,
                reflection_remaining=3.0e-11,
                waves=(10 if name == "calibration_phase1_log" else 9),
                max_reflections=(32 if name == "calibration_phase1_log" else 8),
            )
            text += diagnostic*(44 if name == "calibration_phase1_log" else 1)
        path.write_text(text)
        sources[name] = path

    for name in (
        "smoke_history", "resolution_history", "rsla_history", "physical_history"
    ):
        path = tmp_path/f"{name}.hst"
        write_history(path, final_time=1.0e-2)
        sources[name] = path

    deck = tmp_path/"dci.athinput"
    deck.write_text(
        "<mesh>\nx1min = -0.45\nx1max = 0.75\nx2min = -0.6\nx2max = 0.6\n"
        "x3min = -0.6\nx3max = 0.6\n"
        "<thermal_radiation>\nc_light = 30\nn_groups = 20\n"
        "<laser>\nnbeams = 4\npulse0_nsections = 2\n"
        "pulse0_time_0 = 0\npulse0_time_1 = 1\n"
        "pulse0_power_0 = 1.77455e19\npulse0_power_1 = 1.77455e19\n"
        "max_reflections_per_ray = 64\n"
        "max_mpi_waves = 1024\n"
        "beam0_start_time = 0\nbeam0_end_time = 5\n")
    sources["production_input"] = deck
    sources["calibration_input"] = deck
    for name in ("smoke_fluid_volume", "smoke_3t_volume",
                 "smoke_laser_volume", "smoke_restart",
                 "smoke_material_manifest"):
        path = tmp_path/name
        path.write_bytes(b"synthetic evidence\n")
        sources[name] = path

    monkeypatch.setattr(gate, "check_3t_binary", lambda path: {
        "cycle": 50, "time": .008, "grid_nx1": 100, "grid_nx2": 64,
        "grid_nx3": 64, "meshblock_count": 8, "cell_count": 100*64*64,
        "radiation_group_count": 20, "minimum": 0.0,
        "eos_trace_cell_count": 1000, "eos_energy_floor_cell_count": 1000,
        "eos_disallowed_cell_count": 0,
        "eos_maximum_flag": 1,
        "negative_tolerance": 1e-14})
    settings = {
        "artifacts": artifacts,
        "energy_relative_tolerance": 5e-4,
        "mass_relative_tolerance": 1e-8,
        "resolution_relative_tolerance": .35,
        "sensitivity_relative_tolerance": .05,
        "radiation_deposited_relative_tolerance": .01,
        "minimum_causal_dt_fraction": 1e-4,
        "maximum_eos_energy_floor_fraction": .05,
    }
    checks = gate.evaluate_checks(sources, settings)
    assert set(checks) == set(gate.CHECK_NAMES)
    assert all(record["passed"] for record in checks.values()), checks
    memory_check = checks["gpu_memory_60_80_all"]
    assert memory_check["metrics"]["requested_cycle_limit"] == 22
    assert memory_check["metrics"]["final_cycle"] == 22
    assert memory_check["metrics"]["laser_diagnostic_count"] == 44
    assert checks["reduced_light_speed_sensitivity"]["metrics"]["erad_E"] == 0.0

    calibration_status["phase1_mpi_command"] = ["time/nlim=21"]
    sources["calibration_status"].write_text(json.dumps(calibration_status))
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["gpu_memory_60_80_all"]["passed"]
    calibration_status["phase1_mpi_command"] = ["time/nlim=22"]
    sources["calibration_status"].write_text(json.dumps(calibration_status))

    calibration_log_text = sources["calibration_phase1_log"].read_text()
    sources["calibration_phase1_log"].write_text(
        calibration_log_text.replace(
            "cycle=22 time=4.40000000e-03",
            "cycle=21 time=4.20000000e-03",
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["gpu_memory_60_80_all"]["passed"]
    sources["calibration_phase1_log"].write_text(calibration_log_text)

    sources["calibration_phase1_log"].write_text(
        calibration_log_text.replace(
            "cycle=22 time=4.40000000e-03",
            "cycle=23 time=4.60000000e-03",
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["gpu_memory_60_80_all"]["passed"]
    sources["calibration_phase1_log"].write_text(calibration_log_text)

    sources["calibration_phase1_log"].write_text(
        calibration_log_text.replace(
            laser_diagnostic(
                remaining=5.0e-11,
                wave_remaining=2.0e-11,
                reflection_remaining=3.0e-11,
                waves=10,
                max_reflections=32,
            ),
            "",
            1,
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["gpu_memory_60_80_all"]["passed"]
    sources["calibration_phase1_log"].write_text(calibration_log_text)

    sources["calibration_phase1_log"].write_text(
        calibration_log_text + laser_diagnostic(
            remaining=5.0e-11,
            wave_remaining=2.0e-11,
            reflection_remaining=3.0e-11,
            waves=10,
            max_reflections=32,
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["gpu_memory_60_80_all"]["passed"]
    sources["calibration_phase1_log"].write_text(calibration_log_text)
    checks = gate.evaluate_checks(sources, settings)
    assert checks["physical_light_speed_sensitivity"]["metrics"]["final_cycle"] == 650
    closure = checks["laser_and_boundary_energy_closure"]
    assert closure["metrics"]["diagnostic_count"] == 51
    assert closure["metrics"]["maximum_remainder_fraction"] == 5.0e-11
    assert closure["metrics"]["maximum_observed_reflections"] == 32
    assert closure["metrics"]["maximum_observed_transport_waves"] == 10
    assert closure["metrics"]["configured_max_reflections_per_ray"] == 64
    assert closure["metrics"]["maximum_allowed_observed_reflections"] == 32.0
    assert closure["metrics"]["reflection_headroom_passed"]
    assert closure["metrics"]["configured_max_mpi_waves"] == 1024
    assert closure["metrics"]["maximum_allowed_observed_transport_waves"] == 512.0
    assert closure["metrics"]["transport_wave_headroom_passed"]
    assert set(closure["source_ids"]) >= set(gate.LASER_DIAGNOSTIC_SOURCE_IDS)

    resolution_phase1_text = sources["resolution_phase1_log"].read_text()
    sources["resolution_phase1_log"].write_text(
        resolution_phase1_text
        + laser_diagnostic(
            wave_remaining=0.0,
            reflection_remaining=0.0,
            max_reflections=33,
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    closure = checks["laser_and_boundary_energy_closure"]
    assert not closure["passed"]
    assert closure["metrics"]["maximum_observed_reflections"] == 33
    assert not closure["metrics"]["reflection_headroom_passed"]
    sources["resolution_phase1_log"].write_text(resolution_phase1_text)

    sources["resolution_phase1_log"].write_text(
        resolution_phase1_text
        + laser_diagnostic(
            wave_remaining=0.0,
            reflection_remaining=0.0,
            waves=512,
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    closure = checks["laser_and_boundary_energy_closure"]
    assert closure["passed"]
    assert closure["metrics"]["maximum_observed_transport_waves"] == 512
    assert closure["metrics"]["transport_wave_headroom_passed"]
    sources["resolution_phase1_log"].write_text(resolution_phase1_text)

    sources["resolution_phase1_log"].write_text(
        resolution_phase1_text
        + laser_diagnostic(
            wave_remaining=0.0,
            reflection_remaining=0.0,
            waves=513,
        )
    )
    checks = gate.evaluate_checks(sources, settings)
    closure = checks["laser_and_boundary_energy_closure"]
    assert not closure["passed"]
    assert closure["metrics"]["maximum_observed_transport_waves"] == 513
    assert not closure["metrics"]["transport_wave_headroom_passed"]
    sources["resolution_phase1_log"].write_text(resolution_phase1_text)

    smoke_phase1_text = sources["smoke_phase1_log"].read_text()
    sources["smoke_phase1_log"].write_text(
        smoke_phase1_text
        + "laser: launched=1.81818181818178609e-03 "
        "deposited=4.64369493175387910e-04 "
        "escaped=5.52403361617890502e-04 "
        "remaining=8.01408963388458183e-04 "
        "residual=2.71917904859369308e-14 active=2067 reflected=3929 "
        "max_reflections=64 suppressed_turns=0 reflection_rearms=0 "
        "transfers=10890 waves=1024 iterations=262144 segments=1965976 "
        "path=1.19796497351896141e+04 "
        "dispersion=0.00000000000000000e+00\n"
    )
    checks = gate.evaluate_checks(sources, settings)
    closure = checks["laser_and_boundary_energy_closure"]
    assert not closure["passed"]
    assert closure["metrics"]["relative_residual"] <= settings["energy_relative_tolerance"]
    assert 0.4407 < closure["metrics"]["maximum_remainder_fraction"] < 0.4409
    sources["smoke_phase1_log"].write_text(smoke_phase1_text)

    smoke_phase2_text = sources["smoke_phase2_log"].read_text()
    sources["smoke_phase2_log"].write_text(
        "cycle=50 time=8.00000000e-03 dt=2.00000000e-04\n"
        "cycle=60 time=1.00000000e-02 dt=2.00000000e-04\n"
    )
    checks = gate.evaluate_checks(sources, settings)
    closure = checks["laser_and_boundary_energy_closure"]
    assert not closure["passed"]
    assert not closure["metrics"]["diagnostics_present"]
    sources["smoke_phase2_log"].write_text(smoke_phase2_text)

    physical_status["phase1_mpi_command"] = ["time/nlim=649"]
    sources["physical_status"].write_text(json.dumps(physical_status))
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]
    physical_status["phase1_mpi_command"] = ["time/nlim=650"]
    sources["physical_status"].write_text(json.dumps(physical_status))

    sources["physical_phase1_log"].write_text(
        "cycle=0 time=0.00000000e+00 dt=2.00000000e-05\n"
        "cycle=649 time=1.20000000e-02 dt=2.00000000e-05\n"
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]
    sources["physical_phase1_log"].write_text(
        "cycle=0 time=0.00000000e+00 dt=2.00000000e-05\n"
        "cycle=650 time=1.20000000e-02 dt=2.00000000e-05\n"
    )

    physical_status["radiation_c_light_override"] = 300.0
    sources["physical_status"].write_text(json.dumps(physical_status))
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]
    physical_status["radiation_c_light_override"] = 299.792458
    sources["physical_status"].write_text(json.dumps(physical_status))

    rsla_status["radiation_c_light_override"] = 30.0
    sources["rsla_status"].write_text(json.dumps(rsla_status))
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["reduced_light_speed_sensitivity"]["passed"]
    rsla_status["radiation_c_light_override"] = 10.0
    sources["rsla_status"].write_text(json.dumps(rsla_status))

    write_history(
        sources["physical_history"], final_time=1.0e-2, eos_bad_cells=1.0
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]
    write_history(
        sources["physical_history"], final_time=1.0e-2,
        eos_floor_cells=0.06*100*64*64
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]
    write_history(sources["physical_history"], final_time=1.0e-2)

    deck.write_text(
        "<mesh>\nx1min = -0.45\nx1max = 0.75\nx2min = -0.6\nx2max = 0.6\n"
        "x3min = -0.6\nx3max = 0.6\n"
        "<thermal_radiation>\nc_light = 10\nn_groups = 20\n"
        "<laser>\nnbeams = 4\npulse0_nsections = 2\n"
        "pulse0_time_0 = 0\npulse0_time_1 = 1\n"
        "pulse0_power_0 = 1.77455e19\npulse0_power_1 = 1.77455e19\n"
        "max_reflections_per_ray = 64\n"
        "max_mpi_waves = 1024\n"
        "beam0_start_time = 0\nbeam0_end_time = 5\n")
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["compact_20group_50step"]["passed"]
    deck.write_text(
        "<mesh>\nx1min = -0.45\nx1max = 0.75\nx2min = -0.6\nx2max = 0.6\n"
        "x3min = -0.6\nx3max = 0.6\n"
        "<thermal_radiation>\nc_light = 30\nn_groups = 20\n"
        "<laser>\nnbeams = 4\npulse0_nsections = 2\n"
        "pulse0_time_0 = 0\npulse0_time_1 = 1\n"
        "pulse0_power_0 = 1.77455e19\npulse0_power_1 = 1.77455e19\n"
        "max_reflections_per_ray = 64\n"
        "max_mpi_waves = 1024\n"
        "beam0_start_time = 0\nbeam0_end_time = 5\n")

    write_history(
        sources["smoke_history"], final_time=1.0e-2, eos_bad_cells=1.0
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["finite_nonnegative_3t"]["passed"]

    write_history(sources["smoke_history"], final_time=1.0e-2)
    write_history(sources["rsla_history"], scale=1.2, final_time=1.0e-2)
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["reduced_light_speed_sensitivity"]["passed"]

    write_history(sources["rsla_history"], final_time=1.0e-2)
    write_history(sources["physical_history"], scale=1.2, final_time=1.0e-2)
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]

    write_history(sources["physical_history"], final_time=1.0e-2)
    write_history(
        sources["rsla_history"], final_time=1.0e-2, radiation_scale=5.0
    )
    checks = gate.evaluate_checks(sources, settings)
    rsla_check = checks["reduced_light_speed_sensitivity"]
    assert rsla_check["passed"]
    assert rsla_check["metrics"]["erad_E"] > 0.5
    assert rsla_check["metrics"]["erad_absolute_difference_over_deposited"] < .01

    write_history(
        sources["rsla_history"], final_time=1.0e-2, radiation_scale=20.0
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["reduced_light_speed_sensitivity"]["passed"]

    write_history(sources["rsla_history"], final_time=1.0e-2)
    write_history(sources["physical_history"], final_time=5.0e-3)
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["physical_light_speed_sensitivity"]["passed"]

    write_history(
        sources["smoke_history"], final_time=1.0e-2,
        eos_floor_cells=0.06*100*64*64
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["finite_nonnegative_3t"]["passed"]
