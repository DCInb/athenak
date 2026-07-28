"""Deterministic unit tests for the DCI production-gate verifier."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_production_gate as gate
import run_case


def write_history(path: Path, scale: float = 1.0,
                  final_time: float = 1.0e-4,
                  eos_bad_cells: float = 0.0,
                  eos_floor_cells: float = 0.0) -> None:
    names = ["time", "dt", "laser_Edep", "rad_Pesc", "CH_mass", "eion_E",
             "eele_E", "erad_E", "erad_soft", "erad_mid", "erad_hard",
             "chain_E", "laser_x", "eos_floor", "eos_bad"]
    chain_delta = (1.0e-3-0.5*2.0e-4*final_time)*scale
    rows = (
        (0.0, 1e-4, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, .3, .4, .3,
         3.0, 0.0, 0.0, 0.0),
        (final_time, 1e-4, 1e-3*scale, 2e-4*scale, 2.0, 1.0,
         1.0+4e-4*scale, 1.0+4e-4*scale, .3, .4, .3004,
         3.0+chain_delta, -5e-4*scale, eos_floor_cells, eos_bad_cells),
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
    assert metrics["laser_centroid"] <= 1.0e-12


def test_cycle_parser(tmp_path):
    log = tmp_path/"phase.log"
    log.write_text(
        "cycle=0 time=0.000000e+00 dt=2.000000e-04\n"
        "cycle=50 time=1.000000e-02 dt=1.500000e-04\n")
    rows = gate.parse_cycle_log(log)
    assert rows[0] == {"cycle": 0, "time": 0.0, "dt": 2.0e-4}
    assert rows[-1]["cycle"] == 50


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


def test_all_checks_are_derived_from_artifacts(tmp_path, monkeypatch):
    artifacts = {"athena_binary": "abc123"}

    def status(mode, scale=1, c_light=None):
        return {
            "mode": mode, "compact_scale": scale,
            "radiation_c_light_override": c_light,
            "phase1_exit_code": 0, "phase2_exit_code": 0,
            "case_artifacts": artifacts,
            "phase1_mpi_command": ["time/nlim=50"],
            "baseline_processes": {str(index): [] for index in range(8)},
        }

    smoke_status = status("smoke")
    resolution_status = status("smoke", scale=2)
    rsla_status = status("smoke", c_light=30.0)
    calibration_status = status("calibrate")
    calibration_status["phase1_mpi_command"] = ["time/nlim=2"]
    calibration_status["phase1_memory"] = {
        "errors": [],
        "devices": {
            str(index): {
                "name": "Tesla V100-SXM2-16GB", "peak_fraction": 0.7,
                "within_60_80_percent": True,
            } for index in range(8)
        },
    }

    sources = {}
    for name, value in (
        ("smoke_status", smoke_status),
        ("resolution_status", resolution_status),
        ("rsla_status", rsla_status),
        ("calibration_status", calibration_status),
    ):
        path = tmp_path/f"{name}.json"
        import json
        path.write_text(json.dumps(value))
        sources[name] = path

    for name, rows in (
        ("smoke_phase1_log", ((0, 0.0, 2e-4), (50, .008, 2e-4))),
        ("smoke_phase2_log", ((50, .008, 2e-4), (60, .01, 2e-4))),
        ("resolution_phase1_log", ((0, 0.0, 1e-4), (100, .008, 1e-4))),
        ("resolution_phase2_log", ((100, .008, 1e-4), (120, .01, 1e-4))),
        ("rsla_phase1_log", ((0, 0.0, 7e-5), (150, .008, 7e-5))),
        ("rsla_phase2_log", ((150, .008, 7e-5), (180, .01, 7e-5))),
        ("calibration_phase1_log", ((0, 0.0, 2e-4), (2, 4e-4, 2e-4))),
    ):
        path = tmp_path/f"{name}.log"
        path.write_text("".join(
            f"cycle={cycle} time={time:.8e} dt={dt:.8e}\n"
            for cycle, time, dt in rows))
        sources[name] = path

    for name in ("smoke_history", "resolution_history", "rsla_history"):
        path = tmp_path/f"{name}.hst"
        write_history(path, final_time=1.0e-2)
        sources[name] = path

    deck = tmp_path/"dci.athinput"
    deck.write_text(
        "<thermal_radiation>\nc_light = 10\nn_groups = 20\n"
        "<laser>\nbeam0_power = 2e19\n"
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
        "rsla_relative_tolerance": .30,
        "minimum_causal_dt_fraction": 1e-4,
        "maximum_eos_energy_floor_fraction": .05,
    }
    checks = gate.evaluate_checks(sources, settings)
    assert set(checks) == set(gate.CHECK_NAMES)
    assert all(record["passed"] for record in checks.values()), checks

    write_history(
        sources["smoke_history"], final_time=1.0e-2, eos_bad_cells=1.0
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["finite_nonnegative_3t"]["passed"]

    write_history(
        sources["smoke_history"], final_time=1.0e-2,
        eos_floor_cells=0.06*100*64*64
    )
    checks = gate.evaluate_checks(sources, settings)
    assert not checks["finite_nonnegative_3t"]["passed"]
