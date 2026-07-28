#!/usr/bin/env python3
"""Derive the DCI_3D production gate from immutable run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
SCHEMA = 4
PRODUCTION_C_LIGHT = 30.0
RSLA_COMPARISON_C_LIGHT = 10.0
PHYSICAL_C_LIGHT = 299.792458
SENSITIVITY_RELATIVE_FIELDS = (
    "laser_Edep",
    "eion_E",
    "eele_E",
    "chain_E",
    "laser_centroid",
)
RESOLUTION_RELATIVE_FIELDS = (
    "laser_Edep",
    "eele_E",
    "erad_E",
    "chain_E",
    "laser_centroid",
)
CHECK_NAMES = (
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def unique_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} below {directory}, found {len(matches)}"
        )
    return matches[0]


def latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise ValueError(f"No {pattern!r} files below {directory}")
    return matches[-1]


def discover_sources(
    smoke: Path,
    resolution: Path,
    rsla: Path,
    physical: Path,
    calibration: Path,
) -> dict[str, Path]:
    sources = {
        "production_input": CASE_DIR / "dci_3d.athinput",
        "calibration_input": CASE_DIR / "dci_3d_calibration.athinput",
        "smoke_status": smoke / "run_status.json",
        "smoke_phase1_log": smoke / "phase1.log",
        "smoke_phase2_log": smoke / "phase2.log",
        "smoke_history": unique_file(smoke, "*.hst"),
        "smoke_fluid_volume": latest_file(smoke / "bin", "*.fluid.*.bin"),
        "smoke_3t_volume": latest_file(smoke / "bin", "*.three_t.*.bin"),
        "smoke_laser_volume": latest_file(smoke / "bin", "*.laser.*.bin"),
        "smoke_restart": latest_file(smoke / "rst", "*.rst"),
        "smoke_material_manifest": smoke / "material_tables" / "manifest.json",
        "resolution_status": resolution / "run_status.json",
        "resolution_phase1_log": resolution / "phase1.log",
        "resolution_phase2_log": resolution / "phase2.log",
        "resolution_history": unique_file(resolution, "*.hst"),
        "rsla_status": rsla / "run_status.json",
        "rsla_phase1_log": rsla / "phase1.log",
        "rsla_phase2_log": rsla / "phase2.log",
        "rsla_history": unique_file(rsla, "*.hst"),
        "physical_status": physical / "run_status.json",
        "physical_phase1_log": physical / "phase1.log",
        "physical_history": unique_file(physical, "*.hst"),
        "calibration_status": calibration / "run_status.json",
        "calibration_phase1_log": calibration / "phase1.log",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing production-gate evidence:\n" + "\n".join(missing))
    return sources


def parse_cycle_log(path: Path) -> list[dict[str, float | int]]:
    pattern = re.compile(
        r"cycle=(\d+)\s+time=([+\-0-9.eE]+)\s+dt=([+\-0-9.eE]+)"
    )
    rows = []
    for cycle, time, dt in pattern.findall(path.read_text(encoding="utf-8")):
        rows.append({"cycle": int(cycle), "time": float(time), "dt": float(dt)})
    if not rows:
        raise ValueError(f"No cycle diagnostics in {path}")
    return rows


def read_history(path: Path) -> dict[str, list[float]]:
    names: list[str] | None = None
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and "[1]=" in line and names is None:
            names = re.findall(r"\[\d+\]=([^\s]+)", line)
        elif line.strip() and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    if not names or not rows or any(len(row) != len(names) for row in rows):
        raise ValueError(f"Malformed history file: {path}")
    result = {name: [row[index] for row in rows] for index, name in enumerate(names)}
    if any(not math.isfinite(value) for values in result.values() for value in values):
        raise ValueError(f"Nonfinite history value: {path}")
    return result


def interpolate(history: dict[str, list[float]], field: str, target: float) -> float:
    times = history["time"]
    values = history[field]
    if target <= times[0]:
        return values[0]
    for index in range(1, len(times)):
        if target <= times[index]:
            span = times[index]-times[index-1]
            fraction = 0.0 if span == 0.0 else (target-times[index-1])/span
            return values[index-1]+fraction*(values[index]-values[index-1])
    return values[-1]


def reset_aware_series(
    values: list[float], reset_indicator: list[float]
) -> list[float]:
    """Turn a process-local cumulative field into a restart-continuous series."""
    if not values or len(values) != len(reset_indicator):
        raise ValueError("Cumulative diagnostic samples are missing or mismatched")
    if any(not math.isfinite(value) for value in values+reset_indicator):
        raise ValueError("Cumulative diagnostic must be finite")
    if any(value < 0.0 for value in reset_indicator):
        raise ValueError("Cumulative reset indicator must be nonnegative")
    result = [0.0]
    total = 0.0
    for index in range(1, len(values)):
        reset = reset_indicator[index] < reset_indicator[index-1]
        # At a reset, the new process accumulated values[index] from zero between the
        # checkpoint and its first history sample. Signed moments use the same reset
        # indicator while retaining their sign.
        total += values[index] if reset else values[index]-values[index-1]
        result.append(total)
    return result


def interpolate_samples(times: list[float], values: list[float], target: float) -> float:
    if target <= times[0]:
        return values[0]
    for index in range(1, len(times)):
        if target <= times[index]:
            span = times[index]-times[index-1]
            fraction = 0.0 if span == 0.0 else (target-times[index-1])/span
            return values[index-1]+fraction*(values[index]-values[index-1])
    return values[-1]


def reset_aware_cumulative_delta(values: list[float]) -> float:
    return reset_aware_series(values, values)[-1]


def reset_aware_value_at(
    history: dict[str, list[float]], field: str, target: float
) -> float:
    series = reset_aware_series(history[field], history["laser_Edep"])
    return interpolate_samples(history["time"], series, target)


def relative_change_difference(
    reference: dict[str, list[float]], comparison: dict[str, list[float]], field: str,
    target: float,
) -> float:
    if field == "laser_Edep":
        ref = reset_aware_value_at(reference, field, target)
        cmp = reset_aware_value_at(comparison, field, target)
    else:
        ref = interpolate(reference, field, target)-reference[field][0]
        cmp = interpolate(comparison, field, target)-comparison[field][0]
    floor = 1.0e-10*max(abs(reference[field][0]), abs(comparison[field][0]), 1.0e-30)
    return abs(ref-cmp)/max(abs(ref), abs(cmp), floor)


def history_sensitivity(
    reference: dict[str, list[float]], comparison: dict[str, list[float]]
) -> dict[str, float]:
    target = min(reference["time"][-1], comparison["time"][-1])
    fields = ("laser_Edep", "eion_E", "eele_E", "erad_E", "chain_E")
    metrics = {field: relative_change_difference(reference, comparison, field, target)
               for field in fields}
    metrics["common_time"] = target
    ref_laser = reset_aware_value_at(reference, "laser_Edep", target)
    cmp_laser = reset_aware_value_at(comparison, "laser_Edep", target)
    deposited_scale = max(abs(ref_laser), abs(cmp_laser), 1.0e-30)
    ref_radiation = interpolate(reference, "erad_E", target)-reference["erad_E"][0]
    cmp_radiation = interpolate(comparison, "erad_E", target)-comparison["erad_E"][0]
    ref_ion = interpolate(reference, "eion_E", target)-reference["eion_E"][0]
    cmp_ion = interpolate(comparison, "eion_E", target)-comparison["eion_E"][0]
    metrics["eion_absolute_difference_over_deposited"] = (
        abs(ref_ion-cmp_ion)/deposited_scale
    )
    metrics["erad_absolute_difference_over_deposited"] = (
        abs(ref_radiation-cmp_radiation)/deposited_scale
    )
    if ref_laser > 0.0 and cmp_laser > 0.0:
        ref_x = reset_aware_value_at(reference, "laser_x", target)/ref_laser
        cmp_x = reset_aware_value_at(comparison, "laser_x", target)/cmp_laser
        metrics["laser_centroid"] = abs(ref_x-cmp_x)/max(abs(ref_x), abs(cmp_x), 1.0e-12)
    return metrics


def sensitivity_is_accepted(
    metrics: dict[str, float], settings: dict[str, Any]
) -> bool:
    relative_values = [metrics.get(field, math.inf)
                       for field in SENSITIVITY_RELATIVE_FIELDS]
    radiation_impact = metrics.get(
        "erad_absolute_difference_over_deposited", math.inf
    )
    return (
        all(math.isfinite(value) for value in relative_values)
        and max(relative_values) <= settings["sensitivity_relative_tolerance"]
        and math.isfinite(radiation_impact)
        and radiation_impact <=
            settings["radiation_deposited_relative_tolerance"]
    )


def read_deck_value(path: Path, key: str) -> float:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*([^#\s]+)",
                      path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    return float(match.group(1))


def command_override(status: dict[str, Any], phase: int, key: str) -> str | None:
    command = status.get(f"phase{phase}_mpi_command", [])
    prefix = key+"="
    for item in command:
        if isinstance(item, str) and item.startswith(prefix):
            return item[len(prefix):]
    return None


def check_3t_binary(path: Path) -> dict[str, float | int]:
    sys.path.insert(0, str(REPO / "vis" / "python"))
    import bin_convert  # type: ignore
    import numpy as np

    data = bin_convert.read_binary(str(path))
    names = set(data["var_names"])
    group_names = {f"erad{group:02d}" for group in range(20)}
    physical_names = {"eion", "eele", "tion", "tele", "erad", "trad"} | group_names
    expected = physical_names | {"eos_flags"}
    missing = sorted(expected-names)
    if missing:
        raise ValueError(f"Missing 3T fields in {path}: {missing}")
    minimum = math.inf
    maximum = 0.0
    cell_count = 0
    eos_trace_cell_count = 0
    eos_energy_floor_cell_count = 0
    eos_disallowed_cell_count = 0
    eos_maximum_flag = 0
    for name in sorted(expected):
        arrays = data["mb_data"][name]
        if name == "eion":
            cell_count = sum(int(array.size) for array in arrays)
        for array in arrays:
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Nonfinite {name} in {path}")
            if name in physical_names:
                minimum = min(minimum, float(np.min(array)))
                maximum = max(maximum, float(np.max(np.abs(array))))
            if name == "eos_flags":
                rounded = np.rint(array)
                if not np.all(array == rounded) or np.any(rounded < 0.0):
                    raise ValueError(f"Nonintegral or negative eos_flags in {path}")
                flags = rounded.astype(np.int64)
                if np.any((flags & ~0x3f) != 0):
                    raise ValueError(f"Unknown eos_flags bits in {path}")
                eos_trace_cell_count += int(np.count_nonzero(flags & 0x01))
                eos_energy_floor_cell_count += int(np.count_nonzero(flags & 0x10))
                eos_disallowed_cell_count += int(np.count_nonzero(flags & 0x2e))
                eos_maximum_flag = max(eos_maximum_flag, int(np.max(flags)))
    tolerance = 256.0*float(np.finfo(float).eps)*max(maximum, 1.0)
    if minimum < -tolerance:
        raise ValueError(f"Negative 3T field minimum {minimum} in {path}")
    return {
        "cycle": int(data["cycle"]),
        "time": float(data["time"]),
        "grid_nx1": int(data["Nx1"]),
        "grid_nx2": int(data["Nx2"]),
        "grid_nx3": int(data["Nx3"]),
        "meshblock_count": int(data["n_mbs"]),
        "cell_count": cell_count,
        "radiation_group_count": len(group_names & names),
        "eos_trace_cell_count": eos_trace_cell_count,
        "eos_energy_floor_cell_count": eos_energy_floor_cell_count,
        "eos_disallowed_cell_count": eos_disallowed_cell_count,
        "eos_maximum_flag": eos_maximum_flag,
        "minimum": minimum,
        "negative_tolerance": tolerance,
    }


def evidence_check(passed: bool, source_ids: list[str], **metrics: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "source_ids": source_ids, "metrics": metrics}


def evaluate_checks(sources: dict[str, Path], settings: dict[str, Any]) -> dict[str, Any]:
    smoke_status = read_json(sources["smoke_status"])
    resolution_status = read_json(sources["resolution_status"])
    rsla_status = read_json(sources["rsla_status"])
    physical_status = read_json(sources["physical_status"])
    calibration_status = read_json(sources["calibration_status"])
    smoke1 = parse_cycle_log(sources["smoke_phase1_log"])
    smoke2 = parse_cycle_log(sources["smoke_phase2_log"])
    physical1 = parse_cycle_log(sources["physical_phase1_log"])
    history = read_history(sources["smoke_history"])
    resolution_history = read_history(sources["resolution_history"])
    rsla_history = read_history(sources["rsla_history"])
    physical_history = read_history(sources["physical_history"])

    try:
        field_metrics = check_3t_binary(sources["smoke_3t_volume"])
        binary_3t_pass = True
    except Exception as exc:
        field_metrics = {"error": str(exc)}
        binary_3t_pass = False

    expected_artifacts = settings["artifacts"]
    statuses = (
        smoke_status,
        resolution_status,
        rsla_status,
        physical_status,
        calibration_status,
    )
    same_artifacts = all(status.get("case_artifacts") == expected_artifacts
                         for status in statuses)

    production_groups = int(read_deck_value(sources["production_input"], "n_groups"))
    calibration_groups = int(read_deck_value(
        sources["calibration_input"], "n_groups"))
    production_c_light = read_deck_value(sources["production_input"], "c_light")
    calibration_c_light = read_deck_value(sources["calibration_input"], "c_light")
    scale = int(smoke_status.get("compact_scale", 1))
    expected_shape = (100*scale, 64*scale, 64*scale)
    expected_cells = math.prod(expected_shape)
    compact_pass = (
        same_artifacts and smoke_status.get("mode") == "smoke"
        and scale == 1
        and smoke_status.get("radiation_c_light_override") is None
        and smoke_status.get("phase1_exit_code") == 0
        and smoke_status.get("phase2_exit_code") == 0
        and max(int(row["cycle"]) for row in smoke1) >= 50
        and command_override(smoke_status, 1, "time/nlim") == "50"
        and production_groups == 20 and calibration_groups == 20
        and production_c_light == PRODUCTION_C_LIGHT
        and calibration_c_light == PRODUCTION_C_LIGHT
        and binary_3t_pass and int(field_metrics.get("cycle", -1)) >= 50
        and int(field_metrics.get("radiation_group_count", -1)) == 20
        and tuple(int(field_metrics.get(f"grid_nx{axis}", -1))
                  for axis in (1, 2, 3)) == expected_shape
        and int(field_metrics.get("cell_count", -1)) == expected_cells
    )
    checks: dict[str, Any] = {}
    checks["compact_20group_50step"] = evidence_check(
        compact_pass,
        ["production_input", "calibration_input", "smoke_status",
         "smoke_phase1_log", "smoke_phase2_log", "smoke_3t_volume"],
        phase1_last_cycle=max(int(row["cycle"]) for row in smoke1),
        phase2_last_cycle=max(int(row["cycle"]) for row in smoke2),
        production_group_count=production_groups,
        calibration_group_count=calibration_groups,
        production_c_light=production_c_light,
        calibration_c_light=calibration_c_light,
        expected_volume_shape=list(expected_shape),
        expected_volume_cells=expected_cells,
        volume_shape=[field_metrics.get(f"grid_nx{axis}") for axis in (1, 2, 3)],
        volume_cycle=field_metrics.get("cycle"),
        volume_cell_count=field_metrics.get("cell_count"),
        volume_radiation_group_count=field_metrics.get("radiation_group_count"),
    )

    output_pass = all(sources[name].is_file() for name in (
        "smoke_history", "smoke_fluid_volume", "smoke_3t_volume",
        "smoke_laser_volume", "smoke_restart"))
    checks["compact_output_and_restart"] = evidence_check(
        compact_pass and output_pass,
        ["smoke_status", "smoke_history", "smoke_fluid_volume",
         "smoke_3t_volume", "smoke_laser_volume", "smoke_restart"],
        restart_bytes=sources["smoke_restart"].stat().st_size,
    )

    binary_floor_fraction = (
        float(field_metrics.get("eos_energy_floor_cell_count", math.inf))
        / max(int(field_metrics.get("cell_count", 0)), 1)
    )
    history_floor_fraction = max(history["eos_floor"])/expected_cells
    finite_pass = (
        binary_3t_pass
        and int(field_metrics.get("eos_disallowed_cell_count", -1)) == 0
        and binary_floor_fraction <= settings["maximum_eos_energy_floor_fraction"]
        and history_floor_fraction <= settings["maximum_eos_energy_floor_fraction"]
        and max(history["eos_bad"]) == 0.0
    )
    positive_history_fields = ("eion_E", "eele_E", "erad_E", "erad_soft",
                               "erad_mid", "erad_hard")
    finite_pass = finite_pass and all(min(history[field]) >= 0.0
                                      for field in positive_history_fields)
    checks["finite_nonnegative_3t"] = evidence_check(
        finite_pass, ["smoke_3t_volume", "smoke_history"],
        **field_metrics, binary_eos_energy_floor_fraction=binary_floor_fraction,
        history_eos_energy_floor_fraction=history_floor_fraction,
        maximum_eos_energy_floor_fraction=
            settings["maximum_eos_energy_floor_fraction"])

    c_light = float(production_c_light)
    dx_min = min(3.5/(100*scale), 2.0/(64*scale))
    causal_dt = dx_min/c_light
    timesteps = [float(row["dt"]) for row in smoke1+smoke2 if float(row["dt"]) > 0.0]
    min_ratio = min(timesteps)/causal_dt
    median_ratio = statistics.median(timesteps)/causal_dt
    timestep_pass = min_ratio >= settings["minimum_causal_dt_fraction"] and \
        median_ratio >= 10.0*settings["minimum_causal_dt_fraction"]
    checks["causal_timestep_no_collapse"] = evidence_check(
        timestep_pass, ["smoke_phase1_log", "smoke_phase2_log"],
        causal_dt=causal_dt, minimum_dt=min(timesteps), median_dt=statistics.median(timesteps),
        minimum_causal_ratio=min_ratio, median_causal_ratio=median_ratio)

    times = history["time"]
    escaped = sum(0.5*(history["rad_Pesc"][index-1]+history["rad_Pesc"][index])
                  *(times[index]-times[index-1]) for index in range(1, len(times)))
    deposited = reset_aware_cumulative_delta(history["laser_Edep"])
    chain_delta = history["chain_E"][-1]-history["chain_E"][0]
    residual = chain_delta+escaped-deposited
    energy_scale = max(abs(deposited), abs(chain_delta)+abs(escaped), 1.0e-30)
    energy_relative = abs(residual)/energy_scale
    power = read_deck_value(sources["production_input"], "beam0_power")
    start = read_deck_value(sources["production_input"], "beam0_start_time")
    end = read_deck_value(sources["production_input"], "beam0_end_time")
    incident_joules = power*(end-start)*1.0e-9/1.0e7
    energy_pass = energy_relative <= settings["energy_relative_tolerance"] and \
        abs(incident_joules-10000.0) <= 1.0e-9
    checks["laser_and_boundary_energy_closure"] = evidence_check(
        energy_pass, ["production_input", "smoke_history"],
        deposited=deposited, chain_delta=chain_delta, integrated_radiation_escape=escaped,
        residual=residual, relative_residual=energy_relative,
        configured_incident_joules=incident_joules)

    ch = history["CH_mass"]
    ch_relative = (max(ch)-min(ch))/max(abs(ch[0]), 1.0e-30)
    checks["ch_mass_conservation"] = evidence_check(
        ch_relative <= settings["mass_relative_tolerance"], ["smoke_history"],
        initial=ch[0], minimum=min(ch), maximum=max(ch), relative_range=ch_relative)

    restart_dt_ratio = float(smoke2[0]["dt"])/float(smoke1[-1]["dt"])
    restart_pass = (
        int(smoke2[-1]["cycle"]) > int(smoke1[-1]["cycle"])
        and 0.25 <= restart_dt_ratio <= 4.0
        and history["time"][-1] >= float(smoke2[-1]["time"])
    )
    checks["restart_continuity"] = evidence_check(
        restart_pass,
        ["smoke_status", "smoke_phase1_log", "smoke_phase2_log", "smoke_history",
         "smoke_restart", "smoke_material_manifest"],
        phase1_last_cycle=int(smoke1[-1]["cycle"]),
        phase2_first_cycle=int(smoke2[0]["cycle"]),
        phase2_last_cycle=int(smoke2[-1]["cycle"]), dt_ratio=restart_dt_ratio)

    resolution_metrics = history_sensitivity(history, resolution_history)
    resolution_floor_fraction = max(resolution_history["eos_floor"])/(
        200*128*128)
    resolution_pass = (
        same_artifacts and resolution_status.get("compact_scale") == 2
        and resolution_status.get("phase1_exit_code") == 0
        and resolution_status.get("phase2_exit_code") == 0
        and max(resolution_history["eos_bad"]) == 0.0
        and resolution_floor_fraction <=
            settings["maximum_eos_energy_floor_fraction"]
        and max(resolution_metrics.get(field, math.inf)
                for field in RESOLUTION_RELATIVE_FIELDS) <=
            settings["resolution_relative_tolerance"]
        and resolution_metrics.get(
            "eion_absolute_difference_over_deposited", math.inf
        ) <= settings["sensitivity_relative_tolerance"]
    )
    checks["resolution_or_opacity_sensitivity"] = evidence_check(
        resolution_pass,
        ["resolution_status", "resolution_phase1_log", "resolution_phase2_log",
         "resolution_history", "smoke_history"], **resolution_metrics,
        maximum_relative_difference=max(
            resolution_metrics.get(field, math.inf)
            for field in RESOLUTION_RELATIVE_FIELDS
        ),
        maximum_relative_difference_tolerance=
            settings["resolution_relative_tolerance"],
        eion_deposited_relative_tolerance=
            settings["sensitivity_relative_tolerance"],
        eos_energy_floor_fraction=resolution_floor_fraction)

    rsla_metrics = history_sensitivity(history, rsla_history)
    rsla_floor_fraction = max(rsla_history["eos_floor"])/(100*64*64)
    rsla_coverage = rsla_history["time"][-1] >= history["time"][-1]
    rsla_pass = (
        compact_pass and same_artifacts and rsla_status.get("mode") == "smoke"
        and rsla_status.get("compact_scale") == 1
        and float(rsla_status.get("radiation_c_light_override", 0.0)) ==
            RSLA_COMPARISON_C_LIGHT
        and rsla_status.get("phase1_exit_code") == 0
        and rsla_status.get("phase2_exit_code") == 0
        and max(rsla_history["eos_bad"]) == 0.0
        and rsla_floor_fraction <= settings["maximum_eos_energy_floor_fraction"]
        and rsla_coverage
        and sensitivity_is_accepted(rsla_metrics, settings)
    )
    checks["reduced_light_speed_sensitivity"] = evidence_check(
        rsla_pass,
        ["production_input", "smoke_status", "smoke_history", "rsla_status",
         "rsla_phase1_log", "rsla_phase2_log", "rsla_history"],
        **rsla_metrics,
        history_covers_production_baseline=rsla_coverage,
        maximum_matter_relative_difference=max(
            rsla_metrics.get(field, math.inf)
            for field in SENSITIVITY_RELATIVE_FIELDS
        ),
        sensitivity_relative_tolerance=settings["sensitivity_relative_tolerance"],
        radiation_deposited_relative_tolerance=
            settings["radiation_deposited_relative_tolerance"],
        eos_energy_floor_fraction=rsla_floor_fraction)

    physical_metrics = history_sensitivity(history, physical_history)
    physical_floor_fraction = max(physical_history["eos_floor"])/(100*64*64)
    physical_nlim_text = command_override(physical_status, 1, "time/nlim")
    try:
        physical_nlim = int(physical_nlim_text) if physical_nlim_text is not None else -1
    except ValueError:
        physical_nlim = -1
    physical_last_cycle = max(int(row["cycle"]) for row in physical1)
    physical_last_time = max(float(row["time"]) for row in physical1)
    physical_coverage = (
        physical_history["time"][-1] >= history["time"][-1]
        and physical_last_time >= history["time"][-1]
    )
    physical_pass = (
        compact_pass and same_artifacts and physical_status.get("mode") == "smoke"
        and physical_status.get("compact_scale") == 1
        and float(physical_status.get("radiation_c_light_override", 0.0)) ==
            PHYSICAL_C_LIGHT
        and physical_status.get("phase1_exit_code") == 0
        and physical_nlim >= 650
        and physical_last_cycle >= 650
        and max(physical_history["eos_bad"]) == 0.0
        and physical_floor_fraction <= settings["maximum_eos_energy_floor_fraction"]
        and physical_coverage
        and sensitivity_is_accepted(physical_metrics, settings)
    )
    checks["physical_light_speed_sensitivity"] = evidence_check(
        physical_pass,
        ["production_input", "smoke_status", "smoke_history", "physical_status",
         "physical_phase1_log", "physical_history"],
        **physical_metrics,
        requested_cycle_limit=physical_nlim,
        final_cycle=physical_last_cycle,
        final_time=physical_last_time,
        history_covers_production_baseline=physical_coverage,
        maximum_matter_relative_difference=max(
            physical_metrics.get(field, math.inf)
            for field in SENSITIVITY_RELATIVE_FIELDS
        ),
        sensitivity_relative_tolerance=settings["sensitivity_relative_tolerance"],
        radiation_deposited_relative_tolerance=
            settings["radiation_deposited_relative_tolerance"],
        eos_energy_floor_fraction=physical_floor_fraction)

    memory = calibration_status.get("phase1_memory", {})
    devices = memory.get("devices", {}) if isinstance(memory, dict) else {}
    fractions = {
        name: record.get("peak_fraction") for name, record in devices.items()
        if isinstance(record, dict)
    }
    memory_pass = (
        same_artifacts and calibration_status.get("mode") == "calibrate"
        and calibration_status.get("phase1_exit_code") == 0
        and command_override(calibration_status, 1, "time/nlim") == "2"
        and len(devices) == 8 and not memory.get("errors")
        and all(isinstance(record, dict)
                and "V100" in str(record.get("name", ""))
                and record.get("within_60_80_percent") is True
                for record in devices.values())
        and not any(calibration_status.get("baseline_processes", {}).values())
    )
    checks["gpu_memory_60_80_all"] = evidence_check(
        memory_pass, ["calibration_status", "calibration_phase1_log"],
        peak_fractions=fractions, device_count=len(devices))
    return checks


def resolve_gate_sources(gate: dict[str, Any], gate_path: Path) -> dict[str, Path]:
    records = gate.get("sources")
    if not isinstance(records, dict):
        raise ValueError("Gate sources must be an object")
    result = {}
    for name, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Malformed source record: {name}")
        path = Path(record["path"]).expanduser()
        result[name] = path if path.is_absolute() else gate_path.parent/path
    return result


def recompute_checks_from_gate(gate: dict[str, Any], gate_path: Path) -> dict[str, Any]:
    return evaluate_checks(resolve_gate_sources(gate, gate_path), gate["settings"])


def source_record(path: Path, output: Path) -> dict[str, str]:
    try:
        display = str(path.resolve().relative_to(output.parent.resolve()))
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": sha256_path(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=CASE_DIR/"runs"/"smoke")
    parser.add_argument("--resolution-dir", type=Path,
                        default=CASE_DIR/"runs"/"resolution2")
    parser.add_argument("--rsla-dir", type=Path, default=CASE_DIR/"runs"/"rsla10")
    parser.add_argument("--physical-c-dir", type=Path,
                        default=CASE_DIR/"runs"/"cphys650")
    parser.add_argument("--calibration-dir", type=Path,
                        default=CASE_DIR/"runs"/"calibrate")
    parser.add_argument("--output", type=Path, default=CASE_DIR/"production_gate.json")
    parser.add_argument("--energy-rtol", type=float, default=5.0e-4)
    parser.add_argument("--mass-rtol", type=float, default=1.0e-8)
    parser.add_argument("--resolution-rtol", type=float, default=0.35)
    parser.add_argument("--sensitivity-rtol", type=float, default=0.05)
    parser.add_argument("--radiation-deposited-rtol", type=float, default=0.01)
    parser.add_argument("--minimum-causal-dt-fraction", type=float, default=1.0e-4)
    parser.add_argument("--maximum-eos-energy-floor-fraction", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and print the gate without writing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    sys.path.insert(0, str(CASE_DIR))
    import run_case

    try:
        sources = discover_sources(
            args.smoke_dir.expanduser().resolve(),
            args.resolution_dir.expanduser().resolve(),
            args.rsla_dir.expanduser().resolve(),
            args.physical_c_dir.expanduser().resolve(),
            args.calibration_dir.expanduser().resolve())
        artifacts = run_case.gate_artifact_hashes()
        settings = {
            "artifacts": artifacts,
            "energy_relative_tolerance": args.energy_rtol,
            "mass_relative_tolerance": args.mass_rtol,
            "resolution_relative_tolerance": args.resolution_rtol,
            "sensitivity_relative_tolerance": args.sensitivity_rtol,
            "radiation_deposited_relative_tolerance":
                args.radiation_deposited_rtol,
            "minimum_causal_dt_fraction": args.minimum_causal_dt_fraction,
            "maximum_eos_energy_floor_fraction":
                args.maximum_eos_energy_floor_fraction,
        }
        checks = evaluate_checks(sources, settings)
    except Exception as exc:
        print(f"Cannot derive production gate: {exc}", file=sys.stderr)
        return 2
    gate = {
        "schema": SCHEMA,
        "generator": {"path": Path(__file__).name, "sha256": sha256_path(Path(__file__))},
        "artifacts": artifacts,
        "settings": settings,
        "sources": {name: source_record(path, output) for name, path in sources.items()},
        "checks": checks,
    }
    payload = json.dumps(gate, indent=2, sort_keys=True)+"\n"
    if args.dry_run:
        print(payload, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"Wrote {output}")
    failed = [name for name, record in checks.items() if not record["passed"]]
    if failed:
        print("FAILED production checks: " + ", ".join(failed), file=sys.stderr)
        return 2
    print("All pre-production checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
