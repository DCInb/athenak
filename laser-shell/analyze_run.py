#!/usr/bin/env python3
"""Verify the CH shell geometry, incident pulse, outputs, and 10 ns production run."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
INPUT_PATH = CASE_DIR / "laser_shell.athinput"
STATUS_PATH = CASE_DIR / "run_status.json"
DEFAULT_RUN_DIR = Path("/home/mengqi/data/athenak-2t/laser-shell/run")
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ATOMIC_MASS_UNIT_CGS = 1.660538921e-24
BOLTZMANN_CGS = 1.3806488e-16
EXPECTED_BIN_COUNT = 101
EXPECTED_RESTART_COUNT = 11
BIN_TIME_TOLERANCE_NS = 1.0e-9
INPUT_TEMPERATURE_TOLERANCE_K = 1.0e-10
# Athena's binary fields default to float32 even for a double-precision executable.
FIELD_TEMPERATURE_TOLERANCE_K = 1.0e-3
RESTART_REAL_BYTES = 8
REGION_SIZE_REAL_COUNT = 9
REGION_INDCS_INT_COUNT = 19


def input_blocks() -> dict[str, dict[str, str]]:
    """Parse the simple Athena input syntax into block/key/value strings."""
    blocks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in INPUT_PATH.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, {})
            continue
        if current is not None and "=" in line:
            key, value = line.split("=", 1)
            blocks[current][key.strip()] = value.strip()
    return blocks


def input_parameter(block: str, name: str) -> str:
    blocks = input_blocks()
    try:
        return blocks[block][name].split()[0]
    except KeyError as exc:
        raise RuntimeError(f"Missing input parameter <{block}>/{name}") from exc


def block_value(block: str, name: str) -> float:
    value = input_parameter(block, name)
    if re.fullmatch(FLOAT, value) is None:
        raise RuntimeError(f"Expected numeric <{block}>/{name}, found {value!r}")
    return float(value)


def block_string(block: str, name: str) -> str:
    return input_parameter(block, name)


def input_value(name: str) -> float:
    """Read a numeric parameter whose name is unique across all input blocks."""
    matches = [
        value
        for parameters in input_blocks().values()
        for key, value in parameters.items()
        if key == name
    ]
    if len(matches) != 1 or re.fullmatch(FLOAT, matches[0].split()[0]) is None:
        raise RuntimeError(f"Expected one numeric {name}, found {len(matches)}")
    return float(matches[0].split()[0])


def input_string(name: str) -> str:
    """Read a string parameter whose name is unique across all input blocks."""
    matches = [
        value.split()[0]
        for parameters in input_blocks().values()
        for key, value in parameters.items()
        if key == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def resolve_status_path(status: dict[str, Any], key: str) -> Path | None:
    value = status.get(key)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else CASE_DIR / path


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of the candidate files exist: " + ", ".join(map(str, paths)))


def read_history(path: Path) -> dict[str, np.ndarray]:
    header = [line for line in path.read_text().splitlines() if line.startswith("#")]
    labels = re.findall(r"\[\d+\]=([^\s]+)", header[-1])
    values = np.loadtxt(path)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    return {label: values[:, index] for index, label in enumerate(labels)}


def parse_laser_log(path: Path) -> list[dict[str, float]]:
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("laser:"):
            continue
        records.append(
            {
                key: float(value)
                for key, value in re.findall(rf"([a-z_]+)=({FLOAT})", line)
            }
        )
    if not records:
        raise RuntimeError(f"No laser diagnostics in {path}")
    return records


def final_time(path: Path) -> float:
    matches = re.findall(rf"(?m)^time=({FLOAT})\s+cycle=", path.read_text())
    if not matches:
        raise RuntimeError(f"No final time record in {path}")
    return float(matches[-1])


def binary_metadata(path: Path) -> dict[str, Any]:
    """Read only the textual prefix of an Athena binary dump."""
    with path.open("rb") as stream:
        code_header = stream.readline().split()
        if not code_header or code_header[0] != b"Athena":
            raise TypeError(f"{path} is not an Athena binary dump")
        version = code_header[-1].split(b"=")[-1]
        if version != b"1.1":
            raise TypeError(f"Unsupported binary version {version!r} in {path}")
        parameter_count = int(stream.readline().split(b"=")[-1])
        parameters: dict[str, str] = {}
        for _ in range(parameter_count - 1):
            key, value = [
                item.strip()
                for item in stream.readline().decode("utf-8").split("=", 1)
            ]
            parameters[key] = value
        nvars = int(stream.readline().split(b"=")[-1])
        variable_line = stream.readline().split()
        variable_names = [item.decode("utf-8") for item in variable_line[1:]]
        if len(variable_names) != nvars:
            raise RuntimeError(f"Variable-count mismatch in {path}")
        header_size = int(stream.readline().split(b"=")[-1])
        header_lines = [
            line.decode("utf-8").split("#", 1)[0].strip()
            for line in stream.read(header_size).split(b"\n")
        ]
    runtime_blocks: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in header_lines:
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            current = match.group(1)
            runtime_blocks.setdefault(current, {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            runtime_blocks[current][key.strip()] = value.strip()
    return {
        "time": float(parameters["time"]),
        "cycle": int(parameters["cycle"]),
        "variables": variable_names,
        "input_blocks": runtime_blocks,
    }


def restart_metadata(path: Path) -> dict[str, Any]:
    """Read the physical time, dt, and cycle from an AthenaK restart header."""
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Missing <par_end> in restart file {path}")
            if line.strip() == b"<par_end>":
                break
        fixed_prefix_size = (
            2*struct.calcsize("<i")
            + REGION_SIZE_REAL_COUNT*RESTART_REAL_BYTES
            + 2*REGION_INDCS_INT_COUNT*struct.calcsize("<i")
        )
        header = stream.read(
            fixed_prefix_size + 2*RESTART_REAL_BYTES + struct.calcsize("<i")
        )
    required = fixed_prefix_size + 2*RESTART_REAL_BYTES + struct.calcsize("<i")
    if len(header) != required:
        raise RuntimeError(f"Truncated mesh header in restart file {path}")
    nmb_total, root_level = struct.unpack_from("<ii", header, 0)
    time, dt = struct.unpack_from("<dd", header, fixed_prefix_size)
    cycle = struct.unpack_from("<i", header, fixed_prefix_size+2*RESTART_REAL_BYTES)[0]
    if (
        nmb_total <= 0 or root_level < 0 or cycle < 0
        or not math.isfinite(time) or time < 0.0
        or not math.isfinite(dt) or dt <= 0.0
    ):
        raise RuntimeError(f"Invalid mesh metadata in restart file {path}")
    return {
        "time": time,
        "dt": dt,
        "cycle": cycle,
        "nmb_total": nmb_total,
        "root_level": root_level,
    }


def dump_index(path: Path) -> int:
    match = re.search(r"\.(\d{5})\.(?:bin|rst)$", path.name)
    if match is None:
        raise RuntimeError(f"Cannot parse dump index from {path}")
    return int(match.group(1))


def assemble(
    path: Path, selected_names: tuple[str, ...] | None = None
) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
    raw = read_binary(str(path))
    nx = (int(raw["Nx1"]), int(raw["Nx2"]), int(raw["Nx3"]))
    mb_nx = (
        int(raw["nx1_out_mb"]),
        int(raw["nx2_out_mb"]),
        int(raw["nx3_out_mb"]),
    )
    names = tuple(raw["var_names"]) if selected_names is None else selected_names
    missing = set(names).difference(raw["var_names"])
    if missing:
        raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")
    fields = {
        name: np.zeros((nx[2], nx[1], nx[0]), dtype=np.float64)
        for name in names
    }
    for block, logical in enumerate(raw["mb_logical"]):
        i0 = int(logical[0])*mb_nx[0]
        j0 = int(logical[1])*mb_nx[1]
        k0 = int(logical[2])*mb_nx[2]
        for name in names:
            values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            nz, ny, nx_local = values.shape
            fields[name][k0:k0+nz, j0:j0+ny, i0:i0+nx_local] = values
    extent = (
        float(raw["x1min"]), float(raw["x1max"]),
        float(raw["x2min"]), float(raw["x2max"]),
        float(raw["x3min"]), float(raw["x3max"]),
    )
    return fields, extent


def coordinates(shape: tuple[int, ...], extent: tuple[float, ...]):
    nz, ny, nx = shape
    x1 = extent[0] + (np.arange(nx)+0.5)*(extent[1]-extent[0])/nx
    x2 = extent[2] + (np.arange(ny)+0.5)*(extent[3]-extent[2])/ny
    x3 = extent[4] + (np.arange(nz)+0.5)*(extent[5]-extent[4])/nz
    return x1, x2, x3


def temperature_scale_kelvin() -> float:
    velocity_cgs = block_value("units", "length_cgs")/block_value("units", "time_cgs")
    return (
        velocity_cgs**2*block_value("units", "mu")*ATOMIC_MASS_UNIT_CGS
        / BOLTZMANN_CGS
    )


def geometry_temperature_metrics(run_dir: Path) -> dict[str, float]:
    fluid_path = sorted((run_dir / "bin").glob("laser_shell.fluid.*.bin"))[0]
    two_t_path = sorted(
        (run_dir / "bin").glob("laser_shell.two_temperature.*.bin")
    )[0]
    fields, extent = assemble(fluid_path, ("dens",))
    two_t, two_t_extent = assemble(two_t_path, ("tion", "tele"))
    if extent != two_t_extent or fields["dens"].shape != two_t["tion"].shape:
        raise RuntimeError("Initial fluid and two-temperature dumps are not co-spatial")

    density = fields["dens"]
    x1, x2, x3 = coordinates(density.shape, extent)
    x1_grid = np.broadcast_to(x1[np.newaxis, np.newaxis, :], density.shape)
    radius = np.sqrt(
        x1[np.newaxis, np.newaxis, :]**2
        + x2[np.newaxis, :, np.newaxis]**2
        + x3[:, np.newaxis, np.newaxis]**2
    )
    axis_cosine = np.divide(
        -x1[np.newaxis, np.newaxis, :],
        radius,
        out=np.full_like(radius, -1.0),
        where=radius > 0.0,
    )
    ambient = block_value("problem", "ambient_density")
    solid = block_value("problem", "solid_density")
    fraction = np.clip((density-ambient)/(solid-ambient), 0.0, 1.0)
    shell_mask = fraction > 0.5
    ambient_mask = fraction < 1.0e-4
    if not np.any(shell_mask) or not np.any(ambient_mask):
        raise RuntimeError("Initial density dump lacks resolved shell or ambient cells")
    theta = np.degrees(np.arccos(np.clip(axis_cosine[shell_mask], -1.0, 1.0)))
    cell_volume = (
        (extent[1]-extent[0])/density.shape[2]
        * (extent[3]-extent[2])/density.shape[1]
        * (extent[5]-extent[4])/density.shape[0]
    )
    mass_cgs = block_value("units", "mass_cgs")
    length_cgs = block_value("units", "length_cgs")
    density_scale_cgs = mass_cgs/length_cgs**3
    excess_mass = float(np.sum(density-ambient)*cell_volume*mass_cgs)
    half_angle = block_value("problem", "opening_half_angle_deg")
    inner_radius = block_value("problem", "inner_radius")
    beam_radius = block_value("laser", "beam0_radius")
    projected_inner_radius = inner_radius*math.sin(math.radians(half_angle))

    temperature_scale = temperature_scale_kelvin()
    tion_kelvin = two_t["tion"]*temperature_scale
    tele_kelvin = two_t["tele"]*temperature_scale
    return {
        "measured_inner_radius_mm": float(np.min(radius[shell_mask])),
        "measured_outer_radius_mm": float(np.max(radius[shell_mask])),
        "measured_shell_thickness_mm": float(
            np.max(radius[shell_mask])-np.min(radius[shell_mask])
        ),
        "measured_peak_density_g_cc": float(np.max(density)*density_scale_cgs),
        "measured_full_opening_angle_deg": 2.0*float(np.max(theta)),
        "cap_min_x1_mm": float(np.min(x1_grid[shell_mask])),
        "cap_max_x1_mm": float(np.max(x1_grid[shell_mask])),
        "initial_excess_ch_mass_mg": 1.0e3*excess_mass,
        "projected_inner_radius_mm": projected_inner_radius,
        "gaussian_radius_mm": beam_radius,
        "projected_area_coverage": min((beam_radius/projected_inner_radius)**2, 1.0),
        "temperature_scale_kelvin": temperature_scale,
        "input_temperature_kelvin": (
            block_value("problem", "temperature")*temperature_scale
        ),
        "shell_tion_min_kelvin": float(np.min(tion_kelvin[shell_mask])),
        "shell_tion_max_kelvin": float(np.max(tion_kelvin[shell_mask])),
        "shell_tele_min_kelvin": float(np.min(tele_kelvin[shell_mask])),
        "shell_tele_max_kelvin": float(np.max(tele_kelvin[shell_mask])),
        "ambient_tion_min_kelvin": float(np.min(tion_kelvin[ambient_mask])),
        "ambient_tion_max_kelvin": float(np.max(tion_kelvin[ambient_mask])),
        "ambient_tele_min_kelvin": float(np.min(tele_kelvin[ambient_mask])),
        "ambient_tele_max_kelvin": float(np.max(tele_kelvin[ambient_mask])),
    }


def binary_cadence_metrics(run_dir: Path, time_cgs: float) -> dict[str, Any]:
    patterns = {
        "fluid": "laser_shell.fluid.*.bin",
        "two_temperature": "laser_shell.two_temperature.*.bin",
        "laser": "laser_shell.laser.*.bin",
    }
    expected_indices = list(range(EXPECTED_BIN_COUNT))
    expected_times_ns = np.arange(EXPECTED_BIN_COUNT, dtype=float)*0.1
    streams: dict[str, Any] = {}
    time_arrays = []
    for name, pattern in patterns.items():
        paths = sorted((run_dir / "bin").glob(pattern), key=dump_index)
        indices = [dump_index(path) for path in paths]
        times_ns = np.asarray(
            [binary_metadata(path)["time"]*time_cgs/1.0e-9 for path in paths]
        )
        comparable = min(len(times_ns), len(expected_times_ns))
        schedule_error = (
            float(np.max(np.abs(times_ns[:comparable]-expected_times_ns[:comparable])))
            if comparable else math.inf
        )
        streams[name] = {
            "count": len(paths),
            "indices_complete": indices == expected_indices,
            "first_time_ns": float(times_ns[0]) if len(times_ns) else math.nan,
            "last_time_ns": float(times_ns[-1]) if len(times_ns) else math.nan,
            "max_schedule_error_ns": schedule_error,
            "max_interval_error_ns": (
                float(np.max(np.abs(np.diff(times_ns)-0.1)))
                if len(times_ns) > 1 else math.inf
            ),
        }
        time_arrays.append(times_ns)
    synchronized = all(len(times) == len(time_arrays[0]) for times in time_arrays)
    sync_error = math.inf
    if synchronized and len(time_arrays[0]):
        sync_error = max(
            float(np.max(np.abs(times-time_arrays[0]))) for times in time_arrays[1:]
        )
    return {
        "configured_bin_dt_ns": {
            f"output{index}": block_value(f"output{index}", "dt")*time_cgs/1.0e-9
            for index in (2, 3, 4)
        },
        "streams": streams,
        "synchronized": synchronized,
        "max_stream_sync_error_ns": sync_error,
    }


def restart_metrics(run_dir: Path, status: dict[str, Any], time_cgs: float):
    paths = sorted(run_dir.rglob("laser_shell.*.rst"), key=dump_index)
    indices = [dump_index(path) for path in paths]
    configured_dt_ns = block_value("output5", "dt")*time_cgs/1.0e-9
    metadata = [restart_metadata(path) for path in paths]
    actual_times_ns = np.asarray(
        [item["time"]*time_cgs/1.0e-9 for item in metadata], dtype=float
    )
    expected_times_ns = np.arange(EXPECTED_RESTART_COUNT, dtype=float)
    comparable = min(len(actual_times_ns), len(expected_times_ns))
    schedule_error = (
        float(np.max(np.abs(actual_times_ns[:comparable]-expected_times_ns[:comparable])))
        if comparable else math.inf
    )
    seam = resolve_status_path(status, "restart")
    return {
        "configured_dt_ns": configured_dt_ns,
        "count": len(paths),
        "indices_complete": indices == list(range(EXPECTED_RESTART_COUNT)),
        "actual_times_ns": actual_times_ns.tolist(),
        "cycles": [item["cycle"] for item in metadata],
        "max_schedule_error_ns": schedule_error,
        "max_interval_error_ns": (
            float(np.max(np.abs(np.diff(actual_times_ns)-1.0)))
            if len(actual_times_ns) > 1 else math.inf
        ),
        "phase1_seam_index": dump_index(seam) if seam is not None else None,
    }


def incidence_metrics(run_dir: Path) -> dict[str, float]:
    laser_paths = sorted(
        (run_dir / "bin").glob("laser_shell.laser.*.bin"), key=dump_index
    )
    for path in laser_paths[1:]:
        fields, extent = assemble(
            path, ("laser_path", "laser_dir1", "laser_energy")
        )
        path_length = fields["laser_path"]
        if np.any(path_length > 0.0):
            break
    else:
        raise RuntimeError("No active laser binary dump was found")

    x1, _, _ = coordinates(path_length.shape, extent)
    x1_grid = np.broadcast_to(x1[np.newaxis, np.newaxis, :], path_length.shape)
    source_path_mask = (x1_grid > 0.0) & (path_length > 0.0)
    source_path = float(np.sum(path_length[source_path_mask]))
    direction_x1 = (
        float(np.sum(fields["laser_dir1"][source_path_mask]))/source_path
        if source_path > 0.0 else math.nan
    )
    deposited = np.maximum(fields["laser_energy"], 0.0)
    total_deposited = float(np.sum(deposited))
    upstream_deposited = float(np.sum(deposited[x1_grid > 0.0]))
    centroid_x1 = (
        float(np.sum(x1_grid*deposited))/total_deposited
        if total_deposited > 0.0 else math.nan
    )
    metadata = binary_metadata(path)
    runtime_laser = metadata["input_blocks"]["laser"]
    return {
        "first_active_dump_index": dump_index(path),
        "first_active_dump_time_ns": metadata["time"]*block_value(
            "units", "time_cgs"
        )/1.0e-9,
        "source_side_path_weighted_direction_x1": direction_x1,
        "upstream_deposited_fraction": (
            upstream_deposited/total_deposited if total_deposited > 0.0 else math.inf
        ),
        "deposition_centroid_x1_mm": centroid_x1,
        "runtime_header_origin_x1": float(runtime_laser["beam0_origin_x1"]),
        "runtime_header_direction_x1": float(runtime_laser["beam0_direction_x1"]),
        "runtime_header_direction_x2": float(runtime_laser["beam0_direction_x2"]),
        "runtime_header_direction_x3": float(runtime_laser["beam0_direction_x3"]),
    }


def gpu_memory_metrics(status: dict[str, Any]) -> dict[str, Any]:
    memory = status.get("gpu_memory")
    if not isinstance(memory, dict):
        return {
            "recorded": False,
            "devices": {},
            "ranks": status.get("ranks"),
            "visible_devices": [],
            "baseline_compute_processes": [],
            "eight_unique_devices": False,
            "eight_ranks": False,
            "all_devices_are_v100": False,
            "baseline_was_idle": False,
        }
    device_ids = [str(device) for device in memory.get("device_ids", [])]
    models = memory.get("model_names", {})
    totals = memory.get("total_mib", {})
    baselines = memory.get("baseline_used_mib", {})
    peaks = memory.get("peak_used_mib", {})
    deltas = memory.get("peak_delta_mib", {})
    fractions = memory.get("peak_fraction", {})
    baseline_processes = memory.get("baseline_compute_processes", [])
    devices: dict[str, Any] = {}
    for device in device_ids:
        try:
            model = str(models[device])
            total = float(totals[device])
            baseline = float(baselines[device])
            peak = float(peaks[device])
            reported_delta = float(deltas[device])
            reported_fraction = float(fractions[device])
        except (KeyError, TypeError, ValueError):
            devices[device] = {"complete": False}
            continue
        computed_delta = max(peak-baseline, 0.0)
        computed_fraction = computed_delta/total if total > 0.0 else math.inf
        devices[device] = {
            "complete": True,
            "model": model,
            "is_v100": "V100" in model,
            "total_mib": total,
            "baseline_used_mib": baseline,
            "peak_used_mib": peak,
            "peak_delta_mib": reported_delta,
            "peak_fraction": reported_fraction,
            "computed_peak_fraction": computed_fraction,
            "delta_consistency_error_mib": abs(reported_delta-computed_delta),
            "fraction_consistency_error": abs(
                reported_fraction-computed_fraction
            ),
            "within_required_band": 0.60 <= reported_fraction <= 0.80,
        }
    visible_devices = [
        item.strip() for item in str(status.get("visible_devices", "")).split(",")
        if item.strip()
    ]
    ranks = status.get("ranks")
    return {
        "recorded": bool(device_ids),
        "devices": devices,
        "ranks": ranks,
        "visible_devices": visible_devices,
        "baseline_compute_processes": baseline_processes,
        "eight_unique_devices": (
            len(device_ids) == 8 and len(set(device_ids)) == 8
            and len(visible_devices) == 8 and len(set(visible_devices)) == 8
            and set(device_ids) == set(visible_devices)
        ),
        "eight_ranks": ranks == 8,
        "all_devices_are_v100": (
            len(devices) == 8
            and all(device.get("is_v100", False) for device in devices.values())
        ),
        "baseline_was_idle": baseline_processes == [],
    }


def main() -> int:
    status = json.loads(STATUS_PATH.read_text())
    run_value = status.get("run_dir")
    run_dir = Path(run_value).expanduser() if isinstance(run_value, str) else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = CASE_DIR / run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Production run directory does not exist: {run_dir}")

    phase1_log = first_existing([
        path for path in (
            resolve_status_path(status, "phase1_log"),
            CASE_DIR / "logs" / "phase1_laser_on.log",
            run_dir.parent / "logs" / "phase1_laser_on.log",
        ) if path is not None
    ])
    phase2_log = first_existing([
        path for path in (
            resolve_status_path(status, "phase2_log"),
            CASE_DIR / "logs" / "phase2_laser_off.log",
            run_dir.parent / "logs" / "phase2_laser_off.log",
        ) if path is not None
    ])
    phase1_records = parse_laser_log(phase1_log)
    phase2_records = parse_laser_log(phase2_log)
    active = [record for record in phase1_records if record["launched"] > 0.0]
    if not active:
        raise RuntimeError("Phase 1 has no illuminated laser stages")

    length_cgs = block_value("units", "length_cgs")
    mass_cgs = block_value("units", "mass_cgs")
    time_cgs = block_value("units", "time_cgs")
    velocity_cgs = length_cgs/time_cgs
    energy_scale_erg = mass_cgs*velocity_cgs**2
    power_scale_erg_s = energy_scale_erg/time_cgs
    beam_power_erg_s = block_value("laser", "beam0_power")
    pulse_duration_s = (
        block_value("laser", "beam0_end_time")
        - block_value("laser", "beam0_start_time")
    )*time_cgs
    incident_energy_erg = beam_power_erg_s*pulse_duration_s
    expected_code_power = beam_power_erg_s/power_scale_erg_s

    phase1_history = read_history(run_dir / "phase1_laser_shell.user.hst")
    full_history = read_history(run_dir / "laser_shell.user.hst")
    deposited_energy_erg = float(phase1_history["laser_E"][-1])*energy_scale_erg
    material_energy_change_erg = float(
        full_history["mat_E"][-1]-full_history["mat_E"][0]
    )*energy_scale_erg

    geometry = geometry_temperature_metrics(run_dir)
    cadence = binary_cadence_metrics(run_dir, time_cgs)
    restarts = restart_metrics(run_dir, status, time_cgs)
    incidence = incidence_metrics(run_dir)
    gpu_memory = gpu_memory_metrics(status)
    laser = {
        "profile": block_string("laser", "beam0_profile"),
        "wavelength_um": block_value("laser", "beam0_wavelength")*1.0e4,
        "power_tw": beam_power_erg_s/1.0e19,
        "duration_ns": pulse_duration_s/1.0e-9,
        "incident_energy_kj": incident_energy_erg/1.0e10,
        "deposited_energy_kj": deposited_energy_erg/1.0e10,
        "deposited_fraction": deposited_energy_erg/incident_energy_erg,
        "material_energy_change_kj": material_energy_change_erg/1.0e10,
        "active_stage_count": len(active),
        "max_active_power_relative_error": max(
            abs(record["launched"]-expected_code_power)/expected_code_power
            for record in active
        ),
        "max_transport_residual": max(
            record["residual"] for record in phase1_records+phase2_records
        ),
        "max_phase2_launched_code_power": max(
            record["launched"] for record in phase2_records
        ),
        "min_phase1_launched_code_power": min(
            record["launched"] for record in phase1_records
        ),
        "max_unresolved_power_fraction": max(
            record["remaining"]/record["launched"] for record in active
        ),
        "reflected_ray_events": int(sum(
            record["reflected"] for record in phase1_records
        )),
    }
    run = {
        "run_dir": str(run_dir),
        "state": status.get("state"),
        "phase1_exit_code": int(status["phase1_exit_code"]),
        "phase2_exit_code": int(status["phase2_exit_code"]),
        "phase1_final_time_ns": final_time(phase1_log),
        "phase2_final_time_ns": final_time(phase2_log),
        "phase1_elapsed_seconds": float(status["phase1_elapsed_seconds"]),
        "phase2_elapsed_seconds": float(status["phase2_elapsed_seconds"]),
    }
    metrics = {
        "run": run,
        "laser": laser,
        "geometry": geometry,
        "incidence": incidence,
        "binary_outputs": cadence,
        "restarts": restarts,
        "gpu_memory": gpu_memory,
    }

    stream_values = list(cadence["streams"].values())
    gpu_devices = list(gpu_memory["devices"].values())
    direction = (
        block_value("laser", "beam0_direction_x1"),
        block_value("laser", "beam0_direction_x2"),
        block_value("laser", "beam0_direction_x3"),
    )
    checks = {
        "both_phases_completed": (
            run["state"] == "complete"
            and run["phase1_exit_code"] == 0 and run["phase2_exit_code"] == 0
        ),
        "stopped_exactly_at_5_and_10_ns": (
            abs(run["phase1_final_time_ns"]-5.0) < 1.0e-12
            and abs(run["phase2_final_time_ns"]-10.0) < 1.0e-12
        ),
        "incident_energy_is_10_kj": abs(laser["incident_energy_kj"]-10.0) < 1.0e-12,
        "spatial_profile_is_gaussian": laser["profile"] == "gaussian",
        "one_omega_wavelength": abs(laser["wavelength_um"]-1.053) < 1.0e-12,
        "fully_ionized_ch_composition": (
            abs(block_value("mhd", "electron_heat_capacity_fraction")-7.0/9.0) < 1.0e-14
            and abs(block_value("laser", "electron_number_per_gram")
                    -3.242691463909014e23)/3.242691463909014e23 < 1.0e-14
            and abs(block_value("laser", "beam0_zeff")-37.0/7.0) < 1.0e-14
        ),
        "active_power_matches_2_tw": laser["max_active_power_relative_error"] < 1.0e-12,
        "square_pulse_active_for_all_phase1_stages": (
            laser["min_phase1_launched_code_power"] > 0.0
        ),
        "laser_off_after_restart": laser["max_phase2_launched_code_power"] == 0.0,
        "transport_conserved": laser["max_transport_residual"] <= 1.0e-10,
        "inner_radius_resolved": abs(geometry["measured_inner_radius_mm"]-0.8) < 0.04,
        "outer_radius_resolved": abs(geometry["measured_outer_radius_mm"]-1.0) < 0.04,
        "shell_thickness_resolved": (
            abs(geometry["measured_shell_thickness_mm"]-0.2) < 0.04
        ),
        "initial_density_is_1p1_g_cc": (
            abs(geometry["measured_peak_density_g_cc"]-1.1) < 0.02
        ),
        "opening_angle_resolved": (
            abs(geometry["measured_full_opening_angle_deg"]-50.0) < 3.0
        ),
        "spot_covers_most_projected_area": geometry["projected_area_coverage"] > 0.75,
        "cap_is_on_negative_x1_side": geometry["cap_max_x1_mm"] < 0.0,
        "input_temperature_is_exactly_300_k": (
            abs(geometry["input_temperature_kelvin"]-300.0)
            < INPUT_TEMPERATURE_TOLERANCE_K
        ),
        "initial_shell_temperature_is_300_k": max(
            abs(geometry[name]-300.0) for name in (
                "shell_tion_min_kelvin", "shell_tion_max_kelvin",
                "shell_tele_min_kelvin", "shell_tele_max_kelvin",
            )
        ) < FIELD_TEMPERATURE_TOLERANCE_K,
        "initial_ambient_temperature_is_300_k": max(
            abs(geometry[name]-300.0) for name in (
                "ambient_tion_min_kelvin", "ambient_tion_max_kelvin",
                "ambient_tele_min_kelvin", "ambient_tele_max_kelvin",
            )
        ) < FIELD_TEMPERATURE_TOLERANCE_K,
        "beam_configured_from_right": (
            abs(block_value("laser", "beam0_origin_x1")
                -block_value("mesh", "x1max")) < 1.0e-14
            and direction == (-1.0, 0.0, 0.0)
        ),
        "runtime_ray_direction_is_toward_negative_x1": (
            incidence["source_side_path_weighted_direction_x1"] < 0.0
        ),
        "runtime_header_beam_is_from_right": (
            abs(incidence["runtime_header_origin_x1"]
                -block_value("mesh", "x1max")) < 1.0e-14
            and incidence["runtime_header_direction_x1"] == -1.0
            and incidence["runtime_header_direction_x2"] == 0.0
            and incidence["runtime_header_direction_x3"] == 0.0
        ),
        "upstream_absorption_is_small": incidence["upstream_deposited_fraction"] < 0.05,
        "early_deposition_is_on_target_side": incidence["deposition_centroid_x1_mm"] < 0.0,
        "binary_dt_configured_to_0p1_ns": all(
            abs(value-0.1) < 1.0e-14
            for value in cadence["configured_bin_dt_ns"].values()
        ),
        "all_101_binary_triplets_present": all(
            stream["count"] == EXPECTED_BIN_COUNT and stream["indices_complete"]
            for stream in stream_values
        ),
        "binary_triplets_are_synchronized": (
            cadence["synchronized"]
            and cadence["max_stream_sync_error_ns"] < BIN_TIME_TOLERANCE_NS
        ),
        "binary_timestamps_are_0p1_ns": all(
            stream["max_schedule_error_ns"] < BIN_TIME_TOLERANCE_NS
            for stream in stream_values
        ),
        "restart_dt_configured_to_1_ns": abs(restarts["configured_dt_ns"]-1.0) < 1.0e-14,
        "all_11_restart_outputs_present": (
            restarts["count"] == EXPECTED_RESTART_COUNT
            and restarts["indices_complete"]
            and restarts["phase1_seam_index"] == 5
        ),
        "restart_timestamps_are_0_to_10_ns": (
            restarts["max_schedule_error_ns"] < BIN_TIME_TOLERANCE_NS
            and restarts["max_interval_error_ns"] < BIN_TIME_TOLERANCE_NS
        ),
        "production_used_eight_v100_gpus": (
            gpu_memory["eight_ranks"]
            and gpu_memory["eight_unique_devices"]
            and gpu_memory["all_devices_are_v100"]
        ),
        "gpu_memory_baseline_was_idle": gpu_memory["baseline_was_idle"],
        "gpu_peak_memory_recorded": (
            gpu_memory["recorded"] and bool(gpu_devices)
            and all(device.get("complete", False) for device in gpu_devices)
        ),
        "gpu_peak_memory_fraction_is_60_to_80_percent": (
            bool(gpu_devices)
            and all(
                device.get("within_required_band", False)
                and device.get("fraction_consistency_error", math.inf) < 1.0e-6
                and device.get("delta_consistency_error_mib", math.inf) < 1.0e-6
                for device in gpu_devices
            )
        ),
    }
    warnings = []
    if laser["max_unresolved_power_fraction"] >= 5.0e-2:
        warnings.append(
            "The evolving plasma creates multiply reflecting grazing rays; "
            "remaining power is conserved but is neither deposited nor escaped."
        )
    if incidence["upstream_deposited_fraction"] >= 1.0e-2:
        warnings.append(
            "More than one percent of early deposited energy lies upstream at x1>0."
        )
    metrics["checks"] = checks
    metrics["warnings"] = warnings
    metrics["pass"] = all(checks.values())
    (CASE_DIR / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")

    peak_lines = [
        f"device {device} ({values.get('model', 'unknown')}): "
        f"delta {values.get('peak_delta_mib', math.nan):.1f}/"
        f"{values.get('total_mib', math.nan):.1f} MiB "
        f"({100.0*values.get('peak_fraction', math.nan):.2f}%), "
        f"baseline {values.get('baseline_used_mib', math.nan):.1f} MiB"
        for device, values in gpu_memory["devices"].items()
    ]
    lines = [
        "# Laser-shell run diagnostics",
        "",
        f"Overall: **{'PASS' if metrics['pass'] else 'FAIL'}**",
        "",
        "## Requested setup",
        "",
        f"- Production run: {run_dir}",
        f"- Incident pulse: {laser['incident_energy_kj']:.9g} kJ, "
        f"{laser['power_tw']:.9g} TW for {laser['duration_ns']:.9g} ns",
        f"- Wavelength: {laser['wavelength_um']:.9g} um (1 omega assumption)",
        f"- Deposited energy: {laser['deposited_energy_kj']:.9g} kJ "
        f"({100.0*laser['deposited_fraction']:.3f}% of incident)",
        f"- Maximum unresolved grazing-ray power: "
        f"{100.0*laser['max_unresolved_power_fraction']:.3f}%",
        f"- Resolved shell radii: {geometry['measured_inner_radius_mm']:.5f} to "
        f"{geometry['measured_outer_radius_mm']:.5f} mm",
        f"- Resolved shell thickness: {geometry['measured_shell_thickness_mm']:.5f} mm",
        f"- Cap x1 extent: {geometry['cap_min_x1_mm']:.5f} to "
        f"{geometry['cap_max_x1_mm']:.5f} mm",
        f"- Initial peak CH density: {geometry['measured_peak_density_g_cc']:.6g} g/cc",
        f"- Resolved full opening angle: "
        f"{geometry['measured_full_opening_angle_deg']:.5f} deg",
        f"- Initial CH Ti/Te ranges: "
        f"{geometry['shell_tion_min_kelvin']:.9g}--"
        f"{geometry['shell_tion_max_kelvin']:.9g} K / "
        f"{geometry['shell_tele_min_kelvin']:.9g}--"
        f"{geometry['shell_tele_max_kelvin']:.9g} K",
        f"- Initial ambient Ti/Te ranges: "
        f"{geometry['ambient_tion_min_kelvin']:.9g}--"
        f"{geometry['ambient_tion_max_kelvin']:.9g} K / "
        f"{geometry['ambient_tele_min_kelvin']:.9g}--"
        f"{geometry['ambient_tele_max_kelvin']:.9g} K",
        f"- Source-side path-weighted ray direction x1: "
        f"{incidence['source_side_path_weighted_direction_x1']:.6g}",
        f"- Early upstream deposited fraction: "
        f"{100.0*incidence['upstream_deposited_fraction']:.6g}%",
        f"- Early deposition centroid x1: "
        f"{incidence['deposition_centroid_x1_mm']:.6g} mm",
        f"- Binary triplets: {stream_values[0]['count']} at nominal 0.1 ns cadence",
        f"- Restart files: {restarts['count']}; maximum actual-time schedule error "
        f"{restarts['max_schedule_error_ns']:.3g} ns",
        *(f"- GPU peak memory: {line}" for line in peak_lines),
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (CASE_DIR / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
