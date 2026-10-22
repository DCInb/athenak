#!/usr/bin/env python3
"""Validate collision AMR evolution and restart checkpoints without loading all cells.

Usage: python3 verify_amr_run.py runs/amr-validation-JOB_ID
Checks conserved state, refinement ages, cycle progression, and the input/binary
manifests of all five stages produced by validate_amr.slurm. Requires NumPy.
Use --require-normal-timesteps to require a completed mixed-level step after
the initial 2x timestep-growth limit has ceased to determine the timestep.
Default validation is fresh. The harness can explicitly prepare a frozen
AMR_START_CHECKPOINT, then use --checkpoint-start to certify only its resumed
segment while retaining fresh initialization and both capacity stages.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import struct

import numpy as np


# Only these solver-work controls may differ from a historical start checkpoint.
# Accuracy tolerances, AMR order, mesh, material and physical timestep controls
# remain part of the strict physics comparison.
START_SOLVER_DEFAULTS = {
    "implicit_preconditioner": "jacobi", "implicit_max_iterations": "400",
    "implicit_residual_check_interval": "50", "implicit_gmres_restart": "30",
    "implicit_amr_inner_iterations": "0", "implicit_gmres_metric": "volume",
    "implicit_amr_inner_norm": "volume",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_parameters(stream):
    parameters = {}
    block = None
    while True:
        line = stream.readline()
        require(line, "Incomplete restart parameter header")
        line = line.decode("ascii").split("#", 1)[0].strip()
        if line == "<par_end>":
            return parameters
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            parameters[block] = {}
        elif "=" in line:
            key, value = line.split("=", 1)
            parameters[block][key.strip()] = value.strip()


def inspect_restart(path, require_adaptive=True, include_parameters=False):
    """Read this case's double precision, hydro-only, shared restart format."""
    with path.open("rb") as stream:
        parameters = read_parameters(stream)
        blocks, root_level = struct.unpack("=ii", stream.read(8))
        geometry = struct.unpack("=9d", stream.read(72))
        mesh_indices = struct.unpack("=19i", stream.read(76))
        block_indices = struct.unpack("=19i", stream.read(76))
        time, dt = struct.unpack("=dd", stream.read(16))
        cycle, = struct.unpack("=i", stream.read(4))
        locations = np.fromfile(stream, dtype=np.int32, count=4*blocks).reshape(blocks, 4)
        costs = np.fromfile(stream, dtype=np.float32, count=blocks)
        bytes_per_block, = struct.unpack("=Q", stream.read(8))
        data_offset = stream.tell()
    require(blocks > 0 and np.all(np.isfinite(costs)), "Invalid restart mesh metadata")
    require(np.all(np.isfinite(geometry)) and np.all(np.asarray(geometry[6:9]) > 0.),
            "Invalid restart cell geometry")
    require(np.isfinite(time) and time >= 0. and np.isfinite(dt) and dt >= 0.,
            "Invalid restart time metadata")
    require(not any(name in parameters for name in ("mhd", "radiation", "z4c", "turb_driving")),
            "Expected the hydro-only collision restart format")
    require(parameters["hydro"]["two_temperature"].lower() == "true",
            "Expected two-temperature hydro")
    nscalars = int(parameters["hydro"]["nscalars"])
    groups = int(parameters["thermal_radiation"]["n_groups"])
    require(nscalars == 5 and groups == 20, "Expected five tracers and twenty radiation groups")
    ng, nx, ny, nz = block_indices[:4]
    root_shape = tuple(int(parameters["mesh"][f"nx{i}"]) for i in range(1, 4))
    block_shape = tuple(int(parameters["meshblock"][f"nx{i}"]) for i in range(1, 4))
    require(mesh_indices[1:4] == root_shape, "Restart root mesh size mismatch")
    require((nx, ny, nz) == block_shape, "Restart MeshBlock size mismatch")
    root_blocks = np.asarray(root_shape)//np.asarray(block_shape)
    require(np.all(root_blocks*np.asarray(block_shape) == root_shape),
            "Invalid root block tiling")
    for axis in range(3):
        require(geometry[axis] == float(parameters["mesh"][f"x{axis+1}min"]) and
                geometry[axis+3] == float(parameters["mesh"][f"x{axis+1}max"]),
                "Restart domain bounds mismatch")
        require(np.isclose(geometry[axis+6],
                           (geometry[axis+3]-geometry[axis])/root_shape[axis],
                           rtol=1.e-14, atol=0.), "Restart cell spacing mismatch")
    shape = (blocks, 5+nscalars+2+groups, nz+2*ng, ny+2*ng, nx+2*ng)
    require(bytes_per_block == int(np.prod(shape[1:]))*8,
            "Restart field layout or precision differs from the collision setup")
    age_version = int(parameters.get("mesh_refinement", {}).get("restart_age_version", "0"))
    require(age_version in (0, 1), "Unknown AMR refinement-age footer version")
    if require_adaptive:
        require(parameters["mesh_refinement"]["refinement"] == "adaptive" and age_version == 1,
                "Adaptive checkpoint is missing its refinement-age footer")
    age_offset = data_offset+blocks*bytes_per_block
    require(path.stat().st_size == age_offset+(4*blocks if age_version else 0),
            "Incomplete or unexpected restart payload")
    ages = np.fromfile(path, dtype=np.int32, count=blocks, offset=age_offset) if age_version else []
    if age_version:
        require(np.all((ages >= 0) & (ages <= cycle)), "Invalid AMR refinement ages")
    levels = locations[:, 3]-root_level
    max_levels = int(parameters.get("mesh_refinement", {}).get("num_levels", "1"))
    require(np.all((levels >= 0) & (levels < max_levels)), "Invalid leaf refinement level")
    require(np.all(locations[:, :3] >= 0) and
            np.all(locations[:, :3] < root_blocks[None, :]*2**levels[:, None]),
            "Leaf logical location is outside the root domain")
    leaves = {tuple(location) for location in locations}
    require(len(leaves) == blocks, "Duplicate logical leaf blocks")
    for location in locations:
        for distance in range(1, int(location[3]-root_level)+1):
            ancestor = (*map(int, location[:3]//2**distance), int(location[3]-distance))
            require(ancestor not in leaves, "Overlapping coarse and fine leaf blocks")
    require(np.sum(2.**(-3*levels)) == np.prod(root_blocks),
            "Leaf blocks do not cover the complete root domain")
    fields = np.memmap(path, dtype=np.float64, mode="r", offset=data_offset, shape=shape)
    minima = np.full(shape[1], np.inf)
    tracer_error, origin_error, origin_l1 = 0., 0., 0.
    fraction_min = np.full(5, np.inf)
    fraction_max = np.full(5, -np.inf)
    origin_excess_cells = np.zeros(3, dtype=np.int64)
    origin_excess_mass = np.zeros(3)
    totals = np.zeros(shape[1])
    root_volume = np.prod(geometry[6:9])
    for block in range(blocks):
        active = fields[block, :, ng:ng+nz, ng:ng+ny, ng:ng+nx]
        require(np.all(np.isfinite(active)), f"Nonfinite conserved state in block {block}")
        minima = np.minimum(minima, active.min(axis=(1, 2, 3)))
        rho = active[0]
        require(np.all(rho > 0.), f"Nonpositive density in block {block}")
        fractions = active[5:10]/rho
        fraction_min = np.minimum(fraction_min, fractions.min(axis=(1, 2, 3)))
        fraction_max = np.maximum(fraction_max, fractions.max(axis=(1, 2, 3)))
        tracer_error = max(tracer_error, float(np.max(np.abs(active[5]+active[6]-rho)/rho)))
        origin_difference = np.abs(active[7]+active[8]+active[9]-active[5])
        origin_error = max(origin_error, float(np.max(origin_difference/rho)))
        volume = root_volume/2.**(3*levels[block])
        origin_excess_cells += np.count_nonzero(fractions[2:] > 1.+1.e-8, axis=(1, 2, 3))
        origin_excess_mass += np.maximum(active[7:10]-rho, 0.).sum(axis=(1, 2, 3))*volume
        origin_l1 += float(origin_difference.sum())*volume
        totals += active.sum(axis=(1, 2, 3))*volume
    require(minima[4] > 0., "Nonpositive material total energy")
    require(np.all(minima[10:] >= 0.), "Negative ion, electron, or radiation energy")
    require(np.all(minima[5:10] >= -1.e-14), "Negative material or origin tracer")
    require(np.all(fraction_min >= -1.e-8), "Negative material or origin mass fraction")
    require(np.all(fraction_max[:2] <= 1.+1.e-8), "EOS material mass fraction above one")
    require(tracer_error < 1.e-8, "CH plus He does not recover total density")
    # Independent scalar limiters and material normalization do not preserve this
    # redundant partition locally. EOS conversion bounds and normalizes CH/He,
    # but only enforces positivity for passive origin tags. Independent conserved
    # prolongation can therefore produce origin/rho > 1 at a refinement event.
    # Report those values and partition drift, while enforcing material bounds
    # and retaining the exact initial-condition consistency check.
    unique, counts = np.unique(levels, return_counts=True)
    mass_unit = float(parameters["units"]["mass_cgs"])
    energy_unit = mass_unit*(float(parameters["units"]["length_cgs"]) /
                             float(parameters["units"]["time_cgs"]))**2
    result = {
        "path": str(path), "cycle": cycle, "time": time, "dt": dt,
        "root_mesh": list(mesh_indices[1:4]), "meshblock": [nx, ny, nz],
        "leaf_blocks": blocks,
        "blocks_per_level": {str(int(level)): int(count) for level, count in zip(unique, counts)},
        "refinement_age_range": [int(np.min(ages)), int(np.max(ages))] if age_version else None,
        "density_min_code": float(minima[0]),
        "ion_energy_min_code": float(minima[10]),
        "electron_energy_min_code": float(minima[11]),
        "radiation_energy_min_code": float(np.min(minima[12:])),
        "material_fraction_sum_error": tracer_error, "origin_fraction_sum_error": origin_error,
        "tracer_fraction_range": [float(fraction_min.min()), float(fraction_max.max())],
        "material_fraction_range": [float(fraction_min[:2].min()),
                                    float(fraction_max[:2].max())],
        "origin_fraction_range": [float(fraction_min[2:].min()),
                                  float(fraction_max[2:].max())],
        "origin_tracers": {
            name: {"fraction_range": [float(fraction_min[index+2]),
                                      float(fraction_max[index+2])],
                   "cells_above_one": int(origin_excess_cells[index]),
                   "excess_mass_relative_CH_mass": float(origin_excess_mass[index]/totals[5])}
            for index, name in enumerate(("upper", "lower", "corona"))},
        "origin_partition_l1_relative_CH_mass": origin_l1/totals[5],
        "origin_partition_signed_relative_CH_mass": float(
            (totals[7]+totals[8]+totals[9]-totals[5])/totals[5]),
        "mass_g": float(totals[0]*mass_unit),
        "total_energy_erg": float((totals[4]+totals[12:].sum())*energy_unit),
    }
    if include_parameters:
        result["parameters"] = parameters
    return result


def compare_physics(reference, actual, label):
    """Compare requested physical settings, allowing output/report/cycle overrides."""
    for block, values in reference.items():
        if block == "job" or block.startswith("output"):
            continue
        for key, expected in values.items():
            if block == "time" and key in ("nlim", "ndiag"):
                continue
            if key in ("implicit_report", "source_report"):
                continue
            value = actual.get(block, {}).get(key)
            require(value is not None, f"Missing {label} parameter {block}/{key}")
            try:
                matches = float(value) == float(expected)
            except ValueError:
                matches = value == expected
            require(matches, f"Changed {label} parameter {block}/{key}: {value} != {expected}")


def compare_start_parameters(reference, source):
    """Return explicit solver overlays; keep physical controls strictly equal."""
    expected = {block: dict(values) for block, values in reference.items()}
    actual = {block: dict(values) for block, values in source.items()}
    overlays, changes, implicit_defaults = {}, {}, {}
    for key, default in START_SOLVER_DEFAULTS.items():
        value = expected.setdefault("thermal_radiation", {}).get(key, default)
        source_solver = actual.setdefault("thermal_radiation", {})
        present = key in source_solver
        previous = source_solver.get(key, default)
        if not present:
            require(value == default,
                    f"Cannot override absent start-checkpoint solver key {key}; "
                    "use fresh validation for a nondefault value")
            implicit_defaults[f"thermal_radiation/{key}"] = default
        expected["thermal_radiation"][key] = value
        actual["thermal_radiation"][key] = value
        if present:
            overlays[f"thermal_radiation/{key}"] = value
        if previous != value:
            changes[f"thermal_radiation/{key}"] = {"source": previous, "candidate": value}
    compare_physics(expected, actual, "start-checkpoint")
    return overlays, changes, implicit_defaults


def check_start_growth(state, parameters, evolve_cycle):
    require(0 < state["cycle"] < evolve_cycle, "Start cycle must be positive and below nlim")
    controls = parameters["time"]
    require(controls.get("align_outputs", "false").lower() in ("true", "1"),
            "Checkpoint-start validation requires saved output-alignment growth metadata")
    require(int(controls.get("restart_growth_cycle", "-1")) == state["cycle"],
            "Start checkpoint growth cycle does not match its state")
    growth_dt = float(controls.get("restart_growth_dt", "nan"))
    require(np.isfinite(growth_dt) and growth_dt > 0., "Invalid start checkpoint growth dt")
    require(controls.get("restart_growth_terminal", "true").lower() in ("false", "0"),
            "Terminal checkpoint cannot certify uninterrupted timestep growth")
    require(0. < state["time"] < float(controls["tlim"]), "Invalid start checkpoint time")
    return growth_dt


def prepare_start_checkpoint(directory, source, evolve_cycle):
    """Freeze and verify an external checkpoint before any candidate GPU stages."""
    directory = directory.resolve(strict=True)
    source = source.resolve(strict=True)
    reference = read_parameters(io.BytesIO(
        (directory / "asymmetric_collision_amr.athinput").read_bytes()+b"\n<par_end>\n"))
    source_dir = source.parent.parent
    source_log = source_dir / "run.log"
    log = source_log.read_text()
    hashes = re.findall(r"^Executable SHA-256: ([0-9a-f]{64})  (.+)$", log, re.MULTILINE)
    require(len(hashes) == 1, "Missing or ambiguous source executable provenance")
    source_executable = Path(hashes[0][1])
    require(source_executable.is_file() and file_sha256(source_executable) == hashes[0][0],
            "Source executable differs from its recorded hash")
    source_tables = source_dir / "material_tables.sha256"
    expected_tables = Path(__file__).resolve().parent.parent / "material_tables.sha256"
    require(source_tables.read_bytes() == expected_tables.read_bytes(),
            "Source checkpoint material-table manifest differs from the current case")
    frozen_dir = directory / "start"
    frozen_dir.mkdir()
    frozen = frozen_dir / "checkpoint.rst"
    original_hash = file_sha256(source)
    shutil.copy2(source, frozen)
    require(file_sha256(frozen) == original_hash, "Source checkpoint changed while freezing")
    state = inspect_restart(frozen, include_parameters=True)
    parameters = state.pop("parameters")
    growth_dt = check_start_growth(state, parameters, evolve_cycle)
    overlays, changes, implicit_defaults = compare_start_parameters(reference, parameters)
    shutil.copy2(source_log, frozen_dir / "source-run.log")
    shutil.copy2(source_tables, frozen_dir / "material_tables.sha256")
    result = {
        "mode": "checkpoint-start", "status": "preflight_passed",
        "scope": "External evolved state; candidate evolution begins at the recorded cycle.",
        "source_path": str(source), "checkpoint_sha256": original_hash,
        "source_executable_path": str(source_executable),
        "source_executable_sha256": hashes[0][0],
        "source_run_log_sha256": file_sha256(frozen_dir / "source-run.log"),
        "material_tables_manifest_sha256": file_sha256(frozen_dir / "material_tables.sha256"),
        "source_state": state, "growth_dt": growth_dt,
        "solver_overlays": overlays, "source_to_candidate_solver_changes": changes,
        "absent_solver_keys_using_constructor_defaults": implicit_defaults,
        "parameter_comparison": ("requested deck settings passed; inherited constructor "
                                 "defaults require final expanded-initial comparison"),
    }
    (frozen_dir / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    # One shell argument per line; values are parsed separately, never evaluated.
    (frozen_dir / "solver-overrides.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in overlays.items()))
    return result


def compare_resolved_start_physics(initial, source):
    """Compare both complete parameter maps, with narrow nonphysics exclusions."""
    runtime_keys = {
        ("job", "basename"), ("time", "nlim"), ("time", "ndiag"),
        ("time", "restart_growth_dt"), ("time", "restart_growth_cycle"),
        ("time", "restart_growth_terminal"),
        ("mesh_refinement", "restart_age_version"),
        ("thermal_radiation", "implicit_report"), ("thermal_radiation", "source_report"),
        ("thermal_radiation", "solver_timers"),
        ("thermal_radiation", "implicit_amr_coarse_solver"),
    }
    # Enabling/disabling a diagnostic inserts these constructor defaults. Output
    # schedules are separately checked by inspect_normal_timesteps. Other output
    # keys remain strict, as do similarly named non-output blocks.
    output_keys = {
        "dt", "dcycle", "last_time", "file_number", "data_format",
        "double_precision_binary", "ghost_zones", "gid", "single_file_per_rank",
    }
    compared = 0
    allowed = {}
    for block in sorted(set(initial) | set(source)):
        expected_values = initial.get(block, {})
        source_values = source.get(block, {})
        if not expected_values and not source_values:
            require(block in initial and block in source,
                    f"Missing resolved start block {block}")
        for key in sorted(set(expected_values) | set(source_values)):
            expected = expected_values.get(key)
            previous = source_values.get(key)
            try:
                matches = float(expected) == float(previous)
            except (TypeError, ValueError):
                matches = expected == previous
            reason = None
            if block == "thermal_radiation" and key in START_SOLVER_DEFAULTS:
                reason = "solver-work control; overlays/defaults checked separately"
            elif (block, key) in runtime_keys:
                reason = "explicit runtime/report metadata"
            elif re.fullmatch(r"output[1-9][0-9]*", block) and key in output_keys:
                reason = "output schedule or activation default"
            if reason is not None:
                if not matches:
                    allowed[f"{block}/{key}"] = {
                        "source": previous, "candidate_initial": expected, "reason": reason}
                continue
            require(expected is not None and previous is not None,
                    f"Missing resolved start parameter {block}/{key}: "
                    f"source={previous!r}, candidate_initial={expected!r}")
            require(matches, f"Changed resolved start parameter {block}/{key}: "
                    f"{previous} != {expected}")
            compared += 1
    return {"status": "passed", "compared_parameters": compared,
            "reference": "current executable fresh nlim0 fluid-header parameters",
            "allowed_differences": allowed}


def read_initial_parameters(directory):
    files = list((directory / "initial/bin").glob("*.fluid.00000.bin"))
    require(len(files) == 1, "Expected one initial fluid snapshot")
    with files[0].open("rb") as stream:
        require(stream.readline().startswith(b"Athena"), "Invalid initial binary header")
        header_count = int(stream.readline().split(b"=")[-1])
        for _ in range(header_count-1):
            stream.readline()
        stream.readline()  # number of variables
        stream.readline()  # variable names
        header_size = int(stream.readline().split(b"=")[-1])
        return read_parameters(io.BytesIO(stream.read(header_size)+b"\n<par_end>\n"))


def inspect_start_checkpoint(directory, reference, evolve_cycle, *, initial_parameters):
    directory = directory.resolve(strict=True)
    frozen_dir = directory / "start"
    result = json.loads((frozen_dir / "validation.json").read_text())
    frozen = frozen_dir / "checkpoint.rst"
    require(result["mode"] == "checkpoint-start", "Invalid start-validation mode")
    require(file_sha256(frozen) == result["checkpoint_sha256"], "Frozen start checkpoint changed")
    require(file_sha256(frozen_dir / "source-run.log") == result["source_run_log_sha256"],
            "Frozen source run manifest changed")
    require(file_sha256(frozen_dir / "material_tables.sha256") ==
            result["material_tables_manifest_sha256"], "Frozen source table manifest changed")
    require((frozen_dir / "material_tables.sha256").read_bytes() ==
            (directory / "initial/material_tables.sha256").read_bytes(),
            "Source and candidate material-table manifests differ")
    source_log = (frozen_dir / "source-run.log").read_text()
    require(re.findall(r"^Executable SHA-256: ([0-9a-f]{64}) ", source_log, re.MULTILINE) ==
            [result["source_executable_sha256"]], "Source executable record changed")
    state = inspect_restart(frozen, include_parameters=True)
    parameters = state.pop("parameters")
    growth_dt = check_start_growth(state, parameters, evolve_cycle)
    require(state == result["source_state"] and growth_dt == result["growth_dt"],
            "Start checkpoint diagnostics differ from their preflight record")
    overlays, changes, implicit_defaults = compare_start_parameters(reference, parameters)
    result["resolved_physics_comparison"] = compare_resolved_start_physics(
        initial_parameters, parameters)
    require(overlays == result["solver_overlays"] and
            changes == result["source_to_candidate_solver_changes"] and
            implicit_defaults == result["absent_solver_keys_using_constructor_defaults"],
            "Start solver controls differ from their recorded comparison")
    log = (directory / "evolve/run.log").read_text()
    commands = re.findall(r"^Command:(.*)$", log, re.MULTILINE)
    require(len(commands) == 1, "Missing checkpoint-start launch command")
    arguments = shlex.split(commands[0])
    require(arguments.count("-r") == 1 and "-i" not in arguments,
            "Checkpoint-start must load its frozen state without a full input overlay")
    require(Path(arguments[arguments.index("-r")+1]).resolve() == frozen.resolve(),
            "Evolution loaded a different start checkpoint")
    for key, value in overlays.items():
        require([arg.split("=", 1)[1] for arg in arguments if arg.startswith(key+"=")] ==
                [value], f"Missing or ambiguous recorded start override {key}")
    allowed_diagnostics = {"time/nlim", "time/ndiag", "thermal_radiation/implicit_report",
                           "thermal_radiation/source_report"}
    for argument in arguments:
        if "=" not in argument or "/" not in argument.split("=", 1)[0]:
            continue
        key = argument.split("=", 1)[0]
        require(key in overlays or key in allowed_diagnostics or
                re.fullmatch(r"output[1-9][0-9]*/[A-Za-z0-9_]+", key),
                f"Unapproved checkpoint-start command override {key}")
    result["parameters"] = parameters
    return result


def inspect_provenance(directory, evolve_cycle, restart_cycle, checkpoint_start=False):
    """Require one immutable executable and the intended deck across all stages."""
    frozen_executable = directory / "athena"
    require(frozen_executable.is_file(), "Missing frozen validation executable")
    frozen_hash = hashlib.sha256()
    with frozen_executable.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            frozen_hash.update(chunk)
    executable_hashes = set()
    reference = None
    input_hash = None
    for stage, cycle in (("initial", 0), ("evolve", evolve_cycle),
                         ("restart", restart_cycle), ("capacity", 2), ("capacity_mixed", 1)):
        stage_dir = directory / stage
        log = (stage_dir / "run.log").read_text()
        require(re.search(r"^Backend: cuda$", log, re.MULTILINE), f"{stage} did not use CUDA")
        hashes = re.findall(r"^Executable SHA-256: ([0-9a-f]{64}) ", log, re.MULTILINE)
        require(len(hashes) == 1, f"Missing or ambiguous {stage} executable hash")
        executable_hashes.add(hashes[0])
        require(re.search(rf"^time=\S+ cycle={cycle}$", log, re.MULTILINE),
                f"{stage} did not finish the expected cycle")
        if stage == "capacity":
            require(re.search(r"^Current number of MeshBlocks = 432$", log, re.MULTILINE),
                    "Full-capacity stage did not reach 432 leaves")
        if stage == "capacity_mixed":
            require(re.search(r"^Total number of MeshBlocks = 425$", log, re.MULTILINE),
                    "Mixed-capacity stage did not start with 425 leaves")
        inputs = list(stage_dir.glob("*.athinput"))
        require(len(inputs) == 1, f"Expected one copied {stage} input")
        contents = inputs[0].read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        recorded = re.findall(r"^Input SHA-256: ([0-9a-f]{64}) ", log, re.MULTILINE)
        require(recorded == [digest], f"Copied {stage} input differs from its run manifest")
        parameters = read_parameters(io.BytesIO(contents+b"\n<par_end>\n"))
        if stage == "initial":
            reference, input_hash = parameters, digest
            require(tuple(int(reference["mesh"][f"nx{i}"]) for i in (1, 2, 3)) ==
                    (108, 108, 108), "Validation did not use the production 108^3 root mesh")
            require(tuple(int(reference["meshblock"][f"nx{i}"]) for i in (1, 2, 3)) ==
                    (18, 36, 36), "Validation did not use the production block dimensions")
            require(reference["mesh_refinement"]["refinement"] == "adaptive" and
                    int(reference["mesh_refinement"]["num_levels"]) == 2 and
                    int(reference["mesh_refinement"]["max_nmb_per_rank"]) == 432,
                    "Validation did not reserve the production two-level AMR capacity")
        elif stage == "capacity_mixed":
            unrefined = {key: value for key, value in parameters.items()
                         if not key.startswith("refined_region")}
            require(unrefined == reference, "Mixed capacity input changed production settings")
        else:
            require(digest == input_hash, f"Production input changed before {stage}")
        if stage in ("evolve", "restart"):
            commands = re.findall(r"^Command:(.*)$", log, re.MULTILINE)
            require(len(commands) == 1, f"Missing {stage} launch command")
            arguments = shlex.split(commands[0])
        if stage == "evolve" and not checkpoint_start:
            require(arguments.count("-i") == 1 and "-r" not in arguments,
                    "Fresh evolution unexpectedly loaded a checkpoint")
        if stage == "restart":
            require(arguments.count("-r") == 1, "Restart stage did not load a checkpoint")
            restart_path = Path(arguments[arguments.index("-r")+1]).resolve()
            expected = sorted((directory / "evolve/rst").glob("*.rst"))[-1].resolve()
            require(restart_path == expected, "Restart stage loaded a different checkpoint")
    require(len(executable_hashes) == 1, "Executable changed between GPU validation stages")
    require(executable_hashes == {frozen_hash.hexdigest()},
            "Frozen executable differs from the recorded validation binary")
    return reference, executable_hashes.pop(), input_hash


def inspect_normal_timesteps(log, parameters, expected_cycle, start=None):
    """Certify completed normal-control steps inside a verified evolution segment.

    Driver logs (cycle, time, dt) BEFORE each step, then logs the unused next dt
    again in Finalize. Only a row followed by cycle+1 certifies a completed step.
    Mesh::NewTimeStep selects min(2*previous_physics_dt, physics limits, tlim).
    Exclude terminal/output clipping and the first step of a restart before interpreting a strict
    departure from doubling as a physical timestep limit. Scientific diagnostics
    have six decimal places; differences below their rounding error are ambiguous.
    """
    commands = re.findall(r"^Command:(.*)$", log, re.MULTILINE)
    require(len(commands) == 1, "Normal-timestep check needs one launch command")
    arguments = shlex.split(commands[0])
    if start is None:
        require("-i" in arguments and not any(arg.startswith("-r") for arg in arguments),
                "Normal-timestep check requires fresh evolution, not a restart")
    else:
        require(arguments.count("-r") == 1 and "-i" not in arguments,
                "Verified checkpoint-start evolution requires one restart input")
    effective = {block: dict(values) for block, values in
                 (parameters if start is None else start["parameters"]).items()}
    for argument in arguments:
        if "=" in argument and "/" in argument.split("=", 1)[0]:
            name, value = argument.split("=", 1)
            block, key = name.split("/", 1)
            effective.setdefault(block, {})[key] = value

    def number(value, label):
        result = float(value)
        require(np.isfinite(result), f"Nonfinite {label}")
        return result

    def close(left, right):
        return (np.isfinite(left) and np.isfinite(right) and
                abs(left-right) <= 2.e-6*max(abs(left), abs(right)))

    time_parameters = effective["time"]
    require(int(time_parameters["ndiag"]) == 1, "Normal-timestep check requires ndiag=1")
    require(int(time_parameters["nlim"]) == expected_cycle and expected_cycle > 1,
            "Normal-timestep check needs the requested cycle limit")
    initial_dt = number(time_parameters["initial_dt"], "initial_dt")
    tlim = number(time_parameters["tlim"], "tlim")
    require(initial_dt > 0. and tlim > 0., "Nonpositive startup or terminal time")
    require(number(time_parameters["cfl_number"], "CFL number") > 0.,
            "Nonpositive CFL number")
    require(log.count("\nTerminating on cycle limit\n") == 1,
            "Normal-timestep evolution did not terminate on its cycle limit")
    summaries = re.findall(r"^time=(\S+) cycle=(\d+)$", log, re.MULTILINE)
    require(len(summaries) == 1 and int(summaries[0][1]) == expected_cycle,
            "Missing or inconsistent final cycle summary")
    final_time = number(summaries[0][0], "final time")
    require(0. < final_time < tlim and not close(final_time, tlim),
            "Terminal time clipping cannot certify a normal timestep")

    # Exclude any possible aligned-output event inside the checked segment.
    # Fresh outputs start at zero; restarted schedules retain their saved counters.
    # A dcycle key overrides dt in Outputs, including dcycle=0 (disabled output).
    align = time_parameters.get("align_outputs", "false").lower()
    require(align in ("true", "false", "1", "0"), "Invalid align_outputs value")
    if align in ("true", "1"):
        for block, values in effective.items():
            if block.startswith("output") and "dcycle" not in values:
                cadence = number(values.get("dt", "0"), f"{block}/dt")
                last_time = (0. if start is None or cadence <= 0. else
                             number(values.get("last_time", "nan"), f"{block}/last_time"))
                require(cadence <= 0. or last_time >= 0.,
                        f"Unknown restarted {block} output schedule")
                next_event = last_time+cadence
                require(cadence <= 0. or
                        (final_time < next_event and not close(final_time, next_event)),
                        f"Aligned {block} output can mimic a physical timestep limit")

    rows = []
    for match in re.finditer(r"^elapsed=.*$", log, re.MULTILINE):
        fields = re.fullmatch(r"elapsed=(\S+) cycle=(\d+) time=(\S+) dt=(\S+)(?: .*)?",
                              match[0])
        require(fields is not None, "Malformed cycle diagnostic")
        elapsed, cycle, time, dt = fields.groups()
        row = {"cycle": int(cycle), "elapsed": number(elapsed, "elapsed time"),
               "time": number(time, "cycle time"), "dt": number(dt, "cycle dt"),
               "start": match.start(), "end": match.end()}
        require(row["elapsed"] >= 0. and row["time"] >= 0. and row["dt"] > 0.,
                "Nonpositive timestep or negative diagnostic time")
        rows.append(row)
    start_cycle = 0 if start is None else start["source_state"]["cycle"]
    require([row["cycle"] for row in rows] == list(range(start_cycle, expected_cycle+1)),
            "Expected contiguous cycle diagnostics from the verified start through nlim")
    if start is None:
        require(rows[0]["time"] == 0. and
                rows[0]["dt"] <= initial_dt*(1.+2.e-6), "Invalid fresh startup timestep")
    else:
        growth_dt = check_start_growth(start["source_state"], start["parameters"], expected_cycle)
        require(close(rows[0]["time"], start["source_state"]["time"]),
                "First diagnostic time differs from the verified start checkpoint")
        require(rows[0]["dt"] <= 2.*growth_dt*(1.+2.e-6),
                "First restarted timestep exceeds the saved 2x growth bound")
    require(close(rows[-1]["time"], final_time), "Final diagnostic time differs from summary")
    for current, following in zip(rows, rows[1:]):
        require(following["elapsed"] >= current["elapsed"] and
                following["time"] > current["time"], "Nonincreasing diagnostic sequence")
        require(close(current["time"]+current["dt"], following["time"]),
                f"Cycle {current['cycle']} did not advance by its logged timestep")
        require(following["dt"] <= 2.*current["dt"]*(1.+2.e-6),
                "Timestep exceeds the 2x growth limit; hidden clipping or restart")

    normal_steps = []
    refined_steps = []
    # All these messages originate in the mixed-level high-order solve. A final
    # refined checkpoint alone cannot prove refinement preceded a completed step.
    mixed_marker = re.compile(
        r"^# implicit radiation (?:AMR (?:fallback:|GMRES\b)|"
        r"phase (?:start|complete):[^\n]* solver=high_order_(?:f?gmres)\b|"
        r"Krylov: solver=(?:FGMRES|GMRES)\b)", re.MULTILINE)
    for index in range(1, len(rows)-1):
        previous, current, following = rows[index-1:index+2]
        if current["dt"] >= 2.*previous["dt"]*(1.-2.e-6):
            continue
        evidence = {"cycle": current["cycle"], "completed_cycle": following["cycle"],
                    "time": current["time"], "completed_time": following["time"],
                    "dt": current["dt"], "previous_dt": previous["dt"],
                    "growth_ratio": current["dt"]/previous["dt"]}
        normal_steps.append(evidence)
        if mixed_marker.search(log[current["end"]:following["start"]]):
            refined_steps.append(evidence)
    require(normal_steps, "No completed timestep left the startup 2x growth limit")
    require(refined_steps, "No completed normal timestep has mixed-level solver evidence")
    return {"status": "passed", "completed_steps": expected_cycle-start_cycle,
            "start_cycle": start_cycle, "end_cycle": expected_cycle,
            "mode": "fresh" if start is None else "checkpoint-start",
            "restart_boundary_step_counted_as_normal": False,
            "normal_control_steps": len(normal_steps),
            "refined_normal_control_steps": len(refined_steps),
            "first_refined_normal_step": refined_steps[0],
            "last_refined_normal_step": refined_steps[-1],
            "final_planned_dt_not_counted": rows[-1]["dt"]}


def verify(directory, evolve_cycle=8, restart_cycle=10, require_normal_timesteps=False, *,
           checkpoint_start=False):
    result = {}
    require(restart_cycle > evolve_cycle >= 0, "Expected increasing checkpoint cycles")
    reference, executable_hash, input_hash = inspect_provenance(
        directory, evolve_cycle, restart_cycle, checkpoint_start)
    initial_parameters = read_initial_parameters(directory)
    compare_physics(reference, initial_parameters, "initial")
    start = None
    if checkpoint_start:
        start = inspect_start_checkpoint(
            directory, reference, evolve_cycle, initial_parameters=initial_parameters)
    result["validation_mode"] = "checkpoint-start" if checkpoint_start else "fresh"
    result["candidate_start_cycle"] = 0 if start is None else start["source_state"]["cycle"]
    result["candidate_evolution_steps"] = evolve_cycle-result["candidate_start_cycle"]
    if start is not None:
        result["start_checkpoint"] = {
            key: value for key, value in start.items() if key != "parameters"}
        result["scope"] = ("Current frozen executable: fresh initialization, checkpoint-start "
                           "evolution, continuation restart and both capacity stages. The "
                           "historical prefix is external evidence, not current-binary "
                           "fresh evolution.")
    if require_normal_timesteps:
        result["normal_timesteps"] = inspect_normal_timesteps(
            (directory / "evolve/run.log").read_text(), reference, evolve_cycle, start)
    for stage, cycle in (("evolve", evolve_cycle), ("restart", restart_cycle)):
        files = sorted((directory / stage / "rst").glob("*.rst"))
        require(files, f"Missing {stage} checkpoint")
        result[stage] = inspect_restart(files[-1], include_parameters=True)
        compare_physics(reference, result[stage].pop("parameters"), stage)
        require(result[stage]["cycle"] == cycle, f"Expected {stage} checkpoint cycle {cycle}")
        require(set(result[stage]["blocks_per_level"]) == {"0", "1"},
                f"Expected both coarse and fine leaves after {stage}")
    if start is not None:
        require(result["evolve"]["time"] > start["source_state"]["time"],
                "Checkpoint-start evolution did not advance time")
    require(result["restart"]["time"] > result["evolve"]["time"], "Restart did not advance time")
    with (directory / "gpu-memory.csv").open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    require(rows, "Missing GPU memory samples")
    used = [float(row["memory.used [MiB]"].split()[0]) for row in rows]
    total = [float(row["memory.total [MiB]"].split()[0]) for row in rows]
    require(np.all(np.isfinite(used)) and np.all(np.isfinite(total)) and
            min(used) >= 0. and max(used) > 0. and min(total) > 0., "Invalid GPU memory samples")
    require(len({row["uuid"] for row in rows}) == 1, "GPU monitor changed devices")
    result["gpu_peak_used_MiB"] = max(used)
    result["gpu_total_MiB"] = min(total)
    require(max(used) < min(total), "GPU memory capacity reached")
    result["executable_sha256"] = executable_hash
    # This immutable validation snapshot includes a restart-output scheduling
    # override; its byte hash is intentionally distinct from the production deck.
    result["validation_input_sha256"] = input_hash
    result["gpu_memory_samples"] = len(rows)
    result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_directory", type=Path)
    parser.add_argument("--evolve-cycle", type=int, default=8)
    parser.add_argument("--restart-cycle", type=int, default=10)
    parser.add_argument("--require-normal-timesteps", action="store_true",
                        help="require a completed mixed-level step after startup dt growth")
    start_mode = parser.add_mutually_exclusive_group()
    start_mode.add_argument("--checkpoint-start", action="store_true",
                            help="validate the prepared external checkpoint-start segment")
    start_mode.add_argument("--prepare-start-checkpoint", type=Path,
                            help="freeze/check an external checkpoint before GPU stages")
    args = parser.parse_args()
    if args.prepare_start_checkpoint is not None:
        result = prepare_start_checkpoint(args.validation_directory, args.prepare_start_checkpoint,
                                          args.evolve_cycle)
    else:
        result = verify(args.validation_directory, args.evolve_cycle, args.restart_cycle,
                        args.require_normal_timesteps, checkpoint_start=args.checkpoint_start)
    print(json.dumps(result, indent=2))
