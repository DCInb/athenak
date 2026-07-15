#!/usr/bin/env python3
"""Analyze the FLASH laser-tube benchmark on an AthenaK static-refinement grid."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
UNIFORM_DIR = BENCH_DIR.parent / "flash-laser-tube"
CODATA_EXIT = 0.7017811414419318
N_RAYS = 8
INTERFACE_X1 = math.pi
DOMAIN_X1_MAX = 2.0 * math.pi


def parse_laser_log(path: Path) -> dict[str, float]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("laser:")]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one laser diagnostic in {path}, found {len(lines)}")
    return {
        key: float(value)
        for key, value in re.findall(
            r"([a-z_]+)=([+-]?[0-9.]+(?:e[+-]?[0-9]+)?)", lines[0]
        )
    }


def read_multilevel_binary(path: Path) -> dict:
    """Read native AthenaK MeshBlocks without flattening different SMR levels."""
    with path.open("rb") as binary_file:
        binary_file.seek(0, 2)
        file_size = binary_file.tell()
        binary_file.seek(0)
        if binary_file.readline().split()[-1] != b"version=1.1":
            raise RuntimeError(f"Unexpected binary version in {path}")

        preheader_lines = int(binary_file.readline().split(b"=")[-1])
        preheader = {}
        for _ in range(preheader_lines - 1):
            key, value = binary_file.readline().decode().split("=")
            preheader[key.strip()] = value.strip()
        location_size = int(preheader["size of location"])
        variable_size = int(preheader["size of variable"])
        nvars = int(binary_file.readline().split(b"=")[-1])
        names = [entry.decode() for entry in binary_file.readline().split()[1:]]
        header_size = int(binary_file.readline().split(b"=")[-1])
        header = binary_file.read(header_size).decode().splitlines()

        parameters = {}
        section = ""
        for line in header:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("<"):
                section = line
            elif "=" in line:
                key, value = line.split("=", 1)
                parameters[(section, key.strip())] = value.strip()

        root_nx = tuple(
            int(parameters[("<mesh>", f"nx{axis}")]) for axis in (1, 2, 3)
        )
        meshblock_nx = tuple(
            int(parameters[("<meshblock>", f"nx{axis}")]) for axis in (1, 2, 3)
        )
        location_dtype = np.float64 if location_size == 8 else np.float32
        variable_dtype = np.float64 if variable_size == 8 else np.float32

        blocks = []
        while binary_file.tell() < file_size:
            indices = np.frombuffer(binary_file.read(24), dtype=np.int32)
            local_nx = indices[1] - indices[0] + 1
            local_ny = indices[3] - indices[2] + 1
            local_nz = indices[5] - indices[4] + 1
            logical = tuple(
                int(value)
                for value in np.frombuffer(binary_file.read(16), dtype=np.int32)
            )
            bounds = tuple(
                float(value)
                for value in np.frombuffer(
                    binary_file.read(6 * location_size), dtype=location_dtype
                )
            )
            data = np.fromfile(
                binary_file,
                dtype=variable_dtype,
                count=nvars * local_nx * local_ny * local_nz,
            ).reshape(nvars, local_nz, local_ny, local_nx)
            blocks.append({
                "logical": logical,
                "bounds": bounds,
                "fields": {name: values for name, values in zip(names, data)},
            })

    return {
        "names": names,
        "root_nx": root_nx,
        "meshblock_nx": meshblock_nx,
        "blocks": blocks,
    }


def block_cell_sizes(block: dict) -> tuple[float, float, float]:
    x1min, x1max, x2min, x2max, x3min, x3max = block["bounds"]
    nz, ny, nx = block["fields"]["laser_path"].shape
    return (x1max - x1min) / nx, (x2max - x2min) / ny, (x3max - x3min) / nz


def field_metrics(dataset: dict) -> dict:
    path_sum = 0.0
    trajectory_error_sum = 0.0
    coarse_path = 0.0
    fine_path = 0.0
    coarse_error_sum = 0.0
    fine_error_sum = 0.0
    deposited_from_field = 0.0
    deposited_by_level = {0: 0.0, 1: 0.0}
    block_counts = {0: 0, 1: 0}
    focus_min_segment_radius = math.inf
    finest_dx1 = min(block_cell_sizes(block)[0] for block in dataset["blocks"])

    for block in dataset["blocks"]:
        level = block["logical"][3]
        block_counts[level] = block_counts.get(level, 0) + 1
        fields = block["fields"]
        path = np.asarray(fields["laser_path"], dtype=np.float64)
        occupied = path > 0.0
        dx1, dx2, dx3 = block_cell_sizes(block)
        deposited = float(np.sum(fields["laser_q"], dtype=np.float64) * dx1 * dx2 * dx3)
        deposited_from_field += deposited
        deposited_by_level[level] = deposited_by_level.get(level, 0.0) + deposited
        if not np.any(occupied):
            continue

        x1 = np.zeros(path.shape, dtype=np.float64)
        x2 = np.zeros(path.shape, dtype=np.float64)
        x3 = np.zeros(path.shape, dtype=np.float64)
        np.divide(fields["laser_x1_moment"], path, out=x1, where=occupied)
        np.divide(fields["laser_x2_moment"], path, out=x2, where=occupied)
        np.divide(fields["laser_x3_moment"], path, out=x3, where=occupied)
        radius = np.hypot(x2, x3)
        exact_radius = 3.0 * np.cos(x1 / 4.0)
        absolute_error = np.abs(radius - exact_radius)

        local_path = float(np.sum(path[occupied]))
        local_error = float(np.sum(path[occupied] * absolute_error[occupied]))
        path_sum += local_path
        trajectory_error_sum += local_error
        if level == 0:
            coarse_path += local_path
            coarse_error_sum += local_error
        elif level == 1:
            fine_path += local_path
            fine_error_sum += local_error

        focus = occupied & (x1 > DOMAIN_X1_MAX - 2.0 * finest_dx1)
        if np.any(focus):
            focus_min_segment_radius = min(
                focus_min_segment_radius, float(np.min(radius[focus]))
            )

    return {
        "meshblocks": len(dataset["blocks"]),
        "level0_meshblocks": block_counts.get(0, 0),
        "level1_meshblocks": block_counts.get(1, 0),
        "path_on_level0_cm": coarse_path,
        "path_on_level1_cm": fine_path,
        "trajectory_path_weighted_l1_cm": trajectory_error_sum / path_sum,
        "trajectory_level0_l1_cm": coarse_error_sum / coarse_path,
        "trajectory_level1_l1_cm": fine_error_sum / fine_path,
        "focus_min_segment_radius_cm": focus_min_segment_radius,
        "deposited_from_field": deposited_from_field,
        "deposited_on_level0": deposited_by_level.get(0, 0.0),
        "deposited_on_level1": deposited_by_level.get(1, 0.0),
    }


def compare_datasets(reference: dict, candidate: dict) -> dict:
    reference_blocks = {
        tuple(block["logical"]): block for block in reference["blocks"]
    }
    candidate_blocks = {
        tuple(block["logical"]): block for block in candidate["blocks"]
    }
    same_layout = reference_blocks.keys() == candidate_blocks.keys()
    max_difference = 0.0
    same_fields = same_layout and reference["names"] == candidate["names"]
    if same_fields:
        for key, reference_block in reference_blocks.items():
            candidate_block = candidate_blocks[key]
            same_fields = same_fields and np.allclose(
                reference_block["bounds"], candidate_block["bounds"], rtol=0.0, atol=0.0
            )
            for name in reference["names"]:
                difference = float(
                    np.max(
                        np.abs(
                            reference_block["fields"][name]
                            - candidate_block["fields"][name]
                        )
                    )
                )
                max_difference = max(max_difference, difference)
                same_fields = same_fields and np.allclose(
                    reference_block["fields"][name],
                    candidate_block["fields"][name],
                    rtol=2.0e-11,
                    atol=2.0e-13,
                )
    return {
        "pass": bool(same_fields),
        "same_layout": bool(same_layout),
        "max_abs_field_difference": max_difference,
    }


def output_path(row: dict[str, str]) -> Path:
    return (
        Path(row["run_dir"])
        / "bin"
        / "flash_laser_tube_smr.flash_tube_smr.00001.bin"
    )


def main() -> int:
    rows = list(csv.DictReader((BENCH_DIR / "run_status.csv").open()))
    results = []
    datasets = {}
    for row in rows:
        diagnostic = parse_laser_log(Path(row["log"]))
        dataset = read_multilevel_binary(output_path(row))
        datasets[row["case"]] = dataset
        metrics = field_metrics(dataset)
        exit_per_ray = diagnostic["escaped"] / N_RAYS
        relative_error = abs(exit_per_ray - CODATA_EXIT) / CODATA_EXIT
        field_log_difference = abs(
            metrics["deposited_from_field"] - diagnostic["deposited"]
        )
        mesh_ok = bool(
            metrics["meshblocks"] == 36
            and metrics["level0_meshblocks"] == 4
            and metrics["level1_meshblocks"] == 32
            and metrics["path_on_level0_cm"] > 0.0
            and metrics["path_on_level1_cm"] > 0.0
        )
        mpi_transport_ok = bool(
            int(row["gpus"]) == 1 or diagnostic["transfers"] > 0.0
        )
        physics_ok = bool(
            int(row["exit_code"]) == 0
            and relative_error < 1.0e-3
            and diagnostic["residual"] < 1.0e-10
            and diagnostic["active"] == 0.0
            and diagnostic["dispersion"] < 5.0e-3
            and metrics["trajectory_path_weighted_l1_cm"] < 1.0e-2
            and field_log_difference < 2.0e-6
            and mesh_ok
            and mpi_transport_ok
        )
        result = {
            "case": row["case"],
            "gpus": int(row["gpus"]),
            "run_ok": int(row["exit_code"]) == 0,
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "meshblocks": metrics["meshblocks"],
            "level0_meshblocks": metrics["level0_meshblocks"],
            "level1_meshblocks": metrics["level1_meshblocks"],
            "exit_power_per_ray": exit_per_ray,
            "relative_error_vs_codata": relative_error,
            "deposited_power": diagnostic["deposited"],
            "deposited_field_log_difference": field_log_difference,
            "deposited_fraction_on_fine_level":
                metrics["deposited_on_level1"] / metrics["deposited_from_field"],
            "conservation_residual": diagnostic["residual"],
            "off_rank_transfers": diagnostic["transfers"],
            "traced_segments": diagnostic["segments"],
            "max_dispersion_error": diagnostic["dispersion"],
            "path_on_level0_cm": metrics["path_on_level0_cm"],
            "path_on_level1_cm": metrics["path_on_level1_cm"],
            "trajectory_path_weighted_l1_cm":
                metrics["trajectory_path_weighted_l1_cm"],
            "trajectory_level0_l1_cm": metrics["trajectory_level0_l1_cm"],
            "trajectory_level1_l1_cm": metrics["trajectory_level1_l1_cm"],
            "focus_min_segment_radius_cm": metrics["focus_min_segment_radius_cm"],
            "mesh_ok": mesh_ok,
            "mpi_transport_ok": mpi_transport_ok,
            "physics_ok": physics_ok,
        }
        results.append(result)

    reference = datasets["smr_gpu1"]
    gpu_comparisons = {
        case: compare_datasets(reference, datasets[case])
        for case in ("smr_gpu2", "smr_gpu4")
    }
    overall = bool(
        all(result["physics_ok"] for result in results)
        and all(comparison["pass"] for comparison in gpu_comparisons.values())
    )

    uniform_reference = {}
    uniform_results_path = UNIFORM_DIR / "results.csv"
    if uniform_results_path.is_file():
        for row in csv.DictReader(uniform_results_path.open()):
            if row["case"] in ("r32_gpu1", "r64_gpu1"):
                uniform_reference[row["case"]] = {
                    "exit_power_per_ray": float(row["exit_power_per_ray"]),
                    "relative_error_vs_codata": float(row["relative_error_vs_codata"]),
                    "max_dispersion_error": float(row["max_dispersion_error"]),
                }

    payload = {
        "overall": overall,
        "codata_exit": CODATA_EXIT,
        "interface_x1": INTERFACE_X1,
        "cases": results,
        "gpu_comparisons": gpu_comparisons,
        "uniform_reference": uniform_reference,
    }
    (BENCH_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (BENCH_DIR / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# FLASH quadratic laser-tube SMR benchmark",
        "",
        f"Overall result: {'PASS' if overall else 'FAIL'}",
        "",
        "Grid configuration:",
        "",
        "- Root grid: 32^3 with 16^3-cell MeshBlocks.",
        "- Static level-1 refinement covers pi <= x1 <= 2*pi over the full transverse domain.",
        "- Leaf layout: 4 level-0 blocks and 32 level-1 blocks (36 total).",
        "- Every ray crosses the x1=pi coarse-fine interface and reaches the focus on the fine grid.",
        "",
        "| Case | GPUs | Exit power/ray | Rel. error | Track L1 cm | Dispersion | Transfers | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | {result['gpus']} | "
            f"{result['exit_power_per_ray']:.10f} | "
            f"{result['relative_error_vs_codata']:.3e} | "
            f"{result['trajectory_path_weighted_l1_cm']:.3e} | "
            f"{result['max_dispersion_error']:.3e} | "
            f"{int(result['off_rank_transfers'])} | "
            f"{'PASS' if result['physics_ok'] else 'FAIL'} |"
        )
    representative = results[0]
    lines.extend([
        "",
        "Representative 1-GPU multilevel diagnostics:",
        "",
        f"- Ray path on level 0: {representative['path_on_level0_cm']:.8e} cm.",
        f"- Ray path on level 1: {representative['path_on_level1_cm']:.8e} cm.",
        f"- Level-0 trajectory L1: {representative['trajectory_level0_l1_cm']:.8e} cm.",
        f"- Level-1 trajectory L1: {representative['trajectory_level1_l1_cm']:.8e} cm.",
        f"- Fine-level deposited-power fraction: {representative['deposited_fraction_on_fine_level']:.8f}.",
        f"- Minimum segment radius near focus: {representative['focus_min_segment_radius_cm']:.8e} cm.",
        f"- Field-integrated versus logged deposition difference: "
        f"{representative['deposited_field_log_difference']:.3e}.",
        "",
        "GPU decomposition comparisons:",
        "",
    ])
    for case, comparison in gpu_comparisons.items():
        lines.append(
            f"- {case}: {'PASS' if comparison['pass'] else 'FAIL'}, "
            f"same leaf layout={comparison['same_layout']}, "
            f"max absolute field difference={comparison['max_abs_field_difference']:.16e}."
        )
    if uniform_reference:
        lines.extend(["", "Uniform-grid reference:", ""])
        for case, reference_result in uniform_reference.items():
            lines.append(
                f"- {case}: exit={reference_result['exit_power_per_ray']:.10f}, "
                f"relative error={reference_result['relative_error_vs_codata']:.3e}, "
                f"dispersion={reference_result['max_dispersion_error']:.3e}."
            )
    lines.extend([
        "",
        "Scope:",
        "",
        "This is an SMR transport/interface regression. It checks refractive stepping,",
        "coarse-fine block lookup, inverse-bremsstrahlung deposition with level-dependent",
        "cell volumes, MPI ray transfers, and decomposition-independent output.",
        "",
    ])
    (BENCH_DIR / "diagnostics.md").write_text("\n".join(lines))

    print(f"Overall result: {'PASS' if overall else 'FAIL'}")
    for result in results:
        print(
            f"{result['case']:9s} exit={result['exit_power_per_ray']:.10f} "
            f"track_l1={result['trajectory_path_weighted_l1_cm']:.3e} "
            f"transfers={int(result['off_rank_transfers'])} "
            f"result={'PASS' if result['physics_ok'] else 'FAIL'}"
        )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
