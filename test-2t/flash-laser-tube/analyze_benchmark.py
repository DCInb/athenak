#!/usr/bin/env python3
"""Analyze the AthenaK adaptation of FLASH Energy Deposition unit test I."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
FLASH_PUBLISHED_EXIT = 0.7017811
CODATA_EXIT = 0.7017811414419318
N_RAYS = 8


def parse_laser_log(path: Path) -> dict[str, float]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("laser:")]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one laser diagnostic in {path}, found {len(lines)}")
    return {key: float(value) for key, value in
            re.findall(r"([a-z_]+)=([+-]?[0-9.]+(?:e[+-]?[0-9]+)?)", lines[0])}


def read_laser_binary(path: Path):
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
        block = ""
        for line in header:
            line = line.split("#")[0].strip()
            if line.startswith("<"):
                block = line
            elif "=" in line:
                key, value = line.split("=", 1)
                parameters[(block, key.strip())] = value.strip()
        nx = tuple(int(parameters[("<mesh>", f"nx{axis}")]) for axis in (1, 2, 3))
        mb = tuple(int(parameters[("<meshblock>", f"nx{axis}")]) for axis in (1, 2, 3))
        fields = {name: np.zeros((nx[2], nx[1], nx[0])) for name in names}
        dtype = np.float64 if variable_size == 8 else np.float32
        while binary_file.tell() < file_size:
            indices = np.frombuffer(binary_file.read(24), dtype=np.int32)
            local_nx = indices[1] - indices[0] + 1
            local_ny = indices[3] - indices[2] + 1
            local_nz = indices[5] - indices[4] + 1
            logical = np.frombuffer(binary_file.read(16), dtype=np.int32)
            binary_file.read(6 * location_size)
            data = np.fromfile(binary_file, dtype=dtype,
                               count=nvars * local_nx * local_ny * local_nz)
            data = data.reshape(nvars, local_nz, local_ny, local_nx)
            i0, j0, k0 = logical[0] * mb[0], logical[1] * mb[1], logical[2] * mb[2]
            for name, values in zip(names, data):
                fields[name][k0:k0+local_nz, j0:j0+local_ny,
                             i0:i0+local_nx] = values
        return fields, nx


def trajectory_error(fields, nx):
    path = fields["laser_path"]
    mask = path > 0.0
    dx1 = 2.0 * math.pi / nx[0]
    dx2 = 10.0 / nx[1]
    dx3 = 10.0 / nx[2]
    k, j, i = np.indices(path.shape)
    x1 = (i + 0.5) * dx1
    x2 = -5.0 + (j + 0.5) * dx2
    x3 = -5.0 + (k + 0.5) * dx3
    radius = np.sqrt(x2*x2 + x3*x3)
    exact_radius = 3.0 * np.cos(x1 / 4.0)
    absolute = np.abs(radius - exact_radius)
    weighted_l1 = float(np.sum(path[mask] * absolute[mask]) / np.sum(path[mask]))
    normalized_l1 = weighted_l1 / math.sqrt(dx2*dx2 + dx3*dx3)
    focus_mask = mask & (x1 > 2.0*math.pi - 2.0*dx1)
    focus_radius = float(np.min(radius[focus_mask])) if np.any(focus_mask) else math.inf
    return weighted_l1, normalized_l1, focus_radius


def output_path(row):
    basename = f"flash_laser_tube_r{row['resolution']}"
    return Path(row["run_dir"]) / "bin" / f"{basename}.flash_tube.00001.bin"


def main() -> int:
    rows = list(csv.DictReader((BENCH_DIR / "run_status.csv").open()))
    results = []
    fields_by_case = {}
    for row in rows:
        diagnostic = parse_laser_log(Path(row["log"]))
        fields, nx = read_laser_binary(output_path(row))
        fields_by_case[row["case"]] = fields
        exit_per_ray = diagnostic["escaped"] / N_RAYS
        relative_error = abs(exit_per_ray - CODATA_EXIT) / CODATA_EXIT
        published_error = abs(exit_per_ray - FLASH_PUBLISHED_EXIT) / FLASH_PUBLISHED_EXIT
        path_l1, path_cells, focus_radius = trajectory_error(fields, nx)
        result = {
            "case": row["case"], "resolution": int(row["resolution"]),
            "gpus": int(row["gpus"]), "run_ok": int(row["exit_code"]) == 0,
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "exit_power_per_ray": exit_per_ray,
            "relative_error_vs_codata": relative_error,
            "relative_error_vs_flash_published": published_error,
            "deposited_power": diagnostic["deposited"],
            "conservation_residual": diagnostic["residual"],
            "active_rays": diagnostic["active"],
            "off_rank_transfers": diagnostic["transfers"],
            "traced_segments": diagnostic["segments"],
            "max_dispersion_error": diagnostic["dispersion"],
            "trajectory_path_weighted_l1_cm": path_l1,
            "trajectory_l1_in_transverse_cells": path_cells,
            "focus_min_cell_radius_cm": focus_radius,
        }
        result["physics_ok"] = bool(
            result["run_ok"] and relative_error < 1.0e-3 and
            result["conservation_residual"] < 1.0e-10 and
            result["active_rays"] == 0.0 and
            result["max_dispersion_error"] < 5.0e-3 and path_cells < 1.5)
        results.append(result)

    r32 = next(item for item in results if item["case"] == "r32_gpu1")
    r64 = next(item for item in results if item["case"] == "r64_gpu1")
    convergence_ok = (r32["relative_error_vs_codata"] < 1.0e-3 and
                      r64["relative_error_vs_codata"] < 1.0e-3 and
                      r64["trajectory_path_weighted_l1_cm"] <
                      0.75*r32["trajectory_path_weighted_l1_cm"] and
                      r64["max_dispersion_error"] <
                      0.75*r32["max_dispersion_error"] and
                      r64["focus_min_cell_radius_cm"] <
                      0.75*r32["focus_min_cell_radius_cm"])

    reference = fields_by_case["r64_gpu1"]
    gpu_comparisons = {}
    for case in ("r64_gpu2", "r64_gpu4"):
        candidate = fields_by_case[case]
        max_difference = 0.0
        same = True
        for name in reference:
            difference = float(np.max(np.abs(candidate[name] - reference[name])))
            max_difference = max(max_difference, difference)
            same = same and np.allclose(candidate[name], reference[name],
                                        rtol=2.0e-11, atol=2.0e-13)
        gpu_comparisons[case] = {"pass": bool(same),
                                 "max_abs_field_difference": max_difference}

    overall = (all(item["physics_ok"] for item in results) and convergence_ok and
               all(item["pass"] for item in gpu_comparisons.values()))
    payload = {"overall": bool(overall), "flash_published_exit": FLASH_PUBLISHED_EXIT,
               "codata_exit": CODATA_EXIT, "convergence_ok": bool(convergence_ok),
               "cases": results, "gpu_comparisons": gpu_comparisons}
    (BENCH_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (BENCH_DIR / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = ["# FLASH quadratic laser-tube benchmark", "",
             f"Overall result: {'PASS' if overall else 'FAIL'}", "",
             "Primary external reference:", "",
             "- FLASH User Guide section 18.4.12, Energy Deposition unit test I:",
             "  https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node122.html",
             "- T. B. Kaiser, Laser ray tracing and power deposition on an unstructured",
             "  three-dimensional grid, Phys. Rev. E 61, 895 (2000):",
             "  https://doi.org/10.1103/PhysRevE.61.895", "",
             "FLASH defines a circular quadratic electron-density tube with nw=nc/2,",
             "radius R=3 cm, wavelength 1 micron, Tw=10 keV, and fixed Coulomb log=1.",
             "The published analytic exit power is 0.7017811 erg/s for a 1 erg/s ray.",
             "FLASH reports non-monotonic finite-resolution power errors for diagonal Ray 5;",
             "this benchmark therefore requires each exit power to be within 0.1 percent",
             "while trajectory, focus radius, and dispersion must converge with resolution.",
             "AthenaK uses the same transverse Hamiltonian and absorption integral; its",
             "dispersive axial group speed moves the focus to x1=2*pi cm.", "",
             "| Case | GPUs | Exit power/ray | Rel. error | Path L1 cm | Dispersion | Transfers | Result |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in results:
        lines.append(f"| {item['case']} | {item['gpus']} | "
                     f"{item['exit_power_per_ray']:.10f} | "
                     f"{item['relative_error_vs_codata']:.3e} | "
                     f"{item['trajectory_path_weighted_l1_cm']:.3e} | "
                     f"{item['max_dispersion_error']:.3e} | "
                     f"{int(item['off_rank_transfers'])} | "
                     f"{'PASS' if item['physics_ok'] else 'FAIL'} |")
    lines.extend(["", f"Resolution convergence: {'PASS' if convergence_ok else 'FAIL'}",
                  "", "GPU decomposition comparisons:", ""])
    for case, comparison in gpu_comparisons.items():
        lines.append(f"- {case}: {'PASS' if comparison['pass'] else 'FAIL'}, "
                     f"max absolute field difference="
                     f"{comparison['max_abs_field_difference']:.16e}")
    lines.extend(["", "Benchmark-driven fixes:", "",
                  "- Added a fixed Coulomb-log option for FLASH's unit-test convention.",
                  "- Added density-power initial electron-temperature profiles.",
                  "- Added second-order density/force reconstruction at the ray position.",
                  "  This fixed zero-length stalls for rays launched on symmetry cell planes.",
                  "- The pre-fix failure is preserved under attempt1.", "",
                  "Scope note:", "",
                  "The production FLASH LaserSlab example also uses cylindrical geometry,",
                  "two materials, tabulated multitemperature EOS, tabulated opacities, and",
                  "six-group radiation diffusion. AthenaK does not yet reproduce that full",
                  "configuration. This analytic FLASH unit test isolates the common laser",
                  "refraction and inverse-bremsstrahlung deposition physics with a published",
                  "numerical target.", ""])
    (BENCH_DIR / "diagnostics.md").write_text("\n".join(lines))
    print(f"Overall result: {'PASS' if overall else 'FAIL'}")
    for item in results:
        print(f"{item['case']:10s} exit={item['exit_power_per_ray']:.10f} "
              f"relerr={item['relative_error_vs_codata']:.3e} "
              f"result={'PASS' if item['physics_ok'] else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
