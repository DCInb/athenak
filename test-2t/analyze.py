#!/usr/bin/env python3
"""Analyze correctness and 1/2/4-GPU consistency for the 2T test matrix."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
REPO = TEST_DIR.parent
sys.path.insert(0, str(REPO / "vis" / "python"))
import athena_read  # noqa: E402

athena_read.check_nan_flag = True


def load_tab(path: Path) -> dict[str, np.ndarray]:
    data = athena_read.tab(str(path))
    return {key: np.atleast_1d(value) for key, value in data.items()}


def sorted_tab(path: Path) -> dict[str, np.ndarray]:
    data = load_tab(path)
    order = np.argsort(data["x1v"])
    return {key: value[order] if value.size == order.size else value
            for key, value in data.items()}


def finite(data: dict[str, np.ndarray], fields: tuple[str, ...]) -> bool:
    return all(np.all(np.isfinite(data[field])) for field in fields)


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def compare_fields(candidate: dict[str, np.ndarray],
                   reference: dict[str, np.ndarray],
                   fields: tuple[str, ...],
                   rtol: float = 2.0e-11,
                   atol: float = 2.0e-12) -> tuple[bool, float]:
    errors = []
    passed = True
    for field in fields:
        if candidate[field].shape != reference[field].shape:
            return False, math.inf
        scale = np.maximum(np.abs(reference[field]), atol)
        error = float(np.max(np.abs(candidate[field] - reference[field]) / scale))
        errors.append(error)
        passed = passed and np.allclose(candidate[field], reference[field],
                                        rtol=rtol, atol=atol)
    return passed, max(errors, default=0.0)


def check_two_temperature(run_dir: Path):
    final = sorted_tab(run_dir / "tab" / "two_temperature.hydro_2t.00001.tab")
    delta_t0 = 1.0 / 0.55 - 0.1 / 0.55
    delta_t = delta_t0 * np.exp(-2.0 * 0.1 / 0.2)
    tion_exact = 1.0 + 0.5 * delta_t
    tele_exact = 1.0 - 0.5 * delta_t
    tion_error = max_abs(final["tion"] - tion_exact)
    tele_error = max_abs(final["tele"] - tele_exact)
    energy_error = max_abs(final["eion"] + final["eele"] - 1.5)
    passed = (finite(final, ("eion", "eele", "tion", "tele")) and
              tion_error <= 2.0e-11 and tele_error <= 2.0e-11 and
              energy_error <= 2.0e-11 and np.all(final["tion"] > 0.0) and
              np.all(final["tele"] > 0.0))
    metrics = {"tion_max_abs_error": tion_error,
               "tele_max_abs_error": tele_error,
               "material_energy_max_abs_error": energy_error}
    return passed, metrics, final, ("eion", "eele", "tion", "tele")


def check_biermann(run_dir: Path):
    field = sorted_tab(run_dir / "tab" /
                       "two_temperature_biermann.biermann.00001.tab")
    two_temp = sorted_tab(run_dir / "tab" /
                          "two_temperature_biermann.two_temperature.00001.tab")
    history = athena_read.hst(str(run_dir / "two_temperature_biermann.mhd.hst"))
    x1 = field["x1v"]
    wave_number = 2.0 * np.pi
    y_slice = 0.0078125
    density = np.exp(0.2 * np.sin(wave_number * y_slice))
    pressure = np.exp(0.2 * np.sin(wave_number * x1))
    exact_rate = (0.1 * (pressure / density) * 0.2 * 0.2 * wave_number**2
                  * np.cos(wave_number * x1) * np.cos(wave_number * y_slice))
    numerical_rate = field["bcc3"] / 1.0e-4
    relative_l2 = float(np.linalg.norm(numerical_rate - exact_rate) /
                        np.linalg.norm(exact_rate))
    b1_max = max_abs(field["bcc1"])
    b2_max = max_abs(field["bcc2"])
    total_energy = np.atleast_1d(history["tot-E"])
    energy_drift = max_abs(total_energy - total_energy[0])
    positive = all(np.all(two_temp[name] > 0.0)
                   for name in ("eion", "eele", "tion", "tele"))
    passed = (finite(field, ("bcc1", "bcc2", "bcc3")) and
              finite(two_temp, ("eion", "eele", "tion", "tele")) and
              relative_l2 < 1.5e-2 and b1_max < 2.0e-12 and
              b2_max < 2.0e-12 and energy_drift < 2.0e-12 and positive)
    metrics = {"b3_rate_relative_l2": relative_l2,
               "max_abs_bcc1": b1_max, "max_abs_bcc2": b2_max,
               "total_energy_max_abs_drift": energy_drift,
               "max_abs_bcc3": max_abs(field["bcc3"])}
    combined = dict(field)
    for name in ("eion", "eele", "tion", "tele"):
        combined[name] = two_temp[name]
    fields = ("bcc1", "bcc2", "bcc3", "eion", "eele", "tion", "tele")
    return passed, metrics, combined, fields


def check_dual_energy(run_dir: Path):
    initial = sorted_tab(run_dir / "tab" /
                         "two_temperature_dual_energy.mhd_2t.00000.tab")
    final = sorted_tab(run_dir / "tab" /
                       "two_temperature_dual_energy.mhd_2t.00001.tab")
    initial_sum_error = max_abs(initial["eion"] + initial["eele"] - 1.5)
    final_sum_error = max_abs(final["eion"] + final["eele"] - 1.5)
    eion_change = max_abs(final["eion"] - initial["eion"])
    eele_change = max_abs(final["eele"] - initial["eele"])
    passed = (finite(final, ("eion", "eele", "tion", "tele")) and
              initial_sum_error < 2.0e-11 and final_sum_error < 2.0e-11 and
              eion_change < 2.0e-11 and eele_change < 2.0e-11 and
              np.all(final["tion"] > 0.0) and np.all(final["tele"] > 0.0))
    metrics = {"initial_material_energy_max_abs_error": initial_sum_error,
               "final_material_energy_max_abs_error": final_sum_error,
               "eion_max_abs_change": eion_change,
               "eele_max_abs_change": eele_change}
    return passed, metrics, final, ("eion", "eele", "tion", "tele")


def check_radiation(run_dir: Path):
    initial = sorted_tab(run_dir / "tab" /
                         "two_temperature_mgfld.hydro_3t.00000.tab")
    final = sorted_tab(run_dir / "tab" /
                       "two_temperature_mgfld.hydro_3t.00001.tab")
    energy_initial = initial["eion"] + initial["eele"] + initial["erad"]
    energy_final = final["eion"] + final["eele"] + final["erad"]
    energy_error = max_abs(energy_final - energy_initial)
    group_error = max_abs(final["erad"] - final["erad00"] -
                          final["erad01"] - final["erad02"])
    tele_drop = float(np.min(initial["tele"] - final["tele"]))
    erad_gain = float(np.min(final["erad"] - initial["erad"]))
    nonnegative = all(np.all(final[name] >= 0.0)
                      for name in ("erad", "erad00", "erad01", "erad02"))
    fields = ("eion", "eele", "tion", "tele", "erad", "trad",
              "erad00", "erad01", "erad02")
    passed = (finite(final, fields) and energy_error < 2.0e-11 and
              group_error < 2.0e-11 and tele_drop > 0.0 and erad_gain > 0.0 and
              nonnegative)
    metrics = {"total_specific_energy_max_abs_drift": energy_error,
               "radiation_group_sum_max_abs_error": group_error,
               "minimum_electron_temperature_drop": tele_drop,
               "minimum_radiation_energy_gain": erad_gain}
    return passed, metrics, final, fields


def check_laser(run_dir: Path):
    laser = sorted_tab(run_dir / "tab" /
                       "two_temperature_laser.laser.00001.tab")
    initial = sorted_tab(run_dir / "tab" /
                         "two_temperature_laser.two_temperature.00000.tab")
    final = sorted_tab(run_dir / "tab" /
                       "two_temperature_laser.two_temperature.00001.tab")
    dx = float(laser["x1v"][1] - laser["x1v"][0])
    left_edge = laser["x1v"] - 0.5 * dx
    expected_q = np.exp(-2.0 * left_edge) * -np.expm1(-2.0 * dx) / dx
    q_error = max_abs(laser["laser_q"] - expected_q)
    cumulative_error = max_abs(laser["laser_energy"] - expected_q * 1.0e-6)
    deposited_power = float(np.sum(laser["laser_q"]) * dx)
    deposited_error = abs(deposited_power - (-np.expm1(-2.0)))
    electron_error = max_abs(final["eele"] - initial["eele"] -
                             laser["laser_energy"])
    ion_error = max_abs(final["eion"] - initial["eion"])
    count_error = max_abs(laser["laser_ray_count"] - 13.0)
    tau_error = max_abs(laser["laser_tau"] - 26.0 * dx)
    path_error = max_abs(laser["laser_path"] - 13.0 * dx)
    fields = ("laser_q", "laser_energy", "laser_ray_count", "laser_tau",
              "laser_path")
    passed = (finite(laser, fields) and
              finite(final, ("eion", "eele", "tion", "tele")) and
              q_error < 2.0e-11 and cumulative_error < 2.0e-11 and
              deposited_error < 2.0e-11 and electron_error < 2.0e-11 and
              ion_error < 2.0e-11 and count_error < 2.0e-12 and
              tau_error < 2.0e-12 and path_error < 2.0e-12 and
              np.all(final["tion"] > 0.0) and np.all(final["tele"] > 0.0))
    metrics = {"laser_q_max_abs_error": q_error,
               "cumulative_energy_max_abs_error": cumulative_error,
               "deposited_power_abs_error": deposited_error,
               "electron_deposition_max_abs_error": electron_error,
               "ion_energy_max_abs_change": ion_error,
               "ray_count_max_abs_error": count_error,
               "optical_depth_max_abs_error": tau_error,
               "path_length_max_abs_error": path_error}
    combined = dict(laser)
    for name in ("eion", "eele", "tion", "tele"):
        combined[name] = final[name]
    compare = fields + ("eion", "eele", "tion", "tele")
    return passed, metrics, combined, compare


CHECKS = {
    "two_temperature": check_two_temperature,
    "biermann_battery": check_biermann,
    "dual_energy": check_dual_energy,
    "thermal_radiation": check_radiation,
    "laser": check_laser,
}


def main() -> int:
    status_rows = list(csv.DictReader((TEST_DIR / "run_status.csv").open()))
    status_by_case = {(row["feature"], int(row["gpus"])): row
                      for row in status_rows}
    results = []
    references: dict[str, tuple[dict[str, np.ndarray], tuple[str, ...]]] = {}

    for feature, check in CHECKS.items():
        for gpus in (1, 2, 4):
            status = status_by_case[(feature, gpus)]
            run_ok = int(status["exit_code"]) == 0
            run_dir = Path(status["run_dir"])
            metrics = {}
            physics_ok = False
            consistency_ok = gpus == 1
            consistency_error = 0.0
            error = ""
            data = None
            fields: tuple[str, ...] = ()
            if run_ok:
                try:
                    physics_ok, metrics, data, fields = check(run_dir)
                    if gpus == 1:
                        references[feature] = (data, fields)
                    else:
                        reference, reference_fields = references[feature]
                        consistency_ok, consistency_error = compare_fields(
                            data, reference, reference_fields)
                except Exception as exception:
                    error = str(exception)
            physics_ok = bool(physics_ok)
            consistency_ok = bool(consistency_ok)
            overall = bool(run_ok and physics_ok and consistency_ok)
            results.append({
                "feature": feature,
                "gpus": gpus,
                "visible_devices": status["visible_devices"],
                "run_ok": run_ok,
                "physics_ok": physics_ok,
                "gpu_consistency_ok": consistency_ok,
                "overall": overall,
                "elapsed_seconds": float(status["elapsed_seconds"]),
                "consistency_max_scaled_error": consistency_error,
                "metrics": metrics,
                "error": error,
            })

    serializable = []
    for result in results:
        copy = dict(result)
        copy["metrics"] = {name: float(value)
                           for name, value in result["metrics"].items()}
        serializable.append(copy)
    (TEST_DIR / "results.json").write_text(json.dumps(serializable, indent=2) + "\n")

    with (TEST_DIR / "results.csv").open("w", newline="") as output:
        fieldnames = ("feature", "gpus", "visible_devices", "run_ok", "physics_ok",
                      "gpu_consistency_ok", "overall", "elapsed_seconds",
                      "consistency_max_scaled_error", "metrics", "error")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in serializable:
            row = dict(result)
            row["metrics"] = json.dumps(row["metrics"], sort_keys=True)
            writer.writerow(row)

    overall_pass = all(result["overall"] for result in results)
    lines = [
        "# AthenaK 2T GPU feature diagnostics",
        "",
        f"Overall result: {'PASS' if overall_pass else 'FAIL'}",
        "",
        "The matrix ran sequentially. GPU count equals MPI rank count; physical GPUs "
        "2, 3, 4, and 5 were selected to avoid the lightly occupied GPUs 0 and 1.",
        "",
        "| Feature | GPUs | Run | Physics | vs 1 GPU | Wall s | Result |",
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['feature']} | {result['gpus']} | "
            f"{'PASS' if result['run_ok'] else 'FAIL'} | "
            f"{'PASS' if result['physics_ok'] else 'FAIL'} | "
            f"{'PASS' if result['gpu_consistency_ok'] else 'FAIL'} | "
            f"{result['elapsed_seconds']:.3f} | "
            f"{'PASS' if result['overall'] else 'FAIL'} |")
    lines.extend(["", "## Feature criteria", "",
                  "- two_temperature: exact ion/electron relaxation and material-energy conservation.",
                  "- biermann_battery: analytic early-time B3 rate, vanishing B1/B2, positivity, and total-energy conservation.",
                  "- dual_energy: retention of gas internal energy under a 1e9 magnetic field with static refinement.",
                  "- thermal_radiation: matter-radiation conservation, group-sum closure, positive groups, and electron-to-radiation transfer.",
                  "- laser: analytic exponential attenuation, ray path/count/optical depth, cumulative deposition, and electron-only heating.",
                  "", "## Detailed metrics", ""])
    for result in results:
        lines.append(f"### {result['feature']} / {result['gpus']} GPU")
        lines.append("")
        if result["error"]:
            lines.append(f"Analysis error: {result['error']}")
        for name, value in result["metrics"].items():
            lines.append(f"- {name}: {value:.16e}")
        lines.append(f"- consistency_max_scaled_error: "
                     f"{result['consistency_max_scaled_error']:.16e}")
        lines.append("")
    lines.extend(["## Artifacts", "",
                  "- build.log: CUDA/MPI configure and build transcript.",
                  "- environment.txt: git state and GPU inventory.",
                  "- run_status.csv: commands, GPU visibility, exit codes, and wall times.",
                  "- results.csv and results.json: machine-readable diagnostics.",
                  "- logs/: one stdout/stderr log per run.",
                  "- runs/: raw AthenaK outputs for every feature/GPU count.", ""])
    (TEST_DIR / "diagnostics.md").write_text("\n".join(lines))
    print(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")
    for result in results:
        print(f"{result['feature']:20s} gpus={result['gpus']} "
              f"result={'PASS' if result['overall'] else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
