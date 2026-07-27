#!/usr/bin/env python3
"""Verify the CH shell geometry, exact incident pulse, transport, and 10 ns run."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parent
RUN_DIR = CASE_DIR / "run"
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def input_value(name: str) -> float:
    text = (CASE_DIR / "laser_shell.athinput").read_text()
    matches = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*({FLOAT})", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one numeric {name}, found {len(matches)}")
    return float(matches[0])


def input_string(name: str) -> str:
    text = (CASE_DIR / "laser_shell.athinput").read_text()
    matches = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*([^\s#]+)", text)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


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


def assemble(path: Path) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
    raw = read_binary(str(path))
    nx = (int(raw["Nx1"]), int(raw["Nx2"]), int(raw["Nx3"]))
    mb_nx = (
        int(raw["nx1_out_mb"]),
        int(raw["nx2_out_mb"]),
        int(raw["nx3_out_mb"]),
    )
    fields = {
        name: np.zeros((nx[2], nx[1], nx[0]), dtype=np.float64)
        for name in raw["var_names"]
    }
    for block, logical in enumerate(raw["mb_logical"]):
        i0 = int(logical[0])*mb_nx[0]
        j0 = int(logical[1])*mb_nx[1]
        k0 = int(logical[2])*mb_nx[2]
        for name in raw["var_names"]:
            values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            nz, ny, nx_local = values.shape
            fields[name][k0:k0+nz, j0:j0+ny, i0:i0+nx_local] = values
    extent = (
        float(raw["x1min"]), float(raw["x1max"]),
        float(raw["x2min"]), float(raw["x2max"]),
        float(raw["x3min"]), float(raw["x3max"]),
    )
    return fields, extent


def geometry_metrics() -> dict[str, float]:
    initial = sorted((RUN_DIR / "bin").glob("laser_shell.fluid.*.bin"))[0]
    fields, extent = assemble(initial)
    density = fields["dens"]
    nz, ny, nx = density.shape
    x1 = extent[0] + (np.arange(nx)+0.5)*(extent[1]-extent[0])/nx
    x2 = extent[2] + (np.arange(ny)+0.5)*(extent[3]-extent[2])/ny
    x3 = extent[4] + (np.arange(nz)+0.5)*(extent[5]-extent[4])/nz
    radius = np.sqrt(
        x1[np.newaxis, np.newaxis, :]**2
        + x2[np.newaxis, :, np.newaxis]**2
        + x3[:, np.newaxis, np.newaxis]**2
    )
    axis_cosine = np.divide(
        x1[np.newaxis, np.newaxis, :],
        radius,
        out=np.full_like(radius, -1.0),
        where=radius > 0.0,
    )
    ambient = input_value("ambient_density")
    solid = input_value("solid_density")
    fraction = np.clip((density-ambient)/(solid-ambient), 0.0, 1.0)
    mask = fraction > 0.5
    theta = np.degrees(np.arccos(np.clip(axis_cosine[mask], -1.0, 1.0)))
    cell_volume = (
        (extent[1]-extent[0])/nx
        * (extent[3]-extent[2])/ny
        * (extent[5]-extent[4])/nz
    )
    mass_cgs = input_value("mass_cgs")
    length_cgs = input_value("length_cgs")
    density_scale_cgs = mass_cgs/length_cgs**3
    excess_mass = float(np.sum(density-ambient)*cell_volume*mass_cgs)
    half_angle = input_value("opening_half_angle_deg")
    inner_radius = input_value("inner_radius")
    beam_radius = input_value("beam0_radius")
    projected_inner_radius = inner_radius*math.sin(math.radians(half_angle))
    return {
        "measured_inner_radius_mm": float(np.min(radius[mask])),
        "measured_outer_radius_mm": float(np.max(radius[mask])),
        "measured_shell_thickness_mm": float(
            np.max(radius[mask])-np.min(radius[mask])
        ),
        "measured_peak_density_g_cc": float(np.max(density)*density_scale_cgs),
        "measured_full_opening_angle_deg": 2.0*float(np.max(theta)),
        "initial_excess_ch_mass_mg": 1.0e3*excess_mass,
        "projected_inner_radius_mm": projected_inner_radius,
        "gaussian_radius_mm": beam_radius,
        "projected_area_coverage": min((beam_radius/projected_inner_radius)**2, 1.0),
    }


def main() -> int:
    status = json.loads((CASE_DIR / "run_status.json").read_text())
    phase1_log = CASE_DIR / "logs" / "phase1_laser_on.log"
    phase2_log = CASE_DIR / "logs" / "phase2_laser_off.log"
    phase1_records = parse_laser_log(phase1_log)
    phase2_records = parse_laser_log(phase2_log)
    active = [record for record in phase1_records if record["launched"] > 0.0]

    length_cgs = input_value("length_cgs")
    mass_cgs = input_value("mass_cgs")
    time_cgs = input_value("time_cgs")
    velocity_cgs = length_cgs/time_cgs
    energy_scale_erg = mass_cgs*velocity_cgs**2
    power_scale_erg_s = energy_scale_erg/time_cgs
    beam_power_erg_s = input_value("beam0_power")
    pulse_duration_s = (
        input_value("beam0_end_time")-input_value("beam0_start_time")
    )*time_cgs
    incident_energy_erg = beam_power_erg_s*pulse_duration_s
    expected_code_power = beam_power_erg_s/power_scale_erg_s

    phase1_history = read_history(RUN_DIR / "phase1_laser_shell.user.hst")
    full_history = read_history(RUN_DIR / "laser_shell.user.hst")
    deposited_energy_erg = float(phase1_history["laser_E"][-1])*energy_scale_erg
    material_energy_change_erg = float(
        full_history["mat_E"][-1]-full_history["mat_E"][0]
    )*energy_scale_erg

    metrics = {
        "run": {
            "phase1_exit_code": int(status["phase1_exit_code"]),
            "phase2_exit_code": int(status["phase2_exit_code"]),
            "phase1_final_time_ns": final_time(phase1_log),
            "phase2_final_time_ns": final_time(phase2_log),
            "phase1_elapsed_seconds": float(status["phase1_elapsed_seconds"]),
            "phase2_elapsed_seconds": float(status["phase2_elapsed_seconds"]),
        },
        "laser": {
            "profile": input_string("beam0_profile"),
            "wavelength_um": input_value("beam0_wavelength")*1.0e4,
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
        },
        "geometry": geometry_metrics(),
    }

    geometry = metrics["geometry"]
    laser = metrics["laser"]
    run = metrics["run"]
    checks = {
        "both_phases_completed": (
            run["phase1_exit_code"] == 0 and run["phase2_exit_code"] == 0
        ),
        "stopped_exactly_at_5_and_10_ns": (
            abs(run["phase1_final_time_ns"]-5.0) < 1.0e-12
            and abs(run["phase2_final_time_ns"]-10.0) < 1.0e-12
        ),
        "incident_energy_is_10_kj": abs(laser["incident_energy_kj"]-10.0) < 1.0e-12,
        "spatial_profile_is_gaussian": laser["profile"] == "gaussian",
        "one_omega_wavelength": abs(laser["wavelength_um"]-1.053) < 1.0e-12,
        "fully_ionized_ch_composition": (
            abs(input_value("electron_heat_capacity_fraction")-7.0/9.0) < 1.0e-14
            and abs(input_value("electron_number_per_gram")
                    -3.242691463909014e23)/3.242691463909014e23 < 1.0e-14
            and abs(input_value("beam0_zeff")-37.0/7.0) < 1.0e-14
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
    }
    warnings = []
    if laser["max_unresolved_power_fraction"] >= 5.0e-2:
        warnings.append(
            "The evolving plasma creates multiply reflecting grazing rays; "
            "the eight-turn transport budget leaves a substantial instantaneous "
            "remaining-power fraction. It is conserved in the ray budget but is "
            "neither deposited nor counted as escaped."
        )
    metrics["checks"] = checks
    metrics["warnings"] = warnings
    metrics["pass"] = all(checks.values())
    (CASE_DIR / "results.json").write_text(json.dumps(metrics, indent=2) + "\n")

    lines = [
        "# Laser-shell run diagnostics",
        "",
        f"Overall: **{'PASS' if metrics['pass'] else 'FAIL'}**",
        "",
        "## Requested setup",
        "",
        f"- Incident pulse: {laser['incident_energy_kj']:.9g} kJ, "
        f"{laser['power_tw']:.9g} TW for {laser['duration_ns']:.9g} ns",
        f"- Wavelength: {laser['wavelength_um']:.9g} um (1 omega assumption)",
        f"- Deposited energy: {laser['deposited_energy_kj']:.9g} kJ "
        f"({100.0*laser['deposited_fraction']:.3f}% of incident)",
        f"- Maximum unresolved grazing-ray power: "
        f"{100.0*laser['max_unresolved_power_fraction']:.3f}%",
        f"- Resolved shell radii: {geometry['measured_inner_radius_mm']:.5f} to "
        f"{geometry['measured_outer_radius_mm']:.5f} mm",
        f"- Resolved shell thickness: "
        f"{geometry['measured_shell_thickness_mm']:.5f} mm",
        f"- Initial peak CH density: "
        f"{geometry['measured_peak_density_g_cc']:.6g} g/cc",
        f"- Resolved full opening angle: "
        f"{geometry['measured_full_opening_angle_deg']:.5f} deg",
        f"- Gaussian spot/projected-inner-cap area coverage: "
        f"{100.0*geometry['projected_area_coverage']:.3f}%",
        f"- Initial CH cap mass: {geometry['initial_excess_ch_mass_mg']:.6g} mg",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items()
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (CASE_DIR / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
