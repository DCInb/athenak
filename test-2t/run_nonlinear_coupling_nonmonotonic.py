#!/usr/bin/env python3
"""Run and validate the nonlinear source solver against a non-monotonic residual."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


TEST_DIR = Path(__file__).resolve().parent
INPUT_FILE = TEST_DIR / "nonlinear_coupling_nonmonotonic.athinput"
OPACITY_FILE = TEST_DIR / "nonlinear_coupling_nonmonotonic.opacity"
DEFAULT_BINARY = TEST_DIR / "build" / "src" / "athena"

ARAD = 1.0
LIGHT_SPEED = 1.0
PLANCK_INFINITY = 6.4939394022668291491


@dataclass(frozen=True)
class OpacityTable:
    density: tuple[float, ...]
    temperature: tuple[float, ...]
    group_bound: tuple[float, ...]
    transport: tuple[float, ...]
    absorption: tuple[float, ...]
    emission: tuple[float, ...]


def read_opacity_table(path: Path) -> OpacityTable:
    """Read the fixed native table used by this regression."""
    logical_lines: list[list[str]] = []
    with path.open(encoding="ascii") as stream:
        for raw_line in stream:
            tokens = raw_line.partition("#")[0].split()
            if tokens:
                logical_lines.append(tokens)

    if not logical_lines or logical_lines[0] != ["athenak_opacity_table", "1"]:
        raise AssertionError(f"Unexpected opacity-table header in {path}")
    if len(logical_lines) < 3 or logical_lines[1][0] != "dimensions":
        raise AssertionError(f"Missing opacity-table dimensions in {path}")
    dimensions = tuple(int(value) for value in logical_lines[1][1:])
    if dimensions != (1, 6, 1):
        raise AssertionError(
            f"This regression requires table dimensions (1, 6, 1), got {dimensions}")

    cursor = 2

    def read_section(name: str, count: int) -> tuple[float, ...]:
        nonlocal cursor
        if cursor >= len(logical_lines) or logical_lines[cursor][0] != name:
            raise AssertionError(f"Missing opacity-table section {name}")
        values = [float(value) for value in logical_lines[cursor][1:]]
        cursor += 1
        while len(values) < count and cursor < len(logical_lines):
            values.extend(float(value) for value in logical_lines[cursor])
            cursor += 1
        if len(values) != count:
            raise AssertionError(
                f"Opacity-table section {name} has {len(values)} values, expected {count}")
        return tuple(values)

    density = read_section("density", dimensions[0])
    temperature = read_section("temperature", dimensions[1])
    group_bound = read_section("group_bound", dimensions[2]+1)
    table_size = dimensions[0]*dimensions[1]*dimensions[2]
    transport = read_section("transport", table_size)
    absorption = read_section("absorption", table_size)
    emission = read_section("emission", table_size)
    if cursor >= len(logical_lines) or logical_lines[cursor] != ["end"]:
        raise AssertionError(f"Missing opacity-table end marker in {path}")
    if cursor+1 != len(logical_lines):
        raise AssertionError(f"Unexpected content after opacity-table end marker in {path}")

    for name, axis in (("density", density), ("temperature", temperature),
                       ("group_bound", group_bound)):
        if not all(math.isfinite(value) for value in axis):
            raise AssertionError(f"Opacity-table {name} axis is non-finite")
        if not all(right > left for left, right in zip(axis, axis[1:])):
            raise AssertionError(f"Opacity-table {name} axis is not strictly increasing")
    if density[0] <= 0.0 or temperature[0] <= 0.0 or group_bound[0] < 0.0:
        raise AssertionError("Opacity-table coordinates are outside their physical range")
    for name, values in (("transport", transport), ("absorption", absorption),
                         ("emission", emission)):
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise AssertionError(f"Opacity-table {name} values must be finite and positive")

    return OpacityTable(density, temperature, group_bound,
                        transport, absorption, emission)


def planck_integral(value: float) -> float:
    """Match the cancellation-safe Planck integral used by ThermalRadiation."""
    if value <= 0.0:
        return 0.0
    if value >= 50.0:
        return PLANCK_INFINITY
    if value < 0.5:
        value2 = value*value
        value3 = value2*value
        return (value3/3.0-value3*value/8.0+value3*value2/60.0
                -value3*value2*value2/5040.0
                +value3*value2*value2*value2/272160.0
                -value3*value2*value2*value2*value2/13305600.0)
    tail = 0.0
    for index in range(1, 65):
        inverse = 1.0/index
        inverse2 = inverse*inverse
        tail += math.exp(-index*value)*(
            value**3*inverse+3.0*value**2*inverse2
            + 6.0*value*inverse2*inverse+6.0*inverse2*inverse2)
    return min(max(PLANCK_INFINITY-tail, 0.0), PLANCK_INFINITY)


def group_fraction(table: OpacityTable, temperature: float) -> float:
    if temperature <= 0.0:
        return 0.0
    return ((planck_integral(table.group_bound[1]/temperature)
             - planck_integral(table.group_bound[0]/temperature))/PLANCK_INFINITY)


def interpolate(axis: tuple[float, ...], values: tuple[float, ...],
                temperature: float) -> float:
    if temperature <= axis[0]:
        return values[0]
    if temperature >= axis[-1]:
        return values[-1]
    for index, upper in enumerate(axis[1:]):
        if temperature <= upper:
            lower = axis[index]
            fraction = (temperature-lower)/(upper-lower)
            return values[index]+fraction*(values[index+1]-values[index])
    raise AssertionError("Temperature interpolation interval was not found")


def tab_data(path: Path) -> tuple[float, dict[str, list[float]]]:
    names: list[str] | None = None
    columns: dict[str, list[float]] = {}
    time = math.nan
    with path.open(encoding="ascii") as stream:
        for line in stream:
            if not math.isfinite(time):
                match = re.search(r"time=([^\s]+)", line)
                if match is not None:
                    time = float(match.group(1))
            if line.startswith("# gid"):
                names = line[1:].split()
                columns = {name: [] for name in names}
            elif names is not None and line.strip() and not line.startswith("#"):
                for name, value in zip(names, line.split()):
                    columns[name].append(float(value))
    if not math.isfinite(time) or not columns:
        raise AssertionError(f"Could not parse formatted-table output {path}")
    return time, columns


def coupled_state(table: OpacityTable, temperature: float, old_group_energy: float,
                  local_energy: float, electron_capacity: float,
                  coupling_depth: float) -> tuple[float, float]:
    absorption = interpolate(table.temperature, table.absorption, temperature)
    emission = interpolate(table.temperature, table.emission, temperature)
    equilibrium = ARAD*temperature**4*group_fraction(table, temperature)
    group_energy = ((old_group_energy+coupling_depth*emission*equilibrium)
                    /(1.0+coupling_depth*absorption))
    residual = electron_capacity*temperature+group_energy-local_energy
    return residual, group_energy


def residual_scan(table: OpacityTable, old_group_energy: float, local_energy: float,
                  electron_capacity: float, coupling_depth: float
                  ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
    upper = local_energy/electron_capacity
    temperatures = [upper*index/4096.0 for index in range(4097)]
    residuals = [coupled_state(
        table, temperature, old_group_energy, local_energy, electron_capacity,
        coupling_depth)[0] for temperature in temperatures]
    root_brackets = []
    for index in range(len(temperatures)-1):
        if residuals[index] == 0.0:
            root_brackets.append((temperatures[index], temperatures[index]))
        elif residuals[index]*residuals[index+1] < 0.0:
            root_brackets.append((temperatures[index], temperatures[index+1]))
    return temperatures, residuals, root_brackets


def bisect_root(table: OpacityTable, lower: float, upper: float,
                old_group_energy: float,
                local_energy: float, electron_capacity: float,
                coupling_depth: float) -> float:
    if lower == upper:
        return lower
    lower_residual = coupled_state(
        table, lower, old_group_energy, local_energy, electron_capacity,
        coupling_depth)[0]
    for _ in range(160):
        trial = 0.5*(lower+upper)
        trial_residual = coupled_state(
            table, trial, old_group_energy, local_energy, electron_capacity,
            coupling_depth)[0]
        if (trial_residual > 0.0) == (lower_residual > 0.0):
            lower = trial
            lower_residual = trial_residual
        else:
            upper = trial
    return 0.5*(lower+upper)


def uniform_value(columns: dict[str, list[float]], name: str,
                  tolerance: float = 2.0e-12) -> float:
    values = columns[name]
    if not values or not all(math.isfinite(value) for value in values):
        raise AssertionError(f"Field {name} is empty or non-finite")
    if max(values)-min(values) > tolerance*max(1.0, abs(values[0])):
        raise AssertionError(f"Uniform source problem produced nonuniform {name}")
    return values[0]


def validate(run_directory: Path, stdout: str,
             table: OpacityTable) -> dict[str, float]:
    basename = "nonlinear_coupling_nonmonotonic.hydro_3t"
    initial_time, initial = tab_data(
        run_directory / "tab" / f"{basename}.00000.tab")
    final_time, final = tab_data(
        run_directory / "tab" / f"{basename}.00001.tab")
    density_time, density_output = tab_data(
        run_directory / "tab"
        / "nonlinear_coupling_nonmonotonic.density.00000.tab")

    source_dt = final_time-initial_time
    if abs(density_time-initial_time) > 1.0e-14:
        raise AssertionError("Density and three-temperature outputs are out of sync")
    density = uniform_value(density_output, "dens")
    if abs(density-table.density[0]) > 1.0e-14:
        raise AssertionError("Problem density is not on the opacity-table density axis")
    old_temperature = uniform_value(initial, "tele")
    old_electron_energy = density*uniform_value(initial, "eele")
    old_group_energy = density*uniform_value(initial, "erad00")
    local_energy = old_electron_energy+old_group_energy
    electron_capacity = old_electron_energy/old_temperature
    coupling_depth = source_dt*LIGHT_SPEED*density

    temperatures, residuals, root_brackets = residual_scan(
        table, old_group_energy, local_energy, electron_capacity, coupling_depth)
    differences = [right-left for left, right in zip(residuals, residuals[1:])]
    slope_signs = [1 if difference > 0.0 else -1
                   for difference in differences if difference != 0.0]
    turns = sum(left != right for left, right in zip(slope_signs, slope_signs[1:]))
    if min(differences) >= -1.0e-6 or max(differences) <= 1.0e-6 or turns < 2:
        raise AssertionError("Constructed opacity table did not make F(Te) non-monotonic")
    residual_at_peak = coupled_state(
        table, 0.5, old_group_energy, local_energy, electron_capacity,
        coupling_depth)[0]
    residual_after_peak = coupled_state(
        table, 0.6, old_group_energy, local_energy, electron_capacity,
        coupling_depth)[0]
    if residual_at_peak-residual_after_peak <= 0.1:
        raise AssertionError("Opacity peak did not produce the intended residual decrease")
    if len(root_brackets) != 1:
        raise AssertionError(
            f"Expected one admissible temperature root, found {len(root_brackets)}")

    expected_temperature = bisect_root(
        table, *root_brackets[0], old_group_energy, local_energy,
        electron_capacity, coupling_depth)
    expected_residual, expected_group = coupled_state(
        table, expected_temperature, old_group_energy, local_energy,
        electron_capacity, coupling_depth)
    expected_electron = electron_capacity*expected_temperature
    final_temperature = uniform_value(final, "tele")
    final_electron_energy = density*uniform_value(final, "eele")
    final_group_energy = density*uniform_value(final, "erad00")
    final_total_group = density*uniform_value(final, "erad")

    state_tolerance = 3.0e-9
    if abs(final_temperature-expected_temperature) > state_tolerance:
        raise AssertionError(
            f"Te={final_temperature:.16e}, expected {expected_temperature:.16e}")
    if abs(final_group_energy-expected_group) > state_tolerance:
        raise AssertionError(
            f"E0={final_group_energy:.16e}, expected {expected_group:.16e}")
    if abs(final_electron_energy-expected_electron) > state_tolerance:
        raise AssertionError(
            f"Ee={final_electron_energy:.16e}, expected {expected_electron:.16e}")
    if abs(final_group_energy-final_total_group) > 2.0e-12:
        raise AssertionError("One-group radiation total does not match group zero")
    if final_electron_energy < 0.0 or final_group_energy < 0.0:
        raise AssertionError("Nonlinear source solve produced a negative energy")

    initial_total = (uniform_value(initial, "eion")+uniform_value(initial, "eele")
                     + uniform_value(initial, "erad"))
    final_total = (uniform_value(final, "eion")+uniform_value(final, "eele")
                   + uniform_value(final, "erad"))
    conservation_error = abs(final_total-initial_total)
    if conservation_error > 3.0e-12:
        raise AssertionError(
            f"Matter+radiation energy drifted by {conservation_error:.3e}")

    report = re.search(
        r"nonlinear thermal radiation source: max_iterations=(\d+) "
        r"fallback_cells=(\d+) max_relative_residual=([^\s]+)", stdout)
    if report is None:
        raise AssertionError("Nonlinear source convergence report was not emitted")
    iterations = int(report.group(1))
    fallback_cells = int(report.group(2))
    reported_residual = float(report.group(3))
    if fallback_cells != 0 or iterations <= 0 or iterations > 80:
        raise AssertionError(
            f"iterations={iterations}, fallback_cells={fallback_cells}")
    if not math.isfinite(reported_residual) or reported_residual > 2.0e-9:
        raise AssertionError(
            f"Reported nonlinear residual {reported_residual:.3e} is too large")

    final_residual = coupled_state(
        table, final_temperature, old_group_energy, local_energy,
        electron_capacity, coupling_depth)[0]
    return {
        "source_dt": source_dt,
        "expected_temperature": expected_temperature,
        "final_temperature": final_temperature,
        "root_residual": abs(expected_residual),
        "final_relative_residual": abs(final_residual)/local_energy,
        "conservation_error": conservation_error,
        "turning_points": float(turns),
        "iterations": float(iterations),
    }


def run(binary: Path) -> dict[str, float]:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"AthenaK binary is missing or not executable: {binary}")
    table = read_opacity_table(OPACITY_FILE)
    with tempfile.TemporaryDirectory(prefix="athenak-nonlinear-opacity-") as temporary:
        run_directory = Path(temporary)
        command = [
            str(binary), "-i", str(INPUT_FILE),
            f"thermal_radiation/opacity_table_file={OPACITY_FILE}",
        ]
        completed = subprocess.run(
            command, cwd=run_directory, text=True, capture_output=True,
            timeout=60.0, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"AthenaK exited with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        return validate(run_directory, completed.stdout, table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path,
        default=Path(os.environ.get("ATHENAK_BINARY", DEFAULT_BINARY)),
        help="AthenaK executable (default: test-2t/build/src/athena)")
    arguments = parser.parse_args()
    metrics = run(arguments.binary.resolve())
    print("non-monotonic nonlinear radiation coupling: PASS")
    for name, value in metrics.items():
        print(f"{name}={value:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
