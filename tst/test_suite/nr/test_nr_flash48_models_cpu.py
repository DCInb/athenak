"""Physical reference checks for the second FLASH 4.8 comparison pass."""

import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import athena_read
from bin_convert import read_binary


REPO = Path(__file__).resolve().parents[3]
ME = 9.1093837015e-28
QE = 4.803204712570263e-10
AMU = 1.660538921e-24
KB = 1.3806488e-16
HBAR = 1.054571817e-27


def run_deck(directory, source, updates, remove=(), success=True, restart=None):
    """Isolated run; environment overrides allow the same checks on MPI builds."""
    directory.mkdir()
    text = source.read_text() if source is not None else ""
    for name in remove:
        text = re.sub(rf"(?m)^\s*{name}\s*=.*$", "", text)
    for block, parameters in updates.items():
        text += f"\n<{block}>\n"
        text += "".join(f"{key} = {value}\n" for key, value in parameters.items())
    deck = directory / "test.athinput"
    deck.write_text(text)
    executable = Path(os.environ.get("ATHENA_FLASH48_TEST_EXE", "./athena")).resolve()
    command = [str(executable), "-i", str(deck)]
    if restart is not None:
        command += ["-r", str(restart)]
    ranks = os.environ.get("ATHENA_FLASH48_TEST_RANKS")
    if ranks:
        command = ["mpirun", "-np", ranks, *command]
    result = subprocess.run(command, cwd=directory, capture_output=True,
                            text=True, timeout=120, check=False)
    log = result.stdout + result.stderr
    (directory / "run.log").write_text(log)
    assert (result.returncode == 0) == success, log
    return log


def coulomb_log(ne, ti, te, zbar, model, floor):
    """Independent direct impact-parameter calculation in cgs."""
    bmax = math.sqrt(KB * te / (4 * math.pi * QE**2 * ne))
    bmin = max(zbar * QE**2 / (3 * KB * te),
               HBAR / (2 * math.sqrt(3 * KB * te * ME)))
    if model == "lee_more":
        bmax /= math.sqrt(1 + zbar * te / ti)
        return max(0.5 * math.log1p((bmax / bmin)**2), floor)
    return max(math.log1p(bmax / bmin), floor)


def write_ideal_table(path):
    """A partially ionized table deliberately differing from input full ionization."""
    densities = (0.1, 10.0)
    temperatures = (1e4, 1e7, 1e10)
    abar, zbar = 6.5, 0.7
    text = ("athenak_two_temperature_eos 1\ndimensions 2 3\nabar 6.5\n"
            "density\n0.1 10\ntemperature\n1e4 1e7 1e10\n")
    for field in ("ion_pressure", "electron_pressure",
                  "ion_specific_internal_energy", "electron_specific_internal_energy",
                  "mean_ionization"):
        text += field + "\n"
        for rho in densities:
            row = []
            for temperature in temperatures:
                value = KB * temperature / (abar * AMU)
                if field == "mean_ionization":
                    value = zbar
                else:
                    value *= rho if "pressure" in field else 1.5
                    if field.startswith("electron"):
                        value *= zbar
                row.append(f"{value:.17e}")
            text += " ".join(row) + "\n"
    path.write_text(text + "end\n")


@pytest.mark.parametrize("model", ("spitzer", "lee_more"))
@pytest.mark.parametrize("tabular,density_scale,floor", (
    (False, 1.0, None), (False, 1e4, None), (False, 1.0, 20.0),
    (True, 1.0, None)))
def test_state_dependent_exchange(tmp_path, model, tabular, density_scale, floor):
    dt = 1e-3
    multiplier = density_scale
    updates = {
        "job": {"basename": "exchange"},
        "meshblock": {"nx1": 8},
        "units": {"mass_cgs": 1e-3 * density_scale},
        "time": {"nlim": 1, "tlim": dt, "initial_dt": dt},
        "mhd": {"t_ei_model": "spitzer", "t_ei_coulomb_log_model": model,
                "t_ei_spitzer_multiplier": multiplier},
        "output1": {"dt": dt, "data_format": "%24.17e"},
        "output2": {"dt": -1},
    }
    if floor is not None:
        updates["mhd"]["t_ei_coulomb_log_floor"] = floor
    if tabular:
        table = tmp_path / "partial.2t_eos"
        write_ideal_table(table)
        updates["mhd"]["rsolver"] = "llf"
        updates["problem"] = {"yl": 1, "yr": 1}
        updates["materials"] = {
            "nmaterials": 1, "material0_eos_table_file": table,
            "eos_table_bounds": "clamp", "eos_table_interpolation": "geometric",
            "eos_table_density_to_cgs": 1,
            "eos_table_temperature_to_kelvin": 1e8,
            "eos_table_pressure_from_cgs": 1e-16,
            "eos_table_specific_energy_from_cgs": 1e-16,
        }
    directory = tmp_path / "run"
    run_deck(directory, REPO / "inputs/mhd/two_material_relax.athinput", updates)
    tables = sorted((directory / "tab").glob("exchange.mhd_2t.*.tab"))
    assert len(tables) == 2
    initial, final = (athena_read.tab(str(path)) for path in tables)
    for left in (True, False):
        mask = (initial["x1v"] < 0) == left
        if tabular:
            abar, zbar, zeff, kelvin_per_code = 6.5, 0.7, (37/7)/3.5*0.7, 1e8
        else:
            abar, zbar, zeff = (6.5, 3.5, 37/7) if left else (2.5, 1.0, 1.0)
            kelvin_per_code = 1e16 * abar / (1 + zbar) * AMU / KB
        tion = initial["tion"][mask][0]
        tele = initial["tele"][mask][0]
        ti, te = tion * kelvin_per_code, tele * kelvin_per_code
        ne = density_scale * zbar / (abar * AMU)
        log_floor = floor if floor is not None else (2 if model == "lee_more" else 1)
        log_lambda = coulomb_log(ne, ti, te, zbar, model, log_floor)
        ion_mass = abar * AMU
        tau = (3 * ME * ion_mass * (KB * te / ME + KB * ti / ion_mass)**1.5
               / (8 * math.sqrt(2 * math.pi) * ne * zeff * QE**4 * log_lambda))
        tau *= multiplier / 1e-9
        fe = zbar / (1 + zbar)
        equilibrium = (1 - fe) * tion + fe * tele
        difference = (tion - tele) * math.exp(-(1 + zbar) * dt / tau)
        np.testing.assert_allclose(final["tion"][mask], equilibrium + fe * difference,
                                   rtol=2e-8, atol=1e-11)
        np.testing.assert_allclose(final["tele"][mask], equilibrium - (1-fe) * difference,
                                   rtol=2e-8, atol=1e-11)
        assert np.max(np.abs(final["tele"][mask] - tele)) > 1e-4
    np.testing.assert_allclose(final["eion"] + final["eele"],
                               initial["eion"] + initial["eele"], rtol=2e-12)


def planck_energy(lower, upper, temperature):
    """Gauss-Legendre quadrature, independent of the production Planck series."""
    if temperature == 0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(160)
    lo, hi = lower / temperature, min(upper / temperature, 100.0)
    x = lo + (nodes + 1) * (hi - lo) / 2
    return (temperature**4 * 15 / math.pi**4 * (hi-lo)/2
            * np.dot(weights, x**3 / np.expm1(x)))


def thermal_updates():
    return {
        "job": {"basename": "bath"},
        "mesh": {"ix1_bc": "outflow", "ox1_bc": "outflow"},
        "time": {"nlim": 1, "tlim": 1e-4, "initial_dt": 1e-4},
        "thermal_radiation": {
            "implicit_x1_inner_boundary": "dirichlet",
            "implicit_x1_outer_boundary": "dirichlet",
            "implicit_x1_inner_temperature": 1.0,
            "implicit_x1_outer_temperature": 0.3,
            "initial_profile": "uniform", "initial_radiation_temperature": 0.5,
            "implicit_amr_order": 1, "implicit_tolerance": 1e-12,
        },
    }


THERMAL_VALUES = ("implicit_x1_inner_value", "implicit_x1_outer_value")


@pytest.mark.parametrize("refined", (False, True))
@pytest.mark.parametrize("upper,outer_temperature",
                         ((2.0, 0.3), (100.0, 0.3), (100.0, 0.0)))
def test_radiation_temperature_boundary(tmp_path, refined, upper, outer_temperature):
    updates = thermal_updates()
    if not refined:
        updates["mesh_refinement"] = {"refinement": "none"}
    updates["thermal_radiation"]["group_bound_2"] = upper
    updates["thermal_radiation"]["implicit_x1_outer_temperature"] = outer_temperature
    directory = tmp_path / "run"
    run_deck(directory, REPO / "inputs/tests/thermal_radiation_amr.athinput",
             updates, remove=THERMAL_VALUES)
    outputs = [read_binary(str(path)) for path in
               sorted((directory / "bin").glob("*.bin"))]
    initial, final = outputs[0], outputs[-1]
    assert final["cycle"] == 1
    centers, widths = [], []
    for geom, block in zip(initial["mb_geometry"], initial["mb_data"]["erad00_d"]):
        count = np.asarray(block).size
        dx = (geom[1] - geom[0]) / count
        centers.extend(geom[0] + (np.arange(count) + 0.5) * dx)
        widths.extend([dx] * count)
    order = np.argsort(centers)
    widths = np.asarray(widths)[order]
    count = len(order)
    dt = final["time"] - initial["time"]
    for g, opacity in enumerate((100.0, 35.0)):
        group = f"erad{g:02d}_d"
        lower, high = ((0.0, 1.0), (1.0, upper))[g]
        bath = [planck_energy(lower, high, temperature)
                for temperature in (1.0, outer_temperature)]
        expected_initial = planck_energy(lower, high, 0.5)
        before = np.asarray(initial["mb_data"][group]).ravel()[order]
        np.testing.assert_allclose(before, expected_initial, rtol=2e-12)
        matrix = np.eye(count)
        rhs = before.copy()
        diffusivity = 1000 / (3 * opacity)
        for cell in range(count):
            for neighbor in (cell-1, cell+1):
                if neighbor in (-1, count):
                    coefficient = 2 * dt * diffusivity / widths[cell]**2
                    matrix[cell, cell] += coefficient
                    rhs[cell] += coefficient * bath[int(neighbor == count)]
                else:
                    coefficient = (dt * diffusivity / widths[cell]
                                   / (0.5 * (widths[cell] + widths[neighbor])))
                    matrix[cell, cell] += coefficient
                    matrix[cell, neighbor] -= coefficient
        expected = np.linalg.solve(matrix, rhs)
        after = np.asarray(final["mb_data"][group]).ravel()[order]
        np.testing.assert_allclose(after, expected, rtol=3e-9, atol=3e-12)
        assert np.all(after >= 0)
        # Boundary flux balances the volume-integrated change, group by group.
        heat = dt * 2 * diffusivity * (
            (bath[0]-after[0])/widths[0] + (bath[1]-after[-1])/widths[-1])
        np.testing.assert_allclose(np.dot(widths, after-before), heat,
                                   rtol=2e-9, atol=2e-12)


@pytest.mark.parametrize("axis", (1, 2, 3))
def test_radiation_bath_equilibrium(tmp_path, axis):
    updates = thermal_updates()
    updates["thermal_radiation"].update({
        "implicit_x1_outer_temperature": 1.0,
        "initial_radiation_temperature": 1.0})
    remove = list(THERMAL_VALUES)
    if axis > 1:
        updates["mesh"].update({f"nx{axis}": 8, f"ix{axis}_bc": "outflow",
                                f"ox{axis}_bc": "outflow"})
        updates["meshblock"] = {f"nx{axis}": 8}
        if axis == 3:
            updates["mesh"]["nx2"] = 8
            updates["meshblock"]["nx2"] = 8
        for end in ("inner", "outer"):
            prefix = f"implicit_x{axis}_{end}"
            updates["thermal_radiation"].update({
                prefix + "_boundary": "dirichlet", prefix + "_temperature": 1.0})
            remove.append(prefix + "_value")
    directory = tmp_path / "run"
    run_deck(directory, REPO / "inputs/tests/thermal_radiation_amr.athinput",
             updates, remove=remove)
    outputs = [read_binary(str(path)) for path in
               sorted((directory / "bin").glob("*.bin"))]
    initial, final = outputs[0], outputs[-1]
    for group in ("erad00_d", "erad01_d"):
        np.testing.assert_allclose(final["mb_data"][group],
                                   initial["mb_data"][group], rtol=3e-11, atol=1e-13)


def test_radiation_bath_restart(tmp_path):
    updates = thermal_updates()
    updates["output2"] = {"file_type": "rst", "dt": -1, "dcycle": 1}
    checkpoint = tmp_path / "checkpoint"
    source = REPO / "inputs/tests/thermal_radiation_amr.athinput"
    run_deck(checkpoint, source, updates, remove=THERMAL_VALUES)
    restart = sorted((checkpoint / "rst").glob("*.rst"))[-1]
    continuation = {"time": {"nlim": 2, "tlim": 2e-4}}
    resumed = tmp_path / "resumed"
    run_deck(resumed, None, continuation, restart=restart)
    updates["time"].update(continuation["time"])
    uninterrupted = tmp_path / "uninterrupted"
    run_deck(uninterrupted, source, updates, remove=THERMAL_VALUES)
    results = [read_binary(str(sorted((path / "bin").glob("*.bin"))[-1]))
               for path in (uninterrupted, resumed)]
    assert results[0]["cycle"] == results[1]["cycle"] == 2
    for group in ("erad00_d", "erad01_d"):
        np.testing.assert_allclose(results[0]["mb_data"][group],
                                   results[1]["mb_data"][group], rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("model,reflection,absorption", (
    ("straight", "true", "constant"),
    ("straight", "true", "inverse_bremsstrahlung"),
    ("straight", "false", "inverse_bremsstrahlung"),
    ("refractive", "false", "inverse_bremsstrahlung")))
def test_laser_overdense_launch(tmp_path, model, reflection, absorption):
    updates = {
        "time": {"nlim": 1},
        "problem": {"rho0": 2.0, "density_gradient": 0.0},
        "laser": {"model": model, "critical_reflection": reflection,
                  "absorption_model": absorption, "report_diagnostics": "false"},
    }
    log = run_deck(tmp_path / "run",
                   REPO / "inputs/mhd/two_temperature_laser_reflection.athinput",
                   updates, success=False)
    assert "Laser transport failed: failed_rays=1" in log
    assert "Invalid laser launches=1" in log
    diagnostic = next(line for line in log.splitlines()
                      if line.startswith("laser: launched="))
    fields = dict(re.findall(r"(\w+)=([^\s]+)", diagnostic))
    assert float(fields["launched"]) == 1.0
    assert float(fields["deposited"]) == 0.0
    assert int(fields["segments"]) == 0


@pytest.mark.parametrize("parameter,value,message", (
    ("implicit_x1_inner_temperature", -1, "must be finite and non-negative"),
    ("implicit_x1_inner_temperature", "nan", "must be finite and non-negative"),
    ("implicit_x1_inner_temperature", "1e100", "blackbody energy overflows"),
    ("implicit_x1_inner_boundary", "neumann", "requires implicit transport"),
    ("implicit_x1_inner_value", 1, "Specify only one")))
def test_radiation_bath_invalid_input(tmp_path, parameter, value, message):
    updates = thermal_updates()
    updates["thermal_radiation"][parameter] = value
    log = run_deck(tmp_path / "run",
                   REPO / "inputs/tests/thermal_radiation_amr.athinput",
                   updates, remove=THERMAL_VALUES, success=False)
    assert message in log


@pytest.mark.parametrize("parameter,value,message", (
    ("t_ei_coulomb_log_model", "unknown", "must be constant, spitzer, or lee_more"),
    ("t_ei_coulomb_log_floor", 0, "log floor, and multiplier must be positive"),
    ("t_ei_coulomb_log_floor", "nan", "log floor, and multiplier must be positive")))
def test_exchange_invalid_input(tmp_path, parameter, value, message):
    updates = {"meshblock": {"nx1": 8},
               "mhd": {"t_ei_model": "spitzer", parameter: value}}
    log = run_deck(tmp_path / "run", REPO / "inputs/mhd/two_material_relax.athinput",
                   updates, success=False)
    assert message in log
