"""Production mixed-material multigroup-radiation regressions on CPUs."""

import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils
from test_suite.nr.mixed_thermal_radiation_utils import (
    GROUP_BOUNDS,
    NGROUPS,
    SOURCE_DT,
    electron_heat_capacity_fraction,
    mixed_opacity,
    prepare_case,
    run_mixed_transport_probe,
    write_opacity_table,
)


input_file = Path("mixed_thermal_radiation.athinput")
material0_table = Path("mixed_opacity_ch.dat")
material1_table = Path("mixed_opacity_he.dat")


def planck_integral(x):
    infinity = 6.4939394022668291491
    if x <= 0.0:
        return 0.0
    if x >= 50.0:
        return infinity
    if x < 0.5:
        x2 = x*x
        x3 = x2*x
        return (x3/3.0 - x3*x/8.0 + x3*x2/60.0
                - x3*x2*x2/5040.0 + x3*x2*x2*x2/272160.0
                - x3*x2*x2*x2*x2/13305600.0)
    tail = 0.0
    for n in range(1, 65):
        invn = 1.0/n
        invn2 = invn*invn
        tail += math.exp(-n*x)*(x*x*x*invn + 3.0*x*x*invn2
                                + 6.0*x*invn2*invn + 6.0*invn2*invn2)
    return min(max(infinity-tail, 0.0), infinity)


def planck_group_fraction(lower, upper, temperature):
    infinity = 6.4939394022668291491
    return ((planck_integral(upper/temperature)
             - planck_integral(lower/temperature))/infinity)


def sorted_tab(path):
    data = athena_read.tab(path)
    order = np.argsort(data["x1v"])
    return {name: np.atleast_1d(values)[order]
            for name, values in data.items() if np.ndim(values) > 0}


def run_source_case(basename, material0_fraction):
    flags = [
        f"job/basename={basename}",
        f"problem/yl={material0_fraction}",
        f"problem/yr={material0_fraction}",
    ]
    assert testutils.run(str(input_file), flags=flags), f"{basename} failed."
    return (sorted_tab(f"tab/{basename}.mhd_3t.00000.tab"),
            sorted_tab(f"tab/{basename}.mhd_3t.00001.tab"))


def check_source_update(initial, final, material0_fraction):
    density = 1.0
    tele = initial["tele"][0]
    blackbody = 0.1*tele**4
    expected_fe = electron_heat_capacity_fraction(material0_fraction)
    expected_tele = 0.25/((1.0-expected_fe)+0.25*expected_fe)
    assert np.allclose(initial["tele"], expected_tele,
                       rtol=3.0e-13, atol=3.0e-13)

    group_fields = [f"erad{group:02d}" for group in range(NGROUPS)]
    assert all(field in final for field in group_fields)
    for group, field in enumerate(group_fields):
        old = density*initial[field]
        kappaa = mixed_opacity(
            "absorption", group, density, material0_fraction)
        kappae = mixed_opacity(
            "emission", group, density, material0_fraction)
        fraction = planck_group_fraction(
            GROUP_BOUNDS[group], GROUP_BOUNDS[group+1], tele)
        expected = ((old + SOURCE_DT*density*kappae*blackbody*fraction)
                    /(1.0 + SOURCE_DT*density*kappaa))
        assert np.allclose(final[field], expected/density,
                           rtol=4.0e-11, atol=3.0e-15), field
        assert np.all(final[field] >= 0.0)

    assert np.all(final["eele"] >= 0.0)
    initial_total = initial["eion"]+initial["eele"]+initial["erad"]
    final_total = final["eion"]+final["eele"]+final["erad"]
    assert np.allclose(final_total, initial_total,
                       rtol=4.0e-13, atol=4.0e-13)


def transport_timestep():
    command = [
        "./athena", "-i", str(input_file),
        "job/basename=mixed_radiation_ap_dt",
        "problem/yl=0.25", "problem/yr=0.25",
        "time/nlim=0", "time/tlim=1.0", "time/initial_dt=-1.0",
        "output1/dt=-1.0", "output2/dt=-1.0",
        "thermal_radiation/couple_matter=false",
        "thermal_radiation/c_light=100.0",
        "thermal_radiation/initial_profile=step",
        "thermal_radiation/initial_radiation_temperature_right=0.05",
        "thermal_radiation/initial_radiation_x1=0.0",
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=60.0, check=False)
    if result.returncode != 0:
        pytest.fail(f"Mixed AP timestep run failed:\n{result.stdout}\n{result.stderr}")
    match = re.search(r"cycle=0\s+time=[^\s]+\s+dt=([^\s]+)", result.stdout)
    assert match is not None, result.stdout
    return float(match.group(1))


def last_group_transport_timestep():
    # Give groups 0--18 finite diffusion rates but make only group 19 thin/AP.
    # The anisotropic vacuum faces then have distinct directional maxima, all
    # selected from the final iteration of the face-local group loop.
    write_opacity_table(material0_table, 0, "last-group-ap")
    write_opacity_table(material1_table, 1, "last-group-ap")
    command = [
        "./athena", "-i", str(input_file),
        "job/basename=mixed_radiation_last_group_dt",
        "mesh/nx1=16", "mesh/nx2=8", "mesh/nx3=4",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=4",
        "mesh/ix2_bc=inflow", "mesh/ox2_bc=inflow",
        "mesh/ix3_bc=inflow", "mesh/ox3_bc=inflow",
        "problem/yl=1.0", "problem/yr=0.0",
        "time/nlim=0", "time/tlim=1.0", "time/initial_dt=-1.0",
        "output1/dt=-1.0", "output2/dt=-1.0", "output3/dt=-1.0",
        "thermal_radiation/couple_matter=false",
        "thermal_radiation/source_cfl=0.0",
        "thermal_radiation/c_light=100.0",
        "thermal_radiation/transport_discretization=asymptotic-preserving",
        "thermal_radiation/initial_profile=step",
        "thermal_radiation/initial_radiation_temperature=1.0",
        "thermal_radiation/initial_radiation_temperature_right=0.5",
        "thermal_radiation/initial_radiation_x1=0.0",
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=60.0, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"Last-group AP timestep run failed:\n{result.stdout}\n{result.stderr}")
    match = re.search(r"cycle=0\s+time=[^\s]+\s+dt=([^\s]+)", result.stdout)
    assert match is not None, result.stdout
    return float(match.group(1))


def source_timestep(material0_fraction):
    command = [
        "./athena", "-i", str(input_file),
        "job/basename=mixed_radiation_source_dt",
        f"problem/yl={material0_fraction}",
        f"problem/yr={material0_fraction}",
        "time/nlim=0", "time/tlim=1.0", "time/initial_dt=-1.0",
        "output1/dt=-1.0", "output2/dt=-1.0",
        "thermal_radiation/source_cfl=0.05",
        "thermal_radiation/arad=1000.0",
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=60.0, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"Mixed source timestep run failed:\n{result.stdout}\n{result.stderr}")
    match = re.search(r"cycle=0\s+time=[^\s]+\s+dt=([^\s]+)", result.stdout)
    assert match is not None, result.stdout
    return float(match.group(1))


def expected_source_timestep(initial, material0_fraction):
    density = 1.0
    arad = 1000.0
    tele = initial["tele"][0]
    blackbody = arad*tele**4
    initial_blackbody = arad*0.1**4
    source_rate = 0.0
    for group in range(NGROUPS):
        equilibrium = blackbody*planck_group_fraction(
            GROUP_BOUNDS[group], GROUP_BOUNDS[group+1], tele)
        energy = initial_blackbody*planck_group_fraction(
            GROUP_BOUNDS[group], GROUP_BOUNDS[group+1], 0.1)
        kappaa = mixed_opacity(
            "absorption", group, density, material0_fraction)
        kappae = mixed_opacity(
            "emission", group, density, material0_fraction)
        source_rate += abs(kappae*equilibrium-kappaa*energy)
    # The driver applies its global CFL number after the source-specific limit.
    return 0.4*0.05*density*initial["eele"][0]/source_rate


def test_run():
    try:
        prepare_case(input_file, material0_table, material1_table)

        # Transport consumes the synchronized temperature field.  The only material-EOS
        # inversion in this module is the once-per-cell refresh after source coupling.
        source = Path("../../../src/two_temperature/thermal_radiation.cpp").read_text()
        assert source.count("mixture.ElectronTemperature(") == 1
        transport = source.split("void ThermalRadiation::AddFluxes", 1)[1]
        transport = transport.split("void ThermalRadiation::Couple", 1)[0]
        timestep = source.split("void ThermalRadiation::NewTimeStep", 1)[1]
        assert "ElectronTemperature(" not in transport
        assert "ElectronTemperature(" not in timestep
        assert "w0, temperature" in transport
        assert "temperature(m, 1" in timestep

        results = {}
        for name, fraction in (("ch", 1.0), ("he", 0.0), ("mixed", 0.25)):
            results[name] = run_source_case(
                f"mixed_radiation_{name}", fraction)
            check_source_update(*results[name], fraction)

        # Composition changes Te even at the same rho, total energy, and prescribed
        # initial Te/Ti ratio; the mixed cell is distinct from both pure limits.
        initial_temperatures = [results[name][0]["tele"][0]
                                for name in ("ch", "he", "mixed")]
        assert len(set(initial_temperatures)) == 3

        # Exercise prepared mixed-opacity locations in the source-CFL reduction at
        # both pure-material endpoints and in a genuinely mixed cell.
        for name, fraction in (("ch", 1.0), ("he", 0.0), ("mixed", 0.25)):
            assert source_timestep(fraction) == pytest.approx(
                expected_source_timestep(results[name][0], fraction), rel=4.0e-6)

        # Twenty mixed-opacity groups retain the causal AP CFL without an ngroups factor.
        assert transport_timestep() == pytest.approx(
            0.4/(16.0*100.0), rel=3.0e-6)

        for output in run_mixed_transport_probe(
                input_file, "mixed_radiation_flux_3d_cpu"):
            output.unlink()

        # Only g=19 is optically thin.  Its AP rates are 0.5/dx in each
        # direction, so after the independent directional maxima are summed the
        # driver CFL gives dt=0.4/[c_hat*(16+8+4)].
        assert last_group_transport_timestep() == pytest.approx(
            0.4/(100.0*(16.0+8.0+4.0)), rel=3.0e-6)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        input_file.unlink(missing_ok=True)
        material0_table.unlink(missing_ok=True)
        material1_table.unlink(missing_ok=True)
        testutils.cleanup()
