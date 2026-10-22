"""Resolve weak Planck bands without subtracting saturated cumulative integrals."""
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis/python"))
import athena_read


def band_fraction(lower, upper, temperature):
    """Independent Gauss-Legendre quadrature of the spectral density."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    a, b = lower/temperature, upper/temperature
    x = a + (nodes+1)*(0.5*(b-a))
    spectrum = x**3*np.exp(-x)/(-np.expm1(-x))
    return 0.5*(b-a)*np.dot(weights, spectrum)/(np.pi**4/15)


@pytest.mark.parametrize("bounds", [
    (20., 25., 30., 40., 50., 80.),
    (30., 30.00001, 40., 40.00001, 80., 150.),
    (1.e-8, 2.e-8, 1.e-4, 0.1, 0.49, 0.5),
])
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
def test_planck_bands(tmp_path, bounds, source, ranks):
    executable = os.environ.get("ATHENA_RADIATION_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_RADIATION_TEST_EXE to a builtin executable or wrapper")
    groups = len(bounds)-1
    text = (REPO / "inputs/hydro/two_temperature_mgfld.athinput").read_text()
    text += f"""
<job>
basename = planck
<meshblock>
nx1 = 8
<time>
nlim = 1
tlim = 1.e-4
<thermal_radiation>
n_groups = {groups}
arad = 1.0
initial_radiation_temperature = {0.0 if source else 1.0}
couple_matter = {str(source).lower()}
source_cfl = 0.0
source_integrator = lagged
<output1>
data_format = %24.17e
dt = 1.e-4
"""
    text += "<thermal_radiation>\n"
    for g, bound in enumerate(bounds):
        text += f"group_bound_{g} = {bound:.17g}\n"
    for g in range(groups):
        text += f"kappa_transport_{g} = 100.0\n"
        text += f"kappa_absorption_{g} = 1.0\nkappa_emission_{g} = 1.0\n"
    deck = tmp_path / "input.athinput"
    deck.write_text(text)
    command = ["mpirun", "-np", str(ranks), str(executable), "-i", str(deck)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True,
                            text=True, timeout=90)
    (tmp_path / "run.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    path = sorted((tmp_path / "tab").glob("planck*.tab"))[-1]
    data = athena_read.tab(str(path))
    time = float(re.search(r"time=(\S+)", path.read_text()).group(1))
    for g in range(groups):
        # A lagged source evaluates the spectrum at the known initial Te=1.
        # This isolates the integral from nonlinear root-finding tolerance.
        temperature = 1.0
        expected = temperature**4*band_fraction(bounds[g], bounds[g+1], temperature)
        if source:
            expected *= time/(1+time)
        assert expected > 0
        np.testing.assert_allclose(data[f"erad{g:02d}"], expected,
                                   rtol=2.e-11, atol=0.0)
