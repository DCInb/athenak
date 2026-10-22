"""Finite-area source normalization, spatial moments, rank ownership and restart."""

import os
import shlex

import numpy as np
import pytest
from scipy.integrate import quad

from test_suite.nr.test_nr_fast_electrons_cpu import run_case
from test_suite.nr.test_nr_fast_electron_hybrid_cpu import (
    executable, multidimensional_deck, particle_footer, restart,
)


def source_deck():
    deck, flags = multidimensional_deck(3, False)
    flags.update({"fast_electrons/electromagnetic": "false",
                  "fast_electrons/source_radius": 0.1,
                  "fast_electrons/source_radial_order": 32,
                  "fast_electrons/source_azimuthal_points": 8,
                  "fast_electrons/source_radial_power": 8,
                  "fast_electrons/source_cutoff": 1.5,
                  "fast_electrons/end_time": 0.004,
                  "time/tlim": 0.004, "time/initial_dt": 0.002})
    spatial = "\n".join(f"{k.split('/')[1]} = {v}" for k, v in flags.items()
                        if k.startswith("fast_electrons/source_"))
    deck = deck.replace("<fast_electrons>", "<fast_electrons>\n"+spatial)
    return deck, flags


def last_packets(result):
    return particle_footer(sorted((result["work"]/"rst").glob("*.rst"))[-1])[1]


def conserved_source(result):
    ledger = result["diagnostics"][-1]
    assert ledger["injected_erg"] == pytest.approx(4e7, rel=1e-12)
    assert abs(ledger["residual_erg"]) < 4e7*1e-12
    p = last_packets(result)
    assert np.sum(p["energy"]*p["weight"]) == pytest.approx(4e7, rel=1e-12)


def test_finite_source_area_and_continued_restart(executable, tmp_path):
    deck, flags = source_deck()
    result = run_case(executable, tmp_path/"full", flags, input_text=deck)
    conserved_source(result)
    packets = last_packets(result)
    w = packets["weight"]/packets["weight"].sum()
    y, z = packets["x"][:, 1]-0.01, packets["x"][:, 2]-0.01
    r2 = y*y+z*z
    expected = 0.1**2*quad(lambda r: r**3*np.exp(-r**8), 0, 1.5)[0]/quad(
        lambda r: r*np.exp(-r**8), 0, 1.5)[0]
    assert np.dot(w, r2) == pytest.approx(expected, rel=1e-9)
    assert abs(np.dot(w, y)) < 1e-15
    assert abs(np.dot(w, z)) < 1e-15
    assert np.dot(w, y*y) == pytest.approx(expected/2, rel=1e-9)
    assert np.dot(w, z*z) == pytest.approx(expected/2, rel=1e-9)
    assert len(np.unique(packets["rng"])) == len(packets)
    checkpoints = sorted((result["work"]/"rst").glob("*.rst"))
    assert len(checkpoints) >= 3
    resumed = restart(executable, tmp_path/"resumed", checkpoints[1])
    np.testing.assert_array_equal(last_packets(resumed), packets)


def test_finite_source_mpi_when_requested(executable, tmp_path):
    launch = os.getenv("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launch:
        pytest.skip("set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    deck, flags = source_deck()
    serial = run_case(executable, tmp_path/"serial", flags, input_text=deck)
    parallel = run_case(executable, tmp_path/"parallel", flags, input_text=deck,
                        launcher=shlex.split(launch))
    conserved_source(parallel)
    np.testing.assert_array_equal(last_packets(parallel), last_packets(serial))


@pytest.mark.parametrize("flags,message", [
    ({"mesh/nx3": 1, "meshblock/nx3": 1}, "requires a 3D mesh"),
    ({"fast_electrons/source_radius": 1.0}, "disk must fit"),
    ({"fast_electrons/source_radial_order": 0}, "must be 1..64"),
    ({"fast_electrons/source_azimuthal_points": 3}, "multiple of four"),
    ({"fast_electrons/max_packets": 8}, "quadrature exceeds max_packets"),
])
def test_invalid_source_geometry(executable, tmp_path, flags, message):
    deck, common = source_deck()
    common.update(flags)
    log = run_case(executable, tmp_path, common, input_text=deck, expect_failure=True)
    assert message in log
