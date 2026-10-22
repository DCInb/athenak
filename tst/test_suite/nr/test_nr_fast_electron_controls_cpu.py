"""Memory and sampling controls; the same tests run on CPU and CUDA/MPI builds."""
import csv
import os
from pathlib import Path
import re
import shlex

import numpy as np
import pytest

from .test_nr_fast_electrons_cpu import INPUT, executable, run_case, speed
from .test_nr_fast_electron_hybrid_cpu import (
    checkpoint_input, hybrid_input, particle_footer, restart,
)


def launch(parallel):
    if not parallel:
        return ()
    value = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not value:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for MPI controls tests")
    return shlex.split(value)


def records(run, kind):
    # Rank-local logs are authoritative; MPI stdout can interleave partial lines.
    pattern = ('fast_electron_sampling.rank*.log' if kind == 'sampling' else
               'fast_electron_memory.rank*.log')
    paths = sorted(run.get('work', Path('/nonexistent')).glob(pattern))
    log = '\n'.join(p.read_text() for p in paths) if paths else run['log']
    return [dict(re.findall(r'([a-z0-9_]+)=([^\s]+)', line))
            for line in log.splitlines()
            if line.startswith(f"# fast_electrons_{kind} ")]


def final_packets(run):
    return particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])


def fixed_flags(spectrum):
    return {"time/tlim": .0037, "time/initial_dt": .0037,
            "fast_electrons/start_time": .0002, "fast_electrons/end_time": .0037,
            "fast_electrons/emission_interval": .001,
            "fast_electrons/packets_per_bin": 2,
            "fast_electrons/distribution_file": spectrum,
            "fast_electrons/origin_x1": .49,
            "fast_electrons/collisions": "false",
            "fast_electrons/memory_diagnostics": "true",
            "fast_electrons/particle_diagnostics_interval": 1,
            "fast_electrons/particle_diagnostics_cells": "true"}


def same_packets(a, b):
    ha, pa = final_packets(a)
    hb, pb = final_packets(b)
    assert ha[6] == hb[6]
    np.testing.assert_array_equal(pa["rng"], pb["rng"])
    for field in ("x", "n", "energy", "weight"):
        np.testing.assert_allclose(pa[field], pb[field], rtol=2e-12, atol=2e-14,
                                   err_msg=field)


@pytest.mark.parametrize("parallel", [False, True])
def test_fixed_emission_independent_of_field_steps_and_mesh(executable, tmp_path, parallel):
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("100 0 0 1\n100 0 0 3\n")
    flags = fixed_flags(spectrum)
    runs = []
    for name, nx, cfl, initial_dt in [("coarse", 64, .25, .0037),
                                      ("time", 64, .08, .0007),
                                      ("mesh", 128, .25, .0037)]:
        run = run_case(executable, tmp_path / name, dict(flags, **{
            "mesh/nx1": nx, "fast_electrons/field_cfl": cfl,
            "fast_electrons/magnetic_diffusivity_cm2_s": 0,
            "time/initial_dt": initial_dt}),
            input_text=checkpoint_input(hybrid_input()), launcher=launch(parallel))
        runs.append(run)
        h, p = final_packets(run)
        assert h[0] == 2 and h[6] == len(p) == 16
        assert len(np.unique(p["rng"])) == 16
        # Four bins, with a half-width final bin, and two midpoint times per bin.
        times = [.00045, .00095, .00145, .00195, .00245, .00295, .003325, .003575]
        # Match the source physical constants; the older helper rounds mc^2 in keV.
        beta = np.sqrt(1-(1+100*1.602176634e-9/8.1871057880e-7)**-2)
        expected_x = np.sort(np.repeat(.49 + 29.9792458*beta*(.0037-np.array(times)), 2))
        np.testing.assert_allclose(np.sort(p["x"][:, 0]), expected_x, atol=3e-14, rtol=0)
        d = run["diagnostics"][-1]
        assert d["injected_erg"] == pytest.approx(35., rel=3e-13)
        assert d["active_erg"] == pytest.approx(35., rel=2e-12)
        assert abs(d["residual_erg"]) < 1e-10
    for run in runs[1:]:
        same_packets(runs[0], run)
    # Changing field CFL really changed the number of accepted steps.
    assert len(records(runs[1], "sampling")) > len(records(runs[0], "sampling"))


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("power", [1e13, 1e200])
def test_local_weighted_counts_and_memory_snapshots(executable, tmp_path, parallel, power):
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("100 0 0 1\n100 0 0 3\n")
    flags = dict(fixed_flags(spectrum), **{
        "time/tlim": .0006, "time/initial_dt": .0006,
        "fast_electrons/power_erg_s": power})
    run = run_case(executable, tmp_path / "counts", flags,
                   input_text=checkpoint_input(INPUT.read_text()), launcher=launch(parallel))
    cells = []
    for path in sorted(run["work"].glob("fast_electron_sampling.rank*.csv")):
        with path.open() as stream:
            cells += list(csv.DictReader(stream))
    assert cells
    nonempty = [r for r in cells if int(r["ppc"])]
    assert {r["phase"] for r in nonempty} == {"injected", "transported"}
    assert all(int(r["ppc"]) == 2 for r in nonempty)
    assert all(float(r["neff"]) == pytest.approx(1.6) for r in nonempty)
    for r in cells:
        if not int(r["ppc"]):
            assert float(r["neff"]) == float(r["weight_scale"]) == 0
    for r in records(run, "sampling"):
        selected = [c for c in cells if c["phase"] == r["phase"] and c["rank"] == r["rank"]]
        assert int(r["packets"]) == sum(int(c["ppc"]) for c in selected)
        assert float(r["neff_sum"]) == pytest.approx(sum(float(c["neff"]) for c in selected))
    capacities = records(run, "capacity")
    assert len(list(run["work"].glob("fast_electron_memory.rank*.log"))) == (2 if parallel else 1)
    assert len(list(run["work"].glob("fast_electron_sampling.rank*.log"))) == (2 if parallel else 1)
    assert capacities
    for r in capacities:
        assert int(r["total_bytes"]) == sum(int(r[f"{k}_bytes"]) for k in
                                            ("packet", "source", "mesh", "runtime", "diagnostic"))
        slots = sum(int(r[f"{k}_slots"]) for k in ("storage", "scratch", "work0", "work1", "export"))
        assert int(r["packet_bytes"]) == slots * 104
        assert int(r["live_bytes"]) == int(r["live_packets"]) * 104
        assert int(r["live_packets"]) <= slots
        assert int(r["total_bytes"]) <= int(r["peak_bytes"])
    if parallel:
        assert {r["rank"] for r in capacities} == {"0", "1"}
        assert any(int(r["packets"]) == 0 for r in records(run, "sampling"))
    # These observers do not alter the source, weights, or particle histories.
    quiet = run_case(executable, tmp_path / "quiet", dict(flags, **{
        "fast_electrons/memory_diagnostics": "false",
        "fast_electrons/particle_diagnostics_interval": 0}),
        input_text=checkpoint_input(INPUT.read_text()), launcher=launch(parallel))
    same_packets(run, quiet)


@pytest.mark.parametrize("parallel", [False, True])
def test_fixed_emission_restart_in_middle_of_bin(executable, tmp_path, parallel):
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("100 0 0 1\n100 0 0 3\n")
    flags = dict(fixed_flags(spectrum), **{"time/initial_dt": .0006,
                                          "fast_electrons/magnetic_diffusivity_cm2_s": 0})
    full = run_case(executable, tmp_path / "full", flags,
                    input_text=checkpoint_input(hybrid_input()), launcher=launch(parallel))
    checkpoints = sorted((full["work"] / "rst").glob("*.rst"))
    middle = next(p for p in checkpoints if 0 < particle_footer(p)[0][6] < 16)
    spectrum.unlink()  # The normalized distribution lives in the checkpoint.
    resumed = restart(executable, tmp_path / "resumed", middle,
                      {"fast_electrons/field_cfl": .07}, launcher=launch(parallel))
    same_packets(full, resumed)
    assert resumed["diagnostics"][-1]["injected_erg"] == pytest.approx(35., rel=3e-13)
    for key, value in [("emission_interval", .0005), ("packets_per_bin", 3),
                       ("end_time", .004), ("emission_interval", 0)]:
        restart(executable, tmp_path / f"reject-{key}-{value}", middle,
                {f"fast_electrons/{key}": value}, launcher=launch(parallel),
                failure="fixed emission schedule cannot change on restart")


@pytest.mark.parametrize("parallel", [False, True])
def test_aggregate_budget_includes_mesh_and_growth_overlap(executable, tmp_path, parallel):
    flags = {"time/tlim": .003, "time/initial_dt": .001,
             "fast_electrons/end_time": .003, "fast_electrons/collisions": "false",
             "fast_electrons/packets_per_bin": 300, "fast_electrons/origin_x1": .05,
             "fast_electrons/memory_diagnostics": "true"}
    baseline = run_case(executable, tmp_path / "baseline", flags, launcher=launch(parallel))
    events = records(baseline, "allocation")
    # A storage-growth event includes old and new buffers simultaneously.
    growth = [r for r in events if 'fast-electron-packet-storage' in r["label"] and int(r["live_packets"]) > 0]
    assert growth, events
    r = growth[0]
    assert int(r["live_known"]) == 1
    assert int(r["packet_bytes"]) > int(r["request_bytes"])
    budget = (int(r["total_bytes"]) - 1) / 1048576
    log = run_case(executable, tmp_path / "limited", dict(flags, **{
        "fast_electrons/memory_budget_mib": budget}), expect_failure=True,
        launcher=launch(parallel))
    assert "aggregate memory budget exceeded" in log
    assert "purpose=particle storage growth" in log
    peak_seen = max(int(e["total_bytes"]) for e in records({"log": log}, "allocation"))
    assert peak_seen < int(r["total_bytes"])
    # With no particles at all the mesh still consumes the aggregate budget.
    log = run_case(executable, tmp_path / "mesh-limit", dict(flags, **{
        "fast_electrons/power_erg_s": 0, "fast_electrons/memory_budget_mib": .001}),
        expect_failure=True, launcher=launch(parallel))
    assert "aggregate memory budget exceeded" in log
    # Reserved headroom has the same logical effect as reducing the budget.
    log = run_case(executable, tmp_path / "headroom", dict(flags, **{
        "fast_electrons/memory_budget_mib": budget+1,
        "fast_electrons/memory_headroom_mib": 1}),
        expect_failure=True, launcher=launch(parallel))
    assert "aggregate memory budget exceeded" in log


@pytest.mark.parametrize("key,value", [
    ("emission_interval", -1), ("emission_interval", "nan"),
    ("emission_interval", 1e-30), ("particle_diagnostics_interval", -1),
    ("memory_budget_mib", -1), ("memory_headroom_mib", "nan"),
])
def test_invalid_controls(executable, tmp_path, key, value):
    log = run_case(executable, tmp_path, {f"fast_electrons/{key}": value}, expect_failure=True)
    assert key in log


def test_cannot_enable_fixed_emission_on_legacy_restart(executable, tmp_path):
    run = run_case(executable, tmp_path / "legacy", {"fast_electrons/collisions": "false"},
                   input_text=checkpoint_input(INPUT.read_text()))
    path = sorted((run["work"] / "rst").glob("*.rst"))[-1]
    assert particle_footer(path)[0][0] == 1
    restart(executable, tmp_path / "rejected", path,
            {"fast_electrons/emission_interval": .001},
            failure="cannot enable fixed emission on a legacy checkpoint")


@pytest.mark.parametrize("per_rank", [False, True])
def test_fixed_schedule_mpi_checkpoint_layout_and_repartition(executable, tmp_path, per_rank):
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("100 0 0 1\n")
    flags = dict(fixed_flags(spectrum), **{"time/tlim": .00045,
                                          "time/initial_dt": .00045})
    first = run_case(executable, tmp_path / "first", flags,
                     input_text=checkpoint_input(INPUT.read_text(), per_rank),
                     launcher=launch(True))
    # The first birth lies exactly on the upper boundary, so is consumed only
    # by the resumed half-open interval, once and without skipping an event.
    assert first["diagnostics"][-1]["injected_erg"] == 0
    path = sorted((first["work"] / "rst").rglob("*.rst"))[0]
    resumed = restart(executable, tmp_path / "resumed", path,
                      {"time/tlim": .0037}, launcher=launch(per_rank))
    d = resumed["diagnostics"][-1]
    assert d["packets"] == 8
    assert d["injected_erg"] == pytest.approx(35., rel=1e-12)
    assert abs(d["residual_erg"]) < 1e-10


def test_fixed_sampling_diagnostics_on_refined_mesh(executable, tmp_path):
    from .test_nr_fast_electron_hybrid_cpu import multidimensional_deck
    deck, flags = multidimensional_deck(3, True)
    flags.update({"fast_electrons/emission_interval": .001,
                  "fast_electrons/memory_diagnostics": "true",
                  "fast_electrons/particle_diagnostics_interval": 1,
                  "fast_electrons/particle_diagnostics_cells": "true",
                  "fast_electrons/power_erg_s": 1e13,
                  "fast_electrons/magnetic_diffusivity_cm2_s": 0})
    run = run_case(executable, tmp_path, flags, input_text=deck, launcher=launch(True))
    assert final_packets(run)[0][6] == 2
    for rank in (0, 1):
        with (run["work"]/f"fast_electron_sampling.rank{rank}.csv").open() as stream:
            cells = list(csv.DictReader(stream))
        last = records(run, 'sampling')
        last = [r for r in last if r['rank'] == str(rank)][-1]
        active = [r for r in cells if r['substep'] == last['substep'] and r['phase'] == last['phase']]
        assert len(active) == int(last['cells'])
        assert sum(int(r['ppc']) for r in active) == int(last['packets'])
