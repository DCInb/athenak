"""Opt-in solver timers preserve physics and provide additive rank-local accounting."""

import os
from pathlib import Path
import re

import numpy as np
import pytest

from test_suite.nr import test_nr_thermal_radiation_amr_cholmod_mpicpu as cholmod
from test_suite.nr import thermal_radiation_amr_utils as amr_utils
from test_suite.nr.thermal_radiation_amr_utils import INPUT, ordered_fields, run_case


TIMER_PREFIX = "# radiation solver timer: "
GROUP_PREFIX = "# radiation solver group timer: "
COUNTER_PREFIX = "# radiation solver counters: "
COUNTERS = ("inner_iterations", "outer_iterations", "cg_iterations", "inner_cap_hits")


def _timer_case(monkeypatch, directory, *args, enabled=None, **kwargs):
    # Parameter overrides require keys already declared in the input. Keep the
    # canonical regression deck unchanged, and preserve an actually omitted flag
    # for the default-path check.
    input_file = directory.parent / f"{directory.name}.athinput"
    extras = ("\n<thermal_radiation>\nsource_integrator = nonlinear\n"
              "source_report = false\nkappa_absorption_0 = 0.1\n"
              "kappa_absorption_1 = 0.1\nkappa_emission_0 = 0.1\n"
              "kappa_emission_1 = 0.1\n")
    if enabled is not None:
        extras += f"solver_timers = {str(enabled).lower()}\n"
    input_file.write_text(INPUT.read_text()+extras)
    with monkeypatch.context() as patch:
        patch.setattr(amr_utils, "INPUT", input_file)
        return run_case(directory, *args, **kwargs)


def _records(log, prefix):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in log.splitlines() if line.startswith(prefix)]


def _assert_identical(left, right):
    assert len(left) == len(right)
    for before, after in zip(left, right):
        assert before["cycle"] == after["cycle"] and before["time"] == after["time"]
        before_topology, before_fields = ordered_fields(before)
        after_topology, after_fields = ordered_fields(after)
        np.testing.assert_array_equal(before_topology, after_topology)
        assert before_fields.keys() == after_fields.keys()
        for field in before_fields:
            np.testing.assert_array_equal(before_fields[field], after_fields[field])


def _check_accounting(log, ranks, *, source=True):
    timers = _records(log, TIMER_PREFIX)
    groups = _records(log, GROUP_PREFIX)
    counters = _records(log, COUNTER_PREFIX)
    assert timers and groups and counters
    assert {int(row["rank"]) for row in timers} == set(range(ranks))
    summaries = {}
    for row in timers:
        assert row["gpu_phase_fences"] == "1"
        assert int(row["calls"]) > 0
        wall, own = float(row["wall_seconds"]), float(row["self_seconds"])
        assert np.isfinite(wall) and np.isfinite(own)
        assert wall >= 0 and own >= 0 and own <= wall
        key = int(row["rank"]), int(row["cycle"]), row["solve"]
        phases = summaries.setdefault(key, {})
        assert row["phase"] not in phases
        phases[row["phase"]] = row
    expected_solves = {"transport", "source"} if source else {"transport"}
    assert {key[2] for key in summaries} == expected_solves
    for (rank, cycle, solve), phases in summaries.items():
        total = float(phases[solve]["wall_seconds"])
        # Only self times are additive: inclusive child times overlap with parents.
        np.testing.assert_allclose(sum(float(row["self_seconds"])
                                       for row in phases.values()), total,
                                   rtol=2e-8, atol=1e-12)
        for row in phases.values():
            assert float(row["wall_seconds"]) <= total*(1+2e-8)+1e-12
        summary = [row for row in counters if int(row["rank"]) == rank
                   and int(row["cycle"]) == cycle and row["solve"] == solve]
        assert len(summary) == 1
        if solve == "source":
            assert all(int(summary[0][counter]) == 0 for counter in COUNTERS)
            continue
        attempts = [row for row in groups if int(row["rank"]) == rank
                    and int(row["cycle"]) == cycle]
        assert len(attempts) == int(phases["group"]["calls"])
        np.testing.assert_allclose(sum(float(row["wall_seconds"]) for row in attempts),
                                   float(phases["group"]["wall_seconds"]),
                                   rtol=2e-8, atol=1e-12)
        for counter in COUNTERS:
            assert sum(int(row[counter]) for row in attempts) == int(summary[0][counter])
        for row in attempts:
            assert int(row["order"]) in (1, 2)
            assert int(row["attempt"]) >= 0
            assert all(int(row[counter]) >= 0 for counter in COUNTERS)
    return summaries, groups, counters


@pytest.mark.parametrize("order,inner,ranks,uniform", (
    (1, 0, 1, False), (2, 0, 1, False), (2, 20, 2, False), (2, 0, 1, True),
))
def test_solver_timers_preserve_solution(tmp_path, monkeypatch, order, inner, ranks, uniform):
    # Nonlinear matter coupling is a separate measured stage. Both report flags
    # remain off, so timings must work independently of convergence diagnostics.
    flags = ("time/nlim=1", "time/tlim=0.0001", "problem/pl=0.75", "problem/pr=0.75",
             f"thermal_radiation/implicit_amr_order={order}",
             f"thermal_radiation/implicit_amr_inner_iterations={inner}",
             "thermal_radiation/implicit_report=false",
             "thermal_radiation/source_report=false",
             "thermal_radiation/couple_matter=true",
             "thermal_radiation/source_integrator=nonlinear",
             "thermal_radiation/kappa_absorption_0=0.1",
             "thermal_radiation/kappa_absorption_1=0.1",
             "thermal_radiation/kappa_emission_0=0.1",
             "thermal_radiation/kappa_emission_1=0.1")
    if uniform:
        flags += ("mesh_refinement/refinement=none",)
    outputs = []
    for name, setting in (("default", None), ("disabled", False), ("enabled", True)):
        directory = tmp_path / name
        output = _timer_case(monkeypatch, directory, 2, ranks=ranks,
                             enabled=setting, extra_flags=flags)
        outputs.append(output)
        log = (directory / "run.log").read_text()
        if name != "enabled":
            assert "# radiation solver " not in log
        else:
            phases, _, counters = _check_accounting(log, ranks)
            assert len(phases) == 2*ranks
            for row in counters:
                if row["solve"] != "transport":
                    continue
                if order == 1 or uniform:
                    assert int(row["cg_iterations"]) > 0
                    assert int(row["outer_iterations"]) == 0
                else:
                    assert int(row["outer_iterations"]) > 0
                    if inner > 0:
                        assert int(row["inner_iterations"]) > 0
                        assert 0 < int(row["maximum_inner_iterations"]) <= inner
            assert "# implicit thermal radiation:" not in log
            assert "# nonlinear thermal radiation source:" not in log
    _assert_identical(outputs[0], outputs[1])
    _assert_identical(outputs[0], outputs[2])


def _select_cholmod(monkeypatch):
    executable = os.environ.get("ATHENA_CHOLMOD_TEST_EXE")
    if not executable:
        pytest.skip("Requires ATHENA_CHOLMOD_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)


def test_solver_timers_cholmod_phases(tmp_path, monkeypatch):
    _select_cholmod(monkeypatch)
    flags = (*cholmod.BLOCK18, "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_report=false")
    reference = _timer_case(monkeypatch, tmp_path / "reference", 2, "amr-aggregate",
                            ranks=2, extra_flags=flags)
    timed = _timer_case(monkeypatch, tmp_path / "timed", 2, "amr-aggregate", ranks=2,
                        enabled=True, extra_flags=flags)
    _assert_identical(reference, timed)
    phases, _, _ = _check_accounting(
        (tmp_path / "timed/run.log").read_text(), 2, source=False)
    for phase in phases.values():
        for name in ("aggregate_build", "aggregate_numeric_d2h", "aggregate_graph",
                     "cholmod_analyze", "cholmod_factor", "aggregate_apply",
                     "aggregate_restrict", "aggregate_rhs_d2h", "aggregate_apply_mpi",
                     "cholmod_solve", "aggregate_correction_h2d", "aggregate_prolong"):
            assert name in phase, name
        assert int(phase["cholmod_factor"]["calls"]) >= 2
        assert int(phase["cholmod_solve"]["calls"]) == int(phase["aggregate_apply"]["calls"])


def test_solver_timers_count_positivity_retry(tmp_path, monkeypatch):
    _select_cholmod(monkeypatch)
    original = cholmod._run

    def with_timers(directory, arguments, ranks=None):
        arguments = list(arguments)
        if "-i" in arguments:
            index = arguments.index("-i")+1
            input_file = tmp_path / "timer-impulse.athinput"
            input_file.write_text(Path(arguments[index]).read_text() +
                                  "\n<thermal_radiation>\nsolver_timers = true\n")
            arguments[index] = str(input_file)
        # The subsequent -r calls inherit the flag from this initial checkpoint.
        return original(directory, arguments, ranks)

    monkeypatch.setattr(cholmod, "_run", with_timers)
    # Retain the existing independent equation, mass, positivity and MPI checks.
    cholmod.test_global_aggregate_positivity_fallback(tmp_path)
    log = (tmp_path / "order2/run.log").read_text()
    phases, groups, _ = _check_accounting(log, 2, source=False)
    for rank in (0, 1):
        attempts = [row for row in groups if int(row["rank"]) == rank]
        assert len(attempts) >= 2
        assert {int(row["group"]) for row in attempts} == {0}
        assert {int(row["order"]) for row in attempts} == {1, 2}
        assert len({int(row["attempt"]) for row in attempts}) == len(attempts)
        assert int(phases[rank, 0, "transport"]["cholmod_factor"]["calls"]) >= 2
