"""Batched Arnoldi projections preserve the independent conservative AMR equations."""
import re

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr import test_nr_thermal_radiation_amr_accuracy_cpu as accuracy
from test_suite.nr import test_nr_thermal_radiation_amr_roundoff_cpu as roundoff
from test_suite.nr import test_nr_thermal_radiation_cell_amg_amr as amr
from test_suite.nr.test_nr_thermal_radiation_cell_amg import cell_amg_executable
from test_suite.nr.test_nr_thermal_radiation_cell_amg_batch import input_deck, compare_groups
from test_suite.nr.test_nr_thermal_radiation_amr_restart_mpicpu import execute


@pytest.fixture(autouse=True)
def cgs2_deck(tmp_path, monkeypatch):
    deck = tmp_path / "cgs2.athinput"
    deck.write_text(utils.INPUT.read_text()+"\n<thermal_radiation>\n"
                    "implicit_gmres_orthogonalization = cgs2\nsolver_summary = true\n")
    monkeypatch.setattr(utils, "INPUT", deck)
    monkeypatch.setattr(accuracy, "INPUT", deck)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("dimensions,normal,variable,inner", (
    (2, 0, True, 0), (3, 1, False, 0), (3, 2, False, 20),
))
def test_cgs2_composite_equation(tmp_path, cell_amg_executable, ranks,
                                dimensions, normal, variable, inner):
    amr.test_cell_amr_composite_equation(tmp_path, cell_amg_executable, ranks,
                                       dimensions, normal, 2, variable, inner)


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("inner", (0, 20))
def test_cgs2_stiff_reference(tmp_path, monkeypatch, cell_amg_executable, ranks, inner):
    roundoff.test_stiff_amr_backward_error(tmp_path, monkeypatch,
                                          cell_amg_executable, ranks, inner)


@pytest.mark.parametrize("halo", ("packed", "host"))
def test_cgs2_regrid_tail_restart(tmp_path, cell_amg_executable, monkeypatch, halo):
    amr.test_cell_amr_batch_same_count_regrid(
        tmp_path, cell_amg_executable, monkeypatch, halo, reuse=False)


@pytest.mark.parametrize("metric,restart", (("volume", 2), ("preconditioner", 30)))
def test_cgs2_metric_and_short_restart(tmp_path, cell_amg_executable, metric, restart):
    with utils.INPUT.open('a') as deck:
        deck.write(f'\n<thermal_radiation>\nimplicit_gmres_metric = {metric}\n'
                   f'implicit_gmres_restart = {restart}\n')
    amr.test_cell_amr_composite_equation(tmp_path, cell_amg_executable, 2,
                                       2, 0, 2, True, 0)


def test_summary_accounting_and_physics(tmp_path, cell_amg_executable):
    deck = input_deck(tmp_path, 5)
    deck.write_text(deck.read_text()+"\n<refined_region1>\nlevel = 1\n"
        "x1min = 0\nx1max = 0.5\nx2min = -0.5\nx2max = 0.5\n"
        "x3min = -0.5\nx3max = 0.5\n\n<thermal_radiation>\n"
        "solver_timers = false\nimplicit_amr_order = 2\n")
    flags = ["-i", str(deck), "mesh/nx1=16", "mesh/nx2=16", "meshblock/nx1=8",
             "meshblock/nx2=8", "time/nlim=1", "time/tlim=0.0001"]
    reference = execute(tmp_path / "off", 2, [*flags, "thermal_radiation/solver_summary=false"])
    timed = execute(tmp_path / "on", 2, flags)
    for a, b in zip(reference, timed):
        compare_groups(a, b, 5)
    off_log = (tmp_path / "off/run.log").read_text()
    log = (tmp_path / "on/run.log").read_text()
    assert "# radiation performance:" not in off_log
    assert "# radiation batch timing:" not in off_log
    assert "# radiation solver timer:" not in log
    records = [dict(re.findall(r"(\w+)=([^\s]+)", line))
               for line in re.split(r"\n|(?=# radiation )", log)
               if line.startswith("# radiation batch timing:")]
    assert len(records) == 2 and {r["rank"] for r in records} == {"0", "1"}
    for row in records:
        parts = [float(row[k+"_seconds"]) for k in
                 ("amg_setup", "amg_apply", "halo", "reduction", "other")]
        assert all(np.isfinite(x) and x >= 0 for x in parts)
        np.testing.assert_allclose(sum(parts), float(row["transport_seconds"]), rtol=2e-8)
        assert row["exclusive_dispatch"] == "1"
        assert row["setup_gpu_fences"] == "0"
        assert row["setup_existing_gpu_completion"] == "1"
    assert "orthogonalization=cgs2" in log
