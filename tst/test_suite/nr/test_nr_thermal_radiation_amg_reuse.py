"""Input-selected transfer reuse, refresh controls, and restart persistence."""

import os
from pathlib import Path
import re

import pytest

from test_suite.nr.test_nr_thermal_radiation_amr_restart_mpicpu import execute
from test_suite.nr.test_nr_thermal_radiation_cell_amg_batch import (
    compare_groups, input_deck, records, seed_3d,
)


@pytest.fixture
def reuse_executable(monkeypatch):
    executable = os.environ.get("ATHENA_AMG_REUSE_TEST_EXE")
    if not executable:
        pytest.skip("Requires a reuse-enabled builtin executable in ATHENA_AMG_REUSE_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", str(Path(executable).resolve()))
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")


def reuse_records(directory):
    return [dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in (directory / "run.log").read_text().splitlines()
            if line.startswith("# cell-amg reuse:")]


def test_reuse_input_and_restart(tmp_path, reuse_executable, monkeypatch):
    deck = input_deck(tmp_path, 5)
    checkpoint = seed_3d(tmp_path / "seed", deck, 5)
    flags = ["time/nlim=3", "time/tlim=0.0003"]
    # Old environment controls must not turn on an absent/false input feature.
    monkeypatch.setenv("ATHENA_AMG_REUSE_STEPS", "4")
    monkeypatch.setenv("ATHENA_AMG_REUSE_SLOTS", "5")
    monkeypatch.setenv("ATHENA_AMG_REUSE_DRIFT", "64")
    reference = execute(tmp_path / "default", 4, ["-r", str(checkpoint), *flags])
    assert not reuse_records(tmp_path / "default")
    # Conversely, the input enables reuse even with conflicting old env values.
    monkeypatch.setenv("ATHENA_AMG_REUSE_STEPS", "0")
    monkeypatch.setenv("ATHENA_AMG_REUSE_SLOTS", "0")
    monkeypatch.setenv("ATHENA_AMG_REUSE_DRIFT", "0")
    deck.write_text(deck.read_text()+"\n<thermal_radiation>\nimplicit_amg_reuse = true\n")
    enabled_seed = seed_3d(tmp_path / "enabled_seed", deck, 5)
    candidate = execute(tmp_path / "enabled", 4, ["-r", str(enabled_seed), *flags])
    assert len(reference) == len(candidate) == 3
    for left, right in zip(reference, candidate):
        compare_groups(left, right, 5)
    cache = reuse_records(tmp_path / "enabled")
    assert cache and cache[0]["reused"] == "0"
    assert any(row["reused"] == "1" for row in cache)
    assert all(row["scalar_retries"] == "0" for row in records(tmp_path / "enabled"))
    log = (tmp_path / "enabled/run.log").read_text()
    assert "enabled=1 steps=4 slots=5 drift=64" in log
    saved = sorted((tmp_path / "enabled/rst").glob("*.rst"))[0]
    resumed = execute(tmp_path / "resumed", 4, ["-r", str(saved), *flags])
    compare_groups(candidate[-1], resumed[-1], 5)
    assert reuse_records(tmp_path / "resumed")[0]["reused"] == "0"
    monkeypatch.setenv("ATHENA_AMG_REUSE_STEPS", "4")
    disabled = execute(tmp_path / "disabled", 4, ["-r", str(saved), *flags,
                       "thermal_radiation/implicit_amg_reuse=false"])
    compare_groups(candidate[-1], disabled[-1], 5)
    assert not reuse_records(tmp_path / "disabled")


def test_reuse_refresh_interval(tmp_path, reuse_executable):
    deck = input_deck(tmp_path, 5)
    deck.write_text(deck.read_text()+"\n<thermal_radiation>\nimplicit_amg_reuse = true\n"
                    "implicit_amg_reuse_steps = 1\nimplicit_amg_reuse_slots = 1\n"
                    "implicit_amg_reuse_drift = 0\n")
    checkpoint = seed_3d(tmp_path / "seed", deck, 5)
    flags = ["-r", str(checkpoint), "time/nlim=3", "time/tlim=0.0003"]
    candidate = execute(tmp_path / "refresh", 4, flags)
    reference = execute(tmp_path / "reference", 4,
                        [*flags, "thermal_radiation/implicit_amg_reuse=false"])
    compare_groups(reference[-1], candidate[-1], 5)
    cache = reuse_records(tmp_path / "refresh")
    assert len(cache) >= 3 and all(row["reused"] == "0" for row in cache)
    assert all(int(row["cached_cohorts"]) <= 1 for row in cache)
    assert "enabled=1 steps=1 slots=1 drift=0" in (tmp_path / "refresh/run.log").read_text()


@pytest.mark.parametrize("setting,message", (
    ("implicit_amg_reuse_steps=0", "must be positive"),
    ("implicit_amg_reuse_slots=-1", "must be positive"),
    ("implicit_amg_reuse_drift=-0.5", "finite and nonnegative"),
    ("implicit_amg_reuse_drift=nan", "finite and nonnegative"),
    ("implicit_group_batch_size=1", "requires implicit_group_batch_size >= 2"),
    ("n_groups=1", "and n_groups >= 2"),
))
def test_reuse_invalid_input(tmp_path, reuse_executable, setting, message):
    deck = input_deck(tmp_path, 4)
    deck.write_text(deck.read_text()+"\n<thermal_radiation>\nimplicit_amg_reuse = true\n"
                    +setting.replace("=", " = ", 1)+"\n")
    execute(tmp_path / "invalid", 2, ["-i", str(deck), "time/nlim=0"],
        expected_error=message)


def test_reuse_requires_cell_amg(tmp_path, reuse_executable):
    deck = input_deck(tmp_path, 4)
    deck.write_text(deck.read_text()+"\n<thermal_radiation>\nimplicit_amg_reuse = true\n")
    execute(tmp_path / "invalid", 2, ["-i", str(deck), "time/nlim=0",
        "thermal_radiation/implicit_preconditioner=jacobi",
        "thermal_radiation/implicit_group_batch_size=1"],
        expected_error="implicit_amg_reuse requires implicit_preconditioner=cell-amg")


def test_reuse_requires_build_support(tmp_path, monkeypatch):
    executable = os.environ.get("ATHENA_AMG_REUSE_DISABLED_TEST_EXE")
    if not executable:
        pytest.skip("Requires an ordinary HYPRE builtin executable")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", str(Path(executable).resolve()))
    deck = input_deck(tmp_path, 4)
    execute(tmp_path / "ordinary", 2, ["-i", str(deck), "time/nlim=1"])
    assert not reuse_records(tmp_path / "ordinary")
    deck.write_text(deck.read_text()+"\n<thermal_radiation>\nimplicit_amg_reuse = true\n")
    execute(tmp_path / "invalid", 2, ["-i", str(deck), "time/nlim=0"],
        expected_error="implicit_amg_reuse requires Athena_ENABLE_HYPRE_REUSE=ON")
