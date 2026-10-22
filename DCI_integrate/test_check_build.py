"""A normal production launch must not silently reuse an outdated executable."""

import os
from pathlib import Path
import subprocess

import pytest

from check_build import newer_source


@pytest.mark.parametrize("source", [
    "src/diffusion/conduction_implicit.cpp",
    "src/materials/material_mixture.hpp",
    "DCI_integrate/dci_integrate.cpp",
    "DCI_integrate/dci_amr.hpp",
    "src/CMakeLists.txt",
    "cmake/FindAthenaHYPRE.cmake",
    "config.hpp.in",
])
def test_requires_rebuild_after_source_change(tmp_path, source):
    repo = tmp_path
    case = repo / "DCI_integrate"
    executable = case / "build/src/athena"
    executable.parent.mkdir(parents=True)
    executable.write_text("test executable")
    changed = repo / source
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("test source")
    os.utime(executable, ns=(100, 100))
    os.utime(changed, ns=(200, 200))
    assert newer_source(executable, repo, case) == changed
    os.utime(executable, ns=(300, 300))
    assert newer_source(executable, repo, case) is None


def test_input_and_logs_do_not_require_rebuild(tmp_path):
    case = tmp_path / "DCI_integrate"
    case.mkdir()
    executable = case / "athena"
    executable.touch()
    os.utime(executable, ns=(100, 100))
    for name in ("dci_integrate.athinput", "README.md", "run.log"):
        (case / name).touch()
    assert newer_source(executable, tmp_path, case) is None


def test_launcher_stops_before_staging_stale_default(tmp_path):
    case = Path(__file__).resolve().parent
    build = tmp_path / "build"
    executable = build / "src/athena"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o755)
    os.utime(executable, ns=(100, 100))
    environment = tmp_path / "environment.sh"
    environment.write_text("# No cluster modules needed for the preflight check.\n")
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(case / "run.slurm")], cwd=case, text=True,
        capture_output=True, timeout=30,
        env={**os.environ, "ATHENAK_ENV": str(environment),
             "DCI_CASE_DIR": str(case), "BUILD_DIR": str(build),
             "ATHENA_EXE": "", "RUN_DIR": str(run_dir)},
    )
    assert result.returncode != 0
    assert "Default executable is older than source" in result.stderr
    assert "make.slurm" in result.stderr
    assert not run_dir.exists()
