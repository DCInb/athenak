"""MPI ownership changes and remote coarse/fine implicit radiation fluxes."""

import os
from pathlib import Path
import subprocess

import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    INPUT, assert_boundary_transport, assert_same_solution, assert_transport,
    boundary_flags, run_case,
)


@pytest.mark.parametrize("dimensions,adaptive", (
    (1, None), (2, None), (3, None), (2, "refine"), (2, "derefine"),
))
def test_amr_decomposition(tmp_path, dimensions, adaptive):
    one = run_case(tmp_path / "one", dimensions, adaptive=adaptive, ranks=1)
    two = run_case(tmp_path / "two", dimensions, adaptive=adaptive, ranks=2)
    assert_transport(one, adaptive)
    assert_transport(two, adaptive)
    assert_same_solution(one, two)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_amr_boundary_decomposition(tmp_path, boundary):
    flags = boundary_flags(boundary) + ["problem/dr=2.0"]
    one = run_case(tmp_path / "one", dimensions=2, ranks=1, extra_flags=flags)
    two = run_case(tmp_path / "two", dimensions=2, ranks=2, extra_flags=flags)
    assert_boundary_transport(one, boundary)
    assert_boundary_transport(two, boundary)
    assert_same_solution(one, two)


def test_amr_flux_limiter(tmp_path):
    # Strong coefficient contrast needs a larger Krylov space than smooth
    # diffusion; retain the same tight true-residual tolerance.
    flags = ["thermal_radiation/flux_limiter=larsen",
             "thermal_radiation/kappa_transport_0=0.01",
             "thermal_radiation/kappa_transport_1=0.02",
             "thermal_radiation/implicit_gmres_restart=100"]
    one = run_case(tmp_path / "one", dimensions=2, ranks=1, extra_flags=flags)
    two = run_case(tmp_path / "two", dimensions=2, ranks=2, extra_flags=flags)
    assert_transport(one)
    assert_transport(two)
    assert_same_solution(one, two)


def test_derefinement_rejects_empty_ranks(tmp_path):
    # Four root blocks plus one refined root produce five initial leaves. After
    # two transport steps, derefinement would leave only four leaves on five ranks.
    executable = Path(os.environ.get("ATHENA_AMR_TEST_EXE", "./athena")).resolve()
    command = ["mpirun", "-np", "5", str(executable), "-i", str(INPUT),
               "mesh_refinement/refinement=adaptive"]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True,
                            text=True, timeout=60, check=False, env=environment)
    output = result.stdout + result.stderr
    (tmp_path / "run.log").write_text(output)
    assert result.returncode != 0, output
    assert "Total number of MeshBlocks = 5" in output, output
    assert ("AMR would leave fewer MeshBlocks (nmb_total=4) than MPI ranks "
            "(nranks=5). Each rank must own at least one MeshBlock.") in output, output
    assert "MPI_ERR_TAG" not in output, output
