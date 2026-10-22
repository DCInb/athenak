"""Compare compact solver halos with legacy fields and actual MPI payloads.

Set ATHENA_AMR_TEST_EXE and ATHENA_CONDUCTION_TEST_EXE to a built-in-pgen
executable. ATHENA_HALO_TEST_GPU_ROUNDOFF=1 permits parallel CUDA roundoff.
The PMPI probe counts all nonblocking point-to-point sends, including unchanged
hydro traffic, so a reduction cannot come merely from a smaller allocated view.
"""

import os
import re
import shlex
import subprocess

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as radiation
from test_suite.nr import test_nr_conduction_amr_cpu as conduction


@pytest.fixture(scope="module")
def mpi_payload_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scalar-halo-mpi")
    source = directory / "payload.cpp"
    source.write_text(r"""
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <atomic>
#include <cstdio>
static std::atomic<unsigned long long> bytes{0}, messages{0};
extern "C" int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm, MPI_Request *request) {
  int size = 0;
  PMPI_Type_size(datatype, &size);
  bytes += static_cast<unsigned long long>(count)*size;
  ++messages;
  return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}
extern "C" int MPI_Finalize() {
  int rank = -1;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::fprintf(stderr, "SOLVER_HALO_MPI rank=%d bytes=%llu messages=%llu\n",
               rank, bytes.load(), messages.load());
  return PMPI_Finalize();
}
""")
    library = directory / "payload.so"
    command = [*shlex.split(os.environ.get("MPICXX", "mpicxx")), "-std=c++17",
               "-shared", "-fPIC", str(source), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return library


def payload(directory):
    matches = re.findall(
        r"SOLVER_HALO_MPI rank=(\d+) bytes=(\d+) messages=(\d+)",
        (directory / "run.log").read_text())
    assert sorted(int(rank) for rank, _, _ in matches) == [0, 1]
    return {name: sum(int(row[column]) for row in matches)
            for name, column in (("bytes", 1), ("messages", 2))}


def compare_fields(before, after):
    assert len(before) == len(after)
    for left, right in zip(before, after):
        assert left["cycle"] == right["cycle"] and left["time"] == right["time"]
        topology_a, fields_a = radiation.ordered_fields(left)
        topology_b, fields_b = radiation.ordered_fields(right)
        np.testing.assert_array_equal(topology_a, topology_b)
        assert fields_a.keys() == fields_b.keys()
        for name in fields_a:
            a, b = fields_a[name], fields_b[name]
            assert np.isfinite(a).all() and np.isfinite(b).all(), name
            if os.environ.get("ATHENA_HALO_TEST_GPU_ROUNDOFF") == "1":
                roundoff = 64*np.finfo(np.float64).eps
                np.testing.assert_allclose(a, b, rtol=roundoff, atol=roundoff,
                                           err_msg=name)
            else:
                np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64),
                                              err_msg=name)


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("mode", ("packed", "host"))
@pytest.mark.parametrize("order", (1, 2))
def test_radiation_compact_payload(tmp_path, monkeypatch, mpi_payload_probe,
                                   dimensions, mode, order):
    monkeypatch.setenv("LD_PRELOAD", ":".join(filter(None, (
        str(mpi_payload_probe), os.environ.get("LD_PRELOAD", "")))))
    flags = ("mesh/nghost=4", f"thermal_radiation/implicit_amr_order={order}",
             "thermal_radiation/implicit_amr_inner_iterations=0",
             "thermal_radiation/implicit_tolerance=1e-12")
    before = radiation.run_case(tmp_path / "legacy", dimensions, ranks=2, extra_flags=(
        *flags, "thermal_radiation/implicit_halo_exchange=legacy"))
    after = radiation.run_case(tmp_path / "compact", dimensions, ranks=2, extra_flags=(
        *flags, f"thermal_radiation/implicit_halo_exchange={mode}"))
    radiation.assert_transport(after)
    compare_fields(before, after)
    old, new = payload(tmp_path / "legacy"), payload(tmp_path / "compact")
    assert new["bytes"] < old["bytes"], (old, new)
    assert new["messages"] <= old["messages"], (old, new)
    (tmp_path / "payload.txt").write_text(f"legacy={old}\ncompact={new}\n")


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("mode", ("packed", "host"))
def test_conduction_compact_payload(tmp_path, monkeypatch, mpi_payload_probe,
                                    dimensions, mode):
    monkeypatch.setenv("LD_PRELOAD", ":".join(filter(None, (
        str(mpi_payload_probe), os.environ.get("LD_PRELOAD", "")))))
    before = conduction.run(tmp_path / "legacy", dimensions, ranks=2, flags=(
        "mesh/nghost=4", "hydro/conduction_halo_exchange=legacy"))
    after = conduction.run(tmp_path / "compact", dimensions, ranks=2, flags=(
        "mesh/nghost=4", f"hydro/conduction_halo_exchange={mode}"))
    conduction.assert_closed(after)
    compare_fields(before, after)
    old, new = payload(tmp_path / "legacy"), payload(tmp_path / "compact")
    assert new["bytes"] < old["bytes"], (old, new)
    assert new["messages"] <= old["messages"], (old, new)
    (tmp_path / "payload.txt").write_text(f"legacy={old}\ncompact={new}\n")


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
@pytest.mark.parametrize("physics", ("radiation", "conduction"))
def test_compact_halo_regrid(tmp_path, adaptive, physics):
    if physics == "radiation":
        before = radiation.run_case(tmp_path / "legacy", 2, ranks=2, adaptive=adaptive,
                                     extra_flags=("mesh/nghost=4",
                                         "thermal_radiation/implicit_halo_exchange=legacy"))
        after = radiation.run_case(tmp_path / "compact", 2, ranks=2, adaptive=adaptive,
                                    extra_flags=("mesh/nghost=4",))
        radiation.assert_transport(after, adaptive)
    else:
        flags = ("mesh/nghost=4", "mesh_refinement/refinement=adaptive",
                 f"amr_criterion0/value_max={0.5 if adaptive == 'refine' else 2.0}")
        before = conduction.run(tmp_path / "legacy", 2, ranks=2, flags=(
            *flags, "hydro/conduction_halo_exchange=legacy"))
        after = conduction.run(tmp_path / "compact", 2, ranks=2, flags=flags)
        conduction.assert_closed(after)
        assert not np.array_equal(radiation.ordered_fields(after[0])[0],
                                  radiation.ordered_fields(after[-1])[0])
    compare_fields(before, after)


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("mode", ("packed", "host"))
@pytest.mark.parametrize("physics", ("radiation", "conduction"))
def test_uniform_compact_payload(tmp_path, monkeypatch, mpi_payload_probe,
                                dimensions, mode, physics):
    """Uniform meshes skip coarse storage and still need physical face ghosts."""
    monkeypatch.setenv("LD_PRELOAD", ":".join(filter(None, (
        str(mpi_payload_probe), os.environ.get("LD_PRELOAD", "")))))
    flags = ("mesh/nghost=4", "mesh_refinement/refinement=none")
    if physics == "radiation":
        def run(directory, halo):
            return radiation.run_case(directory, dimensions, ranks=2, extra_flags=(
                *flags, f"thermal_radiation/implicit_halo_exchange={halo}"))
        fields = radiation.GROUPS
        integral = lambda output, field: radiation.moments(output, field)[0]
    else:
        def run(directory, halo):
            return conduction.run(directory, dimensions, ranks=2, flags=(
                *flags, f"hydro/conduction_halo_exchange={halo}"))
        fields = ("eele_d", "eion_d", "ener")
        integral = conduction.integral
    before = run(tmp_path / "legacy", "legacy")
    after = run(tmp_path / "compact", mode)
    compare_fields(before, after)
    for field in fields:
        for output in after[1:]:
            np.testing.assert_allclose(integral(output, field), integral(after[0], field),
                                       rtol=2e-11, atol=2e-13)
    old, new = payload(tmp_path / "legacy"), payload(tmp_path / "compact")
    (tmp_path / "payload.txt").write_text(f"legacy={old}\ncompact={new}\n")
    assert new["bytes"] < old["bytes"], (old, new)
    assert new["messages"] <= old["messages"], (old, new)
