"""Check compact 2T allocations alongside conservative AMR transport.

Use ATHENA_AMR_TEST_EXE and ATHENA_CONDUCTION_TEST_EXE for a bounds-checked
executable. ATHENA_COMPACT_STORAGE_RANKS=2 also exercises remote reflux.
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
def allocation_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("compact-2t-allocation")
    source = directory / "allocations.cpp"
    source.write_text(r"""
#include <cstdio>
#include <cstring>
#include "impl/Kokkos_Profiling_C_Interface.h"

extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle,
                                     const char *label, const void *,
                                     uint64_t size) {
  if (std::strncmp(label, "cond_amr_", 9) == 0 ||
      std::strcmp(label, "rad_amr_diagonal") == 0 ||
      std::strcmp(label, "rad_amr_gradient") == 0 ||
      std::strcmp(label, "cprim") == 0) {
    std::fprintf(stderr, "COMPACT_2T_ALLOCATION %s %llu\n", label,
                 static_cast<unsigned long long>(size));
  }
}
""")
    library = directory / "allocations.so"
    command = [*shlex.split(os.environ.get("HOST_CXX", "g++")),
               "-std=c++17", "-shared", "-fPIC",
               "-I" + str(radiation.REPO / "kokkos/core/src"),
               str(source), "-o", str(library)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return library


def allocations(directory):
    result = {}
    for label, size in re.findall(
            r"COMPACT_2T_ALLOCATION (\S+) (\d+)",
            (directory / "run.log").read_text()):
        result[label] = max(result.get(label, 0), int(size))
    return result


def mesh_flags(dimensions):
    nx, ny, nz = 8, 6 if dimensions > 1 else 1, 4 if dimensions > 2 else 1
    capacity = 64
    flags = (
        f"mesh/nx2={2*ny if dimensions > 1 else 1}",
        f"mesh/nx3={2*nz if dimensions > 2 else 1}",
        f"meshblock/nx2={ny}", f"meshblock/nx3={nz}",
        "mesh_refinement/refinement=adaptive",
        "mesh_refinement/refinement_interval=100",
        f"mesh_refinement/max_nmb_per_rank={capacity}",
    )
    return (nx, ny, nz), capacity, flags


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("halo", ("legacy", "packed", "host"))
def test_radiation_active_cell_storage(tmp_path, monkeypatch, allocation_probe,
                                       dimensions, halo):
    monkeypatch.setenv("KOKKOS_TOOLS_LIBS", str(allocation_probe))
    ranks = int(os.environ.get("ATHENA_COMPACT_STORAGE_RANKS", "1"))
    shape, capacity, flags = mesh_flags(dimensions)
    cell_bytes = capacity*int(np.prod(shape))*np.dtype(np.float64).itemsize
    outputs = []
    for ghosts in (2, 4):
        directory = tmp_path / f"ghosts-{ghosts}"
        output = radiation.run_case(directory, dimensions, ranks=ranks, extra_flags=(
            *flags, f"mesh/nghost={ghosts}",
            f"thermal_radiation/implicit_halo_exchange={halo}",
            "thermal_radiation/implicit_amr_order=2",
            "thermal_radiation/implicit_tolerance=1e-12"))
        radiation.assert_transport(output)
        outputs.append(output)
        sizes = allocations(directory)
        assert sizes["rad_amr_diagonal"] == cell_bytes
        assert sizes["rad_amr_gradient"] == 3*cell_bytes
        assert sizes.get("cprim", 0) <= 128  # Constructor placeholder only.
    radiation.assert_same_solution(*outputs)


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("prolong_primitives", (None, False, True))
def test_conduction_faces_and_optional_coarse_primitives(
        tmp_path, monkeypatch, allocation_probe, dimensions, prolong_primitives):
    monkeypatch.setenv("KOKKOS_TOOLS_LIBS", str(allocation_probe))
    ranks = int(os.environ.get("ATHENA_COMPACT_STORAGE_RANKS", "1"))
    (nx, ny, nz), capacity, flags = mesh_flags(dimensions)
    extra = "" if prolong_primitives is None else (
        "\n<mesh_refinement>\nprolong_primitives = "
        f"{str(prolong_primitives).lower()}\n")
    outputs = []
    for ghosts in (2, 4):
        directory = tmp_path / f"ghosts-{ghosts}"
        output = conduction.run(directory, dimensions,
                                (*flags, f"mesh/nghost={ghosts}"),
                                ranks=ranks, extra=extra)
        conduction.assert_closed(output)
        outputs.append(output)
        sizes = allocations(directory)
        counts = ((nx+1)*ny*nz, nx*(ny+1 if dimensions > 1 else 1)*nz,
                  nx*ny*(nz+1 if dimensions > 2 else 1))
        for field in ("cond_amr_flux", "cond_amr_conductance"):
            for direction, cells in enumerate(counts, 1):
                assert sizes[f"{field}.x{direction}f"] == (
                    capacity*cells*np.dtype(np.float64).itemsize)
        if prolong_primitives:
            coarse_cells = int(np.prod([
                n//2+2*ghosts if n > 1 else 1 for n in (nx, ny, nz)]))
            # This deck has five hydro variables and two component energies.
            assert sizes["cprim"] == capacity*7*coarse_cells*8
        else:
            assert sizes.get("cprim", 0) <= 128
    assert len(outputs[0]) == len(outputs[1])
    for left, right in zip(*outputs):
        np.testing.assert_array_equal(
            radiation.ordered_fields(left)[0], radiation.ordered_fields(right)[0])
        for field in left["mb_data"]:
            np.testing.assert_allclose(conduction.canonical(left, field),
                                       conduction.canonical(right, field),
                                       rtol=3e-10, atol=3e-12)
