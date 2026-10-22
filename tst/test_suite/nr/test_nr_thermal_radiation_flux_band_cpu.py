"""Compact implicit faces preserve transport and use no ghost-region storage.

Select a bounds-checked executable with ATHENA_AMR_TEST_EXE. Set
ATHENA_RADIATION_FACE_RANKS=2 to exercise remote coarse/fine exchanges.
"""

import os
import re
import shlex
import subprocess

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    REPO, assert_same_solution, assert_transport, run_case,
)


@pytest.fixture(scope="module")
def allocation_probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("radiation-face-allocation")
    source = directory / "allocations.cpp"
    source.write_text(r"""
#include <cstdio>
#include <cstring>
#include "impl/Kokkos_Profiling_C_Interface.h"

extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle,
                                     const char *label, const void *,
                                     uint64_t size) {
  if (std::strncmp(label, "rad_amr_flux.", 13) == 0 ||
      std::strncmp(label, "rad_amr_conductance.", 20) == 0) {
    std::fprintf(stderr, "RADIATION_FACE_ALLOCATION %s %llu\n", label,
                 static_cast<unsigned long long>(size));
  }
}
""")
    library = directory / "allocations.so"
    command = [*shlex.split(os.environ.get("HOST_CXX", "g++")),
               "-std=c++17", "-shared", "-fPIC",
               "-I" + str(REPO / "kokkos/core/src"), str(source), "-o", str(library)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return library


@pytest.mark.parametrize("dimensions", (1, 2, 3))
@pytest.mark.parametrize("halo", ("legacy", "packed", "host"))
def test_compact_faces_ignore_ghost_width(tmp_path, monkeypatch, allocation_probe,
                                        dimensions, halo):
    monkeypatch.setenv("KOKKOS_TOOLS_LIBS", str(allocation_probe))
    ranks = int(os.environ.get("ATHENA_RADIATION_FACE_RANKS", "1"))
    nx, ny, nz = 8, 6 if dimensions > 1 else 1, 4 if dimensions > 2 else 1
    capacity = 64
    flags = (
        f"mesh/nx2={2*ny if dimensions > 1 else 1}",
        f"mesh/nx3={2*nz if dimensions > 2 else 1}",
        f"meshblock/nx2={ny}", f"meshblock/nx3={nz}",
        # Reserve AMR capacity while keeping the seeded topology for these three steps.
        "mesh_refinement/refinement=adaptive",
        "mesh_refinement/refinement_interval=100",
        f"mesh_refinement/max_nmb_per_rank={capacity}",
        f"thermal_radiation/implicit_halo_exchange={halo}",
        "thermal_radiation/implicit_amr_order=2",
        "thermal_radiation/implicit_tolerance=1e-12",
    )
    outputs = []
    for ghosts in (2, 4):
        directory = tmp_path / f"ghosts-{ghosts}"
        output = run_case(directory, dimensions, ranks=ranks,
                          extra_flags=(*flags, f"mesh/nghost={ghosts}"))
        assert_transport(output)
        outputs.append(output)
        allocations = {}
        for label, size in re.findall(
                r"RADIATION_FACE_ALLOCATION (\S+) (\d+)",
                (directory / "run.log").read_text()):
            allocations.setdefault(label, set()).add(int(size))
        counts = ((nx+1)*ny*nz, nx*(ny+1 if dimensions > 1 else 1)*nz,
                  nx*ny*(nz+1 if dimensions > 2 else 1))
        for field in ("rad_amr_flux", "rad_amr_conductance"):
            for direction, cells in enumerate(counts, 1):
                # Kokkos may report a small allocation for the constructor's
                # zero-extent placeholder before allocating the actual face field.
                assert max(allocations[f"{field}.x{direction}f"]) == (
                    capacity*cells*np.dtype(np.float64).itemsize)
    assert_same_solution(*outputs)
