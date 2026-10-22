"""Three-dimensional physical boundaries on the local AMR aggregate hierarchy."""

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, assert_same_solution, assert_transport, moments, run_case,
)


@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_local_aggregate_3d_boundaries(tmp_path, boundary):
    # All active block dimensions admit the 3/9 hierarchy. Refining half the
    # domain exercises refluxed aggregate diagonals, while physical boundaries
    # in every direction check their Neumann/Dirichlet/vacuum face weights.
    flags = ["mesh_refinement/max_nmb_per_rank=64", "time/nlim=1",
             "time/tlim=0.0001", "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-12",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "refined_region1/x3max=0.5"]
    for axis in (1, 2, 3):
        flags += [f"mesh/nx{axis}=36", f"meshblock/nx{axis}=18"]
        for side, end in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x{axis}_bc=outflow",
                      f"thermal_radiation/implicit_x{axis}_{end}_boundary={boundary}",
                      f"thermal_radiation/implicit_x{axis}_{end}_value=0.1"]
    reference = run_case(tmp_path / "jacobi", 3, "jacobi", extra_flags=flags)
    candidate = run_case(tmp_path / "aggregate", 3, extra_flags=flags)
    assert_same_solution(reference, candidate)
    initial, final = candidate[0], candidate[-1]
    assert final["cycle"] == 1
    assert set(np.asarray(final["mb_logical"])[:, 3]) == {0, 1}
    assert len(final["mb_logical"]) == 36
    for group in GROUPS:
        values = np.asarray(final["mb_data"][group])
        assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
        if boundary == "neumann":
            np.testing.assert_allclose(moments(final, group)[0],
                                       moments(initial, group)[0], rtol=2e-10, atol=2e-12)


def test_local_aggregate_regrid(tmp_path):
    # The cached aggregate diagonals must be rebuilt for newly refined blocks.
    # This also switches from mixed-level flexible GMRES to equal-level CG.
    flags = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
             "mesh_refinement/max_nmb_per_rank=32", "refined_region1/x1max=0.5",
             "refined_region1/x2max=0.5", "thermal_radiation/implicit_amr_order=2",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", adaptive="refine",
                         extra_flags=flags)
    candidate = run_case(tmp_path / "aggregate", 2, adaptive="refine", extra_flags=flags)
    assert_transport(reference, "refine")
    assert_transport(candidate, "refine")
    assert_same_solution(reference, candidate)
