"""Stiff diffusion with both local aggregate levels and distributed AMR faces."""

import numpy as np
import pytest

from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import composite_diffusion
from test_suite.nr.test_nr_thermal_radiation_amr_preconditioner_mpicpu import (
    monotone_diffusion,
)
from test_suite.nr.thermal_radiation_amr_utils import GROUPS, ordered_fields, run_case


def aggregate_leaf_geometry(output):
    """Independently rasterize the actual 18-cell leaves on the 72-square fine grid."""
    shape = np.asarray(output["mb_data"][GROUPS[0]]).shape
    assert shape[1:] == (1, 18, 18)
    owner = np.full((72, 72), -1, dtype=int)
    widths = np.empty(shape[0]*18*18)
    for block, bounds in enumerate(output["mb_geometry"]):
        dx = (bounds[1]-bounds[0])/18
        np.testing.assert_allclose(dx, (bounds[3]-bounds[2])/18, rtol=0, atol=1e-15)
        first = block*18*18
        widths[first:first+18*18] = dx
        cells = np.arange(first, first+18*18).reshape(18, 18)
        scale = int(round(72*dx))
        assert scale in (1, 2)
        cells = np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1)
        i = int(round(72*(bounds[0]+0.5)))
        j = int(round(72*(bounds[2]+0.5)))
        assert np.all(owner[j:j+18*scale, i:i+18*scale] == -1)
        owner[j:j+18*scale, i:i+18*scale] = cells
    assert np.all(owner >= 0)
    return owner, widths


@pytest.mark.parametrize("amr_order,inner_iterations", ((1, 0), (2, 0), (2, 20)))
def test_stiff_local_aggregate_mpi(tmp_path, amr_order, inner_iterations):
    # Eighteen cells activate both factor3 and factor9 local corrections.
    # A million-fold material jump lies inside a coarse block; dt*K/h^2 reaches
    # 1.3e5. The monotone solve must fit within 40 iterations (the previous
    # block-only correction required 45 on this same equation).
    flags = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
             "problem/dr=1000000", "problem/xshock=-0.125",
             "thermal_radiation/initial_radiation_x1=-0.125",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "thermal_radiation/c_light=300000000", "time/nlim=1", "time/tlim=0.0001",
             "thermal_radiation/implicit_max_iterations=40",
             f"thermal_radiation/implicit_tolerance={1e-13 if amr_order == 1 else 2e-14}",
             f"thermal_radiation/implicit_amr_order={amr_order}",
             f"thermal_radiation/implicit_amr_inner_iterations={inner_iterations}")
    runs = []
    for ranks in (1, 2):
        directory = tmp_path / f"aggregate-ranks{ranks}"
        output = run_case(directory, dimensions=2, ranks=ranks, extra_flags=flags)
        initial, final = output[0], output[-1]
        assert final["cycle"] == 1 and np.isclose(final["time"], 1e-4)
        assert len(final["mb_logical"]) == 10
        assert set(np.asarray(final["mb_logical"])[:, 3]) == {0, 1}
        np.testing.assert_array_equal(initial["mb_logical"], final["mb_logical"])
        assert "AMR fallback" not in (directory / "run.log").read_text()
        owner, widths = aggregate_leaf_geometry(final)
        weights = widths**2
        density = np.asarray(initial["mb_data"]["dens"]).ravel()
        for group, opacity in zip(GROUPS, (100.0, 35.0)):
            before = np.asarray(initial["mb_data"][group]).ravel()
            after = np.asarray(final["mb_data"][group]).ravel()
            assert np.all(np.isfinite(after)) and np.all(after >= 0.)
            coefficient = 3e8/(3.0*density*opacity)
            if amr_order == 1:
                action = monotone_diffusion(after, owner, widths, coefficient)
            else:
                action = composite_diffusion(after, owner, widths, coefficient=coefficient)
            residual = after+1e-4*action-before
            relative = np.sqrt(np.dot(weights, residual**2)/np.dot(weights, before**2))
            assert relative < 1e-7
            np.testing.assert_allclose(np.dot(weights, after), np.dot(weights, before),
                                       rtol=2e-9, atol=2e-12)
        runs.append(ordered_fields(final))
    # Pointwise comparison also checks restriction/prolongation across rank boundaries.
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    for group in GROUPS:
        np.testing.assert_allclose(runs[0][1][group], runs[1][1][group],
                                   rtol=2e-8, atol=2e-9)
    if amr_order == 1:
        reference = run_case(
            tmp_path / "jacobi", dimensions=2, preconditioner="jacobi", ranks=1,
            extra_flags=(*flags, "thermal_radiation/implicit_max_iterations=1000"))
        topology, fields = ordered_fields(reference[-1])
        np.testing.assert_array_equal(topology, runs[0][0])
        for group in GROUPS:
            np.testing.assert_allclose(fields[group], runs[0][1][group],
                                       rtol=2e-8, atol=2e-9)
