"""Packed scalar halos and flux reflux preserve the legacy MPI/AMR operator."""

import os

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import GROUPS, ordered_fields, run_case


def assert_same_fields(reference, candidate):
    assert len(reference) == len(candidate)
    for before, after in zip(reference, candidate):
        assert before['cycle'] == after['cycle']
        assert before['time'] == after['time']
        old_topology, old_fields = ordered_fields(before)
        new_topology, new_fields = ordered_fields(after)
        np.testing.assert_array_equal(old_topology, new_topology)
        for group in GROUPS:
            assert np.isfinite(old_fields[group]).all(), group
            assert np.isfinite(new_fields[group]).all(), group
            if os.environ.get('ATHENA_HALO_TEST_GPU_ROUNDOFF') == '1':
                # Unchanged CUDA runs vary in their last bits through parallel
                # reductions. This gate stays far below the 1e-11 solve tolerance.
                roundoff = 64*np.finfo(np.float64).eps
                np.testing.assert_allclose(old_fields[group], new_fields[group],
                                           rtol=roundoff, atol=roundoff)
            else:
                np.testing.assert_array_equal(old_fields[group].view(np.uint64),
                                              new_fields[group].view(np.uint64))


@pytest.mark.parametrize("mode,faces", (("packed", False), ("packed", True),
                                       ("host", False), ("host", True)))
@pytest.mark.parametrize("dimensions,ranks", ((1, 2), (2, 3), (3, 2)))
def test_packed_halo_matches_legacy(tmp_path, mode, faces, dimensions, ranks):
    # Inner PCG uses a face stencil; the outer quadratic operator still requires
    # edge/corner donors. Periodic neighbors may share gids but have distinct slots.
    flags = ("thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-11")
    reference = run_case(tmp_path / "legacy", dimensions, ranks=ranks,
                         extra_flags=(*flags, "thermal_radiation/implicit_halo_exchange=legacy"))
    candidate = run_case(tmp_path / "candidate", dimensions, ranks=ranks, extra_flags=(
        *flags, f"thermal_radiation/implicit_halo_exchange={mode}",
        f"thermal_radiation/implicit_halo_faces={str(faces).lower()}"))
    assert_same_fields(reference, candidate)


@pytest.mark.parametrize("mode", ("packed", "host"))
@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_packed_halo_regrid(tmp_path, mode, adaptive):
    flags = ("thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-11")
    reference = run_case(tmp_path / "legacy", 2, ranks=2, adaptive=adaptive,
                         extra_flags=(*flags, "thermal_radiation/implicit_halo_exchange=legacy"))
    candidate = run_case(tmp_path / "candidate", 2, ranks=2, adaptive=adaptive,
                         extra_flags=(*flags, f"thermal_radiation/implicit_halo_exchange={mode}"))
    assert_same_fields(reference, candidate)
