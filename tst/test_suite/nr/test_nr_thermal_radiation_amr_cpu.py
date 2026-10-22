"""Implicit radiation diffusion on static and changing coarse/fine meshes."""

import re

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_1d_reference, assert_boundary_transport, assert_same_solution,
    assert_transport, boundary_flags, moments, read_binary, run_case,
)
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import _run


@pytest.mark.parametrize("amr_order", (1, 2))
@pytest.mark.parametrize("dimensions", (1, 2, 3))
def test_static_amr(tmp_path, dimensions, amr_order):
    # Different preconditioners reach the permitted true-residual bound at
    # different iterations. Solve more tightly for this pointwise comparison.
    flags = (f"thermal_radiation/implicit_amr_order={amr_order}",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", dimensions, "jacobi", extra_flags=flags)
    candidate = run_case(tmp_path / "block_coarse", dimensions, extra_flags=flags)
    assert_transport(reference)
    assert_transport(candidate)
    assert_same_solution(reference, candidate)
    if dimensions == 1 and amr_order == 1:
        assert_1d_reference(candidate)


@pytest.mark.parametrize("level", (0, 1))
def test_equal_level_amr_uses_cg(tmp_path, level):
    # AMR-enabled root-only and fully refined meshes have an identical symmetric
    # operator at both configured orders, and must take the same CG iterations.
    text = INPUT.read_text()
    if level == 0:
        text = re.sub(r"<refined_region1>\n.*?(?=\n<)", "", text, flags=re.S)
    input_file = tmp_path / "equal_level.athinput"
    input_file.write_text(text)
    results, iterations = [], []
    for order in (1, 2):
        directory = tmp_path / f"order{order}"
        flags = ["mesh/nx2=16", "meshblock/nx2=8",
                 "thermal_radiation/implicit_tolerance=1e-12",
                 f"thermal_radiation/implicit_amr_order={order}",
                 "thermal_radiation/implicit_gmres_restart=2"]
        if level == 1:
            flags += [f"refined_region1/x{axis}{side}={value}"
                      for axis in (1, 2, 3)
                      for side, value in (("min", -0.5), ("max", 0.5))]
        _run(directory, ["-i", str(input_file), *flags])
        outputs = [read_binary(str(path))
                   for path in sorted((directory / "bin").glob("*.bin"))]
        assert outputs[-1]["cycle"] == 3
        assert set(np.asarray(outputs[-1]["mb_logical"])[:, 3]) == {level}
        assert len(outputs[-1]["mb_logical"]) == 8 * 4**level
        for group in GROUPS:
            assert np.min(outputs[-1]["mb_data"][group]) >= 0
            np.testing.assert_allclose(moments(outputs[-1], group)[0],
                                       moments(outputs[0], group)[0], rtol=2e-11)
        iterations.append(re.findall(
            r"^# implicit thermal radiation:.* max_iterations=(\d+)",
            (directory / "run.log").read_text(), re.MULTILINE))
        results.append(outputs)
    assert len(iterations[0]) == 3
    assert iterations[0] == iterations[1]
    assert_same_solution(*results)
    for left, right in zip(*results):
        for group in GROUPS:
            np.testing.assert_array_equal(left["mb_data"][group], right["mb_data"][group])


@pytest.mark.parametrize("adaptive", ("refine", "derefine"))
def test_dynamic_amr(tmp_path, adaptive):
    outputs = run_case(tmp_path / adaptive, dimensions=2, adaptive=adaptive)
    assert_transport(outputs, adaptive)


@pytest.mark.parametrize("amr_order", (1, 2))
def test_amr_block_hierarchy_fallback(tmp_path, amr_order):
    # An 18-cell block supports the uniform solver's block hierarchy. Mixed leaf
    # levels must still use the AMR preconditioner and conservative face operator.
    flags = ("mesh/nx1=72", "meshblock/nx1=18",
             f"thermal_radiation/implicit_amr_order={amr_order}")
    reference = run_case(tmp_path / "jacobi", preconditioner="jacobi",
                         extra_flags=flags)
    candidate = run_case(tmp_path / "block_coarse", extra_flags=flags)
    assert_transport(reference)
    assert_transport(candidate)
    assert_same_solution(reference, candidate)
    if amr_order == 1:
        assert_1d_reference(candidate)


@pytest.mark.parametrize("amr_order", (1, 2))
@pytest.mark.parametrize("dimensions", (1, 2))
def test_density_jump(tmp_path, dimensions, amr_order):
    # Constant pressure and zero velocity leave a stationary density contact.
    # Radiation energy density is conserved despite the jump in D=c/(3*rho*kappa).
    flags = ("problem/dr=2.0", f"thermal_radiation/implicit_amr_order={amr_order}")
    reference = run_case(tmp_path / "jacobi", dimensions, "jacobi",
                         extra_flags=flags)
    candidate = run_case(tmp_path / "block_coarse", dimensions, extra_flags=flags)
    assert_transport(reference)
    assert_transport(candidate)
    assert_same_solution(reference, candidate)
    if dimensions == 1 and amr_order == 1:
        assert_1d_reference(candidate)


@pytest.mark.parametrize("amr_order", (1, 2))
@pytest.mark.parametrize("boundary", ("neumann", "dirichlet", "vacuum"))
def test_physical_boundaries(tmp_path, boundary, amr_order):
    flags = boundary_flags(boundary) + [
        f"thermal_radiation/implicit_amr_order={amr_order}"]
    reference = run_case(tmp_path / "jacobi", preconditioner="jacobi",
                         extra_flags=flags)
    candidate = run_case(tmp_path / "block_coarse", extra_flags=flags)
    assert_boundary_transport(reference, boundary)
    assert_boundary_transport(candidate, boundary)
    assert_same_solution(reference, candidate)
    if amr_order == 1:
        assert_1d_reference(candidate, boundary)
