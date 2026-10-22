"""Inner stopping norms leave the final physical AMR equation unchanged."""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr import test_nr_thermal_radiation_amr_aggregate_mpicpu as aggregate
from test_suite.nr import test_nr_thermal_radiation_amr_cholmod_mpicpu as cholmod
from test_suite.nr.test_nr_thermal_radiation_amr_accuracy_cpu import (
    _manufactured, composite_diffusion, leaf_geometry, manufactured_run,
)
from test_suite.nr.thermal_radiation_amr_utils import INPUT, ordered_fields


@pytest.mark.parametrize("preconditioner,metric,dimensions,inner", (
    ("jacobi", "volume", 2, 1), ("jacobi", "preconditioner", 2, 20),
    ("block-coarse", "volume", 3, 20), ("block-coarse", "preconditioner", 2, 20),
))
def test_inner_preconditioner_norm_manufactured_mpi(
        tmp_path, preconditioner, metric, dimensions, inner):
    # Cover both outer metrics, capped inner solves, mixed transverse terms and
    # varying positive coefficients. The oracle is the physical equation, not
    # either recursive stopping norm used inside the solver.
    variable = dimensions == 2
    results = []
    for ranks in (1, 2):
        directory = tmp_path / f"ranks{ranks}"
        output = manufactured_run(
            directory, dimensions=dimensions, variable_coefficient=variable, ranks=ranks,
            extra_flags=(f"thermal_radiation/implicit_preconditioner={preconditioner}",
                         f"thermal_radiation/implicit_gmres_metric={metric}",
                         "thermal_radiation/implicit_gmres_restart=3",
                         f"thermal_radiation/implicit_amr_inner_iterations={inner}",
                         "thermal_radiation/implicit_amr_inner_norm=preconditioner"))
        owner, xyz, widths = leaf_geometry(output, 16, dimensions)
        _, rhs = _manufactured(*xyz, widths, dimensions, 0.01, variable)
        actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()
        coefficient = 1+0.25*np.cos(2*np.pi*xyz[1]) if variable else np.ones_like(actual)
        residual = actual+0.01*composite_diffusion(
            actual, owner, widths, coefficient=coefficient)-rhs
        assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
        np.testing.assert_allclose(residual, 0, rtol=0, atol=3e-10)
        np.testing.assert_allclose(np.dot(widths**dimensions, actual),
                                   np.dot(widths**dimensions, rhs), rtol=0, atol=2e-11)
        log = (directory / "solve/run.log").read_text()
        assert "AMR fallback" not in log
        finished = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                    for line in log.splitlines() if "GMRES finished:" in line]
        assert any(row.get("inner_norm") == "preconditioner" and
                   row.get("metric") == metric and row.get("status") == "converged"
                   for row in finished)
        if inner == 1:
            assert max(map(int, re.findall(r"inner_cap_hits=(\d+)", log))) > 0
        results.append(ordered_fields(output))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1]["erad00_d"], results[1][1]["erad00_d"],
                               rtol=0, atol=3e-10)


@pytest.mark.parametrize("preconditioner,metric", (
    ("block-coarse", "volume"), ("block-coarse", "preconditioner"),
    ("amr-aggregate", "volume"), ("amr-aggregate", "preconditioner"),
))
def test_inner_preconditioner_norm_stiff_mpi(tmp_path, monkeypatch, preconditioner, metric):
    if preconditioner == "amr-aggregate":
        executable = os.environ.get("ATHENA_CHOLMOD_TEST_EXE")
        if not executable:
            pytest.skip("Requires ATHENA_CHOLMOD_TEST_EXE")
        monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)
        module = cholmod
    else:
        module = aggregate
    original = module.run_case

    def with_inner_norm(*args, **kwargs):
        kwargs["extra_flags"] = (*kwargs.get("extra_flags", ()),
                                 "thermal_radiation/implicit_amr_inner_norm=preconditioner",
                                 f"thermal_radiation/implicit_gmres_metric={metric}",
                                 "thermal_radiation/implicit_max_iterations=100")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_case", with_inner_norm)
    if preconditioner == "amr-aggregate":
        module.test_global_aggregate_stiff_mpi(tmp_path, 2, 20)
    else:
        module.test_stiff_local_aggregate_mpi(tmp_path, 2, 20)


@pytest.mark.parametrize("metric", ("volume", "preconditioner"))
def test_inner_preconditioner_norm_positivity_fallback(tmp_path, monkeypatch, metric):
    executable = os.environ.get("ATHENA_CHOLMOD_TEST_EXE")
    if not executable:
        pytest.skip("Requires ATHENA_CHOLMOD_TEST_EXE")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)
    original = cholmod._run

    def with_inner_norm(directory, arguments, ranks=None):
        return original(directory, [*arguments,
                                    "thermal_radiation/implicit_amr_inner_norm=preconditioner",
                                    f"thermal_radiation/implicit_gmres_metric={metric}"], ranks)

    monkeypatch.setattr(cholmod, "_run", with_inner_norm)
    cholmod.test_global_aggregate_positivity_fallback(tmp_path)


@pytest.mark.parametrize("flags,message", (
    (("thermal_radiation/implicit_amr_inner_norm=unknown",),
     "implicit_amr_inner_norm must be volume or preconditioner"),
    (("thermal_radiation/implicit_amr_inner_norm=preconditioner",
      "mesh_refinement/refinement=none"),
     "requires implicit transport on an SMR/AMR mesh"),
    (("thermal_radiation/implicit_amr_inner_norm=preconditioner",
      "thermal_radiation/transport_integrator=explicit"),
     "requires implicit transport on an SMR/AMR mesh"),
))
def test_invalid_inner_norm(tmp_path, flags, message):
    executable = Path(os.environ.get("ATHENA_AMR_TEST_EXE", "./athena")).resolve()
    result = subprocess.run([str(executable), "-i", str(INPUT), "time/nlim=0", *flags],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert message in result.stdout+result.stderr
