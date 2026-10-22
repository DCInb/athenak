"""Coarse topology reuse and host/device solves against independent transport oracles."""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr import thermal_radiation_amr_utils as utils
from test_suite.nr.thermal_radiation_amr_utils import (
    GROUPS, INPUT, assert_1d_reference, assert_same_solution, moments,
    ordered_fields, run_case,
)


@pytest.fixture(params=("cpu-host", "cuda-device"))
def coarse_backend(request, tmp_path, monkeypatch):
    variable = ("ATHENA_CUDA_COARSE_TEST_EXE" if request.param == "cuda-device"
                else "ATHENA_CHOLMOD_TEST_EXE")
    executable = os.environ.get(variable)
    if not executable:
        pytest.skip(f"Requires {variable}")
    monkeypatch.setenv("ATHENA_AMR_TEST_EXE", executable)
    # Two small CPU ranks, or two ranks sharing one GPU, fit within the test job.
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    input_file = tmp_path / "coarse.athinput"
    input_file.write_text(INPUT.read_text() +
                          "\n<thermal_radiation>\nsolver_timers = true\n")
    monkeypatch.setattr(utils, "INPUT", input_file)
    return "device" if request.param == "cuda-device" else "host"


def _backend_flag(backend):
    return f"thermal_radiation/implicit_amr_coarse_solver={backend}"


def _check_backend(directory, backend, ranks=2):
    log = (directory / "run.log").read_text()
    assert "AMR fallback" not in log
    if backend == "device":
        assert "phase=aggregate_device_solve " in log
        assert "phase=cholmod_solve " not in log
        if ranks == 1:
            assert "phase=aggregate_rhs_d2h " not in log
            assert "phase=aggregate_correction_h2d " not in log
    else:
        assert "phase=cholmod_solve " in log
        assert "phase=aggregate_device_solve " not in log
    return log


def test_coarse_cache_same_count_regrid(tmp_path, coarse_backend):
    # Initially the right half is refined. A stationary density contact asks AMR
    # to refine the left half and derefine the right half at cycle 2. Both global
    # and two-rank local leaf counts remain unchanged: count-only cache keys fail.
    flags = ("mesh/nx1=36", "mesh/nx2=36", "meshblock/nx1=18", "meshblock/nx2=18",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "mesh_refinement/refinement=adaptive", "mesh_refinement/refinement_interval=2",
             "mesh_refinement/max_nmb_per_rank=32", "amr_criterion0/value_max=1.5",
             "problem/dl=2", "problem/dr=1", "problem/xshock=0",
             "thermal_radiation/implicit_amr_inner_iterations=20",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", extra_flags=flags)
    candidates = []
    for ranks in (1, 2):
        directory = tmp_path / f"aggregate-{ranks}"
        outputs = run_case(directory, 2, "amr-aggregate", ranks=ranks,
                           extra_flags=(*flags, _backend_flag(coarse_backend)))
        assert_same_solution(reference, outputs)
        assert all(len(out["mb_logical"]) == 10 for out in outputs)
        initial, final = ordered_fields(outputs[0])[0], ordered_fields(outputs[-1])[0]
        assert outputs[-1]["cycle"] == 3
        assert not np.array_equal(initial, final)
        # Bin outputs are written before regrid. Cycle 3 must advance radiation on
        # the new topology, so passing by inspecting only the AMR operation is impossible.
        cycle_two = next(out for out in outputs if out["cycle"] == 2)
        np.testing.assert_array_equal(ordered_fields(cycle_two)[0], initial)
        assert np.isclose(outputs[-1]["time"], 7e-4, rtol=0, atol=2e-15)
        for group in GROUPS:
            assert np.all(np.asarray(outputs[-1]["mb_data"][group]) >= 0)
            np.testing.assert_allclose(moments(outputs[-1], group)[0],
                                       moments(outputs[0], group)[0],
                                       rtol=2e-10, atol=2e-12)
        log = _check_backend(directory, coarse_backend, ranks)
        if ranks == 1:
            cache = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                     for line in log.splitlines()
                     if line.startswith("# radiation aggregate cache:")]
            # Two groups per step: build once initially and after the same-count
            # regrid, reuse for the other groups. Count alone cannot pass this.
            assert [int(row["topology_rebuilt"]) for row in cache] == [1, 0, 0, 0, 1, 0]
            expected_backend = "device_dense" if coarse_backend == "device" else "host"
            assert {row["backend"] for row in cache} == {expected_backend}
            assert len({row["nodes"] for row in cache}) == 1
        candidates.append(outputs)
    assert_same_solution(candidates[0], candidates[1])


def test_coarse_periodic_spd_dense_reference(tmp_path, coarse_backend):
    # The periodic diffusion graph has a constant null mode; adding the positive
    # cell-volume mass makes the implicit matrix SPD. The oracle assembles and
    # solves the actual nonuniform-cell equation independently with NumPy.
    directory = tmp_path / "periodic"
    outputs = run_case(directory, 1, "amr-aggregate", ranks=2, extra_flags=(
        "mesh/nx1=36", "meshblock/nx1=18", "refined_region1/x1max=0.5",
        "thermal_radiation/implicit_amr_order=1",
        "thermal_radiation/implicit_tolerance=1e-12", _backend_flag(coarse_backend)))
    assert_1d_reference(outputs, "periodic")
    for group in GROUPS:
        np.testing.assert_allclose(moments(outputs[-1], group)[0],
                                   moments(outputs[0], group)[0], rtol=2e-10)
    _check_backend(directory, coarse_backend)


def test_coarse_periodic_self_neighbor(tmp_path, coarse_backend):
    # One root block spans x2: periodic faces can share a block gid while joining
    # distinct aggregates. Reusing a compact graph must retain those edges once.
    flags = ("mesh/nx1=36", "mesh/nx2=18", "meshblock/nx1=18", "meshblock/nx2=18",
             "refined_region1/x1max=0.5", "refined_region1/x2max=0.5",
             "time/nlim=1", "time/tlim=0.0001", "problem/shock_dir=2",
             "problem/xshock=-0.125", "problem/dr=100",
             "thermal_radiation/implicit_tolerance=1e-12")
    reference = run_case(tmp_path / "jacobi", 2, "jacobi", extra_flags=flags)
    directory = tmp_path / "aggregate"
    candidate = run_case(directory, 2, "amr-aggregate", ranks=2,
                         extra_flags=(*flags, _backend_flag(coarse_backend)))
    assert_same_solution(reference, candidate)
    for group in GROUPS:
        np.testing.assert_allclose(moments(candidate[-1], group)[0],
                                   moments(candidate[0], group)[0], rtol=2e-10)
    _check_backend(directory, coarse_backend)


def test_cpu_rejects_explicit_device_coarse_solver(tmp_path):
    executable = os.environ.get("ATHENA_CHOLMOD_TEST_EXE")
    if not executable:
        pytest.skip("Requires a CPU CHOLMOD build in ATHENA_CHOLMOD_TEST_EXE")
    result = subprocess.run([str(Path(executable).resolve()), "-i", str(INPUT),
                             "mesh/nx1=36", "meshblock/nx1=18",
                             "thermal_radiation/implicit_preconditioner=amr-aggregate",
                             "thermal_radiation/implicit_amr_coarse_solver=device",
                             "time/nlim=0"], cwd=tmp_path, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode != 0
    assert "implicit_amr_coarse_solver=device requires" in result.stdout+result.stderr
