"""Resolve faint cells after stiff diffusion reaches the Krylov rounding floor."""

import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_suite.nr.thermal_radiation_amr_utils import INPUT, read_binary


@pytest.mark.parametrize("ranks", (1, 2, 4))
@pytest.mark.parametrize("stiffness", (1e4, 1e6, 1e12, 1e14))
def test_stiff_and_faint_components(tmp_path, monkeypatch, ranks, stiffness):
    executable = os.environ.get("ATHENA_NATIVE_TEST_EXE")
    if not executable:
        pytest.skip("Set ATHENA_NATIVE_TEST_EXE to a built_in_pgens binary")
    executable = str(Path(executable).resolve())
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    count, block_size, ghosts, dt = 4096, 64, 2, 1e-20
    dx = 1.0 / count
    input_file = tmp_path / "refinement.athinput"
    input_file.write_text(INPUT.read_text() +
                          "\n<output2>\nfile_type = rst\ndt = -1\ndcycle = 1\n")
    # A stationary contact sets K through rho, with constant gas pressure.
    # Fully refined leaves retain the AMR operator while admitting an exact
    # one-dimensional tridiagonal reference. The very short physical step
    # prevents hydrodynamic evolution; dt*K/dx^2 has the prescribed stiffness.
    flags = [f"mesh/nx1={count // 2}", f"meshblock/nx1={block_size}",
             "mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
             "refined_region1/x1min=-0.5", "refined_region1/x1max=0.5",
             "mesh_refinement/max_nmb_per_rank=256", "time/nlim=0",
             f"time/initial_dt={dt}", f"time/tlim={dt}",
             "time/align_outputs=false", "problem/pl=1e20", "problem/pr=1e20",
             "thermal_radiation/n_groups=1", "thermal_radiation/group_bound_1=100",
             "thermal_radiation/initial_profile=uniform",
             "thermal_radiation/kappa_transport_0=1",
             f"thermal_radiation/c_light={3 * stiffness * dx * dx / dt}",
             "thermal_radiation/implicit_preconditioner=native-mg",
             "thermal_radiation/implicit_amr_order=1",
             "thermal_radiation/implicit_tolerance=1e-10",
             "thermal_radiation/implicit_max_iterations=2000"]

    def run(name, arguments, processes):
        directory = tmp_path / name
        directory.mkdir()
        result = subprocess.run(
            ["mpirun", "--oversubscribe", "-np", str(processes), executable, *arguments],
            cwd=directory, capture_output=True, text=True, timeout=240)
        (directory / "run.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        paths = sorted((directory / "bin").glob("*.bin"))
        assert paths
        return read_binary(str(paths[-1]))

    geometry = run("setup", ["-i", str(input_file), *flags], 1)
    assert geometry["n_mbs"] * block_size == count
    checkpoint = next((tmp_path / "setup/rst").glob("*.rst"))
    payload = bytearray(checkpoint.read_bytes())
    extent = block_size + 2 * ghosts
    shape = (geometry["n_mbs"], 8, 1, 1, extent)
    per_block = int(np.prod(shape[1:]) * 8)
    start = len(payload) - shape[0] * per_block
    # Refuse a different checkpoint layout instead of modifying another field.
    assert np.frombuffer(payload, dtype=np.uint64, count=1,
                         offset=start - 8)[0] == per_block
    state = np.frombuffer(payload, dtype=np.float64, offset=start).reshape(shape)
    for block, bounds in enumerate(geometry["mb_geometry"]):
        x = bounds[0] + (np.arange(extent) - ghosts + 0.5) * (
            bounds[1] - bounds[0]) / block_size
        state[block, 0, 0, 0] = np.where(x < 0.25, 1.0, 4 * stiffness)
        state[block, 7, 0, 0] = np.where(x < 0.25, 1.0, 1e-80)
    injected = tmp_path / "refinement.rst"
    injected.write_bytes(payload)
    output = run("solve", ["-r", str(injected), "time/nlim=1"], ranks)
    assert output["cycle"] == 1
    np.testing.assert_allclose(output["time"], dt, rtol=0, atol=1e-35)
    centers = np.concatenate([
        bounds[0] + (np.arange(block_size) + 0.5) * (
            bounds[1] - bounds[0]) / block_size for bounds in output["mb_geometry"]])
    order = np.argsort(centers)
    actual = np.asarray(output["mb_data"]["erad00_d"]).ravel()[order]
    rhs = np.where(centers[order] < 0.25, 1.0, 1e-80)
    coefficient = np.where(centers[order] < 0.25, stiffness, 0.25)
    edge = 0.5 * (coefficient[:-1] + coefficient[1:])

    # Independent long-double tridiagonal elimination. Carry positive mass
    # separately to avoid cancellation of the large opposing diffusion terms.
    mass = np.ones(count, dtype=np.longdouble)
    diagonal = np.zeros(count, dtype=np.longdouble)
    forcing = rhs.astype(np.longdouble)
    expected = diagonal.copy()
    diagonal[0] = mass[0] + edge[0]
    for i in range(1, count):
        fraction = np.longdouble(edge[i - 1]) / diagonal[i - 1]
        mass[i] += fraction * mass[i - 1]
        diagonal[i] = mass[i] + (edge[i] if i + 1 < count else 0.0)
        forcing[i] += fraction * forcing[i - 1]
    expected[-1] = forcing[-1] / diagonal[-1]
    for i in range(count - 2, -1, -1):
        expected[i] = (forcing[i] + np.longdouble(edge[i]) * expected[i + 1]) / diagonal[i]
    residual = rhs - actual
    flux = edge * (actual[:-1] - actual[1:])
    residual[:-1] -= flux
    residual[1:] += flux
    absolute = np.abs(actual)
    denominator = np.abs(rhs) + absolute
    face_absolute = edge * (absolute[:-1] + absolute[1:])
    denominator[:-1] += face_absolute
    denominator[1:] += face_absolute
    relative = np.linalg.norm(residual) / np.linalg.norm(rhs)
    backward = np.max(np.abs(residual) / denominator)
    assert relative <= 16e-10 or backward <= 1e-10
    # Check relative accuracy even in the 1e-80 tail: a global norm alone
    # would conceal precisely the error that caused the failed native solve.
    assert np.all(np.isfinite(actual)) and np.min(actual) >= 0.0
    assert np.max(np.abs((actual - expected) / expected)) < 1e-8
    mass_error = abs(np.sum(actual - rhs, dtype=np.longdouble)) / np.sum(rhs)
    assert mass_error < 1e-10
    log = (tmp_path / "solve/run.log").read_text()
    reports = re.findall(
        r"(?:componentwise|weak|component metric|accepted component) refinement:.* iterations=(\d+) (?:sweeps|corrections)=(\d+)", log)
    # A final phase can find that an earlier correction already met its
    # target. Require useful refinement somewhere and enforce the total cap
    # for every report; a zero-work final check is not a numerical failure.
    assert reports and any(int(work) > 0 for _, work in reports)
    assert all(0 <= int(work) <= int(total) <= 2000 for total, work in reports)
