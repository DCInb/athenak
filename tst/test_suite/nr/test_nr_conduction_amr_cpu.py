"""Composite electron conduction: independent leaf-face balances and restarts.

Set ATHENA_CONDUCTION_TEST_EXE to a built_in_pgens executable. The leaf-face
reference is assembled from output geometry, independently of MeshBlock halos.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "vis/python"))
from bin_convert import read_binary  # noqa: E402

INPUT = REPO / "inputs/tests/conduction_amr.athinput"


def run(directory, dimensions=1, flags=(), ranks=None, restart=None, extra="",
        input_text=None):
    directory.mkdir(parents=True)
    exe = Path(os.environ.get("ATHENA_CONDUCTION_TEST_EXE", "./athena")).resolve()
    source = INPUT
    if extra or input_text is not None:
        source = directory / "test.athinput"
        source.write_text((INPUT.read_text() if input_text is None else input_text) + extra)
    args = ["-r", str(restart)] if restart else ["-i", str(source)]
    if not restart:
        for axis in (2, 3):
            args += [f"mesh/nx{axis}={16 if axis <= dimensions else 1}",
                     f"meshblock/nx{axis}={8 if axis <= dimensions else 1}"]
    command = [str(exe), *args, *flags]
    if ranks is not None:
        command = ["mpirun", "-np", str(ranks), *command]
    result = subprocess.run(command, cwd=directory, text=True,
                            capture_output=True, timeout=180, check=False)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = [read_binary(str(p)) for p in sorted((directory / "bin").glob("*.bin"))]
    assert outputs
    return outputs


def field(output, name):
    return np.asarray(output["mb_data"][name], dtype=float)


def volumes(output):
    geometry = np.asarray(output["mb_geometry"])
    return np.prod(geometry[:, 1::2] - geometry[:, ::2], axis=1) / np.prod(
        field(output, "eele_d").shape[1:])


def integral(output, name):
    return np.sum(field(output, name) * volumes(output)[:, None, None, None])


def canonical(output, name):
    order = np.lexsort(np.asarray(output["mb_logical"]).T[::-1])
    return field(output, name)[order]


def leaf_laplacian(output, values, coefficient):
    """Independent finite-volume flux sum on a finest-cell ownership raster."""
    shape = field(output, "eele_d").shape[1:]
    dimensions = sum(n > 1 for n in shape)
    cells_per_block = int(np.prod(shape))
    widths = np.repeat(
        (np.asarray(output["mb_geometry"])[:, 1] -
         np.asarray(output["mb_geometry"])[:, 0]) / shape[-1], cells_per_block)
    finest = np.min(widths)
    length = int(round(1.0 / finest))
    owner = np.full((length,) * dimensions, -1, dtype=int)
    for block, bounds in enumerate(output["mb_geometry"]):
        width = widths[block * cells_per_block]
        scale = int(round(width / finest))
        ids = np.arange(block*cells_per_block, (block+1)*cells_per_block)
        ids = ids.reshape(shape[-dimensions:])
        for axis in range(dimensions):
            ids = np.repeat(ids, scale, axis=axis)
        start = [int(round((bounds[2*a]+0.5)/finest))
                 for a in range(dimensions)][::-1]
        owner[tuple(slice(s, s+n) for s, n in zip(start, ids.shape))] = ids
    assert np.all(owner >= 0)
    lap = np.zeros_like(values)
    for axis in range(dimensions):
        neighbor = np.roll(owner, -1, axis=axis)
        mask = owner != neighbor
        left, right = owner[mask], neighbor[mask]
        hl, hr = widths[left], widths[right]
        kl, kr = coefficient[left], coefficient[right]
        conductance = np.zeros_like(kl)
        same = hl == hr
        conductance[same] = 0.5 * (kl[same]+kr[same]) / hl[same]
        different = ~same & (kl > 0) & (kr > 0)
        conductance[different] = 1.0 / (
            0.5*hl[different]/kl[different] + 0.5*hr[different]/kr[different])
        flux = finest**(dimensions-1) * conductance * (values[right]-values[left])
        np.add.at(lap, left, flux)
        np.add.at(lap, right, -flux)
    return lap / widths**dimensions


def assert_closed(outputs, changed=False):
    initial, final = outputs[0], outputs[-1]
    assert final["cycle"] == 3
    assert np.isclose(final["time"], 0.07, rtol=0, atol=1e-14)
    assert set(np.asarray(initial["mb_logical"])[:, 3]) == {0, 1}
    if changed:
        assert len(final["mb_logical"]) != len(initial["mb_logical"])
    for name in ("eele_d", "eion_d", "ener"):
        expected = integral(initial, name)
        for output in outputs[1:]:
            np.testing.assert_allclose(integral(output, name), expected,
                                       rtol=2e-11, atol=2e-13)
            assert np.all(np.isfinite(field(output, name)))
            assert np.min(field(output, name)) > 0
    low, high = np.min(field(initial, "eele_d")), np.max(field(initial, "eele_d"))
    assert np.min(field(final, "eele_d")) > low
    assert np.max(field(final, "eele_d")) < high


@pytest.mark.parametrize("dimensions,axis", [(1, 1), (2, 1), (2, 2),
                                            (3, 1), (3, 2), (3, 3)])
@pytest.mark.parametrize("ghosts", [2, 4])
def test_composite_equation(tmp_path, dimensions, axis, ghosts):
    outputs = run(tmp_path / "solve", dimensions,
                  [f"mesh/nghost={ghosts}", f"problem/shock_dir={axis}"])
    assert_closed(outputs)
    for previous, output in zip(outputs, outputs[1:]):
        dt = output["time"] - previous["time"]
        if dt <= 0:
            continue
        e0 = field(previous, "eele_d").ravel()
        e1 = field(output, "eele_d").ravel()
        rho = field(output, "dens").ravel()
        lap = leaf_laplacian(output, e1/(0.75*rho), 0.5*rho)
        np.testing.assert_allclose(e1 - e0, dt*lap, rtol=0, atol=3e-10)
        np.testing.assert_allclose(field(output, "eion_d"),
                                   field(previous, "eion_d"), rtol=0, atol=2e-13)
        np.testing.assert_allclose(field(output, "ener")-field(previous, "ener"),
                                   (e1-e0).reshape(field(output, "ener").shape),
                                   rtol=0, atol=2e-13)


@pytest.mark.parametrize("boundary", ["periodic", "neumann", "dirichlet"])
@pytest.mark.parametrize("density_right", [1.0, 4.0])
@pytest.mark.parametrize("refinement", ["none", "static"])
def test_dense_reference(tmp_path, boundary, density_right, refinement):
    flags = [f"problem/dr={density_right}"]
    flags.append(f"mesh_refinement/refinement={refinement}")
    if boundary != "periodic":
        for side, word in (("i", "inner"), ("o", "outer")):
            flags += [f"mesh/{side}x1_bc=outflow",
                      f"hydro/conduction_x1_{word}_boundary={boundary}",
                      f"hydro/conduction_x1_{word}_value=0.2"]
    outputs = run(tmp_path / "solve", flags=flags)
    cells = []
    for m, bounds in enumerate(outputs[0]["mb_geometry"]):
        n = field(outputs[0], "eele_d").shape[-1]
        h = (bounds[1]-bounds[0])/n
        cells.extend((bounds[0]+(i+0.5)*h, h, m, i) for i in range(n))
    cells.sort()
    widths = np.array([cell[1] for cell in cells])
    def ordered(output, name):
        return np.array([field(output, name)[m, 0, 0, i] for _, _, m, i in cells])
    rho = ordered(outputs[0], "dens")
    capacity, coefficient = 0.75*rho, 0.5*rho
    count = len(cells)
    operator = np.zeros((count, count))
    source = np.zeros(count)
    for left in range(count if boundary == "periodic" else count-1):
        right = (left+1) % count
        if widths[left] == widths[right]:
            g = 0.5*(coefficient[left]+coefficient[right])/widths[left]
        else:
            g = 1/(0.5*widths[left]/coefficient[left] +
                   0.5*widths[right]/coefficient[right])
        for a, b in ((left, right), (right, left)):
            operator[a, b] += g/widths[a]
            operator[a, a] -= g/widths[a]
    if boundary == "dirichlet":
        for a in (0, count-1):
            g = 2*coefficient[a]/widths[a]**2
            operator[a, a] -= g
            source[a] += g*0.2
    expected = ordered(outputs[0], "eele_d")/capacity
    for before, after in zip(outputs, outputs[1:]):
        dt = after["time"]-before["time"]
        expected = np.linalg.solve(np.diag(capacity)-dt*operator,
                                   capacity*expected+dt*source)
        np.testing.assert_allclose(ordered(after, "eele_d"), capacity*expected,
                                   rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_three_levels(tmp_path, dimensions):
    outputs = run(tmp_path / "solve", dimensions, ["refined_region1/level=2",
                  "mesh_refinement/num_levels=3", "mesh/nghost=4",
                  "hydro/conduction_linear_tolerance=1e-13",
                  "hydro/conduction_nonlinear_tolerance=1e-13"])
    assert set(np.asarray(outputs[0]["mb_logical"])[:, 3]) >= {1, 2}
    for before, after in zip(outputs, outputs[1:]):
        dt = after["time"]-before["time"]
        e0, e1 = field(before, "eele_d").ravel(), field(after, "eele_d").ravel()
        rho = field(after, "dens").ravel()
        np.testing.assert_allclose(e1-e0, dt*leaf_laplacian(
            after, e1/(0.75*rho), 0.5*rho), rtol=0, atol=5e-10)
        for name in ("eele_d", "eion_d", "ener"):
            np.testing.assert_allclose(integral(after, name), integral(before, name),
                                       rtol=2e-11, atol=2e-13)


def test_isothermal_equilibrium(tmp_path):
    outputs = run(tmp_path / "solve", 3, ["problem/pr=1.0", "mesh/nghost=4"])
    for name in ("eele_d", "eion_d", "ener"):
        np.testing.assert_array_equal(field(outputs[-1], name), field(outputs[0], name))


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("change", ["refine", "derefine"])
def test_regrid(tmp_path, dimensions, change):
    flags = ["mesh_refinement/refinement=adaptive"]
    if change == "refine":
        flags.append("amr_criterion0/value_max=0.5")
    outputs = run(tmp_path / "solve", dimensions, flags)
    assert_closed(outputs, changed=True)
    assert set(np.asarray(outputs[-1]["mb_logical"])[:, 3]) == {
        1 if change == "refine" else 0}


@pytest.mark.parametrize("change", ["refine", "derefine"])
def test_restart_after_regrid(tmp_path, change):
    flags = ["mesh_refinement/refinement=adaptive"]
    if change == "refine":
        flags.append("amr_criterion0/value_max=0.5")
    reference = run(tmp_path / "reference", 2, flags)
    run(tmp_path / "partial", 2, [*flags, "time/nlim=2"])
    checkpoint = sorted((tmp_path / "partial/rst").glob("*.rst"))[-1]
    resumed = run(tmp_path / "resumed", restart=checkpoint, flags=["time/nlim=3"])
    assert resumed[-1]["cycle"] == 3
    for name in ("dens", "ener", "eele_d", "eion_d"):
        np.testing.assert_allclose(canonical(resumed[-1], name),
                                   canonical(reference[-1], name),
                                   rtol=2e-11, atol=2e-13)


def tabular_spitzer_settings(directory):
    """Nonlinear caloric table with partial ionization, in physical cgs units."""
    table = directory / "conduction.2t_eos"
    density = np.array([0.01, 100.0])
    temperature = np.logspace(3, 10, 8)
    gas_constant = 1.3806488e-16 / (6.5 * 1.660538921e-24)
    text = ["athenak_two_temperature_eos 1", "dimensions 2 8", "abar 6.5",
            "density", "0.01 100", "temperature",
            " ".join(map(str, temperature))]
    for name, values in (
        ("ion_pressure", gas_constant*density[:, None]*temperature),
        ("electron_pressure", 0.7*gas_constant*density[:, None]*temperature),
        ("ion_specific_internal_energy", np.tile(1.5*gas_constant*temperature, (2, 1))),
        ("electron_specific_internal_energy",
         np.tile(1.05*gas_constant*temperature*(temperature/1e7)**0.3, (2, 1))),
        ("mean_ionization", np.full((2, 8), 0.7)),
    ):
        text += [name] + [" ".join(f"{v:.17e}" for v in row) for row in values]
    table.write_text("\n".join(text + ["end", ""]))
    return f"""
<units>
length_cgs = 0.1
mass_cgs = 0.001
time_cgs = 1e-9
<hydro>
nscalars = 1
rsolver = hllc
alpha_iso = 0.0
alpha_spitzer = true
conduction_flux_limiter = larsen
conduction_flux_limit_coefficient = 0.06
conduction_max_nonlinear_iterations = 20
<time>
evolution = dynamic
nlim = 1
tlim = 0.01
<materials>
nmaterials = 1
scalar_index = 0
material0_abar = 6.5
material0_zbar = 3.5
material0_eos_table_file = {table}
eos_table_interpolation = geometric
eos_table_bounds = flash-extrapolate
eos_table_temperature_to_kelvin = 1e8
<problem>
pl = 0.01
pr = 0.005
"""


@pytest.mark.parametrize("dimensions", [1, 3])
def test_tabular_spitzer(tmp_path, dimensions):
    extra = tabular_spitzer_settings(tmp_path)
    outputs = run(tmp_path / "solve", dimensions, extra=extra)
    control = run(tmp_path / "control", dimensions, extra=extra.replace(
        "alpha_spitzer = true", "alpha_spitzer = false\nconduction_integrator = explicit"))
    assert outputs[-1]["cycle"] == 1
    np.testing.assert_allclose(outputs[-1]["time"], control[-1]["time"], atol=1e-15)
    for name in ("eele_d", "ener"):
        np.testing.assert_allclose(integral(outputs[-1], name),
                                   integral(control[-1], name), rtol=2e-11, atol=1e-13)
        assert np.min(field(outputs[-1], name)) > 0
        assert np.max(np.abs(field(outputs[-1], name)-field(control[-1], name))) > 1e-8
    np.testing.assert_allclose(field(outputs[-1], "eion_d"),
                               field(control[-1], "eion_d"), rtol=2e-11, atol=1e-13)
