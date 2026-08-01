"""Post-macro source synchronization for subcycled Biermann AMR states."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


BASE_INPUT = Path("../../../inputs/mhd/two_temperature_biermann.athinput")
SOURCE_INPUT = Path("biermann_subcycle_source_amr.athinput")
FINAL_TIME = 1.0e-2
NULL_TIME_LIMIT = 2.0e-2
UNIFORM_RESOLUTION = 64
SMR_ROOT_RESOLUTION = 32
BLOCK_SIZE = 8
ACTIVE_BASENAMES = (
    "biermann_subcycle_source_active_uniform",
    "biermann_subcycle_source_active_smr",
)
NULL_BASENAMES = (
    "biermann_subcycle_source_null_uniform",
    "biermann_subcycle_source_null_smr",
)
ALL_BASENAMES = ACTIVE_BASENAMES + NULL_BASENAMES


def write_input():
    """Add mixed CH/He exchange to the smooth Biermann test problem."""
    text = BASE_INPUT.read_text(encoding="ascii")
    text = text.replace("<mhd>\n", "<mhd>\nnscalars = 1\n", 1)
    text = text.replace(
        "initial_electron_temperature_ratio = 1.0\n",
        "initial_electron_temperature_ratio = 0.25\n"
        "t_ei_model = constant\n",
        1,
    )
    text = text.replace(
        "<problem>\n",
        "<problem>\n"
        "material0_fraction = 0.5\n"
        "material0_fraction_x1_amplitude = 0.2\n"
        "material0_fraction_x2_amplitude = 0.2\n",
        1,
    )
    text += """

<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
material0_zbar = 3.5
material0_zeff = 5.285714285714286
material0_t_ei = 0.002
material1_name = He
material1_abar = 4.0
material1_zbar = 2.0
material1_zeff = 2.0
material1_t_ei = 0.05
"""
    SOURCE_INPUT.write_text(text, encoding="ascii")


def common_flags(basename, refined, active):
    resolution = (SMR_ROOT_RESOLUTION if refined or not active
                  else UNIFORM_RESOLUTION)
    coefficient = 5.0 if active else 100.0
    amplitude = 0.1 if active else 0.0
    end_time = FINAL_TIME if active else NULL_TIME_LIMIT
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}", f"mesh/nx2={resolution}",
        f"meshblock/nx1={BLOCK_SIZE}", f"meshblock/nx2={BLOCK_SIZE}",
        f"mesh_refinement/refinement={'static' if refined else 'none'}",
        "refined_region1/x1min=-0.25", "refined_region1/x1max=0.25",
        "refined_region1/x2min=-0.25", "refined_region1/x2max=0.25",
        "problem/rho0=1.0", "problem/p0=1.0",
        f"problem/density_amplitude={amplitude}",
        f"problem/pressure_amplitude={amplitude}",
        "problem/pressure_x2_amplitude=0.0",
        f"mhd/biermann_coefficient={coefficient}",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "mhd/biermann_shock_suppression=false",
        f"time/tlim={end_time}",
        f"output1/dt={end_time}", "output2/dt=-1", "output3/dt=-1",
        f"output4/dt={end_time}", f"output5/dt={end_time}",
        f"output6/dt={end_time}", "output7/dt=-1",
    ]
    flags.extend(["time/nlim=-1", "time/align_outputs=true"] if active
                 else ["time/nlim=1"])
    return flags


def run_case(basename, refined, active):
    flags = common_flags(basename, refined, active)
    assert testutils.run(str(SOURCE_INPUT), flags=flags, timeout=120.0), basename
    return {
        "field": bin_convert.read_binary(
            f"bin/{basename}.biermann_full.00001.bin"),
        "initial_thermal": bin_convert.read_binary(
            f"bin/{basename}.two_temperature_full.00000.bin"),
        "thermal": bin_convert.read_binary(
            f"bin/{basename}.two_temperature_full.00001.bin"),
        "divergence": bin_convert.read_binary(
            f"bin/{basename}.divb.00001.bin"),
        "history": athena_read.hst(f"{basename}.mhd.hst"),
    }


def flatten(output, variable):
    return np.concatenate([
        np.asarray(block).ravel() for block in output["mb_data"][variable]
    ])


def assert_common_invariants(result, refined, expected_time, tele_change):
    field = result["field"]
    initial_thermal = result["initial_thermal"]
    thermal = result["thermal"]
    divergence = result["divergence"]
    history = result["history"]

    if expected_time is not None:
        assert np.isclose(field["time"], expected_time,
                          rtol=0.0, atol=2.0e-15)
        assert np.isclose(thermal["time"], expected_time,
                          rtol=0.0, atol=2.0e-15)
    else:
        assert field["cycle"] == thermal["cycle"] == 1
        assert 0.0 < field["time"] < NULL_TIME_LIMIT
    expected_levels = {0, 1} if refined else {0}
    assert set(np.asarray(field["mb_logical"])[:, 3]) == expected_levels
    for name in ("dens", "eint", "bcc1", "bcc2", "bcc3"):
        values = flatten(field, name)
        assert np.all(np.isfinite(values)), name
    for name in ("eion", "eele", "tion", "tele"):
        values = flatten(thermal, name)
        assert np.all(np.isfinite(values)), name
        assert np.all(values > 0.0), name

    density = flatten(field, "dens")
    internal = flatten(field, "eint")
    eion = flatten(thermal, "eion")
    eele = flatten(thermal, "eele")
    assert np.all(density > 0.0)
    assert np.all(internal > 0.0)
    assert np.allclose(density*(eion+eele), internal,
                       rtol=3.0e-11, atol=3.0e-12)

    initial_tele = flatten(initial_thermal, "tele")
    final_tele = flatten(thermal, "tele")
    assert (np.linalg.norm(final_tele-initial_tele)
            / np.linalg.norm(initial_tele)) > tele_change

    divb = flatten(divergence, "divb")
    assert np.all(np.isfinite(divb))
    assert np.max(np.abs(divb)) < 5.0e-13
    for key in ("mass", "tot-E"):
        assert np.allclose(history[key], history[key][0],
                           rtol=3.0e-12, atol=3.0e-12), key


def block_cell_values(result, block, variable):
    field = result["field"]
    thermal = result["thermal"]
    if variable in ("eion_d", "eele_d"):
        specific = "eion" if variable == "eion_d" else "eele"
        return (np.asarray(field["mb_data"]["dens"][block])
                * np.asarray(thermal["mb_data"][specific][block]))
    return np.asarray(field["mb_data"][variable][block])


def assemble_uniform(result, variable):
    """Assemble one uniform 64^2 cell field in global index order."""
    field = result["field"]
    assert (field["Nx1"], field["Nx2"]) == (
        UNIFORM_RESOLUTION, UNIFORM_RESOLUTION)
    grid = np.empty((UNIFORM_RESOLUTION, UNIFORM_RESOLUTION))
    dx1 = (field["x1max"]-field["x1min"])/UNIFORM_RESOLUTION
    dx2 = (field["x2max"]-field["x2min"])/UNIFORM_RESOLUTION
    filled = np.zeros_like(grid, dtype=bool)
    for block, geometry in enumerate(field["mb_geometry"]):
        x1min, _, x2min, _, _, _ = geometry
        values = block_cell_values(result, block, variable)
        assert values.shape[0] == 1
        ny, nx = values.shape[1:]
        i0 = int(round((x1min-field["x1min"])/dx1))
        j0 = int(round((x2min-field["x2min"])/dx2))
        grid[j0:j0+ny, i0:i0+nx] = values[0]
        filled[j0:j0+ny, i0:i0+nx] = True
    assert np.all(filled)
    return grid


def compare_smr_to_uniform(uniform, smr, variable):
    """Volume-weight an SMR leaf comparison to a finest-grid reference."""
    field = smr["field"]
    reference = assemble_uniform(uniform, variable)
    dx1_ref = (field["x1max"]-field["x1min"])/UNIFORM_RESOLUTION
    dx2_ref = (field["x2max"]-field["x2min"])/UNIFORM_RESOLUTION
    error_squared = 0.0
    reference_squared = 0.0
    error_maximum = 0.0
    reference_maximum = 0.0
    for block, geometry in enumerate(field["mb_geometry"]):
        x1min, x1max, x2min, x2max, x3min, x3max = geometry
        candidate = block_cell_values(smr, block, variable)
        nz, ny, nx = candidate.shape
        assert nz == 1
        dx1 = (x1max-x1min)/nx
        dx2 = (x2max-x2min)/ny
        ratio1 = int(round(dx1/dx1_ref))
        ratio2 = int(round(dx2/dx2_ref))
        assert ratio1 in (1, 2) and ratio2 in (1, 2)
        block_reference = np.empty_like(candidate)
        for j in range(ny):
            j0 = int(round(
                (x2min+j*dx2-field["x2min"])/dx2_ref))
            for i in range(nx):
                i0 = int(round(
                    (x1min+i*dx1-field["x1min"])/dx1_ref))
                block_reference[0, j, i] = np.mean(
                    reference[j0:j0+ratio2, i0:i0+ratio1])
        difference = candidate-block_reference
        cell_volume = dx1*dx2*(x3max-x3min)
        error_squared += np.sum(difference*difference)*cell_volume
        reference_squared += np.sum(
            block_reference*block_reference)*cell_volume
        error_maximum = max(error_maximum, np.max(np.abs(difference)))
        reference_maximum = max(
            reference_maximum, np.max(np.abs(block_reference)))
    assert reference_squared > 0.0 and reference_maximum > 0.0
    return (np.sqrt(error_squared/reference_squared),
            error_maximum/reference_maximum)


def magnetic_energy(result):
    field = result["field"]
    total = 0.0
    for block, geometry in enumerate(field["mb_geometry"]):
        x1min, x1max, x2min, x2max, x3min, x3max = geometry
        b1 = np.asarray(field["mb_data"]["bcc1"][block])
        b2 = np.asarray(field["mb_data"]["bcc2"][block])
        b3 = np.asarray(field["mb_data"]["bcc3"][block])
        nz, ny, nx = b1.shape
        cell_volume = ((x1max-x1min)/nx * (x2max-x2min)/ny
                       * (x3max-x3min)/nz)
        total += np.sum(0.5*(b1*b1+b2*b2+b3*b3))*cell_volume
    return total


def check_active_agreement(uniform, smr):
    for result, refined in ((uniform, False), (smr, True)):
        assert_common_invariants(result, refined, FINAL_TIME, 0.5)
        b1 = flatten(result["field"], "bcc1")
        b2 = flatten(result["field"], "bcc2")
        b3 = flatten(result["field"], "bcc3")
        assert np.max(np.abs(b1)) < 5.0e-13
        assert np.max(np.abs(b2)) < 5.0e-13
        assert np.max(np.abs(b3)) > 2.0e-2

    limits = {
        "dens": (5.0e-3, 2.0e-2),
        "eint": (5.0e-3, 2.0e-2),
        "eion_d": (5.0e-3, 2.0e-2),
        "eele_d": (5.0e-3, 2.0e-2),
        "bcc3": (3.0e-2, 1.0e-1),
    }
    for variable, (l2_limit, linf_limit) in limits.items():
        relative_l2, relative_linf = compare_smr_to_uniform(
            uniform, smr, variable)
        assert relative_l2 < l2_limit, (variable, relative_l2)
        assert relative_linf < linf_limit, (variable, relative_linf)
    energy_uniform = magnetic_energy(uniform)
    energy_smr = magnetic_energy(smr)
    assert abs(energy_smr/energy_uniform-1.0) < 3.0e-2


def check_null_case(result, refined):
    assert_common_invariants(result, refined, None, 0.1)
    maximum = max(
        np.max(np.abs(flatten(result["field"], name)))
        for name in ("bcc1", "bcc2", "bcc3")
    )
    tolerance = 4.0e-5 if not refined else 3.0e-2
    assert maximum < tolerance, maximum


def test_run():
    try:
        write_input()
        active_results = [
            run_case(basename, bool(index), True)
            for index, basename in enumerate(ACTIVE_BASENAMES)
        ]
        check_active_agreement(*active_results)

        for index, basename in enumerate(NULL_BASENAMES):
            check_null_case(
                run_case(basename, bool(index), False), bool(index))
    finally:
        SOURCE_INPUT.unlink(missing_ok=True)
        testutils.cleanup()
        for basename in ALL_BASENAMES:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for directory in (Path("bin"), Path("rst")):
                for path in directory.glob(f"{basename}.*"):
                    path.unlink(missing_ok=True)
