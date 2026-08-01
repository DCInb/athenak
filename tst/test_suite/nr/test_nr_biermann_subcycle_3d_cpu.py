"""True-3D CPU coverage for constrained-transport Biermann subcycling."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


INPUT_FILE = "../../../inputs/mhd/two_temperature_biermann.athinput"
BASENAME = "biermann_subcycle_3d"
RESOLUTION = 16
BLOCK_SIZE = 8
FINAL_TIME = 1.0e-3

AMR_RESOLUTIONS = (16, 32, 64)
AMR_FINAL_TIME = 1.0e-8
AMR_COEFFICIENT = 5.0
PRESSURE_AMPLITUDES = (0.20, 0.20, 0.05)
DENSITY_AMPLITUDES = (0.10, 0.20)


def amr_basename(resolution):
    return f"biermann_subcycle_3d_amr_{resolution}"


def run_amr_case(resolution):
    basename = amr_basename(resolution)
    block_size = resolution//2
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}",
        f"mesh/nx2={resolution}",
        f"mesh/nx3={resolution}",
        f"meshblock/nx1={block_size}",
        f"meshblock/nx2={block_size}",
        f"meshblock/nx3={block_size}",
        "mesh_refinement/refinement=static",
        # Select only the lower root MeshBlock.  Its eight children meet the
        # other seven root blocks at one true-3D coarse/fine corner.
        "refined_region1/x1min=-0.5",
        "refined_region1/x1max=-0.49",
        "refined_region1/x2min=-0.5",
        "refined_region1/x2max=-0.49",
        "refined_region1/x3min=-0.5",
        "refined_region1/x3max=-0.49",
        f"problem/pressure_amplitude={PRESSURE_AMPLITUDES[0]}",
        f"problem/pressure_x2_amplitude={PRESSURE_AMPLITUDES[1]}",
        f"problem/pressure_x3_amplitude={PRESSURE_AMPLITUDES[2]}",
        f"problem/density_amplitude={DENSITY_AMPLITUDES[0]}",
        f"problem/density_x3_amplitude={DENSITY_AMPLITUDES[1]}",
        "mhd/biermann_shock_suppression=false",
        f"mhd/biermann_coefficient={AMR_COEFFICIENT}",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "time/align_outputs=true",
        f"time/tlim={AMR_FINAL_TIME}",
        f"output1/dt={AMR_FINAL_TIME}",
        "output2/dt=-1",
        "output3/dt=-1",
        "output4/variable=mhd_divb",
        "output4/id=divb",
        "output4/double_precision_binary=true",
        f"output4/dt={AMR_FINAL_TIME}",
        f"output5/dt={AMR_FINAL_TIME}",
        "output6/dt=-1",
        "output7/dt=-1",
    ]
    assert testutils.run(INPUT_FILE, flags=flags, timeout=120.0), (
        f"{basename} run failed")


def analytic_biermann_source(x1, x2, x3):
    """Return C*(p/rho)*cross(grad(log(p)), grad(log(rho)))."""
    ax, ay, az = PRESSURE_AMPLITUDES
    by, bz = DENSITY_AMPLITUDES
    wave_number = 2.0*np.pi
    pressure_over_density = np.exp(
        ax*np.sin(wave_number*x1)
        + (ay-by)*np.sin(wave_number*x2)
        + (az-bz)*np.sin(wave_number*x3))
    common = AMR_COEFFICIENT*pressure_over_density*wave_number**2
    return np.array([
        common*np.cos(wave_number*x2)*np.cos(wave_number*x3)
        * (ay*bz-az*by),
        -common*np.cos(wave_number*x1)*np.cos(wave_number*x3)*ax*bz,
        common*np.cos(wave_number*x1)*np.cos(wave_number*x2)*ax*by,
    ])


def assert_corner_topology(output, resolution):
    assert (output["Nx1"], output["Nx2"], output["Nx3"]) == (
        resolution, resolution, resolution)
    assert (output["nx1_mb"], output["nx2_mb"], output["nx3_mb"]) == (
        resolution//2, resolution//2, resolution//2)

    logical = np.asarray(output["mb_logical"])
    coarse = {tuple(location[:3]) for location in logical
              if location[3] == 0}
    fine = {tuple(location[:3]) for location in logical
            if location[3] == 1}
    root = {(i, j, k) for k in range(2) for j in range(2)
            for i in range(2)}
    assert coarse == root-{(0, 0, 0)}, coarse
    assert fine == root, fine


def analyze_amr_case(resolution):
    basename = amr_basename(resolution)
    field = bin_convert.read_binary(
        f"bin/{basename}.biermann_full.00001.bin")
    divergence = bin_convert.read_binary(
        f"bin/{basename}.divb.00001.bin")
    assert field["cycle"] == 1
    assert divergence["cycle"] == 1
    assert_corner_topology(field, resolution)
    assert np.array_equal(field["mb_logical"], divergence["mb_logical"])

    error_squared = 0.0
    exact_squared = 0.0
    component_error_squared = np.zeros(3)
    component_exact_squared = np.zeros(3)
    numerical_maximum = np.zeros(3)
    corner_maximum = 0.0
    corner_cell_count = 0
    total_volume = 0.0
    maximum_cell_width = 0.0

    root_spacing = (field["x1max"]-field["x1min"])/resolution
    refinement_corner = np.array([
        0.5*(field["x1min"]+field["x1max"]),
        0.5*(field["x2min"]+field["x2max"]),
        0.5*(field["x3min"]+field["x3max"]),
    ])
    for block, geometry in enumerate(field["mb_geometry"]):
        x1min, x1max, x2min, x2max, x3min, x3max = geometry
        shape = np.asarray(field["mb_data"]["bcc1"][block]).shape
        nx3, nx2, nx1 = shape
        dx1 = (x1max-x1min)/nx1
        dx2 = (x2max-x2min)/nx2
        dx3 = (x3max-x3min)/nx3
        maximum_cell_width = max(maximum_cell_width, dx1, dx2, dx3)
        x1 = x1min+(np.arange(nx1)+0.5)*dx1
        x2 = x2min+(np.arange(nx2)+0.5)*dx2
        x3 = x3min+(np.arange(nx3)+0.5)*dx3
        zz, yy, xx = np.meshgrid(x3, x2, x1, indexing="ij")

        exact = analytic_biermann_source(xx, yy, zz)
        numerical = np.array([
            np.asarray(field["mb_data"][component][block])/AMR_FINAL_TIME
            for component in ("bcc1", "bcc2", "bcc3")
        ])
        assert np.all(np.isfinite(numerical))
        error = numerical-exact
        cell_volume = dx1*dx2*dx3
        error_squared += np.sum(error**2)*cell_volume
        exact_squared += np.sum(exact**2)*cell_volume
        component_error_squared += np.sum(
            error**2, axis=(1, 2, 3))*cell_volume
        component_exact_squared += np.sum(
            exact**2, axis=(1, 2, 3))*cell_volume
        numerical_maximum = np.maximum(
            numerical_maximum,
            np.max(np.abs(numerical), axis=(1, 2, 3)))
        total_volume += np.prod(shape)*cell_volume

        # Keep a fixed number of coarse and fine stencil layers around the point
        # where all three refinement faces meet.  This is stricter than an L2 norm:
        # an O(dx) corner defect cannot hide in its shrinking physical volume.
        corner_mask = (
            (np.abs(xx-refinement_corner[0]) <= 2.0*root_spacing)
            & (np.abs(yy-refinement_corner[1]) <= 2.0*root_spacing)
            & (np.abs(zz-refinement_corner[2]) <= 2.0*root_spacing))
        if np.any(corner_mask):
            vector_error = np.sqrt(np.sum(error**2, axis=0))
            corner_maximum = max(
                corner_maximum, np.max(vector_error[corner_mask]))
            corner_cell_count += np.count_nonzero(corner_mask)

    assert np.isclose(total_volume, 1.0, rtol=0.0, atol=2.0e-15)
    assert corner_cell_count == 120, corner_cell_count
    assert np.all(component_exact_squared > 0.0)
    assert np.all(numerical_maximum > 1.0), numerical_maximum

    divb = np.concatenate([
        np.asarray(block).ravel()
        for block in divergence["mb_data"]["divb"]
    ])
    assert np.all(np.isfinite(divb))
    max_divb = np.max(np.abs(divb))
    magnetic_maximum = AMR_FINAL_TIME*np.max(numerical_maximum)
    normalized_divb = max_divb*maximum_cell_width/magnetic_maximum
    assert max_divb < 5.0e-13, max_divb
    assert normalized_divb < 64.0*np.finfo(float).eps, (
        max_divb, normalized_divb)

    return {
        "relative_l2": np.sqrt(error_squared/exact_squared),
        "component_relative_l2": np.sqrt(
            component_error_squared/component_exact_squared),
        "corner_maximum": corner_maximum,
        "max_divb": max_divb,
        "normalized_divb": normalized_divb,
    }


def assert_amr_convergence():
    results = []
    for resolution in AMR_RESOLUTIONS:
        run_amr_case(resolution)
        results.append(analyze_amr_case(resolution))

    l2_errors = np.array([result["relative_l2"] for result in results])
    corner_errors = np.array([
        result["corner_maximum"] for result in results
    ])
    l2_rates = np.log(l2_errors[:-1]/l2_errors[1:])/np.log(2.0)
    corner_rates = np.log(
        corner_errors[:-1]/corner_errors[1:])/np.log(2.0)

    assert np.all(l2_rates > 1.75), (l2_errors, l2_rates)
    assert np.all(corner_rates > 1.05), (corner_errors, corner_rates)
    assert np.all(results[-1]["component_relative_l2"] < 1.0e-2), (
        results[-1]["component_relative_l2"])


def test_run():
    try:
        Path(f"{BASENAME}.mhd.hst").unlink(missing_ok=True)
        flags = [
            f"job/basename={BASENAME}",
            f"mesh/nx1={RESOLUTION}",
            f"mesh/nx2={RESOLUTION}",
            f"mesh/nx3={RESOLUTION}",
            f"meshblock/nx1={BLOCK_SIZE}",
            f"meshblock/nx2={BLOCK_SIZE}",
            f"meshblock/nx3={BLOCK_SIZE}",
            "problem/pressure_x2_amplitude=0.15",
            "problem/density_x3_amplitude=0.15",
            "mhd/biermann_shock_suppression=false",
            "mhd/biermann_coefficient=20.0",
            "mhd/biermann_subcycle=true",
            "mhd/biermann_subcycle_cfl=0.15",
            f"time/tlim={FINAL_TIME}",
            f"output1/dt={FINAL_TIME}",
            f"output2/dt={FINAL_TIME}",
            f"output3/dt={FINAL_TIME}",
            f"output4/dt={FINAL_TIME}",
        ]
        assert testutils.run(INPUT_FILE, flags=flags, timeout=120.0), (
            f"{BASENAME} run failed")

        field = athena_read.tab(
            f"tab/{BASENAME}.biermann.00001.tab")
        two_temperature = athena_read.tab(
            f"tab/{BASENAME}.two_temperature.00001.tab")
        divb_output = bin_convert.read_binary(
            f"bin/{BASENAME}.divb.00001.bin")
        divb = np.concatenate([
            np.asarray(block).ravel()
            for block in divb_output["mb_data"]["divb"]
        ])
        history = athena_read.hst(f"{BASENAME}.mhd.hst")

        # The hydro/MHD macro limit exceeds this short interval, while the battery
        # limit forces multiple SSPRK2 microsteps across the two Strang half-steps.
        assert len(history["time"])-1 == 1
        assert np.isclose(history["dt"][0], FINAL_TIME,
                          rtol=2.0e-13, atol=0.0)

        magnetic_scale = 0.0
        for key in ("bcc1", "bcc2", "bcc3"):
            component = np.asarray(field[key])
            assert np.all(np.isfinite(component))
            component_max = np.max(np.abs(component))
            assert component_max > 1.0e-4, (key, component_max)
            magnetic_scale = max(magnetic_scale, component_max)

        # mhd_divb is evaluated directly from the face-centered CT fields over the
        # complete 3D volume, not from the cell-centered slice above.
        assert divb_output["cycle"] == 1
        assert (divb_output["Nx1"], divb_output["Nx2"],
                divb_output["Nx3"]) == (RESOLUTION,)*3
        assert divb.size == RESOLUTION**3
        assert np.all(np.isfinite(divb))
        max_divb = np.max(np.abs(divb))
        assert max_divb < 5.0e-13, max_divb
        assert (max_divb/RESOLUTION < 1.0e-12*magnetic_scale), (
            max_divb, magnetic_scale)

        for key in ("dens", "eint"):
            assert np.all(np.isfinite(field[key]))
            assert np.all(field[key] > 0.0)
        for key in ("eion", "eele", "tion", "tele"):
            assert np.all(np.isfinite(two_temperature[key]))
            assert np.all(two_temperature[key] > 0.0)

        # eint is an energy density; eion and eele are specific energies.
        component_energy_density = field["dens"]*(
            two_temperature["eion"]+two_temperature["eele"])
        assert np.allclose(field["eint"], component_energy_density,
                           rtol=3.0e-11, atol=3.0e-12)
        assert np.all(np.isfinite(history["tot-E"]))
        assert np.allclose(history["tot-E"], history["tot-E"][0],
                           rtol=3.0e-12, atol=3.0e-12)

        assert_amr_convergence()
    finally:
        testutils.cleanup()
        for basename in (BASENAME, *(amr_basename(n)
                                      for n in AMR_RESOLUTIONS)):
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for path in Path("bin").glob(f"{basename}.*.bin"):
                path.unlink(missing_ok=True)
