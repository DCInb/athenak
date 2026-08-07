"""Material-floor consistency at a subcycled Biermann AMR mortar."""

from pathlib import Path

import numpy as np

import athena_read
import bin_convert
import test_suite.testutils as testutils


BASE_INPUT = Path("../../../inputs/mhd/two_temperature_biermann.athinput")
MATERIAL_INPUT = Path("tabular_biermann_floor_amr.athinput")
TABLE = Path("tabular_biermann_floor_amr.2t_eos")
# Keep the intervening macro-MHD perturbation at O(dt^2), below the null
# tolerance, while retaining five orders of separation from the O(dt) mortar
# error produced when restricted component energies bypass the material floor.
FINAL_TIME = 1.0e-8
FLOOR = 1.0
KINDS = ("temperature", "pressure")


def write_table(kind):
    """Write a positive table whose configured floor does not commute with restriction."""
    densities = (0.1, 10.0)
    temperatures = (0.25, 4.0)

    def ion_pressure(rho, temp):
        if kind == "temperature":
            return 0.2*rho*temp
        return 0.2*np.sqrt(rho)*temp

    def electron_pressure(rho, temp):
        if kind == "temperature":
            return 0.8*rho*temp
        return 0.8*np.sqrt(rho)*temp

    def ion_energy(rho, temp):
        if kind == "temperature":
            return 1.5*temp
        return 1.5*temp/np.sqrt(rho)

    def electron_energy(rho, temp):
        if kind == "temperature":
            return 1.2*temp/np.sqrt(rho)
        return 1.2*temp

    # At T_floor, p_e/n_e is constant; at p_floor, p_e is constant.  Both are
    # exact-null Biermann states.  In both cases rho*e_e at the selected floor is
    # proportional to sqrt(rho), so restriction produces an electron energy just
    # below the floor evaluated at the coarse density.

    def rows(function):
        return [" ".join(f"{function(rho, temp):.17g}" for temp in temperatures)
                for rho in densities]

    lines = [
        "athenak_two_temperature_eos 1",
        "dimensions 2 2",
        "abar 1.0",
        "density",
        " ".join(str(value) for value in densities),
        "temperature",
        " ".join(str(value) for value in temperatures),
        "ion_pressure", *rows(ion_pressure),
        "electron_pressure", *rows(electron_pressure),
        "ion_specific_internal_energy", *rows(ion_energy),
        "electron_specific_internal_energy", *rows(electron_energy),
        "mean_ionization", "1.0 1.0", "1.0 1.0",
        "end", "",
    ]
    TABLE.write_text("\n".join(lines), encoding="ascii")


def write_input():
    text = BASE_INPUT.read_text(encoding="ascii")
    text = text.replace("<mhd>\n", "<mhd>\nnscalars = 2\n", 1)
    text = text.replace(
        "gamma = 1.6666666666666667\n",
        "gamma = 1.6666666666666667\npfloor = 0.0\ntfloor = 0.0\n", 1)
    text = text.replace("rsolver = hlle", "rsolver = llf", 1)
    text = text.replace(
        "<problem>\n", "<problem>\nmaterial0_fraction = 1.0\n", 1)
    table = TABLE.resolve()
    text += f"""

<units>
length_cgs = 1.0
mass_cgs = 1.0
time_cgs = 1.0
mu = 1.0

<materials>
nmaterials = 2
scalar_index = 0
material0_name = floor_material
material0_abar = 1.0
material0_zbar = 1.0
material0_zeff = 1.0
material1_name = unused_material
material1_abar = 1.0
material1_zbar = 1.0
material1_zeff = 1.0
material0_eos_table_file = {table}
material1_eos_table_file = {table}
eos_table_bounds = clamp
eos_table_interpolation = geometric
eos_table_density_to_cgs = 1.0
eos_table_temperature_to_kelvin = 1.0
eos_table_pressure_from_cgs = 1.0
eos_table_specific_energy_from_cgs = 1.0
"""
    MATERIAL_INPUT.write_text(text, encoding="ascii")


def run_case(kind, refined):
    suffix = "smr" if refined else "uniform"
    basename = f"tabular_biermann_{kind}_floor_{suffix}"
    Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
    flags = [
        f"job/basename={basename}",
        "mesh/nx1=32", "mesh/nx2=32",
        "meshblock/nx1=8", "meshblock/nx2=8",
        f"mesh_refinement/refinement={'static' if refined else 'none'}",
        "refined_region1/x1min=-0.25", "refined_region1/x1max=-0.24",
        "refined_region1/x2min=-0.25", "refined_region1/x2max=-0.24",
        "problem/rho0=1.0", "problem/p0=1.0e-6",
        "problem/density_amplitude=1.0",
        "problem/pressure_amplitude=0.0",
        "problem/pressure_x2_amplitude=0.0",
        "problem/density_x3_amplitude=0.0",
        "problem/compression_rate_x1=0.0",
        "problem/compression_rate_x2=0.0",
        "mhd/biermann_coefficient=100.0",
        "mhd/biermann_subcycle=true",
        "mhd/biermann_subcycle_cfl=0.15",
        "mhd/biermann_reduced_closure=true",
        "mhd/biermann_shock_suppression=false",
        f"mhd/pfloor={FLOOR if kind == 'pressure' else 0.0}",
        f"mhd/tfloor={FLOOR if kind == 'temperature' else 0.0}",
        "time/align_outputs=true", f"time/tlim={FINAL_TIME}",
        f"output1/dt={FINAL_TIME}",
        "output2/dt=-1", "output3/dt=-1",
        f"output4/dt={FINAL_TIME}", f"output5/dt={FINAL_TIME}",
        f"output6/dt={FINAL_TIME}", "output7/dt=-1",
    ]
    assert testutils.run(str(MATERIAL_INPUT), flags=flags, timeout=120.0), basename
    return basename


def flattened(output, variable):
    return np.concatenate([
        np.asarray(block).ravel() for block in output["mb_data"][variable]
    ])


def check_case(kind, basename, refined):
    field = bin_convert.read_binary(f"bin/{basename}.biermann_full.00001.bin")
    thermal = bin_convert.read_binary(
        f"bin/{basename}.two_temperature_full.00001.bin")
    divergence = bin_convert.read_binary(f"bin/{basename}.divb.00001.bin")
    history = athena_read.hst(f"{basename}.mhd.hst")

    expected_topology = {0: 15, 1: 4} if refined else {0: 16}
    levels, counts = np.unique(np.asarray(field["mb_logical"])[:, 3],
                               return_counts=True)
    assert dict(zip(levels.tolist(), counts.tolist())) == expected_topology

    density = flattened(field, "dens")
    electron_temperature = flattened(thermal, "tele")
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(electron_temperature))
    if kind == "temperature":
        floor_coordinate = electron_temperature
    else:
        floor_coordinate = np.sqrt(density)*electron_temperature
    assert np.min(floor_coordinate) >= FLOOR*(1.0-3.0e-12), (
        basename, np.min(floor_coordinate))
    # The initial energy corresponds to a temperature below the native table bound;
    # staying close to the requested floor proves this is a floor-active, not merely
    # floor-configured, calculation.
    assert np.median(floor_coordinate) < FLOOR*1.001, (
        basename, np.median(floor_coordinate))

    magnetic_maximum = max(
        np.max(np.abs(flattened(field, component)))
        for component in ("bcc1", "bcc2", "bcc3"))
    face_energy = np.array([
        history[key][-1] for key in ("1-ME", "2-ME", "3-ME")
    ])
    assert magnetic_maximum < 2.0e-12, (basename, magnetic_maximum)
    assert np.all(face_energy < 2.0e-24), (basename, face_energy)

    divb = flattened(divergence, "divb")
    assert np.max(np.abs(divb)) < 5.0e-13, (
        basename, np.max(np.abs(divb)))
    total_energy = np.asarray(history["tot-E"])
    assert np.all(np.isfinite(total_energy))
    relative_floor_work = (total_energy[-1]-total_energy[0])/total_energy[0]
    assert -3.0e-12 < relative_floor_work < 1.0e-5, (
        basename, relative_floor_work)


def test_run():
    basenames = []
    try:
        write_input()
        for kind in KINDS:
            write_table(kind)
            for refined in (False, True):
                basename = run_case(kind, refined)
                basenames.append(basename)
                check_case(kind, basename, refined)
    finally:
        MATERIAL_INPUT.unlink(missing_ok=True)
        TABLE.unlink(missing_ok=True)
        testutils.cleanup()
        for basename in basenames:
            Path(f"{basename}.mhd.hst").unlink(missing_ok=True)
            for directory in ("bin", "rst"):
                for path in Path(directory).glob(f"{basename}.*"):
                    path.unlink(missing_ok=True)
