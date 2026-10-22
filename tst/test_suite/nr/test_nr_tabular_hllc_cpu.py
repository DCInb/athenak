"""HLLC device identities and integration with both Newtonian table EOS interfaces."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

import athena_read


source_root = Path(__file__).resolve().parents[3]


def execute(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_flux_identities(tmp_path):
    athena_build = Path.cwd().parent
    cmake = f"""
cmake_minimum_required(VERSION 3.16)
project(tabular_hllc_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos CONFIG REQUIRED
  PATHS "{athena_build / 'cmake_packages/Kokkos'}" NO_DEFAULT_PATH)
add_executable(tabular_hllc_driver
  "{source_root / 'tst/test_suite/nr/tabular_hllc_driver.cpp'}")
target_include_directories(tabular_hllc_driver PRIVATE
  "{source_root / 'src'}" "{athena_build}")
target_link_libraries(tabular_hllc_driver PRIVATE Kokkos::kokkos)
"""
    (tmp_path / "CMakeLists.txt").write_text(cmake, encoding="ascii")
    execute(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
            tmp_path)
    execute(["cmake", "--build", str(tmp_path / "build"), "--parallel", "2"],
            tmp_path)
    print(execute([str(tmp_path / "build/tabular_hllc_driver")], tmp_path))


def make_input(path, axis=1, material=False, contact=False):
    # Broad synthetic tables keep these tests inside their EOS domain. Material
    # pressure is 3*rho*T, with eps=7*T (CH) or 14*T (He), unlike the carrier.
    extra = ""
    if material:
        for name, abar in (("ch", 6.5), ("he", 4.0)):
            heat_capacity_scale = 1.0 if name == "ch" else 2.0
            rho = np.array([1.0e-6, 1.0e4])
            temp = np.array([1.0e-6, 1.0e4])
            table = ["athenak_two_temperature_eos 1", "dimensions 2 2",
                     f"abar {abar}", "density", "1.0e-6 1.0e4",
                     "temperature", "1.0e-6 1.0e4"]
            for label, values in (
                ("ion_pressure", rho[:, None] * temp[None, :]),
                ("electron_pressure", 2 * rho[:, None] * temp[None, :]),
                ("ion_specific_internal_energy",
                 np.tile(2 * heat_capacity_scale * temp, (2, 1))),
                ("electron_specific_internal_energy",
                 np.tile(5 * heat_capacity_scale * temp, (2, 1))),
                ("mean_ionization", np.ones((2, 2))),
            ):
                table += [label] + [" ".join(f"{v:.17e}" for v in row)
                                    for row in values]
            (path.parent / f"{name}.2t_eos").write_text(
                "\n".join(table + ["end", ""]), encoding="ascii")
        hydro = """eos = ideal
gamma = 1.6666666666666667
nscalars = 2
two_temperature = true
dual_energy = true
t_ei = 0.0
initial_electron_temperature_ratio = 1.0
"""
        extra = f"""
<units>
length_cgs = 1.0
mass_cgs = 1.0
time_cgs = 1.0
mu = 1.0
<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
material0_zbar = 3.5
material1_name = He
material1_abar = 4.0
material1_zbar = 2.0
material0_eos_table_file = {path.parent / 'ch.2t_eos'}
material1_eos_table_file = {path.parent / 'he.2t_eos'}
eos_table_density_to_cgs = 1.0
eos_table_temperature_to_kelvin = 1.0
eos_table_pressure_from_cgs = 1.0
eos_table_specific_energy_from_cgs = 1.0
eos_table_bounds = error
"""
    else:
        hydro = f"""eos = table
gamma = 1.4
table_file = {source_root / 'inputs/hydro/gamma_law_eos_table.dat'}
table_unit_system = code
table_bounds = error
"""
    mesh, blocks, slices = [], [], []
    for direction in (1, 2, 3):
        count = 96 if direction == axis else (4 if direction < axis else 1)
        mesh += [f"nx{direction} = {count}", f"x{direction}min = -0.5",
                 f"x{direction}max = 0.5", f"ix{direction}_bc = outflow",
                 f"ox{direction}_bc = outflow"]
        blocks += [f"nx{direction} = {32 if direction == axis else count}"]
        if direction != axis:
            slices += [f"slice_x{direction} = 0.0"]
    text = f"""
<job>
basename = hllc
<mesh>
nghost = 3
{chr(10).join(mesh)}
<meshblock>
{chr(10).join(blocks)}
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.3
tlim = 0.1
ndiag = 1000
<hydro>
{hydro}
rsolver = hllc
reconstruct = plm
dfloor = 1.0e-10
pfloor = 1.0e-10
{extra}
<problem>
pgen_name = shock_tube
shock_dir = {axis}
xshock = 0.0
dl = 1.0
dr = {0.25 if contact else 0.125}
pl = 1.0
pr = {(2.0 if material else 1.0) if contact else 0.1}
ul = 0.0
ur = 0.0
vl = 0.0
vr = 0.0
wl = 0.0
wr = 0.0
yl = 1.0
yr = 0.0
<output1>
file_type = tab
variable = hydro_w
data_format = %24.16e
dt = 0.1
{chr(10).join(slices)}
"""
    path.write_text(text, encoding="ascii")


@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("reconstruction", ["plm", "ppm4", "ppmx"])
def test_stationary_contact(tmp_path, material, axis, reconstruction):
    input_file = tmp_path / "contact.athinput"
    make_input(input_file, axis=axis, material=material, contact=True)
    execute([str(Path.cwd() / "athena"), "-i", str(input_file),
             f"hydro/reconstruct={reconstruction}"], tmp_path)
    initial = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00000.tab"))
    final = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00001.tab"))
    if material:
        # Equal material pressures require different energies for the two tables.
        # An accidental return to the carrier EOS would launch spurious waves.
        assert np.ptp(initial["eint"]) > 1.0
    for field in initial:
        if field in ("time", "cycle"):
            continue
        np.testing.assert_allclose(final[field], initial[field],
                                   rtol=2.0e-11, atol=2.0e-11, err_msg=field)


@pytest.mark.parametrize("material", [False, True])
def test_shock_conservation(tmp_path, material):
    input_file = tmp_path / "shock.athinput"
    make_input(input_file, material=material)
    execute([str(Path.cwd() / "athena"), "-i", str(input_file)], tmp_path)
    initial = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00000.tab"))
    final = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00001.tab"))
    for field in final:
        assert np.all(np.isfinite(final[field])), field
    assert np.all(final["dens"] > 0.0)
    assert np.all(final["eint"] > 0.0)
    assert np.max(final["velx"]) > 0.1
    # No waves reach the boundaries by t=0.1, so mass and total energy are conserved.

    def conserved(data):
        density = data["dens"]
        kinetic = 0.5*density*sum(data[v]**2 for v in ("velx", "vely", "velz"))
        return np.array([np.sum(density), np.sum(data["eint"] + kinetic)])
    np.testing.assert_allclose(conserved(final), conserved(initial), rtol=2.0e-10)
    if material:
        np.testing.assert_allclose(final["s_00"] + final["s_01"], 1.0,
                                   rtol=2.0e-12, atol=2.0e-12)
        assert np.all(final["eion"] > 0.0)
        assert np.all(final["eele"] > 0.0)
        np.testing.assert_allclose(
            final["eint"], final["dens"]*(final["eion"] + final["eele"]),
            rtol=2.0e-11, atol=2.0e-11)


def test_density_dependent_energy_contact(tmp_path):
    # This table has p=2*rho*T, eps=3*sqrt(rho)*T. Equal pressure therefore
    # requires unequal internal energy densities: any gamma-law flux moves it.
    input_file = tmp_path / "non_gamma_contact.athinput"
    make_input(input_file, contact=True)
    table = source_root / "inputs/hydro/material_eos_table_v2.dat"
    execute([str(Path.cwd() / "athena"), "-i", str(input_file),
             f"hydro/table_file={table}", "problem/dl=1.0", "problem/dr=4.0",
             "problem/pl=8.0", "problem/pr=8.0"], tmp_path)
    initial = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00000.tab"))
    final = athena_read.tab(str(tmp_path / "tab/hllc.hydro_w.00001.tab"))
    assert np.ptp(initial["eint"]) > 1.0
    for field in ("dens", "eint", "velx", "vely", "velz"):
        np.testing.assert_allclose(final[field], initial[field],
                                   rtol=2.0e-11, atol=2.0e-11, err_msg=field)


def test_sod_convergence(tmp_path):
    from test_suite.nr.test_nr_sod_cpu import compute_error

    input_file = tmp_path / "sod.athinput"
    make_input(input_file)
    errors = []
    for resolution in (96, 192):
        basename = f"sod_{resolution}"
        execute([str(Path.cwd() / "athena"), "-i", str(input_file),
                 f"mesh/nx1={resolution}", f"job/basename={basename}"], tmp_path)
        data = athena_read.tab(str(tmp_path / f"tab/{basename}.hydro_w.00001.tab"))
        errors.append(compute_error(data, tlim=0.1))
    assert errors[1] < 0.015
    assert errors[1] / errors[0] < 0.75, errors
