"""D-T burn, nonlocal alpha transport, fluid feedback, and persistent packet tests.

Set ATHENAK_DT_EXE to an existing CPU or CUDA AthenaK build. The same tests run on
both backends. ATHENAK_DT_MPI_LAUNCH optionally enables the two-rank comparison.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "inputs/hydro/dt_fusion.athinput"
AMU = 1.660538921e-24
ENERGY_UNIT = 1e18
VOLUME = 0.02**3


@pytest.fixture(scope="module")
def executable():
    requested = os.environ.get("ATHENAK_DT_EXE")
    choices = [Path(requested)] if requested else [
        Path.cwd() / "athena", ROOT / "tst/build/src/athena"]
    for path in choices:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    pytest.fail("Set ATHENAK_DT_EXE to a CMake-built executable")


def execute(command, work, failure=False):
    p = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=180)
    (work / "run.log").write_text(p.stdout)
    if failure:
        assert p.returncode != 0, p.stdout
        return p.stdout
    assert p.returncode == 0, p.stdout
    return p.stdout


def run(executable, work, overrides=None, extra="", restart=None, launcher=(),
        input_text=None):
    work.mkdir(parents=True, exist_ok=True)
    path = work / "input.athinput"
    path.write_text((input_text or INPUT.read_text()) + extra)
    options = {"job/basename": "case", "time/tlim": 0.002,
               "time/initial_dt": 0.001, "time/nlim": 2,
               "dt_fusion/scattering": "false",
               "mesh/ix1_bc": "periodic", "mesh/ox1_bc": "periodic",
               "dt_fusion/periodic_transport": "true"}
    for i in range(1, 5):
        options[f"output{i}/dt"] = 0.001
    options.update(overrides or {})
    command = [*launcher, str(executable), "-r" if restart else "-i",
               str(restart or path)]
    command += [f"{k}={v}" for k, v in options.items()]
    log = execute(command, work)
    diagnostics = [dict((k, float(v)) for k, v in re.findall(
        r"([a-z_]+)=([-+0-9.eE]+)", line)) for line in log.splitlines()
        if line.startswith("# dt_fusion ")]
    assert diagnostics, log
    return work, diagnostics[-1]


def table(work, name, initial=False):
    paths = sorted((work / "tab").glob(f"case.{name}.*.tab"))
    assert paths
    path = paths[0] if initial else paths[-1]
    names = path.read_text().splitlines()[1].lstrip("#").split()
    values = np.loadtxt(path, comments="#", ndmin=2)
    order = np.argsort(values[:, names.index("x1v")])
    return {n: values[order, i] for i, n in enumerate(names)}


def check_ledger(d):
    assert d["alpha_born_erg"] == pytest.approx(
        d["deposited_erg"] + d["escaped_erg"] + d["active_erg"], rel=2e-10)
    assert d["neutron_erg"] / d["alpha_born_erg"] == pytest.approx(14.1 / 3.5)
    assert d["neutron_mass_g"] == pytest.approx(d["reactions"] * AMU, rel=1e-12)


def test_device_physics(executable, tmp_path):
    build = executable.parents[1]
    cache = (build / "CMakeCache.txt").read_text()
    compiler = next(x.split("=", 1)[1] for x in cache.splitlines()
                    if x.startswith("CMAKE_CXX_COMPILER:"))
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.16)
project(dt_physics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Kokkos CONFIG REQUIRED PATHS "{build}/cmake_packages/Kokkos" NO_DEFAULT_PATH)
add_executable(dt_physics "{Path(__file__).with_name('dt_physics_driver.cpp')}")
target_include_directories(dt_physics PRIVATE "{ROOT}/src" "{build}")
target_compile_definitions(dt_physics PRIVATE KOKKOS_DEPENDENCE)
target_link_libraries(dt_physics PRIVATE Kokkos::kokkos)
""")
    execute(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build"),
             f"-DCMAKE_CXX_COMPILER={compiler}"], tmp_path)
    execute(["cmake", "--build", str(tmp_path / "build"), "-j2"], tmp_path)
    execute([str(tmp_path / "build/dt_physics")], tmp_path)


@pytest.mark.parametrize("collisions,scattering", [("false", "false"),
                                                 ("true", "false"), ("true", "true")])
def test_conservation(executable, tmp_path, collisions, scattering):
    work, d = run(executable, tmp_path, {"dt_fusion/collisions": collisions,
                                       "dt_fusion/scattering": scattering})
    check_ledger(d)
    old, new = table(work, "fluid", True), table(work, "fluid")
    volume = VOLUME / len(new["dens"])
    assert np.sum(old["dens"] - new["dens"]) * volume == pytest.approx(
        d["neutron_mass_g"], rel=2e-7)
    assert np.sum(new["ener"] - old["ener"]) * volume * ENERGY_UNIT == pytest.approx(
        d["deposited_erg"] - d["neutron_bulk_erg"], rel=2e-8, abs=2e-5)
    # Scalar order: D, He4, T, E_i, E_e. Nuclear stoichiometry and charge.
    species = table(work, "species")
    scalars = [n for n in species if n.startswith("r_")]
    assert len(scalars) == 3, species.keys()
    sd, sa, st = [species[n] for n in scalars[:3]]
    assert np.all(sd >= 0) and np.all(st >= 0) and np.all(sa >= 0)
    assert np.max(abs(sd + sa + st - new["dens"])) < 1e-11
    count = np.sum(sa) * volume / (4 * AMU)
    assert count == pytest.approx(d["reactions"], rel=2e-9)
    if collisions == "false":
        assert d["deposited_erg"] == 0
    else:
        assert d["deposited_erg"] > 0
        fusion = table(work, "fusion")
        assert np.sum(fusion["alpha_eion"]) > 0
        assert np.sum(fusion["alpha_eele"]) > 0
        assert np.max(abs(fusion["alpha_dp1"])) > 0


def test_nonlocal_heating(executable, tmp_path):
    # Right half has only tritium, at the same pressure: no local D-T births.
    work, d = run(executable, tmp_path, {"problem/yr": 0.0})
    check_ledger(d)
    fusion = table(work, "fusion")
    right = (fusion["x1v"] > 0.011) & (fusion["x1v"] < 0.014)
    assert np.sum(fusion["alpha_energy"][right]) > 0
    assert np.sum(fusion["alpha_eele"][right]) > 0
    assert np.sum(fusion["alpha_dp1"][right]) > 0
    assert np.max(fusion["dt_rate"][right]) < 1e-8 * np.max(fusion["dt_rate"])


def test_escape(executable, tmp_path):
    work, d = run(executable, tmp_path, {"mesh/ix1_bc": "outflow",
        "mesh/ox1_bc": "outflow", "dt_fusion/periodic_transport": "false",
        "dt_fusion/collisions": "false"})
    check_ledger(d)
    assert d["escaped_erg"] > 0 and d["active_erg"] > 0


@pytest.mark.parametrize("fast_electrons", [False, True])
def test_restart(executable, tmp_path, fast_electrons):
    extra = "\n<output5>\nfile_type = rst\ndt = 0.001\n"
    if fast_electrons:
        extra += """
<fast_electrons>
power_erg_s = 1e14
energy_kev = 100.0
origin_x1 = 0.002
collisions = false
periodic_transport = true
"""
    settings = {"dt_fusion/scattering": "true"}
    full, reference = run(executable, tmp_path / "full", settings, extra=extra)
    first, _ = run(executable, tmp_path / "first", {**settings, "time/nlim": 1}, extra=extra)
    restart = sorted((first / "rst").glob("*.rst"))[-1]
    resumed, d = run(executable, tmp_path / "resumed", settings, restart=restart)
    check_ledger(d)
    for key in reference:
        if key == "residual_erg":
            continue  # summation order is not part of the checkpoint physics
        assert d[key] == pytest.approx(reference[key], rel=2e-10, abs=1e-8), key
    for key, values in table(full, "fluid").items():
        assert np.allclose(table(resumed, "fluid")[key], values, rtol=2e-11, atol=1e-12)


def test_invalid_input(executable, tmp_path):
    log = execute([str(executable), "-i", str(INPUT),
                   "dt_fusion/packets_per_cell=3"], tmp_path, failure=True)
    assert "packets_per_cell" in log


def test_packet_capacity(executable, tmp_path):
    path = tmp_path / "capacity.athinput"
    path.write_text(INPUT.read_text() + "\n<dt_fusion>\nmax_packets=512\n")
    log = execute([str(executable), "-i", str(path)], tmp_path, failure=True)
    assert "max_packets" in log


def test_empty_packet_restart(executable, tmp_path):
    # All new alphas thermalize immediately. Exercise deletion of the complete
    # device population, subsequent buffer reuse, and a zero-packet checkpoint.
    extra = "\n<dt_fusion>\nthermalization_kev=3500\n<output5>\nfile_type=rst\ndt=0.001\n"
    _, reference = run(executable, tmp_path / "full", extra=extra)
    first, _ = run(executable, tmp_path / "first", {"time/nlim": 1}, extra=extra)
    restart = sorted((first / "rst").glob("*.rst"))[-1]
    _, d = run(executable, tmp_path / "resumed", restart=restart)
    check_ledger(d)
    assert d["packets"] == 0 and d["active_erg"] == 0 and d["escaped_erg"] == 0
    assert d["deposited_erg"] == pytest.approx(d["alpha_born_erg"], rel=2e-10)
    for key in ("reactions", "alpha_born_erg", "deposited_erg"):
        assert d[key] == pytest.approx(reference[key], rel=2e-10)


@pytest.mark.parametrize("mpi", [False, True])
def test_transport_waves(executable, tmp_path, mpi):
    # Sparse births and multiple routing waves must agree with uninterrupted
    # trajectories, including ranks which only receive alpha packets.
    launcher = ()
    if mpi:
        command = os.environ.get("ATHENAK_DT_MPI_LAUNCH")
        if not command:
            pytest.skip("ATHENAK_DT_MPI_LAUNCH is unset")
        launcher = shlex.split(command)
    settings = {"problem/yr": 0.0}
    _, reference = run(executable, tmp_path / "full", settings)
    _, d = run(executable, tmp_path / "waves", settings, launcher=launcher,
               extra="\n<dt_fusion>\nmax_segments=1\n")
    check_ledger(d)
    for key in ("reactions", "alpha_born_erg", "deposited_erg", "active_erg", "packets"):
        assert d[key] == pytest.approx(reference[key], rel=2e-10)


def test_no_reactions(executable, tmp_path):
    work, d = run(executable, tmp_path, {"dt_fusion/rate_multiplier": 0.0})
    assert d["reactions"] == 0 and d["packets"] == 0
    for name, before in table(work, "fluid", True).items():
        assert np.array_equal(table(work, "fluid")[name], before)


def test_mhd_carrier(executable, tmp_path):
    text = INPUT.read_text().replace("<hydro>", "<mhd>")
    text = text.replace("hydro_", "mhd_").replace("rsolver = hllc", "rsolver = hlld")
    text = text.replace("<problem>", "<problem>\nbxl=0\nbyl=0\nbzl=0\nbxr=0\nbyr=0\nbzr=0")
    work, d = run(executable, tmp_path, input_text=text)
    check_ledger(d)
    old, new = table(work, "fluid", True), table(work, "fluid")
    assert np.sum(new["ener"] - old["ener"]) * VOLUME / len(new["dens"]) * ENERGY_UNIT == pytest.approx(
        d["deposited_erg"] - d["neutron_bulk_erg"], rel=2e-8, abs=2e-5)


@pytest.mark.parametrize("adaptive", [False, True])
def test_refinement(executable, tmp_path, adaptive):
    extra = f"""
<mesh_refinement>
refinement = {'adaptive' if adaptive else 'static'}
num_levels = 2
refinement_interval = 1
max_nmb_per_rank = 64
<refined_region1>
level = 1
x1min = 0.005
x1max = 0.01
x2min = -0.01
x2max = 0.01
x3min = -0.01
x3max = 0.01
<amr_criterion0>
method = min_max
variable = hydro_u_d
value_max = 9.0
"""
    work, d = run(executable, tmp_path, extra=extra)
    check_ledger(d)
    new = table(work, "fluid")
    assert len(new["dens"]) > 64
    if adaptive:
        assert len(new["dens"]) > len(table(work, "fluid", True)["dens"])
    # Recover each contiguous 1D leaf-cell width from its center and preceding edge.
    edge, widths = 0.0, []
    for x in new["x1v"]:
        width = 2 * (x-edge)
        widths.append(width)
        edge += width
    assert edge == pytest.approx(0.02)
    assert np.dot(new["dens"], widths) * 0.02**2 == pytest.approx(
        10 * VOLUME - d["neutron_mass_g"], rel=2e-10)


def test_tabulated_eos(executable, tmp_path):
    workdir = tmp_path / "tables"
    workdir.mkdir()
    extra = "\n<materials>\neos_table_bounds=clamp\n"
    for n, (a, z) in enumerate(((2, 1), (4, 2), (3, 1))):
        rho = np.array([1e-12, 1e4])
        temp = np.array([1e3, 1e10])
        cv = 1.5 * 1.3806488e-16 / (a * AMU)
        lines = ["athenak_two_temperature_eos 1", "dimensions 2 2", f"abar {a}",
                 "density", "1e-12 1e4", "temperature", "1e3 1e10"]
        for label, data in (
            ("ion_pressure", rho[:, None]*temp[None, :]*cv*2/3),
            ("electron_pressure", rho[:, None]*temp[None, :]*cv*2/3*z),
            ("ion_specific_internal_energy", np.tile(cv*temp, (2, 1))),
            ("electron_specific_internal_energy", np.tile(cv*temp*z, (2, 1))),
            ("mean_ionization", np.ones((2, 2))*z),
        ):
            lines += [label] + [" ".join(f"{v:.17e}" for v in row) for row in data]
        path = workdir / f"material{n}.2t_eos"
        path.write_text("\n".join(lines + ["end", ""]))
        extra += f"material{n}_eos_table_file={path}\n"
    _, reference = run(executable, tmp_path / "ideal")
    _, d = run(executable, tmp_path / "tabular", extra=extra)
    check_ledger(d)
    for name in ("reactions", "alpha_born_erg", "deposited_erg", "active_erg"):
        assert d[name] == pytest.approx(reference[name], rel=2e-7)


def test_mpi(executable, tmp_path):
    launcher = os.environ.get("ATHENAK_DT_MPI_LAUNCH")
    if not launcher:
        pytest.skip("ATHENAK_DT_MPI_LAUNCH is unset")
    _, serial = run(executable, tmp_path / "one")
    _, parallel = run(executable, tmp_path / "two", launcher=shlex.split(launcher))
    check_ledger(parallel)
    for name in ("reactions", "alpha_born_erg", "deposited_erg", "active_erg"):
        assert parallel[name] == pytest.approx(serial[name], rel=2e-10)


def test_mpi_restart_repartition(executable, tmp_path):
    launcher = os.environ.get("ATHENAK_DT_MPI_LAUNCH")
    if not launcher:
        pytest.skip("ATHENAK_DT_MPI_LAUNCH is unset")
    extra = "\n<output5>\nfile_type = rst\ndt = 0.001\n"
    settings = {"dt_fusion/scattering": "true"}
    _, reference = run(executable, tmp_path / "full", settings)
    first, _ = run(executable, tmp_path / "first", {**settings, "time/nlim": 1}, extra=extra)
    restart = sorted((first / "rst").glob("*.rst"))[-1]
    _, d = run(executable, tmp_path / "two", settings, restart=restart,
               launcher=shlex.split(launcher))
    check_ledger(d)
    for name in ("reactions", "alpha_born_erg", "deposited_erg", "active_erg"):
        assert d[name] == pytest.approx(reference[name], rel=2e-10)


def test_mpi_source_subcycles(executable, tmp_path):
    # Multiple nuclear substeps share one final fluid halo exchange. Nonuniform
    # fuel and transported feedback must still give the same cell states across
    # a rank boundary, including the EOS projection between source operators.
    launcher = os.environ.get("ATHENAK_DT_MPI_LAUNCH")
    if not launcher:
        pytest.skip("ATHENAK_DT_MPI_LAUNCH is unset")
    settings = {"problem/yr": 0.0, "dt_fusion/scattering": "true"}
    extra = "\n<dt_fusion>\nmax_source_dt=0.0001\n"
    one, reference = run(executable, tmp_path / "one", settings, extra=extra)
    two, d = run(executable, tmp_path / "two", settings,
                 extra=extra, launcher=shlex.split(launcher))
    check_ledger(d)
    for name in ("reactions", "alpha_born_erg", "deposited_erg", "active_erg", "packets"):
        assert d[name] == pytest.approx(reference[name], rel=2e-10)
    for field in ("fluid", "species"):
        parallel = table(two, field)
        for name, values in table(one, field).items():
            assert np.allclose(parallel[name], values, rtol=2e-10, atol=2e-12), name
