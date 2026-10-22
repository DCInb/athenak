"""Fast-electron transport regressions using an existing AthenaK executable.

Run with ATHENAK_FAST_ELECTRONS_EXE=/absolute/path/to/athena pytest -q <this file>.
The usual AthenaK harness is also supported: its current directory contains athena.
Each run uses a fresh pytest temporary directory and never changes a build tree.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "inputs/particles/fast_electrons.athinput"
KEV_ERG = 1.602176634e-9
MC2_KEV = 510.99895069
LIGHT_SPEED_CODE = 29.9792458
ENERGY_CGS = 1.0e18
TIME_CGS = 1.0e-9
DT = 0.02
POWER = 1.0e13
SOURCE_X = 0.05
DX = 1.0 / 64


@pytest.fixture(scope="module")
def executable():
    requested = os.environ.get("ATHENAK_FAST_ELECTRONS_EXE")
    candidates = [Path(requested)] if requested else [
        Path.cwd() / "athena", ROOT / "tst/build/src/athena",
        ROOT / "build/src/athena",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    if requested:
        pytest.fail(f"ATHENAK_FAST_ELECTRONS_EXE is not executable: {requested}")
    pytest.skip("Set ATHENAK_FAST_ELECTRONS_EXE to an existing AthenaK build")


def run_case(executable, work, overrides=None, input_text=None, expect_failure=False,
             launcher=()):
    work.mkdir(parents=True, exist_ok=True)
    input_path = INPUT
    if input_text is not None:
        input_path = work / "input.athinput"
        input_path.write_text(input_text)
    options = {
        "job/basename": "case",
        "time/tlim": DT,
        "time/initial_dt": DT,
        "fast_electrons/end_time": DT,
        "fast_electrons/scattering": "false",
        "output1/dt": DT,
        "output2/dt": DT,
        "output3/dt": DT,
    }
    options.update(overrides or {})
    command = [*launcher, str(executable), "-i", str(input_path)]
    command.extend(f"{name}={value}" for name, value in options.items())
    result = subprocess.run(command, cwd=work, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90, check=False)
    (work / "run.log").write_text(result.stdout)
    if expect_failure:
        assert result.returncode != 0, result.stdout
        assert "FATAL" in result.stdout or "ERROR" in result.stdout, result.stdout
        return result.stdout
    assert result.returncode == 0, result.stdout
    diagnostics = []
    for line in result.stdout.splitlines():
        if line.startswith("# fast_electrons "):
            diagnostics.append({key: float(value) for key, value in re.findall(
                r"([a-z_]+)=([-+0-9.eE]+)", line)})
    assert diagnostics, result.stdout
    return {"work": work, "diagnostics": diagnostics, "log": result.stdout}


def table(run, name, initial=False):
    files = sorted((run["work"] / "tab").glob(f"case.{name}.*.tab"))
    assert len(files) >= 2, f"Missing initial/final {name} outputs"
    path = files[0] if initial else files[-1]
    names = path.read_text().splitlines()[1].lstrip("#").split()
    values = np.loadtxt(path, comments="#", ndmin=2)
    assert values.shape[1] == len(names)
    order = np.argsort(values[:, names.index("x1v")])
    return {name: values[order, i] for i, name in enumerate(names)}


def conserved(run, expected_energy=POWER * DT * TIME_CGS):
    final = run["diagnostics"][-1]
    assert final["injected_erg"] == pytest.approx(expected_energy, rel=2.0e-12)
    total = sum(final[name] for name in
                ("deposited_erg", "escaped_erg", "active_erg"))
    assert total == pytest.approx(expected_energy, rel=2.0e-11, abs=1.0e-12)
    assert abs(final["residual_erg"]) <= 2.0e-11 * max(expected_energy, 1.0)
    fields = table(run, "fast_electrons")
    assert np.sum(fields["fe_energy"]) * DX * ENERGY_CGS == pytest.approx(
        final["deposited_erg"], rel=2.0e-11, abs=1.0e-10)
    assert np.sum(fields["fe_kinetic"]) * DX * ENERGY_CGS == pytest.approx(
        final["active_erg"], rel=2.0e-11, abs=1.0e-10)
    return final


def speed(energy_kev):
    gamma = 1.0 + energy_kev / MC2_KEV
    return LIGHT_SPEED_CODE * np.sqrt(1.0 - gamma**-2)


def binary_population(run):
    """Read full-domain moments and occupied-cell centers from version-1.1 output."""
    files = sorted((run["work"] / "bin").glob("case.population.*.bin"))
    assert len(files) >= 2
    counts, energies, centers = [], [], []
    with files[-1].open("rb") as stream:
        assert stream.readline().split()[-1] == b"version=1.1"
        header_lines = int(stream.readline().split(b"=")[-1])
        header = {}
        for _ in range(header_lines - 1):
            key, value = stream.readline().decode().split("=")
            header[key.strip()] = value.strip()
        nvars = int(stream.readline().split(b"=")[-1])
        names = stream.readline().decode().split()[1:]
        offset = int(stream.readline().split(b"=")[-1])
        stream.read(offset)
        location_type = np.float64 if int(header["size of location"]) == 8 else np.float32
        data_type = np.float64 if int(header["size of variable"]) == 8 else np.float32
        while True:
            index_bytes = stream.read(24)
            if not index_bytes:
                break
            indices = np.frombuffer(index_bytes, dtype=np.int32)
            shape = indices[1::2] - indices[::2] + 1
            stream.read(16)  # Logical location and level.
            bounds = np.fromfile(stream, dtype=location_type, count=6).reshape(3, 2)
            values = np.fromfile(stream, dtype=data_type,
                                 count=nvars * int(np.prod(shape)))
            values = values.reshape((nvars, *shape[::-1]))
            spacing = (bounds[:, 1] - bounds[:, 0]) / shape
            volume = np.prod(spacing)
            number = values[names.index("fe_number")]
            kinetic = values[names.index("fe_kinetic")]
            counts.append(np.sum(number) * volume)
            energies.append(np.sum(kinetic) * volume * ENERGY_CGS)
            for k, j, i in np.argwhere(number > 0.0):
                centers.append(bounds[:, 0] + (np.array([i, j, k]) + 0.5) * spacing)
    return sum(counts), sum(energies), np.asarray(centers)


def test_transparent_finite_speed(executable, tmp_path):
    run = run_case(executable, tmp_path, {"fast_electrons/collisions": "false"})
    diagnostics = conserved(run)
    assert diagnostics["deposited_erg"] == 0.0
    assert diagnostics["escaped_erg"] == 0.0
    fields = table(run, "fast_electrons")
    occupied = fields["fe_number"] > 0.0
    assert np.count_nonzero(occupied) == 1
    expected_x = SOURCE_X + speed(100.0) * DT / 2.0
    assert abs(fields["x1v"][occupied][0] - expected_x) <= DX / 2.0
    number = np.sum(fields["fe_number"]) * DX
    assert number == pytest.approx(POWER * DT * TIME_CGS / (100 * KEV_ERG),
                                   rel=2.0e-12)
    assert np.array_equal(table(run, "fluid")["ener"],
                          table(run, "fluid", initial=True)["ener"])


@pytest.mark.parametrize("scattering", ["false", "true"])
def test_conservative_electron_heating(executable, tmp_path, scattering):
    run = run_case(executable, tmp_path, {"fast_electrons/scattering": scattering})
    diagnostics = conserved(run)
    assert diagnostics["deposited_erg"] > 0.0
    fluid0, fluid1 = table(run, "fluid", True), table(run, "fluid")
    temp0, temp1 = table(run, "temperatures", True), table(run, "temperatures")
    delta_total = (fluid1["ener"] - fluid0["ener"]) * ENERGY_CGS
    delta_electron = (fluid1["dens"] * temp1["eele"]
                      - fluid0["dens"] * temp0["eele"]) * ENERGY_CGS
    delta_ion = (fluid1["dens"] * temp1["eion"]
                 - fluid0["dens"] * temp0["eion"]) * ENERGY_CGS
    assert np.sum(delta_total) * DX == pytest.approx(
        diagnostics["deposited_erg"], rel=2.0e-8, abs=1.0e-7)
    np.testing.assert_allclose(delta_electron, delta_total, rtol=2.0e-8, atol=5.0e-8)
    np.testing.assert_allclose(delta_ion, 0.0, atol=5.0e-8)


def test_block_and_integrator_invariance(executable, tmp_path):
    reference = None
    for block_size, integrator in ((64, "rk1"), (16, "rk2"), (8, "rk3")):
        run = run_case(executable, tmp_path / f"{block_size}_{integrator}", {
            "meshblock/nx1": block_size, "time/integrator": integrator,
        })
        conserved(run)
        candidate = table(run, "fast_electrons")
        if reference is None:
            reference = candidate
        else:
            for name in ("fe_q", "fe_energy", "fe_number", "fe_kinetic"):
                np.testing.assert_allclose(candidate[name], reference[name],
                                           rtol=2.0e-10, atol=1.0e-27)


def test_mhd_carrier_matches_zero_field_hydro(executable, tmp_path):
    hydro = run_case(executable, tmp_path / "hydro")
    deck = INPUT.read_text().replace("<hydro>", "<mhd>")
    deck = deck.replace("rsolver = hllc", "rsolver = hlle")
    deck = deck.replace("hydro_2t", "mhd_2t").replace("hydro_u", "mhd_u")
    magnetic_fields = "\n".join(f"{component}{side} = 0.0"
                                for component in ("bx", "by", "bz")
                                for side in ("l", "r"))
    deck = deck.replace("<problem>", "<problem>\n" + magnetic_fields)
    mhd = run_case(executable, tmp_path / "mhd", input_text=deck)
    conserved(mhd)
    left, right = table(hydro, "fast_electrons"), table(mhd, "fast_electrons")
    for name in ("fe_q", "fe_energy", "fe_number", "fe_kinetic"):
        np.testing.assert_allclose(left[name], right[name], rtol=2.0e-10, atol=1.0e-27)


def test_clipped_injection_and_persistent_population(executable, tmp_path):
    run = run_case(executable, tmp_path, {
        "time/tlim": 2 * DT,
        "fast_electrons/start_time": 0.004,
        "fast_electrons/end_time": 0.012,
        "fast_electrons/collisions": "false",
    })
    diagnostics = conserved(run, POWER * 0.008 * TIME_CGS)
    assert len(run["diagnostics"]) == 2
    assert diagnostics["deposited_erg"] == 0.0
    fields = table(run, "fast_electrons")
    occupied = fields["fe_number"] > 0.0
    expected_x = SOURCE_X + speed(100.0) * (2 * DT - 0.008)
    assert abs(fields["x1v"][occupied][0] - expected_x) <= DX / 2.0


def test_periodic_transport(executable, tmp_path):
    duration = 0.2
    run = run_case(executable, tmp_path, {
        "time/tlim": duration, "time/initial_dt": duration,
        "fast_electrons/end_time": duration,
        "fast_electrons/collisions": "false",
        "fast_electrons/periodic_transport": "true",
        "mesh/ix1_bc": "periodic", "mesh/ox1_bc": "periodic",
    })
    diagnostics = conserved(run, POWER * duration * TIME_CGS)
    assert diagnostics["deposited_erg"] == diagnostics["escaped_erg"] == 0.0
    fields = table(run, "fast_electrons")
    occupied = fields["fe_number"] > 0.0
    assert np.count_nonzero(occupied) == 1
    expected_x = (SOURCE_X + speed(100.0) * duration / 2.0) % 1.0
    assert abs(fields["x1v"][occupied][0] - expected_x) <= DX / 2.0


@pytest.mark.parametrize("ndim", [2, 3])
def test_multidimensional_ballistic_geometry(executable, tmp_path, ndim):
    theta, phi = 50.0, 60.0 if ndim == 3 else 0.0
    deck = INPUT.read_text() + (
        "\n<output4>\nfile_type = bin\nvariable = fast_electrons\n"
        f"id = population\ndt = {DT}\ndouble_precision_binary = true\n")
    run = run_case(executable, tmp_path, {
        "mesh/nx2": 16, "meshblock/nx2": 4,
        "mesh/nx3": 16 if ndim == 3 else 1,
        "meshblock/nx3": 4 if ndim == 3 else 1,
        "fast_electrons/theta_deg": theta, "fast_electrons/phi_deg": phi,
        "fast_electrons/collisions": "false",
        "output1/dt": -1.0, "output2/dt": -1.0, "output3/dt": -1.0,
    }, input_text=deck)
    number, energy, positions = binary_population(run)
    assert number == pytest.approx(POWER * DT * TIME_CGS / (100 * KEV_ERG),
                                   rel=2.0e-12)
    assert energy == pytest.approx(POWER * DT * TIME_CGS, rel=2.0e-12)
    assert positions.shape == (1, 3)
    theta, phi = np.deg2rad([theta, phi])
    direction = np.array([np.cos(theta), np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi)])
    expected = np.array([SOURCE_X, 0.0, 0.0]) + speed(100) * DT / 2 * direction
    tolerance = np.array([DX, 1 / 16, 1 / 16 if ndim == 3 else 1.0]) / 2
    assert np.all(np.abs(positions[0] - expected) <= tolerance)
    assert run["diagnostics"][-1]["deposited_erg"] == 0.0


def test_cold_target_slowing_convergence(executable, tmp_path):
    # Independent exact integral of dE/dt=-4*pi*r_e^2*m_e*c^3*ne*lnLambda/beta:
    # G(q0)-G(q)=4*pi*r_e^2*c*ne*lnLambda*t, q=E/(m_e*c^2),
    # G(q)=sqrt(q*(q+2))-acos(1/(1+q)).
    duration = 0.005
    density = 0.1
    prefactor = 4 * np.pi * (2.8179403205e-13)**2 * 2.99792458e10
    ne = density * 6.02214076e23
    q0 = 100 / MC2_KEV

    def integral(q):
        return np.sqrt(q * (q + 2)) - np.arccos(1 / (1 + q))

    target = integral(q0) - prefactor * ne * 10 * duration * TIME_CGS / 2
    assert target > 0.0
    lo, hi = 0.0, q0
    for _ in range(80):
        mid = (lo + hi) / 2
        if integral(mid) > target:
            hi = mid
        else:
            lo = mid
    expected_ratio = (lo + hi) / (2 * q0)
    errors = []
    for fraction in (0.1, 0.025, 0.00625):
        run = run_case(executable, tmp_path / str(fraction), {
            "time/tlim": duration, "time/initial_dt": duration,
            "fast_electrons/end_time": duration,
            "fast_electrons/max_fractional_loss": fraction,
            "problem/dl": density, "problem/dr": density,
            "problem/pl": 1.0e-9, "problem/pr": 1.0e-9,
        })
        result = conserved(run, POWER * duration * TIME_CGS)
        measured_ratio = result["active_erg"] / result["injected_erg"]
        errors.append(abs(measured_ratio - expected_ratio))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 0.002


def test_joint_pic_number_weights(executable, tmp_path):
    distribution = tmp_path / "pic.csv"
    distribution.write_text("# E_keV theta_deg phi_deg relative_number_weight\n"
                            "50, 0, 0, 1\n150, 80, 90, 3\n")
    run = run_case(executable, tmp_path / "run", {
        "fast_electrons/distribution_file": distribution,
        "fast_electrons/collisions": "false",
        "fast_electrons/packets_per_bin": 3,
    })
    conserved(run)
    fields = table(run, "fast_electrons")
    energy = POWER * DT * TIME_CGS
    expected_number = energy / (125 * KEV_ERG)
    assert np.sum(fields["fe_number"]) * DX == pytest.approx(
        expected_number, rel=2.0e-12)
    expected_n = np.zeros(64)
    expected_e = np.zeros(64)
    for e_kev, theta, number_fraction in ((50, 0, 0.25), (150, 80, 0.75)):
        for sample in range(3):
            flight_time = DT * (1.0 - (sample + 0.5) / 3.0)
            position = SOURCE_X + speed(e_kev) * flight_time * np.cos(np.deg2rad(theta))
            cell = np.argmin(np.abs(fields["x1v"] - position))
            expected_n[cell] += expected_number * number_fraction / 3.0
            expected_e[cell] += expected_number * number_fraction / 3.0 * e_kev * KEV_ERG
    np.testing.assert_allclose(fields["fe_number"] * DX, expected_n, rtol=2.0e-12)
    np.testing.assert_allclose(fields["fe_kinetic"] * DX * ENERGY_CGS,
                               expected_e, rtol=2.0e-12)


def test_thermalization_and_escape(executable, tmp_path):
    absorbed = run_case(executable, tmp_path / "absorbed", {
        "fast_electrons/thermalization_kev": 200.0,
    })
    diagnostics = conserved(absorbed)
    assert diagnostics["active_erg"] == 0.0
    assert diagnostics["escaped_erg"] == 0.0
    assert diagnostics["deposited_erg"] == pytest.approx(POWER * DT * TIME_CGS)
    escaped = run_case(executable, tmp_path / "escaped", {
        "time/tlim": 0.2, "time/initial_dt": 0.2,
        "fast_electrons/end_time": 0.2,
        "fast_electrons/collisions": "false",
    })
    diagnostics = conserved(escaped, POWER * 0.2 * TIME_CGS)
    assert diagnostics["deposited_erg"] == 0.0
    assert diagnostics["active_erg"] == 0.0
    assert diagnostics["escaped_erg"] == pytest.approx(POWER * 0.2 * TIME_CGS)


@pytest.mark.parametrize("row", [
    "100 0 0", "100 0 0 -1", "-100 0 0 1", "nan 0 0 1",
    "100 0 0 0", "100 0 0 1 extra",
])
def test_invalid_pic_distribution(executable, tmp_path, row):
    distribution = tmp_path / "invalid.dat"
    distribution.write_text(row + "\n")
    log = run_case(executable, tmp_path / "run", {
        "fast_electrons/distribution_file": distribution,
    }, expect_failure=True)
    assert "<fast_electrons>" in log and "distribution" in log, log


@pytest.mark.parametrize("change", ["units", "two_temperature"])
def test_unsupported_configuration(executable, tmp_path, change):
    deck = INPUT.read_text()
    overrides = {}
    if change == "units":
        deck = re.sub(r"<units>.*?(?=<mesh>)", "", deck, flags=re.DOTALL)
    elif change == "two_temperature":
        deck = deck.replace("two_temperature = true", "two_temperature = false")
        deck = deck.replace("dual_energy = true", "dual_energy = false")
    log = run_case(executable, tmp_path, overrides, input_text=deck, expect_failure=True)
    assert "fast_electrons" in log, log


@pytest.mark.parametrize("overrides", [
    {"fast_electrons/origin_x1": 0.49},
    {"fast_electrons/origin_x1": 0.5, "fast_electrons/theta_deg": 180.0,
     "fast_electrons/thermalization_kev": 200.0},
    {"fast_electrons/origin_x1": 0.49, "fast_electrons/scattering": "true",
     "fast_electrons/packets_per_bin": 32, "fast_electrons/max_segments": 1},
    {"fast_electrons/origin_x1": 0.99, "fast_electrons/collisions": "false",
     "fast_electrons/periodic_transport": "true", "time/tlim": 0.2,
     "time/initial_dt": 0.2, "fast_electrons/end_time": 0.2},
])
def test_mpi_packet_migration_when_requested(executable, tmp_path, overrides):
    """Opt in with ATHENAK_FAST_ELECTRONS_MPI_LAUNCH='mpiexec -n 2'."""
    launcher = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launcher:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH to test MPI transport")
    serial = run_case(executable, tmp_path / "serial", overrides)
    parallel = run_case(executable, tmp_path / "parallel", overrides,
                        launcher=shlex.split(launcher))
    expected = POWER * overrides.get("fast_electrons/end_time", DT) * TIME_CGS
    conserved(serial, expected)
    conserved(parallel, expected)
    left, right = table(serial, "fast_electrons"), table(parallel, "fast_electrons")
    for name in ("fe_q", "fe_energy", "fe_number", "fe_kinetic"):
        np.testing.assert_allclose(left[name], right[name], rtol=2.0e-10, atol=1.0e-27)


@pytest.mark.parametrize("origin,theta,periodic,cell", [
    (0.5, 180.0, False, 31), (0.5, 0.0, False, 32),
    (0.0, 180.0, True, 63), (1.0, 0.0, True, 0),
])
def test_face_ownership_before_thermalization(
        executable, tmp_path, origin, theta, periodic, cell):
    run = run_case(executable, tmp_path, {
        "fast_electrons/origin_x1": origin, "fast_electrons/theta_deg": theta,
        "fast_electrons/periodic_transport": str(periodic).lower(),
        "fast_electrons/thermalization_kev": 200.0,
    })
    final = conserved(run)
    assert final["deposited_erg"] == pytest.approx(final["injected_erg"])
    deposited = table(run, "fast_electrons")["fe_energy"]
    assert np.flatnonzero(deposited).tolist() == [cell]


@pytest.mark.parametrize("tabular", [False, True])
def test_material_density_and_charge_override_source_values(
        executable, tmp_path, tabular):
    atomic_mass = 1.660538921e-24
    charge, density_scale = (0.5, 0.2) if tabular else (2.0, 1.0)
    overrides = {"fast_electrons/electron_number_per_gram":
                 density_scale * charge / (4 * atomic_mass),
                 "fast_electrons/zeff": charge, "fast_electrons/scattering": "true",
                 "fast_electrons/packets_per_bin": 16,
                 "fast_electrons/thermalization_factor": 0.0}
    reference = run_case(executable, tmp_path / "reference", overrides)
    deck = INPUT.read_text().replace("<hydro>", "<hydro>\nnscalars = 1")
    deck += ("\n<materials>\nnmaterials = 1\nscalar_index = 0\n"
             "material0_abar = 4.0\nmaterial0_zbar = 2.0\nmaterial0_zeff = 2.0\n")
    if tabular:
        # Independently generated ideal caloric table with partial ionization.
        # Deliberately distinct EOS density/temperature units exercise the table
        # state path; this validates free-electron transport, not bound losses.
        temperatures = (1.0e3, 1.0e8)
        densities = (1.0e-4, 1.0)
        gas_constant = 1.380649e-16 / (4 * atomic_mass)
        sections = ["athenak_two_temperature_eos 1", "dimensions 2 2", "abar 4",
                    "density", " ".join(map(str, densities)), "temperature",
                    " ".join(map(str, temperatures))]
        for component, factor in (("ion", 1.0), ("electron", charge)):
            sections.append(f"{component}_pressure")
            sections.extend(" ".join(str(rho * gas_constant * factor * t)
                                     for t in temperatures) for rho in densities)
        for component, factor in (("ion", 1.0), ("electron", charge)):
            sections.append(f"{component}_specific_internal_energy")
            sections.extend(" ".join(str(1.5 * gas_constant * factor * t)
                                     for t in temperatures) for _ in densities)
        sections += ["mean_ionization", "0.5 0.5", "0.5 0.5", "end", ""]
        eos_table = tmp_path / "helium.2t_eos"
        eos_table.write_text("\n".join(sections))
        deck += (f"material0_eos_table_file = {eos_table}\neos_table_bounds = clamp\n"
                 "eos_table_density_to_cgs = 0.2\n"
                 "eos_table_temperature_to_kelvin = 1000\n")
    overrides.update({"fast_electrons/electron_number_per_gram": 1.0,
                      "fast_electrons/zeff": 99.0})
    material = run_case(executable, tmp_path / "material", overrides, input_text=deck)
    conserved(reference)
    conserved(material)
    for name in ("fe_q", "fe_energy", "fe_number", "fe_kinetic"):
        np.testing.assert_allclose(table(material, "fast_electrons")[name],
                                   table(reference, "fast_electrons")[name],
                                   rtol=2.0e-11, atol=1.0e-27)


def test_physical_unit_rescaling(executable, tmp_path):
    reference = run_case(executable, tmp_path / "reference")
    scaled = run_case(executable, tmp_path / "scaled", {
        "units/length_cgs": 10.0, "units/mass_cgs": 100.0,
        "units/time_cgs": 1.0e-8, "fast_electrons/power_erg_s": POWER * 10,
    })
    # n_e decreases by ten and the physical time increases by ten. Drag, travel
    # distance in cells, and code-unit heating therefore match the reference.
    left, right = table(reference, "fast_electrons"), table(scaled, "fast_electrons")
    for name in ("fe_q", "fe_energy", "fe_kinetic"):
        np.testing.assert_allclose(left[name], right[name], rtol=2.0e-11, atol=1.0e-27)
    np.testing.assert_allclose(left["fe_number"], 10 * right["fe_number"],
                               rtol=2.0e-11)
    for name in ("injected_erg", "deposited_erg", "escaped_erg", "active_erg"):
        assert scaled["diagnostics"][-1][name] == pytest.approx(
            100 * reference["diagnostics"][-1][name], rel=2.0e-11)


def test_heating_updates_next_fluid_timestep(executable, tmp_path):
    # Fully absorb an energetic pulse in one cell. The next fluid CFL must be
    # computed from the heated state, without waiting an extra fluid cycle.
    run = run_case(executable, tmp_path, {
        "time/tlim": 1.0, "time/nlim": 1,
        "fast_electrons/power_erg_s": 1.0e27,
        "fast_electrons/thermalization_kev": 200.0,
    })
    fields = table(run, "fluid")
    gamma = 5 / 3
    speed_squared = sum(fields[key]**2 for key in ("mom1", "mom2", "mom3"))
    thermal = fields["ener"] - 0.5 * speed_squared / fields["dens"]
    sound = np.sqrt(gamma * (gamma - 1) * thermal / fields["dens"])
    expected = 0.3 * DX / np.max(sound + np.abs(fields["mom1"] / fields["dens"]))
    steps = re.findall(r"cycle=\s*1\s+time=\s*\S+\s+dt=\s*(\S+)", run["log"])
    assert steps, run["log"]
    assert float(steps[-1]) == pytest.approx(expected, rel=2.0e-5)
