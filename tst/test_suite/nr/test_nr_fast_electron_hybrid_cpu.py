"""Hybrid Ohm-Faraday, refined-mesh transport, and persistent particle regressions."""

import os
import re
import shlex
import subprocess

import numpy as np
import pytest

from test_suite.nr.test_nr_fast_electrons_cpu import (
    INPUT, ENERGY_CGS, TIME_CGS, DX, KEV_ERG,
    conserved, run_case, speed, table,
)
from test_suite.nr.test_nr_fast_electrons_cpu import executable as _executable


@pytest.fixture(scope="module")
def executable():
    return _executable.__wrapped__()


def hybrid_input():
    deck = INPUT.read_text().replace("<hydro>", "<mhd>")
    deck = deck.replace("rsolver = hllc", "rsolver = hlle")
    deck = deck.replace("hydro_2t", "mhd_2t").replace("hydro_u", "mhd_u")
    magnetic = "\n".join(f"{component}{side} = 0.0"
                         for component in ("bx", "by", "bz") for side in ("l", "r"))
    return deck.replace("<problem>", "<problem>\n" + magnetic).replace(
        "<fast_electrons>", "<fast_electrons>\nelectromagnetic = true\n"
        "magnetic_diffusivity_cm2_s = 1.0e5\nfield_cfl = 0.2")


def checkpoint_input(deck, per_rank=False):
    return deck + ("\n<output4>\nfile_type = rst\ndt = -1\ndcycle = 1\n"
                   f"single_file_per_rank = {str(per_rank).lower()}\n")


def restart(exe, directory, checkpoint, overrides=None, launcher=(), failure=None):
    directory.mkdir(parents=True)
    command = [*launcher, str(exe), "-r", str(checkpoint)]
    command.extend(f"{k}={v}" for k, v in (overrides or {}).items())
    proc = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=180, check=False)
    (directory / "run.log").write_text(proc.stdout)
    if failure:
        assert proc.returncode != 0, proc.stdout
        assert failure in proc.stdout, proc.stdout
        return None
    assert proc.returncode == 0, proc.stdout
    diagnostics = [{k: float(v) for k, v in re.findall(r"([a-z_]+)=([-+0-9.eE]+)", line)}
                   for line in proc.stdout.splitlines()
                   if line.startswith("# fast_electrons ")]
    return {"work": directory, "diagnostics": diagnostics, "log": proc.stdout}


def particle_footer(path):
    """Read the documented explicit footer, independently of the C++ packet layout."""
    payload = path.read_bytes()
    offset = payload.rindex(b"FEPIC001") + 8
    header = np.frombuffer(payload, dtype=np.uint64, count=13, offset=offset)
    offset += 13 * 8
    assert header[0] in (1, 2) and tuple(header[1:3]) == (8, 11)
    offset += 2 * 8
    if header[0] == 2:
        offset += 3 * 8 + 2 * 8  # Fixed emission times and quadrature dimensions.
    offset += int(header[5]) * 5 * 8
    counts = np.frombuffer(payload, dtype=np.uint64, count=int(header[4]), offset=offset)
    offset += len(counts) * 8
    dtype = np.dtype([("x", "f8", 3), ("n", "f8", 3), ("energy", "f8"),
                      ("weight", "f8"), ("rng", "u8")])
    assert dtype.itemsize == header[12]
    packets = []
    for gid in range(int(header[8]), int(header[8] + header[9])):
        offset += int(header[2] * header[3]) * 8
        count = int(counts[gid])
        packets.append(np.frombuffer(payload, dtype=dtype, count=count, offset=offset))
        offset += count * dtype.itemsize
    result = np.concatenate(packets)
    return header, np.sort(result, order="rng")


def binary_blocks(path):
    result = []
    with path.open("rb") as stream:
        assert stream.readline().split()[-1] == b"version=1.1"
        lines = int(stream.readline().split(b"=")[-1])
        header = dict(stream.readline().decode().strip().split("=")
                      for _ in range(lines - 1))
        header = {k.strip(): v.strip() for k, v in header.items()}
        nvars = int(stream.readline().split(b"=")[-1])
        names = stream.readline().decode().split()[1:]
        stream.read(int(stream.readline().split(b"=")[-1]))
        location = np.float64 if int(header["size of location"]) == 8 else np.float32
        dtype = np.float64 if int(header["size of variable"]) == 8 else np.float32
        while True:
            raw = stream.read(24)
            if not raw:
                break
            indices = np.frombuffer(raw, dtype=np.int32)
            shape = indices[1::2] - indices[::2] + 1
            logical = np.frombuffer(stream.read(16), dtype=np.int32)
            bounds = np.fromfile(stream, dtype=location, count=6).reshape(3, 2)
            values = np.fromfile(stream, dtype=dtype, count=nvars * int(np.prod(shape)))
            values = values.reshape((nvars, *shape[::-1]))
            volume = np.prod((bounds[:, 1] - bounds[:, 0]) / shape)
            result.append((logical, bounds, volume, dict(zip(names, values))))
    return result


def integral(path, field):
    return sum(volume * np.sum(data[field]) for _, _, volume, data in binary_blocks(path))


def test_resistive_return_current_heats_electrons(executable, tmp_path):
    run = run_case(executable, tmp_path, {
        "time/tlim": 0.002, "time/initial_dt": 0.002,
        "fast_electrons/end_time": 0.002, "fast_electrons/power_erg_s": 1.0e21,
        "fast_electrons/collisions": "false", "fast_electrons/origin_x1": 0.49,
        "output1/dt": 0.002, "output2/dt": 0.002, "output3/dt": 0.002,
    }, input_text=hybrid_input())
    ledger = conserved(run, 2.0e9)
    assert ledger["deposited_erg"] > 1.0e5
    data = table(run, "fast_electrons")
    assert np.max(data["fe_e1"]) > 0.0
    assert np.min(data["fe_j1"]) < 0.0
    assert np.min(data["fe_ohmic"]) >= 0.0
    assert np.sum(data["fe_ohmic"]) > 0.0
    # Actual conservative electron heating approaches eta*J_return^2 as dt -> 0.
    assert np.sum(data["fe_q"]) == pytest.approx(np.sum(data["fe_ohmic"]), rel=0.08)
    fluid0, fluid1 = table(run, "fluid", True), table(run, "fluid")
    temp0, temp1 = table(run, "temperatures", True), table(run, "temperatures")
    assert DX * ENERGY_CGS * np.sum(fluid1["ener"] - fluid0["ener"]) == pytest.approx(
        ledger["deposited_erg"], rel=2.0e-10)
    np.testing.assert_allclose(temp1["eion"], temp0["eion"], rtol=2.0e-9)
    assert np.sum(temp1["eele"]) > np.sum(temp0["eele"])


@pytest.mark.parametrize("mode", ["static", "refine", "derefine"])
def test_refined_particle_transport_and_energy(executable, tmp_path, mode):
    deck = INPUT.read_text().replace(
        "file_type = tab", "file_type = bin\ndouble_precision_binary = true")
    deck = checkpoint_input(deck) + f"""
<mesh_refinement>
refinement = {'static' if mode == 'static' else 'adaptive'}
num_levels = 2
refinement_interval = 2
max_nmb_per_rank = 64
<refined_region1>
level = 1
x1min = 0.25
x1max = 0.75
<amr_criterion0>
method = min_max
variable = hydro_u_d
value_max = {0.005 if mode == 'refine' else 0.02}
"""
    run = run_case(executable, tmp_path, {
        "time/tlim": 0.05, "time/initial_dt": 0.005, "time/nlim": 3,
        "fast_electrons/end_time": 0.005, "fast_electrons/energy_kev": 1000,
        "fast_electrons/origin_x1": 0.49, "fast_electrons/packets_per_bin": 16,
        "fast_electrons/electron_number_per_gram": 1.0e22,
        "output1/dt": 0.005, "output2/dt": 0.005, "output3/dt": 0.005,
    }, input_text=deck)
    final = run["diagnostics"][-1]
    assert final["injected_erg"] == pytest.approx(50.0)
    assert final["deposited_erg"] > 0.0
    assert abs(final["residual_erg"]) < 1.0e-8
    paths = sorted((run["work"] / "bin").glob("case.fast_electrons.*.bin"))
    assert integral(paths[-1], "fe_energy") * ENERGY_CGS == pytest.approx(
        final["deposited_erg"], rel=2.0e-11)
    initial = {int(block[0][3]) for block in binary_blocks(paths[0])}
    last = {int(block[0][3]) for block in binary_blocks(paths[-1])}
    assert len(initial) == 2
    if mode != "static":
        assert last == {max(initial) if mode == "refine" else min(initial)}


@pytest.mark.parametrize("hybrid", [False, True])
def test_particle_restart_preserves_source_and_rng(executable, tmp_path, hybrid):
    deck = checkpoint_input(hybrid_input() if hybrid else INPUT.read_text())
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("100 15 20 1\n300 5 45 2\n")
    full = run_case(executable, tmp_path / "full", {
        "time/tlim": 0.007, "time/initial_dt": 0.001,
        "fast_electrons/end_time": 0.007, "fast_electrons/scattering": "true",
        "fast_electrons/distribution_file": spectrum,
        "fast_electrons/packets_per_bin": 4,
        "fast_electrons/origin_x1": 0.49,
        "output1/dt": 0.001, "output2/dt": 0.001, "output3/dt": 0.001,
    }, input_text=deck)
    checkpoints = sorted((full["work"] / "rst").glob("*.rst"))
    assert len(checkpoints) >= 4
    spectrum.unlink()  # Restart must use the saved normalized joint distribution.
    resumed = restart(executable, tmp_path / "resumed", checkpoints[1])
    for name in ("fast_electrons", "fluid", "temperatures"):
        reference, actual = table(full, name), table(resumed, name)
        for key in reference:
            np.testing.assert_allclose(actual[key], reference[key], rtol=5.0e-9,
                                       atol=1.0e-24, err_msg=f"{name}/{key}")
    saved = sorted((resumed["work"] / "rst").glob("*.rst"))[-1]
    left_h, left = particle_footer(checkpoints[-1])
    right_h, right = particle_footer(saved)
    assert left_h[6] == right_h[6]  # source sequence
    np.testing.assert_array_equal(left["rng"], right["rng"])
    for name in ("x", "n", "energy", "weight"):
        np.testing.assert_allclose(left[name], right[name], rtol=5.0e-10, atol=1.0e-25)


def test_reject_truncated_particle_checkpoint(executable, tmp_path):
    run = run_case(executable, tmp_path / "source", {
        "fast_electrons/collisions": "false",
    }, input_text=checkpoint_input(INPUT.read_text()))
    path = sorted((run["work"] / "rst").glob("*.rst"))[-1]
    broken = tmp_path / "truncated.rst"
    broken.write_bytes(path.read_bytes()[:-13])
    restart(executable, tmp_path / "truncated", broken,
            failure="Incomplete fast-electron restart footer")


@pytest.mark.parametrize("mode", ["refine", "derefine", "beam"])
def test_mpi_hybrid_amr_restart_when_requested(executable, tmp_path, mode):
    launch = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launch:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for MPI/repartition tests")
    deck = checkpoint_input(hybrid_input()) + f"""
<mesh_refinement>
refinement = adaptive
num_levels = 2
refinement_interval = 2
max_nmb_per_rank = 64
<refined_region1>
level = 1
x1min = 0.25
x1max = 0.75
<amr_criterion0>
method = min_max
variable = {"fe_number" if mode == "beam" else "mhd_u_d"}
value_max = {100.0 if mode == "beam" else 0.005 if mode == "refine" else 0.02}
"""
    flags = {"time/tlim": 0.007, "time/initial_dt": 0.001,
             "fast_electrons/end_time": 0.007, "fast_electrons/scattering": "true",
             "fast_electrons/origin_x1": 0.49,
             "output1/dt": 0.001, "output2/dt": 0.001, "output3/dt": 0.001}
    full = run_case(executable, tmp_path / "full", flags, input_text=deck)
    path = sorted((full["work"] / "rst").glob("*.rst"))[1]
    resumed = restart(executable, tmp_path / "resumed", path,
                      launcher=shlex.split(launch))
    for name in ("fast_electrons", "fluid", "temperatures"):
        a, b = table(full, name), table(resumed, name)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=2.0e-8, atol=1.0e-23,
                                       err_msg=f"{name}/{key}")


def multidimensional_deck(dim, refined):
    deck = hybrid_input().replace("file_type = tab",
                                  "file_type = bin\ndouble_precision_binary = true")
    deck = deck.replace("variable = mhd_u\n", "variable = mhd_u_bcc\n")
    deck = re.sub(r"^slice_x[23] = .*\n", "", deck, flags=re.MULTILINE)
    deck = deck.replace("= outflow", "= periodic")
    deck += "\n<output5>\nfile_type = bin\nvariable = mhd_divb\nid = divb\n"
    deck += "double_precision_binary = true\ndt = 0.002\n"
    if refined:
        deck += """
<mesh_refinement>
refinement = static
num_levels = 2
max_nmb_per_rank = 256
<refined_region1>
level = 1
x1min = 0.5
x1max = 0.75
x2min = 0.0
x2max = 0.25
x3min = 0.0
x3max = 0.25
"""
    flags = {"mesh/nx1": 32 if dim == 2 else 16, "meshblock/nx1": 8,
             "mesh/nx2": 16 if dim == 2 else 8, "meshblock/nx2": 8 if dim == 2 else 4,
             "mesh/nx3": 8 if dim == 3 else 1, "meshblock/nx3": 4 if dim == 3 else 1,
             "time/tlim": 0.002, "time/initial_dt": 0.002,
             "fast_electrons/end_time": 0.002, "fast_electrons/power_erg_s": 1.0e19,
             "fast_electrons/collisions": "false", "fast_electrons/origin_x1": 0.49,
             "fast_electrons/origin_x2": 0.01, "fast_electrons/origin_x3": 0.01,
             "fast_electrons/periodic_transport": "true",
             "output1/dt": 0.002, "output2/dt": 0.002, "output3/dt": 0.002}
    return checkpoint_input(deck), flags


def assert_field_generation(run):
    fluid = sorted((run["work"] / "bin").glob("case.fluid.*.bin"))
    fe = sorted((run["work"] / "bin").glob("case.fast_electrons.*.bin"))
    div = sorted((run["work"] / "bin").glob("case.divb.*.bin"))
    final = run["diagnostics"][-1]
    assert abs(final["residual_erg"]) < 1.0e-8 * final["injected_erg"]
    delta = (integral(fluid[-1], "ener") - integral(fluid[0], "ener")) * ENERGY_CGS
    assert delta == pytest.approx(final["deposited_erg"], rel=2.0e-9, abs=1.0e-5)
    assert integral(fe[-1], "fe_kinetic") * ENERGY_CGS == pytest.approx(
        final["active_erg"], rel=2.0e-11)
    max_b, magnetic_energy = 0.0, 0.0
    for _, _, volume, data in binary_blocks(fluid[-1]):
        for component in ("bcc1", "bcc2", "bcc3"):
            max_b = max(max_b, np.max(np.abs(data[component])))
            magnetic_energy += 0.5 * volume * np.sum(data[component]**2)
    assert max_b > 1.0e-12
    assert magnetic_energy > 0.0
    max_div = max(np.max(np.abs(data["divb"]))
                  for _, _, _, data in binary_blocks(div[-1]))
    assert max_div < 1.0e-10 * max_b
    _, packets = particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])
    rest_energy = 8.1871057880e-7
    q = packets["energy"] / rest_energy
    momentum = rest_energy / 2.99792458e10 * np.sqrt(q * (q + 2))
    fast_p = np.sum((packets["weight"] * momentum)[:, None] * packets["n"], axis=0)
    fluid_p = np.array([integral(fluid[-1], f"mom{d}") for d in (1, 2, 3)]) * 1.0e9
    q0 = 100.0 * KEV_ERG / rest_energy
    source_p = final["injected_erg"] / 2.99792458e10 * np.sqrt(q0 * (q0 + 2)) / q0
    np.testing.assert_allclose(fluid_p + fast_p, [source_p, 0.0, 0.0],
                               rtol=2.0e-10, atol=1.0e-14)


@pytest.mark.parametrize("dim,refined", [(2, False), (2, True), (3, False), (3, True)])
def test_self_generated_fields_and_ct(executable, tmp_path, dim, refined):
    deck, flags = multidimensional_deck(dim, refined)
    run = run_case(executable, tmp_path, flags, input_text=deck)
    assert_field_generation(run)


@pytest.mark.parametrize("dim,ideal,refined", [
    (2, False, True), (2, True, True), (3, True, True),
    (2, False, False), (3, False, False), (3, True, False),
])
def test_mpi_hybrid_fields_when_requested(executable, tmp_path, dim, ideal, refined):
    launch = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launch:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for MPI hybrid fields")
    deck, flags = multidimensional_deck(dim, refined)
    if ideal:
        # A localized beam crosses the refined interface at x=0.5. Lorentz
        # feedback creates nonuniform gas velocities and hence nonzero ideal E.
        # Most ranks/cells start empty, but must participate in the timestep and
        # migration collectives. Exercise the active-cell closure without halos.
        flags.update({"fast_electrons/magnetic_diffusivity_cm2_s": 0.0,
                      "problem/bzl": 1.0e-6, "problem/bzr": 1.0e-6})
    a = run_case(executable, tmp_path / "serial", flags, input_text=deck)
    b = run_case(executable, tmp_path / "parallel", flags, input_text=deck,
                 launcher=shlex.split(launch))
    assert_field_generation(a)
    assert_field_generation(b)
    if ideal:
        for result in (a, b):
            final = sorted((result["work"] / "rst").glob("*.rst"))[-1]
            _, packets = particle_footer(final)
            assert np.any(packets["x"][:, 0] > 0.5)
            fields = binary_blocks(sorted((result["work"] / "bin").glob(
                "case.fast_electrons.*.bin"))[-1])
            assert max(np.max(np.abs(v["fe_e1"])) for _, _, _, v in fields) > 0.0
            assert all(np.all(v["fe_ohmic"] == 0.0) for _, _, _, v in fields)
    for name in ("fast_electrons", "fluid", "temperatures"):
        left = binary_blocks(sorted((a["work"] / "bin").glob(f"case.{name}.*.bin"))[-1])
        right = binary_blocks(sorted((b["work"] / "bin").glob(f"case.{name}.*.bin"))[-1])
        assert len(left) == len(right)
        for block_a, block_b in zip(left, right):
            np.testing.assert_array_equal(block_a[0], block_b[0])
            for key, value in block_a[3].items():
                np.testing.assert_allclose(value, block_b[3][key], rtol=1.0e-8,
                                           atol=1.0e-22, err_msg=f"{name}/{key}")


def test_magnetic_orbit_converges_under_field_subcycling(executable, tmp_path):
    deck, flags = multidimensional_deck(2, False)
    flags.update({"problem/bzl": 1.0e-6, "problem/bzr": 1.0e-6,
                  "fast_electrons/magnetic_diffusivity_cm2_s": 0.0,
                  "fast_electrons/power_erg_s": 1.0,
                  "fast_electrons/start_time": 0.0, "fast_electrons/end_time": 1.0e-5,
                  "fast_electrons/origin_x1": 0.2, "fast_electrons/origin_x2": 0.1,
                  "time/tlim": 0.01, "time/initial_dt": 0.01})
    # Independent CGS gyrofrequency and analytic circular orbit for a 100 keV electron.
    gamma = 1.0 + 100.0 / 510.99895069
    b_gauss = np.sqrt(4.0 * np.pi * ENERGY_CGS) * 1.0e-6
    omega = (4.80320471257e-10 * b_gauss * TIME_CGS /
             (9.1093837139e-28 * 2.99792458e10 * gamma))
    phase = omega * (0.01 - 0.5e-5)
    radius = speed(100.0) / omega
    expected = np.array([0.2 + radius * np.sin(phase),
                         0.1 + radius * (1 - np.cos(phase))])
    errors = []
    for cfl in (0.2, 0.1):
        run = run_case(executable, tmp_path / str(cfl), {
            **flags, "fast_electrons/field_cfl": cfl,
        }, input_text=deck)
        _, packets = particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])
        assert len(packets) == 1
        p = packets[0]
        assert p["energy"] == pytest.approx(100.0 * KEV_ERG, rel=2.0e-11)
        np.testing.assert_allclose(p["n"][:2], [np.cos(phase), np.sin(phase)],
                                   atol=1.0e-5)
        errors.append(np.linalg.norm(p["x"][:2] - expected))
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.001


def test_resistive_diffusion_without_fast_particles(executable, tmp_path):
    deck = hybrid_input().replace("= outflow", "= periodic")
    deck = deck.replace("variable = mhd_u\n", "variable = mhd_u_bcc\n")
    errors = []
    for cells in (64, 128):
        run = run_case(executable, tmp_path / str(cells), {
            "mesh/nx1": cells, "fast_electrons/power_erg_s": 0.0,
            "fast_electrons/magnetic_diffusivity_cm2_s": 1.0e8,
            "problem/byl": 1.0e-8, "problem/byr": -1.0e-8,
        }, input_text=deck)
        initial, final = table(run, "fluid", True), table(run, "fluid")
        x = final["x1v"]
        analytic = np.zeros_like(x)
        for mode in range(1, 101, 2):
            k = 2.0 * np.pi * mode
            analytic += (4.0e-8 / (np.pi * mode) * np.sin(k * x) *
                         np.exp(-0.1 * k**2 * 0.02) * np.sinc(mode / cells))
        errors.append(np.mean(np.abs(final["bcc2"] - analytic)) / 1.0e-8)
        np.testing.assert_allclose(np.sum(final["ener"]), np.sum(initial["ener"]),
                                   rtol=2.0e-12)
        assert np.sum(table(run, "fast_electrons")["fe_ohmic"]) > 0.0
        assert run["diagnostics"][-1]["packets"] == 0
    assert errors[1] < 0.4 * errors[0]
    assert errors[1] < 0.001


@pytest.mark.parametrize("per_rank", [False, True])
def test_mpi_hybrid_checkpoint_layouts_when_requested(executable, tmp_path, per_rank):
    launch = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launch:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for MPI checkpoint layouts")
    launcher = shlex.split(launch)
    deck = checkpoint_input(hybrid_input(), per_rank)
    flags = {"time/tlim": 0.007, "time/initial_dt": 0.001,
             "fast_electrons/end_time": 0.007, "fast_electrons/scattering": "true",
             "fast_electrons/origin_x1": 0.49, "fast_electrons/packets_per_bin": 4,
             "output1/dt": 0.001, "output2/dt": 0.001, "output3/dt": 0.001}
    full = run_case(executable, tmp_path / "full", flags, input_text=deck,
                    launcher=launcher)
    directory = full["work"] / "rst"
    if per_rank:
        directory /= "rank_00000000"
    checkpoint = sorted(directory.glob("*.rst"))[1]
    # Shared file: repartition from two ranks to one. Per-rank: preserve two ranks.
    resumed = restart(executable, tmp_path / "resumed", checkpoint,
                      launcher=launcher if per_rank else ())
    for name in ("fast_electrons", "fluid", "temperatures"):
        a, b = table(full, name), table(resumed, name)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=2.0e-8, atol=1.0e-23,
                                       err_msg=f"{name}/{key}")


def test_amr_follows_the_fast_population(executable, tmp_path):
    deck = checkpoint_input(INPUT.read_text()) + """
<mesh_refinement>
refinement = adaptive
num_levels = 2
refinement_interval = 1
max_nmb_per_rank = 64
<amr_criterion0>
method = min_max
variable = fe_number
value_max = 1.0
"""
    run = run_case(executable, tmp_path, {
        "time/tlim": 0.035, "time/initial_dt": 0.005,
        "fast_electrons/start_time": 0.0, "fast_electrons/end_time": 0.001,
        "fast_electrons/collisions": "false", "fast_electrons/origin_x1": 0.2,
    }, input_text=deck)
    header, particles = particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])
    assert header[4] > 4  # Uniform root mesh had four blocks; the occupied leaf refined.
    assert len(particles) == 1
    assert particles["x"][0, 0] == pytest.approx(
        0.2 + speed(100.0) * 0.0345, rel=1.0e-11)
    assert run["diagnostics"][-1]["active_erg"] == pytest.approx(10.0)


@pytest.mark.parametrize("hybrid", [False, True])
def test_packet_growth_compaction_and_transport_waves(executable, tmp_path, hybrid):
    """Buffer growth and removal must preserve survivors, source order, and RNG."""
    spectrum = tmp_path / "spectrum.dat"
    # The first row thermalizes immediately; the others survive or escape. More
    # than one initial buffer's worth is emitted each time the source is sampled.
    spectrum.write_text("0.5 0 0 1\n100 15 20 2\n1000 5 45 3\n")
    deck = checkpoint_input(hybrid_input() if hybrid else INPUT.read_text())
    runs = []
    for segments in (1, 256):
        runs.append(run_case(executable, tmp_path / str(segments), {
            "time/tlim": 0.006, "time/initial_dt": 0.001,
            "fast_electrons/end_time": 0.004,
            "fast_electrons/distribution_file": spectrum,
            "fast_electrons/packets_per_bin": 200,
            "fast_electrons/origin_x1": 0.95,
            "fast_electrons/scattering": "true",
            "fast_electrons/electron_number_per_gram": 1e22,
            "fast_electrons/max_segments": segments,
            "output1/dt": 0.001, "output2/dt": 0.001, "output3/dt": 0.001,
        }, input_text=deck))
    for run in runs:
        ledger = conserved(run, 40.0)
        assert ledger["deposited_erg"] > 0.0
        assert ledger["escaped_erg"] > 0.0
        assert ledger["packets"] > 0
    headers, particles = zip(*(particle_footer(
        sorted((run["work"] / "rst").glob("*.rst"))[-1]) for run in runs))
    assert headers[0][6] == headers[1][6]
    np.testing.assert_array_equal(particles[0]["rng"], particles[1]["rng"])
    for field in ("x", "n", "energy", "weight"):
        np.testing.assert_allclose(particles[0][field], particles[1][field],
                                   rtol=2e-10, atol=1e-25)
    for name in ("fast_electrons", "fluid", "temperatures"):
        a, b = table(runs[0], name), table(runs[1], name)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=2e-9, atol=1e-23,
                                       err_msg=f"{name}/{key}")


def test_mpi_mixed_population_preserves_packet_histories(executable, tmp_path):
    """Migration, early stopping and short flight segments preserve each packet."""
    launch = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
    if not launch:
        pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for MPI packet histories")
    spectrum = tmp_path / "spectrum.dat"
    spectrum.write_text("0.5 0 0 1\n100 15 20 2\n1000 175 45 3\n")
    flags = {
        "time/tlim": 0.006, "time/initial_dt": 0.001,
        "fast_electrons/end_time": 0.004,
        "fast_electrons/distribution_file": spectrum,
        "fast_electrons/packets_per_bin": 200,
        "fast_electrons/origin_x1": 0.49,
        "fast_electrons/scattering": "true",
        "fast_electrons/electron_number_per_gram": 1e22,
    }
    runs = [run_case(executable, tmp_path / label,
                    dict(flags, **{"fast_electrons/max_segments": segments}),
                    input_text=checkpoint_input(INPUT.read_text()), launcher=launcher)
            for label, segments, launcher in [
                ("serial", 256, []),
                ("parallel", 256, shlex.split(launch)),
                ("short-segments", 1, shlex.split(launch))]]
    reference = None
    for run in runs:
        ledger = conserved(run, 40.0)
        assert ledger["deposited_erg"] > 0.0 and ledger["packets"] > 0
        _, packets = particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])
        assert np.any(packets["x"][:, 0] < 0.49)
        assert np.any(packets["x"][:, 0] > 0.5)
        packets = packets[np.argsort(packets["rng"])]
        if reference is None:
            reference = packets
            continue
        np.testing.assert_array_equal(packets["rng"], reference["rng"])
        for field in ("x", "n", "energy", "weight"):
            np.testing.assert_allclose(packets[field], reference[field],
                                       rtol=2e-10, atol=1e-25)


def test_dense_deposition_matches_single_cell_quadrature(executable, tmp_path):
    """The dense GPU deposit path must match the direct path for a short drift."""
    runs = []
    for samples in (1, 8192):
        runs.append(run_case(executable, tmp_path / str(samples), {
            "time/tlim": 1e-6, "time/initial_dt": 1e-6,
            "fast_electrons/end_time": 1e-6,
            "fast_electrons/packets_per_bin": samples,
            "fast_electrons/max_packets": samples,
            "fast_electrons/origin_x1": 0.1,
            "fast_electrons/thermalization_factor": 0,
            "fast_electrons/electron_number_per_gram": 1e22,
            "fast_electrons/magnetic_diffusivity_cm2_s": 0,
            "output1/dt": 1e-6, "output2/dt": 1e-6, "output3/dt": 1e-6,
        }, input_text=hybrid_input()))
        ledger = conserved(runs[-1], 0.01)
        assert ledger["packets"] == samples
        assert ledger["deposited_erg"] > 0
    # No particle reaches a cell face. One initial-rate collision segment per
    # particle makes the midpoint quadrature for mean flight time exact.
    a, b = table(runs[0], "fast_electrons"), table(runs[1], "fast_electrons")
    for field in ("fe_number", "fe_kinetic", "fe_energy", "fe_q", "fe_j1"):
        np.testing.assert_allclose(a[field], b[field], rtol=2e-7, atol=1e-30,
                                   err_msg=field)


def test_live_population_limit_before_device_append(executable, tmp_path):
    log = run_case(executable, tmp_path, {
        "time/tlim": 0.003, "time/initial_dt": 0.001,
        "fast_electrons/end_time": 0.003,
        "fast_electrons/collisions": "false",
        "fast_electrons/packets_per_bin": 200,
        "fast_electrons/max_packets": 300,
    }, expect_failure=True)
    assert "max_packets" in log


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("survivors", [False, True])
def test_reclaim_packet_memory_preserves_physics(executable, tmp_path, parallel,
                                                hybrid, survivors):
    """Reclaim after stopping, then reinject, migrate and restart without losses."""
    launcher = ()
    if parallel:
        launch = os.environ.get("ATHENAK_FAST_ELECTRONS_MPI_LAUNCH")
        if not launch:
            pytest.skip("Set ATHENAK_FAST_ELECTRONS_MPI_LAUNCH for reclamation on MPI")
        launcher = shlex.split(launch)
    spectrum = tmp_path / "spectrum.dat"
    # Most packets thermalize before their first drift. The two energetic rows
    # travel in opposite directions, including across the rank boundary.
    rows = ["0.5 0 0 1\n"] * 62
    rows += (["1000 5 45 1\n", "1000 175 45 1\n"] if survivors else
             ["0.5 0 0 1\n"] * 2)
    spectrum.write_text("".join(rows))
    flags = {
        "time/tlim": 0.006, "time/initial_dt": 0.001,
        "fast_electrons/end_time": 0.004,
        "fast_electrons/distribution_file": spectrum,
        "fast_electrons/packets_per_bin": 64,
        "fast_electrons/origin_x1": 0.49,
        "fast_electrons/electron_number_per_gram": 1e22,
        "fast_electrons/thermalization_factor": 0,
        "fast_electrons/scattering": "true",
        "output1/dt": 0.001, "output2/dt": 0.001, "output3/dt": 0.001,
    }
    deck = checkpoint_input(hybrid_input() if hybrid else INPUT.read_text())
    runs = [run_case(executable, tmp_path / name,
                    dict(flags, **{"fast_electrons/reclaim_packet_memory": enabled}),
                    input_text=deck, launcher=launcher)
            for name, enabled in [("retain", "false"), ("reclaim", "true")]]
    events = [{key: int(value) for key, value in re.findall(r"([a-z_]+)=(\d+)", line)}
              for line in runs[1]["log"].splitlines()
              if line.startswith("# fast_electrons_memory ")]
    assert events, runs[1]["log"]
    assert "# fast_electrons_memory " not in runs[0]["log"]
    assert all(e["reserved_after"] < e["reserved_before"] and e["released_bytes"] > 0
               for e in events)
    assert max(e["released_bytes"] for e in events) > 100000
    if survivors:
        assert any(e["live_packets"] > 0 for e in events)
    else:
        assert all(e["live_packets"] == e["reserved_after"] == 0 for e in events)
    for run in runs:
        ledger = conserved(run, 40.0)
        assert ledger["deposited_erg"] > 0.0
        if survivors:
            assert ledger["packets"] > 0
        else:
            assert ledger["packets"] == ledger["active_erg"] == 0
            assert ledger["deposited_erg"] == pytest.approx(40.0, rel=2e-11)
    if survivors:
        checkpoints = sorted((runs[1]["work"] / "rst").glob("*.rst"))
        assert len(checkpoints) >= 3
        runs.append(restart(executable, tmp_path / "restart", checkpoints[1],
                            launcher=launcher))
        conserved(runs[-1], 40.0)
    _, reference = particle_footer(sorted((runs[0]["work"] / "rst").glob("*.rst"))[-1])
    for run in runs[1:]:
        _, packets = particle_footer(sorted((run["work"] / "rst").glob("*.rst"))[-1])
        np.testing.assert_array_equal(packets["rng"], reference["rng"])
        for field in ("x", "n", "energy", "weight"):
            np.testing.assert_allclose(packets[field], reference[field],
                                       rtol=2e-10, atol=1e-25)
        for name in ("fast_electrons", "fluid", "temperatures"):
            a, b = table(runs[0], name), table(run, name)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=2e-9, atol=1e-23,
                                           err_msg=f"{name}/{key}")
