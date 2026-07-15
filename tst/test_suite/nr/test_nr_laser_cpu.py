"""Regression tests for device DDA laser-ray transport and 2T deposition."""

import shutil

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


input_file = "../../../inputs/mhd/two_temperature_laser.athinput"
SOURCE_DT = 1.0e-6


def reference_cell_path(origin, direction, lower, upper):
    """Return the ray length inside one Cartesian cell."""
    enter = 0.0
    leave = np.inf
    for x0, ray_dir, cell_lo, cell_hi in zip(
            origin, direction, lower, upper):
        if abs(ray_dir) < 1.0e-15:
            if x0 < cell_lo or x0 >= cell_hi:
                return 0.0
            continue
        first = (cell_lo - x0) / ray_dir
        second = (cell_hi - x0) / ray_dir
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
    return max(leave - enter, 0.0)


def reference_2d_paths(origin_y, direction, resolution):
    ray_dir = np.asarray(direction, dtype=float)
    ray_dir /= np.linalg.norm(ray_dir)
    result = np.zeros((resolution, resolution))
    dx = 1.0 / resolution
    dy = 1.0 / resolution
    for j in range(resolution):
        for i in range(resolution):
            result[j, i] = reference_cell_path(
                (0.0, origin_y), ray_dir,
                (i * dx, -0.5 + j * dy),
                ((i + 1) * dx, -0.5 + (j + 1) * dy))
    return result


def reference_3d_paths(origin, direction, resolution):
    ray_dir = np.asarray(direction, dtype=float)
    ray_dir /= np.linalg.norm(ray_dir)
    result = np.zeros((resolution, resolution, resolution))
    spacing = 1.0 / resolution
    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
                result[k, j, i] = reference_cell_path(
                    origin, ray_dir,
                    (i * spacing, -0.5 + j * spacing,
                     -0.5 + k * spacing),
                    ((i + 1) * spacing, -0.5 + (j + 1) * spacing,
                     -0.5 + (k + 1) * spacing))
    return result


def run_transparent_case(basename, flags):
    common = [
        f"job/basename={basename}",
        "laser/absorption_coefficient=0.0",
        "time/tlim=1.0e-7",
        "output1/dt=-1.0",
        "output2/dt=-1.0",
        "output3/dt=1.0e-7",
    ]
    assert testutils.run(input_file, flags=common + flags), (
        f"{basename} laser transport run failed.")
    data = athena_read.tab(f"tab/{basename}.laser.00001.tab")
    order = np.argsort(data["x1v"])
    fields = ("x1v", "laser_q", "laser_energy", "laser_ray_count",
              "laser_tau", "laser_path")
    return {name: np.atleast_1d(data[name])[order] for name in fields}


def read_laser_binary(filename):
    """Read the uniform-grid subset of AthenaK's version-1.1 binary format."""
    with open(filename, "rb") as binary_file:
        binary_file.seek(0, 2)
        file_size = binary_file.tell()
        binary_file.seek(0)
        assert binary_file.readline().split()[-1] == b"version=1.1"
        preheader_lines = int(binary_file.readline().split(b"=")[-1])
        preheader = {}
        for _ in range(preheader_lines - 1):
            key, value = binary_file.readline().decode().split("=")
            preheader[key.strip()] = value.strip()
        location_size = int(preheader["size of location"])
        variable_size = int(preheader["size of variable"])
        nvars = int(binary_file.readline().split(b"=")[-1])
        var_names = [entry.decode()
                     for entry in binary_file.readline().split()[1:]]
        header_size = int(binary_file.readline().split(b"=")[-1])
        header = binary_file.read(header_size).decode().splitlines()

        parameters = {}
        block = ""
        for line in header:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("<"):
                block = line
            elif "=" in line:
                key, value = line.split("=", 1)
                parameters[(block, key.strip())] = value.strip()

        result = {
            "Nx1": int(parameters[("<mesh>", "nx1")]),
            "Nx2": int(parameters[("<mesh>", "nx2")]),
            "Nx3": int(parameters[("<mesh>", "nx3")]),
            "nx1_mb": int(parameters[("<meshblock>", "nx1")]),
            "nx2_mb": int(parameters[("<meshblock>", "nx2")]),
            "nx3_mb": int(parameters[("<meshblock>", "nx3")]),
            "mb_logical": [],
            "mb_data": {name: [] for name in var_names},
        }
        variable_dtype = np.float64 if variable_size == 8 else np.float32
        while binary_file.tell() < file_size:
            indices = np.frombuffer(binary_file.read(24), dtype=np.int32)
            nx = indices[1] - indices[0] + 1
            ny = indices[3] - indices[2] + 1
            nz = indices[5] - indices[4] + 1
            logical = np.frombuffer(binary_file.read(16), dtype=np.int32)
            result["mb_logical"].append(logical)
            binary_file.read(6 * location_size)
            block_data = np.fromfile(
                binary_file, dtype=variable_dtype, count=nvars * nx * ny * nz)
            block_data = block_data.reshape(nvars, nz, ny, nx)
            for variable, values in zip(var_names, block_data):
                result["mb_data"][variable].append(values)
        return result


def assemble_binary_field(file_data, field):
    result = np.zeros((file_data["Nx3"], file_data["Nx2"],
                       file_data["Nx1"]))
    for block, location in enumerate(file_data["mb_logical"]):
        block_data = np.asarray(file_data["mb_data"][field][block])
        nz, ny, nx = block_data.shape
        i0 = location[0] * file_data["nx1_mb"]
        j0 = location[1] * file_data["nx2_mb"]
        k0 = location[2] * file_data["nx3_mb"]
        result[k0:k0+nz, j0:j0+ny, i0:i0+nx] = block_data
    return result


def run_full_transparent_case(basename, origin, direction, resolution=8):
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}", f"mesh/nx2={resolution}",
        f"mesh/nx3={resolution}",
        "meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4",
        "laser/absorption_coefficient=0.0",
        f"laser/beam0_origin_x1={origin[0]}",
        f"laser/beam0_origin_x2={origin[1]}",
        f"laser/beam0_origin_x3={origin[2]}",
        f"laser/beam0_direction_x1={direction[0]}",
        f"laser/beam0_direction_x2={direction[1]}",
        f"laser/beam0_direction_x3={direction[2]}",
        "laser/report_diagnostics=false",
        "time/tlim=1.0e-7",
        "output1/dt=-1.0", "output2/dt=-1.0", "output3/dt=-1.0",
        "output4/dt=1.0e-7",
    ]
    assert testutils.run(input_file, flags=flags), (
        f"{basename} full-grid laser transport run failed.")
    return read_laser_binary(f"bin/{basename}.laser_full.00001.bin")


def sorted_tab(filename):
    data = athena_read.tab(filename)
    order = np.argsort(data["x1v"])
    return {name: np.atleast_1d(values)[order]
            for name, values in data.items()
            if np.ndim(values) > 0}


def run_absorption_case(basename, flags):
    common = [
        f"job/basename={basename}",
        "time/integrator=rk1",
        f"time/tlim={SOURCE_DT}",
        "output1/dt=-1.0",
        f"output2/dt={SOURCE_DT}",
        f"output3/dt={SOURCE_DT}",
    ]
    assert testutils.run(input_file, flags=common + flags), (
        f"{basename} laser deposition run failed.")
    laser = sorted_tab(f"tab/{basename}.laser.00001.tab")
    initial = sorted_tab(f"tab/{basename}.two_temperature.00000.tab")
    final = sorted_tab(f"tab/{basename}.two_temperature.00001.tab")
    return laser, initial, final


def constant_absorption_profile(x1, coefficient):
    dx = x1[1] - x1[0]
    left_edge = x1 - 0.5 * dx
    return (np.exp(-coefficient * left_edge)
            * -np.expm1(-coefficient * dx) / dx)


def inverse_bremsstrahlung_coefficient(ne, te, zeff, wavelength):
    electron_charge = 4.803204712570263e-10
    electron_mass = 9.1093837015e-28
    boltzmann = 1.380649e-16
    light_speed = 2.99792458e10
    critical_density = (electron_mass * np.pi * light_speed**2
                        / (electron_charge**2 * wavelength**2))
    density_ratio = min(ne / critical_density, 1.0 - 1.0e-12)
    coulomb_argument = (
        3.0 / (2.0 * zeff * electron_charge**3)
        * np.sqrt(boltzmann**3 * te**3 / (np.pi * ne)))
    coulomb_log = max(np.log(max(coulomb_argument, 1.0)), 1.0)
    collision_frequency = (
        4.0 / 3.0 * np.sqrt(2.0 * np.pi / electron_mass)
        * ne * zeff * electron_charge**4 * coulomb_log
        / (boltzmann * te)**1.5)
    group_speed = light_speed * np.sqrt(max(1.0 - density_ratio, 1.0e-12))
    return density_ratio * collision_frequency / group_speed


def test_run():
    try:
        resolution = 8
        cases = (
            (-0.31, (1.0, 0.37)),
            (0.22, (1.0, -0.65)),
            (-0.47, (1.0, 1.60)),
        )
        for case_index, (origin_y, direction) in enumerate(cases):
            expected = reference_2d_paths(origin_y, direction, resolution)
            measured = np.zeros_like(expected)
            counts = np.zeros_like(expected)
            for j in range(resolution):
                slice_y = -0.5 + (j + 0.5) / resolution
                basename = f"laser_dda_{case_index}_{j}"
                data = run_transparent_case(basename, [
                    f"mesh/nx1={resolution}",
                    f"mesh/nx2={resolution}",
                    "meshblock/nx1=4",
                    "meshblock/nx2=4",
                    f"laser/beam0_origin_x2={origin_y}",
                    f"laser/beam0_direction_x1={direction[0]}",
                    f"laser/beam0_direction_x2={direction[1]}",
                    f"output3/slice_x2={slice_y}",
                ])
                measured[j, :] = data["laser_path"]
                counts[j, :] = data["laser_ray_count"]
                assert np.count_nonzero(data["laser_q"]) == 0
                assert np.count_nonzero(data["laser_energy"]) == 0
                assert np.count_nonzero(data["laser_tau"]) == 0

            # The formatted-table input uses 12 digits after the decimal point.
            assert np.allclose(measured, expected, rtol=2.0e-11,
                               atol=2.0e-13)
            assert np.array_equal(counts, (expected > 0.0).astype(float))

        # Exercise three-dimensional indexing and transfers between eight
        # same-rank MeshBlocks with an exactly axis-aligned ray.
        data = run_transparent_case("laser_dda_3d", [
            "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8",
            "meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4",
            "laser/beam0_origin_x2=0.0",
            "laser/beam0_origin_x3=0.0",
            "laser/beam0_direction_x1=1.0",
            "laser/beam0_direction_x2=0.0",
            "laser/beam0_direction_x3=0.0",
        ])
        assert np.allclose(data["laser_path"], 1.0 / 8.0,
                           rtol=2.0e-11, atol=2.0e-13)
        assert np.array_equal(data["laser_ray_count"], np.ones(8))

        # Cover all direction signs, exact face/edge/corner starts, zero direction
        # components, diagonals, and a deterministic randomized ray set in 3D.
        geometry_cases = [
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            ((0.5, -0.5, 0.0), (0.0, 1.0, 0.0)),
            ((0.5, 0.5, 0.0), (0.0, -1.0, 0.0)),
            ((0.5, 0.0, -0.5), (0.0, 0.0, 1.0)),
            ((0.5, 0.0, 0.5), (0.0, 0.0, -1.0)),
            ((0.0, -0.5, -0.5), (1.0, 1.0, 1.0)),
            ((0.5, -0.25, -0.25), (1.0, 0.3, 0.2)),
            ((1.0e-14, -0.5 + 1.0e-14, -0.5 + 1.0e-14),
             (1.0, 0.71, 0.43)),
        ]
        random_generator = np.random.default_rng(271828)
        for _ in range(6):
            origin = np.array((0.0,
                               random_generator.uniform(-0.45, 0.45),
                               random_generator.uniform(-0.45, 0.45)))
            direction = np.array((random_generator.uniform(0.2, 1.0),
                                  random_generator.uniform(-1.0, 1.0),
                                  random_generator.uniform(-1.0, 1.0)))
            geometry_cases.append((origin, direction))

        for case_index, (origin, direction) in enumerate(geometry_cases):
            basename = f"laser_full_dda_{case_index}"
            file_data = run_full_transparent_case(
                basename, origin, direction)
            expected = reference_3d_paths(origin, direction, 8)
            measured = assemble_binary_field(file_data, "laser_path")
            counts = assemble_binary_field(file_data, "laser_ray_count")
            assert np.allclose(measured, expected, rtol=2.0e-6,
                               atol=2.0e-7)
            assert np.array_equal(counts, (expected > 0.0).astype(float))
            assert np.count_nonzero(
                assemble_binary_field(file_data, "laser_q")) == 0
            assert np.isclose(np.sum(measured), np.sum(expected),
                              rtol=2.0e-6, atol=2.0e-7)

        # Exact exponential attenuation, normalized across multiple ray packets,
        # deposits into total energy and the redundant 2T electron equation.
        coefficient = 2.0
        nrays = 13
        laser, initial, final = run_absorption_case("laser_absorption", [
            f"laser/absorption_coefficient={coefficient}",
            f"laser/beam0_nrays={nrays}",
        ])
        expected_q = constant_absorption_profile(laser["x1v"], coefficient)
        dx = laser["x1v"][1] - laser["x1v"][0]
        assert np.allclose(laser["laser_q"], expected_q,
                           rtol=2.0e-11, atol=2.0e-13)
        assert np.allclose(laser["laser_energy"], expected_q * SOURCE_DT,
                           rtol=2.0e-11, atol=2.0e-18)
        assert np.allclose(laser["laser_ray_count"], nrays)
        assert np.allclose(laser["laser_tau"], nrays * coefficient * dx)
        assert np.allclose(laser["laser_path"], nrays * dx)
        deposited_power = np.sum(laser["laser_q"]) * dx
        assert np.isclose(deposited_power, -np.expm1(-coefficient),
                          rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["eion"], initial["eion"],
                           rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["eele"],
                           initial["eele"] + expected_q * SOURCE_DT,
                           rtol=2.0e-12, atol=2.0e-13)
        expected_tele = 1.0 + 4.0 / 3.0 * expected_q * SOURCE_DT
        assert np.allclose(final["tele"], expected_tele,
                           rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["tion"], 1.0,
                           rtol=2.0e-12, atol=2.0e-13)

        # With B^2/2=5e17, the total-energy increment is below its floating-point
        # spacing. The dual 2T electron equation must still retain laser heating.
        laser, initial, final = run_absorption_case("laser_magnetic_dual", [
            "time/tlim=1.0", "time/nlim=1",
            "output2/dt=1.0e-20", "output3/dt=1.0e-20",
            f"laser/absorption_coefficient={coefficient}",
            "problem/bxl=1.0e9", "problem/bxr=1.0e9",
        ])
        assert np.allclose(final["eion"], initial["eion"],
                           rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["eele"],
                           initial["eele"] + laser["laser_energy"],
                           rtol=2.0e-12, atol=2.0e-13)

        # Total-energy targeting leaves the pre-source electron fraction unchanged.
        laser, initial, final = run_absorption_case("laser_total", [
            "laser/deposition_target=total",
            f"laser/absorption_coefficient={coefficient}",
        ])
        expected_q = constant_absorption_profile(laser["x1v"], coefficient)
        half_increment = 0.5 * expected_q * SOURCE_DT
        assert np.allclose(final["eion"], initial["eion"] + half_increment,
                           rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["eele"], initial["eele"] + half_increment,
                           rtol=2.0e-12, atol=2.0e-13)
        assert np.allclose(final["tion"], final["tele"],
                           rtol=2.0e-12, atol=2.0e-13)

        # Cumulative laser energy follows the same low-storage RK recurrence as u0.
        laser, _, _ = run_absorption_case("laser_rk3", [
            "time/integrator=rk3",
            f"laser/absorption_coefficient={coefficient}",
        ])
        assert np.allclose(laser["laser_energy"],
                           laser["laser_q"] * SOURCE_DT,
                           rtol=2.0e-12, atol=2.0e-18)

        # The configured low-power cutoff deposits, rather than discards, the
        # unresolved tail in an optically thick target.
        laser, _, _ = run_absorption_case("laser_thick", [
            "laser/absorption_coefficient=1000.0",
        ])
        assert np.isclose(np.sum(laser["laser_q"]) * dx, 1.0,
                          rtol=2.0e-12, atol=2.0e-13)
        assert np.all(laser["laser_q"] >= 0.0)
        assert np.allclose(laser["laser_energy"],
                           laser["laser_q"] * SOURCE_DT,
                           rtol=2.0e-12, atol=2.0e-18)

        # Evaluate the physical FLASH inverse-bremsstrahlung model below the
        # critical surface and compare its cell optical depths independently.
        length_scale = 1.0e-2
        density_scale = 1.0e-3
        temperature_scale = 1.0e6
        electrons_per_gram = 6.02214076e23
        wavelength_code = 1.0e-2
        laser, _, _ = run_absorption_case("laser_inverse_brems", [
            "laser/absorption_model=inverse_bremsstrahlung",
            f"laser/length_scale_cgs={length_scale}",
            f"laser/density_scale_cgs={density_scale}",
            f"laser/temperature_scale_cgs={temperature_scale}",
            f"laser/electron_number_per_gram={electrons_per_gram}",
            f"laser/beam0_wavelength={wavelength_code}",
        ])
        coefficient_cgs = inverse_bremsstrahlung_coefficient(
            density_scale * electrons_per_gram, temperature_scale, 1.0,
            wavelength_code * length_scale)
        coefficient_code = coefficient_cgs * length_scale
        expected_q = constant_absorption_profile(
            laser["x1v"], coefficient_code)
        assert np.allclose(laser["laser_q"], expected_q,
                           rtol=2.0e-11, atol=2.0e-13)
        assert np.allclose(laser["laser_tau"], coefficient_code * dx,
                           rtol=2.0e-11, atol=2.0e-13)
    except Exception as exc:
        pytest.fail(str(exc))
    finally:
        testutils.cleanup()
        shutil.rmtree("bin", ignore_errors=True)
