"""Analytic transfer, ray geometry, opacity, and native binary diagnostic tests."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from radiation_image import main, product_path
from radiation_io import paired_paths, read_pair, read_volume, unit_scales
from radiation_rays import LeafMesh, make_camera, trace_image
from radiation_transfer import (OpacityTable, PLANCK_INTEGRAL, PLANCK_PREFACTOR,
                                PlasmaEmission, formal_step, planck_groups)
from test_amr_planes import octree


def uniform(n=2):
    raw = {"n_mbs": 1, "mb_logical": np.array([[0, 0, 0, 0]]),
           "mb_index": np.array([[0, n-1, 0, n-1, 0, n-1]]),
           "mb_geometry": np.array([[-1., 1., -1., 1., -1., 1.]])}
    for i in (1, 2, 3):
        raw.update({f"Nx{i}": n, f"nx{i}_mb": n, f"x{i}min": -1., f"x{i}max": 1.})
    return raw


def write_table(path, absorption=2., emission=3.):
    values = {"transport": 1., "absorption": absorption, "emission": emission}
    path.write_text("athenak_opacity_table 1\ndimensions 2 2 2\n"
                    "density\n1 10\ntemperature\n1 10\ngroup_bound\n0 2 100\n"
                    + "\n".join(kind + "\n" + " ".join([str(value)] * 8)
                                for kind, value in values.items()) + "\nend\n")


def write_dump(path, kind, *, time=1., shape=(2, 2, 2), loc_size=8, var_size=4):
    """Small actual v1.1 record, including an unused field and separate tables."""
    fields = ({"dens": 2., "velx": 123., "s_00": .25, "s_01": .75,
               "s_02": .05, "s_03": .10, "s_04": .10} if kind == "fluid"
              else {"tion": 300., "tele": 1.})
    names = list(fields)
    header = ("<mesh>\nnghost=2\nnx1=2\nnx2=2\nnx3=2\n"
              "x1min=-1\nx1max=1\nx2min=-1\nx2max=1\nx3min=-1\nx3max=1\n"
              "<meshblock>\nnx1=2\nnx2=2\nnx3=2\n"
              "<units>\nlength_cgs=0.1\nmass_cgs=0.001\ntime_cgs=1e-5\nmu=1\n"
              "<materials>\nnmaterials=2\nscalar_index=0\n"
              "material0_name=CH\nmaterial1_name=He\n"
              "material0_opacity_table_file=material_tables/a.opacity\n"
              "material1_opacity_table_file=material_tables/b.opacity\n"
              "<thermal_radiation>\nopacity_model=mixed-table\nn_groups=2\n"
              "group_bound_0=0\ngroup_bound_1=2\ngroup_bound_2=100\n").encode()
    with path.open("wb") as stream:
        stream.write((f"Athena binary output version=1.1\n  size of preheader=5\n"
                      f"  time={time}\n  cycle=4\n  size of location={loc_size}\n"
                      f"  size of variable={var_size}\n  number of variables={len(names)}\n"
                      f"  variables: {' '.join(names)}\n  header offset={len(header)}\n").encode())
        stream.write(header)
        stream.write(struct.pack("=6i", 2, shape[0]+1, 2, shape[1]+1, 2, shape[2]+1))
        stream.write(struct.pack("=4i", 0, 0, 0, 0))
        stream.write(np.array([-1, 1, -1, 1, -1, 1], dtype=f"f{loc_size}").tobytes())
        for name in names:
            stream.write(np.full(shape[::-1], fields[name], dtype=f"f{var_size}").tobytes())


class TransferTests(unittest.TestCase):
    def test_analytic_slab_from_transparent_to_thick(self):
        tau = np.array([0., 1e-20, 1e-7, 0.5, 100., 1e6])
        actual = formal_step(2., tau, tau * 7., 1.)
        np.testing.assert_allclose(actual, 2 * np.exp(-tau) + 7 * -np.expm1(-tau), rtol=2e-15)
        self.assertEqual(actual[-1], 7.)
        # Zero absorption with finite emission must retain the j*ds limit.
        self.assertEqual(formal_step(2., np.array(0.), 3., 4.), 14.)
        self.assertAlmostEqual(float(formal_step(0., np.array(1e-20), 7e-20, 1.)), 7e-20, delta=1e-34)

    def test_blackbody_integral_and_truncated_bands(self):
        temperature = np.array([0., 1., 10.])
        total = planck_groups(temperature, [0., 0.1, 0.5, 3., 20., 1e4]).sum(axis=1)
        np.testing.assert_allclose(total, PLANCK_PREFACTOR * PLANCK_INTEGRAL * temperature**4, rtol=1e-12)
        self.assertLess(planck_groups(1., [1., 2.])[0], total[1])
        # Independent high-order quadrature checks Rayleigh-Jeans and Wien bands.
        nodes, weights = np.polynomial.legendre.leggauss(100)
        for low, high in ((1e-5, .01), (.45, .55), (1., 5.), (60., 61.)):
            x = low + (nodes + 1) * (high - low) / 2
            expected = PLANCK_PREFACTOR * np.sum(weights * x**3 / np.expm1(x)) * (high - low) / 2
            np.testing.assert_allclose(planck_groups(1., [low, high])[0], expected, rtol=3e-11)


class RayTests(unittest.TestCase):
    def test_uniform_amr_slab_axis_and_oblique_chords(self):
        raw = octree()
        mesh = LeafMesh(raw)
        absorption = np.array([0., .3, 1e5])
        emissivity = np.array([2., .9, 7e5])
        background = np.array([.1, .2, .3])
        def coefficients(block):
            return np.tile(absorption, (8, 1)), np.tile(emissivity, (8, 1))
        for direction in ([0, 0, 1], [0, 0, -1], [1, 2, 3], [-2, 1, -3]):
            camera = make_camera(mesh, (13, 11), direction)
            result = trace_image(mesh, camera, coefficients, background)
            # Independent intersection with the full cube.
            chords = []
            for point in camera.origins:
                lo, hi = -np.inf, np.inf
                for p, d in zip(point, camera.direction):
                    if d:
                        a, b = (-1-p)/d, (1-p)/d
                        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
                    elif not -1 <= p < 1:
                        hi = -np.inf
                chords.append(max(hi-lo, 0))
            chords = np.array(chords).reshape(11, 13)
            expected = formal_step(background[:, None, None], absorption[:, None, None],
                                   emissivity[:, None, None], chords)
            np.testing.assert_allclose(result["intensity"], expected, rtol=2e-12, atol=1e-12)
            np.testing.assert_allclose(result["path_length"], chords, atol=1e-13)
            np.testing.assert_allclose(result["optical_depth"], absorption[:, None, None] * chords, atol=1e-9)

    def test_two_layers_reverse_observer_changes_occlusion(self):
        mesh = LeafMesh(uniform())
        alpha = np.repeat([.7, 2.], 4)[:, None]
        emission = alpha * np.repeat([3., 9.], 4)[:, None]
        results = []
        for sign in (1, -1):
            camera = make_camera(mesh, (3, 3), (0, 0, sign))
            result = trace_image(mesh, camera, lambda b: (alpha, emission), [1.])
            sequence = [0, 4] if sign == 1 else [4, 0]
            expected = 1.
            for cell in sequence:
                expected = formal_step(expected, alpha[cell], emission[cell], 1.)
            np.testing.assert_allclose(result["intensity"], float(expected[0]), rtol=2e-15)
            results.append(result["intensity"][0, 0, 0])
        self.assertGreater(results[0], results[1])

    def test_nonuniform_amr_matches_exact_fine_representation(self):
        raw = octree()
        fine_alpha, fine_emission = np.empty((8, 8, 8)), np.empty((8, 8, 8))
        coefficients = []
        for b, (x, y, z, level) in enumerate(raw["mb_logical"]):
            repeat = 2**(1-level)
            alpha = raw["mb_data"]["dens"][b] / 10
            emission = alpha * raw["mb_data"]["tele"][b]
            coefficients.append((alpha.reshape(-1, 1), emission.reshape(-1, 1)))
            region = tuple(slice(i * 2 * repeat, (i+1) * 2 * repeat) for i in (z, y, x))
            for target, value in ((fine_alpha, alpha), (fine_emission, emission)):
                target[region] = value.repeat(repeat, 0).repeat(repeat, 1).repeat(repeat, 2)
        mesh, fine = LeafMesh(raw), LeafMesh(uniform(8))
        camera = make_camera(mesh, (15, 13), (1, -2, 3))
        actual = trace_image(mesh, camera, lambda b: coefficients[b], [0.])
        expected = trace_image(fine, camera, lambda b: (fine_alpha.reshape(-1, 1),
                                                       fine_emission.reshape(-1, 1)), [0.])
        np.testing.assert_allclose(actual["intensity"], expected["intensity"], rtol=2e-12, atol=1e-11)
        np.testing.assert_allclose(actual["optical_depth"], expected["optical_depth"], atol=1e-12)
        # Disk block ordering is unrelated to ray depth.
        mesh.bounds = mesh.bounds[::-1].copy()
        reverse = trace_image(mesh, camera, lambda b: coefficients[-1-b], [0.])
        np.testing.assert_array_equal(actual["intensity"], reverse["intensity"])

    def test_geometry_rejects_holes_parents_and_slices(self):
        raw = octree()
        hole = deepcopy(raw)
        hole["n_mbs"] -= 1
        for key in ("mb_geometry", "mb_logical", "mb_index"):
            hole[key] = hole[key][:-1]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            LeafMesh(hole)
        overlap = deepcopy(raw)
        overlap["n_mbs"] += 1
        for key, value in (("mb_geometry", [0, 1, 0, 1, 0, 1]),
                           ("mb_index", [0, 1, 0, 1, 0, 1]), ("mb_logical", [1, 1, 1, 0])):
            overlap[key] = np.vstack((overlap[key], value))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            LeafMesh(overlap)
        raw["mb_index"][0, 5] = 0
        with self.assertRaisesRegex(ValueError, "full-volume"):
            LeafMesh(raw)

    def test_camera_and_missed_rays(self):
        mesh = LeafMesh(uniform())
        with self.assertRaisesRegex(ValueError, "parallel"):
            make_camera(mesh, observer=(1, 0, 0))
        camera = make_camera(mesh, (1, 1), extent=[2, 3, 2, 3])
        result = trace_image(mesh, camera, lambda b: self.fail("Missed rays must not load coefficients"), [5.])
        self.assertEqual(result["intensity"][0, 0, 0], 5)
        self.assertEqual(result["optical_depth"][0, 0, 0], 0)
        np.testing.assert_array_equal(camera.horizontal, [0, 1, 0])
        np.testing.assert_array_equal(camera.vertical, [1, 0, 0])


class OpacityTests(unittest.TestCase):
    def test_log_coordinates_geometric_values_and_zero_fallback(self):
        axis = np.array([1., 10.])
        grid = (axis[:, None]**2 * axis[None, :]**3)[None, :, :]
        table = OpacityTable(axis, axis, np.array([0., 1.]), {"absorption": grid})
        actual = table.evaluate("absorption", np.array([np.sqrt(10), .01, 100]),
                                np.array([np.sqrt(10), .01, 100]), [0])[:, 0]
        np.testing.assert_allclose(actual, [10**2.5, 1, 1e5])
        table.values["absorption"][0, 0, 0] = 0
        actual = table.evaluate("absorption", np.array([np.sqrt(10)]), np.array([np.sqrt(10)]), [0])
        self.assertAlmostEqual(actual[0, 0], grid.mean())

    def test_partial_density_mixture_and_emission_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            for name in ("a", "b"):
                write_table(root / f"material_tables/{name}.opacity")
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin")
            model = PlasmaEmission(fluid, thermal, scales, root)
            # Set kappa_a(rho)=rho so partial-density evaluation is distinguishable
            # from evaluating both materials at the total density.
            for table in model.tables:
                table.values["absorption"][:] = table.density[None, :, None]
            alpha, emissivity = model.coefficients(0)
            # rho_CH=.5 clamps to 1; rho_He=1.5 interpolates to 1.5.
            np.testing.assert_allclose(alpha, .5 * 1 + 1.5 * 1.5)
            b = planck_groups(scales["temperature_kelvin"], model.bounds * scales["temperature_kelvin"])
            np.testing.assert_allclose(emissivity, np.tile(2 * 3 * b, (8, 1)))
            self.assertEqual(model.clamp_counts[0]["density_low"], 8)
            # All-zero material fractions use the last material, as in the solver.
            fluid["mb_data"]["s_00"] = np.zeros((1, 2, 2, 2))
            fluid["mb_data"]["s_01"] = np.zeros((1, 2, 2, 2))
            np.testing.assert_allclose(model.coefficients(0)[0], 4.)
            del model, fluid, thermal  # Release mmap handles before NFS cleanup.

    def test_ch_only_retains_partial_density_and_helium_is_transparent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            write_table(root / "material_tables/a.opacity", absorption=2, emission=3)
            # No He table is needed when only CH contributes.
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin")
            model = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"])
            alpha, emissivity = model.coefficients(0)
            b = planck_groups(scales["temperature_kelvin"], model.bounds * scales["temperature_kelvin"])
            # rho=2, Y_CH=.25: neither opacity nor emission may use total rho.
            np.testing.assert_allclose(alpha, 2 * .25 * 2)
            np.testing.assert_allclose(emissivity, np.tile(2 * .25 * 3 * b, (8, 1)))
            self.assertEqual(model.material_indices, [0])
            self.assertEqual(model.material_names, ["CH"])
            self.assertEqual(len(model.provenance), 1)
            self.assertFalse(model.clamp_counts[1]["enabled"])
            self.assertEqual(model.clamp_counts[1]["sampled_cells"], 0)
            # Verify the density coordinate too, not just the prefactor.
            model.tables[0].values["absorption"][:] = model.tables[0].density[None, :, None]
            fluid["mb_data"]["s_00"] = np.full((1, 2, 2, 2), .75)
            fluid["mb_data"]["s_01"] = np.full((1, 2, 2, 2), .25)
            np.testing.assert_allclose(model.coefficients(0)[0], 1.5**2)
            # Selecting CH must not alter normalization/fallback over all materials.
            fluid["mb_data"]["s_00"] = np.zeros((1, 2, 2, 2))
            for helium in (0., 1.):
                fluid["mb_data"]["s_01"] = np.full((1, 2, 2, 2), helium)
                alpha, emissivity = model.coefficients(0)
                np.testing.assert_array_equal(alpha, 0.)
                np.testing.assert_array_equal(emissivity, 0.)
                mesh = LeafMesh(fluid)
                result = trace_image(mesh, make_camera(mesh, (3, 3)), model.coefficients, [4., 7.])
                np.testing.assert_array_equal(result["optical_depth"], 0.)
                np.testing.assert_array_equal(result["intensity"], np.broadcast_to(
                    np.array([4., 7.])[:, None, None], (2, 3, 3)))
            # The pure-CH limit remains the usual material opacity.
            fluid["mb_data"]["s_00"] = np.ones((1, 2, 2, 2))
            fluid["mb_data"]["s_01"] = np.zeros((1, 2, 2, 2))
            np.testing.assert_allclose(model.coefficients(0)[0], 2**2)
            del model, fluid, thermal

    def test_material_selection_names_indices_and_invalid_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            write_table(root / "material_tables/a.opacity")
            write_table(root / "material_tables/b.opacity", absorption=5, emission=7)
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin")
            for choice, index, expected_alpha in ((["ch"], 0, 1.), (["0"], 0, 1.),
                                                 (["He"], 1, 7.5)):
                model = PlasmaEmission(fluid, thermal, scales, root, material_selection=choice)
                self.assertEqual(model.material_indices, [index])
                np.testing.assert_allclose(model.coefficients(0)[0], expected_alpha)
                del model
            for choice in (["Au"], ["-1"], ["2"], ["CH", "0"], []):
                with self.assertRaises(ValueError):
                    PlasmaEmission(fluid, thermal, scales, root, material_selection=choice)
            del fluid, thermal

    def test_origin_emission_partition_keeps_full_ch_absorption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            write_table(root / "material_tables/a.opacity")
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin",
                                               ["s_02", "s_03", "s_04"])
            whole = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"])
            shells = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"],
                                     emission_tracers=["s_02", "s_03"])
            corona = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"],
                                     emission_tracers=["s_04"])
            # Origin ratios must remain correct even if material fractions need
            # normalization. Both sources share the saved mixture's opacity.
            for helium in (.75, 1.):
                fluid["mb_data"]["s_01"] = np.full((1, 2, 2, 2), helium)
                alpha, emission = whole.coefficients(0)
                shell_alpha, shell_emission = shells.coefficients(0)
                corona_alpha, corona_emission = corona.coefficients(0)
                np.testing.assert_array_equal(shell_alpha, alpha)
                np.testing.assert_array_equal(corona_alpha, alpha)
                np.testing.assert_allclose(shell_emission, .6 * emission, rtol=1e-7)
                np.testing.assert_allclose(shell_emission + corona_emission, emission, rtol=1e-7)
            # Pure corona emits nothing in shell mode, but remains absorbing.
            fluid["mb_data"]["s_02"] = np.zeros((1, 2, 2, 2))
            fluid["mb_data"]["s_03"] = np.zeros((1, 2, 2, 2))
            np.testing.assert_array_equal(shells.coefficients(0)[1], 0.)
            np.testing.assert_array_equal(shells.coefficients(0)[0], alpha)
            # Roundoff-sized origin overshoot is bounded by the total CH source.
            fluid["mb_data"]["s_02"][:] = .25000001
            np.testing.assert_array_equal(shells.coefficients(0)[1], emission)
            # A zero material fraction is nonemitting even with nonzero tags.
            fluid["mb_data"]["s_00"] = np.zeros((1, 2, 2, 2))
            np.testing.assert_array_equal(shells.coefficients(0), 0.)
            del whole, shells, corona, fluid, thermal

    def test_corona_foreground_absorbs_without_emitting_in_shell_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            write_table(root / "material_tables/a.opacity")
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin", ["s_02", "s_03"])
            # Each z layer is one code length thick: shell behind pure corona.
            fluid["mb_data"]["s_02"] = np.zeros((1, 2, 2, 2))
            fluid["mb_data"]["s_02"][:, 0] = .25
            fluid["mb_data"]["s_03"] = np.zeros((1, 2, 2, 2))
            model = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"],
                                    emission_tracers=["s_02", "s_03"])
            mesh = LeafMesh(fluid)
            background = np.array([4., 7.])
            actual = trace_image(mesh, make_camera(mesh, (3, 3)), model.coefficients, background)
            b = planck_groups(scales["temperature_kelvin"], model.bounds * scales["temperature_kelvin"])
            expected = background * np.exp(-2) + 1.5 * b * -np.expm1(-1) * np.exp(-1)
            np.testing.assert_allclose(actual["intensity"], np.broadcast_to(expected[:, None, None], (2, 3, 3)))
            np.testing.assert_allclose(actual["optical_depth"], 2.)
            del model, fluid, thermal

    def test_origin_selection_rejects_ambiguous_or_invalid_tracers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "material_tables").mkdir()
            write_table(root / "material_tables/a.opacity")
            write_dump(root / "f.bin", "fluid")
            write_dump(root / "t.bin", "thermal")
            fluid, thermal, scales = read_pair(root / "f.bin", root / "t.bin", ["s_02", "s_03"])
            with self.assertRaisesRegex(ValueError, "exactly one selected material"):
                PlasmaEmission(fluid, thermal, scales, root, emission_tracers=["s_02"])
            for selection in ([], ["s_02", "s_02"], ["s_00"], ["dens"], ["s_04"]):
                with self.assertRaises(ValueError):
                    PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"],
                                   emission_tracers=selection)
            model = PlasmaEmission(fluid, thermal, scales, root, material_selection=["CH"],
                                    emission_tracers=["s_02", "s_03"])
            fluid["mb_data"]["s_02"] = np.full((1, 2, 2, 2), np.nan)
            with self.assertRaisesRegex(ValueError, "Invalid emission origin"):
                model.coefficients(0)
            del model, fluid, thermal


class BinaryAndCLITests(unittest.TestCase):
    def test_native_binary_precisions_and_pair_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for loc, var in ((4, 8), (8, 4)):
                write_dump(root / "f.bin", "fluid", loc_size=loc, var_size=var)
                write_dump(root / "t.bin", "thermal", loc_size=loc, var_size=var)
                fluid, thermal, _ = read_pair(root / "f.bin", root / "t.bin")
                self.assertEqual(set(fluid["mb_data"]), {"dens", "s_00", "s_01"})
                self.assertFalse(fluid["mb_data"]["dens"].flags.writeable)
                np.testing.assert_array_equal(fluid["mb_data"]["dens"], 2.)
                self.assertEqual(thermal["mb_data"]["tele"].shape, (1, 2, 2, 2))
                del fluid, thermal
            write_dump(root / "t.bin", "thermal", time=2.)
            with self.assertRaisesRegex(ValueError, "mismatched time"):
                read_pair(root / "f.bin", root / "t.bin")
            write_dump(root / "f.bin", "fluid", shape=(2, 2, 1))
            with self.assertRaisesRegex(ValueError, "sliced"):
                read_volume(root / "f.bin", ["dens"])

    def test_cli_outputs_have_physical_units_and_reproducible_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "material_tables").mkdir()
            for name in ("a", "b"):
                write_table(root / f"material_tables/{name}.opacity")
            write_dump(root / "bin/test.fluid.00000.bin", "fluid")
            write_dump(root / "bin/test.three_t.00000.bin", "thermal")
            main([str(root), "--pixels", "3", "5", "--groups", "1", "--no-plot"])
            output = root / "radiation/test_radiation_00000"
            self.assertEqual(product_path(root / "a.b_radiation_00000", ".npz").name,
                             "a.b_radiation_00000.npz")
            with np.load(output.with_suffix(".npz")) as saved:
                self.assertEqual(saved["intensity_cgs"].shape, (1, 5, 3))
                np.testing.assert_allclose(saved["optical_depth"], 2 * 2 * 2)
                np.testing.assert_allclose(saved["path_length_cm"], .2)
                np.testing.assert_allclose(saved["directional_power_cgs"],
                                           saved["intensity_cgs"].sum(axis=(1, 2)) * saved["pixel_area_cm2"])
                self.assertEqual(saved["group_indices"].tolist(), [1])
                snapshot = read_volume(root / "bin/test.three_t.00000.bin", ["tele"])
                temperature = unit_scales(snapshot["header"])["temperature_kelvin"]
                source = planck_groups(temperature, np.array([0, 2, 100]) * temperature)[1]
                np.testing.assert_allclose(saved["intensity_cgs"], 1.5 * source * -np.expm1(-8))
                del snapshot
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["camera"]["horizontal_direction"], [0., 1., 0.])
            self.assertEqual(len(metadata["tables"][0]["sha256"]), 64)
            main([str(root), "--pixels", "3", "5", "--groups", "1", "--materials", "CH",
                  "--output-dir", str(root / "ch_only"), "--no-plot"])
            ch_output = root / "ch_only/test_radiation_00000"
            with np.load(ch_output.with_suffix(".npz")) as saved:
                np.testing.assert_array_equal(saved["material_indices"], [0])
                np.testing.assert_array_equal(saved["material_names"], ["CH"])
                np.testing.assert_allclose(saved["optical_depth"], 2.)
                np.testing.assert_allclose(saved["intensity_cgs"], 1.5 * source * -np.expm1(-2))
            ch_metadata = json.loads(ch_output.with_suffix(".json").read_text())
            self.assertEqual(ch_metadata["material_names"], ["CH"])
            self.assertEqual(len(ch_metadata["tables"]), 1)
            self.assertEqual(ch_metadata["tables"][0]["material"], "CH")
            main([str(root), "--pixels", "3", "5", "--groups", "1", "--materials", "CH",
                  "--emission-tracers", "s_02", "s_03", "--output-dir", str(root / "shells"), "--no-plot"])
            shell_output = root / "shells/test_radiation_00000"
            with np.load(shell_output.with_suffix(".npz")) as saved:
                np.testing.assert_array_equal(saved["emission_tracers"], ["s_02", "s_03"])
                np.testing.assert_allclose(saved["optical_depth"], 2.)
                np.testing.assert_allclose(saved["intensity_cgs"], .6 * 1.5 * source * -np.expm1(-2), rtol=1e-7)
            shell_metadata = json.loads(shell_output.with_suffix(".json").read_text())
            self.assertEqual(shell_metadata["emission_selection"]["tracers"], ["s_02", "s_03"])
            self.assertEqual(shell_metadata["emission_selection"]["material"], "CH")
            # An incomplete newest output must not silently select an earlier time.
            write_dump(root / "bin/test.fluid.00001.bin", "fluid")
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                paired_paths(root)
            self.assertEqual(len(paired_paths(root, "0")), 1)


if __name__ == "__main__":
    unittest.main()
