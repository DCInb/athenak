"""Check AMR slice geometry using a small, analytically specified octree."""

from copy import deepcopy
from itertools import product
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from amr_planes import load_amr_plane


SCALES = {"length_um": 1.0, "density_gcc": 1.0,
          "velocity_kms": 1.0, "temperature_ev": 1.0}


def octree():
    """A 4^3 root grid with only its positive octant refined to level one."""
    blocks = [(x, y, z, 0) for x, y, z in product(range(2), repeat=3)
              if (x, y, z) != (1, 1, 1)]
    blocks += [(x, y, z, 1) for x, y, z in product(range(2, 4), repeat=3)]
    raw = {"n_mbs": len(blocks), "mb_logical": np.array(blocks),
           "mb_index": np.tile([0, 1, 0, 1, 0, 1], (len(blocks), 1)),
           "mb_geometry": [], "mb_data": {}}
    for axis in (1, 2, 3):
        raw.update({f"Nx{axis}": 4, f"nx{axis}_mb": 2,
                    f"x{axis}min": -1.0, f"x{axis}max": 1.0})
    for name in ("dens", "velx", "vely", "velz", "s_02", "s_03", "tion", "tele"):
        raw["mb_data"][name] = []
    for x, y, z, level in blocks:
        size = 1.0 / 2 ** level
        lower = -1.0 + np.array([x, y, z]) * size
        raw["mb_geometry"].append(np.column_stack((lower, lower + size)).ravel())
        coordinates = [value + np.array([0.25, 0.75]) * size for value in lower]
        zz, yy, xx = np.meshgrid(coordinates[2], coordinates[1], coordinates[0], indexing="ij")
        values = {"dens": 10 + xx + 2 * yy + 4 * zz,
                  "velx": xx, "vely": yy, "velz": zz,
                  "s_02": np.ones_like(xx), "s_03": np.zeros_like(xx),
                  "tion": 100 + xx, "tele": 200 + yy + 2 * zz}
        for name, value in values.items():
            raw["mb_data"][name].append(value)
    raw["mb_geometry"] = np.asarray(raw["mb_geometry"])
    return raw


def saved_xy(raw):
    saved = deepcopy(raw)
    keep = np.flatnonzero((raw["mb_geometry"][:, 4] <= 0) & (raw["mb_geometry"][:, 5] > 0))
    # In this octree every selected block starts at z=0, so its first cell is saved.
    saved["n_mbs"] = len(keep)
    for name in ("mb_geometry", "mb_index", "mb_logical"):
        saved[name] = saved[name][keep].copy()
    saved["mb_index"][:, 4:6] = 0
    for name, blocks in raw["mb_data"].items():
        saved["mb_data"][name] = [blocks[block][0:1, :, :] for block in keep]
    return saved


class AMRPlaneTests(unittest.TestCase):
    def test_xy_native_values_and_geometry(self):
        raw = octree()
        data, xedges, yedges, normal, mesh = load_amr_plane(raw, raw, "xy", SCALES)
        self.assertEqual(data["density"].shape, (8, 8))
        np.testing.assert_array_equal(xedges, np.linspace(-1, 1, 9))
        np.testing.assert_array_equal(yedges, xedges)
        # Native coarse values occupy 2x2 pixels; finest cells remain independent.
        np.testing.assert_array_equal(data["density"][:2, :2], np.full((2, 2), 8.75))
        self.assertEqual(data["density"][4, 4], 10.875)
        self.assertEqual(data["density"][5, 4], 11.125)
        self.assertEqual(data["cell_x"][0, 0], -0.75)
        self.assertEqual(data["cell_y"][0, 0], -0.75)
        self.assertEqual(data["cell_x"][4, 4], 0.125)
        self.assertIsNone(normal)
        self.assertEqual(mesh["normal_cell_center_range_um"], [0.125, 0.25])
        self.assertEqual(mesh["cell_width_range_um"]["x1"], [0.25, 0.5])
        self.assertEqual(mesh["native_slice_cells"], 28)
        self.assertEqual(mesh["slice_blocks_by_level"], {"0": 3, "1": 4})

    def test_xz_axis_orientation_and_normal_selection(self):
        raw = octree()
        data, _, _, _, _ = load_amr_plane(raw, raw, "xz", SCALES)
        # At x=-0.75, y=+0.25, z=-0.75: rho=10+x+2y+4z.
        self.assertEqual(data["density"][0, 0], 6.75)
        self.assertEqual(data["density"][4, 4], 10.875)
        np.testing.assert_array_equal(data["horizontal_velocity"], data["v3"])

    def test_saved_xy_agrees_with_volume(self):
        raw = octree()
        saved = saved_xy(raw)
        volume = load_amr_plane(raw, raw, "xy", SCALES)
        sliced = load_amr_plane(saved, saved, "xy", SCALES)
        for name in volume[0]:
            np.testing.assert_array_equal(volume[0][name], sliced[0][name])
        self.assertEqual(volume[4], sliced[4])

    def test_hole_is_rejected(self):
        raw = saved_xy(octree())
        raw["n_mbs"] -= 1
        for name in ("mb_logical", "mb_index", "mb_geometry"):
            raw[name] = raw[name][:-1]
        for name in raw["mb_data"]:
            raw["mb_data"][name] = raw["mb_data"][name][:-1]
        with self.assertRaisesRegex(ValueError, "Incomplete AMR plane"):
            load_amr_plane(raw, raw, "xy", SCALES)

    def test_overlap_is_rejected(self):
        raw = saved_xy(octree())
        raw["n_mbs"] += 1
        for name in ("mb_logical", "mb_index", "mb_geometry"):
            raw[name] = np.concatenate((raw[name], raw[name][:1]))
        for name in raw["mb_data"]:
            raw["mb_data"][name].append(raw["mb_data"][name][0])
        with self.assertRaisesRegex(ValueError, "Overlapping AMR leaf cells"):
            load_amr_plane(raw, raw, "xy", SCALES)

    def test_nonfinite_value_is_rejected(self):
        raw = saved_xy(octree())
        raw["mb_data"]["dens"][0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_amr_plane(raw, raw, "xy", SCALES)


if __name__ == "__main__":
    unittest.main()
