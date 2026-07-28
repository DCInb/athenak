"""Focused tests for the DCI laser-ray plotting tool."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_laser_rays as rays


def synthetic_collective() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    shape = (4, 4, 4)  # z, y, x
    z, y, x = np.indices(shape)
    path = 1.0+x+10.0*y+100.0*z
    global_fields = {
        "laser_path": path,
        "laser_ray_count": 2.0+path,
        "laser_dir1": -path,
        "laser_dir2": 0.25*path,
        "laser_dir3": 0.50*path,
    }
    logical = []
    indices = []
    geometry = []
    fields = {name: [] for name in global_fields}
    for block_z in range(2):
        for block_y in range(2):
            for block_x in range(2):
                logical.append((block_x, block_y, block_z, 0))
                indices.append((0, 1, 0, 1, 0, 1))
                geometry.append((
                    -1.0+block_x, -1.0+block_x+1.0,
                    -1.0+block_y, -1.0+block_y+1.0,
                    -1.0+block_z, -1.0+block_z+1.0,
                ))
                block_slice = np.s_[
                    2*block_z:2*block_z+2,
                    2*block_y:2*block_y+2,
                    2*block_x:2*block_x+2,
                ]
                for name, values in global_fields.items():
                    fields[name].append(np.asarray(values[block_slice], dtype=np.float64))
    raw: dict[str, object] = {
        "header": [
            "<units>", "length_cgs = 1.0e-1", "time_cgs = 1.0e-9",
            "<problem>", "inner_radius = 0.8", "outer_radius = 1.0",
            "opening_half_angle_deg = 50.0",
        ],
        "time": 0.25,
        "cycle": 50,
        "var_names": list(global_fields),
        "Nx1": 4, "Nx2": 4, "Nx3": 4,
        "nx1_mb": 2, "nx2_mb": 2, "nx3_mb": 2,
        "n_mbs": 8,
        "x1min": -1.0, "x1max": 1.0,
        "x2min": -1.0, "x2max": 1.0,
        "x3min": -1.0, "x3max": 1.0,
        "mb_logical": np.asarray(logical, dtype=np.int64),
        "mb_index": np.asarray(indices, dtype=np.int64),
        "mb_geometry": np.asarray(geometry, dtype=np.float64),
        "mb_data": fields,
    }
    return raw, global_fields


def test_collective_xy_and_xz_plane_assembly():
    raw, global_fields = synthetic_collective()
    names = ("laser_path", "laser_dir1")
    xy, x_edges, y_edges, xy_location = rays.assemble_fields(raw, "xy", 0.0, names)
    xz, _, z_edges, xz_location = rays.assemble_fields(raw, "xz", 0.0, names)

    # AthenaK selects the upper cell when zero lies on an even-grid face: index 2.
    np.testing.assert_allclose(xy["laser_path"], global_fields["laser_path"][2, :, :])
    np.testing.assert_allclose(xz["laser_path"], global_fields["laser_path"][:, 2, :])
    np.testing.assert_allclose(x_edges, np.linspace(-1.0, 1.0, 5))
    np.testing.assert_allclose(y_edges, np.linspace(-1.0, 1.0, 5))
    np.testing.assert_allclose(z_edges, np.linspace(-1.0, 1.0, 5))
    assert xy_location == pytest.approx(0.25)
    assert xz_location == pytest.approx(0.25)


def test_pre_sliced_collective_xy_plane_assembly():
    raw, global_fields = synthetic_collective()
    selected = [
        index for index, logical in enumerate(raw["mb_logical"])
        if int(logical[2]) == 1
    ]
    raw["n_mbs"] = len(selected)
    raw["mb_logical"] = raw["mb_logical"][selected]
    raw["mb_index"] = raw["mb_index"][selected].copy()
    raw["mb_index"][:, 4:6] = 0
    raw["mb_geometry"] = raw["mb_geometry"][selected]
    for name, values in raw["mb_data"].items():
        raw["mb_data"][name] = [values[index][0:1, :, :] for index in selected]

    fields, _, _, location = rays.assemble_fields(
        raw, "xy", 0.0, ("laser_path",),
    )
    np.testing.assert_allclose(
        fields["laser_path"], global_fields["laser_path"][2, :, :],
    )
    assert location == pytest.approx(0.25)


def test_load_plane_recovers_path_weighted_direction(monkeypatch, tmp_path):
    raw, global_fields = synthetic_collective()
    monkeypatch.setattr(rays, "read_binary", lambda _: raw)
    plane = rays.load_laser_plane(tmp_path/"synthetic.laser.00000.bin", "xz", 0.0)

    np.testing.assert_allclose(plane.values, global_fields["laser_path"][:, 2, :])
    np.testing.assert_allclose(plane.direction_x, -1.0)
    np.testing.assert_allclose(plane.direction_y, 0.50)
    np.testing.assert_allclose(plane.direction_coherence, np.sqrt(1.3125))
    assert plane.time_ns == pytest.approx(0.25)
    assert plane.normal_location_mm == pytest.approx(0.25)
    counts = rays.load_laser_plane(
        tmp_path/"synthetic.laser.00000.bin", "xz", 0.0, "ray-count",
    )
    np.testing.assert_allclose(
        counts.values, global_fields["laser_ray_count"][:, 2, :],
    )


def test_incomplete_rank_shard_is_rejected():
    raw, _ = synthetic_collective()
    raw["n_mbs"] = 7
    raw["mb_logical"] = raw["mb_logical"][:7]
    raw["mb_index"] = raw["mb_index"][:7]
    raw["mb_geometry"] = raw["mb_geometry"][:7]
    for values in raw["mb_data"].values():
        del values[7:]
    with pytest.raises(RuntimeError, match="complete collective"):
        rays.assemble_fields(raw, "xy", 0.0, ("laser_path",))


def test_shell_cap_outline_matches_central_geometry():
    segments = rays.shell_cap_outline(0.8, 1.0, 50.0, 0.0, samples=101)
    assert len(segments) == 4
    inner_x, inner_y = segments[0]
    assert inner_x[50] == pytest.approx(-0.8)
    assert inner_y[0] == pytest.approx(-0.8*np.sin(np.deg2rad(50.0)))
    assert inner_x[0] == pytest.approx(-0.8*np.cos(np.deg2rad(50.0)))


def test_directory_resolution_prefers_requested_plane(tmp_path):
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    xy = bin_dir/"dci_3d.laser_xy.00001.bin"
    xz = bin_dir/"dci_3d.laser_xz.00001.bin"
    volume = bin_dir/"dci_3d.laser.00001.bin"
    for path in (xy, xz, volume):
        path.touch()
    assert rays.resolve_inputs((str(tmp_path),), "xy") == [xy.resolve()]
    assert rays.resolve_inputs((str(tmp_path),), "xz") == [xz.resolve()]


def test_png_render_is_deterministic(monkeypatch, tmp_path):
    raw, _ = synthetic_collective()
    monkeypatch.setattr(rays, "read_binary", lambda _: raw)
    plane = rays.load_laser_plane(tmp_path/"synthetic.laser.00000.bin", "xy", 0.0)
    options = rays.RenderOptions(dpi=72, shell_outline=True)
    first = rays.render_laser_plane(plane, tmp_path/"first.png", options)
    second = rays.render_laser_plane(plane, tmp_path/"second.png", options)
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
    assert first_hash == second_hash
    assert first.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_zero_ray_frame_renders_without_fabricating_paths(monkeypatch, tmp_path):
    raw, _ = synthetic_collective()
    monkeypatch.setattr(rays, "read_binary", lambda _: raw)
    plane = rays.load_laser_plane(tmp_path/"synthetic.laser.00000.bin", "xy", 0.0)
    zero = np.zeros_like(plane.values)
    quiet = replace(
        plane, values=zero, direction_x=zero, direction_y=zero,
        direction_coherence=zero,
    )
    output = rays.render_laser_plane(
        quiet, tmp_path/"quiet.png", rays.RenderOptions(dpi=72),
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_streamline_overlay_renders(monkeypatch, tmp_path):
    raw, _ = synthetic_collective()
    monkeypatch.setattr(rays, "read_binary", lambda _: raw)
    plane = rays.load_laser_plane(tmp_path/"synthetic.laser.00000.bin", "xz", 0.0)
    output = rays.render_laser_plane(
        plane, tmp_path/"stream.png",
        rays.RenderOptions(dpi=72, vectors="stream", stream_max_resolution=16),
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
