#!/usr/bin/env python3
"""Check the initial binary dumps against the requested physical setup and mesh.

Usage: python3 verify_initial.py RUN_DIRECTORY [--nx 108]
Use --nx 96 or --nx 100 for archived initializations.
Requires NumPy. Reuses AthenaK's binary reader; HDF5 support is not needed.
Physical cap parameters and unit conversions come from each dump's saved input.
Mass integrals use individual leaf-block volumes for uniform or refined meshes.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis" / "python"))
try:
    from bin_convert import read_binary
except ModuleNotFoundError as exc:
    # The shared converter imports h5py only for its separate HDF5 conversion functions.
    if exc.name != "h5py":
        raise
    sys.modules["h5py"] = types.ModuleType("h5py")
    from bin_convert import read_binary


def require(condition, message):
    if not condition:
        raise ValueError(message)


def header_value(header, block, name, default=None):
    """Read the saved input, so archived corona-free initializations remain valid."""
    current = None
    for line in header:
        line = line.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            current = line[1:-1]
        elif current == block and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip()
    if default is not None:
        return default
    raise ValueError(f"Missing saved parameter <{block}>/{name}")


def header_parameter(header, block, name):
    return float(header_value(header, block, name))


def verify(run_dir, nx=108):
    fluid_files = list((run_dir / "bin").glob("*.fluid.00000.bin"))
    thermal_files = list((run_dir / "bin").glob("*.three_t.00000.bin"))
    require(len(fluid_files) == len(thermal_files) == 1, "Expected initial full-volume dumps")
    fluid = read_binary(str(fluid_files[0]))
    thermal = read_binary(str(thermal_files[0]))
    require(fluid["time"] == thermal["time"] == 0.0, "Dump is not at t=0")
    mesh_shape = tuple(fluid["N"+axis] for axis in ("x1", "x2", "x3"))
    require(mesh_shape == (nx, nx, nx), "Expected a %d^3 mesh" % nx)
    require(tuple(thermal["N"+axis] for axis in ("x1", "x2", "x3")) == mesh_shape,
            "Fluid and thermal mesh sizes differ")
    header = fluid["header"]
    length_unit = header_parameter(header, "units", "length_cgs")
    mass_unit = header_parameter(header, "units", "mass_cgs")
    time_unit = header_parameter(header, "units", "time_cgs")
    length_um = length_unit*1.e4
    velocity_kms = length_unit/time_unit/1.e5
    density_unit = mass_unit/length_unit**3
    for axis in ("x1", "x2", "x3"):
        require(np.isclose(fluid[axis+"min"]*length_um, -150.) and
                np.isclose(fluid[axis+"max"]*length_um, 150.),
                "Expected a centred 300 um cube")
    require(np.array_equal(fluid["mb_logical"], thermal["mb_logical"]),
            "Fluid and thermal block ordering differs")
    require(np.array_equal(fluid["mb_geometry"], thermal["mb_geometry"]),
            "Fluid and thermal block geometry differs")
    shell_density = header_parameter(header, "problem", "shell_density_gcc")
    ambient_density = header_parameter(header, "problem", "ambient_density_gcc")
    half_thickness = .5*header_parameter(header, "problem", "shell_thickness_um")
    radial_width = header_parameter(header, "problem", "interface_half_width_um")
    angular_width = header_parameter(header, "problem", "angular_half_width")
    cosine = math.cos(math.radians(header_parameter(
        header, "problem", "opening_half_angle_deg")))
    caps = [(name, sign, header_parameter(header, "problem", name+"_radius_um"),
             header_parameter(header, "problem", name+"_speed_kms"))
            for name, sign in (("upper", 1), ("lower", -1))]
    shell_ti = header_parameter(fluid["header"], "problem", "shell_temperature_ev")
    shell_te = shell_ti * header_parameter(
        fluid["header"], "hydro", "initial_electron_temperature_ratio")
    corona_enabled = header_value(
        fluid["header"], "problem", "corona_enabled", "false").lower() == "true"
    corona_density = float(header_value(
        fluid["header"], "problem", "corona_density_gcc", "0.01"))
    corona_ti = float(header_value(
        fluid["header"], "problem", "corona_temperature_ev", "1000.0"))
    corona_te = corona_ti * header_parameter(
        fluid["header"], "hydro", "initial_electron_temperature_ratio")
    require(np.isfinite(shell_ti) and np.isfinite(shell_te) and
            shell_ti > 0 and shell_te > 0, "Invalid expected shell temperatures")
    data, temp = fluid["mb_data"], thermal["mb_data"]
    for fields in (data, temp):
        require(all(np.all(np.isfinite(values)) for values in fields.values()),
                "Non-finite initial field")
    density = np.asarray(data["dens"], dtype=float)*density_unit
    ch, he = np.asarray(data["s_00"]), np.asarray(data["s_01"])
    upper, lower = np.asarray(data["s_02"]), np.asarray(data["s_03"])
    require(not corona_enabled or "s_04" in data, "Missing corona origin tracer")
    corona = np.asarray(data["s_04"]) if "s_04" in data else np.zeros_like(ch)
    require(np.all(density > 0.0) and np.isclose(density.max(), shell_density),
            "Shell peak density differs from the saved input")
    require(np.allclose(ch+he, 1.0, atol=1e-7) and
            np.allclose(upper+lower+corona, ch, atol=2e-7),
            "CH must equal upper-shell + lower-shell + corona mass fractions")
    require(all(np.all((v >= -1e-7) & (v <= 1.+1e-7))
                for v in (ch, he, upper, lower, corona)), "Invalid mass fraction")
    if not corona_enabled:
        require(np.all(corona == 0.0), "Corona is present in a corona-disabled run")
    require(all(np.all(np.asarray(data[key]) >= 0.0)
                for key in ("eion", "eele", "eint")), "Negative material energy")
    groups = [key for key in temp if key.startswith("erad") and len(key) == 6]
    require(len(groups) == 20 and all(np.all(np.asarray(temp[key]) >= 0.0) for key in groups),
            "Expected 20 nonnegative radiation groups")
    # Match Units::temperature_cgs(), including its tabulated physical constants.
    temperature_scale_ev = ((length_unit/time_unit)**2 *
                            header_parameter(header, "units", "mu") *
                            1.660538921e-24/1.3806488e-16/11604.518121550082)
    ti = np.asarray(temp["tion"])*temperature_scale_ev
    te = np.asarray(temp["tele"])*temperature_scale_ev
    require(np.all(ti > 0.0) and np.all(te > 0.0), "Nonpositive Ti or Te")
    # Trace partial densities may lie below a table's density edge; other initial
    # bounds/floor flags indicate a thermodynamic initialization problem.
    flags = np.asarray(temp["eos_flags"], dtype=np.int64)
    require(np.all((flags & ~1) == 0), "Unexpected initial EOS bounds/floor flag")
    ambient = ch == 0.0
    require(np.allclose(density[ambient], ambient_density, rtol=2e-6),
            "Incorrect He density")
    ambient_ti = header_parameter(fluid["header"], "problem", "ambient_temperature_ev")
    ambient_te = ambient_ti * header_parameter(
        fluid["header"], "hydro", "initial_electron_temperature_ratio")
    require(np.allclose(ti[ambient], ambient_ti, rtol=2e-5) and
            np.allclose(te[ambient], ambient_te, rtol=2e-5), "Cold He has been heated initially")
    for key in ("velx", "vely", "velz"):
        require(np.all(np.asarray(data[key])[ambient] == 0.0), "Ambient is not stationary")
    # Independently integrate each spherical cap and check all sampled core velocities.
    cell_width = (fluid["x1max"]-fluid["x1min"])*length_um/nx
    block_shape = density.shape[1:]
    leaf_widths = [(geometry[1]-geometry[0])*length_um/block_shape[2]
                   for geometry in fluid["mb_geometry"]]
    result = {"mesh": list(mesh_shape), "cell_width_um": cell_width, "radiation_groups": 20,
              "leaf_blocks": len(leaf_widths),
              "leaf_cell_width_um": [float(min(leaf_widths)), float(max(leaf_widths))],
              "shell_thickness_um": 2.*half_thickness,
              "expected_shell_Ti_eV": shell_ti, "expected_shell_Te_eV": shell_te,
              "corona_enabled": corona_enabled}
    for (label, sign, radius, speed), tag in zip(caps, (upper, lower)):
        mass = 0.0
        speeds, temperatures, ion_temperatures, radii = [], [], [], []
        for block, geometry in enumerate(fluid["mb_geometry"]):
            shape = density[block].shape
            dz = (geometry[5]-geometry[4])/shape[0]
            dy = (geometry[3]-geometry[2])/shape[1]
            dx = (geometry[1]-geometry[0])/shape[2]
            z, y, x = np.meshgrid(
                geometry[4]+(np.arange(shape[0])+0.5)*dz,
                geometry[2]+(np.arange(shape[1])+0.5)*dy,
                geometry[0]+(np.arange(shape[2])+0.5)*dx, indexing="ij")
            r = np.sqrt(x*x+y*y+z*z)
            mu = np.divide(x, r, out=np.zeros_like(r), where=r > 0.)
            # At a 10^6 density contrast a mixed-volume cell can still have a CH
            # mass fraction indistinguishable from one in float output. Select the
            # pure core geometrically, beyond both interface transition regions.
            core = ((np.abs(r*length_um-radius) < half_thickness-radial_width-1.e-4) &
                    (sign*mu > cosine+angular_width+1.e-6))
            present = tag[block] > 1e-7
            if np.any(present):
                require(np.all(sign*x[present] > 0.0), "Shell lies in wrong hemisphere")
                require(np.all(np.abs(r[present]*length_um-radius) <=
                               half_thickness+radial_width+1.e-5),
                        "Shell extends outside its width plus interface smoothing")
                require(np.all(sign*mu[present] >= cosine-angular_width-1.e-6),
                        "Wrong opening angle")
            if np.any(core):
                v = [np.asarray(data[key][block], dtype=float)*velocity_kms
                     for key in ("velx", "vely", "velz")]
                for component, coordinate in zip(v, (x, y, z)):
                    require(np.allclose(component[core], -speed*coordinate[core]/r[core],
                                        rtol=2e-6, atol=2e-4),
                            "Velocity is not the requested inward radial vector")
                require(np.allclose(density[block][core], shell_density, rtol=2e-6),
                        "Incorrect core density")
                require(np.allclose(ti[block][core], shell_ti, rtol=2e-5) and
                        np.allclose(te[block][core], shell_te, rtol=2e-5), "Incorrect Ti/Te")
                speeds.extend(np.sqrt(sum(component[core]**2 for component in v)))
                temperatures.extend(te[block][core])
                ion_temperatures.extend(ti[block][core])
                radii.extend(r[core]*length_um)
            mass += np.sum(density[block]*tag[block])*dx*dy*dz*length_unit**3
        require(len(speeds) > 0, "No resolved pure-shell core cells")
        analytic_mass = shell_density*2*math.pi*(1-cosine)/3 * (
            ((radius+half_thickness)*1e-4)**3-((radius-half_thickness)*1e-4)**3)
        mass_error = abs(mass/analytic_mass-1.0)
        require(mass_error < 0.03, "Cap mass differs from analytic geometry by >3%")
        result[label] = {
            "mass_g": float(mass), "sharp_cap_mass_g": analytic_mass,
            "relative_mass_error": mass_error, "core_cells": len(speeds),
            "core_speed_kms": [float(min(speeds)), float(max(speeds))],
            "core_Ti_eV": [float(min(ion_temperatures)), float(max(ion_temperatures))],
            "core_Te_eV": [float(min(temperatures)), float(max(temperatures))],
            "core_radius_um": [float(min(radii)), float(max(radii))],
            "ballistic_midpoint_arrival_ns": radius/speed,
        }
    if corona_enabled:
        mass = 0.0
        core_count = [0, 0]
        corona_ti_values, corona_te_values = [], []
        outer_upper, outer_lower = [cap[2]+half_thickness for cap in caps]
        for block, geometry in enumerate(fluid["mb_geometry"]):
            nz, ny, nx_block = density[block].shape
            dx, dy, dz = [(geometry[2*i+1]-geometry[2*i])/n
                          for i, n in enumerate((nx_block, ny, nz))]
            z, y, x = np.meshgrid(
                geometry[4]+(np.arange(nz)+.5)*dz,
                geometry[2]+(np.arange(ny)+.5)*dy,
                geometry[0]+(np.arange(nx_block)+.5)*dx, indexing="ij")
            radius = np.sqrt(x*x+y*y+z*z)
            mu = np.divide(x, radius, out=np.zeros_like(radius), where=radius > 0.)
            r_um = radius*length_um
            allowed = (((mu >= cosine-angular_width-1e-6) &
                        (r_um >= outer_upper-radial_width-1e-5)) |
                       ((-mu >= cosine-angular_width-1e-6) &
                        (r_um >= outer_lower-radial_width-1e-5)))
            require(np.all(corona[block][~allowed] == 0.),
                    "Corona appears ahead of a shell or outside its cap angle")
            for side, sign, outer in ((0, 1, outer_upper), (1, -1, outer_lower)):
                core = ((sign*mu > cosine+angular_width+1e-6) &
                        (r_um > outer+radial_width+1e-5))
                core_count[side] += int(core.sum())
                require(np.allclose(corona[block][core], 1., atol=1e-7) and
                        np.all(upper[block][core] == 0.) and np.all(lower[block][core] == 0.),
                        "Rear corona must be pure CH with no dense-shell origin tags")
                require(np.allclose(density[block][core], corona_density, rtol=2e-6),
                        "Incorrect rear corona density")
                require(np.allclose(ti[block][core], corona_ti, rtol=2e-5) and
                        np.allclose(te[block][core], corona_te, rtol=2e-5),
                        "Incorrect corona Ti/Te")
                for key in ("velx", "vely", "velz"):
                    require(np.all(np.asarray(data[key][block])[core] == 0.),
                            "Rear corona is not stationary")
                corona_ti_values.extend(ti[block][core])
                corona_te_values.extend(te[block][core])
            mass += np.sum(density[block]*corona[block])*dx*dy*dz*length_unit**3
        require(all(n > 0 for n in core_count), "No resolved corona behind one or both shells")
        result["corona"] = {
            "mass_g": float(mass), "core_cells_upper_lower": core_count,
            "density_gcc": corona_density, "speed_kms": 0.,
            "core_Ti_eV": [float(min(corona_ti_values)), float(max(corona_ti_values))],
            "core_Te_eV": [float(min(corona_te_values)), float(max(corona_te_values))],
            "outer_shell_radii_um": [outer_upper, outer_lower],
        }
        histories = list(run_dir.glob("*.user.hst"))
        require(len(histories) == 1, "Expected one history to validate corona mass accounting")
        labels = [word.split("=", 1)[1] for word in histories[0].read_text().splitlines()[1].split()
                  if "=" in word]
        initial = np.atleast_2d(np.loadtxt(histories[0]))[0]
        history = dict(zip(labels, initial))
        require(history["time"] == 0.0 and "coro_mass" in history,
                "Initial history has no corona mass column")
        require(np.isclose(history["CH_mass"], history["upper_mass"] +
                           history["lower_mass"]+history["coro_mass"], rtol=1e-12),
                "History counts corona as dense-shell mass or loses CH mass")
        require(np.isclose(history["coro_mass"]*mass_unit, mass, rtol=2e-6),
                "Corona history mass differs from the volume dump")
    result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--nx", type=int, default=108,
                        help="expected cells per axis (default: 108; archived: 96 or 100)")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_directory, args.nx), indent=2))
