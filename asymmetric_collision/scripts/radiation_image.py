#!/usr/bin/env python3
"""Ray-trace multigroup thermal emission through a saved collision plasma.

Writes a synthetic intensity image (PNG/PDF), group intensities and optical
depths (NPZ), and provenance (JSON). Requires paired full 3D fluid/three_t binary
dumps and the run's opacity tables. This is a frozen-snapshot, straight-ray,
emission/absorption diagnostic; it does not evolve the simulation.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from radiation_io import paired_paths, read_pair
from radiation_rays import LeafMesh, make_camera, trace_image
from radiation_transfer import KELVIN_PER_EV, PlasmaEmission, planck_groups


def product_path(stem, suffix):
    """Append the format suffix without stripping dots from a job basename."""
    return Path(str(stem) + suffix)


def render_image(path, intensity, camera, length_um, time_ns, energy_intervals, dpi=180,
                 material_names=None, emission_tracers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    image = intensity.sum(axis=0)
    positive = image[image > 0]
    if positive.size and positive.min() < positive.max():
        norm = LogNorm(vmin=max(float(positive.min()), float(positive.max()) * 1e-8),
                       vmax=float(positive.max()))
        display = np.ma.masked_less_equal(image, 0)
    else:
        norm, display = Normalize(0, max(float(image.max()), 1)), image
    def axis_label(vector, letter):
        for axis in range(3):
            if np.allclose(vector, np.eye(3)[axis], rtol=0, atol=1e-12):
                return rf"$x_{axis + 1}$ offset [$\mu$m]"
        return rf"Image ${letter}$ [$\mu$m]"
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(cmap(0.0))  # A zero-intensity pixel is dark on a logarithmic map.
    plotted = ax.pcolormesh(camera.u_edges * length_um, camera.v_edges * length_um,
                           display, shading="flat", cmap=cmap, norm=norm, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel(axis_label(camera.horizontal, "u"))
    ax.set_ylabel(axis_label(camera.vertical, "v"))
    contiguous = np.allclose(energy_intervals[:-1, 1], energy_intervals[1:, 0])
    band = (f"{energy_intervals[0, 0]:g}–{energy_intervals[-1, 1]:g} eV" if contiguous
            else f"{len(energy_intervals)} selected photon groups")
    contributors = "" if material_names is None else " | " + " + ".join(material_names)
    if emission_tracers:
        contributors += " emission: " + " + ".join(emission_tracers)
    ax.set_title(f"Synthetic thermal radiation | t = {time_ns:.3f} ns\n{band}{contributors}")
    colorbar = fig.colorbar(plotted, ax=ax, pad=0.03)
    colorbar.set_label(r"Group-summed intensity [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")
    for suffix in (".png", ".pdf"):
        fig.savefig(product_path(path, suffix), dpi=dpi)
    plt.close(fig)


def process_snapshot(fluid_path, thermal_path, args):
    start = time.monotonic()
    fluid, thermal, scales = read_pair(fluid_path, thermal_path, args.emission_tracers or ())
    mesh = LeafMesh(fluid)
    model = PlasmaEmission(fluid, thermal, scales, args.run_dir, args.groups, args.table_dir,
                           material_selection=args.materials, emission_tracers=args.emission_tracers)
    center = (None if args.center_um is None
              else np.asarray(args.center_um) / scales["length_um"])
    extent = (None if args.extent_um is None
              else np.asarray(args.extent_um) / scales["length_um"])
    camera = make_camera(mesh, args.pixels, args.observer, args.up, center, extent)
    bounds_kelvin = model.bounds * scales["temperature_kelvin"]
    background = planck_groups(args.background_temperature_ev * KELVIN_PER_EV,
                               bounds_kelvin)[model.groups]
    time_ns = fluid["time"] * scales["time_ns"]
    print(f"Tracing {fluid_path.name}: t={time_ns:g} ns, {fluid['n_mbs']} leaves, "
          f"{args.pixels[0]}x{args.pixels[1]} rays, {len(model.groups)} groups, "
          f"materials={'+'.join(model.material_names)}, "
          f"emission origins={'+'.join(model.emission_tracers) or 'all'}", flush=True)
    last_update = time.monotonic()
    def progress(done, total):
        nonlocal last_update
        if time.monotonic() - last_update >= 10 or done == total:
            print(f"  integrated {done}/{total} blocks ({time.monotonic() - start:.1f} s)", flush=True)
            last_update = time.monotonic()
    result = trace_image(mesh, camera, model.coefficients, background, progress)
    output = args.output_dir or Path(args.run_dir) / "radiation"
    output.mkdir(parents=True, exist_ok=True)
    base, _, dump, _ = fluid_path.name.rsplit(".", 3)
    stem = output / f"{base}_radiation_{dump}"
    intervals = np.column_stack((bounds_kelvin[model.groups], bounds_kelvin[model.groups + 1])) / KELVIN_PER_EV
    area_cm2 = float(np.diff(camera.u_edges)[0] * np.diff(camera.v_edges)[0] * scales["length_cm"]**2)
    np.savez_compressed(product_path(stem, ".npz"),
                        intensity_cgs=result["intensity"],
                        optical_depth=result["optical_depth"],
                        transmission=np.exp(-result["optical_depth"]),
                        band_intensity_cgs=result["intensity"].sum(axis=0),
                        group_indices=model.groups, energy_intervals_ev=intervals,
                        material_indices=np.array(model.material_indices, dtype=int),
                        material_names=np.array(model.material_names, dtype=str),
                        emission_tracers=np.array(model.emission_tracers, dtype=str),
                        u_edges_um=camera.u_edges * scales["length_um"],
                        v_edges_um=camera.v_edges * scales["length_um"],
                        path_length_cm=result["path_length"] * scales["length_cm"],
                        block_crossings=result["block_crossings"],
                        observer_direction=camera.direction,
                        horizontal_direction=camera.horizontal, vertical_direction=camera.vertical,
                        center_um=camera.center * scales["length_um"],
                        background_intensity_cgs=background, pixel_area_cm2=area_cm2,
                        directional_power_cgs=result["intensity"].sum(axis=(1, 2)) * area_cm2,
                        time_ns=time_ns, cycle=fluid["cycle"])
    if not args.no_plot:
        render_image(stem, result["intensity"], camera, scales["length_um"], time_ns, intervals,
                     args.dpi, model.material_names, model.emission_tracers)
    metadata = {
        "model": "Frozen-snapshot multigroup thermal emission and absorption; orthographic straight rays",
        "equation": "dI_g/ds = j_g - alpha_g I_g; alpha_g=sum_selected rho*Y_m*kappa_abs,m,g; j_g=w_origin*B_g(Te)*sum_selected rho*Y_m*kappa_em,m,g",
        "cell_integration": "Exact exponential solution for piecewise-constant native leaf cells",
        "intensity_units": "erg s^-1 cm^-2 sr^-1 (integrated over each photon group)",
        "directional_power_units": "erg s^-1 sr^-1; sum of intensity times object-plane pixel area",
        "array_axes": {"intensity_cgs": ["group", "image_v", "image_u"],
                       "optical_depth": ["group", "image_v", "image_u"]},
        "limitations": ["No scattering, refraction, Doppler shifts, or light-travel-time evolution",
                        "Uses group-mean thermal opacities, not a resolved line spectrum or non-LTE population solve",
                        "One ray at each pixel center; no detector response, PSF, distance, or exposure model",
                        "Radiation moments in the dump are not substituted for directional intensity"],
        "time_ns": time_ns, "cycle": fluid["cycle"], "scales": scales,
        "sources": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                     "mtime_ns": path.stat().st_mtime_ns} for path in (fluid_path, thermal_path)],
        "tables": model.provenance,
        "material_indices": model.material_indices, "material_names": model.material_names,
        "material_selection_rule": "Only selected materials emit and absorb; use their partial densities after normalizing the full simulated mixture, without renormalizing the selected subset",
        "emission_selection": model.emission_selection,
        "opacity_clamps": model.clamp_counts,
        "opacity_clamp_count_scope": "All positive-partial-density cells in blocks intersected by at least one ray",
        "camera": {"pixels": args.pixels, "observer_direction": camera.direction.tolist(),
                   "horizontal_direction": camera.horizontal.tolist(),
                   "vertical_direction": camera.vertical.tolist(),
                   "center_um": (camera.center * scales["length_um"]).tolist(),
                   "extent_um": (np.array([camera.u_edges[0], camera.u_edges[-1],
                                            camera.v_edges[0], camera.v_edges[-1]]) * scales["length_um"]).tolist()},
        "background_temperature_ev": args.background_temperature_ev,
        "group_indices": model.groups.tolist(), "energy_intervals_ev": intervals.tolist(),
        "leaf_blocks": fluid["n_mbs"], "levels": np.unique(mesh.levels).tolist(),
        "band_intensity_range_cgs": [float(result["intensity"].sum(axis=0).min()),
                                     float(result["intensity"].sum(axis=0).max())],
        "optical_depth_range": [float(result["optical_depth"].min()), float(result["optical_depth"].max())],
        "elapsed_seconds": time.monotonic() - start,
    }
    if model.emission_tracers:
        metadata["limitations"].append(
            "Origin tracers partition emissivity at the saved cell temperature and material partial density; "
            "they do not separate temperatures in mixed cells or remove prior corona influence on evolution")
    product_path(stem, ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {stem}.npz/.json" + ("" if args.no_plot else " and .png/.pdf"), flush=True)
    return stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing bin/ and material_tables/")
    parser.add_argument("--dump", default="latest", help="dump number, latest (default), or all")
    parser.add_argument("--pixels", type=int, nargs=2, default=[256, 256], metavar=("NX", "NY"))
    parser.add_argument("--observer", type=float, nargs=3, default=[0, 0, 1], metavar=("X1", "X2", "X3"),
                        help="vector from plasma to observer (default: 0 0 1)")
    parser.add_argument("--up", type=float, nargs=3, default=[1, 0, 0], metavar=("X1", "X2", "X3"),
                        help="image-up reference vector, not parallel to observer (default: 1 0 0)")
    parser.add_argument("--center-um", type=float, nargs=3, metavar=("X1", "X2", "X3"),
                        help="image-plane origin in micrometers (default: domain center)")
    parser.add_argument("--extent-um", type=float, nargs=4, metavar=("UMIN", "UMAX", "VMIN", "VMAX"),
                        help="image bounds relative to center (default: full projected domain)")
    parser.add_argument("--groups", type=int, nargs="+", help="zero-based photon group indices (default: all)")
    parser.add_argument("--materials", nargs="+", metavar="NAME_OR_INDEX",
                        help="emitting and absorbing materials by name or zero-based index, e.g. CH; "
                             "default: all; selected fractions retain their original partial densities")
    parser.add_argument("--emission-tracers", nargs="+", metavar="S_NN",
                        help="origin mass fractions selecting emission within one --materials choice, "
                             "e.g. s_02 s_03 for the CH shells; all selected-material absorption is retained")
    parser.add_argument("--background-temperature-ev", type=float, default=0,
                        help="blackbody behind the plasma; zero means no incident radiation")
    parser.add_argument("--table-dir", type=Path, help="explicit replacement directory for the saved opacity tables")
    parser.add_argument("--output-dir", type=Path, help="default: RUN_DIR/radiation; reruns replace matching products")
    parser.add_argument("--no-plot", action="store_true", help="write only numerical NPZ and JSON (NumPy only)")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        if (min(args.pixels) < 1 or args.dpi < 1 or not np.isfinite(args.background_temperature_ev)
                or args.background_temperature_ev < 0):
            raise ValueError("Pixels/DPI must be positive and background temperature finite and non-negative")
        for pair in paired_paths(args.run_dir, args.dump):
            process_snapshot(*pair, args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"radiation_image: {exc}\n")


if __name__ == "__main__":
    main()
