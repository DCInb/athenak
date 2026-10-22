#!/usr/bin/env python3
"""Measure same-cell thermal separation and prepare reproducible EOS queries."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_dumps import paired_paths, read_pair


def table_query(executable, run, rows, mode="temperature"):
    text = io.StringIO()
    np.savetxt(text, np.atleast_2d(rows), fmt="%.17g")
    command = [str(executable), str(run / "material_tables/ch.2t_eos"),
               str(run / "material_tables/he.2t_eos"), mode]
    result = subprocess.run(command, input=text.getvalue(), text=True,
                            capture_output=True, check=True)
    return np.atleast_1d(np.genfromtxt(io.StringIO(result.stdout), delimiter=",", names=True))


def cell_position(raw, flat):
    shape = np.asarray(raw["mb_data"]["dens"]).shape
    b, k, j, i = np.unravel_index(flat, shape)
    bounds = np.asarray(raw["mb_geometry"])[b]
    widths = (bounds[1::2] - bounds[::2]) / np.array(
        [raw[f"nx{axis}_mb"] for axis in (1, 2, 3)])
    index = np.asarray(raw["mb_index"])[b, ::2] + [i, j, k]
    return bounds[::2] + (index + .5) * widths


def raster(raw, values):
    """Display native saved XY cells, repeating coarse cells without smoothing."""
    locations = np.asarray(raw["mb_logical"])
    level = locations[:, 3].max()
    shape = (raw["Nx1"] * 2 ** level, raw["Nx2"] * 2 ** level)
    image = np.full(shape, np.nan)
    cover = np.zeros(shape, dtype=np.uint8)
    for b, (ix, iy, iz, block_level) in enumerate(locations):
        repeat = 2 ** (level - block_level)
        start = (np.array([ix, iy]) * [raw["nx1_mb"], raw["nx2_mb"]]
                 + np.asarray(raw["mb_index"])[b, [0, 2]]) * repeat
        cells = values[b, 0].T.repeat(repeat, axis=0).repeat(repeat, axis=1)
        region = (slice(start[0], start[0] + cells.shape[0]),
                  slice(start[1], start[1] + cells.shape[1]))
        image[region] = cells
        cover[region] += 1
    assert np.all(cover == 1) and np.isfinite(image).all()
    return image


def inspect(run, output, executable):
    records, representative_rows, representatives = [], [], []
    for kind, thermal_kind in (("fluid_xy", "three_t_xy"), ("fluid", "three_t")):
        for paths in paired_paths(run, kind, thermal_kind):
            fluid, thermal, scales = read_pair(*paths)
            d = {name: np.asarray(fluid["mb_data"][name], dtype=np.float64)
                 for name in ("dens", "s_00", "s_01", "s_02", "s_03")}
            ti_code, te_code = [np.asarray(thermal["mb_data"][name], dtype=np.float64)
                                for name in ("tion", "tele")]
            rho = d["dens"] * scales["density_gcc"]
            ti, te = ti_code * scales["temperature_ev"], te_code * scales["temperature_ev"]
            y = np.clip(d["s_00"], 0, 1)
            y /= np.maximum(y + np.clip(d["s_01"], 0, 1), 1.e-100)
            ratio = ti / te
            flags = np.asarray(thermal["mb_data"]["eos_flags"], dtype=np.int64)
            widths = (np.asarray(fluid["mb_geometry"])[:, 1::2]
                      - np.asarray(fluid["mb_geometry"])[:, ::2]) / np.array(
                          [fluid[f"nx{axis}_mb"] for axis in (1, 2, 3)])
            measure = np.prod(widths[:, :2 if kind == "fluid_xy" else 3], axis=1)
            weight = rho * measure[:, None, None, None]
            dense = (rho >= 1.) & (y >= .5)
            hot_gap = (ratio > 2.) & (ti >= 100.)
            time = float(fluid["time"] * scales["time_ns"])
            record = {"kind": kind, "dump": paths[0].name.rsplit(".", 2)[1],
                      "time_ns": time, "cycle": int(fluid["cycle"]),
                      "native_cells": int(rho.size), "Ti_max_ev": float(ti.max()),
                      "Te_max_ev": float(te.max()), "max_Ti_Te": float(ratio.max()),
                      "density_weighted_hot_gap_fraction": float(weight[hot_gap].sum() / weight.sum()),
                      "hot_gap_volume_or_area_fraction": float(
                          np.sum(np.count_nonzero(hot_gap, axis=(1, 2, 3)) * measure)
                          / np.sum(np.prod(rho.shape[1:]) * measure)),
                      "dense_CH_max_Ti_Te": float(ratio[dense].max()),
                      "dense_CH_max_Ti_ev": float(ti[dense].max()),
                      "dense_CH_max_Te_ev": float(te[dense].max()),
                      "dense_CH_weighted_abs_relative_gap": float(
                          (weight[dense] * np.abs(ratio[dense] - 1)).sum() / weight[dense].sum()),
                      "eos_flags_present": [int(x) for x in np.unique(flags)]}
            peak_indices = {"hottest_ion": int(np.argmax(ti)),
                            "hottest_electron": int(np.argmax(te)),
                            "densest": int(np.argmax(rho)),
                            "largest_dense_gap": int(np.argmax(np.where(dense, ratio, -1.)))}
            record["cells"] = {}
            for label, flat in peak_indices.items():
                cell = {"flat_index": flat, "rho_gcc": float(rho.flat[flat]),
                        "Ti_ev": float(ti.flat[flat]), "Te_ev": float(te.flat[flat]),
                        "Y_CH": float(y.flat[flat]),
                        "shell_origin_fraction": float((d["s_02"] + d["s_03"]).flat[flat]),
                        "position_um": (cell_position(fluid, flat) * scales["length_um"]).tolist(),
                        "eos_flags": int(flags.flat[flat]),
                        "query_index": len(representative_rows)}
                record["cells"][label] = cell
                representative_rows.append([d["dens"].flat[flat], ti_code.flat[flat],
                                            te_code.flat[flat], y.flat[flat]])
                representatives.append(cell)
            records.append(record)
            if kind == "fluid_xy" and record["dump"] in ("00013", "00031", "00051"):
                rows = np.column_stack([d["dens"].ravel(), ti_code.ravel(),
                                        te_code.ravel(), y.ravel()])
                queried = table_query(executable, run, rows)
                for name in queried.dtype.names:
                    assert np.isfinite(queried[name]).all(), name
                assert np.all(queried["exchange_fallback"] != 2), "An EOS exchange solve failed"
                np.savez_compressed(output / f"cell_query_{record['dump']}.npz",
                                    query=queried, widths_code=widths,
                                    shape=np.array(rho.shape))
                tau = queried["tau_difference_ns"].reshape(rho.shape)
                # Classical, local isotropic 3D explicit limit; saturation is not included.
                dt_exp = 1.e9 / (2 * queried["chi_cgs"].reshape(rho.shape)
                                    * np.sum((widths * .1) ** -2, axis=1)[:, None, None, None])
                record["conduction_classical_explicit_min_ns"] = float(dt_exp.min())
                record["coupling_tau_difference_range_ns"] = [float(tau.min()), float(tau.max())]
                record["queried_exchange_fallback_counts"] = {
                    str(int(k)): int(v) for k, v in zip(*np.unique(
                        queried["exchange_fallback"], return_counts=True))}
                if record["dump"] == "00013":
                    panels = [rho, ti, te, ratio, tau, dt_exp]
                    np.savez_compressed(output / "thermal_maps_data.npz",
                                        **{key: raster(fluid, value) for key, value in zip(
                                            ("rho", "Ti", "Te", "ratio", "tau", "dt_exp"), panels)})
                    plot_maps(output, fluid, panels, time)
            print(f"Read {kind} t={time:.2f} ns: Ti/Te(dense CH) <= "
                  f"{record['dense_CH_max_Ti_Te']:.6g}, hot-gap weight fraction="
                  f"{record['density_weighted_hot_gap_fraction']:.4g}", flush=True)
            del fluid, thermal, d, rho, ti, te, y, ratio, weight, flags
    rows = np.asarray(representative_rows)
    np.savetxt(output / "representative_cells.ssv", rows, fmt="%.17g")
    queried = table_query(executable, run, rows)
    for cell, values in zip(representatives, queried):
        cell["eos_query"] = {name: float(values[name]) for name in queried.dtype.names}
    manifest = {"run": str(run), "density_unit_gcc": scales["density_gcc"],
                "temperature_unit_ev": scales["temperature_ev"],
                "query_executable": str(executable),
                "query_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "mass_weight_definition": "rho times native cell volume for 3D; rho times cell area for XY",
                "dense_CH_definition": "rho >= 1 g/cm^3 and Y_CH >= 0.5",
                "hot_gap_definition": "Ti/Te > 2 and Ti >= 100 eV",
                "snapshots": records}
    (output / "thermal_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_history(output, records)
    return manifest


def plot_maps(output, fluid, panels, time):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    fig.suptitle(f"Where the temperature separation occurs · t = {time:.2f} ns", fontsize=18)
    limits = [(1.e-6, 1.e3), (1, 5.e4), (1, 5.e4), (.3, 300), (1.e-6, 100), (1.e-9, .1)]
    titles = ["Density [g/cm³]", "Ion temperature [eV]", "Electron temperature [eV]",
              "Same-cell Ti / Te", "Local difference relaxation time [ns]",
              "Classical explicit conduction limit [ns]"]
    for ax, values, bounds, title in zip(axes.flat, panels, limits, titles):
        arr = raster(fluid, values)
        im = ax.imshow(arr, origin="lower", extent=(-150, 150, -150, 150),
                       interpolation="nearest", cmap="viridis" if title.startswith("Density") else "magma",
                       norm=LogNorm(*bounds))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x₂ [µm]"); ax.set_ylabel("x₁ [µm]")
        fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"thermal_maps.{ext}", dpi=170)
    plt.close(fig)


def plot_history(output, records):
    xy = [r for r in records if r["kind"] == "fluid_xy"]
    volume = [r for r in records if r["kind"] == "fluid"]
    time = np.array([r["time_ns"] for r in xy])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key, label in (("Ti_ev", "Ti at the hottest-ion cell"),
                       ("Te_ev", "Te in that same cell")):
        axes[0, 0].semilogy(time, [r["cells"]["hottest_ion"][key] for r in xy], label=label)
    axes[0, 0].set_ylabel("Temperature [eV]")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title("Hot, dilute cells dominate the temperature maximum")
    axes[0, 1].plot(time, [r["dense_CH_max_Ti_Te"] for r in xy], label="XY slices")
    axes[0, 1].plot([r["time_ns"] for r in volume],
                    [r["dense_CH_max_Ti_Te"] for r in volume], "o", label="Full volumes")
    axes[0, 1].set_ylabel("Maximum Ti / Te in dense CH")
    axes[0, 1].set_title("Dense CH: ρ ≥ 1 g/cm³, CH mass fraction ≥ 0.5")
    axes[0, 1].legend(fontsize=9)
    for label in ("hottest_ion", "densest"):
        axes[1, 0].semilogy(time, [r["cells"][label]["eos_query"]["tau_difference_ns"] for r in xy],
                            label=label.replace("_", " "))
    axes[1, 0].axhline(.8, ls="--", color="gray", label="Run duration")
    axes[1, 0].set_ylabel("Local temperature-difference relaxation time [ns]")
    axes[1, 0].set_title("Spitzer rate evaluated with the production EOS")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot([r["time_ns"] for r in volume],
                    [r["density_weighted_hot_gap_fraction"] for r in volume], "o-")
    axes[1, 1].set_yscale("symlog", linthresh=1.e-6)
    axes[1, 1].set_ylabel("Fraction of total mass")
    axes[1, 1].set_title("Ti > 2 Te and Ti ≥ 100 eV · full-volume measurement")
    for ax in axes.flat:
        ax.set_xlabel("Time [ns]"); ax.grid(alpha=.2); ax.set_xlim(0, .8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"thermal_history.{ext}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--query-executable", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    inspect(args.run_directory.resolve(), output, args.query_executable.resolve())


if __name__ == "__main__":
    main()
