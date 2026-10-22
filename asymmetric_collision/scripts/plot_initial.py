#!/usr/bin/env python3
"""Plot measured initial density, Ti, Te, and velocity from the saved test-run dumps.

Requires NumPy and Matplotlib. No simulation is launched and no input deck is changed.
Outputs PNG/PDF overviews and signed axial velocity maps for the x1-x2 and x1-x3 planes.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np

CASE_DIR = Path(__file__).resolve().parent.parent
# Reuse the uniform-grid assembly/reader already used by DCI_3D plotting. It handles
# the MeshBlock order and AthenaK's upper-cell selection at an exact cell-face tie.
sys.path.insert(0, str(CASE_DIR.parent / "DCI_3D"))
from plot_laser_rays import assemble_fields, input_header, read_binary


def one_file(run_dir, field):
    matches = list((run_dir / "bin").glob("*." + field + ".00000.bin"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one initial full-volume " + field + " dump")
    return matches[0]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_scales(header):
    units = input_header(header)["units"]
    length, mass, time, mu = (
        float(units[name]) for name in ("length_cgs", "mass_cgs", "time_cgs", "mu"))
    # The same constants and temperature convention as src/units/units.hpp/.cpp.
    velocity = length/time
    temperature_kelvin = velocity**2*mu*1.660538921e-24/1.3806488e-16
    return {"length_um": length*1e4, "density_gcc": mass/length**3,
            "velocity_kms": velocity/1e5,
            "temperature_ev": temperature_kelvin/11604.518121550082,
            "time_ns": time*1e9}


def load_plane(fluid, thermal, plane, scales):
    names = ("dens", "velx", "vely", "velz", "s_02", "s_03")
    values, axial_edges, transverse_edges, normal = assemble_fields(
        fluid, plane, 0.0, names)
    temperatures, t_axial, t_transverse, t_normal = assemble_fields(
        thermal, plane, 0.0, ("tion", "tele"))
    if (not np.array_equal(axial_edges, t_axial) or
            not np.array_equal(transverse_edges, t_transverse) or normal != t_normal):
        raise ValueError("Fluid and temperature slices do not coincide")
    # The reusable assembler places x1 horizontally; transpose every field so the
    # collision axis is vertical and the named upper/lower shells appear up/down.
    physical = {"density": values["dens"].T*scales["density_gcc"],
                "Ti": temperatures["tion"].T*scales["temperature_ev"],
                "Te": temperatures["tele"].T*scales["temperature_ev"],
                "v1": values["velx"].T*scales["velocity_kms"],
                "v2": values["vely"].T*scales["velocity_kms"],
                "v3": values["velz"].T*scales["velocity_kms"]}
    physical["speed"] = np.sqrt(sum(physical[key]**2 for key in ("v1", "v2", "v3")))
    physical["horizontal_velocity"] = physical["v2" if plane == "xy" else "v3"]
    physical["upper_tag"] = values["s_02"].T
    physical["lower_tag"] = values["s_03"].T
    return (physical, transverse_edges*scales["length_um"],
            axial_edges*scales["length_um"], normal*scales["length_um"])


def style_axis(ax, xedges, yedges, plane):
    ax.set_aspect("equal")
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xticks(np.linspace(xedges[0], xedges[-1], 5))
    ax.set_yticks(np.linspace(yedges[0], yedges[-1], 5))
    ax.set_xlabel(r"$x_2$ [$\mu$m]" if plane == "xy" else r"$x_3$ [$\mu$m]")
    ax.set_ylabel(r"$x_1$ (collision axis) [$\mu$m]")
    ax.plot(0.0, 0.0, "+", color="white", markersize=7, markeredgewidth=1,
            path_effects=[effects.withStroke(linewidth=1.7, foreground="#454545")])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def velocity_arrows(ax, data, xedges, yedges):
    xc, yc = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    x, y = np.meshgrid(xc, yc)
    # Choose well-separated actual cell samples within dense shell material.
    # All arrow components are read from the dump, not reconstructed from a model.
    candidates = np.argwhere(data["density"] > 0.8*data["density"].max())
    samples = []
    for row, column in candidates:
        if all((x[row, column]-x[r, c])**2+(y[row, column]-y[r, c])**2 >= 17.0**2
               for r, c in samples):
            samples.append((row, column))
    if not samples:
        return
    rows, columns = np.asarray(samples).T
    quiver = ax.quiver(
        x[rows, columns], y[rows, columns],
        data["horizontal_velocity"][rows, columns], data["v1"][rows, columns],
        angles="xy", scale_units="xy", scale=1/0.075, pivot="tail", color="black",
        width=0.005, headwidth=3.2, headlength=4, zorder=4)
    quiver.set_path_effects([effects.withStroke(linewidth=1.1, foreground="white")])
    ax.quiverkey(quiver, 0.60, 0.065, 200.0, "200 km/s", labelpos="E", color="white",
                 labelcolor="white", fontproperties={"size": 9})


def panel(fig, ax, values, xedges, yedges, plane, title, cmap, norm, label, ticks):
    # Rasterize the actual cells in the PDF too: no spatial smoothing/interpolation.
    mesh = ax.pcolormesh(xedges, yedges, values, cmap=cmap, norm=norm,
                         shading="flat", rasterized=True)
    style_axis(ax, xedges, yedges, plane)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=9)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.046, ticks=ticks)
    colorbar.set_label(label)
    colorbar.outline.set_linewidth(0.6)
    return colorbar


def render(data, xedges, yedges, normal, plane, output_dir, time_ns, grid, dpi):
    normal_axis = "3" if plane == "xy" else "2"
    transverse_axis = "2" if plane == "xy" else "3"
    # Use measured temperatures so archived and revised initializations are labeled
    # correctly. Round float32 output noise away from otherwise exact color limits.
    temperature_max = float(f"{max(np.max(data['Ti']), np.max(data['Te'])):.6g}")
    temperature_ticks = np.linspace(0, temperature_max, 6)
    core = data["density"] >= 0.999 * np.max(data["density"])
    core_ti, core_te = (float(np.median(data[key][core])) for key in ("Ti", "Te"))
    hot_corona = temperature_max > 5*max(core_ti, core_te)
    temperature_norm = LogNorm(1., temperature_max) if hot_corona else Normalize(0, temperature_max)
    if hot_corona:
        temperature_ticks = sorted(set([1., 10., round(core_ti), 100., temperature_max]))
    temperature_label = " [eV] (log scale)" if hot_corona else " [eV]"
    subtitle = (rf"$t={time_ns:g}$ ns  |  $x_{normal_axis}={normal:+.1f}$ $\mu$m"
                rf"  |  ${grid[0]}\times{grid[1]}\times{grid[2]}$ grid"
                rf"  |  $\Delta x={yedges[1]-yedges[0]:g}$ $\mu$m")
    with plt.rc_context({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.labelsize": 10, "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.5))
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.10, top=0.87,
                            wspace=0.34, hspace=0.34)
        fig.suptitle("Initial compressed-shell collision", y=0.97,
                     fontsize=17, fontweight="bold")
        fig.text(0.5, 0.927, subtitle, ha="center", fontsize=11, color="#444444")
        bar = panel(fig, axes[0, 0], data["density"], xedges, yedges, plane,
                    "(a) Mass density", "viridis", LogNorm(1e-5, 10.0),
                    r"$\rho$ [g cm$^{-3}$] (log scale)", [1e-5, 1e-3, 1e-1, 10.0])
        bar.formatter = LogFormatterMathtext()
        bar.update_ticks()
        panel(fig, axes[0, 1], data["Ti"], xedges, yedges, plane,
              "(b) Ion temperature", "magma", temperature_norm,
              r"$T_i$"+temperature_label, temperature_ticks)
        panel(fig, axes[1, 0], data["Te"], xedges, yedges, plane,
              "(c) Electron temperature", "magma", temperature_norm,
              r"$T_e$"+temperature_label, temperature_ticks)
        panel(fig, axes[1, 1], data["speed"], xedges, yedges, plane,
              "(d) Speed and in-plane velocity", "cividis", Normalize(0, 300),
              r"$|\mathbf{v}|$ [km s$^{-1}$]", [0, 100, 200, 300])
        velocity_arrows(axes[1, 1], data, xedges, yedges)
        vertical = .5*(yedges[1:]+yedges[:-1])
        column = data["density"].shape[1]//2
        for tag, label, boundary in (("upper_tag", "Upper shell", yedges[-1]),
                                     ("lower_tag", "Lower shell", yedges[0])):
            row = np.argmax(data["density"][:, column]*data[tag][:, column])
            y = vertical[row]
            axes[0, 0].text(0, y, label, ha="center", va="center", color="white",
                            fontsize=9, path_effects=[effects.withStroke(
                                linewidth=1.5, foreground="black")])
            if hot_corona:
                for ax in (axes[0, 1], axes[1, 0]):
                    ax.text(0, .5*(y+boundary), "Hot CH corona", ha="center",
                            va="center", color="black", fontsize=8)
        fig.text(0.5, 0.035,
                 "Saved simulation cells at initialization; no spatial interpolation. "
                 rf"Pure-shell $T_i\simeq{core_ti:g}$ eV, $T_e\simeq{core_te:g}$ eV." "\n"
                 "Color in (d) is the full 3D speed; arrows show the two components "
                 "in this plane. + marks the origin.",
                 ha="center", va="center", fontsize=9, color="#555555", linespacing=1.6)
        paths = []
        for extension in ("png", "pdf"):
            path = output_dir / ("initial_"+plane+"."+extension)
            fig.savefig(path, dpi=dpi)
            paths.append(str(path))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 6.4))
        fig.subplots_adjust(left=0.13, right=0.86, bottom=0.16, top=0.83)
        fig.suptitle("Initial axial velocity", y=0.97, fontsize=16, fontweight="bold")
        fig.text(0.5, 0.917, subtitle, ha="center", fontsize=10, color="#444444")
        panel(fig, ax, data["v1"], xedges, yedges, plane,
              rf"$x_1$–$x_{transverse_axis}$ slice", "RdBu_r",
              TwoSlopeNorm(vmin=-300, vcenter=0, vmax=300),
              r"$v_1$ [km s$^{-1}$]", [-300, -200, -100, 0, 100, 200, 300])
        fig.text(0.5, 0.045,
                 "Blue: downward flow from the upper shell.\n"
                 "Red: upward flow from the lower shell. White: stationary background.",
                 ha="center", fontsize=9, color="#555555", linespacing=1.6)
        for extension in ("png", "pdf"):
            path = output_dir / ("initial_"+plane+"_axial_velocity."+extension)
            fig.savefig(path, dpi=dpi)
            paths.append(str(path))
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path,
                        default=CASE_DIR / "runs" / "validation-initial")
    parser.add_argument("--output-dir", type=Path, default=CASE_DIR / "plots" / "initial")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    fluid_path, thermal_path = (
        one_file(args.run_directory, kind) for kind in ("fluid", "three_t"))
    fluid, thermal = read_binary(str(fluid_path)), read_binary(str(thermal_path))
    if fluid["time"] != 0.0 or thermal["time"] != 0.0:
        raise ValueError("Expected t=0 initial snapshots")
    if not np.array_equal(fluid["mb_logical"], thermal["mb_logical"]):
        raise ValueError("Fluid and thermal blocks differ")
    scales = unit_scales(fluid["header"])
    if scales != unit_scales(thermal["header"]):
        raise ValueError("Fluid and temperature units differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = [fluid["Nx"+str(axis)] for axis in (1, 2, 3)]
    print("Source:", args.run_directory.resolve(), "time=0 ns, grid=", grid)
    for plane in ("xy", "xz"):
        data, xedges, yedges, normal = load_plane(fluid, thermal, plane, scales)
        # Compare the volume extraction with the independently written xy plane dump.
        if plane == "xy":
            pairs = (("fluid_xy", ("dens", "velx", "vely", "velz"), fluid),
                     ("three_t_xy", ("tion", "tele"), thermal))
            for kind, names, reference in pairs:
                saved = read_binary(str(one_file(args.run_directory, kind)))
                separate, _, _, _ = assemble_fields(saved, plane, 0.0, names)
                extracted, _, _, _ = assemble_fields(reference, plane, 0.0, names)
                if any(not np.array_equal(separate[name], extracted[name]) for name in names):
                    raise ValueError("Volume extraction differs from the saved xy slice")
        output_paths = render(data, xedges, yedges, normal, plane, args.output_dir,
                              fluid["time"]*scales["time_ns"], grid, args.dpi)
        print(plane, "normal [um] =", normal,
              "max |Ti-Te| [eV] =", float(np.max(np.abs(data["Ti"]-data["Te"]))))
        print("\n".join(output_paths))
    print("Source SHA-256:", file_hash(fluid_path), file_hash(thermal_path))


if __name__ == "__main__":
    main()
