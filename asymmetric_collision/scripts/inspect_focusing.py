#!/usr/bin/env python3
"""Measure shell focusing from existing dumps; never launch or modify a simulation.

Example:
  python inspect_focusing.py runs/case_a runs/case_b

Outputs JSON, CSV, and a PNG/PDF comparison in diagnostics/focusing by default.
Slice peaks are labelled separately from the less frequent full-volume peaks.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_initial import assemble_fields, input_header, read_binary, unit_scales


def initial_physics(run, header):
    """Pure-CH table interpolation and the same sound-speed derivative as C++."""
    path = run / header["materials"]["material0_eos_table_file"]
    table, key = {}, None
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line[0].isalpha():
            key = line if " " not in line and line != "end" else None
            if key:
                table[key] = []
        elif key:
            table[key].extend(map(float, line.split()))
    density_axis = np.array(table["density"])
    temperature_axis = np.array(table["temperature"])/11604.518121550082

    def value(field, rho, temp):
        if not (density_axis[0] < rho < density_axis[-1] and
                temperature_axis[0] < temp < temperature_axis[-1]):
            raise ValueError("Initial EOS diagnostic requires an interior table query")
        i, j = np.searchsorted(density_axis, rho)-1, np.searchsorted(temperature_axis, temp)-1
        wr = np.log(rho/density_axis[i])/np.log(density_axis[i+1]/density_axis[i])
        wt = np.log(temp/temperature_axis[j])/np.log(temperature_axis[j+1]/temperature_axis[j])
        a = np.array(table[field]).reshape(len(density_axis), len(temperature_axis))[i:i+2, j:j+2]
        weights = np.array([[(1-wr)*(1-wt), (1-wr)*wt], [wr*(1-wt), wr*wt]])
        if np.all(a > 0):
            return np.exp((weights*np.log(a)).sum())
        return (weights*a).sum()

    if header["materials"]["eos_table_interpolation"] != "geometric":
        raise ValueError("Initial EOS diagnostic expects geometric interpolation")
    problem = header["problem"]
    rho, temp = float(problem["shell_density_gcc"]), float(problem["shell_temperature_ev"])
    h, pressure, cs2 = .001, 0., 0.
    rl, rh, tl, th = rho*np.exp(-h), rho*np.exp(h), temp*np.exp(-h), temp*np.exp(h)
    for component in ("ion", "electron"):
        pf, ef = component+"_pressure", component+"_specific_internal_energy"
        pc = value(pf, rho, temp)
        pressure += pc
        dpdr = (value(pf, rh, temp)-value(pf, rl, temp))/(rh-rl)
        dedr = (value(ef, rh, temp)-value(ef, rl, temp))/(rh-rl)
        dpdt = (value(pf, rho, th)-value(pf, rho, tl))/(th-tl)
        dedt = (value(ef, rho, th)-value(ef, rho, tl))/(th-tl)
        cs2 += dpdr+dpdt*(pc/rho**2-dedr)/dedt
    cs = np.sqrt(max(cs2, pressure/rho))/1e5
    width = float(problem["shell_thickness_um"])
    return dict(pressure_mbar=float(pressure/1e12), sound_speed_kms=float(cs),
                sound_speed_excludes_numerical_safety_factor=True,
                sound_crossing_ns=float(width/cs), rarefaction_overlap_estimate_ns=float(width/(2*cs)),
                initial_radial_column_gcm2=float(rho*width*1e-4),
                upper_ballistic_midpoint_ns=float(problem["upper_radius_um"])/float(problem["upper_speed_kms"]),
                lower_ballistic_midpoint_ns=float(problem["lower_radius_um"])/float(problem["lower_speed_kms"]))


def fwhm(x, y):
    """Width of the contiguous dominant peak, interpolating half-height crossings."""
    i = int(np.argmax(y))
    half = y[i] / 2
    if half <= 0:
        return float("nan")
    lo = hi = i
    while lo > 0 and y[lo - 1] >= half:
        lo -= 1
    while hi < len(y) - 1 and y[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == len(y) - 1:
        return float("nan")
    left = np.interp(half, [y[lo - 1], y[lo]], [x[lo - 1], x[lo]])
    right = np.interp(half, [y[hi + 1], y[hi]], [x[hi + 1], x[hi]])
    return float(right - left)


def slice_metrics(run):
    rows, profiles = [], {}
    for fp in sorted((run / "bin").glob("*.fluid_xy.*.bin")):
        tp = fp.with_name(fp.name.replace(".fluid_xy.", ".three_t_xy."))
        f, t = read_binary(str(fp)), read_binary(str(tp))
        assert f["time"] == t["time"] and f["cycle"] == t["cycle"]
        s = unit_scales(f["header"])
        a, xe, ye, normal = assemble_fields(
            f, "xy", 0., ("dens", "s_00", "s_02", "s_03"))
        b, tx, ty, tn = assemble_fields(t, "xy", 0., ("tion", "tele"))
        assert np.array_equal(xe, tx) and np.array_equal(ye, ty) and normal == tn
        rho = a["dens"].astype(float) * s["density_gcc"]
        ti, te = (b[n].astype(float) * s["temperature_ev"] for n in ("tion", "tele"))
        x, y = (e[1:] + e[:-1] for e in (xe, ye))
        x, y = x * s["length_um"] / 2, y * s["length_um"] / 2
        # Reader arrays are [x2, x1]; take the upper central cell, as AthenaK does.
        j = len(y) // 2
        p = np.unravel_index(np.argmax(rho), rho.shape)
        hot = np.unravel_index(np.argmax(ti), ti.shape)
        dense = (rho >= 1.) & (a["s_00"] >= .5)
        row = dict(time_ns=float(f["time"] * s["time_ns"]),
                   rho_peak_gcc=float(rho[p]), ti_at_rho_peak_ev=float(ti[p]),
                   te_at_rho_peak_ev=float(te[p]), peak_x1_um=float(x[p[1]]),
                   peak_x2_um=float(y[p[0]]), normal_x3_um=float(normal*s["length_um"]),
                   ti_global_max_ev=float(ti[hot]), rho_at_ti_global_max_gcc=float(rho[hot]),
                   ch_at_ti_global_max=float(a["s_00"][hot]),
                   ti_dense_max_ev=float(ti[dense].max()) if np.any(dense) else 0.,
                   upper_axis_fwhm_um=fwhm(x, rho[j]*a["s_02"][j]),
                   lower_axis_fwhm_um=fwhm(x, rho[j]*a["s_03"][j]),
                   transverse_ch_column_max_gcm2=float(
                       (rho*a["s_00"]*np.diff(ye)[:, None]*s["length_um"]*1e-4)
                       .sum(axis=0).max()))
        rows.append(row)
        if round(row["time_ns"]*100) in (0, 10, 20, 30, 40):
            profiles[str(round(row["time_ns"], 2))] = dict(x1_um=x.tolist(), rho_gcc=rho[j].tolist())
    if not rows:
        raise ValueError(f"No saved xy slices in {run}")
    return rows, profiles


def volume_metrics(run):
    rows = []
    for fp in sorted((run / "bin").glob("*.fluid.*.bin")):
        f = read_binary(str(fp))
        tp = fp.with_name(fp.name.replace(".fluid.", ".three_t."))
        t = read_binary(str(tp))
        assert f["time"] == t["time"] and f["cycle"] == t["cycle"]
        s = unit_scales(f["header"])
        a, b = f["mb_data"], t["mb_data"]
        rho = np.asarray(a["dens"], dtype=float)*s["density_gcc"]
        ch = np.asarray(a["s_00"], dtype=float)
        ti = np.asarray(b["tion"], dtype=float)*s["temperature_ev"]
        te = np.asarray(b["tele"], dtype=float)*s["temperature_ev"]
        flags = np.asarray(b["eos_flags"], dtype=np.int32)
        # Table energy-below-range bit, accumulated at fixed cells by the solver.
        floor = (flags & 16) != 0
        p = np.unravel_index(np.argmax(rho), rho.shape)
        n = [int(f[f"Nx{i}"]) for i in (1, 2, 3)]
        dv = np.prod([(f[f"x{i}max"]-f[f"x{i}min"])*s["length_um"]*1e-4/n[i-1]
                      for i in (1, 2, 3)])
        mass = rho*ch*dv
        row = dict(time_ns=float(f["time"]*s["time_ns"]),
                   rho_peak_gcc=float(rho[p]), ti_at_rho_peak_ev=float(ti[p]),
                   te_at_rho_peak_ev=float(te[p]), ch_mass_g=float(mass.sum()),
                   ch_mass_above_100_gcc_g=float(mass[rho >= 100.].sum()),
                   ch_mass_above_100_fraction=float(mass[rho >= 100.].sum()/mass.sum()),
                   ch_mass_in_flagged_cells_fraction=float(mass[floor].sum()/mass.sum()),
                   flagged_cell_count=int(floor.sum()),
                   flagged_dense_cell_count=int((floor & (rho >= 1.) & (ch >= .5)).sum()))
        # Compare radial thickness within a 20-degree axial cone before impact.
        if row["time_ns"] <= .200001:
            r_all, mu_all = [], []
            for m, g in enumerate(f["mb_geometry"]):
                nz, ny, nx = rho[m].shape
                x, y, z = [np.linspace(g[2*i], g[2*i+1], k, endpoint=False)
                           + (g[2*i+1]-g[2*i])/(2*k)
                           for i, k in enumerate((nx, ny, nz))]
                zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
                r = np.sqrt(xx*xx+yy*yy+zz*zz)
                r_all.append(r*s["length_um"])
                mu_all.append(xx/r)
            radius, mu = np.asarray(r_all), np.asarray(mu_all)
            for name, tag, sign in (("upper", "s_02", 1), ("lower", "s_03", -1)):
                w = rho*np.asarray(a[tag])*(sign*mu >= np.cos(np.deg2rad(20)))
                avg = (w*radius).sum()/w.sum()
                sigma = np.sqrt((w*(radius-avg)**2).sum()/w.sum())
                row[name+"_core_mean_radius_um"] = float(avg)
                row[name+"_core_equivalent_width_um"] = float(np.sqrt(12)*sigma)
        rows.append(row)
    return rows


def history_metrics(run):
    files = list(run.glob("*.user.hst"))
    if len(files) != 1:
        raise ValueError(f"Expected one history in {run}")
    names = [p.split("=", 1)[1] for p in files[0].read_text().splitlines()[1].split()
             if "=" in p]
    data = np.loadtxt(files[0])
    assert data.shape[1] == len(names)
    h = dict(zip(names, data.T))
    # Trapezoidal integral is approximate: the saved history cadence is 5 ps.
    loss = np.r_[0., np.cumsum(.5*np.diff(h["time"])*(h["rad_Pesc"][1:]+h["rad_Pesc"][:-1]))]
    return [dict(time_ns=float(h["time"][i]), mass_fraction=float(h["mass"][i]/h["mass"][0]),
                 energy_fraction=float(h["total_E"][i]/h["total_E"][0]),
                 kinetic_fraction=float(h["kin_E"][i]/h["kin_E"][0]),
                 radiated_energy_fraction_approx=float(loss[i]/h["total_E"][0]),
                 eos_bad=int(h["eos_bad"][i]), eos_floor=int(h["eos_floor"][i]),
                 com_velocity_kms=float(h["mom1"][i]/h["mass"][i]*1000),
                 upper_centroid_um=float(h["upper_x1"][i]/h["upper_mass"][i]*1000),
                 lower_centroid_um=float(h["lower_x1"][i]/h["lower_mass"][i]*1000))
            for i in range(len(data))]


def plot(results, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for color, result in zip(plt.cm.tab10.colors, results):
        rows = result["slices"]
        t = [r["time_ns"] for r in rows]
        label = f'{result["initial_temperature_ev"]:g} eV'
        axes[0, 0].plot(t, [r["rho_peak_gcc"] for r in rows], label=label, color=color)
        axes[0, 0].scatter([r["time_ns"] for r in result["volumes"]],
                           [r["rho_peak_gcc"] for r in result["volumes"]], s=14, color=color)
        axes[0, 1].plot(t, [r["transverse_ch_column_max_gcm2"] for r in rows], color=color)
        pre = [r for r in rows if r["time_ns"] <= .20]
        for cap, style in (("upper", "-"), ("lower", "--")):
            axes[0, 2].plot([r["time_ns"] for r in pre],
                           [r[cap+"_axis_fwhm_um"] for r in pre], style, color=color,
                           label=f"{label}, {cap}")
        axes[1, 0].plot(t, [r["ti_at_rho_peak_ev"] for r in rows], color=color, label=label+", ions")
        axes[1, 0].plot(t, [r["te_at_rho_peak_ev"] for r in rows], "--", color=color, label=label+", electrons")
        h = result["history"]
        axes[1, 1].plot([r["time_ns"] for r in h], [r["energy_fraction"] for r in h], color=color, label=label+", energy")
        axes[1, 1].plot([r["time_ns"] for r in h], [r["mass_fraction"] for r in h], "--", color=color, label=label+", mass")
    for time, p in results[-1]["axis_profiles"].items():
        axes[1, 2].plot(p["x1_um"], p["rho_gcc"], label=f"{time} ns")
    titles = ["Peak density: slice lines, volume dots", "Maximum transverse CH column (slice)",
              "Shell width on the nearest axial line", "Temperature at the slice density maximum",
              "Domain totals / initial totals", f'Axial density, {results[-1]["initial_temperature_ev"]:g} eV run']
    units = [r"Density [g cm$^{-3}$]", r"CH column [g cm$^{-2}$]", r"FWHM [$\mu$m]",
             "Temperature [eV]", "Fraction", r"Density [g cm$^{-3}$]"]
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.set_xlabel("Time [ns]")
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[1, 2].set_xlabel(r"$x_1$ [$\mu$m]")
    axes[1, 2].set_ylim(0, 150)
    fig.suptitle("Measured shell evolution: existing 108³ runs", fontsize=15)
    fig.savefig(out / "focusing_comparison.png", dpi=170)
    fig.savefig(out / "focusing_comparison.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "diagnostics" / "focusing")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        print(f"Reading {run}", flush=True)
        f = read_binary(str(next((run / "bin").glob("*.fluid_xy.00000.bin"))))
        header = input_header(f["header"])
        rows, profiles = slice_metrics(run)
        result = dict(run=str(run.resolve()),
                      initial_temperature_ev=float(header["problem"]["shell_temperature_ev"]),
                      initial_physics=initial_physics(run, header),
                      slices=rows, volumes=volume_metrics(run),
                      history=history_metrics(run), axis_profiles=profiles)
        results.append(result)
        with (args.output_dir / (run.name+"_slices.csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        peak = max(rows, key=lambda row: row["rho_peak_gcc"])
        print("Peak:", json.dumps(peak), flush=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    plot(results, args.output_dir)
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
