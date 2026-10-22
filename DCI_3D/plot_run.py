#!/usr/bin/env python3
"""Visualize saved DCI XY/XZ slices, including adaptive meshes.

Usage: python DCI_3D/plot_run.py DCI_3D/runs/RUN/bin
Requires NumPy, Matplotlib and Pillow. No full volumes are loaded. Physical
scales come from the binary headers; coarse cells are repeated, not smoothed.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import html
import json
import math
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from plot_laser_rays import input_header, read_binary


FIELDS = {
    "density": ("Mass density", r"$\rho$ [g cm$^{-3}$]", "viridis"),
    "Te": ("Electron temperature", r"$T_e$ [eV]", "inferno"),
    "Ti": ("Ion temperature", r"$T_i$ [eV]", "inferno"),
    "Tr": ("Radiation temperature", r"$T_r$ [eV]", "inferno"),
    "speed": ("Flow speed", r"$|v|$ [km s$^{-1}$]", "cividis"),
    "Q": ("Laser deposition", r"$Q_{\rm laser}$ [W cm$^{-3}$]", "magma"),
    "path": ("Summed laser path per cell", "Ray-segment path sum [mm]", "magma"),
    "level": ("Adaptive mesh", "Refinement level", "viridis"),
}
OVERVIEW = ("density", "Te", "Ti", "Tr", "speed", "Q")


def unit_scales(header):
    units = input_header(header)["units"]
    length, mass, time, mu = (
        float(units[k]) for k in ("length_cgs", "mass_cgs", "time_cgs", "mu"))
    velocity = length / time
    # Match AthenaK src/units/units.cpp and units.hpp, then convert kelvin to eV.
    kelvin = velocity**2 * mu * 1.660538921e-24 / 1.3806488e-16
    return {"length_mm": length * 10, "time_ns": time * 1e9,
            "density": mass / length**3, "velocity": velocity / 1e5,
            "temperature": kelvin / 11604.518121550082,
            "Q": mass * velocity**2 / length**3 / time * 1e-7}


def assemble(raw, plane, names):
    """Return a complete leaf-cell plane, with x1 horizontal, in code units."""
    normal, vertical = (2, 1) if plane == "xy" else (1, 2)
    root = np.array([raw[f"Nx{i}"] for i in (1, 2, 3)], dtype=int)
    block_shape = np.array([raw[f"nx{i}_mb"] for i in (1, 2, 3)], dtype=int)
    low = np.array([raw[f"x{i}min"] for i in (1, 2, 3)])
    high = np.array([raw[f"x{i}max"] for i in (1, 2, 3)])
    logical, indices = np.asarray(raw["mb_logical"]), np.asarray(raw["mb_index"])
    geometry = np.asarray(raw["mb_geometry"]).reshape(-1, 3, 2)
    selected = [i for i, b in enumerate(geometry) if b[normal, 0] <= 0 < b[normal, 1]]
    if not selected or np.any(logical[:, 3] < 0):
        raise ValueError("No valid leaf blocks at the requested zero plane")
    max_level = int(logical[selected, 3].max())
    grid = root * 2**max_level
    shape = (int(grid[vertical]), int(grid[0]))
    data = {name: np.full(shape, np.nan) for name in (*names, "level")}
    coverage = np.zeros(shape, dtype=np.uint8)
    centers, widths, rectangles = [], [], []
    level_counts = Counter()
    native_cells = 0
    for block in selected:
        level = int(logical[block, 3])
        repeat = 2**(max_level - level)
        dx = (high - low) / (root * 2**level)
        bounds = geometry[block]
        expected_low = low + logical[block, :3] * block_shape * dx
        if not (np.allclose(bounds[:, 0], expected_low, rtol=0, atol=1e-12)
                and np.allclose(bounds[:, 1] - bounds[:, 0], block_shape * dx,
                                rtol=1e-10, atol=1e-12)):
            raise ValueError("Block bounds disagree with logical AMR coordinates")
        starts, ends = indices[block, ::2], indices[block, 1::2] + 1
        if np.any(starts < 0) or np.any(ends > block_shape):
            raise ValueError("Expected active-cell outputs without ghost zones")
        index = int((0 - bounds[normal, 0]) / (bounds[normal, 1] - bounds[normal, 0])
                    * block_shape[normal])
        if not starts[normal] <= index < ends[normal]:
            raise ValueError("Saved cells do not contain the requested zero plane")
        start = (logical[block, :3] * block_shape + starts) * repeat
        end = (logical[block, :3] * block_shape + ends) * repeat
        if np.any(start < 0) or np.any(end > grid):
            raise ValueError("Leaf cells extend beyond the domain")
        region = (slice(start[vertical], end[vertical]), slice(start[0], end[0]))
        if np.any(coverage[region]):
            raise ValueError("Overlapping leaf cells")
        coverage[region] = 1
        for name in names:
            values = np.asarray(raw["mb_data"][name][block], dtype=float)
            if values.shape != tuple((ends - starts)[::-1]):
                raise ValueError(f"Unexpected data dimensions for {name}")
            local = index - starts[normal]
            values = values[local, :, :] if plane == "xy" else values[:, local, :]
            data[name][region] = values.repeat(repeat, 0).repeat(repeat, 1)
        data["level"][region] = level
        centers.append(bounds[normal, 0] + (index + 0.5) * dx[normal])
        widths.append(dx[0])
        rectangles.append(bounds[[0, vertical], :].ravel().tolist())
        level_counts[level] += 1
        native_cells += int((ends[0] - starts[0]) * (ends[vertical] - starts[vertical]))
    if np.any(coverage != 1) or any(np.any(~np.isfinite(v)) for v in data.values()):
        raise ValueError("Incomplete plane or non-finite data")
    edges = [np.linspace(low[a], high[a], grid[a] + 1) for a in (0, vertical)]
    mesh = {"levels": sorted(level_counts), "blocks_by_level": dict(level_counts),
            "native_cells": native_cells, "display_shape": list(shape),
            "normal_cell_centers_code": [min(centers), max(centers)],
            "cell_width_code": [min(widths), max(widths)],
            "coverage_complete_without_overlap": True,
            "rasterization": "piecewise constant leaf cells; no interpolation"}
    return data, edges, mesh, rectangles


def read_frame(paths, plane, dump):
    fluid, thermal, laser = [read_binary(str(p)) for p in paths]
    keys = ("time", "cycle", "Nx1", "Nx2", "Nx3", "n_mbs", "nx1_mb", "nx2_mb",
            "nx3_mb", "mb_logical", "mb_index", "mb_geometry", "x1min", "x1max",
            "x2min", "x2max", "x3min", "x3max")
    scales = unit_scales(fluid["header"])
    for other in (thermal, laser):
        for key in keys:
            if not np.array_equal(fluid[key], other[key]):
                raise ValueError(f"Mismatched {key}: {paths}")
        if scales != unit_scales(other["header"]):
            raise ValueError("Unit scales differ between field files")
    f, edges, mesh, rectangles = assemble(fluid, plane, ("dens", "velx", "vely", "velz"))
    t, _, _, _ = assemble(thermal, plane, ("tele", "tion", "trad", "eos_flags"))
    l, _, _, _ = assemble(laser, plane, ("laser_q", "laser_path"))
    data = {"density": f["dens"] * scales["density"], "level": f["level"],
            "Te": t["tele"] * scales["temperature"],
            "Ti": t["tion"] * scales["temperature"],
            "Tr": t["trad"] * scales["temperature"],
            "speed": np.sqrt(f["velx"]**2 + f["vely"]**2 + f["velz"]**2)
                     * scales["velocity"],
            "Q": l["laser_q"] * scales["Q"],
            "path": l["laser_path"] * scales["length_mm"]}
    if any(np.any(data[k] <= 0) for k in ("density", "Ti", "Te", "Tr")):
        raise ValueError("Logarithmic density/temperature requires positive values")
    if np.any(data["Q"] < 0) or np.any(data["path"] < 0):
        raise ValueError("Negative laser diagnostic")
    return {"plane": plane, "dump": dump, "time_ns": float(fluid["time"] * scales["time_ns"]),
            "cycle": int(fluid["cycle"]), "data": data, "mesh": mesh, "scales": scales,
            "edges": [e * scales["length_mm"] for e in edges],
            "rectangles": np.array(rectangles) * scales["length_mm"],
            "sources": [str(p.resolve()) for p in paths]}


def discover(bin_dir):
    frames = []
    for plane in ("xy", "xz"):
        families = []
        for family in ("fluid", "three_t", "laser"):
            found = {}
            for p in sorted(bin_dir.glob(f"*.{family}_{plane}.*.bin")):
                base, _, dump, _ = p.name.rsplit(".", 3)
                found[(base, dump)] = p
            families.append(found)
        common = set.intersection(*(set(x) for x in families))
        incomplete = set.union(*(set(x) for x in families)) - common
        if incomplete:
            print(f"Skipping incomplete {plane} groups: {sorted(incomplete)}", flush=True)
        for key in sorted(common):
            print(f"Reading {plane} {key[1]}", flush=True)
            frames.append(read_frame([x[key] for x in families], plane, key[1]))
    if not frames:
        raise ValueError("No complete fluid/three_t/laser XY or XZ slice groups found")
    return frames


def color_limits(frames):
    extrema = {k: [min(float(f["data"][k].min()) for f in frames),
                   max(float(f["data"][k].max()) for f in frames)] for k in FIELDS}
    # Share one temperature scale across species, planes, and times.
    tmin = min(extrema[k][0] for k in ("Ti", "Te", "Tr"))
    tmax = max(extrema[k][1] for k in ("Ti", "Te", "Tr"))
    limits = {"density": [extrema["density"][0], extrema["density"][1]],
              "speed": [0, max(1, 100 * math.ceil(extrema["speed"][1] / 100))],
              "level": [-0.5, max(0.5, extrema["level"][1] + 0.5)]}
    for k in ("Ti", "Te", "Tr"):
        limits[k] = [tmin, max(tmax, tmin * 1.01)]
    for k in ("Q", "path"):
        maximum = max(extrema[k][1], 1e-20)
        positives = [f["data"][k][f["data"][k] > 0] for f in frames]
        minimum = min((float(a.min()) for a in positives if a.size), default=maximum / 1e6)
        limits[k] = [max(minimum, maximum / 1e6), maximum]
        if limits[k][0] == limits[k][1]:
            limits[k][0] /= 10
    if limits["density"][0] == limits["density"][1]:
        limits["density"][1] *= 1.01
    return limits, extrema


def panel(fig, ax, frame, key, limits, title=None):
    values = frame["data"][key]
    name, unit, cmap_name = FIELDS[key]
    linear = key in ("speed", "level")
    norm = Normalize(*limits[key]) if linear else LogNorm(*limits[key])
    cmap = plt.get_cmap(cmap_name, int(limits[key][1] + 0.5) + 1 if key == "level" else None).copy()
    cmap.set_bad("#e9eef3")
    if key in ("Q", "path"):
        values = np.ma.masked_less_equal(values, 0)
    x, y = frame["edges"]
    plot = ax.pcolormesh(x, y, values, shading="flat", cmap=cmap, norm=norm, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ [mm]")
    ax.set_ylabel(r"$x_2$ [mm]" if frame["plane"] == "xy" else r"$x_3$ [mm]")
    ax.set_xticks(np.linspace(-0.4, 0.6, 6))
    ax.set_yticks(np.arange(-0.6, 0.61, 0.3))
    ax.tick_params(labelsize=8)
    ax.set_title(title or name, loc="left", fontsize=11, fontweight="bold", pad=8)
    if key in ("Te", "Ti", "Tr", "Q", "path", "speed"):
        xc, yc = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
        if frame["data"]["density"].min() < 0.1 < frame["data"]["density"].max():
            ax.contour(xc, yc, frame["data"]["density"], levels=[0.1],
                       colors=["#b6c5d3"], linewidths=0.45, alpha=0.7)
    if key == "level":
        for a, b, c, d in frame["rectangles"]:
            ax.plot([a, b, b, a, a], [c, c, d, d, c], color="white", lw=0.3, alpha=0.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    extend = "min" if key in ("Q", "path") and np.any((values > 0) & (values < limits[key][0])) else "neither"
    bar = fig.colorbar(plot, ax=ax, pad=0.025, fraction=0.045, extend=extend)
    bar.set_label(unit + (" · log" if not linear else ""), fontsize=9)
    bar.ax.tick_params(labelsize=8)
    if key == "level":
        bar.set_ticks(range(int(limits[key][1] + 0.5)))
    if key in ("Q", "path") and np.ma.count(values) == 0:
        ax.text(0.5, 0.95, "No laser signal in this dump", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#26384a")


def decorate(fig, title, subtitle, footer):
    height = fig.get_figheight()
    # Reserve header space in inches so the shorter mesh figures keep the same
    # text separation as the taller field overviews.
    fig.text(0.065, 1 - 0.16 / height, title, ha="left", va="top", fontsize=18,
             fontweight="bold", color="#17304a")
    fig.text(0.065, 1 - 0.52 / height, subtitle, ha="left", va="top", fontsize=10,
             color="#4d6073")
    fig.text(0.065, 0.014, footer, ha="left", fontsize=8, color="#4d6073")
    fig.tight_layout(rect=(0.025, 0.045, 0.99, 1 - 0.82 / height), w_pad=2.2, h_pad=1.6)


def save_figure(fig, stem, dpi, pdf=False):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    if pdf:
        fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)


def render_frame(frame, limits, out, run_name, dpi):
    stem = f"dci_{frame['plane']}_{frame['dump']}"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
    for ax, key in zip(axes.flat, OVERVIEW):
        panel(fig, ax, frame, key, limits)
    normal = "x3" if frame["plane"] == "xy" else "x2"
    dx = frame["mesh"]["cell_width_code"][0] * frame["scales"]["length_mm"] * 1000
    decorate(fig, f"DCI 3D  /  {frame['plane'].upper()} plane",
             f"{run_name}  ·  t = {frame['time_ns']:g} ns  ·  cycle {frame['cycle']}  ·  "
             f"requested {normal} = 0  ·  finest cell = {dx:.3f} µm",
             "Saved cell values; no spatial interpolation. Shared color scales across all frames. "
             "Gray contours: density = 0.1 g/cm³. Light gray laser cells: zero deposition.")
    save_figure(fig, out / stem, dpi, pdf=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, key in zip(axes, ("path", "level")):
        panel(fig, ax, frame, key, limits)
    decorate(fig, f"Laser sampling & mesh  /  {frame['plane'].upper()}",
             f"{run_name}  ·  t = {frame['time_ns']:g} ns  ·  "
             f"{frame['mesh']['native_cells']:,} native cells in this slice",
             "Path is the summed ray-segment length per native cell, not individual ray tracks; "
             "it depends on cell size and ray sampling.")
    save_figure(fig, out / (stem + "_mesh"), dpi)
    return stem


def render_summary(frames, limits, out, run_name, dpi):
    latest = [max((f for f in frames if f["plane"] == p), key=lambda f: f["time_ns"])
              for p in ("xy", "xz") if any(f["plane"] == p for f in frames)]
    fig, axes = plt.subplots(len(latest), 3, figsize=(15, 5.2 * len(latest)), squeeze=False)
    for row, frame in zip(axes, latest):
        for ax, key in zip(row, ("density", "Te", "Q")):
            panel(fig, ax, frame, key, limits, f"{frame['plane'].upper()} · {FIELDS[key][0]}")
    times = ", ".join(f"{f['plane'].upper()}: {f['time_ns']:g} ns (cycle {f['cycle']})" for f in latest)
    decorate(fig, f"DCI 3D  /  {run_name}", times + "  ·  latest saved central slices",
             "Shared scales across both planes. No spatial interpolation. "
             "Gray contours: density = 0.1 g/cm³. Light gray laser cells: zero deposition.")
    save_figure(fig, out / "latest_overview", dpi, pdf=True)
    for plane in ("xy", "xz"):
        selected = sorted((f for f in frames if f["plane"] == plane), key=lambda f: f["time_ns"])
        if len(selected) < 2:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
        for row, frame in zip(axes, (selected[0], selected[-1])):
            for ax, key in zip(row, ("density", "Te", "Q")):
                panel(fig, ax, frame, key, limits, f"{frame['time_ns']:g} ns · {FIELDS[key][0]}")
        decorate(fig, f"Evolution  /  {plane.upper()} plane", f"{run_name}  ·  first and latest saved dumps",
                 "Identical color scales at both times. Gray contours: density = 0.1 g/cm³. "
                 "Light gray laser cells: zero deposition.")
        save_figure(fig, out / f"evolution_{plane}", dpi, pdf=True)


def write_animations(out, manifest):
    """Animate the exported panels using one GIF palette for the whole series."""
    from PIL import Image

    animations = []
    for plane in ("xy", "xz"):
        entries = sorted((f for f in manifest["frames"] if f["plane"] == plane),
                         key=lambda f: f["time_ns"])
        if len(entries) < 2:
            continue
        frames, thumbnails = [], []
        for entry in entries:
            with Image.open(out / entry["png"]) as source:
                size = (1400, round(source.height * 1400 / source.width))
                frame = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            frames.append(frame)
            thumbnails.append(frame.resize((350, round(size[1] / 4)), Image.Resampling.LANCZOS))
        # A common palette prevents per-frame changes in the color mapping.
        contact = Image.new("RGB", (350, sum(f.height for f in thumbnails)), "white")
        y = 0
        for thumbnail in thumbnails:
            contact.paste(thumbnail, (0, y))
            y += thumbnail.height
        palette = contact.quantize(colors=256)
        indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        name = f"evolution_{plane}.gif"
        indexed[0].save(out / name, save_all=True, append_images=indexed[1:],
                        duration=750, loop=0, disposal=2, optimize=False)
        animations.append({"plane": plane, "file": name, "frames": len(entries),
                           "times_ns": [f["time_ns"] for f in entries],
                           "interpolation": "none", "duration_per_frame_ms": 750})
        for image in frames + thumbnails + indexed + [contact, palette]:
            image.close()
        print(f"Rendered {name} ({len(entries)} saved times)", flush=True)
    manifest["animations"] = animations


def write_viewer(out, manifest, run_name):
    # Local assets and plain JavaScript: works offline by opening index.html.
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCI 3D — RUN_NAME</title>
<style>body{font:16px system-ui,sans-serif;background:#edf2f7;color:#17304a;margin:0}
main{max-width:1500px;margin:auto;padding:28px}h1{margin:0 0 8px}p{line-height:1.5}
nav{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:16px 0}
button,select{font:inherit;padding:8px 12px;border:1px solid #bccad8;border-radius:6px;background:white}
img{width:100%;height:auto;background:white;border:1px solid #d9e2ec;border-radius:8px}
a{color:#175f9d}#time{font-weight:600}small{display:block;line-height:1.5;margin:12px 0}
</style><main><h1>DCI 3D · RUN_NAME</h1>
<p>Saved central slices. Compare planes and times with identical physical color scales.</p>
<nav><label>Plane <select id="plane"><option>xy</option><option>xz</option></select></label>
<label>View <select id="view"><option value="">Density, temperatures, speed & deposition</option>
<option value="_mesh">Laser path & adaptive mesh</option></select></label>
<button id="prev" aria-label="Previous saved time">←</button>
<input id="step" type="range" min="0" value="0" aria-label="Saved snapshot">
<button id="next" aria-label="Next saved time">→</button><span id="time"></span></nav>
<img id="figure" alt="DCI simulation fields">
<p><a id="png">Open PNG</a> · <a id="pdf">Open PDF</a> ·
<a href="latest_overview.png">Latest XY/XZ overview</a> ·
<a id="comparison">First/latest comparison</a> · <a id="animation">Animation (GIF)</a> ·
<a href="manifest.json">Data & validation</a></p>
<small>Temperatures are in eV, density in g/cm³, speed in km/s, laser deposition in W/cm³.
Cells intersecting the requested zero plane are used; native cell-center offsets vary with AMR level.
Coarse values are repeated without interpolation. Contours show density = 0.1 g/cm³.
Zero laser values are light gray. Laser maps use six decades at most; values below the range use
the low-end color and an extended colorbar. Path sums are cell diagnostics, not reconstructed rays.</small>
<small>This is a snapshot of available output files, not a live monitor. Full 3D volumes are not
loaded. Rerun plot_run.py to include newly written slice dumps.</small></main>
<script>const frames=FRAME_DATA;const animations=ANIMATION_DATA;const $=id=>document.getElementById(id);
function selected(){return frames.filter(f=>f.plane===$('plane').value)}
function update(reset=false){let list=selected();if(!list.length)return;
$('step').max=list.length-1;if(reset)$('step').value=list.length-1;
let f=list[Number($('step').value)];let suffix=$('view').value;
let stem='dci_'+f.plane+'_'+f.dump+suffix;
$('figure').src=stem+'.png';$('png').href=stem+'.png';$('pdf').href=stem+'.pdf';
$('pdf').hidden=!!suffix;$('comparison').href='evolution_'+f.plane+'.png';
$('comparison').hidden=list.length<2;
let animation=animations.find(a=>a.plane===f.plane);
$('animation').hidden=!animation;if(animation)$('animation').href=animation.file;
$('time').textContent='t = '+Number(f.time_ns.toPrecision(8))+' ns · cycle '+f.cycle;
$('prev').disabled=Number($('step').value)===0;$('next').disabled=Number($('step').value)===list.length-1;}
function move(d){$('step').value=Math.max(0,Math.min(Number($('step').max),Number($('step').value)+d));update()}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('step').oninput=()=>update();
$('plane').onchange=()=>update(true);$('view').onchange=()=>update();
document.onkeydown=e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)};
for(let o of Array.from($('plane').options)){if(!frames.some(f=>f.plane===o.value))o.remove()}
update(true);</script></html>'''
    rows = [{k: f[k] for k in ("plane", "dump", "time_ns", "cycle")} for f in manifest["frames"]]
    page = page.replace("RUN_NAME", html.escape(run_name)).replace("FRAME_DATA", json.dumps(rows))
    page = page.replace("ANIMATION_DATA", json.dumps(manifest.get("animations", [])))
    (out / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="DCI run directory or its bin directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of rendering processes (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    bin_dir = args.path.resolve()
    if (bin_dir / "bin").is_dir():
        bin_dir /= "bin"
    out = args.output_dir or bin_dir.parent / "plots" / "fields"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    frames = discover(bin_dir)
    limits, extrema = color_limits(frames)
    manifest = {"source": str(bin_dir), "color_limits": limits, "extrema": extrema,
                "temperature_unit": "eV", "density_unit": "g/cm^3",
                "velocity_unit": "km/s", "deposition_unit": "W/cm^3", "frames": []}
    render = partial(render_frame, limits=limits, out=out,
                     run_name=bin_dir.parent.name, dpi=args.dpi)
    with ExitStack() as stack:
        if args.workers > 1:
            # Fork preserves the plotting style and shares the loaded arrays on
            # this Linux analysis host. Each process writes distinct files.
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context("fork")))
            rendered = pool.map(render, frames)
        else:
            rendered = map(render, frames)
        for frame, stem in zip(frames, rendered):
            entry = {k: frame[k] for k in ("plane", "dump", "time_ns", "cycle", "mesh", "scales", "sources")}
            entry["png"] = stem + ".png"
            entry["extrema"] = {k: [float(v.min()), float(v.max())] for k, v in frame["data"].items()}
            manifest["frames"].append(entry)
            print(f"Rendered {stem}", flush=True)
    render_summary(frames, limits, out, bin_dir.parent.name, args.dpi)
    write_animations(out, manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_viewer(out, manifest, bin_dir.parent.name)
    print(f"Viewer: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
