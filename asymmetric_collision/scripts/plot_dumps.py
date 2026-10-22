#!/usr/bin/env python3
"""Plot density, Ti, Te, and velocity for every paired dump in one collision run.

The saved xy slices supply the high-cadence series. Full volumes supply xz slices
and independently verify xy at shared times. All frames share physical color
limits; temperature is logarithmic to retain both cold and hot structures.
"""

import argparse
from dataclasses import dataclass
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext
import numpy as np

from plot_initial import CASE_DIR, input_header, load_plane, panel, read_binary, unit_scales
from amr_planes import load_amr_plane


@dataclass
class Frame:
    plane: str
    dump: str
    time_ns: float
    cycle: int
    grid: tuple
    data: dict
    xedges: np.ndarray
    yedges: np.ndarray
    normal: float
    sources: list
    xy_verified: bool = False
    mesh: dict = None

    @property
    def stem(self):
        return f"{self.plane}/collision_{self.plane}_{self.dump}"


def paired_paths(run_dir, fluid_kind, thermal_kind):
    def discover(kind):
        found = {}
        for path in sorted((run_dir / "bin").glob(f"*.{kind}.*.bin")):
            basename, _, index, _ = path.name.rsplit(".", 3)
            key = (basename, int(index))
            if key in found:
                raise ValueError(f"Duplicate {kind} dump: {key}")
            found[key] = path
        return found

    fluid, thermal = discover(fluid_kind), discover(thermal_kind)
    if fluid.keys() != thermal.keys():
        raise ValueError(f"Unpaired {fluid_kind}/{thermal_kind} dumps: "
                         f"{sorted(fluid.keys() ^ thermal.keys())}")
    return [(fluid[key], thermal[key]) for key in sorted(fluid)]


def read_pair(fluid_path, thermal_path):
    fluid = read_binary(str(fluid_path))
    thermal = read_binary(str(thermal_path))
    for key in ("time", "cycle", "Nx1", "Nx2", "Nx3", "n_mbs",
                "nx1_mb", "nx2_mb", "nx3_mb", "mb_logical", "mb_index",
                "mb_geometry", "x1min", "x1max", "x2min", "x2max",
                "x3min", "x3max"):
        if not np.array_equal(fluid[key], thermal[key]):
            raise ValueError(f"Mismatched {key}: {fluid_path}, {thermal_path}")
    scales = unit_scales(fluid["header"])
    if scales != unit_scales(thermal["header"]):
        raise ValueError("Fluid and thermal units differ")
    return fluid, thermal, scales


def extract_frame(fluid, thermal, scales, plane, paths):
    refinement = input_header(fluid["header"]).get("mesh_refinement", {}).get("refinement")
    mesh = None
    if refinement in ("adaptive", "static") or np.any(np.asarray(fluid["mb_logical"])[:, 3] != 0):
        data, xedges, yedges, normal, mesh = load_amr_plane(fluid, thermal, plane, scales)
    else:
        data, xedges, yedges, normal = load_plane(fluid, thermal, plane, scales)
    if np.any(data["density"] <= 0) or any(np.any(data[k] <= 0) for k in ("Ti", "Te")):
        raise ValueError("Logarithmic density/temperature requires positive values")
    return Frame(plane, paths[0].name.rsplit(".", 2)[1],
                 float(fluid["time"] * scales["time_ns"]), int(fluid["cycle"]),
                 tuple(int(fluid[f"Nx{i}"]) for i in (1, 2, 3)),
                 data, xedges, yedges, normal, list(paths), mesh=mesh)


def load_frames(run_dir):
    frames = {}
    expected_scales = None
    verified = 0
    for fluid_kind, thermal_kind, planes in (
            ("fluid_xy", "three_t_xy", ("xy",)),
            ("fluid", "three_t", ("xy", "xz"))):
        pairs = paired_paths(run_dir, fluid_kind, thermal_kind)
        print(f"Reading {len(pairs)} {fluid_kind}/{thermal_kind} pairs", flush=True)
        for paths in pairs:
            fluid, thermal, scales = read_pair(*paths)
            if expected_scales is not None and scales != expected_scales:
                raise ValueError("Unit scales changed between dumps")
            expected_scales = scales
            for plane in planes:
                frame = extract_frame(fluid, thermal, scales, plane, paths)
                key = (plane, frame.cycle)
                if key in frames:
                    saved = frames[key]
                    if fluid_kind != "fluid" or plane != "xy":
                        raise ValueError(f"Duplicate snapshot: {key}")
                    if (saved.time_ns != frame.time_ns or saved.grid != frame.grid
                            or saved.normal != frame.normal or saved.mesh != frame.mesh
                            or not np.array_equal(saved.xedges, frame.xedges)
                            or not np.array_equal(saved.yedges, frame.yedges)
                            or any(not np.array_equal(saved.data[k], frame.data[k])
                                   for k in frame.data)):
                        raise ValueError(f"Full volume disagrees with saved xy: {paths[0]}")
                    saved.sources.extend(paths)
                    saved.xy_verified = True
                    verified += 1
                else:
                    frames[key] = frame
            # Only the extracted 2D arrays are retained, not the large 3D dumps.
            del fluid, thermal
    ordered = sorted(frames.values(), key=lambda frame: (frame.plane, frame.time_ns))
    if not ordered:
        raise ValueError(f"No paired dumps found in {run_dir / 'bin'}")
    stems = [frame.stem for frame in ordered]
    if len(stems) != len(set(stems)):
        raise ValueError("Dump numbering would produce duplicate output filenames")
    print(f"Verified {verified} xy snapshots against full volumes", flush=True)
    return ordered


def color_limits(frames):
    extrema = {name: [min(float(f.data[name].min()) for f in frames),
                      max(float(f.data[name].max()) for f in frames)]
               for name in ("density", "Ti", "Te", "speed")}

    def decade_bounds(lo, hi):
        # Float32 dumps have O(1e-7) roundoff at exact powers of ten.
        lower = 10.0 ** np.floor(np.log10(lo) + 1e-6)
        upper = 10.0 ** np.ceil(np.log10(hi) - 1e-6)
        if upper <= lower:
            upper = lower * 10
        return [float(lower), float(upper)]

    temperature = decade_bounds(min(extrema[k][0] for k in ("Ti", "Te")),
                                max(extrema[k][1] for k in ("Ti", "Te")))
    peak_speed = extrema["speed"][1]
    step = 10.0 ** np.floor(np.log10(max(peak_speed, 1.0))) / 2
    speed_max = max(step, float(np.ceil(peak_speed / step) * step))
    return {"density": decade_bounds(*extrema["density"]),
            "temperature": temperature, "speed": [0.0, speed_max]}, extrema


def velocity_arrows(ax, frame):
    data = frame.data
    xc = 0.5 * (frame.xedges[1:] + frame.xedges[:-1])
    yc = 0.5 * (frame.yedges[1:] + frame.yedges[:-1])
    x, y = np.meshgrid(xc, yc)
    if frame.mesh:
        x, y = data["cell_x"], data["cell_y"]
    # Prefer dense CH cells so tenuous ambient peaks cannot dominate the arrows.
    mask = ((data["upper_tag"] + data["lower_tag"] >= 0.5)
            & (data["density"] >= 0.01 * data["density"].max()))
    candidates = np.argwhere(mask)
    candidates = sorted(candidates, key=lambda rc: -data["density"][tuple(rc)])
    samples = []
    for row, column in candidates:
        if all((x[row, column] - x[r, c]) ** 2 + (y[row, column] - y[r, c]) ** 2
               >= 17.0 ** 2 for r, c in samples):
            samples.append((row, column))
    if not samples:
        return
    rows, columns = np.asarray(samples).T
    quiver = ax.quiver(x[rows, columns], y[rows, columns],
                      data["horizontal_velocity"][rows, columns],
                      data["v1"][rows, columns], angles="xy", scale_units="xy",
                      scale=1 / 0.075, pivot="tail", color="black", width=0.005,
                      headwidth=3.2, headlength=4, zorder=4)
    quiver.set_path_effects([effects.withStroke(linewidth=1.1, foreground="white")])
    key = ax.quiverkey(quiver, 0.60, 0.065, 200.0, "200 km/s", labelpos="E",
                       color="white", labelcolor="white", fontproperties={"size": 9})
    key.text.set_path_effects([effects.withStroke(linewidth=2, foreground="#333333")])


def log_ticks(bounds):
    low, high = [int(round(np.log10(value))) for value in bounds]
    step = max(1, int(np.ceil((high - low) / 5)))
    exponents = sorted(set(range(low, high + 1, step)) | {high})
    return [10.0 ** exponent for exponent in exponents]


def meshblock_grid(ax, frame):
    """Outline the actual leaf blocks intersecting this slice, in plotting axes."""
    if not frame.mesh:
        return
    outlines = []
    for block in frame.mesh["slice_blocks"]:
        left, right, bottom, top = block["bounds_um"]
        outlines.append([(left, bottom), (right, bottom), (right, top),
                         (left, top), (left, bottom)])
    lines = LineCollection(outlines, colors="white", linewidths=0.55, zorder=1.9)
    lines.set_path_effects([effects.withStroke(linewidth=1.05, foreground="#20242b")])
    ax.add_collection(lines)


def render_frame(frame, limits, output_dir, dpi):
    normal_axis = "3" if frame.plane == "xy" else "2"
    grid = r"\times".join(str(n) for n in frame.grid)
    if frame.mesh:
        dx_min, dx_max = frame.mesh["cell_width_range_um"]["x1"]
        dx = f"{dx_min:.3f}" if np.isclose(dx_min, dx_max) else f"{dx_min:.3f}–{dx_max:.3f}"
        subtitle = (rf"$t={frame.time_ns:.2f}$ ns  |  $x_{normal_axis}=0$ cell slice"
                    rf"  |  AMR, ${grid}$ root  |  $\Delta x={dx}$ $\mu$m")
    else:
        subtitle = (rf"$t={frame.time_ns:.2f}$ ns  |  $x_{normal_axis}={frame.normal:+.2f}$ $\mu$m"
                    rf"  |  ${grid}$ grid  |  $\Delta x={frame.yedges[1]-frame.yedges[0]:.3f}$ $\mu$m")
    with plt.rc_context({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.labelsize": 10, "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.5))
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.10, top=0.87,
                            wspace=0.34, hspace=0.34)
        fig.suptitle("Asymmetric compressed-shell collision", y=0.97,
                     fontsize=17, fontweight="bold")
        fig.text(0.5, 0.927, subtitle, ha="center", fontsize=11, color="#444444")
        definitions = (
            ("density", "(a) Mass density", "viridis", LogNorm(*limits["density"]),
             r"$\rho$ [g cm$^{-3}$] (log scale)", log_ticks(limits["density"])),
            ("Ti", "(b) Ion temperature", "magma", LogNorm(*limits["temperature"]),
             r"$T_i$ [eV] (log scale)", log_ticks(limits["temperature"])),
            ("Te", "(c) Electron temperature", "magma", LogNorm(*limits["temperature"]),
             r"$T_e$ [eV] (log scale)", log_ticks(limits["temperature"])),
            ("speed", "(d) Speed and in-plane velocity", "cividis", Normalize(*limits["speed"]),
             r"$|\mathbf{v}|$ [km s$^{-1}$]", np.linspace(*limits["speed"], 6)),
        )
        for ax, (name, title, cmap, norm, label, ticks) in zip(axes.flat, definitions):
            bar = panel(fig, ax, frame.data[name], frame.xedges, frame.yedges,
                        frame.plane, title, cmap, norm, label, ticks)
            if isinstance(norm, LogNorm):
                bar.formatter = LogFormatterMathtext()
                bar.update_ticks()
        meshblock_grid(axes[0, 0], frame)
        velocity_arrows(axes[1, 1], frame)
        if frame.time_ns == 0:
            vertical = 0.5 * (frame.yedges[1:] + frame.yedges[:-1])
            column = frame.data["density"].shape[1] // 2
            for tag, label in (("upper_tag", "Upper shell"), ("lower_tag", "Lower shell")):
                row = np.argmax(frame.data["density"][:, column] * frame.data[tag][:, column])
                y = vertical[row]
                axes[0, 0].text(0, y, label, ha="center", va="center",
                                color="white", fontsize=9,
                                path_effects=[effects.withStroke(
                                    linewidth=1.5, foreground="black")])
        density_note = ("White lines in (a): MeshBlock boundaries." if frame.mesh
                        else "Color scales fixed across all dumps.")
        fig.text(0.5, 0.035,
                 "Saved simulation cells; no spatial interpolation. " + density_note + "\n"
                 "Color in (d): full 3D speed. Arrows: in-plane velocity in shell-origin material. + marks the origin.",
                 ha="center", va="center", fontsize=9, color="#555555", linespacing=1.6)
        for extension in ("png", "pdf"):
            path = output_dir / f"{frame.stem}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
        plt.close(fig)


def write_catalog(frames, limits, extrema, run_dir, output_dir, dpi):
    records = []
    for frame in frames:
        records.append({"plane": frame.plane, "dump": frame.dump,
                        "time_ns": frame.time_ns, "cycle": frame.cycle,
                        "grid": frame.grid, "normal_um": frame.normal,
                        "sources": [str(p.relative_to(run_dir)) for p in frame.sources],
                        "xy_verified_against_volume": frame.xy_verified,
                        "png": f"{frame.stem}.png", "pdf": f"{frame.stem}.pdf",
                        "extrema": {name: [float(frame.data[name].min()),
                                           float(frame.data[name].max())]
                                    for name in extrema}})
        if frame.mesh:
            records[-1]["mesh"] = frame.mesh
    manifest = {"run_directory": str(run_dir), "dpi": dpi, "color_limits": limits,
                "extrema": extrema, "frames": records}
    is_amr = any(frame.mesh for frame in frames)
    if is_amr:
        manifest["slice_method"] = ("Cell containing zero in each AMR leaf block, using AthenaK's "
                                    "cell selection; coarse values repeated without interpolation.")
        manifest["density_meshblock_grid"] = {
            "source": "Saved geometry of leaf MeshBlocks intersecting the slice",
            "color": "white", "linewidth_pt": 0.55,
            "outline_color": "#20242b", "outline_linewidth_pt": 1.05,
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    title = f"Compressed-shell collision — {run_dir.name}"
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body { margin: 0 auto; max-width: 1100px; padding: 24px; color: #20242b; background: #f4f5f7;
       font: 16px/1.5 system-ui, sans-serif; }
h1 { font-size: 26px; margin-bottom: 8px; }
p { color: #505866; }
nav { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding: 12px 0; }
button, select { font: inherit; padding: 6px 10px; }
input { flex: 1; min-width: 180px; }
output { min-width: 235px; font-variant-numeric: tabular-nums; }
img { display: block; width: 100%; background: white; border-radius: 8px; }
a { color: #164caf; }
</style>
<h1>__TITLE__</h1>
<p>__COUNTS__. All frames use the same color scales. Density and temperatures are logarithmic.
The collision axis x₁ is vertical; the nearest saved cell plane is shown.</p>
<nav aria-label="Snapshot controls">
<label>Plane <select id="plane"><option value="xy">x₁–x₂</option><option value="xz">x₁–x₃</option></select></label>
<button id="previous" aria-label="Previous snapshot">←</button>
<button id="play">Play</button><button id="next" aria-label="Next snapshot">→</button>
<input id="slider" type="range" min="0" value="0" step="1" aria-label="Snapshot">
<output id="time" aria-live="polite"></output>
<a id="png">PNG</a><a id="pdf">PDF</a>
</nav>
<img id="snapshot" alt="Density, ion temperature, electron temperature, speed and velocity">
<p><a href="README.md">Plot notes and all files</a> · <a href="manifest.json">Source and range metadata</a></p>
<script>
const frames = __FRAMES__;
const plane = document.getElementById('plane'), slider = document.getElementById('slider');
const play = document.getElementById('play');
let timer = null;
function selected() { return frames.filter(f => f.plane === plane.value); }
function show() {
  const list = selected(); slider.max = list.length - 1;
  slider.value = Math.min(Number(slider.value), list.length - 1);
  const f = list[Number(slider.value)];
  document.getElementById('snapshot').src = f.png;
  document.getElementById('snapshot').alt = `${plane.value} at ${f.time_ns.toFixed(2)} ns: density, Ti, Te, speed and velocity`;
  document.getElementById('time').textContent = `t = ${f.time_ns.toFixed(2)} ns · ${Number(slider.value)+1} / ${list.length}`;
  document.getElementById('png').href = f.png; document.getElementById('pdf').href = f.pdf;
}
function move(step) { slider.value = (Number(slider.value) + step + selected().length) % selected().length; show(); }
function stop() { clearInterval(timer); timer = null; play.textContent = 'Play'; }
for (const option of [...plane.options]) {
  if (!frames.some(f => f.plane === option.value)) option.remove();
}
plane.addEventListener('change', () => { stop(); slider.value = 0; show(); });
slider.addEventListener('input', () => { stop(); show(); });
document.getElementById('previous').onclick = () => { stop(); move(-1); };
document.getElementById('next').onclick = () => { stop(); move(1); };
play.onclick = () => {
  if (timer) { stop(); return; }
  play.textContent = 'Pause'; timer = setInterval(() => move(1), 300);
};
show();
</script></html>
'''
    counts = "; ".join(f"{sum(f.plane == p for f in frames)} {p} snapshots"
                       for p in ("xy", "xz"))
    page = (page.replace("__TITLE__", html.escape(title)).replace("__COUNTS__", counts)
            .replace("__FRAMES__", json.dumps(records).replace("<", "\\u003c")))
    if is_amr:
        page = page.replace("the nearest saved cell plane is shown.",
                            "each AMR leaf contributes its cell containing the zero plane. "
                            "Coarse-cell values are preserved without interpolation. "
                            "White lines on density mark MeshBlock boundaries.")
    (output_dir / "index.html").write_text(page)
    lines = [f"# {title}", "", f"Source: `{run_dir}`.", "",
             f"{counts}, each in PNG and PDF. [Interactive viewer](index.html).", "",
             "Collision axis x1 is vertical. Unit conversions come from each saved header; "
             "the nearest cell to the zero plane is selected, without spatial interpolation.", "",
             f"Shared color limits: density {limits['density']} g/cm³ (log), "
             f"Ti and Te {limits['temperature']} eV (log), "
             f"speed {limits['speed']} km/s (linear). Limits cover the evolved data across "
             "all plotted frames. Both temperatures use the same scale.", "",
             "Speed colors include all three velocity components. Arrows show the measured "
             "in-plane components where the upper-plus-lower shell-origin fraction is ≥ 0.5 "
             "and density ≥ 1% of that slice's "
             "peak, preferring dense cells with at least 17 µm separation. The common arrow "
             "scale is 0.075 µm per km/s; the key is 200 km/s. Colors include every saved cell, "
             "including corona and ambient plasma.", "",
             f"{sum(f.xy_verified for f in frames)} saved xy slices were checked against their "
             "full-volume counterparts: all plotted fields and origin tags agree exactly. "
             "Fluid/thermal times, cycles, block geometry and units are checked for every pair. "
             "Per-frame extrema, source files, cycles and coordinates are in [manifest.json](manifest.json).", "",
             "Regenerate from the case directory with NumPy and Matplotlib installed:", "",
             "```bash", f"python plot_dumps.py runs/{run_dir.name} --output-dir plots/{output_dir.name}",
             "```", "", "| Plane | Dump | Time (ns) | Cycle | PNG | PDF |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    if is_amr:
        lines[6] = ("Collision axis x1 is vertical. Unit conversions come from each saved header. "
                    "Each leaf block contributes the cell containing the requested zero plane, "
                    "using AthenaK's half-open bounds and cell-index rule. Coarse cells are "
                    "repeated on the finest slice grid with no spatial interpolation; arrows "
                    "remain at native cell centers. The root grid, slice refinement levels, "
                    "actual cell widths, and normal cell-center range are recorded per frame "
                    "in manifest.json. A null normal_um means that the normal cell center "
                    "varies with refinement level. Every slice is checked for gaps and overlaps.")
        lines[6] += (" White lines with a dark outline on the density panel mark the saved "
                     "leaf MeshBlock boundaries; their projected bounds and refinement levels "
                     "are recorded in each frame's mesh.slice_blocks metadata.")
    lines.extend(f"| {f.plane} | {f.dump} | {f.time_ns:.2f} | {f.cycle} | "
                 f"[PNG]({f.stem}.png) | [PDF]({f.stem}.pdf) |" for f in frames)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    run_dir = args.run_directory.expanduser().resolve()
    output_dir = (args.output_dir or CASE_DIR / "plots" / run_dir.name).resolve()
    frames = load_frames(run_dir)
    limits, extrema = color_limits(frames)
    print("Color limits:", json.dumps(limits), flush=True)
    print("Measured extrema:", json.dumps(extrema), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames, 1):
        render_frame(frame, limits, output_dir, args.dpi)
        print(f"[{index}/{len(frames)}] {frame.stem}: t={frame.time_ns:.2f} ns", flush=True)
    write_catalog(frames, limits, extrema, run_dir, output_dir, args.dpi)
    print(f"Wrote {len(frames)} PNG/PDF pairs and {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
