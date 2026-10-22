#!/usr/bin/env python3
"""Make a density movie from saved plane or 3D dumps with a fixed logarithmic colorbar."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
import numpy as np
from PIL import Image

from plot_initial import (CASE_DIR, assemble_fields, file_hash, input_header,
                          read_binary, style_axis, unit_scales)


def nice_bound(value, upper):
    scale = 10.0**np.floor(np.log10(value))
    choices = scale*np.array([1., 2., 5., 10.])
    candidates = choices[choices >= value] if upper else choices[choices <= value]
    return float(candidates[0] if upper else candidates[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--plane', choices=['xy', 'xz'], default='xz')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--snapshot-fps', type=int, choices=[1, 2, 4, 5, 10, 20], default=1,
                        help='Saved snapshots shown per playback second')
    args = parser.parse_args()
    run = args.run_directory.resolve()
    output = (args.output_dir or CASE_DIR/'plots'/run.name/f'density_{args.plane}_movie').resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'frames').mkdir(exist_ok=True)

    plane_paths = sorted((run/'bin').glob(f'*.fluid_{args.plane}.*.bin'))
    source_kind = f'fluid_{args.plane}' if plane_paths else 'fluid'
    source_paths = plane_paths or sorted((run/'bin').glob('*.fluid.*.bin'))
    snapshots, fields = [], []
    expected = None
    for path in source_paths:
        before = path.stat()
        raw = read_binary(str(path))
        scales = unit_scales(raw['header'])
        values, axial, transverse, normal = assemble_fields(raw, args.plane, 0., ('dens',))
        density = values['dens'].T*scales['density_gcc']
        if not np.isfinite(density).all() or np.any(density <= 0):
            raise ValueError(f'Density must be finite and positive: {path}')
        xedges = transverse*scales['length_um']
        yedges = axial*scales['length_um']
        normal_um = normal*scales['length_um']
        hydro = input_header(raw['header'])['hydro']
        settings = dict(scales=scales, normal_um=normal_um, shape=density.shape,
                        hydro_solver=hydro['rsolver'], reconstruction=hydro['reconstruct'])
        if expected is not None:
            assert settings == expected
            assert np.array_equal(xedges, saved_xedges) and np.array_equal(yedges, saved_yedges)
        else:
            expected = settings
            saved_xedges, saved_yedges = xedges.copy(), yedges.copy()
        checksum = file_hash(path)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        record = dict(source=str(path), source_sha256=checksum,
                      dump=path.name.rsplit('.', 2)[1], time_ns=raw['time']*scales['time_ns'],
                      cycle=int(raw['cycle']), minimum_gcc=float(density.min()),
                      maximum_gcc=float(density.max()))
        snapshots.append(record)
        fields.append(density)
        print(f'Read {path.name}: t={record["time_ns"]:.3f} ns, '
              f'density={density.min():.6g}..{density.max():.6g} g/cm^3', flush=True)
        del raw, values
    assert len(snapshots) > 1, 'At least two complete dumps of the requested plane are required'
    times = np.array([r['time_ns'] for r in snapshots])
    spacing = np.diff(times)
    assert np.all(spacing > 0) and np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-12), \
        'The fixed-rate movie requires uniformly spaced output times'

    minimum = min(r['minimum_gcc'] for r in snapshots)
    maximum = max(r['maximum_gcc'] for r in snapshots)
    low, high = nice_bound(minimum, False), nice_bound(maximum, True)
    if high <= low:
        high = low*10
    norm = LogNorm(low, high)
    ticks = [low] + [10.**e for e in range(int(np.ceil(np.log10(low))),
                                           int(np.floor(np.log10(high)))+1)] + [high]
    ticks = sorted(set(ticks))
    # Repeated video frames hold each actual saved state without temporal interpolation.
    fps = 20
    repeat = fps//args.snapshot_fps
    seconds_per_snapshot = 1/args.snapshot_fps
    stem = output/f'density_{args.plane}_fixed'
    rgb_frames = []
    normal_axis = '2' if args.plane == 'xz' else '3'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(8.8, 8.0), dpi=125)
        ax = fig.add_axes([0.115, 0.16, 0.70, 0.71])
        color_ax = fig.add_axes([0.835, 0.16, 0.025, 0.71])
        mesh = ax.pcolormesh(xedges, yedges, fields[0], shading='flat',
                             norm=norm, cmap='viridis', rasterized=True)
        style_axis(ax, xedges, yedges, args.plane)
        ax.set_xlabel(r'$x_3$ [$\mu$m]' if args.plane == 'xz' else r'$x_2$ [$\mu$m]', fontsize=13)
        ax.set_ylabel(r'$x_1$ (collision axis) [$\mu$m]', fontsize=13)
        bar = fig.colorbar(mesh, cax=color_ax, ticks=ticks,
                           format=LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))
        bar.minorticks_off()
        bar.set_label(r'Density $\rho$ [g cm$^{-3}$]  |  fixed log scale', fontsize=12)
        title = f'{run.name} | {expected["hydro_solver"].upper()} + {expected["reconstruction"].upper()}'
        fig.suptitle(title, x=0.5, y=0.975, fontsize=19, fontweight='bold')
        subtitle = fig.text(0.5, 0.918, '', ha='center', fontsize=12, color='#374151')
        fig.text(0.5, 0.048,
                 f'{len(snapshots)} saved snapshots | spacing {spacing[0]:.2f} ns | '
                 f'fixed density range {low:g}–{high:g} g/cm³\n'
                 f'Actual cell values; each snapshot held for {seconds_per_snapshot:g} s. '
                 'No interpolation between snapshots.',
                 ha='center', va='center', fontsize=10, color='#4b5563', linespacing=1.6)
        for i, (record, density) in enumerate(zip(snapshots, fields)):
            mesh.set_array(density.ravel())
            subtitle.set_text(rf'$t={record["time_ns"]:.2f}$ ns | '
                              rf'$x_{normal_axis}={normal_um:+.3f}$ $\mu$m | '
                              f'dump {record["dump"]} | cycle {record["cycle"]}')
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            rgb_frames.append(rgb)
            frame_path = output/'frames'/f'density_{args.plane}_{record["dump"]}.png'
            Image.fromarray(rgb).save(frame_path)
            record['png'] = str(frame_path.relative_to(output))
        plt.close(fig)

    with av.open(str(stem.with_suffix('.mp4')), mode='w', options={'movflags': '+faststart'}) as movie:
        stream = movie.add_stream('libx264', rate=fps)
        stream.width, stream.height = rgb_frames[0].shape[1], rgb_frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '17', 'preset': 'medium'}
        for rgb in rgb_frames:
            for _ in range(repeat):
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                for packet in stream.encode(frame):
                    movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)

    # A single GIF palette keeps the fixed color mapping identical in every frame.
    samples = np.concatenate([rgb[::5, ::5] for rgb in rgb_frames], axis=0)
    palette = Image.fromarray(samples).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE)
                  for rgb in rgb_frames]
    gif_frames[0].save(stem.with_suffix('.gif'), save_all=True, append_images=gif_frames[1:],
                       duration=round(1000*seconds_per_snapshot), loop=0,
                       disposal=2, optimize=False)
    peak = int(np.argmax([r['maximum_gcc'] for r in snapshots]))
    Image.fromarray(rgb_frames[peak]).save(output/'preview.png')

    with av.open(str(stem.with_suffix('.mp4'))) as movie:
        decoded = list(movie.decode(video=0))
        assert len(decoded) == len(snapshots)*repeat
        assert all((f.width, f.height) == (stream.width, stream.height) for f in decoded)
        duration = float(movie.duration/av.time_base)
        assert np.isclose(duration, len(snapshots)*seconds_per_snapshot, atol=1/fps)
    with Image.open(stem.with_suffix('.gif')) as gif:
        assert gif.n_frames == len(snapshots)
        assert gif.size == (stream.width, stream.height)
        assert sum((gif.seek(i), gif.info['duration'])[1] for i in range(gif.n_frames)) == \
            round(1000*seconds_per_snapshot)*len(snapshots)

    manifest = dict(run=str(run), created_utc=datetime.now(timezone.utc).isoformat(),
                    plane=args.plane, source_kind=source_kind, requested_normal_um=0.,
                    actual_normal_um=normal_um,
                    x_axis='x3' if args.plane == 'xz' else 'x2', y_axis='x1',
                    density_unit='g/cm^3', scale='logarithmic', cmap='viridis',
                    fixed_color_limits=[low, high], global_density_extrema=[minimum, maximum],
                    snapshot_count=len(snapshots), simulation_spacing_ns=float(spacing[0]),
                    seconds_per_snapshot=seconds_per_snapshot, encoded_fps=fps,
                    encoded_frame_count=len(decoded),
                    duration_seconds=duration, spatial_interpolation=False,
                    temporal_interpolation=False, mp4=stem.with_suffix('.mp4').name,
                    gif=stem.with_suffix('.gif').name, snapshots=snapshots)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{run.name} — {args.plane} density movie</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:17px system-ui;background:#f4f5f7;'
        'color:#263140;padding:0 16px}video{width:100%;background:white}a{color:#1759b6}</style>'
        f'<h1>{args.plane.upper()} density evolution</h1>'
        f'<p>{times[0]:.2f}–{times[-1]:.2f} ns; fixed logarithmic colorbar '
        f'{low:g}–{high:g} g/cm³. Collision axis x₁ is vertical.</p>'
        f'<video controls loop playsinline preload="metadata" poster="preview.png">'
        f'<source src="{stem.with_suffix(".mp4").name}" type="video/mp4"></video>'
        f'<p><a href="{stem.with_suffix(".mp4").name}">Download MP4</a> · '
        f'<a href="{stem.with_suffix(".gif").name}">Animated GIF</a> · '
        '<a href="manifest.json">Source data and metadata</a></p></html>')
    print(json.dumps({k: manifest[k] for k in ('snapshot_count', 'fixed_color_limits',
                     'global_density_extrema', 'duration_seconds', 'mp4', 'gif')}, indent=2))
    print(f'Output directory: {output}', flush=True)


if __name__ == '__main__':
    main()
