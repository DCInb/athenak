# 10 kJ laser drive of an open CH spherical shell

This case models a three-dimensional spherical-cap shell with outer radius 1 mm,
thickness 0.2 mm, and initial CH density 1.1 g/cm^3. The requested 50 degree opening is
interpreted as the full included angle, giving a 25 degree half-angle. The cap is
centered on the `-x1` axis. The laser enters from the `+x1` boundary, crosses the open
side, and illuminates the concave inner surface.

The beam uses AthenaK's Gaussian aperture, `exp(-2 r^2/R^2)`, with a hard and 1/e^2
radius of 0.32 mm. The inner surface projects to 0.3381 mm, so the spot covers 89.6% of
its projected area. The rays are parallel; this is broad spot coverage rather than
geometric convergence to a focal point.

## Physical normalization

The code units are 1 mm, 1 ns, and 1.1 g/cm^3. Fully ionized equimolar CH is represented
by a fixed electron heat-capacity fraction 7/9, electron number 3.242691e23 per gram,
collision-weighted `Zeff = 37/7`, and ideal-gas `gamma = 5/3`. The initial ion and
electron temperatures are both exactly 300 K.

The fully ionized inverse-bremsstrahlung law is not valid for cold, neutral CH and would
make even tenuous 300 K material artificially opaque. The hydrodynamic state remains at
300 K, while `inverse_bremsstrahlung_temperature_floor` evaluates only the absorption
coefficient at no less than 1 keV. The ambient density is `1e-8` code units to keep the
source-to-target path initially transparent.

`1 omega` is taken to mean the 1.053 micrometre Nd:glass fundamental. The square pulse is
2 TW for 5 ns, hence exactly 10 kJ incident energy. The 1,024 deterministic equal-area
Fibonacci rays carry Gaussian power weights normalized to exactly 2 TW. The aperture is
truncated at its 1/e^2 radius, so its edge ray has about 13.5% of the central intensity.

The straight tracer includes oblique critical-density turning and specular reflection
from the local density-gradient normal. Each ray is allowed one physical reflection,
and distributed transport is capped at 64 MPI waves. Unresolved grazing-ray power is
reported separately from deposited and escaped power.

## Build and run

From the repository root:

```bash
python3 laser-shell/run_case.py --build --clean
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/analyze_run.py
```

The launcher defaults to eight MPI ranks on GPUs `0-7` and writes the production data to
`/home/mengqi/data/athenak-2t/laser-shell/run`. Override that location with `--run-dir`
or `ATHENAK_LASER_SHELL_RUN_DIR`; for cleanup safety, a custom path must end in
`laser-shell/run`. Do not put this run on the repository filesystem: the 101 synchronized
full-volume output epochs and eleven restarts require roughly 1.3 TB.

The production mesh is `700 x 400 x 400`, or 112 million cells, with isotropic 5 micron
spacing and 40 cells through the shell thickness. It contains 896 `50^3` MeshBlocks,
exactly 112 per rank. Calibration measured about 12,200 MiB on each 16 GiB V100, or
74-75% of nominal device memory.

Fluid, two-temperature, and laser binary dumps are aligned exactly every 0.1 ns.
Restarts are written every 1 ns. The first phase lands exactly on the 5 ns pulse edge;
`run_case.py` restarts with the laser disabled and advances the same state to 10 ns.

### Long pauses and lower-I/O diagnostics

The completed production run wrote 1.2 TiB to the network filesystem. Each 0.1 ns
output event writes an 11.648 GB triplet (fluid, two-temperature, and laser), and each
1 ns restart is 11.350 GB. Measured output pauses were about 194--487 seconds, during
which the cycle log does not advance even though the MPI job is still writing. A long
pause near a scheduled output time is therefore expected and is not evidence of a
deadlock. Check file sizes and timestamps in the run directory before stopping the job.

For iteration runs, increase `output2/dt`, `output3/dt`, and `output4/dt`, and disable
restarts with `output5/dt = -1.0`. For a much smaller visualization run, use a scratch
copy of the input and add `slice_x3 = 0.0` to outputs 2--4; this writes the central
`x1-x2` plane instead of all 112 million cells. Keep the production settings for the
full 3D acceptance analysis.

## Ray visualization

Render the exact configured launch bundle and, when matching production dumps exist,
the 2.5 and 5 ns traced-path frames with:

```bash
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/plot_rays.py
```

The launch-only view does not require simulation output:

```bash
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/plot_rays.py --mode launch
```

To choose a run and physical frame times explicitly:

```bash
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/plot_rays.py \
  --mode traced \
  --run-dir /home/mengqi/data/athenak-2t/laser-shell/run \
  --times 2.5 5.0
```

The launch plot reproduces every aperture sample and its Gaussian power directly from
the same formula as `BuildInitialRays`. The traced plots show cell-rasterized,
power-unweighted path length and path-weighted mean direction over CH density contours.
AthenaK does not retain individual ray identities or polylines in binary dumps, so
reflected paths cannot be reconstructed as connected per-ray curves.

## Verification status

On 2026-07-28, a compact current-geometry run verified the 300 K initialization and
exact binary timestamps at 0, 0.1, and 0.2 ns. Full-grid timing tests selected 1,024 rays:
both 4,096 and 25,600 rays developed severe reflected-ray load imbalance as the plasma
evolved. A subsequent 1,024-ray run showed the same issue near 0.28 ns with eight allowed
reflections, motivating the explicit one-reflection and 64-wave transport bounds. That
bounded configuration completed a 1,200-cycle full-grid qualification through 0.314 ns
in 565 seconds. The generated launch and traced-path images are under `plots/`.

The revised 112-million-cell run subsequently completed both phases through 10 ns.
`analyze_run.py` passes its checks for completion, output and restart cadence, geometry,
300 K fields, right-side incidence, transport, and per-GPU memory use. The phase runtimes
were 18,491 seconds laser-on and 15,044 seconds laser-off; most of the visible multi-minute
pauses in the latter were synchronous output writes rather than hydrodynamic work.

## Model limits

This is an idealized laser-plasma calculation, not a predictive cold-solid CH model.
AthenaK currently uses an ideal-gas EOS with fixed ionization and omits solid strength,
phase transitions, electron thermal conduction, state-dependent equilibration, dynamic
ionization, and thermal radiation. The opacity-only temperature floor is an explicit
modeling approximation, not a replacement for an ionization-aware cold-material model.
Incident and deposited energies are reported separately because inverse bremsstrahlung,
critical-surface reflection, escape, and the reflection cap determine target coupling.
