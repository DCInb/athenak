# 10 kJ laser drive of an open CH spherical shell

This case models a three-dimensional spherical-cap shell with outer radius 1 mm,
thickness 0.2 mm, and initial CH density 1.1 g/cm^3. The requested 50 degree opening is
interpreted as the **full included angle**, giving a 25 degree half-angle. The cap is
centered on the +x1 axis; a laser entering from negative x1 crosses the open side and
illuminates the concave inner surface.

The beam uses AthenaK's Gaussian aperture, `exp(-2 r^2/R^2)`, with a hard/1/e^2 radius
of 0.32 mm. The inner surface projects to 0.3381 mm, so the spot covers 89.6% of its
projected area. AthenaK's current beam rays are
parallel: this is spot coverage, not geometric ray convergence to a focal point.

## Physical normalization

The code units are 1 mm, 1 ns, and 1.1 g/cm^3. Fully ionized equimolar CH is represented
by a fixed electron heat-capacity fraction 7/9, electron number 3.242691e23 per gram,
collision-weighted Zeff = 37/7, and an ideal-gas gamma of 5/3. The initial ion and
electron temperatures are 1 keV and are equilibrated after each step. The temperature
was not specified in the request; 1 keV keeps the fixed fully ionized approximation
self-consistent and avoids an unphysical cold-plasma absorption singularity.

`1 omega` is taken to mean the 1.053 micrometre Nd:glass fundamental. The square pulse is
2 TW for 5 ns, hence exactly 10 kJ incident energy. Gaussian ray weights are normalized
so their powers sum exactly to 2 TW.

The straight tracer includes oblique critical-density turning and specular reflection
from the local density-gradient normal. Evolving grazing rays are capped after eight
turns; their unresolved power is kept separate from deposited and escaped power. The
analysis reports this quantity prominently rather than treating it as target coupling.

The laser implementation evaluates its on/off state at the cycle start and does not
shorten a step at a beam endpoint. `run_case.py` therefore advances to exactly 5 ns,
writes a restart, disables the laser, and advances the same physical state to 10 ns.
This avoids an O(dt) pulse-energy overshoot.

## Build and run

From the repository root:

```bash
python3 laser-shell/run_case.py --build --clean --ranks 1 --gpus 0
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/analyze_run.py
```

The launcher uses the local AthenaK CUDA/MPI helper and builds the external problem with
`-DPROBLEM=../../laser-shell/laser_shell`. Outputs at 0, 2.5, 5, 7.5, and 10 ns include
fluid, ion/electron, and laser-deposition fields. `diagnostics.md` and `results.json`
verify the geometry, incident energy, pulse cutoff, transport conservation, and final
simulation time.

The production mesh is `140 x 80 x 80` (0.025 mm cells), resolving the shell thickness
with eight cells. The 1,024 equal-area rays provide about two samples per illuminated
transverse cell. One V100 is the default because this sub-million-cell mesh is too small
to amortize laser ray handoff across multiple MPI ranks.

## Verified run

The 2026-07-27 production run completed at 5 ns and restarted to exactly 10 ns. Measured
initial geometry was 0.80029--0.99961 mm in radius, 0.19932 mm thick, with a 49.8876 degree
full opening and 89.583% projected spot coverage. The launched pulse was 2 TW for 5 ns
(10 kJ) to roundoff; 3.10083 kJ was deposited. The maximum instantaneous unresolved
grazing-ray fraction was 66.695%, so this run establishes the requested setup and idealized
hydrodynamic response but is not a converged multiple-reflection coupling calculation.

Generated central-plane visualizations are under `plots/`, including
`density_evolution.gif`, `electron_temperature_evolution.gif`, and the 2.5/5 ns cumulative
laser-energy maps. Generate the ray-path occupancy frames and animation with:

```bash
/home/mengqi/.venvs/athenak-vis/bin/python laser-shell/plot_rays.py
```

This writes `plots/laser_rays/*.png` and `plots/laser_rays.gif`. These plots show the
cell-rasterized, power-unweighted ray path length and path-weighted mean direction over
CH density contours. AthenaK does not retain the individual ray polylines in its binary
dumps, so the plot brightness is occupancy rather than the Gaussian beam intensity.

## Model limits

This is an idealized laser-plasma calculation, not a predictive cold-solid CH model.
AthenaK currently uses an ideal-gas EOS with fixed ionization and omits solid strength,
phase transitions, electron thermal conduction, state-dependent Spitzer equilibration,
and dynamic ionization. Thermal radiation is deliberately omitted because no CH opacity
table was supplied. Incident energy and deposited energy are reported separately because
inverse bremsstrahlung, critical-surface reflection, and escape determine how much of the
10 kJ actually couples to the target.
