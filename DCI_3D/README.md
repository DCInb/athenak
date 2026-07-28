# Provisional DCI_3D case

The archive audit and every reference-versus-user override are recorded in
[`REFERENCE_MAPPING.md`](REFERENCE_MAPPING.md).

This directory is an executable first-version scaffold for a three-dimensional,
laser-driven CH target with separate ion, electron, and multigroup-radiation energies,
ion-electron exchange, dual-energy MHD, a Biermann battery, and a conservative material
tracer.

It is not yet a reconstruction of the supplied reference case.  On 2026-07-28,
`/home/mengqi/Research/athenak-2t/3d_zb.zip` was replaced with a 4,006,600-byte archive
containing 61 entries, including FLASH sources, parameter decks, and material tables
(SHA-256 `952708009c9e3bc00dc645e11c9c0f804614def9c70cc999b78c92f16c8a96cf`).  This
scaffold predates that populated archive and has not yet been reconciled with it.  Every
choice below is therefore either inherited from the already validated
`laser-shell`/`laser-target` cases or explicitly labelled as an assumption; audit the
reference archive before making physical-fidelity claims.

## Files

- `dci_3d.cpp`: target initialization, vacuum FLD boundary, CH scalar, and 20 integrated
  diagnostics.
- `dci_3d.athinput`: production mesh and 5 ns laser-on phase.  The launcher restarts it
  laser-off to 10 ns.
- `dci_3d_calibration.athinput`: identical allocation and physics with `nlim=2` and every
  output disabled.
- `ch_surrogate.opacity`: provisional single-material CH-like opacity, not reference
  data.
- `run_case.py`: guarded eight-GPU builder/launcher.

## Explicit assumptions

### Target and normalization

The code units are inherited from `laser-shell`:

| Quantity | Code unit |
| --- | ---: |
| length | 1 mm |
| time | 1 ns |
| density | 1.1 g/cm3 |
| velocity | 1.0e8 cm/s |
| pressure | 1.1e16 erg/cm3 |
| temperature | 1.737267449e8 K = 14.970613 keV |
| power | 1.1e22 erg/s |
| magnetic field | 3.71793e8 G |

The target is the same far-side spherical cap as `laser-shell`, not a demonstrated
double-cone target and not a full spherical shell with a polar hole.  It occupies

```text
0.8 mm <= r <= 1.0 mm,    -x1/r >= cos(25 degrees),
```

with density 1.1 g/cm3, 0.02 mm radial smoothing, and 0.01 smoothing in angular cosine.
The full included opening is interpreted as 50 degrees.  The ambient density is
1.1e-8 g/cm3.  Ion, electron, and radiation temperatures initially equal 300 K.

Fully ionized equimolar CH is approximated by `gamma=5/3`, ion-averaged `Abar=6.5`,
`Zbar=3.5`, electron heat-capacity fraction `7/9`, electron number
`3.242691464e23/g`, and collision-weighted `Zeff=37/7`.  This fully ionized ideal-gas
model is not a predictive cold-solid EOS.

### Material scalar and mixed cells

One user passive scalar stores CH partial mass density `rho*X_CH`; the ambient fraction is
`1-X_CH`.  The smooth geometric field `alpha` is a volume fraction, so initialization is

```text
rho_CH  = alpha * rho_CH,pure
rho_amb = (1-alpha) * rho_amb,pure
rho     = rho_CH + rho_amb
X_CH    = rho_CH/rho.
```

This avoids incorrectly treating a volume fraction as a mass fraction across the
eight-order density jump.  The scalar is advected conservatively and `CH_mass` and a
mixed-mass indicator are recorded in history output.

The shared AthenaK core now contains an ideal two-material closure, local material
properties for laser absorption and Biermann terms, a standalone mixed-opacity
evaluator, and a separate-ion/electron IONMIX table loader.  The requested end-to-end
FLASH-like closure is still incomplete:

- the IONMIX loader is standalone infrastructure and is not connected to the MHD
  pressure, wave-speed, work-partition, laser, or radiation paths;
- thermal radiation still constructs one opacity table and does not yet call the
  mixed-opacity evaluator with the advected material fraction;
- this provisional DCI deck does not configure the new material closure.

Consequently, the deck uses ideal CH and a named, single-material surrogate opacity
table.  The table contains constant values over a minimal density/temperature grid:
transport opacities of 100, 10, and 1 cm2/g and Planck absorption/emission opacities of
10, 5, and 1 cm2/g in the three groups.  These are numerical assumptions, not data from
`3d_zb.zip`.  A tracer-only run must not be reported as satisfying material-specific EOS
or FLASH mixing fidelity.

The audited CH/He tables can be regenerated locally from the untracked reference archive
with `--regenerate-material-tables`, but the provisional deck does not consume them.  A
run must not be described as material-specific until the loader and mixed opacity are
wired into the production thermodynamic and radiation paths.

### Three-temperature physics

- Three thermal groups have assumed boundaries `[0, 0.1, 1, 100] keV`, or
  `[0, 0.00667975338, 0.06679753383, 6.679753383]` in code temperature.
- `arad=626.50300896` follows the shared physical normalization.
- `c_light=10` is a reduced light speed of 1.0e9 cm/s.  Physical light speed would be
  `299.792458` in code units and is not practical for this explicit FLD first version.
- The Levermore-Pomraning limiter, electron-radiation coupling, and a source CFL of 0.1
  are enabled.
- The constant ion-electron exchange time is assumed to be 0.05 ns.  Compact sensitivity
  runs at 0.01 and 0.1 ns are required before physical interpretation.
- The Biermann coefficient `4.026493224e-4` is dimensionally consistent with the CH unit
  system.  Shock suppression is enabled at threshold 0.8.
- All fluid boundaries are outflow.  The problem generator replaces only radiation-group
  ghosts with zero energy, allowing FLD energy to escape instead of making the ordinary
  zero-gradient outflow an insulated radiation wall.

### Laser

One 1.053 micrometre Nd:glass fundamental beam supplies 2 TW for exactly 5 ns: 10 kJ
incident energy.  Its 1,024 deterministic rays deposit into electron energy through
inverse bremsstrahlung, use a 1 keV absorption-only temperature floor, and permit one
critical-surface reflection.

The new lens geometry is used instead of the parallel bundle in `laser-shell`:

```text
lens center   = (+1.5, 0, 0) mm
aperture      = 0.40 mm hard and Gaussian 1/e2 radius
target center = (-0.8, 0, 0) mm
target radius = 0.32 mm
```

The 20 percent convergence maps the Gaussian 1/e2 radius to 0.32 mm.  The projected
inner-cap radius is `0.8*sin(25 deg)=0.3381 mm`, so the target spot covers 89.6 percent
of its projected area.  Straight tracing is the robust first version; refractive tracing
is a later sensitivity study.

## Uniform eight-GPU mesh

The production grid is `600 x 368 x 368` = 81,254,400 uniform cells.  MeshBlocks are
`50 x 46 x 46`, producing a `12 x 8 x 8` block lattice: 768 blocks and exactly 96 per
MPI rank/GPU.  Cell widths are 5.833, 5.435, and 5.435 micrometres, resolving the shell
thickness with roughly 34--37 cells.

The prior 2T laser-shell case measured 12,242 MiB/GPU for 112 `50^3` blocks.  Accounting
for four additional transported scalars, radiation diagnostics, and Biermann scratch
arrays predicts roughly 12.6--13.0 GiB, or 77--79 percent of each 16 GiB V100.  This is
not acceptance evidence.  The full-allocation calibration must measure every GPU:

```bash
python3 DCI_3D/run_case.py --build --clean --mode calibrate
```

The launcher returns status 2 if any peak allocation is outside 60--80 percent.  If the
mesh exceeds 80 percent, the documented fallback is `600 x 352 x 352` with
`50 x 44 x 44` blocks, followed by a fresh measurement.

## Build and safe validation

The launcher uses the local AthenaK build helper, MPI/CUDA/NVHPC environment, Volta-70
target, and one MPI rank per visible GPU.  It refuses duplicate GPU IDs, a rank count
other than eight, nonempty production directories without its sentinel, and occupied
GPUs unless explicitly overridden.

Print the exact build/run command without changing state:

```bash
python3 DCI_3D/run_case.py --build --mode validate --dry-run
```

Compile and validate initialization on a compact `2 x 2 x 2` MeshBlock grid (`nlim=0`):

```bash
python3 DCI_3D/run_case.py --build --clean --mode validate
```

Advance the same compact grid for two steps with every output disabled:

```bash
python3 DCI_3D/run_case.py --clean --mode smoke
```

The compact modes change only global cell counts to `100 x 92 x 92`; production-size
MeshBlocks and all physics allocations remain active.  `--nlim 0` or `--nlim 2` can
override either non-production validation mode.

Print the intended production command for the 5 ns laser phase and 10 ns restart with:

```bash
python3 DCI_3D/run_case.py --mode production --dry-run
```

An actual production launch is intentionally disabled.  The new AP radiation flux removes
the former optically thin timestep collapse: an eight-GPU compact run advanced 50 RK2
steps to `7.21e-3 ns` with positive finite fields, conserved CH mass, and an energy-closure
residual below `1e-7` of deposited laser energy.  That is stability evidence, not
production acceptance.  Resolution/opacity convergence, restart behavior, the final
20-group CH/He material path, full-mesh memory use, and evolution to both 5 ns and 10 ns
remain unverified.  Keep the guard until those checks and the reference reconciliation
are complete.

The intended production staging directory is the local, ignored `DCI_3D/run`.  A
dedicated, non-deleting configuration is provided in `tranfile_config.json`; it preserves
the run-directory tree under `~/data/DCI_3D` on `192.168.3.20`.  Validate its SSH and
rsync setup after creating the local staging directory:

```bash
mkdir -p DCI_3D/run
python3 /home/mengqi/Research/TranFile/file_watcher.py \
  --config DCI_3D/tranfile_config.json --validate-only
```

Start that dedicated watcher before production with the same command but without
`--validate-only`.  It requires four unchanged size/mtime checks spaced 15 seconds apart,
uses separate DCI log/state files, and never deletes local output.  Do not use the global
TranFile configuration for this case: its unrelated settings delete transferred `.bin`
and `.rst` files.

## Outputs and diagnostics

History is written every 0.025 ns.  It records deposited laser energy/power, total and CH
mass, material/kinetic/magnetic/ion/electron energy, all three radiation-group energies,
matter-plus-radiation energy, integrated `|B|` and Biermann-source estimate, laser and
radiation x1 moments, mixed mass, x1 momentum, and volume.

Central `x1-x2` and `x1-x3` fluid/3T/laser slices are written every 0.1 ns.  Full-volume
fluid, 3T, and laser fields are written every 0.5 ns, and restarts every 1 ns.  This lower
cadence is intentional: the earlier 112-million-cell case paused for 194--487 seconds at
each 0.1 ns full-volume output and produced 1.2 TiB.  Even after the radiation timestep
blocker is resolved, reserve several hundred GiB and remeasure both compute and I/O cost
before estimating the production runtime.

Minimum acceptance requires exact 5 and 10 ns endpoints, 10 kJ incident energy, laser
power closure, nonnegative component/group energies, conserved CH mass, nonzero
ion-electron/radiation/Biermann coupling, matter-plus-radiation energy accounting including
boundary loss, no NaNs or floor runaway, and measured 60--80 percent memory on all eight
GPUs.
