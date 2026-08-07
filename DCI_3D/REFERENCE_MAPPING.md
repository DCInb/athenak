# DCI_3D reference mapping

This case uses the populated `3d_zb.zip` as a material/physics reference while honoring
the user's explicit target and laser requirements where the archive conflicts.  The
document records the target mapping and explicit modeling choices implemented by the two
DCI decks.  Passing the production acceptance gates remains a separate requirement;
current evidence and launch controls are described in `README.md`.

The archive audited on 2026-07-28 has:

- size `4,006,600` bytes;
- 61 ZIP entries (59 files and two directories), `15,534,392` bytes uncompressed;
- SHA-256 `952708009c9e3bc00dc645e11c9c0f804614def9c70cc999b78c92f16c8a96cf`.

## Why this is a mapping, not a literal reconstruction

The archive does not contain one self-consistent buildable case.  The historical
`DCI3D_eighth1um` setup requests a missing `flash_eighth1um.par`.  Its closest surviving
deck, `flash_eighth.par`, is an AMR positive-octant case.  The newest full-domain source
and `ParDir/1l_4beam_BB.par` instead describe a different, smaller CH shell with an Au
cone and four 3ω uniform beams.  `RUN.sh` selects the corresponding no-laser test deck.

A stale `Simulation_initBlock4beams.F90` contains a 1.0 mm outer radius and 0.2 mm
combined shell, but divides it into CH, DT ice, and DT gas and refers to undeclared
variables.  It cannot be paired coherently with the included `Config` and
`Simulation_data.F90`.

The exact FLASH multi-material EOS and opacity mixing implementations are also absent.
`ParDir/Eos.F90` proves only the composition reductions

```text
1/Abar = sum_s X_s/A_s
Zbar   = Abar sum_s X_s Z_s/A_s.
```

Consequently, AthenaK's mixed-cell closure is an explicit documented choice and must not
be described as bit-for-bit FLASH behavior.

## User requirements that override the archive

The AthenaK case now follows the archive for target shape, domain, materials, beams, and
pulse.  Only these user requirements remain as overrides:

- 1.1 g/cm3 initial CH density (the archive agrees: `sim_rhoFoam = 1.1`);
- evolution through 10 ns;
- uniform mesh using 60--80 percent of every available V100 GPU.

Two earlier overrides have been withdrawn.  The first version used a requested
single-beam drive (axial 1ω Gaussian, square 10 kJ/5 ns pulse); the laser now follows
`ParDir/1l_4beam_BB.par` directly.  The first version also used a requested 1.0 mm CH
shell of 0.2 mm thickness; the target is now the archive's own 0.52--0.55 mm cap plus its
Au cone, so the archived beam coordinates apply without rescaling.

That change is self-validating: sampling all four 0.275 mm apertures, every ray lands on
the CH cap, and the outermost strikes it at 49.89 degrees against the 50-degree cap edge.
The archived beams were evidently sized for exactly this target, a fit that the earlier
1.0 mm shell obscured (rays then hit at only ~34 degrees, lighting an annulus).

## Reference choices carried into AthenaK

The newest laser/Biermann reference is `3d_zb/ParDir/1l_4beam_BB.par` together with
`3d_zb/Simulation_initBlock.F90` and `Config`.  It supplies the following choices:

- the archived `angle=100` geometry means a 50-degree polar **half-angle** (100-degree
  full cone), not a 25-degree half-angle;
- the ambient material is helium, not DT;
- initial ion, electron, and radiation temperatures are 11,606 K (approximately 1 eV);
- 3T MHD, electron-ion heat exchange, 20-group radiation, vacuum radiation boundaries,
  Biermann battery, and zero initial magnetic field are enabled;
- the tabular-material Biermann coefficient is
  `1.0364e-4/[0.1 sqrt(4 pi 1.1)] = 2.7875722321043606e-4`; material electron density is
  `rho sum_s(Y_s Z_s/A_s)`, so the legacy constant-heat-capacity mean-weight factor is
  not applied;
- Hall and explicit resistivity are disabled;
- all full-domain hydro boundaries are outflow;
- the run ends at 10 ns;
- CH, gold, and helium use IONMIX material tables.

The archived Au cone is now included: `MaterialMixture` accepts any positive
`nmaterials`, and the deck explicitly advects all three mass fractions
(`rho*Y_CH`, `rho*Y_Au`, and `rho*Y_He`).  Electron conduction
is a reference feature (SpitzerHighZ with Larsen limiter 0.06) but is not an explicit
first-version acceptance requirement; any omission must remain documented until a
material-aware conduction model is implemented and tested.

## Material sources

| Role | Material | Abar | Zbar | Initial density | EOS source | Opacity source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| shell | equimolar CH | 6.5 | 3.5 | 1.1 g/cm3 | `C16H1620gPROP.cn4` | `feos_snop_CH_20g.cn4` |
| cone | Au | 196.96655 | 79 | 19.2 g/cm3 | `feos_snop_Au.cn4` | `feos_snop_Au_20g.cn4` |
| ambient | He | 4.002602 | 2 | 1.0e-5 g/cm3 | `He_20G_yr23.cn4` | `He20g.cn4` |

The Au sources are exactly the files the archived deck names (`eos_coneTableFile` and
`op_coneFileName`).  Gold's Zeff is taken as its Zbar, and the laser reads the local
tabular ionization rather than the deck constant, so inverse-bremsstrahlung absorption in
the cone follows the Au table instead of a CH surrogate.

Three conservative passive scalars explicitly store `rho*Y_CH`, `rho*Y_Au`, and
`rho*Y_He`.  All nonnegative fractions are normalized before use; no material is an
implicit remainder. Mixed opacity uses partial material densities and additive extinction,

```text
kappa_mix(rho, Te, Y) = sum_s Y_s kappa_s(rho*Y_s, Te).
```

This recovers both pure-material limits and makes the mixing choice testable.
Geometric interpolation is used in density and temperature.  Smoothed or numerically
mixed cells can produce partial densities below a pure-material table.  Opacity coordinates
are clamped to the nearest endpoint.  EOS energy and ionization use the minimum-density
surface, while trace-material pressure is scaled linearly from that surface to zero as the
partial density vanishes; this is an explicit part of the AthenaK closure.

The DCI inputs use the opt-in `flash-extrapolate` EOS bounds policy.  In-range IONMIX
values are unchanged; positive pressure and caloric surfaces above the final temperature
node follow the final log-log slope, with a continuous `T^1` fallback for a flat or
decreasing endpoint, and ionization holds its endpoint value.  Forward and inverse mixed
closures expand above the native maximum for every present component.  This mirrors
FLASH's broader EOS inversion bracket and prevents a high conserved energy from being
paired with an endpoint-clamped temperature; `clamp` and `error` retain their prior
behavior for other decks.

`generate_reference_tables.py` verifies the archive hash and converts the separate
CH/Au/He ion-electron EOS surfaces plus all three opacity payloads into ignored local
AthenaK tables.
Its manifest records hashes of the archive members and generated files.

## Radiation groups

The 20 reference photon boundaries are, in eV:

```text
1, 10, 50, 100, 150, 250, 400, 550, 700, 900,
1200, 1600, 2000, 2200, 2400, 2600, 2800, 3100, 3500, 4000, 10000.
```

The reference uses a harmonic flux limiter with coefficient one and vacuum conditions on
all radiation faces.  The opacity tables provide Rosseland transport and Planck
absorption/emission coefficients in cm2/g; AthenaK does not inflate opacity.

FLASH keeps physical `c`.  Its MGD equation (FLASH 4.8 guide, equations 25.3--25.7)
evaluates opacity, emission, and flux-limiter coefficients at time level `n` and solves
each group's diffusion operator backward implicitly for `u_g^(n+1)`.  The general
diffusion solver's `theta=1` scheme is backward Euler (guide equation 19.2), and the
archived decks set `dt_diff_factor=1.0e100` with the comment "Disable diffusion dt".
Thus FLASH removes the diffusion stability restriction rather than replacing physical
light speed.

AthenaK maps the time integration to `transport_integrator=implicit`: it evaluates a
harmonic-limited FLD coefficient from the old-state centered cell gradient.  Resolved
gradients retain their physical coefficient; only roundoff-flat limited cells receive
the grid-scale `D <= alpha*dx_min/2` regularization.  AthenaK exchanges the frozen
coefficient halo before PCG so an MPI-shared face remains symmetric even after the
preceding implicit-conduction solve changed only interior temperatures.  It then
arithmetic-averages the coefficient to faces and advances the centered finite-volume
operator with backward Euler while retaining `c_light=299.792458 mm/ns` in both
transport and matter coupling.  A vacuum face additionally caps
`D_face <= alpha*dx_normal/2`; the operator, Jacobi diagonal, and `rad_Pesc` diagnostic
all use the identical capped value.  This is not the explicit AP/upwind face
discretization.  The deck's
`transport_discretization=asymptotic-preserving` and `ap_*` thresholds are retained only
as explicit-mode fallback controls and are inactive in the implicit solve.  All six
implicit boundaries are pinned to `vacuum`, matching the DCI zero-radiation ghost
treatment instead of the solver's zero-gradient default.
The archive also writes `rt_dtFactor=0.02`, but its
`rt_computeDt` switch is absent and defaults false.  That FLASH control does not map
directly to AthenaK's source-splitting guard: AthenaK time-lags its local emission source
and does not perform FLASH's fully coupled nonlinear source solve.  DCI therefore keeps
`source_cfl=0.1` as an independent source-accuracy limit.  It constrains source changes
only; the implicit transport operator remains free of the explicit `c*dt/dx` restriction.
The guard itself still scales with physical `c`: at initialization it lowers the DCI
candidate timestep from `1.068285e-6 ns` with the guard disabled to
`7.88743694e-7 ns`, approximately 26 percent.  Lower `--radiation-c-light` values are
non-production sensitivity diagnostics, not the mechanism used to relax the transport
timestep.

The correspondence to FLASH is limited to physical `c`, time-lagged coefficients, and
backward-Euler transport; it is not a claim of identical spatial discretization or a
fully coupled nonlinear radiation/material solve.  AthenaK's current implicit solver is
uniform-grid only and rejects SMR/AMR, advances groups sequentially with frozen
coefficients, and uses Jacobi-preconditioned CG at the DCI-pinned tolerance `1e-10` and
2000-iteration ceiling.  Recursive convergence is verified with the true `b-Ax`
residual.  The incoming conserved radiation state must be finite and nonnegative.  Only
finite tolerance-scale negative solver roundoff may undergo a globally
volume-conservative positive rescaling, followed by another true-residual check; larger
negativity and non-finite states abort.  Matter coupling remains a separate time-lagged
local source under `source_cfl=0.1`.

Focused dense-reference tests now cover constant periodic, harmonic-limited periodic,
and harmonic-limited vacuum matrices.  Ten focused CPU tests, the CUDA/MPI build, and
compact initialization validation passed.  The exact current-tree seven-GPU 50-cycle
phase-1 smoke passed in 27.36 s to `t=6.542658936097e-05 ns`, with `eos_bad=0` and
domain-integrated CH/Au/He fractions
`0.07224481712971552/0.9276974187345620/5.776413572244726e-05`, whose sum is one within
roundoff.  This is roughly 382 times short of the historical `0.025 ns` first-failure
time and does not prove long-time removal of the DCI symptom.  Production-scale
20-group Jacobi-PCG memory, convergence, and performance are also unproven; multigrid or
a stronger preconditioner remains future work.  The DCI production gate must separately
supply long-horizon, restart, energy-budget, and full-scale evidence for the exact hashed
deck and binary.

## Laser drive adopted from the archive

The newest archived laser deck, `ParDir/1l_4beam_BB.par`, is now adopted directly.  It
defines four uniform beams at 0.351 micrometres (3ω) inclined 50 degrees from the cone
axis at the four diagonal azimuths, with equal 0.275 mm lens/target radii (collimated)
and one shared 13-section picket pulse that integrates to 1774.55 J per beam, 7.0982 kJ
total.  The AthenaK decks map the archive's cone axis `+z` to `+x1` (`x -> x2`,
`y -> x3`) and convert cm to the 0.1 cm code length unit:

```text
ed_lens (+/-0.14142, +/-0.14142, 0.2037) cm -> lens (2.037, +/-1.4142, +/-1.4142)
ed_target (0, 0, 0.0359) cm               -> target (0.359, 0, 0)
ed_lens/targetSemiAxis 0.0275 cm          -> aperture_radius = target_radius = 0.275
ed_wavelength 0.351 um                    -> wavelength 3.51e-5 cm (unit_system=cgs)
ed_time/ed_power pulse 1 (s, W)           -> pulse0 knots (code ns, erg/s)
```

The pulse is written inline through the laser module's FLASH-style shared tables
(`npulses`, `pulse0_nsections`, `pulse0_time_S`/`pulse0_power_S`) and each beam
references it with `beamN_pulse_number`, which is an absolute power table exactly like
`ed_pulseNumber`.  Explicit modeling choices in this mapping:

- The archived `ed_numberOfRays = 50000` is used as each beam's deterministic
  Fibonacci-aperture ray count.  FLASH's `delta2D` grid at the archived 1 um spacing
  would instead produce about 237,000 grid rays per beam; the parameter file's stated
  ray count is honored rather than the implied delta-grid count.
- The beams and the target now come from the same archived deck, so no aim rescaling is
  needed.  Sampling all four apertures, every ray reaches the CH cap and the outermost
  lands at 49.89 degrees against the 50-degree cap edge, confirming the archived spot
  size was chosen for this shell.
- Beam `start_time`/`end_time` remain `0`/`5 ns` as an independent gate; the pulse
  itself is zero outside `[0, 4.71] ns`.

Incident, deposited, and escaped-radiation power are recorded for energy closure.
Initial compact and production-layout sweeps (performed with the earlier single-beam
drive) establish a hard reflection cap of 64.  An evolved 100-cycle doubled-mesh sweep
additionally brackets the one-percent underdense rearm band, the `1e-5`-cell normal
offset, and caps 64/128; it retains zero terminal power with at most 19 observed turns
per ray.  Those chatter-control selections carry over unchanged.  Wave-cap and
reflection-cap remainders are reported separately, are fatal above `1e-10` of launched
power, and are checked in every baseline, resolution, light-speed, and calibration
laser record by the production gate.

## Exchange and numerical controls

FLASH requests Spitzer electron-ion exchange; no constant exchange time appears in the
archive.  AthenaK therefore uses the state- and mixture-dependent Spitzer option.
Dual energy is enabled because the user explicitly requires it even though the archived
runtime default for FLASH's auxiliary-energy equation is unavailable.

The archived Biermann implementation suppresses source generation in shocks and very
low density.  AthenaK applies its own tested shock suppression and evaluates local
electron density and heat capacity from the material mixture.
