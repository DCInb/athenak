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

The first AthenaK version retains the explicitly requested:

- all-CH shell with outer radius 1.0 mm and thickness 0.2 mm;
- 1.1 g/cm3 initial CH density;
- Gaussian beam in space;
- square pulse in time;
- 1ω wavelength, taken as 1.053 micrometres;
- 10 kJ total incident energy over 5 ns;
- a converging spot covering most of the projected open shell;
- evolution through 10 ns;
- uniform mesh using 60--80 percent of every one of eight V100 GPUs.

These choices are not claimed to reproduce the archived 3ω, uniform, four-beam pulse.

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
- CH and helium use IONMIX material tables.

The archived Au cone is omitted because the requested target is an open spherical CH
shell and one conservative scalar is reserved for CH versus helium.  Electron conduction
is a reference feature (SpitzerHighZ with Larsen limiter 0.06) but is not an explicit
first-version acceptance requirement; any omission must remain documented until a
material-aware conduction model is implemented and tested.

## Material sources

| Role | Material | Abar | Zbar | Initial density | EOS source | Opacity source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| shell | equimolar CH | 6.5 | 3.5 | 1.1 g/cm3 | `C16H1620gPROP.cn4` | `feos_snop_CH_20g.cn4` |
| ambient | He | 4.002602 | 2 | 1.0e-5 g/cm3 | `He_20G_yr23.cn4` | `He20g.cn4` |

The conservative passive scalar stores `rho*Y_CH`; `Y_He=1-Y_CH`.  Mixed opacity uses
partial material densities and additive extinction,

```text
kappa_mix(rho, Te, Y) = sum_s Y_s kappa_s(rho*Y_s, Te).
```

This recovers both pure-material limits and makes the mixing choice testable.
Geometric interpolation is used in density and temperature.  Smoothed or numerically
mixed cells can produce partial densities below a pure-material table.  Opacity coordinates
are clamped to the nearest endpoint.  EOS energy and ionization use the minimum-density
surface, while trace-material pressure is scaled linearly from that surface to zero as the
partial density vanishes; this is an explicit part of the AthenaK closure.

`generate_reference_tables.py` verifies the archive hash and converts the separate CH/He
ion-electron EOS surfaces plus both opacity payloads into ignored local AthenaK tables.
Its manifest records hashes of the archive members and generated files.

## Radiation groups

The 20 reference photon boundaries are, in eV:

```text
1, 10, 50, 100, 150, 250, 400, 550, 700, 900,
1200, 1600, 2000, 2200, 2400, 2600, 2800, 3100, 3500, 4000, 10000.
```

The reference uses a harmonic flux limiter with coefficient one and vacuum conditions on
all radiation faces.  The opacity tables provide Rosseland transport and Planck
absorption/emission coefficients in cm2/g.  AthenaK uses its explicit
asymptotic-preserving face flux and does not inflate opacity.  The production candidate
uses the documented reduced value `c_hat=1.0e9 cm/s` (about `c/30`) so a 10 ns run is
feasible; a `c_hat=10` versus `30` compact sensitivity comparison, plus a short physical-c
check where practical, is a mandatory production gate.

## Laser comparison

The newest archived laser deck has four 50,000-ray uniform beams at 0.351 micrometres.
They are inclined approximately 50 degrees from `-z`, have equal 0.275 mm lens/target
radii (collimated), and use a shaped pulse inferred to integrate to 7.0982 kJ total.
Those settings demonstrate the archive's ray and symmetry conventions but conflict with
the requested 1ω Gaussian converging 10 kJ square drive.  The AthenaK laser therefore
uses one axial 4,096-ray beam with a 0.72 mm Gaussian aperture focused to 0.58 mm at the
inner cap.  The target spot covers 89.5 percent of the cap's projected area.  Incident,
deposited, and escaped-radiation power are recorded for energy closure.

## Exchange and numerical controls

FLASH requests Spitzer electron-ion exchange; no constant exchange time appears in the
archive.  AthenaK therefore uses the state- and mixture-dependent Spitzer option.
Dual energy is enabled because the user explicitly requires it even though the archived
runtime default for FLASH's auxiliary-energy equation is unavailable.

The archived Biermann implementation suppresses source generation in shocks and very
low density.  AthenaK applies its own tested shock suppression and evaluates local
electron density and heat capacity from the material mixture.
