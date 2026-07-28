# Tabulated Equation Of State

AthenaK can use a single-material density-temperature EOS table for Newtonian Hydro or
MHD. The table supplies pressure, specific internal energy, and adiabatic sound speed
squared, and may also supply thermodynamic derivatives and material properties. Runtime
queries use bilinear interpolation in log density and log temperature;
conserved-to-primitive conversion inverts the monotonic internal-energy rows on device.

## Input

Select the table closure in the fluid block:

```text
<hydro>
eos = table
table_file = inputs/hydro/gamma_law_eos_table.dat
table_unit_system = code
table_bounds = clamp
reconstruct = plm
rsolver = llf
```

`eos=tabulated` is an alias. The same parameters work under `<mhd>`, and `table` is an
alias for `table_file`. `table_bounds` may be `clamp` (the default) or `error`.

`table_unit_system=code` interprets every value in code units. With `cgs`, density is in
g/cm3, temperature in K, pressure in erg/cm3, and specific energy and sound-speed squared
in erg/g; a `<units>` block is required. The automatically derived conversions can be
overridden with positive `table_density_scale`, `table_temperature_scale`,
`table_pressure_scale`, `table_specific_eint_scale`, and `table_sound_speed2_scale`.
Each scale multiplies the corresponding stored value to obtain code units.

## Supported configurations

The portable table closure currently supports single-level Newtonian Hydro and MHD with
the LLF Riemann solver. The following combinations fail during initialization rather
than silently using a gamma-law closure:

- special, general, or dynamical relativity;
- SMR or AMR;
- Riemann solvers other than `llf`;
- a meaningful positive `sfloor`, because these tables contain no entropy field;
- `ism_cooling` or `rel_cooling` in `<hydro_srcterms>` or `<mhd_srcterms>`;
- gamma-law thermal conduction;
- `<ion-neutral>` or `<radiation>` fluid blocks;
- two-temperature evolution and its dependent dual-energy and laser models.

Laser deposition therefore remains a gamma-law, two-temperature MHD feature. Opacity
tables likewise remain independent material data for multigroup two-temperature
radiation; selecting a table EOS does not select opacity data or make the two table
types interchangeable.

## File formats

AthenaK detects the native format when the first non-comment token is
`athenak_eos_table`. Otherwise it invokes the existing `TableReader` binary reader.

### Native ASCII

The native format is ASCII and comment-aware. Version 1 contains the three required
thermodynamic fields:

```text
athenak_eos_table 1
dimensions 2 3
density 1.0 2.0
temperature 0.5 1.0 2.0

pressure
# density row 0, then density row 1; temperature varies fastest
0.5 1.0 2.0
1.0 2.0 4.0

specific_internal_energy
1.25 2.5 5.0
1.25 2.5 5.0

sound_speed_squared
0.7 1.4 2.8
0.7 1.4 2.8
end
```

In either version, both axes and all three required field values must be finite and
positive. The density and temperature axes must be strictly increasing but may be
nonuniform. At each density, pressure and specific internal energy must increase with
temperature so inverse queries are unique.

Version 2 appends a counted list of optional material fields after the three required
fields:

```text
athenak_eos_table 2
dimensions 2 3
density 1.0 2.0
temperature 0.5 1.0 2.0

pressure
0.5 1.0 2.0
1.0 2.0 4.0

specific_internal_energy
1.25 2.5 5.0
1.25 2.5 5.0

sound_speed_squared
0.7 1.4 2.8
0.7 1.4 2.8

material_fields 2
gamma1
1.4 1.4 1.4
1.4 1.4 1.4
zbar
0.0 0.5 1.0
0.0 0.5 1.0
end
```

`material_fields` may contain from zero through six fields. Fields may appear in any
order, but each may appear at most once. Every field array uses the same density-major,
temperature-fastest ordering as the required arrays. Required thermodynamic fields are
interpolated logarithmically; optional fields are stored and interpolated as linear
values using the same log-axis interpolation weights. Optional values are not affected
by any `table_*_scale` setting.

The supported optional fields and their returned names are shown below. `gamma3m1`
represents `Gamma_3 - 1`; `zbar` and `zeff` are mean ionization and effective charge,
respectively; and `abar` and `mu` are mean atomic mass and mean molecular weight.

| File field | `ThermoState` value | Availability flag | Valid values |
| --- | --- | --- | --- |
| `gamma1` | `gamma1` | `has_gamma1` | finite and greater than zero |
| `gamma3m1` | `gamma3_minus_one` | `has_gamma3_minus_one` | finite and greater than zero |
| `zbar` | `mean_ionization` | `has_mean_ionization` | finite and non-negative |
| `zeff` | `effective_charge` | `has_effective_charge` | finite and non-negative |
| `abar` | `mean_atomic_mass` | `has_mean_atomic_mass` | finite and greater than zero |
| `mu` | `mean_molecular_weight` | `has_mean_molecular_weight` | finite and greater than zero |

### TableReader binary

The existing AthenaK `TableReader` format is also accepted. Its text header must declare
the two axes in this order and include the three required fields. A header containing
all recognized fields is:

```text
<metadatabegin>
endianness = little
log_axis_base = e
<metadataend>
<scalarsbegin>
<scalarsend>
<pointsbegin>
logrho = 2
logtemp = 3
<pointsend>
<fieldsbegin>
logpress
logeps
logcs2
gamma1
gamma3m1
zbar
zeff
abar
mu
<fieldsend>
```

The header is followed immediately by raw IEEE double arrays in the declared
endianness: the `logrho` axis, the `logtemp` axis, and then every field in header order.
Field arrays are density-major with temperature varying fastest. `logrho`, `logtemp`,
`logpress`, `logeps`, and `logcs2` contain natural logarithms. The six recognized
optional arrays use the same names as native v2 (`gamma1`, `gamma3m1`, `zbar`, `zeff`,
`abar`, and `mu`) and contain direct linear values subject to the validation rules
above. Any subset is accepted. `log_axis_base` may be omitted; when present, it must be
`e`. `endianness` may be omitted for legacy native-endian tables; when present, it must
be `little` or `big`. Other fields and scalar metadata are ignored by this closure.

Both axes need at least two finite, strictly increasing points. `logpress`, `logeps`, and
`logcs2` must be finite, and pressure and specific internal energy must increase with
temperature at each density. Unit conversions and explicit `table_*_scale` values apply
to both formats; for a binary table AthenaK applies the corresponding additive shift in
log space.

## Runtime query API

`EOS_Data` exposes device-callable forward and inverse queries for all closures. For a
table-backed EOS, the forward calls are `PressureFromRhoTemperature()` and
`SpecificEintFromRhoTemperature()`. `TemperatureFromRhoPressure()` performs the pressure
inverse; the existing `TemperatureFromRhoEint()` uses an internal-energy-density
argument, not a specific internal energy. Bounds follow the configured `table_bounds`
policy for both forward and inverse queries.

The following calls return a device-copyable `EOS_Data::ThermoState`:

- `EvalThermoStateFromRhoTemperature(density, temperature)`;
- `EvalThermoStateFromRhoEint(density, internal_energy_density)`;
- `EvalThermoStateFromRhoPressure(density, pressure)`.

Every table result contains `temperature`, `pressure`, `specific_internal_energy`, and
`sound_speed_squared`. It also contains the six optional values and their corresponding
`has_*` flags listed above. When an optional field is absent from a native v1, native
v2, or TableReader table, its availability flag is `false` and its value is quiet NaN;
AthenaK does not synthesize a material-dependent default. Callers must test the flag
before using the value. With `table_bounds=clamp`, the temperature returned in the state
is the bounded table temperature actually used for interpolation.

## MPI loading

In an MPI build, rank 0 alone opens, parses, and validates the EOS file. It broadcasts
the dimensions, axes, thermodynamic and material fields, and availability metadata to
the other ranks, after which each rank initializes its host and device table views. The
file therefore needs to be accessible to rank 0; it is not read independently by every
process. Oversized table payloads that cannot be represented by an MPI `int` count are
rejected before the broadcast. Invalid or overflowing dimensions and truncated binary
payloads are rejected before table storage is allocated or used.

`inputs/hydro/tabulated_eos.athinput` and `inputs/mhd/tabulated_eos.athinput` are runnable
examples. Their supplied table exactly represents a gamma=1.4 gas, which makes it useful
for comparing the table and analytic closures without changing the expected shock-tube
solutions.

Opacity-table format and scale controls are documented in `docs/THERMAL_RADIATION.md`.
