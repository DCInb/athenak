# Tabulated Equation Of State

AthenaK can use a single-material density-temperature EOS table for Newtonian Hydro or
MHD. The table supplies pressure, specific internal energy, and adiabatic sound speed
squared. Runtime queries use bilinear interpolation in log density and log temperature;
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

The native format is ASCII and comment-aware:

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

Both axes and all stored values must be finite and positive. The density and temperature
axes must be strictly increasing but may be nonuniform. At each density, pressure and
specific internal energy must increase with temperature so inverse queries are unique.

### TableReader binary

The existing AthenaK `TableReader` format is also accepted. Its text header must declare
the two axes in this order and include the three required fields:

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
<fieldsend>
```

The header is followed immediately by raw IEEE double arrays in the declared
endianness: the `logrho` axis, the `logtemp` axis, and then every field in header order.
Field arrays are density-major with temperature varying fastest. The required values
are natural logarithms of density, temperature, pressure, specific internal energy, and
sound speed squared. `log_axis_base` may be omitted; when present, it must be `e`.
`endianness` may be omitted for legacy native-endian tables; when present, it must be
`little` or `big`. Additional fields and scalar metadata are ignored by this closure.

Both axes need at least two finite, strictly increasing points. `logpress`, `logeps`, and
`logcs2` must be finite, and pressure and specific internal energy must increase with
temperature at each density. Unit conversions and explicit `table_*_scale` values apply
to both formats; for a binary table AthenaK applies the corresponding additive shift in
log space.

## MPI loading

In an MPI build, rank 0 alone opens, parses, and validates the EOS file. It broadcasts
the dimensions, axes, and three thermodynamic fields to the other ranks, after which
each rank initializes its host and device table views. The file therefore needs to be
accessible to rank 0; it is not read independently by every process. Oversized table
payloads that cannot be represented by an MPI `int` count are rejected before the
broadcast. Invalid or overflowing dimensions and truncated binary payloads are rejected
before table storage is allocated or used.

`inputs/hydro/tabulated_eos.athinput` and `inputs/mhd/tabulated_eos.athinput` are runnable
examples. Their supplied table exactly represents a gamma=1.4 gas, which makes it useful
for comparing the table and analytic closures without changing the expected shock-tube
solutions.

Opacity-table format and scale controls are documented in `docs/THERMAL_RADIATION.md`.
