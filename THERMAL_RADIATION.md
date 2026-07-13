# Multigroup thermal radiation for the two-temperature model

AthenaK can evolve a frequency-dependent thermal radiation field together with the
Newtonian ideal-gas ion/electron model described in `TWO_TEMPERATURE.md`.  This creates a
three-temperature system in the FLASH sense: the ion, electron, and radiation
temperatures are independent, while the radiation spectrum is represented by any number
of energy groups.

## Equations

Group `g` stores the comoving radiation energy density `E_g`.  The transport and local
source update are

\[
\frac{\partial E_g}{\partial t}+\boldsymbol\nabla\!\cdot(E_g\boldsymbol v)
=\boldsymbol\nabla\!\cdot(c_*D_g\boldsymbol\nabla E_g)
+c_*\left(\sigma_{e,g}B_g(T_e)-\sigma_{a,g}E_g\right),
\]

\[
\frac{\partial (\rho e_e)}{\partial t}
=-c_*\sum_g\left(\sigma_{e,g}B_g(T_e)-\sigma_{a,g}E_g\right).
\]

Here `c_*` is `c_light`, which may be the physical or a reduced speed of light,
`sigma=rho*kappa`, and

\[
B_g(T_e)=a_rT_e^4\frac{15}{\pi^4}
\left[P\!\left(\frac{\epsilon_{g+1}}{T_e}\right)
-P\!\left(\frac{\epsilon_g}{T_e}\right)\right],\qquad
P(x)=\int_0^x\frac{t^3}{e^t-1}\,dt.
\]

The group boundaries `epsilon_g=h*nu_g/k_B` are therefore entered in the same code
temperature units as `tele`.  The Planck integral uses a cancellation-safe small-argument
series and an exponentially convergent complementary series.

The FLD flux is

\[
\boldsymbol F_g=-c_*D_g\boldsymbol\nabla E_g,
\qquad D_g=\frac{\lambda(R_g)}{\sigma_{t,g}},\qquad
R_g=\frac{|\boldsymbol\nabla E_g|}{\sigma_{t,g}E_g}.
\]

Available limiter choices are:

- `none`: `lambda=1/3`;
- `harmonic`: FLASH's harmonic limiter;
- `larsen`: FLASH's Larsen limiter;
- `minmax`: FLASH's min/max limiter;
- `levermore-pomraning` (default):
  `lambda=(2+R)/(6+3R+R^2)`.

`flux_limit_coefficient` rescales `R` so that the streaming limit is
`|F_g| <= alpha*c_*E_g`; its physical default is one.

## Numerical update

Radiation groups are appended after the ion/electron energies in the fluid scalar arrays.
Consequently, their advection, boundary communication, AMR restriction/prolongation,
flux correction, and restart storage use AthenaK's existing conservative infrastructure.

FLD fluxes are added to the Hydro or MHD finite-volume fluxes at every Runge--Kutta stage.
AthenaK computes the explicit multidimensional diffusion stability bound

\[
\Delta t_g\leq
\frac{1}{2c_*D_g(\Delta x_1^{-2}+\Delta x_2^{-2}+\Delta x_3^{-2})}
\]

and includes it in the normal CFL timestep.  This differs from FLASH, which time-lags the
coefficients but solves each group diffusion equation implicitly.  The physical FLD
closure and group emission/absorption terms follow FLASH; the current AthenaK transport
update is explicit and does not require HYPRE.

After a complete fluid step, the source update uses FLASH's time-lagged electron
temperature and implicit absorption:

\[
E_g^{n+1}=\frac{E_g^n+\Delta t\,c_*\sigma_{e,g}B_g(T_e^n)}
{1+\Delta t\,c_*\sigma_{a,g}}.
\]

The summed radiation change is removed from the electron energy and the conservative
material total energy.  Emission is limited only when needed to prevent negative electron
energy.  Thus every group remains non-negative and
`rho*e_i + rho*e_e + sum(E_g)` is conserved to roundoff by local source coupling.

## Input parameters

Enable `two_temperature` in either `<hydro>` or `<mhd>`, then add:

```text
<thermal_radiation>
enabled = true
n_groups = 3
arad = 0.1
c_light = 1.0
initial_radiation_temperature = 0.2
flux_limiter = levermore-pomraning
flux_limit_coefficient = 1.0
source_cfl = 0.1
couple_matter = true
opacity_model = constant

group_bound_0 = 0.0
group_bound_1 = 0.5
group_bound_2 = 2.0
group_bound_3 = 100.0

kappa_transport_0 = 100.0
kappa_transport_1 = 100.0
kappa_transport_2 = 100.0
kappa_absorption_0 = 0.5
kappa_absorption_1 = 0.5
kappa_absorption_2 = 0.5
```

There must be `n_groups+1` strictly increasing group boundaries.  Each group requires a
positive `kappa_transport_g`.  `kappa_absorption_g` defaults to zero and
`kappa_emission_g` defaults to the corresponding absorption opacity.

### Tabulated opacities

Set `opacity_model=table` to replace the constant group values with separate tabulated
transport (Rosseland), Planck absorption, and Planck emission mass opacities.  Following
FLASH, each opacity is a function of mass density, electron temperature, and group.  The
lookup is bilinear in density and electron temperature; states outside the table are
clamped to its nearest edge.

```text
opacity_model = table
opacity_table_file = inputs/hydro/two_temperature_opacity_table.dat
opacity_interpolation = linear
```

`opacity_interpolation` may be `linear` (the default) or `log`.  Log interpolation is
performed on the opacity values, not the coordinates, and therefore requires every
stored opacity to be strictly positive.  Linear tables may contain zero absorption or
emission opacity.  Transport opacity must always be positive.

The native, comment-friendly table format is:

```text
athenak_opacity_table 1
dimensions 2 2 2              # n_density, n_temperature, n_groups
density 1.0 3.0
temperature 1.0 2.0
group_bound 0.0 1.0 100.0

transport
# group 0: density row 0, density row 1; temperature varies fastest
100.0 400.0
900.0 3600.0
# group 1
200.0 800.0
1800.0 7200.0

absorption
# 2 groups * 2 densities * 2 temperatures values
0.10 0.40 0.90 3.60
0.20 0.80 1.80 7.20

emission
0.15 0.60 1.35 5.40
0.25 1.00 2.25 9.00
end
```

The group count and boundaries in the table must match the radiation input.  Table
coordinates and values are in code units by default.  The following optional multipliers
permit a physical table to be used without rewriting it:

```text
# Multiply code rho and Te before looking up table coordinates.
opacity_density_scale = 1.0
opacity_temperature_scale = 1.0

# Multiply stored bounds before comparing them with group_bound_g.
opacity_group_bound_scale = 1.0

# Convert stored mass opacities back to code units.  Per-kind values override the common
# scale and correspond to FLASH's transport, absorption, and emission scale factors.
opacity_value_scale = 1.0
opacity_transport_scale = 1.0
opacity_absorption_scale = 1.0
opacity_emission_scale = 1.0
```

The supplied example table is intentionally nonlinear and can be run with either linear
or log interpolation.  The format stores only radiation opacity data rather than the
unrelated EOS arrays required by an IONMIX file; IONMIX data can be converted by writing
its density/temperature axes, group boundaries, Rosseland opacity, Planck absorption,
and Planck emission in the order above.

`source_cfl > 0` limits the fractional electron-energy change for accuracy.  The local
source update remains positive and conservative without this limit.  Set
`couple_matter=false` to test pure radiation advection and diffusion.

Initial radiation can be uniform or a one-dimensional step:

```text
initial_profile = step
initial_radiation_temperature = 1.0
initial_radiation_temperature_right = 0.5
initial_radiation_x1 = 0.0
```

The step option is useful for diffusion tests.  Both profiles initialize a Planck spectrum
over the finite configured group interval.

## Outputs and examples

`hydro_3t` or `mhd_3t` writes `eion`, `eele`, `tion`, `tele`, every group as `erad00`,
`erad01`, ..., the summed specific radiation energy `erad`, and
`trad=(rho*erad/arad)^(1/4)`.  Conservative scalar groups in `hydro_u` or `mhd_u` are
labelled `erad00_d`, `erad01_d`, and so on.

Ready-to-run examples are:

- `inputs/hydro/two_temperature_mgfld.athinput`: uniform electron-radiation relaxation;
- `inputs/hydro/mgfld_diffusion.athinput`: periodic FLD smoothing of a radiation step;
- `inputs/hydro/two_temperature_opacity_table.athinput`: tabulated-opacity relaxation;
- `inputs/hydro/mgfld_opacity_table_diffusion.athinput`: tabulated-opacity diffusion;
- `inputs/mhd/two_temperature_mgfld.athinput`: Brio--Wu MHD with two radiation groups.

## Current scope

This implementation supports standalone Newtonian ideal-gas Hydro and MHD, constant or
single-material density/electron-temperature opacity tables, Cartesian finite-volume
diffusion, and up to 100 groups.  It models thermal transport and electron-radiation
energy exchange.  Radiation pressure, momentum feedback, radiation work,
frequency-space Doppler coupling, multispecies opacity mixing, direct IONMIX parsing, and
an implicit global diffusion solver are not yet included.  It is therefore appropriate
when radiation energy transport and thermal coupling matter but radiation forces are
subdominant.

The design follows the FLASH User's Guide sections on
[3T capabilities](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node103.html),
[multigroup diffusion](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node154.html),
[opacity models and interpolation](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node149.html),
and [flux limiters](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node128.html).
