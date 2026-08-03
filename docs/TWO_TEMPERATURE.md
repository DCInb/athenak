# Ion/electron two-temperature model

AthenaK can evolve separate ion and electron internal energies for standalone Newtonian
ideal-gas hydrodynamics and MHD.  The implementation follows the redundant-energy design
and operator-split heat exchange used by FLASH multitemperature hydrodynamics:

\[
E_{\rm tot}=\rho e_i+\rho e_e+\frac{1}{2}\rho |\boldsymbol v|^2
             +\frac{1}{2}|\boldsymbol B|^2,
\]

while the component equations are

\[
\frac{\partial (\rho e_s)}{\partial t}
+\boldsymbol\nabla\!\cdot(\rho e_s\boldsymbol v)
+p_s\boldsymbol\nabla\!\cdot\boldsymbol v=Q_s,
\qquad s\in\{i,e\},
\]

with \(Q_i=-Q_e\).  Both components currently use the fluid's common ideal-gas index,
so \(p_s=(\gamma-1)\rho e_s\) and \(p_i+p_e=p\).  Consequently, AthenaK's existing
Riemann solvers still evolve the correct total pressure and conservative total energy.

## Hydrodynamic update

The component energies are appended after user passive scalars.  Their conservative
values \(\rho e_i\) and \(\rho e_e\) are advected with fluxes
\(F_s=F_\rho e_s\).  After every Runge--Kutta stage, the internal energy obtained from
the conservative total-energy equation is reconciled with the two advected energies.
The residual work plus shock heating is divided in proportion to partial pressure.  This
is FLASH's default RAGE-like multitemperature method.  With the common gamma used here,
the correction is equivalently

\[
(\rho e_s)^{n+1}=(\rho e_{\rm int})^{n+1}
 \frac{(\rho e_s)^{\rm adv}}
      {(\rho e_i)^{\rm adv}+(\rho e_e)^{\rm adv}}.
\]

This preserves the conservative AthenaK total energy exactly and enforces
\(\rho e_i+\rho e_e=\rho e_{\rm int}\) after every stage.

## MHD dual energy

For two-temperature MHD, `dual_energy` is enabled by default.  The already evolved sum
\(\rho e_i+\rho e_e\) acts as the auxiliary internal-energy equation, avoiding a third
redundant material-energy field.  This protects gas pressure when

\[
\rho e_{\rm int}=E_{\rm tot}-\frac{|\boldsymbol m|^2}{2\rho}
                 -\frac{|\boldsymbol B|^2}{2}
\]

is an ill-conditioned subtraction in a high-Mach or magnetically dominated cell.
Following AthenaK's Newtonian dual-energy design, the update has four parts:

1. The Riemann-solver mass flux advects both component specific energies.
2. The upwind face velocity supplies the common-gamma compression update
   \(e_s\leftarrow e_s\exp[-(\gamma-1)\Delta t\,\nabla\!\cdot\boldsymbol v]\).
3. The component sum is refreshed from conservative total energy only when
   \(e_{\rm int,cons}>\eta_2\max_{\rm local}E_{\rm tot}\), retaining conservative shock
   heating wherever that subtraction is accurate.
4. Conserved-to-primitive conversion uses total-energy-derived internal energy only when
   \(e_{\rm int,cons}>\eta_1 E_{\rm tot}\); otherwise it obtains pressure from the
   independently evolved component sum.

The conservative total-energy field is not replaced when the auxiliary branch is used;
it is changed only if an EOS floor is required.  Static/adaptive mesh interfaces flux
correct the face velocity along with the component-energy fluxes.

### Hydrodynamic dual energy

The same formalism is available in a `<hydro>` block, with the magnetic terms simply
absent: the conservative internal energy is \(E_{\rm tot}-|\boldsymbol m|^2/2\rho\), and
there is no Biermann electron work.  Because that subtraction is usually well conditioned
without magnetic energy, `<hydro>/dual_energy` defaults to `false`; laser-driven or
otherwise strongly kinetic-energy-dominated problems should set it to `true` explicitly.
Everything else — the tabulated `<materials>` closure, multigroup radiation, the material
LLF flux, and the laser deposition module — is shared between the two carriers, so a 3T
deck can be moved between `<mhd>` and `<hydro>` by renaming the block.

## Biermann battery

Two-temperature MHD can optionally add the FLASH-style flux formulation of the Biermann
battery.  It uses electron pressure in the induction equation and transports the full
electron enthalpy with the electron velocity, allowing electron and magnetic energy to
exchange while retaining conservative total energy.  The MHD dual-energy protection
remains active.  See `BIERMANN_BATTERY.md` for equations, stability limits, shock
suppression, and input parameters.

## Ion/electron exchange

Heat exchange is operator split and integrated exactly once per complete time step.  The
constant-\(t_{ei}\) equations match the FLASH Spitzer update once its locally calculated
equilibration time is held fixed over a step:

\[
\frac{d e_i}{dt}=\frac{c_{v,e}}{t_{ei}}(T_e-T_i),\qquad
\frac{d e_e}{dt}=\frac{c_{v,e}}{t_{ei}}(T_i-T_e).
\]

Writing \(m=c_{v,e}/c_{v,i}\), their exact temperature-difference update is

\[
(T_i-T_e)^{n+1}=(T_i-T_e)^n
\exp\!\left[-(1+m)\frac{\Delta t}{t_{ei}}\right].
\]

The code updates component energies, not temperatures, and assigns one component as the
remainder of the fixed total internal energy.  Exchange is therefore conservative to
roundoff and remains positive and stable for \(\Delta t\gg t_{ei}\).

## Implicit electron thermal conduction

Two-temperature Hydro and MHD can operator-split isotropic electron conduction once per
complete time step:

\[
\rho c_{v,e}\frac{T_e^{n+1}-T_e^n}{\Delta t}
=\theta\,\boldsymbol\nabla\!\cdot(K_e^n\boldsymbol\nabla T_e^{n+1})
 +(1-\theta)\,\boldsymbol\nabla\!\cdot(K_e^n\boldsymbol\nabla T_e^n).
\]

The default \(\theta=1\) is backward Euler.  Conductivity and the optional saturated-flux
limiter are frozen at the old state, following FLASH's implicit diffusion linearization.
For a tabular material mixture the electron EOS energy is evaluated nonlinearly and its
finite-difference heat capacity forms Newton systems.  A matrix-free, diagonally
preconditioned conjugate-gradient solve spans all same-level MeshBlocks and MPI ranks.
The converged flux divergence is added identically to electron and conservative total
energy; ion energy is unchanged.  Periodic or block faces use the usual halo exchange,
while physical faces independently select zero-gradient or fixed-temperature conduction
boundaries.

Constant diffusivity uses `alpha_iso`, with \(K_e=\rho\,\mathtt{alpha_iso}\).  Setting
`alpha_spitzer=true` adds the FLASH `SpitzerHighZ` conductivity in physical units.  Its
classical or saturated value can be selected with `none`, `harmonic`, `minmax`, or
`larsen` limiting.  A minimal configuration is:

```text
two_temperature = true
alpha_spitzer = true
conduction_integrator = implicit
conduction_theta = 1.0
conduction_coulomb_log = 10.0
conduction_spitzer_multiplier = 1.0
conduction_flux_limiter = harmonic
conduction_flux_limit_coefficient = 0.06
conduction_linear_tolerance = 1.0e-10
conduction_nonlinear_tolerance = 1.0e-8
conduction_max_iterations = 400
conduction_max_nonlinear_iterations = 8
```

Each face also accepts `conduction_x{1,2,3}_{inner,outer}_boundary`, set to `neumann`
(the default) or `dirichlet`; a Dirichlet face reads the corresponding `_value` in code
temperature.  `conduction_report=true` prints iteration, residual, and net conductive
energy diagnostics.  The explicit conduction path remains the default when
`conduction_integrator` is omitted.  Implicit conduction is not included in the explicit
parabolic timestep restriction, but it currently requires a uniform-level mesh and
rejects SMR/AMR and anisotropic conductivity.

## Input parameters

Add these parameters to either a `<hydro>` or `<mhd>` block:

```text
two_temperature = true
electron_heat_capacity_fraction = 0.5
initial_electron_temperature_ratio = 0.1
t_ei = 0.2
```

- `electron_heat_capacity_fraction` is
  \(f_e=c_{v,e}/(c_{v,i}+c_{v,e})\), strictly between zero and one.
- `initial_electron_temperature_ratio` sets the global initial \(T_e/T_i\) while
  retaining the total pressure supplied by the problem generator.
- `t_ei > 0` is the constant FLASH-style electron-ion equilibration time in code units.
  `t_ei = 0` gives immediate equilibrium and `t_ei < 0` disables exchange.
- In an `<mhd>` block, `dual_energy` defaults to `true` when `two_temperature=true`.
  In a `<hydro>` block it defaults to `false`.
  `dual_energy_eta1` and `dual_energy_eta2` default to `1.0e-3` and `1.0e-4`.
  Set `dual_energy=false` only when reproducing the conservative-only method.

Ready-to-run examples are provided:

- `inputs/hydro/two_temperature_relax.athinput`
- `inputs/mhd/two_temperature_bw.athinput`
- `inputs/mhd/two_temperature_dual_energy.athinput` (magnetically dominated SMR test)
- `inputs/mhd/two_temperature_biermann.athinput` (crossed-gradient battery test)
- `inputs/tests/implicit_conduction.athinput` (stiff periodic backward-Euler regression)
- `inputs/tests/implicit_conduction_dirichlet.athinput` (fixed-temperature boundary test)

For three-temperature ion/electron/thermal-radiation calculations, including multigroup
flux-limited diffusion, see `THERMAL_RADIATION.md`.

## Output variables

`hydro_2t` and `mhd_2t` output `eion`, `eele`, `tion`, and `tele`.  Each field can also
be selected separately with `hydro_eion`, `hydro_eele`, `hydro_tion`, `hydro_tele`, or
the corresponding `mhd_*` name.  The component energies are specific internal energies;
their conservative energy densities also appear as `eion_d` and `eele_d` in the usual
`hydro_u` or `mhd_u` output groups.

## Current scope

This implementation supports standalone Newtonian gamma-law Hydro and MHD, constant or
locally calculated Spitzer ion/electron exchange, and electron-only implicit conduction.
MHD additionally supports composition-aware tabular ion/electron material mixtures.
Optional thermal multigroup radiation is documented separately, and the optional 2D/3D
Cartesian Biermann battery is documented in `BIERMANN_BATTERY.md`.  Relativistic fluids
and ion-neutral two-fluid runs are not supported.  The implicit conduction operator is
currently isotropic and single-level; magnetized conduction, Nernst advection, nonlocal
transport, SMR, and AMR remain out of scope.  MHD dual energy is incompatible with FOFC,
viscosity, resistivity, **explicit** conduction, generic MHD source terms, and shearing-box
evolution; the implicit electron-conduction path is compatible.  Thermal multigroup
radiation is supported because its electron coupling updates the same component-energy
reservoir.  The RAGE-like method also partitions shock heating by pressure; unlike FLASH's
entropy-advection alternative, it does not force all irreversible shock heating into ions.

The numerical design follows the FLASH User's Guide sections
[3T capabilities](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node103.html),
[multitemperature hydrodynamics](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node105.html),
and [heat exchange](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node123.html).
The implicit conduction stencil, theta method, flux limiting, and boundary choices follow
the FLASH 4.6.2 [general implicit and thermal-conduction
solver](https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node122.html),
with its [SpitzerHighZ
conductivity](https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node139.html).
The dual-energy switches and face-velocity compression update follow
[AthenaK PR #753](https://github.com/IAS-Astrophysics/athenak/pull/753).
