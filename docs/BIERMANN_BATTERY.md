# Biermann battery for two-temperature MHD

AthenaK's Biermann battery follows the flux-based, three-temperature formulation
documented for FLASH's unsplit staggered-mesh MHD solver. It is available for 2D and 3D
Cartesian, Newtonian ideal-gas MHD when the two-temperature model is enabled.

## Equations

In normalized units with `mu0 = k_B = 1`, the module uses

```text
n_e = f_e rho
E_B = -C_B grad(p_e)/n_e
v_e = v - C_B curl(B)/n_e
```

where `f_e` is `electron_heat_capacity_fraction` and `C_B` is
`biermann_coefficient`. The legacy stage-coupled path constructs the transverse
components of `E_B` on coordinate faces and arithmetically averages them to
constrained-transport edges. The dedicated subcycle instead constructs the
path-conservative edge integral described below. In both paths, the total-energy
Poynting flux is paired with the electric field used by the magnetic update.

When a `<materials>` closure is active, the electron-density normalization is instead

```text
n_e,code = rho * sum_s(Y_s Zbar_s/A_s).
```

For tabular material EOS tables, `Zbar_s` is the local table ionization. The
two-temperature synchronization caches `p_i`, `p_e`, physical `n_e`, and the effective
charge once per cell. Biermann kernels reuse that cache: `grad(p_e)` uses the tabular
electron pressure, shock detection uses `p_i+p_e`, and both the electric field and
electron drift use cached `n_e`. The physical cache is converted without an EOS query by
`n_e,code = n_e,cgs*m_u/rho_0`, where `rho_0` is the code density unit in g cm^-3.

As in FLASH's flux formulation, the electron component follows its full velocity:

```text
d(epsilon_e)/dt + div(epsilon_e v_e) + p_e div(v_e) = 0.
```

The finite-volume electron equation receives the drift flux `epsilon_e (v_e-v)` and a
matching `p_e div(v_e-v)` work update. Total energy receives the electron enthalpy drift
flux `(epsilon_e+p_e)(v_e-v)`. Together with the Biermann Poynting flux, this keeps the
total-energy equation conservative while allowing magnetic and electron energy to
exchange. The existing 2T dual-energy update remains active, including in magnetically
dominated cells.

For a tabular EOS, the upwind total-energy drift flux uses the cached
`epsilon_e+p_e`. On the legacy stage-coupled path, the positive exponential work update
uses the same frozen `p_e`; source coefficients are held at the stage-start thermodynamic
state until the next two-temperature synchronization. The subcycle instead adds
`-p_e div(v_e-v)` to each SSPRK2 stage RHS and refreshes the accepted thermodynamic state
between stages. There are no per-face table inversions on either path.

## Input

```text
<mhd>
eos = ideal
two_temperature = true
dual_energy = true
biermann_battery = true
biermann_coefficient = 1.0
biermann_subcycle = false
biermann_subcycle_cfl = 0.15
biermann_subcycle_max_steps = 100000
biermann_minimum_electron_fraction = 1.0e-12
biermann_shock_suppression = true
biermann_shock_threshold = 0.25
biermann_shock_compression_threshold = 0.02
```

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `biermann_battery` | `false` | Enable the flux-form Biermann and 2T energy terms. |
| `biermann_coefficient` | `1.0` | Normalized inverse-charge coefficient `C_B`; zero disables all terms. |
| `biermann_subcycle` | `false` | Use the second-order Strang/SSPRK2 multirate update. |
| `biermann_subcycle_cfl` | `min(<time>/cfl_number, 0.15)` | CFL factor for Biermann microsteps; must lie in `(0,0.15]`. |
| `biermann_subcycle_max_steps` | `100000` | Maximum microsteps in either Strang half-step. |
| `biermann_minimum_electron_fraction` | `1.0e-12` | Minimum tabular `q_e=n_e/rho` for a resolved plasma. |
| `biermann_shock_suppression` | `true` | On the legacy path, attenuate Biermann faces adjacent to detected compressive pressure jumps. |
| `biermann_shock_threshold` | `0.25` | Legacy-path threshold for the symmetric centered gas-pressure jump. |
| `biermann_shock_compression_threshold` | `0.02` | Legacy-path directional compression Mach number where suppression starts; activation is complete at twice this value. |

The multirate algorithm, operator ordering, stability range, and production acceptance
gates are specified in [BIERMANN_SUBCYCLING_DESIGN.md](BIERMANN_SUBCYCLING_DESIGN.md).
Subcycling is intentionally opt-in because its endpoint-cochain shock path differs from
the retained legacy shock-masked face operator.

`biermann_subcycle=true` additionally requires `biermann_battery=true`, dynamic time
evolution, and a second- or higher-order macro integrator. The shock-suppression controls
remain accepted for input compatibility but do not alter the endpoint-cochain subcycle.

`biermann_coefficient` is a code-unit coefficient, not a CGS value. A physical-unit setup
must map the inverse electron charge consistently into both the induction and electron
drift equations.

There are two supported electron-density normalizations, so their coefficients are not
interchangeable. For the material normalization above,

```text
C_B,material = 1.0364e-4 / (L_0 * sqrt(4*pi*rho_0)),
```

with `L_0` in cm and `rho_0` in g cm^-3. The legacy no-material model uses
`n_e,code=f_e*rho`; for a fixed composition with
`q_e=sum_s(Y_s Zbar_s/A_s)`, the equivalent value is
`C_B,legacy=C_B,material*f_e/q_e`. For the DCI CH normalization
(`L_0=0.1 cm`, `rho_0=1.1 g cm^-3`, `Abar=6.5`, `Zbar=3.5`) these are
`2.7875722321043606e-4` and `4.026493224150743e-4`, respectively. A deck with
`<materials>` must use the former.

The explicit Biermann stability estimate includes the electron drift speed and FLASH's
thermal-magnetic wave speed. On the legacy path it limits the macro MHD timestep. With
subcycling enabled it selects globally synchronized microsteps and no longer clamps the
macro timestep. FLASH recommends a lower CFL number for this formulation. It also
recommends shock detection because a direct pressure-gradient discretization is not
convergent inside a discontinuity; AthenaK therefore enables symmetric shock suppression
by default for the retained legacy operator.

In the tabular branch, cached electron density is regularized with
`n_e,eff=max(n_e,rho*q_min)`, where `q_min` is
`biermann_minimum_electron_fraction`. This floor is used only in denominators and
`grad(ln n_e)`: it does not turn neutral matter into an artificial plasma. A smooth
activation `S(q_e)` is zero through `q_min`, rises from zero to one between `q_min` and
`2*q_min`, and is exactly one above that interval. It multiplies the electric field,
electron drift, and thermal-magnetic speed. Thus a tabular electron-pressure floor cannot
drive Biermann terms in a cell with no physical free electrons.

The default is far below the initial DCI states: direct table interpolation gives
`q_e=0.170142` for CH at `rho=1.1 g cm^-3`, `T=11606 K`, and
`q_e=1.36190e-4` for helium at `rho=1e-5 g cm^-3`. Both therefore have `S=1` without a
case-specific override.

The thermal-magnetic contribution is evaluated in the underflow-safe form

```text
v_TM = S(q_e) * C_B * sqrt((gamma-1)*p_e)/n_e,eff
       * |grad(ln n_e,eff)|.
```

**Mask shape (legacy stage-coupled path):** each directional suppression component ramps
linearly from 1 at a pressure jump of `threshold/5` to 0 at `threshold`, while its
compression activation ramps smoothly from the configured compression threshold to
twice that value. It attenuates only the matching components of `grad(p_e)` and
`curl(B)`, retaining their energy-exchange pairing without multiplying unrelated
transverse fields. The wide pressure band is deliberate: a mask edge can itself be a
curl source, so the 1-side of the ramp must sit in quiet flow well away from the
discontinuity. The earlier scalar, pressure-only mask was measured on a 64² Orszag–Tang
shock (unsuppressed noise floor
max|B3| = 3.3e-6): the original hard 0/1 edge injected 1.5e-5, a narrow
[threshold/2, threshold] ramp 9.3e-6, and the current wide band 6.5e-6 at the default
`threshold = 0.25` — and `threshold = 0.1` drops *below* the unsuppressed floor
(2.1e-6 max, 3.8e-7 rms) while still zeroing the battery inside shocks. Noise is
monotone in the threshold. Guidance: at marginal resolution prefer `threshold ≈ 0.1`
*provided* your smooth-flow battery gradients stay below jump ≈ threshold/5 per two
cells (the analytic regression setup at 64² has jumps of 0.039, which is why the
default keeps `threshold = 0.25`, giving a mask that is exactly 1 there; the same
setup at 32² has jumps of 0.078 and the partially active mask contaminates the
battery by ~14% — under-resolved smooth gradients look like weak shocks to the
detector, exactly as in FLASH). Positivity and energy conservation are unaffected
in all configurations.

The dedicated Biermann subcycle does not apply this local mask. Its CT edge field is
the path-conservative endpoint integral

```text
I_ab = -C_B (p_e,b - p_e,a) / L(n_e,a,n_e,b),
E_edge = I_ab / |x_b-x_a|,
```

where `L(a,b)=(b-a)/(ln(b)-ln(a))` is the positive logarithmic mean. The edge field is
an exact discrete gradient, and therefore has zero CT curl, for constant `n_e`, constant
`p_e`, and the linear/isothermal relation `p_e=T_e n_e` with constant `T_e`. In the
isothermal case it reduces exactly to `-C_B T_e d(ln p_e)`, the gauge-equivalent
Graziani shock integral. A general nonlinear barotrope `p_e=p_e(n_e)` is only
second-order curl-free in smooth flow; the two-point logarithmic-mean formula is not an
exact line integral for every nonlinear equation of state. The formula remains finite
for resolved or unresolved jumps and is used for all three edge orientations in 2-D
and 3-D. The final Poynting flux is reconstructed from the communicated edge field, so
magnetic and total-energy updates use the same cochain.

For fixed-composition material mixtures, `n_e` is evaluated from the local composition.
For tabular EOS, the cached physical electron density is used with the same logarithmic
mean; neutral activation is included in the endpoint pressure coordinate rather than
multiplied onto an edge, preserving the constant-density telescoping invariant. This
is a path-conservative numerical closure, not a claim that an arbitrary nonideal table
has a unique weak shock path. A thermodynamically exact tabular shock treatment requires
table-provided electron chemical potential and entropy (from
`dp_e=n_e dmu_e+s_e dT_e` plus composition terms) or EOS quadrature along the selected
Hugoniot path. Until those quantities are available, tabular shock results should remain
qualified by the endpoint path choice.

**Analytic and FLASH validation (2026-07):** against the exact nonlinear early-time
solution of the crossed-gradient problem the legacy face operator converges at order
1.994–1.998 (2D 32²–256² and 3D, suppression off), with exact
`biermann_coefficient` linearity and a machine-verified timestep formula
(dt ∝ 1/C_B mantissa-exact). Formulation comparison against the FLASH User Guide
(fetched) and Graziani et al. 2015 (ApJ 802:43, full text) shows that the E-field
convention (`biermann_coefficient` ≡ normalized 1/e), face-flux discretization, flux-CT
edge construction, electron-energy/Poynting coupling, and thermal-magnetic wave-speed
limit agree with FLASH's released flux version (AthenaK additionally limits on electron
drift speed).

The dedicated endpoint-cochain operator is validated separately by smooth 2-D temporal
convergence, true-3-D and static-AMR spatial convergence, exact curl-null fixtures,
coarse/fine mortar tests, and fixed-time comparison with a tighter subcycle. The legacy
stage-coupled path implements the "naive" `∇p_e/(e·n_e)` flux (Graziani Eq. 41), while
the endpoint path has the same smooth limit and the isothermal shock limit of the paper's
`ln(p_e)∇T_e` form (Eq. 40). Deliberate deviations from FLASH remain the legacy ramped
mask (versus FLASH's hard `shockDetect`) and the fixed-ionization `n_e=f_e ρ` model when
no material mixture is present. Material and tabular closures use their cached electron
density, with the nonideal shock-path qualification described above. The 2-D and 3-D
endpoint stencils share the same edge integral; any remaining difference is from the
surrounding CT and thermodynamic reconstruction.

On the legacy stage-coupled path, the face drift velocities used by the electron-work
term are computed per refinement level and are not separately SMR/AMR flux-corrected
(unlike the dual-energy face velocities). The measured consequence is an ion/electron
partition perturbation at coarse/fine boundaries bounded by the coarse-fine truncation
envelope, with total energy and dual-energy closure unaffected. The dedicated subcycle
does not have this limitation: it refluxes the normal drift velocity directly in the
additional `vd_flux_` component, then applies electron work from that corrected face
field (see `BiermannBattery::UseCorrectedDriftFlux`).

The implementation requires at least two ghost zones. One-dimensional meshes are
rejected because crossed gradients cannot generate a Biermann field.

## Regression problem

`inputs/mhd/two_temperature_biermann.athinput` initializes smooth perpendicular density
and electron-pressure gradients. The qualification matrix checks the analytic early-time
field, electron-temperature scaling, temporal and 3-D/AMR convergence, active legacy
shock masking, periodic total-energy conservation, positive component energies,
dual-energy closure, CT divergence, tabular and neutral activation, MPI decomposition,
restart, CUDA, and fixed-time production/tight-subcycle agreement. Exact current results
and provenance are recorded in
[BIERMANN_SUBCYCLING_DESIGN.md](BIERMANN_SUBCYCLING_DESIGN.md#qualification-status-and-provenance).

## References

- [FLASH 4.8 MHD user guide, Biermann battery](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node107.html)
- [FLASH 4.8 release notes](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node6.html)
- [Graziani et al., *The Biermann Catastrophe in Numerical MHD*](https://arxiv.org/abs/1408.4161)
