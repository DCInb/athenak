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
`biermann_coefficient`. The transverse components of `E_B` are constructed on coordinate
faces and arithmetically averaged to constrained-transport edges. The same face fields
add the Biermann Poynting flux `E_B x B` to total energy.

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
`epsilon_e+p_e`. The positive exponential work update uses the same frozen `p_e`; source
coefficients are deliberately held at the stage-start thermodynamic state until the next
two-temperature synchronization. There are no per-face table inversions.

## Input

```text
<mhd>
eos = ideal
two_temperature = true
dual_energy = true
biermann_battery = true
biermann_coefficient = 1.0
biermann_minimum_electron_fraction = 1.0e-12
biermann_shock_suppression = true
biermann_shock_threshold = 0.25
```

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `biermann_battery` | `false` | Enable the flux-form Biermann and 2T energy terms. |
| `biermann_coefficient` | `1.0` | Normalized inverse-charge coefficient `C_B`; zero disables all terms. |
| `biermann_minimum_electron_fraction` | `1.0e-12` | Minimum tabular `q_e=n_e/rho` for a resolved plasma. |
| `biermann_shock_suppression` | `true` | Disable Biermann faces adjacent to detected pressure jumps. |
| `biermann_shock_threshold` | `0.25` | Threshold for the symmetric centered gas-pressure jump. |

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

The explicit timestep includes the electron drift speed and FLASH's thermal-magnetic
wave speed. FLASH recommends a lower CFL number for this formulation. It also recommends
shock detection because a direct pressure-gradient discretization is not convergent
inside a discontinuity; AthenaK therefore enables symmetric shock suppression by default.

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

**Mask shape (2026-07 validation):** the suppression mask ramps linearly from 1 at a
pressure jump of `threshold/5` to 0 at `threshold`. The wide band is deliberate: the
mask edge is itself a curl source whose magnitude is set by |E| at the transition
contour, so the 1-side of the ramp must sit in quiet flow well away from the
discontinuity. Measured on a 64² Orszag–Tang shock (unsuppressed noise floor
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

**Analytic and FLASH validation (2026-07):** against the exact nonlinear early-time
solution of the crossed-gradient problem the module converges at order 1.994–1.998
(2D 32²–256² and 3D, suppression off), with exact `biermann_coefficient` linearity
and a machine-verified timestep formula (dt ∝ 1/C_B mantissa-exact). Formulation
comparison against the FLASH User Guide (fetched) and Graziani et al. 2015 (ApJ
802:43, full text): the E-field convention (`biermann_coefficient` ≡ normalized
1/e), face-flux discretization, flux-CT edge construction, electron-energy/Poynting
coupling, and the thermal-magnetic wave-speed dt limit all agree with FLASH's
released flux version (AthenaK additionally limits on the electron drift speed).
Note both codes implement the "naive" `∇p_e/(e·n_e)` flux (Graziani Eq. 41), not
the paper's shock-convergent `ln(p_e)∇T_e` form (Eq. 40). Deliberate deviations:
the ramped mask above (vs FLASH's hard `shockDetect`) and the fixed-ionization
`n_e = f_e·ρ` model (vs FLASH's 3T EOS Z̄) — no ionization-gradient battery,
single-material plasmas only. The 2D and 3D flux-CT edge stencils differ at O(h²)
(dimensional reduction of the arithmetic average), so z-invariant 3D runs agree
with 2D runs to truncation, not round-off.

Note the face drift velocities used by the electron-work term are computed per
refinement level and are not SMR/AMR flux-corrected (unlike the dual-energy face
velocities); the measured consequence is an ion/electron partition perturbation at
coarse/fine boundaries bounded by the coarse–fine truncation envelope, with total
energy and dual-energy closure unaffected (see the comment on
`BiermannBattery::ApplyElectronWork`).

The implementation requires at least two ghost zones. One-dimensional meshes are
rejected because crossed gradients cannot generate a Biermann field.

## Regression problem

`inputs/mhd/two_temperature_biermann.athinput` initializes smooth perpendicular density
and electron-pressure gradients. The CPU regression checks the analytic early-time
magnetic field, its electron-temperature scaling, periodic total-energy conservation,
positive ion/electron energies, and the 3D flux-CT path.

## References

- [FLASH 4.8 MHD user guide, Biermann battery](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node107.html)
- [FLASH 4.8 release notes](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node6.html)
- [Graziani et al., *The Biermann Catastrophe in Numerical MHD*](https://arxiv.org/abs/1408.4161)
