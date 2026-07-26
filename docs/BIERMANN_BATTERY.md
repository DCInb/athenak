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

## Input

```text
<mhd>
eos = ideal
two_temperature = true
dual_energy = true
biermann_battery = true
biermann_coefficient = 1.0
biermann_shock_suppression = true
biermann_shock_threshold = 0.25
```

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `biermann_battery` | `false` | Enable the flux-form Biermann and 2T energy terms. |
| `biermann_coefficient` | `1.0` | Normalized inverse-charge coefficient `C_B`; zero disables all terms. |
| `biermann_shock_suppression` | `true` | Disable Biermann faces adjacent to detected pressure jumps. |
| `biermann_shock_threshold` | `0.25` | Threshold for the symmetric centered gas-pressure jump. |

`biermann_coefficient` is a code-unit coefficient, not a CGS value. A physical-unit setup
must map the inverse electron charge consistently into both the induction and electron
drift equations.

The explicit timestep includes the electron drift speed and FLASH's thermal-magnetic
wave speed. FLASH recommends a lower CFL number for this formulation. It also recommends
shock detection because a direct pressure-gradient discretization is not convergent
inside a discontinuity; AthenaK therefore enables symmetric shock suppression by default.

**Mask shape (2026-07 validation):** the suppression mask ramps linearly from 1 at a
pressure jump of `threshold/2` to 0 at `threshold` (a hard 0/1 edge was measured to
inject ~4.7× more spurious B3 than it suppressed at a marginally resolved 64²
discontinuity; the ramp halves that artifact at every tested threshold). Suppression
still cannot beat the unsuppressed noise floor at marginal resolution with the default
`threshold=0.25` — a transition in the face E-field is itself a curl source — so near
marginally resolved shocks either lower the threshold (0.1 reached unsuppressed noise
levels in testing while still zeroing strong shocks) or verify the mask's effect at
your resolution. Positivity and energy conservation are unaffected in all cases.

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
