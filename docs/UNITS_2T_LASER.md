# Choosing consistent units for laser + thermal radiation + Biermann runs

AthenaK's hydro/MHD equations are scale-free; **only the laser module reads cgs
scales** (from `<units>` if present, overridable in `<laser>`), while
`<thermal_radiation>` `arad`/`c_light` and `<mhd>` `biermann_coefficient` are raw
code-unit numbers. Nothing cross-checks them — you must set all of them from one
unit system. (Scale-transformation invariance of these rules was verified to 5e-13
in the 2026-07 validation campaign.)

**Step 1 — pick three scales and the material.** L0 [cm], rho0 [g/cc], T0 [K], and
for a fully ionized plasma with mean charge Z and mean mass number A:
`f_e = Z/(Z+1)` (`electron_heat_capacity_fraction`), `N_e = Z/(A*m_u)`
(`electron_number_per_gram`).

**Step 2 — derived scales.** Because code temperature is measured in v0² units via
`p_e = f_e rho T`:

```
v0 = sqrt(kB*T0*Ne/fe)    t0 = L0/v0      p0 = rho0*v0^2
P0 = rho0*L0^2*v0^3       B0 = sqrt(4*pi*rho0)*v0
```

**Step 3 — set every knob.**

```
<laser>    length_scale_cgs = L0     density_scale_cgs = rho0
           temperature_scale_cgs = T0   power_scale_cgs = rho0*L0^2*v0^3
           electron_number_per_gram = Ne    beam powers/wavelengths in cgs if unit_system=cgs
<thermal_radiation>
           arad    = 7.5657e-15 * T0^4 / (rho0*v0^2)
           c_light = chat_cgs / v0          # = 2.998e10/v0 if not reduced
           group_bound_g = (h*nu_g/kB) / T0
           kappa_* = kappa_cgs * rho0 * L0
<mhd>      biermann_coefficient = 1.0364e-4 * mu_bar / (L0*sqrt(4*pi*rho0))
           # mu_bar = fe/(Ne*m_u) = A/(Z+1); this is d_i/L0, the normalized ion inertial length
```

**Worked case — `laser-target/laser_target.athinput`** (L0=100 um, rho0=1 g/cc,
T0=1 keV, CH: N_e=3.243e23/g): v0=3.224e7 cm/s, t0=0.310 ns, p0=1.04 Gbar,
B0=114 MG. Consistent values (with the deck's f_e=0.5): `power_scale_cgs=3.35e18`
(deck: 1e21 — the demo beam is 299x weaker than its label), `arad=0.132` (deck:
0.05), `c_light=930` (deck: 1.0, i.e. an aggressive RSLA with c_hat=v0),
`biermann_coefficient=2.71e-3` (deck: 2e-2, 7.4x overdriven); and CH would require
f_e=0.7778, not 0.5. The shipped deck is therefore a *code-unit demonstration* of
the coupled physics (as its README states), not a calibrated physical scenario —
use this recipe to build one.
