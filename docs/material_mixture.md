# Two-material plasma closure

The optional `<materials>` block enables a two-material closure for Newtonian,
gamma-law MHD. Material 0 is represented by user passive scalar `rho*Y0`; material 1
is its complement. `scalar_index` is relative to the first user scalar in `<mhd>`.
Only `Y0` is clamped, to `[0,1]` (or `[0,rho]` in conserved form).

```text
<mhd>
gamma = 1.6666666666666667
nscalars = 1
two_temperature = true
t_ei_model = constant  # or spitzer

<materials>
nmaterials = 2
scalar_index = 0
material0_name = CH
material0_abar = 6.5
material0_zbar = 3.5
material0_zeff = 5.285714285714286
material0_t_ei = 0.05
material1_name = DT
material1_abar = 2.5
material1_zbar = 1.0
material1_zeff = 1.0
material1_t_ei = 0.1
```

Both materials are fully ionized monatomic gases with common `gamma=5/3`. Following
FLASH, the mixture uses

```text
q_i = sum(Y_s/A_s)
q_e = sum(Y_s Z_s/A_s)
Abar = 1/q_i
Zbar = q_e/q_i
f_e = q_e/(q_i+q_e)
```

`f_e` and `1-f_e` are the local electron and ion heat-capacity fractions. `Zeff` is
electron-density weighted, and the physical electron density is
`rho_cgs*q_e/m_u`. Constant material exchange rates are electron-density weighted.

`t_ei_model=spitzer` requires `<units>` and `<materials>`. It freezes the local
classical Spitzer rate during each operator-split update and integrates the coupled
ion/electron temperatures exponentially, preserving their energy sum and positivity.
The optional controls are `t_ei_coulomb_log` (default 10),
`t_ei_spitzer_multiplier` (default 1), and `t_ei_temperature_floor_kelvin` (default 1).

`MixedOpacityTable` provides the intended radiation-side evaluator, but it is not yet
connected to `ThermalRadiation`; production radiation still uses a single
`<thermal_radiation>/opacity_table_file`.  Once that integration is enabled, the helper
reads `material0_opacity_*` and `material1_opacity_*` parameters from `<materials>`. In
particular:

```text
material0_opacity_table_file = ch.opacity
material1_opacity_table_file = dt.opacity
material0_opacity_interpolation = geometric
material1_opacity_interpolation = geometric
material0_opacity_coordinate_interpolation = log
material1_opacity_coordinate_interpolation = log
```

The mixed mass opacity is
`Y0*kappa0(rho*Y0,Te) + (1-Y0)*kappa1(rho*(1-Y0),Te)`. `geometric` value
interpolation is bilinear in `log(kappa)` when all four corners are positive and falls
back to ordinary bilinear interpolation for a stencil containing a zero. Legacy
`linear` and strict `log` modes are unchanged.
