# Laser transport for two-temperature MHD

## Supported model and task ordering

The laser module is enabled by the presence of a `<laser>` block. It supports Cartesian,
Newtonian, ideal-gas, two-temperature MHD on uniform, statically refined, and adaptively
refined meshes. Each Runge--Kutta stage executes

```text
MHD finite-volume update
-> 2T dual-energy compression work
-> initialize laser rays and optical properties
-> trace and deposit ray power
-> apply laser energy over beta_stage*dt
-> ordinary MHD source, boundary, CT, and primitive tasks
```

The medium is frozen during a ray trace. `model=straight` selects DDA transport and
`model=refractive` selects the Hamiltonian transport described below. Momentum
deposition, ray splitting, and frequency groups are not implemented. Laser cell
diagnostics (`cell_data`, including the cumulative deposited energy density reported
as `laser_energy`) are not written to restart files: on restart the physical state
(which already contains all deposited energy) continues exactly, but the cumulative
diagnostic restarts from zero. Rays transfer
directly between same-rank MeshBlocks. Across ranks, off-rank rays are compacted by destination
into contiguous packets and advanced by a nonblocking composite task. Each transport
wave bounds device work by `max_transport_iterations*max_segments_per_launch` segments
per ray; rays that hit the cap stay active and are re-traced in subsequent waves (in
serial exactly as under MPI), so results are independent of the rank decomposition.
Only after `max_mpi_waves` waves is leftover power booked as remaining. Count exchange,
packet receives/sends, and global completion are polled with `TaskStatus::incomplete`.
`gpu_aware_mpi=true` passes device packet buffers directly to MPI; the default uses
`Kokkos::SharedHostPinnedSpace` staging for MPI implementations without device support.

The global leaf-block map is rebuilt at every stage, including after adaptive refinement.
At coarse/fine interfaces a ray is mapped by its forward-probed physical position into
the containing leaf block; its power and direction are unchanged and it is not split.
Laser cell diagnostics use AthenaK's ordinary CC restriction, prolongation, and AMR
load-balance buffers so cumulative deposited energy follows a changing mesh.

When `critical_reflection=true`, centered electron-density gradients linearly locate a
turning point inside the current cell. Normal incidence uses `n_turn=n_c`; the reduced
oblique model uses `n_turn=n_c*cos(theta)^2`. The ray is reflected specularly about the
local density-gradient normal, displaced by `reflection_offset_fraction` of a cell, and
continues through the ordinary device queue. `max_reflections_per_ray` bounds trapping;
any power stopped by that bound is reported as remaining rather than discarded.

Reflection requires a resolved density gradient: the turning point is located by
linear interpolation of the cell-centered electron-density gradient, so a ray launched
into a *uniform* supercritical medium finds no gradient, never turns, and is silently
absorbed in the first cell even with `critical_reflection=true`. Beams must approach
critical density through a resolved ramp. Also note that when `max_mpi_waves` is the
binding limit (absorbing medium, rays still active at the final wave), the amount
deposited before cutoff depends on the rank decomposition — converged runs
(`remaining=0`) are decomposition-independent, truncated ones are not.

## Refractive transport

The opt-in refractive model evolves the normalized wave vector
`q=c*k/omega0` using `ell=c*t`:

```text
dx/dell = q
dq/dell = -0.5*grad(ne/nc).
```

A second-order kick-drift-kick update uses centered grid gradients and a local linear
density reconstruction. Its step is limited by `refractive_cell_fraction` times the
smallest cell width, `refractive_curvature_fraction*|q|/|dq/dell|`, the nearest cell
face, and `refractive_tau_max` per absorbing step. Refraction uses the same exact
attenuation, electron-energy coupling, MeshBlock queues, MPI packets, and AMR leaf map
as the straight tracer. It therefore must not be combined with the straight-tracer
`critical_reflection` switch; a refractive ray turns continuously as `|q|` approaches
zero.

The runtime monitors the normalized dispersion invariant

```text
epsilon_omega = abs(ne/nc + |q|^2 - 1).
```

The maximum error over all rays and MPI ranks is printed as `dispersion` and is fatal
above `dispersion_tolerance`. Path-weighted cell outputs provide the trajectory needed
for regression and analysis: `laser_dir1..3`, `laser_dispersion_error`, and
`laser_x1_moment..laser_x3_moment`. Divide any of these by `laser_path` for a cell-path
average.

## Energy convention

A ray stores instantaneous power `P` in code energy per code time. In a cell segment of
length `ds`,

```text
tau = K*ds
Pout = Pin*exp(-tau)
delta_P = -Pin*expm1(-tau)
```

The cell array `laser_q` stores deposited power density,

```text
Q_laser = sum(delta_P)/cell_volume,
```

and a stage deposits `Q_laser*beta_stage*dt`. For `deposition_target=electron`, AthenaK
adds the same physical increment to conservative total energy and to the redundant 2T
electron-energy component. This is not double counting: total energy and the auxiliary
electron component are separate stored equations, and both must represent the same
electron heating. For `deposition_target=total`, only conservative total energy is
incremented and the ordinary 2T synchronization partitions the new internal energy.
The cumulative `laser_energy` diagnostic follows the same `gam0`, `gam1`, and `beta`
low-storage Runge--Kutta recurrence as the conserved MHD state, so it records the net
source contribution rather than the sum of intermediate-stage increments.

The per-stage conservation diagnostic is

```text
P_launch = P_deposit + P_escape + P_remaining
epsilon_E = abs(P_launch-P_deposit-P_escape-P_remaining)/P_launch.
```

## Optical models and units

`absorption_model=constant` interprets `absorption_coefficient` in inverse code length and
is the reference/testing model. `absorption_model=inverse_bremsstrahlung` evaluates the
FLASH cell-average collision model in cgs:

```text
nc = me*pi*c^2/(e^2*lambda^2)
nu_ib = (ne/nc)*nu_ei
K_IB = nu_ib/[c*sqrt(1-ne/nc)].
```

Electron density is `rho_cgs*electron_number_per_gram`; electron temperature is the 2T
temperature multiplied by `temperature_scale_cgs`. The returned cgs coefficient is
converted to inverse code length. The weak-collision expression is evaluated only below
the critical surface.

| Quantity | Code-unit input | `unit_system=cgs` input |
| --- | --- | --- |
| Ray position/radius | code length | code length |
| Wavelength | code length | cm |
| Beam power | code energy/time | erg/s |
| `laser_q` | code energy/(volume*time) | stored in code units |
| Deposited energy | code energy/volume | stored in code units |
| `ne` for physical IB | derived through cgs scales | cm^-3 |
| `Te` for physical IB | derived through cgs scales | K |

When a `<units>` block is present, its length, density, temperature, and power scales are
used. They can be overridden by `length_scale_cgs`, `density_scale_cgs`,
`temperature_scale_cgs`, and `power_scale_cgs` in `<laser>`.

## Input parameters

```text
<laser>
model = straight
deposition_target = electron
electron_temperature_model = two_temperature
absorption_model = constant
absorption_coefficient = 1.0
unit_system = code
length_scale_cgs = 1.0
density_scale_cgs = 1.0
temperature_scale_cgs = 1.0
power_scale_cgs = 1.0
electron_number_per_gram = 1.0
inverse_bremsstrahlung_coulomb_log = -1.0
critical_reflection = false
oblique_turning = true
max_reflections_per_ray = 8
reflection_offset_fraction = 1.0e-10
nbeams = 1
max_segments_per_launch = 16
max_transport_iterations = 64
max_mpi_waves = 1024
gpu_aware_mpi = false
minimum_power_fraction = 1.0e-14
conservation_tolerance = 1.0e-10
periodic_transport = false
report_diagnostics = true
refractive_cell_fraction = 0.25
refractive_curvature_fraction = 0.25
refractive_tau_max = 0.25
dispersion_tolerance = 1.0e-3

beam0_power = 1.0
beam0_wavelength = 1.0
beam0_nrays = 1
beam0_origin_x1 = -0.5
beam0_origin_x2 = 0.0
beam0_origin_x3 = 0.0
beam0_direction_x1 = 1.0
beam0_direction_x2 = 0.0
beam0_direction_x3 = 0.0
beam0_radius = 0.0
beam0_profile = uniform
beam0_zeff = 1.0
beam0_start_time = 0.0
beam0_end_time = 1.0
```

For inverse bremsstrahlung, a positive
`inverse_bremsstrahlung_coulomb_log` fixes the Coulomb logarithm to that value.
The default non-positive value evaluates the local FLASH Debye-number expression.

Beam parameters repeat as `beam1_*`, `beam2_*`, and so on. Direction vectors are
normalized during construction. `uniform` and `gaussian` profiles are normalized so the
sum of ray powers equals the specified beam power.

## Ray data fields

The production structure-of-arrays layout contains position, direction, power, current
MeshBlock GID, cell indices, and status for every ray. Two integer queues hold current
and next active ray IDs; queue compaction uses `Kokkos::parallel_scan`. Optional cell
output contains deposited power density, cumulative deposited energy density, number of
traced segments, summed optical depth, summed path length, path-integrated direction,
dispersion error, and segment-midpoint position. Ray MPI packets additionally preserve
the normalized wave vector and maximum dispersion error so refractive paths are
decomposition independent.

## References

- [FLASH Energy Deposition unit](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node122.html)
- [FLASH laser-model tutorial](https://flash.rochester.edu/site/flashcode/user_support/tutorial_talks/RAL_May2012/lasermgd.pdf)
