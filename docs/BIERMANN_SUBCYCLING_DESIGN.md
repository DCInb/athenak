# Biermann multirate time integration

## Scope

The optional Biermann subcycle advances the complete semi-discrete Biermann operator
while the ordinary MHD, thermal-radiation, and laser operators retain the macro
timestep.  It is opt-in so that existing input files continue to use the legacy
stage-coupled update.

For one macro interval `dt`, write `A` for the existing integrator with every Biermann
term disabled and `B` for the complete Biermann operator.  The update is the symmetric
Strang composition

```text
B(dt/2) A(dt) B(dt/2).
```

Each `B` half-interval is tiled exactly by adaptive SSPRK2 (Heun) steps.  For a
microstep `h`,

```text
Y1   = Y0 + h L_B(Y0)
Ynew = 0.5 Y0 + 0.5 (Y1 + h L_B(Y1)).
```

The evolved Biermann state is face-centred magnetic field, conservative total energy,
and electron internal energy.  Ion energy is a redundant algebraic component: after
each predictor/corrector CT update it is reconstructed as total internal minus electron
energy (or retains the auxiliary-energy fallback when conservative subtraction is ill
conditioned).  It is deliberately excluded from the SSPRK blend.  A stage RHS contains
all of the following:

1. electron thermodynamic state, face drift, electron-energy, and electron-enthalpy
   flux construction (the legacy local shock mask is not used);
2. path-conservative CT edge construction and same-level/coarse-fine edge
   reconciliation and communication;
3. Biermann Poynting flux reconstruction from that final edge field, followed by
   coarse/fine conservative correction of the complete energy and drift fluxes;
4. total/electron-energy divergence updates and additive
   `-p_e div(v_e-v)` electron work from the corrected drift field;
5. constrained-transport magnetic update;
6. cell and face boundary exchange, physical boundaries, and AMR prolongation;
7. conservative-to-primitive conversion, cell-centred magnetic field reconstruction,
   algebraic ion-energy closure that preserves the evolved electron energy, and a
   material/two-temperature cache refresh.  The ordinary pressure-partition projection
   is not inserted between predictor and corrector because it modifies both component
   energies and leaves a first-order projection term outside the RK recurrence.

The electron-work term is additive in the SSPRK stage RHS.  The legacy positive
exponential work update is intentionally not reused: composing that frozen-coefficient
map with the low-storage RK weights leaves an extra second-order term and does not give
Heun's method for the complete operator.

## Microstep selection

At the beginning of every accepted microstep, all ranks recompute the raw Biermann
stability limit from the synchronized stage state.  An MPI minimum produces one global
limit,

```text
h_limit = biermann_subcycle_cfl * min_MPI(dt_B_raw),
h       = min(remaining_half_interval, h_limit).
```

The last microstep closes the half-interval exactly.  The limit is recomputed after
every accepted step, and a configurable maximum step count turns an accidental runaway
into a fatal diagnostic instead of a stalled production run.

The supported SSPRK2 method factor is `(0,0.15]`, with a default of
`min(<time>/cfl_number, 0.15)`.  The reproducible nonlinear `128²` regression compares
`0.15` directly with `0.075` at fixed physical time.  The current `B3` relative L2 and
magnetic-energy differences are `5.05e-6` and `2.12e-6`, respectively; the gate requires
them to remain below `1e-4` and `2e-4`.  Values above `0.15` are unqualified and are
rejected during input validation.

## Ordering and invariants

The first half-step follows `before_timeintegrator`, so saved beginning-of-cycle state
still brackets the complete macro update.  The second half-step follows both the final
macro RK stage and the existing `after_timeintegrator` ion/electron exchange and matter
radiation coupling.  Thus the whole existing macro operator, including laser and
radiation work, lies between the symmetric Biermann halves.  The MHD stability state is
refreshed after the trailing half-step.  Orbital/shearing remaps are not repeated inside
microsteps (and 2T MHD already rejects shearing-box configurations).

For time-dependent user boundaries, the leading Biermann map sees the macro start time
and the trailing map sees `t_n+dt`; the driver restores `t_n` before its ordinary time
increment.  This prevents the trailing boundary refresh and its microstages from reusing
the beginning-of-step time.  User-boundary calls inside the existing macro `A` stages
retain AthenaK's established stage-time handling.

On a multilevel mesh, the nonlinear exchange and matter-radiation sources run after the
final macro-stage boundary exchange.  Before the trailing Biermann half-step, the driver
therefore calls `InitBoundaryValuesAndPrimitives()` once more.  This restricts the
source-updated fine state and rebuilds coarse and ghost primitives; applying a nonlinear
source independently to an interpolated ghost state is not equivalent to interpolating
the source-updated fine state.  Each Biermann RK stage likewise closes accepted fine
interior cells before restriction, using the same device-callable thermodynamic closure
as the final full-domain primitive/cache refresh.

The base macro stages skip Biermann fluxes, edge fields, and electron work when
subcycling is enabled.  The Biermann stability limit is still evaluated for diagnostics
and microstep selection, but it no longer clamps the macro MHD timestep.

Periodic total energy remains a conservative finite-volume update, and CT plus edge
communication preserves the discrete face-field divergence.  The independently
evolved two-temperature energies remain an algebraic closure and are synchronized after
each SSPRK stage, as required before the next tabular/material query.

Scheduled restart output on an adaptive mesh is written after the cycle-boundary AMR
pass when subcycling is active.  A resumed trajectory therefore starts from the same
topology selected by the uninterrupted run.  Ordinary field-output ordering and the
legacy Biermann path are unchanged.

The `A+B` composition is second order when the selected macro integrator is at least
second order.  Existing source operators that are themselves applied only after the
macro integrator (notably ion/electron exchange and matter-radiation coupling) are not
made symmetrically split by this feature; whole-application convergence claims must
account for that pre-existing ordering.

## Acceptance gates

Production enablement requires evidence for all of the following:

- disabled-mode compatibility and coefficient-zero controls;
- temporal refinement against a tightly subcycled reference;
- smooth analytic magnetic-field convergence and true three-dimensional generation;
- total-energy conservation, positive component energies, dual-energy closure, and
  face-centred divergence preservation;
- legacy shock-mask compatibility plus subcycle edge-cochain, tabular-material, and
  neutral-activation cases;
- uniform MPI, static/adaptive refinement, restart equivalence, CUDA, and MPI+CUDA;
- fixed-physical-time DCI field/energy comparison and interleaved wall-time brackets.

Fixed-time performance comparisons must use the same executable, input tables, mesh,
macro CFL, output schedule, thread affinity, and physical end time.  The production
subcycle field is compared to a more tightly subcycled run of the *same* endpoint-cochain
operator.  The legacy shock-masked face operator is retained as a timing and
compatibility baseline, not as an apples-to-apples magnetic reference.

## Qualification status and provenance

The following qualification was completed on 2026-08-01.  The implementation was
integrated on branch `2T` from pre-integration base `5ba529da`; the commit containing
this document is the source provenance for the subcycle.  The qualification artifact
also archives the exact tested source snapshot as
`/tmp/dci-biermann-current-source.hR50HvWi/current-source-snapshot.tar`, SHA-256
`922dda6b0ec364b4de31c090782624c45fea0f9a2cc3c738c27a716e251b73ed`.

| Gate | Current evidence | Result |
| --- | --- | --- |
| Serial CPU physics matrix | `/tmp/biermann-cpu-qualification-final-current.junit.xml`; executable SHA-256 `0d695d7dc1318d6dc9d8446b11be3483a91c860acfffdd587a04f7a22aa08808` | 12 passed: analytic/disabled/zero controls, temporal and 3-D convergence, adaptive stability, cochain, closure, SMR/AMR, restart, source synchronization, tabular floors, active tabular material, and neutral activation |
| Supported-CFL comparison | `/tmp/biermann-stability-cpu-current.junit.xml` | `0.15` versus `0.075` passes field, magnetic-energy, positivity, closure, conservation, and divergence gates |
| Legacy shock-mask compatibility | `/tmp/biermann-legacy-mask-cpu-current.junit.xml` and `/tmp/biermann-legacy-mask-gpu-current.junit.xml` | Active mask changes the retained legacy operator on CPU and CUDA while preserving positivity and total energy |
| MPI CPU | `/tmp/biermann-mpicpu-qualification-final.junit.xml`; executable SHA-256 `60d2d1dc1b5788a2eea6c9ae5b4a3c525a8eccade32dd325c6a3a175088316bf` | Uniform, static refinement, and dynamic AMR agree across one and two ranks |
| Serial CUDA | `/tmp/biermann-cuda-qualification-final-current.junit.xml` and `/tmp/biermann-cuda-qualification-current.junit.xml`; executable SHA-256 `97676bace646a66f4d415ddc182b8b2101c2dd9f47d14ecef4c00064a79d08a9` | 11 core/tabular/stability checks plus device old-scratch/fused-RK equivalence pass |
| MPI+CUDA | `/tmp/biermann-mpi-gpu-qualification-current.junit.xml` | Uniform, static refinement, and dynamic AMR agree across one and two ranks |
| Current-source DCI | `/tmp/dci-biermann-current-source.hR50HvWi/analysis-current-vs-frozen.json` | Fixed-time production fields and histories reproduce the frozen qualified result |

The CUDA executable is MPI-enabled.  Direct singleton tests require the local
`hpcx-init-ompi.sh` followed by `hpcx_load`; sourcing only `bashrc_athenaK` leaves the
relocated Open MPI installation with its build-time `/proj/nv/...` prefix and fails in
`MPI_Init` before AthenaK starts.  This was a launch-environment failure, not a device
solver failure.

### Fixed-time DCI comparison

All long comparisons use the same deck (SHA-256
`f05656f30bec0598c8b6ecc9136064c9ec80127b90f7bcdc240a52cd752bd992`), a
`256x256x8` mesh, one rank, 12 threads pinned to cores 0--11, macro CFL `0.25`, and
`t=1e-4`.  The frozen executable SHA-256 is
`9166e2122cbcca6fa89388558f0a1d73e99bdfca6c4eb97f4f242a65e6a7bdfc`; the fresh
current-source executable SHA-256 is
`e8a6f8905532d455d3e14db43bf5fcedb45b1f8052c1200e1daa097317cce3d8`.

| Run | Wall time | Macro cycles | Biermann microsteps |
| --- | ---: | ---: | ---: |
| Frozen legacy timing baseline | 1249.59 s | 108 | n/a |
| Frozen production subcycle, CFL `0.15` | 144.08 s | 6 | 30 |
| Fresh current-source production subcycle, CFL `0.15` | 144.23 s | 6 | 30 |
| Frozen tight subcycle, CFL `0.0375` | 313.02 s | 6 | 102 |

The fresh wall speedup over the frozen legacy baseline is `8.664x` (`8.891x` using
program time); its wall time is only 0.104% above the frozen production run.  Short
interleaved legacy/subcycle timing brackets give a more conservative `1.315x` wall
speedup when initialization dominates.

The production-versus-tight magnetic-vector relative L2 difference is `7.5461e-6`, and
the magnetic-energy relative difference is `1.5903e-6`.  The fresh current-source and
frozen production magnetic vectors are bitwise identical, their magnetic-energy
difference is zero, and their history files are byte-identical.  For the fresh run,
`max|divB|=1.8006e-20`, relative mass drift is `-2.6845e-13`, and the complete
matter/radiation/laser energy-budget residual is `2.8630e-5` of laser deposition.  The
magnetic-energy and boundary-loss calculation is preserved at
`/tmp/dci-biermann-current-source.hR50HvWi/current_production/magnetic_energy_budget/`;
it exactly matches the frozen production CSV.

The legacy and subcycle magnetic energies differ by about `1.92e7` because they use
different discrete Biermann operators (shock-masked face gradients versus the endpoint
cochain).  The legacy run is therefore only a compatibility and timing baseline.  The
tight subcycle is the magnetic-accuracy reference.

These results establish second order for the split `A+B` map and for the complete
Biermann operator.  They do not establish global second order for the source-active DCI
application: ion/electron exchange and matter-radiation coupling remain one-sided after
the macro RK solve, as noted above.
