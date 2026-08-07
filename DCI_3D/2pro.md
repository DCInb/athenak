# DCI_3D bad-EOS and radiation-timestep repair plan

## Objective

Diagnose and eliminate the time-growing `bad_eos` population in the recent DCI_3D
run, then replace the physical-light-speed radiation timestep restriction with a
FLASH-like implicit transport strategy without weakening radiation source coupling
or conservation checks.

## Execution order

1. **Preserve and identify the failing evidence**
   - Locate the newest DCI_3D run, its phase logs, history/output files, input overrides,
     executable hash, and first cycle/time at which `bad_eos` becomes nonzero.
   - Correlate `bad_eos` with EOS query flags, density/temperature floors, material
     fractions, radiation energy, and the affected spatial region.
   - Reproduce the earliest failure with the smallest restart or compact run that retains
     the symptom.

2. **Trace the EOS failure to one code path**
   - Audit where `bad_eos` is incremented and decode its constituent query flags.
   - Separate genuine IONMIX table-range violations from normalization, floor,
     reconstruction, AMR, laser, radiation, or electron-ion exchange artifacts.
   - Add a focused diagnostic/regression that fails for the observed state before
     changing behavior.

3. **Fix `bad_eos` at its source**
   - Correct the responsible state update, closure bound, table query, or diagnostic
     classification; do not merely suppress the counter.
   - Preserve conservative total/component energies and normalized explicit CH/Au/He
     fractions.
   - Verify the focused reproducer, affected CPU regressions, and a compact CUDA DCI run
     show no growing `bad_eos` population.

4. **Map the FLASH radiation timestep strategy**
   - Inspect the archived FLASH configuration/source used by DCI_3D and document how it
     avoids a timestep fixed by the physical speed of light.
   - Compare that rule with AthenaK's `ThermalRadiation::NewTimeStep` and source-coupling
     paths, identifying which signal speed belongs in transport stability and which
     physical constants must remain unchanged.

5. **Implement the FLASH-like relaxation**
   - Add a centered, frozen-coefficient backward-Euler FLD option that removes the explicit
     radiation transport CFL while retaining physical `c` in transport, emission,
     absorption, and energy exchange.
   - Add input validation and regression coverage for the relaxed timestep, optically
     thick diffusion behavior, positivity, and conservation.
   - Update the DCI_3D deck and documentation with the chosen parameter and rationale.

6. **Integrated verification**
   - Run focused material/EOS, radiation, restart, laser, and Biermann regressions.
   - Build CPU and CUDA/MPI targets with the local AthenaK toolchain.
   - Run `DCI_3D/run_case.py --clean --mode validate`, then a compact multi-cycle smoke
     case long enough to cover the former first `bad_eos` event.
   - Confirm: `bad_eos == 0` (or no unexplained growth), normalized three-material
     fractions, finite states, accepted energy budgets, and a radiation timestep no
     longer bounded by physical light speed.

7. **Final review**
   - Run `git diff --check`, inspect all changed files, record verification evidence,
     and leave unrelated workspace changes untouched.

## Acceptance criteria

- The first offending EOS state has a demonstrated root cause and a regression test.
- The fix removes the cause rather than masking `bad_eos` diagnostics.
- DCI_3D explicitly retains helium as the low-density background material.
- Radiation transport uses FLASH-like backward-Euler diffusion at physical light speed;
  source terms retain the same physical scaling.
- Focused tests, CPU/CUDA builds, DCI validation, and the compact smoke run pass; a DCI
  run reaching the historical `0.025 ns` onset remains required before claiming
  long-time elimination of the original symptom.

## Execution record

### 1. Failing evidence and root cause

- The failing run first records `eos_bad=5726` at `t=0.025 ns`; the lifetime count
  reaches `15700077` near `t=1.725 ns`.
- Every disallowed state inspected carries only `ionmix_energy_above_table` (`0x20`).
  No affected state has a high-density or temperature-axis violation.
- The first affected cells are laser-heated, CH-dominated ablation-plume cells.  Their
  electron or ion specific energy is as much as 3.37 times the table endpoint energy,
  while recovered temperatures pin at the CH/He endpoint
  `1.160451812e8 K` (10 keV).
- Laser deposition is conservative and must not be clipped.  The defect is the inverse
  EOS policy: `clamp` retains the deposited energy but returns the endpoint state and an
  above-energy flag.  Thermodynamic flags are deliberately ORed over a cell's lifetime,
  explaining the monotonic history count.
- Repair selected: an opt-in FLASH-motivated high-temperature continuation of the last
  two positive IONMIX planes, with an analytic high-energy inverse.  Existing `clamp`
  and `error` behavior remains unchanged.

### 2. FLASH radiation-timestep mapping

- The archived FLASH setup requests MGD and the unsplit diffusion unit.  Its decks set
  `dt_diff_factor=1.0e100` (explicitly labelled "Disable diffusion dt") and retain the
  physical speed of light.
- FLASH can disable the diffusion timestep because each radiation group is advanced by
  a time-lagged, backward-Euler diffusion solve (FLASH 4.8 guide, equations 25.6 and
  19.2).  Its strategy is not an explicit physical-`c` CFL and not a reduced-light-speed
  substitution.
- The decks also contain `rt_dtFactor=0.02`, but leave `rt_computeDt` unset.  The FLASH
  runtime schema defaults `rt_computeDt` to false, so that coefficient is inactive in
  this reference.  That switch does not map directly to AthenaK's source guard: AthenaK
  time-lags its local emission source rather than using FLASH's fully coupled nonlinear
  source solve.  DCI therefore retains `source_cfl=0.1` as an independent accuracy limit.
- AthenaK's asymptotic-preserving explicit FLD removes the singular optically thin
  diffusion bound, but its remaining streaming stability limit is still proportional
  to `dx/c`.  A reduced `c_light` also changes emission and absorption rates.
- Repair selected: retain explicit FLD as the compatibility default and add an opt-in
  centered, frozen-coefficient backward-Euler transport mode.  DCI_3D selects that mode and
  physical `c`; only the explicit transport `c*dt/dx` restriction is removed from the
  global timestep, while the source-accuracy guard remains active.

### 3. Implementation boundary and evidence scope

- The implicit path is a centered finite-volume diffusion solve: it forms a
  harmonic-limited coefficient from the old-state centered gradient and retains the
  physical value at resolved gradients.  Only roundoff-flat limited cells use
  `D <= alpha*dx_min/2`.  The coefficient halo is exchanged before PCG so MPI-shared
  face coefficients remain symmetric after an implicit-conduction update, then cell
  coefficients are arithmetic-averaged to faces.  It is not the explicit AP/upwind flux
  path.
- Every physical vacuum face enforces `D_face <= alpha*dx_normal/2`.  The centered
  operator, Jacobi preconditioner diagonal, and `rad_Pesc` surface diagnostic use the
  same capped face coefficient.
- DCI pins Jacobi-PCG to `implicit_tolerance=1e-10` and
  `implicit_max_iterations=2000`.  A fresh true `b-Ax` residual validates recursive PCG
  convergence.  The incoming conserved radiation state must already be finite and
  nonnegative.  Only finite tolerance-scale negative solver roundoff can be projected:
  negative cells go to zero, positive cells are volume-rescaled to conserve integrated
  group energy, and the projected state must pass another true-residual check.  Larger
  negatives or non-finite values abort.
- `transport_discretization=asymptotic-preserving` and the `ap_*` thresholds remain in
  the DCI decks as explicit-mode fallback settings; they do not select or alter the
  implicit stencil.
- The FLASH correspondence is the use of physical `c`, time-lagged coefficients, and
  backward-Euler transport.  AthenaK does not claim FLASH-identical spatial
  discretization or its fully coupled nonlinear source solve.
- The current solver rejects SMR/AMR, freezes coefficients for each group solve, and
  treats local matter-radiation exchange as a separate time-lagged implicit source under
  `source_cfl=0.1`.  This guard still contains physical `c`; it lowers the initial DCI
  candidate step from `1.068285e-6 ns` with the guard disabled to `7.88743694e-7 ns`,
  about 26 percent.  Only the transport `c*dt/dx` restriction has been removed.
- Dense-reference focused tests cover constant periodic, harmonic-limited periodic, and
  harmonic-limited vacuum matrices.  Ten focused CPU tests passed, the CUDA/MPI target
  built, and compact seven-GPU initialization validation passed.
- The exact current-tree seven-GPU 50-cycle phase-1 smoke passed in 27.36 s, ending at
  `t=6.542658936097e-05 ns` with `eos_bad=0`.  Its domain-integrated CH/Au/He fractions
  were `0.07224481712971552/0.9276974187345620/5.776413572244726e-05`, summing to one
  within roundoff.
- Exact smoke evidence is recorded in `DCI_3D/runs/final_fix_smoke/phase1.log`,
  `dci_3d_calibration.user.hst`, and `run_status.json`.
- That smoke endpoint is about 382 times earlier than the historical first bad-EOS event
  at `0.025 ns`, so it does not establish long-time elimination.  Full-scale 20-group
  Jacobi-PCG convergence, memory use, and performance are also unproven; multigrid or a
  stronger preconditioner is future work.  Long-horizon behavior, restart continuity,
  chain-energy closure, and production-scale performance require the exact-hash DCI
  evidence gate, and any source or deck change invalidates earlier evidence.
