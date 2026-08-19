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
   - Run `DCI_3D/run_case.py --clean --mode validate`, a compact multi-cycle/restart
     smoke for integration, and a read-only historical restart replay that contains the
     former high-energy `bad_eos` population.
   - Confirm: `bad_eos == 0` (or no unexplained growth), normalized three-material
     fractions, finite states, accepted energy budgets, and a radiation timestep no
     longer bounded by physical light speed.

7. **Final review**
   - Run `git diff --check`, inspect all changed files, record verification evidence,
     and leave unrelated workspace changes untouched.

## Acceptance criteria

- The historical high-energy bad-EOS state class has a demonstrated root cause and
  regression coverage; the exact first offending cell was not preserved.
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
- The historical full-volume outputs contain `3910058`, `13450852`, and `15671476`
  cells with a disallowed EOS bit at `0.5`, `1.0`, and `1.5 ns`, respectively.  The only
  disallowed bit present is `ionmix_energy_above_table` (`0x20`); affected cells can also
  carry the allowed density-below or energy-below bits.
- A read-only scan of all `33554432` active cells in the preserved SHA-256-pinned
  CH/He restart at `1 ns` (cycle 48420) finds `1688558` cells whose current ion or
  electron energy is above the native endpoint under `clamp`.  The worst target/endpoint
  ratio is `8.016194089`; the endpoint temperature is `1.160451812e8 K` (10 keV).
  The same stored states under `flash-extrapolate` have zero high-energy flags, remain
  finite and positive, reach `9.541060299e8 K`, and recover the volume-integrated ion
  and electron energies to relative residuals below `1e-15`.
- Laser deposition is conservative and must not be clipped.  The defect is the inverse
  EOS policy: `clamp` retains the deposited energy but returns the endpoint state and an
  above-energy flag.  Thermodynamic flags are deliberately ORed over a cell's lifetime,
  explaining the monotonic history count.
- Repair selected: an opt-in FLASH-motivated high-temperature continuation of the last
  two positive IONMIX planes, with an analytic high-energy inverse.  Existing `clamp`
  and `error` behavior remains unchanged.
- The scanner and evidence JSON are under `historical_eos_restart_scan/` and `evidence/`.
  This is exact same-state recovery for the legacy two-material CH/He layout, not a clean
  trajectory replay and not evidence for the current CH/Au/He restart layout.  The
  historical producer executable hash and the exact cell state at the first `0.025 ns`
  history observation were not preserved, so claims about the first offending cell's
  location or source remain inferential.

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
  operator, preconditioner diagonals, and `rad_Pesc` surface diagnostic use the same
  capped face coefficient.
- DCI pins PCG to `implicit_tolerance=1e-10`, `implicit_max_iterations=2000`, and the
  fixed linear `block-coarse` preconditioner.  Its factor-three global Galerkin V-cycle
  is `45^3 -> 15^3 -> 5^3 -> one value/MeshBlock`: restriction is exactly `P^T`, MPI
  face values enter every global residual, pre/post red-black sweeps are transposes, and
  the replicated 343-MeshBlock root matrix is solved by exact Cholesky.  A fresh true
  `b-Ax` residual validates recursive PCG convergence.  The incoming conserved radiation
  state must already be finite and nonnegative.  Only finite tolerance-scale negative
  solver roundoff can be projected:
  negative cells go to zero, positive cells are volume-rescaled to conserve integrated
  group energy, and the projected state must pass another true-residual check.  Larger
  negatives or non-finite values abort.
- `transport_discretization=asymptotic-preserving` and the `ap_*` thresholds remain in
  the DCI decks as explicit-mode fallback settings; they do not select or alter the
  implicit stencil.
- The FLASH correspondence is the use of physical `c`, time-lagged coefficients, and
  backward-Euler transport.  AthenaK does not claim FLASH-identical spatial
  discretization or its fully coupled nonlinear source solve.
- The current solver rejects SMR/AMR and shear-periodic boundaries, freezes coefficients
  for each group solve, and treats local matter-radiation exchange as a separate
  time-lagged implicit source under `source_cfl=0.1`.  A factor-three hierarchy also
  rejects an odd periodic graph because its two-color smoother would not be SPD.  The
  source guard still contains physical `c`; it lowers the initial DCI candidate step
  from `1.068285e-6 ns` with the guard disabled to `7.88743694e-7 ns`, about 26 percent.
  Only the transport `c*dt/dx` restriction has been removed.
- Dense-reference focused tests cover constant periodic, harmonic-limited periodic, and
  harmonic-limited vacuum matrices.  Separate comparisons cover the 9-cubed
  three-dimensional hierarchy against Jacobi, the incompatible-block Jacobi fallback,
  and the dense-root allocation cap.  A one-rank/two-rank MPI decomposition regression
  agrees across an ownership boundary; focused CPU material/EOS tests, CPU and CUDA/MPI
  builds, and the exact seven-GPU full-scale probe pass.
- The exact final-source seven-GPU smoke advanced 50 cycles in 28.06 s and restarted for
  10 more cycles in 6.49 s.  It ended at `t=7.389676454491e-05 ns` with `eos_bad=0`.
  Its domain-integrated CH/Au/He fractions were
  `0.07224481712967802/0.9276974187345620/5.776413575991410e-05`, summing to one within
  roundoff.  Its incompatible compact `50 x 32 x 32` MeshBlocks exercise the point-Jacobi
  fallback, not the production hierarchy.  Evidence is recorded in
  `DCI_3D/evidence/smoke_final_20260808.json`; full local artifacts are in the ignored
  `DCI_3D/runs/smoke/` tree.
- CUDA binary `0e3513e15354cf12f17105b417ea30e34f84c003dc9e7f357659d7dbbf80a6c0`
  passed the exact final-source 315-cubed/twenty-group two-cycle probe on seven V100s.
  The monitored run took 25.05 s and peaked at 12620--12622 MiB per GPU.  A separate
  report-enabled repeat of the same binary/input recorded maximum PCG iterations of 67
  and 149 and fresh true residual maxima of `9.782791e-11` and `1.181642e-10`, within the
  guarded recursive-to-true drift allowance.  The compact tracked record is
  `DCI_3D/evidence/radiation_final_probe_20260808.json`; full local monitored-run logs are
  in the ignored `DCI_3D/runs/fullscale_implicit_global_exact_final_20260808/`.
- The restarted smoke endpoint is about 338 times earlier than the historical first
  bad-EOS event at `0.025 ns`; the final full-scale probe is earlier still.  These
  checks do not establish long-time elimination.  Long-horizon behavior, restart
  continuity, chain-energy closure, and sustained production performance require the
  exact-hash DCI evidence gate, and any source or deck change invalidates earlier
  evidence.
