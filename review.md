# Performance code review — laser, thermal radiation, two-temperature, Biermann battery

Date: 2026-08-02
Branch: `2T`, at `733bfbb7` ("perf: reduce tabular Biermann closure overhead")

**Scope.** `src/laser/`, `src/two_temperature/` (incl. `thermal_radiation`, `opacity_table`),
`src/mhd/biermann_battery*.cpp` + `mhd_biermann_subcycle.cpp`, and the `src/materials/`
closure engine those three modules all call into. Orchestration in
`src/driver/driver.cpp` and `src/mhd/mhd_tasks.cpp` is in scope because it is where
most of the cost is scheduled. `src/radiation/` (the GR discrete-ordinates module) is
covered briefly in §2.8 — it is unmodified upstream code and is not instantiated by the
production deck.

**Method.** Source read of the ~11 kloc in the four modules plus the 3.3 kloc materials
backend, cross-checked against the measured 8-rank `nvprof` capture in
`DCI_3D/load_balance_trials/current_profile/` (rank 0, 3 cycles, 23.3 s wall) and the
production deck `DCI_3D/dci_3d.athinput` (512×256×256, 32³ blocks, 1024 blocks / 8 GPUs,
RK2, CH+He tabular EOS, 20-group AP-FLD mixed-table opacity, Biermann subcycling on).

This is a *review*, not an execution plan. `plan_performance.md` (2026-08-01) already
sequences the work; §7 maps findings onto it and flags the ones that are new.

---

## 0. Executive summary

The four modules are, individually, well-engineered numerics. The performance problem is
almost entirely **one algorithm placed in the wrong place in the call graph**, amplified
by an orchestration layer that calls it far more often than the physics requires.

Three sentences:

1. **Algorithm.** The mixed CH/He energy→temperature inversion is a *fixed 48-iteration
   bisection with no convergence test and no early exit*
   (`src/materials/material_mixture.hpp:592`). It is a strictly serial dependency chain of
   ~200 latency-bound transcendentals per component, and it is the innermost loop of every
   hot kernel in the profile.
2. **Architecture.** The Biermann subcycle re-derives that closure — plus a full MPI halo
   exchange — at the *microstep* cadence rather than the macro-step cadence, over a
   ghost-padded domain, gated by two blocking `MPI_Allreduce`s and a host dt readback per
   microstep.
3. **Code detail.** Underneath both, the same transcendental is recomputed several times
   per cell per call (an `exp`/`log` round-trip inside the bisection, a redundant forward
   evaluation after every inverse, 64 unrolled `exp`s in the Planck tail, 4 `log` + 1 `exp`
   per group per opacity lookup).

Measured consequence: **~59% of GPU activity is tabular-EOS closure work, ~13% is
radiation, ~10% is tiny `memcpy`s, ~5% is halo pack/unpack, ~7% is core MHD, ~1.4% is the
Biermann operator's own arithmetic, and ~0.4% is the laser.** The physics the Biermann
subcycle exists to compute costs 1.4% of the run; the bookkeeping around it costs 31%.

**And the device is not the whole story.** GPU activity totals 9.62 s against 23.30 s wall
— **59% of the run is not GPU work at all.** On the host side, 44.85% of CUDA API time
(8.33 s) is spent in 10 058 `cuMemHostRegister`/`cuMemHostUnregister` pairs, and another
45% is the host blocking on the device. See §1.1: the largest single cost in this run is
host-memory pinning, not any kernel. That finding reorders the recommendations, and it is
reflected in §7.

Per-cell cost puts the device side in perspective on an interior-cell basis:

| Kernel | ns/cell | Comment |
|---|---:|---|
| `TwoTemperature::Sync` | 37.7 | up to 4 mixed inverses + floor bisection + 16 sound-speed evals |
| `TwoTemperature::CloseBiermannStage` (reduced) | 24.8 | 1 mixed inverse + 1 redundant forward eval |
| `RefreshMaterialThermodynamics` | 18.1 | 2 mixed inverses + sound speed |
| `mhd_2t_dual_energy_sync` | 12.8 | floor + pressure-partition inverses |
| `ThermalRadiation::Couple` | 9.0 | 20 groups × (opacity + 2 Planck) |
| `MHD::CalculateFluxes` (x1) | 5.6 | **reference: full PLM reconstruction + LLF solve** |
| `MHD::RKUpdate` | 4.8 | **reference: memory-bound update over ~10 variables** |

A single EOS closure costs **5–8× a complete PLM+LLF flux computation on the same cell.**
That ratio is the review's central finding: it is not justified by the physics, and it is
what makes every other cost in the code look small.

---

## 1. Measured baseline (rank 0, 3 cycles, `current_profile`)

Grouped from the `nvprof` log. Percentages are of GPU activity.

| Group | Share | Dominant entries |
|---|---:|---|
| **Tabular EOS closure** | **58.8%** | `CloseBiermannStage` 31.15 (24 calls), `Sync` 14.03 (6), `SynchronizeDualEnergyFromTotal` 4.78 (6), `RefreshMaterialThermodynamics` 3.36 (3), `ApplyDualEnergyFormalism` 3.22 (6), `Exchange` 2.29 (3) |
| **Radiation (AP-FLD)** | **12.9%** | `NewTimeStep` ×4 reducers 8.08 (28), `AddFluxes` ×3 2.66 (18), `Couple` 1.69 (3), `UpdateDiagnostics` 0.43 (30) |
| **Data movement** | **9.6%** | `memcpy HtoD` 4.44 (**45 379 calls**), `memcpy DtoH` 3.49 (**44 256**), `memset` 1.18 (693), `memcpy DtoD` 0.49 (52) |
| **Core MHD** | **7.1%** | `CalculateFluxes` 4.23 (18), `ConsToPrim` 1.64 (30), `RKUpdate` 1.26 (6) |
| **Halo pack/unpack** | **4.9%** | `RecvAndUnpackCC` 3.73 (30), `PackAndSendCC` 0.85 (30), `RecvAndUnpackFC` 0.17, `HydroBCs` 0.15 |
| **One-time init** | **3.9%** | `TwoTemperature::Initialize` 3.51 (1) |
| **Biermann operator** | **1.4%** | AddFluxes/EMF/CT/Poynting ~1.1 (20 each), `NewTimeStep` 0.31 (17) |
| **Laser** | **0.4%** | `TraceStraightRays` 0.35 (**46 calls**, min 1.66 µs, max 3.50 ms) |

Call-count arithmetic that matters:

- `IdealMHD::ConsToPrim` = 30 calls = 6 RK stages + **24 Biermann microstages**.
  **80% of all conservative→primitive conversions and CC halo exchanges are microsteps.**
- 24 Biermann stages / 3 cycles = **4 microsteps per cycle, 2 per Strang half-step.**
- Only *one* `CloseBiermannStage` kernel variant appears in the profile (the reduced
  electron-only branch, `two_temperature.cpp:773`). The reduced-closure optimisation is
  working — and the kernel is *still* number one at 31%.
- `ThermalRadiation::NewTimeStep` x1 reducer (26.2 ms) costs **1.8×** the `AddFluxes` x1
  kernel (14.5 ms) over the identical face set with identical opacity work.
- Startup is over-weighted in a 3-cycle capture (3.9%); steady-state shares are ~4% higher.

Scaling context (from `DCI_3D/load_balance_trials/`): 16 ranks (14.1 s) is *slower* than
8 ranks (10.6 s). The configuration is already communication-bound, which is consistent
with 10% of GPU time in ~90 000 sub-10 µs `memcpy`s.

### 1.1 The host side — and it is bigger than the device side

The kernel table above accounts for **9.62 s of GPU activity** (derived: 2.996 s = 31.15%).
Wall time is **23.30 s**. So **59% of the run is not GPU activity at all.** The CUDA API
breakdown from the same capture explains where it goes:

| CUDA API call | Share of API time | Total | Calls | Avg |
|---|---:|---:|---:|---:|
| `cuMemHostRegister` | **29.98%** | **5.565 s** | 10 058 | 553 µs |
| `cudaEventSynchronize` | 27.88% | 5.175 s | 419 | 12.4 ms |
| `cuMemHostUnregister` | **14.87%** | **2.760 s** | 10 058 | 274 µs |
| `cudaDeviceSynchronize` | 11.91% | 2.211 s | 1 864 | 1.19 ms |
| `cudaStreamSynchronize` | 5.63% | 1.045 s | 1 316 | 794 µs |
| `cuMemcpyAsync` | 1.11% | 206 ms | 88 264 | 2.3 µs |
| `cuEventQuery` | 1.29% | 240 ms | 653 966 | 366 ns |

Two things dominate, and neither is a kernel:

- **Host-memory pinning: 44.85% of API time — 8.33 s in 10 058 register/unregister
  pairs**, ~830 µs per pair, ≈3 350 pairs per cycle. This is the single largest cost in
  the entire run, larger than `CloseBiermannStage`.
- **Host waiting on the device: 45.4% of API time** across `cudaEventSynchronize` (419
  calls, 12.4 ms average, 226 ms maximum), `cudaDeviceSynchronize` (1 864) and
  `cudaStreamSynchronize` (1 316) — the dt readbacks, reduction results, and queue-compaction
  scans catalogued in §3.3. 653 966 `cuEventQuery` calls is the polling loop underneath.

**The mechanism behind the pinning needs one diagnostic step before it is fixed.** The only
explicit pinned allocation in the codebase is `laser.hpp:236-237`
(`SharedHostPinnedSpace`), allocated once — so this is not application code calling
`cudaHostRegister` directly. The two credible sources are (a) UCX/hpcx registering MPI
send/receive buffers for RDMA with its registration cache being defeated — note 221
`cudaMallocAsync` and 277 `cudaFree` calls, which can invalidate such a cache — or (b)
Kokkos staging pageable host buffers for `deep_copy`. The `.nsys-rep` traces already in
`DCI_3D/profile_20260729/` can attribute this; so can re-running with UCX registration-cache
diagnostics enabled.

**This reorders the recommendations.** Enabling `gpu_aware_mpi` and holding persistent host
buffers is not merely "recover part of a 9.6% GPU-time group" — it targets the largest
single cost in the run. Likewise, the host-sync items in §3.3 sit against 45% of API time,
not against the 0.4% the laser kernel occupies.

---

## 2. Perspective 1 — Algorithm

### 2.1 The mixed EOS inverse is the single dominant algorithm, and it is the wrong one

`MixtureComponentFromRhoSpecificEnergyCached` (`material_mixture.hpp:529-604`) inverts
ε(ρ,T) for a two-material mixture with:

```cpp
for (int iteration = 0; iteration < 48; ++iteration) {     // :592
  const Real log_trial = 0.5*(log_low+log_high);
  const Real trial_energy = MixtureComponentEnergyFromCachedDensity(
      component, exp(log_trial), y0, cache, energy_cache);
  if (trial_energy < target_energy) { log_low = log_trial; }
  else                              { log_high = log_trial; }
}
```

Three separate problems, in decreasing order of cost:

- **No convergence test.** 48 iterations always, regardless of how quickly the bracket
  collapses. 48 bisections of a log-temperature bracket spanning ~6 decades gives
  ~2⁻⁴⁸ relative width — far below the table's own interpolation error. The last ~30
  iterations resolve nothing physical.
- **Bisection, not a secant method.** The correct method already exists *in the same
  file*: the exchange solver at `material_mixture.hpp:1183` uses a safeguarded
  regula-falsi that converges in ≤6 iterations with a real tolerance test, falling back to
  bisection only if it stalls (`:1220-1246`). The inverse should use the identical
  structure. Expected 48 → 5–8 evaluations.
- **It is a serial dependency chain.** Each iteration's `exp` depends on the previous
  comparison, so there is zero instruction-level parallelism across the ~200
  transcendentals. This is why `Sync` at 37.7 ns/cell is ~7× a PLM+LLF flux kernel despite
  a FLOP count that does not justify it: the kernel is *latency*-bound, not
  throughput-bound. Cutting the iteration count cuts latency proportionally — this is the
  highest-leverage single change in the codebase.

The same fixed-48 pattern is repeated three more times:
`MinimumPressureEnergyState` (`:1643`, pressure-floor bisection),
`InitialStateFromTotalSpecificEnergy` (`:1536`), `MinimumStateNoSound` (`:1590`).

**Compounding: `Sync` runs up to four of these per cell.** `two_temp_material_sync`
(`two_temperature.cpp:376-455`) calls `MinimumPressureEnergyState` (possibly a 48-step
floor bisection), then `PressureEnergyFromRhoSpecificEnergies` (ion + electron inverse =
2 × 48), then `StateFromRhoSpecificEnergies` (ion + electron again = 2 × 48, plus
`TabularSoundSpeedSquared`). That is ≥400 dependent transcendental evaluations per cell,
per RK stage, over the ghost-inclusive 36³ domain. It explains 14% of the run exactly.

**Recommendation.** Replace the four fixed-48 bisections with the safeguarded secant
already implemented at `:1183`, with a relative tolerance tied to the table's
interpolation error rather than to machine epsilon. This is a roundoff-changing change: it
requires the convergence/conservation gate, not the byte-exact gate.

### 2.2 The Biermann subcycle synchronises globally at the microstep cadence

`Driver::ExecuteBiermannHalfStep` (`driver.cpp:344-431`) tiles each Strang half-interval
with **uniform, globally-minimum SSPRK2 microsteps**. Per microstep:

1. `BiermannSubcycleTimeStepLimit()` → a full-grid reduction + host readback (`:351`);
2. `MPI_Allreduce` on an invalid-limit flag (`:359`);
3. `MPI_Allreduce` MIN on the limit (`:371`);
4. two `ExecuteTaskList("biermann_stage")` (`:407`, `:410`), each a full stage with
   three blocking halo rounds and a ghost-inclusive closure.

Two algorithmic objections:

- **The step size is set by a few extreme cells.** The limit is the electron-drift +
  thermal-magnetic wave speed (`biermann_battery.cpp:1280-1490`), which is spatially
  localised near steep `∇p_e/n_e`. A global MIN forces every one of the 33.5 M cells to
  march at the worst cell's rate. The physically-correct lever is block-local multirate
  substepping with flux correction at block faces. Note that super-time-stepping does
  **not** apply — the limit is hyperbolic, not parabolic; the existing report rejects STS
  correctly.
- **The limit is re-derived every microstep for no benefit.** The drift and
  thermal-magnetic speeds vary on the *macro*-step timescale; recomputing them at
  microstep cadence to 15 digits, then reducing globally and reading back to the host,
  buys nothing. Computing the limit once per half-step (or every N microsteps) and using
  equal substeps removes (N−1)/N of the reductions, readbacks, *and* barriers.
- **Two `Allreduce`s where one suffices.** The invalid-flag MAX and the limit MIN
  (`:359`, `:371`) can be a single reduction — encode "invalid" as a sentinel that MIN
  preserves, or reduce a 2-element buffer with a custom op.

### 2.3 Thermodynamic state is re-derived where it could be propagated

This is the structural reason `CloseBiermannStage` is 31%. After each microstage the code
performs a full halo exchange of the *conserved* state and then **re-inverts the EOS in
the ghost zones** (`mhd_biermann_subcycle.cpp:256-288`) rather than communicating the
already-closed thermodynamic values. The recent reduced-closure work
(`biermann_stage_full_thermodynamics`, driver.cpp:406-411, and the `is-1..ie+1` extent at
`mhd_biermann_subcycle.cpp:268-282`) correctly narrowed *what* and *where*, but the
underlying pattern — halo-exchange conserved state, then redo the expensive nonlinear
closure on both sides of every rank boundary — remains.

Cost of the pattern: the interior 32³ is closed once per stage; the surrounding ghost
shell (34³ − 32³ = 6 536 cells, 17% of the reduced kernel) is closed *redundantly* by both
the owner and the neighbour. Communicating `thermodynamics` (7 components) instead costs
bandwidth but zero inverses.

The trade is not obviously favourable at 7 components; it likely is if the ghost closure
is narrowed to just the 2 fields the Biermann stencil actually reads
(`electron_pressure`, `electron_number_density_cgs` — see
`biermann_battery.cpp:29-37`, `:64-86`). Worth prototyping.

### 2.4 The radiation timestep limiter repeats the flux operator's work — but not redundantly

`ThermalRadiation::NewTimeStep` (`thermal_radiation.cpp:974-1237`) re-derives, per face
per group, the same quantities `AddFluxes` (`:688-845`) computes: the same
`X{1,2,3}FaceMaterialState`, the same `mixed_opacity.Locate`, the same `opacity.Get`, the
same `FLDProperties`. The only new quantity is `FLDFaceStabilityRate`. The cost asymmetry
is large: **8.08% for the limiter versus 2.66% for the fluxes** — the code spends 3× as
long deciding how big the step may be as it does taking it.

**But this is duplicated code, not duplicated computation, and the distinction matters.**
`AddFluxes` runs mid-RK-stage. `NewTimeStep` is reached from `MHD::NewTimeStep`
(`mhd_newdt.cpp:204`) via `MHD::TwoTempExchange` (`mhd_tasks.cpp:676`) — that is, *after*
`Exchange`, `Couple`, and `RefreshMaterialThermodynamics` have run. Those sources change
T_e and the group energies, so the opacity and every FLD face state genuinely differ
between the two calls. `mhd_tasks.cpp:84-90` documents this as deliberate:

> *"Two-temperature exchange and matter-radiation coupling run after the final RK stage and
> can change pressure, opacity, and every source timestep. In that case compute the next
> timestep from the post-source state in TwoTempExchange instead of caching a stale
> pre-source limit here."*

So the two attractive-looking fixes are both invalid:

- **Fusing the `Max` reduction into `AddFluxes`** would reintroduce exactly the stale
  pre-source limit that comment describes removing.
- **Caching the per-face rate during `AddFluxes` and reducing it later** has the same
  defect — the cached rate would be pre-source. (It is also 2.9 GB of arrays, though per
  §3.9 that is a sizing question, not a blocker.)

What is actually available, in order of value:

1. **Make the shared lookup cheaper.** The dominant per-face cost in *both* kernels is the
   geometric opacity interpolation (D6/D7 below: 4 `log` + 1 `exp` per group per material).
   Fixing that once benefits the limiter and the fluxes together, with no correctness
   question at all. This is the primary lever.
2. **Merge the three directional `Max` reductions into one multi-reducer** — the pattern
   already exists at `biermann_battery.cpp:1318-1321`. Removes two launches and two host
   synchronisations per call. Pure win.
3. **Stride the limiter** — recompute the radiation transport dt every N steps with a
   safety margin, since the FLD limit is a smooth function of a smoothly-varying opacity
   field. This is *explicit, monitored* staleness rather than the accidental kind, and is
   defensible where fusion is not — but it needs the convergence gate and must hold
   `causal_timestep_no_collapse`.

### 2.5 The Planck series is O(64) where it should be O(1)+early-exit

`PlanckIntegral` (`thermal_radiation.cpp:54-77`) evaluates the complementary series with a
**fixed 64-term loop and a fresh `exp(-n·x)` per term**:

```cpp
for (int n = 1; n <= 64; ++n) {
  Real term = exp(-rn*x)*(...);      // :71  — 64 independent exp() calls
  tail += term;
}
```

Two independent inefficiencies:

- `exp(-n·x) = q^n` with `q = exp(-x)`. A single `exp` plus a running multiply replaces
  64 `exp`s — algebraically identical, ~64× fewer transcendentals in the tail branch.
- No early exit. Terms decay as e^(−nx); for x ≥ 0.5 the series is converged to double
  precision by n ≈ 10–15 in the worst case and by n ≈ 3 for typical x. A
  `if (term < eps*tail) break;` removes most of the loop. Because x is spatially smooth,
  warp divergence from the early exit is small.

**And the caller doubles the work.** `PlanckGroupFraction` (`:80-86`) evaluates *both*
group boundaries, so a 20-group cell performs 40 `PlanckIntegral` calls where 21 suffice —
every interior boundary is computed twice. The rolling-bound version is already written
correctly in the same file, in the source dt reducer:

```cpp
Real lower_planck = PlanckIntegral(bounds(0)/tele);        // :1203
for (int g = 0; g < ng; ++g) {
  Real upper_planck = PlanckIntegral(bounds(g+1)/tele);    // :1214
  ...
  lower_planck = upper_planck;                             // :1217
}
```

`Couple` (`:910`) and `Initialize` do not use it. Having two different implementations of
the same quantity in one file is both a performance bug and a maintenance hazard.

> **Scaling caveat worth flagging.** `PlanckIntegral`'s cost is *state-dependent*: it
> returns early for x ≤ 0 and x ≥ 50, uses a cheap polynomial for x < 0.5, and only enters
> the 64-term loop in between. In the 3-cycle profile the domain is still cold, so most
> group boundaries land in the x ≥ 50 constant branch. **As the plasma heats, more cells
> enter the expensive branch, so `Couple`'s 1.69% is a floor, not a steady-state value.**
> A profile at a representative later-time restart will look materially different here.

### 2.6 The laser algorithm is sound; its per-segment recomputation is not

The DDA ray march, the SoA ray state, the wave-based distributed transport with a
work cap, and the exact `expm1` attenuation (`laser_physics.hpp:71-76`) are all
well-chosen. The algorithmic issue is that quantities which are **per-cell constants
within a frozen stage** are recomputed **per ray per segment**:

- the inverse-bremsstrahlung coefficient (`laser_trace.cpp:796-843`), which costs a
  `pow(kT,1.5)`, a `log`, and two `sqrt` per segment (`laser_physics.hpp:39-66`);
- the critical-reflection electron-density gradient, a 6-point stencil rebuilt per segment
  (`laser_trace.cpp:683-705` and the `multi_d`/`three_d` blocks following).

With N_ray·N_segment ≫ N_cell in the illuminated region, precomputing a per-cell IB
coefficient and a per-cell ∇n_e once per stage converts O(rays × segments)
transcendentals into O(cells). The laser is only 0.4% of GPU time so the *kernel* payoff
is small — but see §3.8: the launch/sync overhead around it is much larger than the kernel,
and fewer, cheaper segments shortens the whole state machine.

Also algorithmically wasteful: `FindLocalBlock` / `FindGlobalBlock`
(`laser_trace.cpp:49-73`) are **O(N_blocks) linear scans** executed inside the ray kernel
on every block crossing — 1024 iterations worst case on the production mesh. The DCI mesh
is static, uniform and single-level; block lookup is O(1) index arithmetic.

### 2.7 What is algorithmically right — do not regress these

Worth stating explicitly so optimisation does not damage them:

- **The log-mean edge integral** (`biermann_battery.cpp:825-836`, `InverseLogMean` at
  `:126-139`) is the correct path-conservative discretisation: it telescopes exactly for
  constant n_e and reduces to the Graziani shock integral for an isothermal jump, with a
  cancellation-safe series for nearly-equal states. It is also *cheap*.
- **Keeping ion energy out of the SSPRK recurrence** (`mhd_biermann_subcycle.cpp:94-101`)
  is correct: blending a projected ion energy as a zero-RHS variable would silently drop
  the split update to first order. Any fusion of the closure into the update must preserve
  this.
- **The FLD stability estimate from the true flux Jacobian** rather than the optically-thick
  bound (`thermal_radiation.cpp:964-972`) is the right call — it avoids a needlessly small
  dt in streaming regions. The cost problem in §2.4 is duplication, not the estimate.
- **The reduced Biermann closure** (`biermann_stage_full_thermodynamics`) and the
  `is-1..ie+1` extent restriction are correct and already landed; the analysis of what the
  stencil actually reads (`mhd_biermann_subcycle.cpp:262-268`) is sound.
- **The IONMIX interval-energy cache** (`ionmix_two_temperature_table.hpp:413-444`) is a
  well-executed optimisation: it makes a repeat bisection probe within the same
  temperature interval cost one `exp` instead of a full 2-D interpolation.

### 2.8 `src/radiation/` (GR discrete ordinates)

Not instantiated by the production deck (`meshblock_pack.cpp:174` requires a `<radiation>`
block; `dci_3d.athinput` has none) and unmodified from upstream on this branch. Two
observations from a read, for completeness only:

- `RadiationNudt` (`radiation_newdt.cpp:62-90`) carries a nested angle × neighbour loop
  inside the reduction; on a lat-long geodesic grid this is O(n_ang · n_neighbour) per
  cell, which is the standard hot spot for this module.
- `radiation_source.cpp` uses repeated `for (n=0..nang1)` sweeps (`:192, 235, 298, 332,
  380`) over the same angular data; these are fusable if this module ever enters the
  production path.

Neither is actionable now.

---

## 3. Perspective 2 — Implementation architecture

### 3.1 Layering — the cache-based design is right

`TwoTemperature` acting as the *owner of a materialised thermodynamic cache*
(`temperature`, `thermodynamics`, `two_temperature.hpp:58-59`) that Biermann, laser, and
radiation all read is the correct architecture. It is what lets
`biermann_battery.cpp:29-37` (`ElectronPressure`), `laser_trace.cpp:100-116`
(`MaterialElectronDensityCgs`), and the FLD face states read pre-inverted state instead of
querying IONMIX per use. Without it the laser alone would be untenable.

The `eos_query_flags` bitfield accumulated by OR into the cache
(`two_temperature.cpp:76-79`) is a nice touch — lifetime diagnostics with no extra pass.

The problem is not the design; it is that **the cache is invalidated and fully rebuilt far
more often than the physics changes it.** Per cycle the closure is materialised at least:
2× in `Sync`, 8× in `CloseBiermannStage`, 1× in `RefreshMaterialThermodynamics`, 1× in
`Exchange`, plus the two dual-energy kernels. Ten-plus full rebuilds per cycle of a
quantity whose inputs change by O(dt).

### 3.2 The Biermann stage reuses the macro-step communication pattern

`mhd_tasks.cpp:102-152` builds `biermann_stage` as a faithful miniature of the macro RK
stage, which is good for correctness and bad for throughput. Each microstage executes
**three strictly serialised blocking communication rounds**:

```
BiermannEField → SendE  → RecvE            (round 1: CT edge fields)
              → BiermannCompositeEnergyFlux
              → SendFlux/RecvFlux          (multilevel only — skipped here)
BiermannRKUpdate → BiermannCT
              → RestrictU → SendU → RecvU  (round 2: conserved state)
              → RestrictB → SendB → RecvB  (round 3: face B)
              → ApplyPhysicalBCs → Prolongate → BiermannConToPrim
```

Rounds 2 and 3 are serialised only because the chain is written linearly: `RestrictB`
(`:136`) depends on `RecvU` (`:134`), though B's restriction does not need U's halo. On
the macro list this ordering is inherited from upstream and amortised over a much larger
stage; at microstep cadence it is paid 8× per cycle. **Overlapping the U and B exchanges
would remove one full latency round per microstage, 8 per cycle.**

Total per cycle at 4 microsteps: 8 microstages × 3 rounds = **24 blocking halo rounds for
the Biermann operator versus 6 for all of MHD.** This is the direct cause of
`RecvAndUnpackCC` showing 30 calls (24 of them Biermann) at 3.73%, and it is the primary
suspect for the 16-rank regression.

### 3.3 Host-in-the-loop control flow

Three separate places put the host on the critical path at sub-step cadence:

| Site | Per-cycle syncs (measured) | Cause |
|---|---:|---|
| `driver.cpp:351,359,371` | 4 × (1 readback + 2 `Allreduce`) | Biermann dt limit |
| `laser_trace.cpp:1061` (`CompactActiveQueue`) | ~15 | `parallel_scan` returns a host scalar |
| `laser_mpi.cpp:38,50,324` | per MPI wave | count mirror + `CountActiveRays` |

The laser case deserves care in how it is stated. The *bound* is severe —
`TraceStraightRays` loops up to `max_transport_iterations = 256` (deck line 186), and
every iteration ends with `CompactActiveQueue` returning the scan total to the host to
decide whether to break. The *measured* count is modest: 46 `TraceStraightRays` launches
over 3 cycles ≈ 15 per cycle, so the loop terminates after ~8 iterations per RK stage
summed over waves. But the profile also reports a **1.66 µs minimum kernel time against a
3.50 ms maximum** — the tail iterations do essentially no work, and the loop is dominated
by launch plus a full device fence inside a task that is simultaneously driving
non-blocking MPI. That is exactly the shape that produces the launch/sync gaps noted in
the earlier report.

The fix pattern is the same in all three sites: keep the termination condition on-device
(a device-side counter, or check the host copy only every K iterations), and accept an
occasional wasted launch instead of a guaranteed sync.

### 3.4 Kernel granularity — dt limiters are split four ways

`ThermalRadiation::NewTimeStep` issues four separate `parallel_reduce`s (x1, x2, x3
`Max`, plus a source `Min`) — `thermal_radiation.cpp:1020, 1072, 1123, 1180` — each with
its own launch, its own reduction tree, and its own host-visible result. Kokkos supports
multi-reducers (the code already uses one in `BiermannBattery::NewTimeStep`,
`biermann_battery.cpp:1318-1321`, with three `Min` reducers in one launch). Applying the
same pattern here removes three launches and three syncs per call, 7 calls per 3 cycles.

Similar granularity issue on the write side: `BiermannFluxes` and `BiermannEField` zero
their targets with six `Kokkos::deep_copy`s per stage
(`mhd_biermann_subcycle.cpp:67-69, 149-151`) purely because `AddFluxes`/`AddEMFs` use
`+=`. Those six memsets × 8 microstages = 48 per cycle, and the profile shows 693 memsets
at 1.18%. Having the first writer use `=` on first touch eliminates them at zero risk.

### 3.5 Data movement — 90 000 transfers on the device, 10 058 pin/unpin pairs on the host

45 379 HtoD + 44 256 DtoH in 3 cycles ≈ **30 000 transfers per cycle**, averaging 9.4 µs
and 7.6 µs — i.e. dominated by fixed per-transfer overhead, not bandwidth. Sources:

- **Non-GPU-aware MPI staging.** The deck sets `gpu_aware_mpi = false` (line 188). Every
  halo packet round-trips through host memory. With 24 Biermann + 6 MHD halo rounds per
  cycle × 128 blocks × up to 26 neighbours, the count is in the right range.
- **Per-microstep dt readback** (`driver.cpp:351`).
- **Per-call host mirror allocation in the laser.** `PrepareOutgoingRays`
  (`laser_mpi.cpp:37-38, 46-50`) calls `Kokkos::create_mirror_view` and `deep_copy` on
  every invocation rather than holding persistent mirrors. `mpi_host_send_packets_` /
  `mpi_host_recv_packets_` are already persistent pinned views
  (`laser.hpp:236-237`) — the count/offset mirrors should be too.

But per §1.1 the device-side transfers are the *cheap* half of this. The 88 264
`cuMemcpyAsync` calls cost 206 ms of API time; the 10 058 `cuMemHostRegister` /
`cuMemHostUnregister` pairs that accompany them cost **8.33 s**. Whatever is causing host
buffers to be pinned and unpinned ~3 350 times per cycle is the most valuable single thing
to find in this codebase, and it is very likely the same root cause as
`gpu_aware_mpi = false`: staging halo packets through pageable host memory that the MPI
layer must then register for RDMA.

Qualifying and enabling `gpu_aware_mpi = true` is therefore the single largest lever in the
review. It is a pure data-movement change (byte-exact-gateable) and it benefits both the
bvals halo and the laser packet path (`laser_mpi.cpp:258-269, 283-284` already branch on
the flag). Pair it with persistent host buffers so that whatever registration does remain
is amortised rather than repeated.

### 3.6 Domain extent is an architectural parameter, and it is under-exploited

Several kernels run over the ghost-inclusive `0..n1m1` (36³ = 46 656 cells) when the
consumer needs less:

| Kernel | Extent | Cells | vs interior |
|---|---|---:|---:|
| `Sync` (`mhd_tasks.cpp:648`) | `0..n1m1` | 46 656 | 1.42× |
| `Exchange`, `Couple`, `Refresh` (`mhd_tasks.cpp:669-677`) | `0..n1m1` | 46 656 | 1.42× |
| `CloseBiermannStage` reduced (`mhd_biermann_subcycle.cpp:271-282`) | `is-1..ie+1` | 39 304 | 1.20× |
| `SynchronizeDualEnergyFromTotal` (`mhd_dual_energy.cpp:170-175`) | ±ng | 46 656 | 1.42× |

The Biermann path has already been narrowed correctly. The operator-split sources
(`Exchange`, `Couple`, `RefreshMaterialThermodynamics`) are run ghost-inclusive so that the
next stage's reconstruction sees consistent ghosts — but the halo exchange that follows
would supply exactly those values. Each of these is a 1.42× multiplier on a kernel that is
already the most expensive thing in the code; auditing which genuinely need ghosts is
cheap analysis with a direct 30% return on the affected kernels.

### 3.7 Load balance and scaling

`DCI_3D/profile_20260729/.../material_work_imbalance/DIAGNOSIS.md` records ranks 0–3
owning 3.8× more mixed-material cells than ranks 4–7, producing a 2.1× `Sync` spread. The
deck's `x1_rank_map` (athinput lines 34-44) is a deliberate, laser-aware static
partition — and the report is explicit that a material-only repartition was tried and
**rejected** because it raised laser communication faces 12.8% and slowed sustained runs
3%.

The architectural point: **any future balancing must have laser ray residence in the
objective**, not just material work. And because §2.1 shrinks the material cost by a large
factor, the imbalance's absolute weight drops with it — re-measure before investing here.

The 16-rank > 8-rank wall time is the more urgent scaling signal, and §3.2/§3.5 are the
likely causes (24 blocking halo rounds per cycle, host-staged packets), not the imbalance.

### 3.8 Laser transport — right structure, wrong launch dimension

The state machine (`LaserTransportState`, `laser.hpp:38-50`; driven by
`AdvanceDistributedTransport`, `laser_mpi.cpp:155-363`) is a genuinely good design:
non-blocking `Ialltoall` for counts, `Irecv`/`Isend` for packets, `Iallreduce` for global
completion, and the task returns `TaskStatus::incomplete` so the scheduler can interleave
other work. Ray identity is preserved globally so per-ray diagnostics survive migration.

But the queue compaction does not deliver its main benefit:

```cpp
int Laser::CompactActiveQueue(...)                    // :486 — builds a dense active list
...
Kokkos::parallel_for("laser_trace_dda",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),  // :627 — launches ALL 4096 rays
    KOKKOS_LAMBDA(int queue_index) {
      int r = current(queue_index);
      if (r < 0 || status(r) != active) return;       // :630 — most threads exit here
```

The compacted queue is built, then the kernel is launched over the *full* `nrays_` anyway.
Launching over the returned active count would make the last iterations of the DDA loop
genuinely cheap instead of merely idle. (The count is already read to the host at `:1061`,
so it is available at zero extra cost.) The same applies to `PrepareOutgoingRays` and
`BookRemainingRays`, both `RangePolicy(0, nrays_)`.

### 3.9 Memory footprint

Per rank (128 blocks × 36³ cells × 8 B = **47.8 MB per scalar component**):

| Array | Components | Size/rank |
|---|---:|---:|
| `Laser::cell_data` (`laser.cpp:528`) | 12 | **573 MB** |
| `TwoTemperature::thermodynamics` | 7 | 334 MB |
| `ThermalRadiation` groups in `u0`+`w0` | 20 × 2 | ≥1.9 GB (more with RK registers and flux arrays) |
| `TwoTemperature::temperature` | 2 | 96 MB |
| `Laser::cumulative_energy_start_` | 1 | 48 MB |

`cell_data` is the largest single auxiliary array in the code, and **10 of its 12
components are pure diagnostics** (segment count, tau, path, direction×path,
dispersion×path, midpoint×path — see `laser.hpp:140-142`). Production needs components 0
and 1. `report_diagnostics_` exists as a member (`laser.hpp:206`) but does not gate either
the allocation or the atomic writes. Gating both would free ~478 MB/rank and remove 8 of
the ~11 atomics per segment (§4).

Note also that `ClearInstantaneousData` (`laser_tasks.cpp:71-78`) rewrites all 12
components over the interior every stage — ~400 MB of writes per stage for data that is
mostly discarded.

**How much this matters.** The target machine is 8× V100-16GB, the measured peak is
**11 876 MiB = 72.49% of 16 384 MiB** (`PROFILE_REPORT.md:691`), and the production gate
`gpu_memory_60_80_all` requires every GPU to land in 60–80%.

That gate is a **sizing check, not a design constraint**: `DCI_3D/README.md` states the
remedy directly — *"Tune the uniform mesh and repeat if it misses."* The mesh is the free
parameter, so an optimisation that needs more memory is acceptable; it just shifts the
calibrated resolution. Two practical consequences rather than prohibitions:

- Report the memory delta for any card that allocates, so the mesh can be retuned once at
  the end rather than drifting out of band unnoticed.
- Retuning the mesh **invalidates any frozen-reference comparison** — a byte-exact or
  convergence gate cannot span a resolution change. Batch memory-growing changes so the
  reference is rebuilt once.

Freeing the ~478 MB of laser diagnostics is therefore worth doing for its own sake — fewer
atomics per segment and ~400 MB/stage less clearing traffic — not because headroom is
scarce.

---

## 4. Perspective 3 — Code details

Concrete, local items. "Risk" is the validation gate required: **BE** = byte-exact
achievable, **CG** = convergence gate needed (perturbs roundoff).

| # | Location | Issue | Fix | Gain | Risk |
|---|---|---|---|---|---|
| D1 | `material_mixture.hpp:592-601` + `:288` | The bisection computes `exp(log_trial)` to get T, and `MixtureComponentEnergyFromCachedDensity` immediately does `log(temperature)` to get `log_trial` back. A comment states this exists to preserve "the legacy exp-then-log inverse trajectory". | Pass `log_trial` through directly. | 2 transcendentals × 48 iters × 2 components/cell, in the #1 and #2 kernels | CG |
| D2 | `material_mixture.hpp:1389-1392` | `ElectronStateFromRhoSpecificEnergy` runs the inverse, then calls `MixtureComponentFromRhoTemperature` at the returned temperature — a full **uncached** forward evaluation (re-locates density in both tables). The inverse already returns pressure and energy at that temperature. | Return the inverse's own state; if a forward pass is required for flag fidelity, use the cached variant. | one extra 2-material forward eval + 2 density locates per cell, **inside the 31% kernel** | CG |
| D3 | `thermal_radiation.cpp:67-74` | 64 independent `exp(-n*x)` calls per `PlanckIntegral`. | `q = exp(-x); qn *= q;` — one `exp` total. | ~64× fewer transcendentals in the tail branch | CG |
| D4 | `thermal_radiation.cpp:66-74` | No early exit; terms decay as e^(−nx). | `if (term < eps*tail) break;` | typically 64 → <10 iterations | CG |
| D5 | `thermal_radiation.cpp:80-86` vs `:1203-1217` | `PlanckGroupFraction` evaluates both boundaries (40 calls/cell for 20 groups); the rolling form (21 calls) is already implemented in the dt reducer but not used by `Couple` (`:910`) or `Initialize`. | Use the rolling bound in `Couple`/`Initialize`. | ~2× fewer Planck evals in coupling | CG |
| D6 | `opacity_table.hpp:118-122` | Geometric interpolation recomputes **4 `log` + 1 `exp` per group per lookup**. With `mixed-table` × 20 groups × 2 kinds that is ~320 `log` + 80 `exp` per cell in `Couple`. The IONMIX table already pre-stores `log_values` for exactly this reason (`ionmix_two_temperature_table.hpp:288-290`). | Pre-store `log(value)` at load with a zero-safe sentinel; runtime becomes linear-in-log + one `exp`. | large, on all FLD kernels (12.9% group) | BE or CG |
| D7 | `opacity_table.hpp:73-78, 98-103` | `Locate` with `log_coordinates` does `log(d)`, `log(density(lower))`, `log(density(upper))` — 3 `log` per axis, 6 per material, 12 per mixed lookup, on axis values that are compile-time constant per table. | Store the axis in log space at load. | 12 `log`/cell/lookup | BE |
| D8 | `biermann_battery.cpp:1351, 1367, 1372, 1390, ...` | `const Real log_ne = log(ne);` for the *centre* cell is recomputed once per direction inside `biermann_newdt` (x1, x2, x3 blocks each redeclare it). Neighbour `log(ne)` values are recomputed by each adjacent cell. | Hoist the centre `log_ne`; ideally cache `log(ne)` in a scratch pass and read 7 values. | ~3 redundant `log`/cell/microstep, ×12 microsteps/cycle | BE (hoist) |
| D9 | `mhd_biermann_subcycle.cpp:67-69, 149-151` | Six `deep_copy(…, 0.0)` per microstage solely because `AddFluxes`/`AddEMFs` use `+=`. | First writer uses `=`. | 48 memsets/cycle; part of the 1.18% memset total | BE |
| D10 | `laser_trace.cpp:627` | DDA kernel launches `RangePolicy(0, nrays_)` = 4096 threads despite the compacted queue; in the tail iterations nearly every thread exits at `:630` (profile min kernel time 1.66 µs vs max 3.50 ms). | Launch over the active count already returned at `:1061`. | collapses the near-empty tail launches | BE |
| D11 | `laser_trace.cpp:858-876` | 9–11 `atomic_add`s to `cell_data` per segment; 8 of them are diagnostics. `report_diagnostics_` (`laser.hpp:206`) does not gate them. | Gate diagnostics 2–11 behind the flag. | ~3× fewer atomics/segment + 465 MB/rank (§3.9) | BE |
| D12 | `laser_trace.cpp:49-73` | `FindLocalBlock`/`FindGlobalBlock` are O(N_blocks) linear scans in the ray kernel (1024 blocks on this mesh), executed on every block crossing. | O(1) index arithmetic for the static uniform single-level mesh; keep the scan as a fallback. | worst-case 1024→1 per crossing | BE |
| D13 | `laser.cpp:679-720` + `laser_tasks.cpp:52` | `RefreshGlobalBlockInfo` rebuilds all `nmb_total` block descriptors on the host and copies H2D **every stage**, on a static mesh. | Build once; rebuild only on remesh. | 1 H2D + O(N_blocks) host work per stage | BE |
| D14 | `laser_mpi.cpp:37-38, 46-50` | `create_mirror_view` + `deep_copy` for send counts/offsets on **every** call (every MPI wave). | Hold persistent mirrors, as already done for the packet buffers (`laser.hpp:236-237`). | allocations + 2 small copies per wave | BE |
| D15 | `two_temperature.cpp:46-49` | `pow(thermal_speed_squared, 1.5)` in `SpitzerExchangeTime`; `pow(electron_charge_cgs, 4)` and `8.0*sqrt(2.0*pi)` recomputed from constants. | `v*sqrt(v)`; fold the constants into a single `constexpr`. | small but free (the constants may already be folded; the 1.5 power is not) | CG (pow) / BE (constants) |
| D16 | `material_mixture.hpp:803-866` | `TabularSoundSpeedSquared` performs **16 full mixed forward evaluations** per cell (2 components × 4 finite-difference points × 2 materials), re-`Locate`-ing density each time although only 3 distinct densities occur. | Pass a shared `MixedDensityCache`, as the inverse already does. | roughly halves the sound-speed table traffic in `Sync`/`Refresh`/`Initialize` | BE or CG |
| D17 | `material_mixture.hpp:1643-1652` | The floor bisection calls `log(temperature)` on a value it just produced as `exp(0.5*(log_low+log_high))` — same round-trip as D1. | Track the log directly. | 1–2 transcendentals × 48 iters, in `Sync`/`Exchange`/`Couple`/dual-energy | CG |
| D18 | `thermal_radiation.cpp:1020, 1072, 1123, 1180` | Four separate reductions where a multi-reducer would do (the pattern is already used at `biermann_battery.cpp:1318`). | Merge the three directional `Max` reductions. | 3 launches + 3 host syncs per call | BE |
| D19 | `mhd_tasks.cpp:136` | `RestrictB` depends on `RecvU`, serialising the B halo behind the U halo though they are independent. | Branch B's restriction from `BiermannCT` directly and let the scheduler overlap. | one latency round × 8 microstages/cycle | BE |
| D20 | `driver.cpp:359, 371` | Two `MPI_Allreduce` per microstep (invalid-flag MAX, then limit MIN). | Fold into one reduction with a sentinel encoding. | 1 barrier × 4 microsteps/cycle | BE |

---

## 5. Cross-cutting observation: the validation gate is now the constraint

`PROFILE_REPORT.md` documents a disciplined campaign with a **byte-exact frozen-field
gate**, and that gate is why several of the above items are still open — D1, D3, D17 in
particular exist *specifically* to preserve bit-identical arithmetic trajectories
("preserving the legacy exp-then-log inverse trajectory", `material_mixture.hpp:286-288`).

That gate has largely exhausted its headroom. Every remaining large win — the secant
inverse, the Planck recurrence, fewer microsteps, block-multirate — is by construction a
roundoff-perturbing change and *cannot* pass bit-identity. These are not bugs; they are
algorithm changes.

**Recommendation:** stand up a convergence / conservation / physics-equivalence gate
against a frozen later-time restart (energy accounting, `div B`, laser conservation
counters, timestep-limiter sequence, refinement convergence) *before* the next round of
optimisation, and retain the byte-exact gate only for the pure data-movement items
(D7, D9, D10, D11, D12, D13, D14, D18, D19, D20, `gpu_aware_mpi`). Budget this up front —
it gates roughly 80% of the available speedup.

A second methodological note: the current profile is 3 cycles from a cold start. It
over-weights one-time initialisation (3.9%) and, per §2.5, *under*-weights the Planck path,
whose cost grows as the plasma heats. Re-profile at a representative later-time restart
before ranking radiation work against EOS work.

---

## 6. Findings not already covered by `plan_performance.md`

The plan (2026-08-01) is accurate and its priorities hold. These items appear to be new:

0. **§1.1 — the run is host-bound, and the largest single cost is host-memory pinning.**
   GPU activity is 9.62 s of 23.30 s wall; 44.85% of CUDA API time (8.33 s) goes to 10 058
   `cuMemHostRegister`/`cuMemHostUnregister` pairs and another 45% to host-side waiting.
   The plan's A4 anticipated the *symptom* (~8% memcpy time) but not that the accompanying
   pin/unpin traffic costs an order of magnitude more, nor that it outranks every kernel.
1. **D1 / D17 — the `exp`↔`log` round-trip inside both bisection loops.** Two wasted
   transcendentals per iteration, in the two hottest kernels, kept only for bit-exactness.
2. **D2 — the redundant uncached forward evaluation after every electron inverse**
   (`material_mixture.hpp:1389-1392`), inside the 31% kernel.
3. **D3 — `PlanckIntegral` recomputes `exp(-n·x)` 64 times** instead of using a `q^n`
   recurrence. The plan noted the missing early exit (D4) but not the recurrence, which is
   the larger factor.
4. **§2.5 scaling caveat — Planck cost is state-dependent and undersampled** by a cold
   3-cycle profile.
5. **D10 — the laser DDA kernel launches `nrays_` threads despite compacting the queue**,
   so the compaction's main benefit is unrealised.
6. **§3.3 — `CompactActiveQueue` forces a full device fence per DDA iteration** by
   returning the scan total to the host. Measured at ~15 fences per cycle, but each sits
   inside a task that is concurrently driving non-blocking MPI, and the 1.66 µs minimum
   kernel time shows the tail iterations are pure launch+sync. The plan attributed laser
   cost to "launch/sync gaps" without identifying this as the mechanism.
7. **§2.4 quantified — the radiation dt limiter costs 3× the flux operator** (8.08% vs
   2.66%) for the same faces and the same opacity work.
8. **D19 — `RestrictB` is needlessly serialised behind `RecvU`** in the Biermann stage
   list, costing one latency round per microstage.
9. **D8 — `log(ne)` for the centre cell is recomputed once per direction** in
   `biermann_newdt`.
10. **§3.9 — `Laser::cell_data` is 559 MB/rank, 10/12 of it diagnostics**, and
    `report_diagnostics_` gates neither the allocation nor the atomic writes.
11. **§2.1 quantified — `Sync` performs up to four independent 48-step inverses per cell**
    plus a floor bisection plus 16 sound-speed evaluations; ≥400 dependent transcendentals.
12. **§2.1 framing — the inverse is latency-bound, not throughput-bound.** The 48-iteration
    serial dependency chain, not the FLOP count, is why the closure costs 5–8× a PLM+LLF
    flux kernel per cell. This is the reason to expect the secant change to over-deliver
    relative to a naive operation count.

---

## 7. Prioritised summary

Ranked by (expected gain) × (confidence) ÷ (risk). Items map to `plan_performance.md`
initiatives where they exist.

| Rank | Item | Est. impact | Gate | Plan ref |
|---:|---|---|---|---|
| 0 | **Diagnose and eliminate the 10 058 host pin/unpin pairs (§1.1, §3.5)** — start with `gpu_aware_mpi=true` + persistent host buffers | **8.33 s of 18.6 s API time; largest single cost in the run** | BE | A4 |
| 1 | Safeguarded secant replacing all four fixed-48 bisections (§2.1), incl. D1/D17 | Large — hits every kernel in the 58.8% GPU group | CG | A1 |
| 2 | Cut host↔device synchronisation (§3.3): amortise the Biermann dt limit, fold the two `Allreduce`s, on-device laser termination | Against 45% of API time in `cudaEventSynchronize`/`DeviceSynchronize`/`StreamSynchronize` | BE/tight | A3, A5, E1 |
| 3 | Opacity `log`-value pre-storage + log-space axes (D6, D7) | Large fraction of the 12.9% radiation group | BE/CG | C4 |
| 4 | Radiation limiter (§2.4): merge the 3 `Max` reducers (D18), then stride it. **Not fusible into `AddFluxes` — it legitimately runs on post-source state** | launches + syncs; up to ~4% if strided | BE / CG | C5 |
| 5 | D2 redundant forward eval; D16 sound-speed density cache | Direct cuts to the 31% and 14% kernels | CG / BE | (new) / C2 |
| 6 | Planck recurrence + early exit + rolling bound (D3, D4, D5) | ~2–10× on Planck; grows with plasma temperature | CG | C3 (+new) |
| 7 | Ghost-extent audit for `Exchange`/`Couple`/`Refresh` (§3.6) | up to 30% off three kernels | BE if extents provably unread | A2 |
| 8 | Overlap U/B halos in the Biermann stage (D19); drop the 6 memsets (D9) | 8 latency rounds + 48 memsets per cycle | BE | (new) / A3b |
| 9 | Laser: active-count launch (D10), gate diagnostics (D11), O(1) block lookup (D12), one-time block info (D13) | small GPU %, large launch/sync % + 478 MB/rank | BE | E1 (+new) |
| 10 | Block-local multirate Biermann substepping (§2.2) | Largest remaining structural win | Full CG + `div B`/conservation study | A3 |
| 11 | Re-measure load imbalance *after* items 1–8 (§3.7) | TBD — absolute weight falls with the material cost | — | D1 |

**Do not revisit** (profiled and rejected in `PROFILE_REPORT.md`): fused `MHD::RKUpdate`;
rolling-Planck coupling *as a standalone change*; static communication-aware composition
partition; scalar density continuation; Exchange-wide density-cache reuse; strict
mixed-inverse invariant hoisting; normal-face shock-mask relaxation. Super-time-stepping
does not apply to the Biermann limit (hyperbolic, not diffusive).

**Config note, no code change:** `biermann_shock_suppression = true` (athinput line 84) is
effectively inert on the subcycle path — `AddFluxes` skips `ComputeShockMask` when
subcycling (`biermann_battery.cpp:354-355`) and suppression is instead the smooth
`activation` smoothstep computed in `CachedElectronDensityCode`
(`biermann_battery.cpp:64-86`) and folded into the vertex pressure coordinate
(`:947-953` for the 3-D production path). It is a correctness/config flag, not a
performance lever, and costs nothing.
