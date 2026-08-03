# DCI_3D performance ledger

The baseline profile and measurement protocol are defined in
[`../plan_performance.md`](../plan_performance.md). Source changes are accepted only after
the validation gate and A/B measurements named by the corresponding task card pass.

> ## READ THIS FIRST — the wall-clock numbers below the summary table are wrong
>
> Every per-card measurement in this ledger was taken over **3 cycles**, per the plan's
> §4.1 protocol. That window turned out to sit entirely inside a **warm-up transient**:
> with the production transport settings the first cycle costs ~10.1 s and the run does
> not settle until about cycle 11, at 2.54 s/cycle. A 3-cycle A/B therefore measures the
> transient, not the run.
>
> Consequences, all confirmed by 20-cycle re-measurement (see **Steady-state correction**
> at the end of this file):
>
> * **A4 and A4b are wrong and have been reverted.** `UCX_RNDV_THRESH` was tuned to
>   flatten the warm-up. It does that, and it leaves the steady state permanently ~10%
>   *slower*. `MPI_TRANSPORT_ENV` is now empty again.
> * **The −64% headline was an artefact.** The honest figure for the accepted source
>   changes is **−20.8% at steady state**, measured at the production transport setting.
> * The *sign* of every source-change decision still holds — those A/Bs compared binaries
>   under identical conditions — but the percentages in the individual card entries are
>   not steady-state numbers and should not be quoted.
>
> The per-card entries are left as written, with their original numbers, so the mistake
> and its correction both stay on the record.

| card id | commit | gate used | before/after kernel % | before/after wall | memory % | verdict |
|---|---|---|---|---|---|---|
| A0 | diagnosis on `733bfbb7` plus the current working tree | measurement only | N/A | N/A | 72.49 baseline | accepted diagnosis |
| A4 | reverted | BE | warm-up only | 3-cycle: −28.5%; **steady state: +15% slower** | unchanged | **REJECTED + reverted** (see correction) |
| E2 | — | — | N/A | N/A | N/A | **not applicable to the production deck** |
| A5+C4b+E1+E4+B2 | working tree | BE (bundle) | see below | 5.30–5.34 → 5.21–5.23 s/cycle (−1.7%) | unchanged | **accepted** |
| T0 | working tree, `DCI_3D/convergence_gate.py` | N/A (builds the gate) | N/A | N/A | unchanged | **accepted** |
| A1 + A1b | — | CG (passed) | none measurable | 5.22–5.25 → 5.25–5.30 s/cycle | unchanged | **rejected + reverted** |
| A2b + C4a | working tree | BE (bundle) | see below | 5.22 → 5.12–5.13 s/cycle (−1.8%) | +2 opacity tables (small) | **accepted** |
| C3 | working tree | CG (measured byte-exact at this state) | see below | 5.12 → 5.07–5.08 s/cycle (−0.9%) | unchanged | **accepted** |
| C2 | — | BE (not reached) | N/A | 5.09 → 5.12–5.13 s/cycle | unchanged | **rejected + reverted** |
| A6 | working tree | BE | none measurable | 5.07 → 5.07–5.08 s/cycle | unchanged | **accepted as a no-op** |
| C5 step 2 | — | — | N/A | N/A | N/A | **deferred, see below** |
| H1 (new) | working tree | BE (incl. adversarial floor stress) | `RecvAndUnpackCC` halo width 28 → 8 components on 5 of 6 microstages | **steady state −14.2% with B1** | unchanged (74.41%) | **accepted — largest source win** |
| A4b | reverted | BE | warm-up only | 3-cycle: −32.4%; **steady state: +11% slower** | 74.46% | **REJECTED + reverted** (see correction) |
| B1 | working tree | BE | flux-clear traffic 28 → 2 components | 2.67–2.71 → 2.65–2.68 s/cycle (at the noise floor) | unchanged | **accepted** |
| H2 | working tree | BE | `u1` register copy 28 → 2 components | steady 2.542 → 2.530 s (0.2 sd, noise floor) | unchanged | **accepted as marginal** |
| Occupancy (`LaunchBounds`) | — | — | `Sync` regs 166 → 128, occupancy x2 | steady 2.542 → 2.533 s (no gain) | unchanged | **rejected + reverted** |
| **A7** | — | — | N/A | N/A | **89.40% at `nghost=3`** | **BLOCKED — ghost budget, see below** |

## A0 — host pin/unpin attribution

Date: 2026-08-02

Inputs:

- `profile_20260729/dci_rank0.nsys-rep` and its existing SQLite export;
- `profile_20260729/dci_rank3.nsys-rep` and its existing SQLite export;
- the capture environment and application stdout embedded in those reports;
- the boundary and laser MPI call sites in `src/bvals/` and `src/laser/`.

The reports were captured with `SHOW_BACKTRACE=false`, and every
`CUPTI_ACTIVITY_KIND_RUNTIME.callchainId` is null. Direct CPU-stack attribution is
therefore unavailable from these captures. Temporal attribution is nevertheless decisive:

| rank | `cuMemHostRegister` calls / time | immediately after CC/FC/FC-flux halo pack | share of calls / API time | unregisters inside `MPI_Finalize` |
|---:|---:|---:|---:|---:|
| 0 | 4,161 / 3.162136 s | 4,090 / 3.146521 s | 98.29% / 99.51% | 2,589 / 4,161 |
| 3 | 4,546 / 3.178144 s | 4,527 / 3.173129 s | 99.58% / 99.84% | 2,860 / 4,546 |

On rank 0, 1,332 registrations occur directly inside `MPI_Wait`/`MPI_Test`; tracing each
request handle back to its originating nonblocking operation identifies all 1,332 as mesh
boundary requests with tags produced by `CreateBvals_MPI_Tag`. The longest registration
burst follows `MeshBoundaryValuesCC::PackAndSendCC`: 694 registrations and 0.915 s occur
while waiting on one 458,752-byte boundary send. No registration is enclosed by another
CUDA runtime/driver operation, which rules out a `deep_copy` call as the direct caller.
The large cache teardown in `MPI_Finalize` independently identifies the owner as the MPI
transport's registration cache.

### Attribution

The subsystem is the cell-centered, face-centered, and face-flux mesh halo path in
`src/bvals/`. Those paths pass `CudaSpace` boundary-buffer pointers directly to MPI, so
they are already GPU-aware and do not use the laser deck's `gpu_aware_mpi` switch. UCX is
the mechanism: the captured environment sets
`UCX_TLS=self,sm,tcp,cuda_copy`. That list excludes `cuda_ipc`, so same-node GPU payloads
are staged through host shared-memory transport; UCX registers the staging regions with
CUDA and retains many of them until finalization. The installed UCX 1.18.0 build advertises
both `cuda_copy` and `cuda_ipc`, so the missing transport is a configuration restriction,
not a missing capability.

The small remainder after the nearest-kernel classification does not change the result:
it accounts for 0.49% of registration API time on rank 0 and 0.16% on rank 3. There is no
laser kernel adjacent to a material registration burst.

### A4 recommendation

Do not treat `laser/gpu_aware_mpi=true` as the primary A4 fix; it cannot affect the
dominant mesh-halo calls. A4 should first bracket, without a profiler, the current transport
list against either UCX automatic selection (unset `UCX_TLS`) or
`self,sm,tcp,cuda_copy,cuda_ipc`, then repeat the profile with both the GPU-activity and
API-call sections enabled. If CUDA IPC is correct on all eight local V100s, no separate
application-level boundary-buffer reuse change is needed. Persistent mirrors for the small
laser count/offset arrays remain useful but are not the host-pinning fix.

Only if CUDA IPC qualification fails should A4 add persistent pinned staging buffers to
the mesh boundary layer or tune the UCX registration cache. That fallback must be measured
separately because it adds explicit D2H/H2D copies and consumes memory. Acceptance for the
transport fix is a large reduction in `cuMemHostRegister`/`cuMemHostUnregister` count and
time, byte-exact fields, unchanged memory-band compliance, and a sustained wall-clock win
in opposite-order A/B runs.

---

## A4 — GPU-aware MPI / host-buffer pinning

Date: 2026-08-02

A0 attributed the 10 058 `cuMemHostRegister`/`Unregister` pairs to the UCX transport's
registration cache on the `src/bvals/` mesh-halo path, and recommended qualifying CUDA IPC
or UCX automatic transport selection before any application-level buffer-reuse change.

### Transport qualification — both A0 candidates fail

| configuration | cycle-0 elapsed | cycle-1 elapsed | verdict |
|---|---:|---:|---|
| `UCX_TLS=self,sm,tcp,cuda_copy` (current) | 3.30 s | 12.77 s | reference |
| `+cuda_ipc` | 56.2 s | 666.4 s | **rejected** — 64× slower per cycle |
| `UCX_TLS` unset (UCX auto) | 70.3 s | (killed) | **rejected** |

CUDA IPC is available in the installed UCX 1.18.0 build but is catastrophically slow on
these eight SXM2 V100s, so A0's preferred fix is unavailable. This moves the card to A0's
stated fallback: tune the registration behaviour instead.

### The actual lever — rendezvous threshold

The registrations are the rendezvous protocol's staging buffers. The mesh boundary packets
are 458 752 B, comfortably above the automatic rendezvous threshold, so every halo round
drives fresh host registration. Forcing those packets down the eager path removes the
registration path entirely.

Per-cycle wall clock (3 cycles, `nlim=3`, outputs disabled, no profiler; cycle 0 excluded
so init is not counted):

| configuration | per-cycle | note |
|---|---:|---|
| baseline | 7.49 s / 7.38 s | two independent runs |
| `UCX_ZCOPY_THRESH=inf` | 7.01 s | not the lever |
| `UCX_RCACHE_MAX_REGIONS=65536` | 7.25 s | cache capacity is not the constraint |
| `UCX_RNDV_FRAG_SIZE=host:512K,cuda:512K` | 7.25 s | not the lever |
| **`UCX_RNDV_THRESH=1M`** | **5.30 s** | **accepted** |
| `UCX_RNDV_THRESH=512K` / `8M` / `inf` | 5.34 / 5.32 / 5.33 s | equivalent |
| `UCX_RNDV_THRESH=1M` + `UCX_POSIX_SEG_SIZE=512K` | 14.28 s | rejected |
| `UCX_RNDV_THRESH=1M`, `UCX_TLS` without `tcp` | 5.32 s | no further gain |
| `UCX_RNDV_THRESH=1M`, `UCX_RNDV_SCHEME=get_zcopy` | 5.31 s | no further gain |

`1M` is chosen over `inf` because it still bounds unexpected-message buffering for any
message larger than the boundary packets, at identical measured cost.

### Profile delta (nvprof, 3 cycles, rank 1)

| metric | before | after |
|---|---:|---:|
| `cuMemHostRegister` | 16 380 calls / 9.024 s / 51.99% API | **0** |
| `cuMemHostUnregister` | 8 749 calls / 3.038 s / 17.50% API | **0** |
| `cuMemcpyAsync` | 147 083 / 373 ms | 891 392 / 6.866 s |
| `[CUDA memcpy HtoD]` | 74 652 / 679 ms | 447 008 / 740 ms |
| `[CUDA memcpy DtoH]` | 73 833 / 556 ms | 445 889 / 913 ms |

The card's literal acceptance criterion — a ≥50% drop in HtoD+DtoH *count* — is **not**
met: eager delivery fragments each 448 KiB packet into shared-memory segments, so the
count rises 6× while the mean transfer falls from 9.09 µs to 1.65 µs. The plan's own
framing resolves this (§1.1: "the transfers are the cheap half; the pinning around them is
the expensive half"). Registration, the stated 8.33 s target, is eliminated outright, and
the sustained wall-clock win is 28.5% across six runs in two configurations.

### Gate

BE. Full-volume `mhd_w_bcc`, `mhd_3t` and `laser` dumps at every cycle for 3 cycles:
**all 15 files byte-identical**; macro dt sequence, Biermann substep counts and stability
ratios identical; laser structural counters (segments, path, transfers, waves, iterations,
reflections) identical. The `laser:` global power sums differ in the last 1–2 ulps because
their MPI reductions re-associate — this is pre-existing run-to-run nondeterminism, not an
A4 effect: two baseline runs of the unmodified binary differ in the same digits.

### Landing

`UCX_RNDV_THRESH=1M` is applied by `DCI_3D/run_case.py` via `MPI_TRANSPORT_ENV` at every
launch site, and is echoed in the `--dry-run` command by the new `env_prefix` helper so the
printed command reproduces the measured configuration. It is deliberately **not** added to
`/home/mengqi/Research/bashrc_athenaK`: that file is shared with other codes, and an eager
threshold this high is not a safe global default for inter-node jobs.

---

## E2 — gate laser diagnostics: not applicable to this deck

Date: 2026-08-02

The card assumes the ten diagnostic components of `cell_data` are unused in production.
They are not:

- `DCI_3D/dci_3d.athinput:192` sets `report_diagnostics = true`, so the card's proposed
  `report_diagnostics_ == false` guard never fires on the production deck;
- `DCI_3D/plot_laser_rays.py:308` reads `laser_path`, `laser_ray_count` and
  `laser_dir1/2/3` — components 2, 4, 5, 6, 7 — and those plots are a production
  deliverable (commit `5ba529da`, "Automate production laser ray plots");
- `DCI_3D/README.md:147` documents `laser_dir1/2/3` as part of the recorded output.

Implementing the guard as written would therefore be inert here, so it was **not** landed.

A narrower variant does remain available and is *not* claimed by this card: components 8
(`laser_dispersion_error`) and 9–11 (`laser_x1/x2/x3_moment`) have no reader anywhere in
the repository. Dropping them would remove 4 of ~11 `atomic_add`s per segment, a quarter of
the `laser_clear_stage` traffic, and ~191 MB/rank. It changes what `variable = laser`
writes, so it is a deck/analysis decision rather than a pure optimisation, and is left for
the owner to approve.

---

## A5 + C4b + E1 + E4 + B2 — Wave 1 bundle

Date: 2026-08-02 · Gate: BE, bundled per plan §6 "WAVE 1 bundling"

| card | change | file |
|---|---|---|
| A5 | two per-microstep `MPI_Allreduce` (int MAX for the validity flag, Real MIN for the limit) folded into one 2-element MIN. A rank with an unusable limit votes `-1.0` in the flag slot, so MIN reproduces the old MAX exactly, and contributes `+inf` to the limit slot so a NaN can never reach the healthy ranks' minimum. Removes 4 blocking collectives per cycle. | `src/driver/driver.cpp:352-380` |
| C4b | opacity axes pre-logged, removing 4 of the 6 axis `log` calls per material lookup (8 of 12 per mixed lookup). The axis logs are filled by a **device** kernel (`OpacityTable::BuildLogAxes`), not on the host, so each stored value is bit-identical to the one the inline `log()` produced — this is what keeps the card BE rather than CG. | `src/two_temperature/opacity_table.{hpp,cpp}` |
| E1 | the DDA and refractive trace kernels launch over the compacted active count instead of `nrays_` = 4096. `SeedActiveQueue` now returns its scan total and `CompactActiveQueue` takes the current count so it, too, scans only the populated prefix. The loop still enters once with a zero count, preserving the `iterations` diagnostic exactly. | `src/laser/laser_trace.cpp:482-524, 627-635, 1068-1070, 1142-1149` |
| E4 | `RefreshGlobalBlockInfo` caches its result. The descriptors are a pure function of the block tree and rank map, so they are rebuilt only under AMR or a change in `nmb_total`; previously all 1024 were rebuilt on the host and copied H2D on every RK stage. | `src/laser/laser.cpp:679-694, 733` |
| B2 | `RestrictB` branches from the CT update instead of from the completed U halo, so the U and B halo rounds overlap; `ApplyPhysicalBCs` now depends on both. `RestrictB` touches only `b0`/`coarse_b0`, so it never needed U's ghosts. Saves one latency round on each of the 8 microstages per cycle. | `src/mhd/mhd_tasks.cpp:130-148` |

**Gate:** BE. All 15 full-volume field dumps byte-identical to the A4-accepted reference;
macro dt sequence, Biermann substep counts, stability ratios and every laser structural
counter identical.

**Measurement:** 5.30 / 5.32 / 5.33 / 5.34 s/cycle before → 5.21 / 5.22 / 5.23 / 5.23
after. −1.7%.

**Caveat, stated plainly:** the plan asks for an opposite-order bracket for sub-noise
bundles. This is a same-order comparison — the two clusters were measured about 40 minutes
apart. Building a true frozen reference binary was attempted and abandoned: `git stash` of
the touched files also reverts the prior session's uncommitted `opacity_table.hpp`
refactor, which `thermal_radiation.cpp` depends on, so the reference does not compile in
isolation. The evidence is still reported as adequate because the within-configuration
spread is 0.2% (0.04 s and 0.02 s) and the two clusters do not overlap, which is a much
tighter separation than the plan's assumed 3% noise floor. `DCI_3D/perf_work/athena.w1`
is retained as the frozen reference for subsequent BE cards, so later cards do get a
proper bracket.

**Cards not landed in this wave, with reasons:**

- **B1** (drop the six per-microstage memsets) — the memsets clear *all* components of
  `uflx`/`efld` over the full ghost extent, while the subcycle path writes only `IEN` and
  `iele` over `is..ie+1`. Removing them is safe on a uniform mesh but leaves garbage in the
  other components for the `multilevel` flux-correction path, which packs and unpacks the
  whole array. The target is ~0.24% of GPU time (48 of 231 memsets per cycle, in a 1.18%
  group). Deferred as poor value for the correctness surface it opens.
- **E3** (O(1) block lookup) — the plan calls this "the highest-risk BE card", and the
  entire laser group is 0.4% of GPU activity, so the gain is bounded well below noise. It
  is also harder here than the card assumes: this deck uses `meshblock_order = x1_rank_map`,
  so gid is **not** an arithmetic function of the logical index and an inverse
  `(lx3,lx2,lx1) → gid` map would have to be built and cached first. Deferred to be
  re-ranked after Wave 2, when the laser share is re-measured.
- **F1b** (fold the Spitzer constants) — `8.0*sqrt(2.0*pi)` and `pow(electron_charge_cgs, 4)`
  have compile-time-constant operands and nvcc folds both, so the card is a no-op as the
  plan anticipated. Hand-folding them is *not* free of risk: it changes the left-to-right
  association of the `denominator` product and would silently break byte-exactness for no
  gain. Recorded as no-op, not landed.

---

## T0 — convergence gate

Date: 2026-08-02 · Deliverable: `DCI_3D/convergence_gate.py`

The gate factors `verify_production_gate.py`'s metric layer (`read_history`,
`parse_cycle_log`, `parse_laser_diagnostics`, `check_3t_binary`) rather than
reimplementing it, so the two gates cannot disagree about what "energy" or "clamp" means.
It captures runs (`capture`), compares a reference against a candidate (`compare`), does
both in one step (`check`, optionally `--with-refinement`), and validates itself
(`selftest`).

**One supporting code change was required.** §6.0b asks for `div B` drift, and the deck
had no such diagnostic — `abs_B` is the field magnitude, not the divergence. `NHISTORY_VARIABLES`
and `NREDUCTION_VARIABLES` were 20 and `kHistoryFields` was already 20, so both were
widened to 21 and `DCIHistory` now emits volume-integrated `|div B|` from the face fields
(`DCI_3D/dci_3d.cpp`). Measured cost: none — 5.22/5.23 s/cycle with the field against
5.24/5.25 without.

### Tolerance table

| check | limit | rationale |
|---|---:|---|
| `energy_closure_drift` | 5e-9 of `chain_E` | closure is an enforced identity, not a converged quantity; bound admits reduction re-association, not a leak |
| `ch_mass_drift` | 1e-11 | conservative scalar flux only; independent of the EOS closure |
| `divb_ratio` | 1e-10 | CT invariant; an EOS change cannot legitimately raise it |
| `field_relative_l2` | 2e-3 | the one deliberately loose bound: roundoff grows in a driven flow, and 2e-3 is far below the deck's own discretisation error |
| `refinement_convergence_ratio` | 1.5 | the actual convergence test — refining must not amplify the candidate-vs-reference difference |
| `dt_sequence_relative` | 5e-2 | dt may respond, but must not switch limiter branch |
| `dt_collapse_factor` | 0.5 | hard floor for `causal_timestep_no_collapse` |
| `biermann_substep_growth` | 1.25 | a closure change must not move cost into the subcycle count |
| `laser_residual` | 1e-10 of launched | transport accounting identity, EOS-independent |
| `laser_cap_margin` | 2x | §4.1's required margin to the reflection and wave caps |
| `eos_clamp_growth` | 0 additional cells | `finite_nonnegative_3t` admits no new out-of-table states |

### Acceptance (§6.0b) — both halves demonstrated

- **Reference against itself:** PASS, with `field_relative_l2`, `ch_mass_drift` and
  `dt_sequence_relative` all exactly 0.
- **Reference against a deliberately perturbed build** (the plan's own suggestion: the
  mixed inverse cut from 48 to 47 bisections): the gate resolves the perturbation at
  `field_relative_l2 = 2.01e-8` over 20 cycles and correctly still returns PASS — a
  47-step bisection is physically equivalent, and reporting it with a finite, sane
  magnitude rather than failing is precisely the behaviour a CG gate must have.

`refinement_convergence_ratio` reports SKIP unless the coarse pair is captured
(`check --with-refinement` or explicit `--reference-coarse/--candidate-coarse`), so a
partial run can never be mistaken for a complete one.

**Note for future use:** `check_3t_binary` imports `numpy` via `vis/python/bin_convert`,
which the system `python3` does not have. Run the gate with
`/home/mengqi/miniconda3/envs/athenak-vis/bin/python`.

---

## A1 + A1b — safeguarded secant inverse: REJECTED, reverted

Date: 2026-08-02 · Gate: CG (passed) · Verdict: **no measurable gain; reverted**

This was the plan's largest projected win (§6, "LARGEST DEVICE-SIDE WIN", targeting the
58.8% EOS group). It does not exist on this deck.

### What was built

A bracketed false-position solver in log T replacing the fixed 48-step bisection at
`material_mixture.hpp:592`, with A1b threading `log_trial` through instead of the
`exp`-then-`log` round trip. Two formulations were tried:

| version | closure kernel registers | per-cycle |
|---|---:|---:|
| reference (fixed-48 bisection) | 154 / 146 | 5.22–5.25 s |
| Illinois + best-residual tracking + separate bisection fallback | **173 / 171** | 5.28–5.30 s |
| lean false position, periodic bisection, no auxiliary state | 156 / 150 | 5.25–5.28 s |

The first version regressed because the extra live state cost an occupancy block on
VOLTA70 — the same failure mode §7 records for "Strict mixed-inverse invariant hoisting".
The lean rewrite recovered the registers and still showed no gain.

### Why the premise does not hold here

Plan §3.2 argues the loop is latency-bound: ~200 dependent transcendentals, so cutting
48 iterations to ~6 should cut latency ~8x and *over*-deliver. A direct experiment
settles it — a throwaway build with the loop capped at a **single iteration**, i.e. the
absolute floor on what the loop can cost:

| build | per-cycle |
|---|---:|
| loop capped at 1 iteration (wrong answer, timing only) | **5.33 / 5.36 s** |
| unmodified 48-step bisection | 5.24 s |

Deleting the loop entirely does not make the run faster. Its cost is below measurement,
so no reformulation of it can pay. (The 1-iteration build is *slower* because the wrong
temperature perturbs the trajectory; the point is only that it is not faster.)

Two facts explain this. 83.3% of cells take the pure-material early return and never
enter the mixed loop at all — measured from `s_00` at t = 3 ns: 82.44% at `y0 <= 0`,
0.89% at `y0 >= 1`, 16.67% strictly mixed. And the loop is a small part of
`CloseBiermannStage`: the 31% that kernel occupies is dominated by the *forward* table
evaluations, which the plan's per-kernel accounting did not separate from the inverse.

### Consequence for the plan

§1's ranking attributes 58.8% to "tabular EOS closure" and §8 projects that group falling
to 20–30% mainly through A1. That projection should be withdrawn. The measurable
transcendental cost in this deck is in the **forward, per-cell, per-group** paths — which
is exactly what C4b and C4a target, and both of those did pay.

The CG gate passed on the lean version (`field_relative_l2 = 7.13e-8`, conservation and
dt sequence unchanged, no new clamps), so this is a clean negative result and not a
correctness failure. `DCI_3D/perf_work/athena.a1lean` is retained.

---

## A2b + C4a — remove the redundant forward evaluation; pre-store opacity logarithms

Date: 2026-08-02 · Gate: BE (bundle) · Verdict: **accepted**

**A2b** — `ElectronStateFromRhoSpecificEnergy` runs the mixed inverse and then repeats a
forward query at the returned temperature (deliberately, to retain trace-material
pressure scaling below a pure table's minimum density). That repeat used the *uncached*
`MixtureComponentFromRhoTemperature`, re-locating the density in both material tables.
It now reuses the inverse's own `MixedDensityCache` via `MixtureComponentFromCachedDensity`,
which the inverse already uses internally, so the forward evaluation is unchanged and only
the redundant density location is removed.

**C4a** — geometric opacity interpolation evaluated `log()` on all four corners of every
`(kind, group)` lookup. With 20 groups, a two-material table, and `Couple` doing two
kinds, that is ~320 `log` + 80 `exp` per cell, repeated per face by `AddFluxes` and again
by the transport limiter. `log_values` is now built once and only the final `exp` remains.

Two details make this land as **BE** rather than CG:

- the logarithms are filled by a **device** kernel, so each stored value is bit-identical
  to what the inline `log()` produced;
- non-positive entries store a sentinel (`kNonPositiveLog`) instead of a logarithm, so the
  zero-safe linear fallback is selected by exactly the same predicate as before —
  and the geometric path never has to load the linear values just to test their sign.

**Gate:** all 15 full-volume dumps byte-identical to the Wave-1 reference; timestep
sequence identical.

**Measurement:** 5.22 s/cycle (reference, interleaved in the same session) → 5.12 / 5.12 /
5.13 s/cycle. −1.8%.

---

## Running total

| stage | per-cycle | cumulative |
|---|---:|---:|
| baseline (`UCX_TLS=self,sm,tcp,cuda_copy`, pre-A4) | 7.38–7.49 s | — |
| + A4 (`UCX_RNDV_THRESH=1M`) | 5.30–5.34 s | −28.5% |
| + A5, C4b, E1, E4, B2 | 5.21–5.23 s | −30.0% |
| + A2b, C4a | 5.12–5.13 s | −31.2% |
| + C3 | 5.07–5.08 s | −31.9% |
| + A6 (no-op, kept) | 5.07–5.09 s | −31.7% |
| + H1 (narrowed Biermann halo) | 3.95–3.98 s | −46.8% |
| + A4b (`UCX_RNDV_THRESH=48K`) | 2.68–2.69 s | −64.0% |

**This whole table measures a warm-up transient and overstates the result. Superseded by
the steady-state table at the end of this file. The honest figure is −20.8%.**

Confirmed on the final binary (`DCI_3D/perf_work/athena.a6`) at 5.08 / 5.09 / 5.08 s per
cycle. Measured as cycles 1-3 of an `nlim=3` run with outputs disabled, cycle 0 excluded
so one-time initialisation is not counted; 8 ranks, 8x V100-16GB.

Frozen binaries retained in `DCI_3D/perf_work/` for future brackets: `athena.w1`
(Wave 1), `athena.ref` (+divB history), `athena.a1lean` (rejected A1), `athena.c4a`,
`athena.c3`, `athena.c2a6` (rejected C2), `athena.a6` (current).

---

## C3 — Planck recurrence, early exit, rolling group bound

Date: 2026-08-02 · Gate: CG · Verdict: **accepted**

All three items in the card, landed together as §6 C3 requires:

1. **Recurrence.** `exp(-n*x)` is geometric, so `q = exp(-x)` plus a running multiply
   replaces 64 independent transcendentals (`thermal_radiation.cpp:66-84`).
2. **Early exit.** The terms decay like `e^(-n*x)` with `x >= 0.5` on this branch, so the
   series stops as soon as a term can no longer change the double-precision sum.
3. **Rolling group bound.** `PlanckGroupFraction` evaluated *both* boundaries, so a
   20-group cell made 40 integral evaluations where 21 suffice. `Couple` and
   `Initialize` now roll the lower bound forward, which is the construction the
   source-limit reducer at `:1203-1217` already used — the file no longer carries two
   implementations of one quantity.

### Validation of items 1 and 2

The production gate could not exercise them: at this state almost every group boundary
lands in the `x >= 50` early-out, which is why the run comparison came out *exactly*
byte-identical (see below). They were therefore validated directly, by evaluating the old
and new formulations at 495 001 points across the whole `[0.5, 50)` branch:

| metric | result |
|---|---|
| bit-identical results | 489 966 / 495 001 = **98.98%** |
| worst relative error | **4.69e-14** (1.78e-15 absolute), at x = 0.518 |
| iterations to convergence | mean **4.81**, max 63, against a fixed 64 |
| `exp` calls | **1**, against 64 |

A few ulps at the worst point, and the loop is 13x shorter on average.

### Gate

CG. `field_relative_l2 = 0.000000e+00` — byte-exact in this run, with every conservation,
limiter and clamp check equal to the reference. This is the plan's own §6 C3 warning
confirmed: the cold state puts nearly all boundaries in the `x >= 50` branch, so the
measured gain comes from item 3 (halving the number of integral evaluations) while items
1 and 2 sit idle. **`Couple`'s share is a floor, not a steady-state value** — this card
should be re-measured at a later-time restart, where the recurrence actually runs and
where its value should be materially larger than the 0.9% observed here.

### Measurement

5.12 s/cycle (interleaved reference) → 5.07 / 5.08 / 5.08 s/cycle. −0.9%.

---

## Harness defect found and fixed during C3

The first `convergence_gate.py` implementation emitted a full-volume 3T dump on every
cycle. At 512x256x256 with 27 fields each dump is ~3.4 GB, so a 20-cycle capture retained
~70 GB and four captures exhausted the 1.9 TB filesystem mid-run. Only the final state is
ever compared, so `capture_run` now calls `prune_volume_dumps` the moment the run exits,
retaining just the last dump. A capture is now 3.4 GB instead of ~70 GB.

---

## C2 — sound-speed density-location cache: REJECTED, reverted

Date: 2026-08-02 · Verdict: **register pressure, as with A1**

`TabularSoundSpeedSquared` performs sixteen forward evaluations per cell (2 components x
4 finite-difference points x 2 materials) across only **three** distinct densities, so
ten of the sixteen density searches are redundant. Passing three shared
`MixedDensityCache` objects removes them — and loses.

| build | `Sync` registers | `RefreshMaterialThermodynamics` registers | per-cycle |
|---|---:|---:|---:|
| without the cache | 146 | 166 | 5.07–5.09 s |
| with three shared caches | **174** | **197** | 5.12–5.13 s |

Three live `MixedDensityCache` objects cost 28 and 31 registers, and the occupancy that
buys is worth more than ten table searches.

### The pattern this makes explicit

This is the third time the same mechanism has decided a card, and it is worth stating as
a rule for the rest of the plan: **the tabular EOS kernels are occupancy-limited, not
work-limited.** `Sync`, `CloseBiermannStage` and `RefreshMaterialThermodynamics` all sit
at 146–166 registers with 80–88 bytes of stack already spilled, right at the VOLTA70
cliff. Any optimisation that trades *added live state* for *removed work* loses there:

- A1 (Illinois secant with best-residual tracking): 154 → 173, regressed;
- C2 (three density caches): 146/166 → 174/197, regressed;
- §7's already-rejected "Strict mixed-inverse invariant hoisting — Register pressure" and
  "Exchange-wide density-cache reuse — Rejected" are the same mechanism.

What *does* pay in these kernels is moving work **out** of them entirely, with no new
live state — precisely C4b and C4a (precomputed tables consumed by a plain load) and C3
item 3 (fewer calls, no extra state). Future cards should be screened on their register
delta with `cuobjdump -res-usage` **before** being measured; it is a two-second check
that would have predicted both negative results here.

---

## A6 — hoist the centre `log(ne)` in `biermann_newdt`

Date: 2026-08-02 · Gate: BE · Verdict: **accepted as a no-op**

`const Real log_ne = log(ne)` was redeclared in all three directional blocks although
`ne` is assigned once per cell. The nine declarations are now one hoisted variable.

**Byte-exact:** all 15 full-volume dumps identical. **No measurable gain:** 5.07 s/cycle
before, 5.07/5.07/5.08 after — nvcc already common-subexpression-eliminated `log(ne)`
across the blocks, so this only removes the redundancy from the source.

Kept rather than reverted because it is byte-exact, strictly less redundant, and carries
no cost. The card's optional second half — precomputing `log(ne)` into a 47.8 MB scratch
array — was **not** attempted: it adds a global array read into the same
occupancy-limited kernel family that has now rejected A1 and C2 for exactly that reason.

---

## C5 step 2 — merge the three directional `Max` reductions: deferred

Date: 2026-08-02

The card's own instruction is to land C4a first and re-measure, because C4a removes the
dominant per-face cost in both `AddFluxes` and the limiter. That has been done, and on
the arithmetic the residue does not justify the change: the three reducers run about
twice per cycle, so merging them saves ~4 host synchronisations per cycle, ~5 ms against
5 070 ms — roughly 0.1%, far below the noise floor and not bundleable with anything left.

It is also a bigger change than the card implies. The three kernels iterate over
*different* face counts (`nx3*nx2*(nx1+1)`, `nx3*(nx2+1)*nx1`, `(nx3+1)*nx2*nx1`), unlike
the `biermann_newdt` multi-reducer precedent where all three directions share one cell
index space; merging means restructuring all three face loops onto a common index. Since
`fmax` is exactly associative and commutative the result would still be byte-exact, so
this remains available — it is deferred on value, not on correctness.

Step 3 (striding the limiter) is untouched and remains the real lever in this card if the
limiter is still a large share at checkpoint 3.

---

# CHECKPOINT 1

Date: 2026-08-02 · Binary: `DCI_3D/perf_work/athena.a6` · 3 cycles, 8 ranks, `nvprof`,
`UCX_RNDV_THRESH=1M`, outputs disabled. Both sections captured per §4.2. Raw logs in
`DCI_3D/perf_work/prof_checkpoint1/` (and `prof_checkpoint1_noA4/` for the same build
*without* the A4 transport setting, which isolates its API-side effect).

## Memory band

| GPU | peak | fraction | gate |
|---|---:|---:|---|
| 0–7 (all identical) | 12 192 MiB | **74.41%** | **IN BAND** (60–80%) |

Baseline was 11 876 MiB = 72.49%. The +316 MiB is the pre-logged opacity data (C4b axes,
C4a values) and the widened history reduction. `gpu_memory_60_80_all` still passes, so
**no mesh retune is required** and the frozen references stay valid (§2.1).

## API calls — A4's effect, measured on the same build

| call | without `UCX_RNDV_THRESH=1M` | with it |
|---|---:|---:|
| `cuMemHostRegister` | 15 996 calls / **9.997 s** / 54.99% | **absent** |
| `cuMemHostUnregister` | 8 454 calls / **3.338 s** / 18.36% | **absent** |
| `cuMemcpyAsync` | 147 296 / 401 ms / 2.21% | 891 392 / 6.928 s / 44.91% |
| `cuStreamSynchronize` | — | 891 392 / 4.118 s / 26.69% |
| `cudaEventSynchronize` | 415 / 1.755 s / 9.65% | 471 / 1.944 s / 12.60% |

13.3 s of host-memory registration is gone. **The new top host cost is the eager
fragmentation itself**: 891 392 `cuMemcpyAsync` + `cuStreamSynchronize` pairs per 3
cycles, because UCX splits each 448 KiB boundary packet across ~56 shared-memory segments
(`UCX_POSIX_SEG_SIZE = 8256`). Raising the segment size was tried and is much worse
(5.30 → 14.28 s/cycle at 512K), so this is not tunable from the transport side. The
actionable lever is to **send fewer packets**, which is exactly what A7 targets: 24
Biermann halo rounds per cycle against 6 for all of MHD.

## GPU activities — the §1 ranking has changed materially

| group | plan §1 baseline | checkpoint 1 | note |
|---|---:|---:|---|
| Tabular EOS closure | **58.8%** | **~17.9%** | `Sync` 6.74, `CloseBiermannStage` 6.49 + 2.81, `Refresh` 1.87 |
| Data movement | 9.6% | ~29.9% | eager fragments; inflated by `nvprof` instrumentation |
| Core MHD | 7.1% | ~13.3% | `CalculateFluxes` x3 7.39, `RKUpdate` 2.19, `ConsToPrim` 3.67 |
| Radiation (AP-FLD) | 12.9% | ~6.0% | three `NewTimeStep` reducers 2.04 + 2.02 + 1.91 |
| Halo pack/unpack | 4.9% | ~6.7% | `RecvAndUnpackCC` |
| One-time init | 3.9% | 4.59% | `TwoTemperature::Initialize`, single call |

Two consequences for the rest of the plan:

1. **The EOS closure is no longer the dominant group.** Against the plan's own reference
   kernel, `CloseBiermannStage / CalculateFluxes` has gone from 31.15/4.23 = 7.4 to
   9.30/7.39 = 1.3. §8's projection of that group falling to 20–30% *via A1* should be
   replaced: it fell without A1, and A1 has been shown not to help (see the A1 entry).
2. **C5 has largely dissolved, as its own card predicted.** The radiation limiter was
   8.08% across four reducers at baseline and is ~6.0% across three now, after C4a made
   the shared per-face opacity work cheap. Step 2 is not worth its restructure (see the
   C5 entry); step 3 remains the only real lever there.

## Next-highest-value work, re-ranked from this checkpoint

1. **A7 / A3 — reduce the number of Biermann halo rounds and dt reductions.** The eager
   fragment count and `cudaEventSynchronize` are both downstream of 24 blocking halo
   rounds per cycle. A3 is the cheap first step: the measured `biermann_dt_min` and
   `biermann_dt_max` differ by only ~0.15% *within* a cycle, so the limit really does
   vary on the macro-step timescale and amortising it is well founded.
2. **C3 re-measured at a later-time restart**, where the Planck recurrence actually runs.
3. **B3 ghost-extent audit** — `Sync` and `CloseBiermannStage` remain the two largest
   compute kernels and both run over the ghost-inclusive 36³.

Screen every future card with `cuobjdump -res-usage` first (see the C2 entry).

---

## Proposal for the next stage — narrow the Biermann CC halo to the components it changes

Date: 2026-08-02 · **Not implemented.** Scoped and evidenced here for the next session.

This is the largest remaining item found in this campaign, and it is larger than anything
left in Waves 1–3. It is a cheaper cousin of A7 and should be evaluated before it.

### The observation

`MHD::BiermannInitRecv` posts `InitRecv(nmhd + nscalars)` = **28 components**, and the
microstage exchanges all of `u0`. But a Biermann microstep modifies only:

- `IEN` (index 4) and `iele` (index 7), from `BiermannRKUpdate` + `AddElectronWorkRHS`;
- `iion` (index 6), reconstructed by `CloseBiermannStage`.

(`nmhd = 5`; `TwoTemperature` takes `first_component_index = nmhd + nuser_scalars = 6`,
so `iion = 6`, `iele = 7`, and the 20 radiation groups occupy 8–27.)

The 20 radiation group energies, the density, the three momenta and the user scalar — 24
of 28 components — are **untouched by the microstep**, so their ghost values are still the
ones the macro-step exchange delivered. Sending the contiguous range `[IEN .. iele]`
(4 components, including the untouched user scalar at index 5 for contiguity) would cut
the payload **7x** on **24 of the 30 CC halo rounds per cycle**.

### Why it should be worth a lot

From checkpoint 1: `RecvAndUnpackCC` is 6.71% of GPU activity, and the eager transport
fragments — now the single largest host cost at 44.91% + 26.69% of API time across 891 392
`cuMemcpyAsync`/`cuStreamSynchronize` pairs — scale directly with halo bytes. Cutting the
Biermann rounds 7x should take a large bite out of both at once.

### Feasibility

Tractable. `InitRecv(nvars)` only sizes the `MPI_Irecv`; the buffers are allocated at full
width already. `PackAndSendCC` / `RecvAndUnpackCC` index `a(m, v, k, j, i)` with
`v` in `[0, nvar)` and place it at buffer offset `nk*v`, so adding a component offset is a
two-line change per kernel: index `a(m, vbeg+v, ...)` and keep the buffer offset on `v`.
Add `vlo`/`vhi` parameters defaulting to the full range so no other caller changes.

### The audit that must happen first — do not skip this

`MHD::BiermannConToPrim` calls `peos->ConsToPrim(u0, b0, w0, bcc0, ...)` over the
**ghost-inclusive** extent, and EOS floors can write corrected conserved variables back
into `u0`. Today a full re-exchange every microstage overwrites any locally-floored ghost
with the owner's interior value, so ghost/interior consistency is restored by brute force.
With a narrowed exchange, `IDN` and the group energies would keep whatever the *local*
ghost-extent `ConsToPrim` wrote.

The argument that this is still correct is that the ghost cell and the owner's interior
cell run the same floor on the same inputs and therefore produce the same result — but
that argument depends on `bcc0` and every floor input being identical at the shared cell,
and it has not been verified. Confirm it before landing, and note that a 3-cycle
byte-exact run is **not** sufficient evidence: it may never trigger a floor. Gate this on
a run that demonstrably exercises `eos_floor` (the `eos_floor` history column, added for
T0's `div B` work, counts exactly those cells).

Restrict the optimisation to `!multilevel` in the first cut: the coarse-array
prolongation/restriction path assumes the full component count and is not worth
generalising until the uniform-mesh gain is measured.

---

## H1 — narrow the Biermann microstage cell-centered halo

Date: 2026-08-02 · Gate: BE · Verdict: **accepted — the largest single win after A4**

Not a numbered card in the plan; found at checkpoint 1 and written up there as a
proposal. It is a much cheaper cousin of A7 and should be evaluated before it.

### Change

`MHD::BiermannInitRecv` posted `InitRecv(nmhd + nscalars)` = **28 components** and the
microstage exchanged all of `u0`. A Biermann microstep writes only three: `IEN` and
`iele` (`BiermannRKUpdate` + `AddElectronWorkRHS`) and `iion` (reconstructed by
`CloseBiermannStage`). The 20 radiation group energies are untouched, so their ghosts are
still the ones the macro-step exchange delivered.

`MeshBoundaryValuesCC::PackAndSendCC` / `RecvAndUnpackCC` take an optional `nvar_in`
(default −1 = all components, so no other caller changes), and new `MHD::BiermannSendU` /
`BiermannRecvU` tasks size the exchange with `MHD::BiermannHaloNumVars()`.

### The audit — three load-bearing constraints

1. **The range starts at `IDN`, not at `IEN`.** `ConsToPrim` writes corrected conserved
   values back into `u0` when a floor fires (`ideal_mhd.cpp:157-166`), and an intermediate
   microstage runs `ConsToPrim` only over `is-1..ie+1`. A floor firing in an owner's
   *second* interior cell would never be reproduced in the matching ghost at layer `is-2`,
   so `IDN` and the momenta must travel even though Biermann never touches them. The range
   is therefore `[0 .. iele]` = 8 of 28 — which also means only the component *count*
   changes and no index offset is needed anywhere.
2. **The closing stage of each half-step keeps the full-width exchange**, gated on the
   existing `biermann_stage_full_thermodynamics` flag — the same stage that already
   restores the complete ghost domain for the closure. So the radiation groups are
   refreshed before MHD reconstruction is allowed to read the outer ghost layer. With
   3 substeps per half-step this narrows 5 of 6 microstages.
3. **Uniform meshes only.** Restriction and prolongation move the full conserved vector
   through the coarse buffers, so `multilevel` falls back to the full width.

### Gate — BE, including an adversarial test of the audit's own weak point

Standard run, 6 cycles: **all 24 full-volume dumps byte-identical**, macro-dt and Biermann
substep sequences identical, laser counters identical.

That is necessary but not sufficient, because the production floors never fired
(`eos_energy_floor = 0`), leaving the constraint-1 hazard untested. So the comparison was
repeated with the floors raised hard (`dfloor` 1e-8 → 3e-4, `pfloor` 1e-14 → 1e-8) to
force the `ConsToPrim` write-back:

| | reference | narrowed |
|---|---:|---:|
| cells at the energy floor | 32 094 792 | 32 094 792 |
| cells pinned at `dfloor` | 95.47% | 95.47% |
| max EOS flag | 17 | 17 |
| full-volume dumps differing | — | **0 of 24** |

With the density floor active in 95% of cells and the conserved write-back firing across
essentially the whole domain, the narrowed halo still reproduces the full-width halo
bit-for-bit.

### Measurement

5.08 s/cycle (interleaved reference) → 3.95 / 3.98 / 3.95. **−22.1%.**
Memory unchanged at 74.41% on all eight GPUs.

### Why it is worth so much more than its share of `RecvAndUnpackCC`

CC halo volume per cycle falls from 30 rounds x 28 components to
10 x 28 + 20 x 8 = 48% of what it was. That lands on the two costs checkpoint 1 identified
as dominant: `RecvAndUnpackCC` (6.71% of GPU activity) and, more importantly, the eager
transport fragments that A4 traded the host-pinning cost for — 891 392 `cuMemcpyAsync` +
`cuStreamSynchronize` pairs per 3 cycles, 71.6% of all API time, which scale directly with
halo bytes.

### Follow-on

Narrowing further to `[IEN .. iele]` (4 components, 7x) would need an index offset and
would re-expose the constraint-1 hazard, since `IDN` would stop travelling. Given the
floor stress test shows how heavily that write-back can fire, this is **not** recommended
without a stronger argument. A7 remains the way to attack what is left.

---

## A4b — re-tune the rendezvous threshold after H1

Date: 2026-08-02 · Gate: BE · Verdict: **accepted — second-largest win of the campaign**

A4 chose `UCX_RNDV_THRESH=1M` from a sweep of 512K / 1M / 8M / inf, all of which measured
the same. **That sweep never went below 512K, and the optimum is far below it.** H1's
narrower packets prompted a re-sweep, which found it.

| `UCX_RNDV_THRESH` | per-cycle (post-H1) |
|---|---:|
| default (`intra:auto,inter:auto`) | 7.72 s |
| 1M (A4's choice) | 3.97 s |
| 128K | 2.77 s |
| 96K | 2.77 s |
| 64K | 2.76 s |
| **48K** | **2.68 / 2.69 s** |
| **32K** | **2.68 / 2.69 / 2.69 s** |
| 16K | 3.25 s |
| 8K | 3.42 s |

The optimum is a broad, flat plateau at 32–64 KiB — the large face buffers go zero-copy
rendezvous while the small edge and corner buffers stay eager, which is exactly the right
split. **48K** is adopted as the middle of the plateau.

### This was not only an H1 effect

Measured on the **pre-H1** binary (`athena.a6`) to separate the two:

| binary | at 1M | at 32K |
|---|---:|---:|
| pre-H1 (`athena.a6`) | 5.08 s | **3.13 s** |
| post-H1 (`athena.halo`) | 3.97 s | **2.69 s** |

So roughly 38% of this was available before H1 and A4 simply missed it by not sweeping
low enough; H1 then contributes a further 14% on top at the good threshold. Recorded
plainly because it is a lesson about the measurement, not only about the code: **A4's
sweep was too narrow, and a one-parameter sweep should bracket the optimum on both
sides rather than stopping when successive points agree.**

### Profile delta (checkpoint 2 → checkpoint 3, same binary)

| metric | at 1M | at 48K |
|---|---:|---:|
| `cuMemcpyAsync` | 665 488 / 4.930 s / 40.80% | **259 088 / 1.767 s / 20.72%** |
| `cuStreamSynchronize` | 665 488 / 3.023 s / 25.02% | 239 888 / 1.109 s / 13.00% |
| `cuMemHostRegister` | absent | 1 168 / 604 ms / 7.08% |
| `[CUDA memcpy HtoD]` count | 334 052 | **130 852** |

Host registration returns, but at 1 168 calls instead of the 15 996 that motivated A0 —
at these buffer sizes UCX's registration cache actually works, because the rendezvous
buffers are stable and reused rather than churning through the large-message fragment
pool. The transfer count is now ~one per boundary buffer per round, i.e. no fragmentation
left to remove; further reduction needs fewer rounds or fewer buffers (A7).

**Gate:** BE. All 18 full-volume dumps byte-identical, dt and substep sequences identical.
**Memory:** 74.45–74.46% on all eight GPUs, in band.

---

# CHECKPOINT 3

Date: 2026-08-02 · Binary: `DCI_3D/perf_work/athena.halo`, `UCX_RNDV_THRESH=48K`
Raw logs: `DCI_3D/perf_work/prof_checkpoint3/`.

Wall is now **2.68 s/cycle against a 7.44 s/cycle baseline — a factor of 2.8**.
GPU activity is 4.84 s per 3 cycles = 1.61 s/cycle, so the run has gone from 41% GPU-bound
to **60% GPU-bound**. The host-side problem the plan opened with (§1.1: "59% of the run is
not GPU work") is substantially solved.

| kernel / group | share of GPU activity |
|---|---:|
| data movement (HtoD + DtoH + DtoD + memset) | 26.5% |
| `CloseBiermannStage` (both variants) | 10.63% |
| `CalculateFluxes` x3 | 8.46% |
| `TwoTemperature::Sync` | 7.71% |
| `TwoTemperature::Initialize` (one-time) | 5.27% |
| `RecvAndUnpackCC` | 4.99% |
| `ConsToPrim` | 4.20% |
| `RKUpdate` | 2.50% |
| `SynchronizeDualEnergyFromTotal` | 2.45% |
| radiation `NewTimeStep` x3 | ~7% |

Top API cost is now `cudaEventSynchronize` (23.79%, 471 calls at 4.31 ms) — the host
waiting on genuine device work, largely the serial laser transport iterations, not
removable overhead.

`TwoTemperature::Initialize` at 5.27% is an artefact of a 3-cycle measurement; over a
production run it is negligible. Excluding it, the profile is now dominated by real
physics kernels rather than by data movement or synchronisation.

## What is left, ranked

1. **A7** — the only remaining structural lever. Transfers are now ~one per buffer per
   round, so the way to cut them further is fewer rounds: 24 Biermann halo rounds per
   cycle against 6 for all of MHD.
2. **B3** for `Sync` / `ConsToPrim` — both run over the ghost-inclusive 36³ inside
   `MHD::ConToPrim`, which sits *after* the halo exchange, so their ghost values are
   genuinely consumed by the next stage's reconstruction. The plan's premise ("the halo
   exchange that follows would supply exactly those values") does **not** hold for these
   two; it holds only for `Exchange`, which is now too small to matter. A narrower claim
   may survive — `Sync` may only need one ghost layer, not two — worth roughly 1.2% of
   GPU activity, and it needs a careful read of every consumer of `thermodynamics` and
   `temperature` in the ghosts.
3. **B1** (per-microstage memsets) is now ~0.5% of GPU activity, still below noise.
4. **C3 at a later-time restart**, where the Planck recurrence actually runs.

---

## B1 — clear only the flux components the subcycle writes

Date: 2026-08-02 · Gate: BE · Verdict: **accepted; gain at the noise floor**

The card proposed removing the six per-microstage `deep_copy(..., 0.0)` calls by having
the first writer use `=`. That was deferred earlier because the memsets clear *all*
`nmhd+nscalars` components over the full ghost extent, and dropping them outright leaves
garbage for the `multilevel` flux-correction path, which packs the whole array.

H1's structure gives a cleaner answer: keep the clear, but clear only what is written.
`BiermannFluxes` now zeroes just `IEN` and `iele` — 2 of 28 components — on a uniform
mesh, and falls back to the full `deep_copy` under SMR/AMR. The other 26 are provably
dead there: `BiermannRKUpdate` reads only `IEN` and `iele`,
`AddPoyntingFluxFromEdgeEMF` adds only to `IEN`, and `BiermannSendFlux`/`RecvFlux` and
`UseCorrectedDriftFlux` all return immediately when `!multilevel`.

The `efld` clears are left alone: the edge arrays have no variable dimension and are ~27x
smaller than `uflx.x1f`, so they are not worth the churn.

**Gate:** BE. All 18 full-volume dumps byte-identical; dt and substep sequences identical.

**Measurement:** reference 2.71 / 2.67 s/cycle, candidate 2.68 / 2.65 / 2.65 / 2.68. About
−0.9%, which is **at the noise floor** and not separable — reported as such rather than
claimed. Kept anyway, on the same basis as A6: byte-exact, strictly less work (26 of 28
components of memset traffic removed on three face arrays every microstage), and no
downside. It is not counted as a win in the running total.

---

# STEADY-STATE CORRECTION — supersedes every wall-clock number above

Date: 2026-08-02

## What went wrong

The plan's §4.1 protocol specifies a 3-cycle A/B with cycle 0 excluded, and every card in
this ledger followed it. Extending a run to 20 cycles shows that window is inside a
warm-up transient:

| cycle | 1 | 2 | 3 | 5 | 8 | 11 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s/cycle (final build, production transport) | 10.09 | 7.36 | 5.87 | 4.24 | 3.40 | 2.59 | 2.51 | 2.46 |

The first cycle is **4x** the steady state. The cause is UCX's host-memory registration
cache filling: the `cuMemHostRegister` storm that card A0 diagnosed is a startup cost that
amortises, not a per-cycle cost. A 3-cycle measurement is therefore dominated by transport
warm-up, which is exactly what A4 then "optimised".

## A4 / A4b are rejected and reverted

Four transport settings, final build, 20 cycles each. "Steady" is the mean of cycles
11-20; "warm-up excess" is time spent above the steady rate.

| `UCX_RNDV_THRESH` | cycles 1-3 | **steady (11-20)** | 20-cycle total | warm-up excess | projected 1000 cycles |
|---|---:|---:|---:|---:|---:|
| **default (auto)** | 7.77 s | **2.54 s** | 75.7 s | 25.2 s | **42.8 min** |
| 48K (A4b) | 2.68 s | 2.81 s | 59.1 s | 3.5 s | 46.8 min |
| 256K | 2.99 s | 3.38 s | 72.0 s | 5.6 s | 56.4 min |
| 1M (A4) | 3.95 s | 4.35 s | 91.4 s | 5.6 s | 72.6 min |

Forcing eager transport flattens the warm-up and permanently slows the steady state:
halo packets arrive as many small fragments instead of one zero-copy transfer. The
crossover is near **80 cycles**; production runs are thousands. `MPI_TRANSPORT_ENV` is
back to empty, with the reasoning recorded inline in `DCI_3D/run_case.py`.

The same holds on the Wave-1 binary — steady state 3.21 s at default against 3.71 s at
48K — so this is a property of the transport, not an interaction with H1.

## What the source changes are actually worth

All at the production transport setting, 20 cycles, steady state = mean of cycles 11-20:

| build | cycles 1-3 | **steady (11-20)** | 20-cycle total |
|---|---:|---:|---:|
| `athena.w1` — Wave 1 (A5, C4b, E1, E4, B2) | 7.73 s | 3.21 s | 86.7 s |
| `athena.a6` — + A2b, C4a, C3, A6 | 7.50 s | 2.96 s | 82.3 s |
| `athena.b1` — + H1, B1 (final) | 7.77 s | **2.54 s** | 75.7 s |

| group | steady state | 20-cycle total | steady-state gain |
|---|---:|---:|---:|
| Waves 2–3 (A2b, C4a, C3, A6) | 3.21 → 2.96 s | 86.7 → 82.3 s | **−7.7%** |
| H1 alone (narrowed Biermann halo) | 2.96 → 2.58 s | 82.3 → 76.5 s | **−13.0%** |
| B1 alone (narrowed flux clear) | 2.58 → 2.54 s | 76.5 → 75.7 s | −1.4% (within noise) |
| **all source changes from Wave 1** | 3.21 → 2.54 s | 86.7 → 75.7 s | **−20.8%** |

Run-to-run spread on the steady mean is 0.03–0.08 s (1–3%), so H1's −13.0% is firmly
resolved and B1's −1.4% remains at the noise floor — the same conclusion its own entry
reached, now confirmed in the regime that matters. **H1 is the single largest accepted
change in this campaign.**

Note the 3-cycle column is nearly flat across all three builds (7.73 / 7.50 / 7.77) while
the steady column falls monotonically — the transient is transport-bound and hides source
improvements entirely. The 3-cycle protocol did not merely exaggerate; for source work it
was close to blind, and it is only luck that no accepted card was a steady-state
regression.

**Wave 1's own contribution is not quantified at steady state**: no pre-Wave-1 binary was
retained, and the working tree it was built from cannot be reconstructed by `git stash`
because it depends on other uncommitted work. The 7.44 s/cycle "baseline" quoted
throughout this ledger is a 3-cycle warm-up number and is not comparable to any steady
figure here.

## Protocol change for any future work

1. **Measure at least 20 cycles and report the mean of cycles 11-20.** A 3-cycle A/B on
   this deck measures transport warm-up.
2. **Report the per-cycle series, not just the mean.** The decay from 10.1 to 2.5 s is
   obvious in the series and invisible in a 3-cycle average.
3. **Sweep parameters on both sides of the optimum.** A4 swept 512K / 1M / 8M / inf, saw
   agreement, and stopped — the optimum was below the whole range, and the true best value
   was "no setting at all".
4. Keep screening register pressure with `cuobjdump -res-usage` before measuring (C2).

## What still stands

Unaffected by this correction, because they do not depend on wall-clock deltas:

- **T0**, the convergence gate, and its validation.
- **H1's** byte-exactness, including the adversarial floor-stress test.
- **A1, C2** rejections — both were register-pressure findings confirmed by
  `cuobjdump`, and both also regressed on wall clock.
- **C3's** numerical validation of the Planck recurrence.
- All **memory-band** results: 74.4–74.5% on all eight GPUs throughout.
- The **final convergence gate**: the complete accumulated change passes against the
  frozen reference with `field_relative_l2 = 0.000000e+00` and every conservation,
  limiter and clamp check equal.

---

# POST-CORRECTION WORK — two hypotheses tested under the 20-cycle protocol

Date: 2026-08-02 · All measurements: 20 cycles, production transport, steady state =
mean of cycles 11–20, run-to-run spread 0.05–0.07 s.

With the protocol fixed, two things were worth revisiting: an occupancy hypothesis that
the earlier register-pressure findings had suggested, and card **C2**, whose rejection had
rested on a 3-cycle measurement.

## Occupancy on the tabular kernels — REJECTED

The tabular EOS kernels sit at 146–166 registers with 80 bytes of stack already spilled.
On a V100 that is ~8 resident warps against a maximum of 64 — about 12.5% occupancy — so
"raise occupancy" looked like a lever over `Sync`, `CloseBiermannStage`,
`RefreshMaterialThermodynamics`, `Exchange` and `mhd_2t_dual_energy_sync` together, about
14% of wall time. A `par_for_lb` variant with explicit `Kokkos::LaunchBounds` was added to
test it.

| variant | `Sync` registers | stack | steady state | vs baseline |
|---|---:|---:|---:|---:|
| baseline (Kokkos default block size) | 166 | 80 B | **2.542 s** | — |
| `LaunchBounds<128, 1>` | 173 | 80 B | 2.586 s | −1.7% |
| `LaunchBounds<256, 2>` | **128** | **192 B** | 2.533 s | +0.3% (within noise) |

`LaunchBounds<256,2>` did exactly what it was asked to: it forced the register count from
166 down to 128 and **doubled theoretical occupancy**. It bought nothing — the extra
spilling (80 → 192 bytes of stack) precisely offset the gain. `<128,1>` was worse still,
because nvcc simply used the headroom to take *more* registers.

**Conclusion: these kernels are not occupancy-limited.** They are bound by the dependent
chain of scattered table reads. Both `par_for_lb` and all five call sites were reverted;
`src/athena.hpp` is unchanged.

This retires a hypothesis that had been forming across the A1 and C2 entries, and it
means **the register-count screening rule proposed in the C2 entry is wrong** — a register
delta is not a valid proxy for performance here. Measure instead.

## C2 re-tested — rejection stands, for a different reason

C2 caches the three distinct density locations in `TabularSoundSpeedSquared`, removing ten
of sixteen density searches. It was rejected earlier on a register delta plus a 3-cycle
measurement — both now-discredited grounds — so it was re-applied and re-measured.

| | steady state | spread |
|---|---:|---:|
| without C2 | **2.542 s** | 0.072 |
| with C2 | 2.598 s | 0.059 |

−2.2%, a separation of 0.8 sd: not firmly resolved, but the sign agrees with the 3-cycle
result and no measurement has ever shown C2 faster. **Reverted again.** The reason is now
the same as the occupancy finding rather than the register count: removing table
*searches* does not help a kernel that is waiting on table *reads*.

## Where this leaves the remaining work

Steady state is 2.54 s/cycle at **68.3% mean GPU utilisation** across the eight GPUs
(sampled over cycles 11–20), so roughly a third of the run is still host/MPI stall, and
the GPU third that remains is dominated by kernels that have now resisted two different
micro-optimisation strategies.

**A7 was the only substantial lever left — it has since been tested and is blocked by the
ghost budget; see the A7 entry at the end of this file.** The sketch below is retained
because the cost analysis in it turned out to be wrong in an instructive way:

- Pair the two SSPRK stages of each Biermann substep behind one halo exchange instead of
  two, by widening stage 1's update to `is-1..ie+1`. With `nghost = 2` the ghost budget
  is exactly sufficient. That halves the Biermann CC+FC halo rounds, 8 per cycle to 4,
  and it should be byte-exact — a wider update followed by no exchange produces the same
  interior values as a narrow update followed by one.
- The cost is that the stages preceding a skipped exchange must close over the full ghost
  domain rather than the reduced `is-1..ie+1`, which adds ~18% to those
  `CloseBiermannStage` calls. Whether the trade is net positive is genuinely open.

The implementation surface is large — `AddFluxes`, `AddEMFs`, `BiermannRKUpdate`,
`BiermannCT`, `BiermannCloseInterior` and `BiermannConToPrim` all derive extents
independently from `indcs`, and the CT/EMF path is where a mistake would be subtle. This
matches the plan's own decision to schedule A7 in Wave 4 as its own study, and it is where
the next session should start.

---

## A7 — pair the SSPRK stages behind one halo exchange: BLOCKED

Date: 2026-08-02 · Verdict: **not possible at `nghost = 2`; the alternative is out of the
memory band.** This closes the plan's last structural card.

### The design, and why it looked right

Each Biermann substep runs two SSPRK2 stages, each ending in its own CC + FC halo
exchange — 8 exchanges per cycle at the steady-state 4 substeps. Pairing them behind a
single exchange means widening stage 1's update to `is-1..ie+1` so stage 2 can run on
`is..ie` without fresh ghosts. With two ghost layers the budget looked exactly sufficient.

The cost also looked acceptable once `full_thermodynamics` was read properly: that flag
does **not** gate the closure's expense as the C5/B3 discussion had assumed. It selects
between `two_temp_refresh_biermann_electron` (11.1 ms, electron state only) and
`two_temp_refresh_material_components` (36.5 ms, full ion+electron state), and the
Biermann flux needs only the electron pressure. So A7 would have widened the *cheap*
closure by one layer, about +2 ms per substep — not the +25 ms first feared.

### Why it cannot be done

The Biermann edge-integral stencil already consumes the entire ghost budget.
`biermann_face_e2` (`src/mhd/biermann_battery.cpp:466`) runs `i` from **`is-1`** to `ie+1`
and reads `i-1` and `i+1`:

```
"biermann_face_e2", ... , js, je + 1, is - 1, ie + 1,
    ... ElectronPressure(..., m, k, j-1, i-1) ...
```

At `i = is-1` that reads `is-2`, which with `ng = 2` is index **0** — the outermost ghost
cell. Widening the update by one more cell would read index **−1**. The same holds in
`j` and `k` (`biermann_face_e3` already starts at `js-1`).

A7 therefore requires `nghost = 3`. Measured, not estimated:

| `nghost` | peak GPU memory | fraction | gate |
|---|---:|---:|---|
| 2 (production) | 12 246 MiB | 74.74% | **IN BAND** |
| 3 | 14 648 MiB | **89.40%** | **OUT OF BAND** |

+2 402 MiB, +19.6%. That is far outside the mandated 60–80% band, so it would force the
mesh retune that §2.1 forbids while the reference outputs are frozen — and it would also
add ~19% to every ghost-inclusive kernel, which is most of the expensive ones.

**A7 is closed.** Revisiting it means either accepting a mesh retune and re-freezing the
references, or reformulating the Biermann edge integral on a narrower stencil. Both are
larger questions than a performance card.

---

## H2 — copy only the register components the microstep reads

Date: 2026-08-02 · Gate: BE · Verdict: **accepted; gain at the noise floor**

`BiermannCopyCons` deep-copied all 28 components of `u0` into the low-storage register
`u1` at every microstep, but `BiermannRKUpdate` blends exactly two of them (`IEN` and
`iele`), and the macro SSPRK integrator re-seeds the whole register at its own stage 1.
The copy is now those two components; the face registers are still copied whole, because
`BiermannCT` reads all three components of `b1`.

Third instance of the same pattern as H1 and B1: the subcycle moved or cleared the full
conserved vector where it touches two components.

**Gate:** BE — all 18 full-volume dumps byte-identical.
**Measurement:** steady state 2.542 → 2.530 s, +0.5% at 0.2 sd — **not resolved**, and not
counted as a win. Kept on the same basis as A6 and B1: byte-exact, strictly less work
(~12 ms/cycle of device-to-device traffic removed), no downside.

---

# FINAL STATE

Date: 2026-08-02 · Binary: `DCI_3D/perf_work/athena.cc` (= current working tree)

| metric | value |
|---|---|
| steady-state wall | **2.530 s/cycle** (mean of cycles 11–20) |
| against Wave 1, same transport | **−21.2%** (3.21 s/cycle) |
| GPU memory | 74.4–74.7% on all eight GPUs, **in band** |
| mean GPU utilisation, steady state | 68.3% |
| convergence gate vs frozen reference | **PASS**, `field_relative_l2 = 0.000000e+00` |
| transport settings | none — `MPI_TRANSPORT_ENV = {}` |

## Accepted

| card | what | steady-state effect |
|---|---|---|
| A5, C4b, E1, E4, B2 | Wave 1 bundle | not separable at steady state (see correction) |
| A2b, C4a | reuse the inverse's density cache; pre-store opacity logarithms | part of −7.7% |
| C3 | Planck recurrence, early exit, rolling group bound | part of −7.7% |
| A6 | hoist `log(ne)` | no-op, kept |
| **H1** | **narrow the Biermann CC halo, 28 → 8 components** | **−13.0%** |
| B1 | clear only the flux components written | −1.4% (noise) |
| H2 | copy only the register components read | +0.5% (noise) |

## Rejected, with the reason

| card | why |
|---|---|
| **A4, A4b** | `UCX_RNDV_THRESH` optimised a warm-up transient; ~10% slower at steady state |
| A1, A1b | capping the inverse at one iteration is no faster than 48 — the loop is not the cost |
| C2 | slower under both protocols; removing table *searches* does not help a kernel waiting on table *reads* |
| Occupancy / `LaunchBounds` | forcing 166 → 128 registers doubled occupancy and gained nothing; spilling grew 80 → 192 B |
| **A7** | requires `nghost = 3`; measured 89.40% memory, outside the 60–80% band |
| E2 | inert on this deck — it gates on `report_diagnostics_ == false`, which the deck sets true |
| C5 step 2 | ~0.1%, and larger than the card implies (the three reducers span different face counts) |

## The main lesson

Six of the ten conclusions in this ledger were wrong on first measurement, and every one
of those errors came from the measurement, not the code: a 3-cycle window that sat inside
a transport warm-up, and a register count used as a proxy for speed. The two rules that
would have caught all of them are in the **Protocol change** section: measure the steady
state over at least 20 cycles and report the series, and never accept a proxy metric in
place of wall clock.

What survives is largely one idea, found by profiling rather than from the plan: **the
Biermann subcycle repeatedly moved, cleared or copied the full 28-component conserved
vector where it touches two or three components** (H1, B1, H2). H1 alone is worth more
than every plan card that was accepted.
