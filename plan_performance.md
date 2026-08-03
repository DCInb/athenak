# DCI_3D performance improvement plan

Date: 2026-08-02 (rev 2 — regenerated from `review.md`)
Supersedes: rev 1 (2026-08-01)
Scope: `src/laser/`, `src/two_temperature/`, `src/mhd/biermann_battery*` +
`mhd_biermann_subcycle.cpp`, and the `src/materials/` closure engine they all call into.
Target deck: `DCI_3D/dci_3d.athinput` (512×256×256, 32³ blocks = 1024 blocks, 8× V100-16GB,
RK2, tabular CH/He, 20-group AP-FLD mixed-table opacity, Biermann subcycling).

---

## 0. How an agent should use this document

This is a **work program**, not prose. Read §1–§4 once, then execute §6 task cards in the
order given by §5.

**Rules of engagement — do not deviate without saying so explicitly:**

0. **Run card A0 first.** It is pure diagnosis, changes no source, and its answer determines
   the design of A4 — which targets the largest single cost in the run. Do not start A4
   blind.
1. **One task card per change.** Each card in §6 is self-contained: exact file:line, exact
   change, expected effect, validation gate, acceptance criteria. Do not bundle cards
   unless a card says to.
2. **Never start a card whose `Depends` field is unmet.** In particular, no `CG`-gated
   card may begin before **T0** (§6.0b) is complete.
3. **Measure before and after every card** using the exact protocol in §4.2 — **both the
   GPU-activity and the API-call sections**. A card is not done until its acceptance
   criteria are met with measured evidence written to `DCI_3D/perf_ledger.md`.
4. **Respect the hard constraints in §2.** They are gates on production, not preferences.
   A change that improves wall time and breaks a §2 constraint is a failed change.
5. **Do not touch anything in §7** (already profiled and rejected) or §3.3 (algorithmically
   correct, easy to break).
6. If a card's expected gain does not materialise, record the negative result in the ledger
   with the profile delta and **revert**. Do not iterate on a dead card; move to the next.
7. Small changes whose individual signal is below run-to-run noise (≈3%) must be
   **bundled into one A/B bracket** — the cards say when.

Environment: source `/home/mengqi/Research/bashrc_athenaK` first. Build out-of-source; in
this repo the pgen dispatcher means no `-DPROBLEM` (built-in pgens binary). Always launch
via `mpirun`, even at `-n 1`.

---

## 1. Ground truth — measured baseline

Source: `DCI_3D/load_balance_trials/current_profile/nvprof.world-0.*.log`, rank 0,
3 cycles, 23.30 s wall, 8 ranks. Full analysis in `review.md`.

| Group | Share | Dominant entries (calls) |
|---|---:|---|
| **Tabular EOS closure** | **58.8%** | `CloseBiermannStage` 31.15 (24), `Sync` 14.03 (6), `SynchronizeDualEnergyFromTotal` 4.78 (6), `RefreshMaterialThermodynamics` 3.36 (3), `ApplyDualEnergyFormalism` 3.22 (6), `Exchange` 2.29 (3) |
| **Radiation (AP-FLD)** | **12.9%** | `NewTimeStep` ×4 reducers 8.08 (28), `AddFluxes` ×3 2.66 (18), `Couple` 1.69 (3), `UpdateDiagnostics` 0.43 (30) |
| **Data movement** | **9.6%** | `memcpy HtoD` 4.44 (**45 379**), `memcpy DtoH` 3.49 (**44 256**), `memset` 1.18 (693), `DtoD` 0.49 (52) |
| **Core MHD** | **7.1%** | `CalculateFluxes` 4.23 (18), `ConsToPrim` 1.64 (30), `RKUpdate` 1.26 (6) |
| **Halo pack/unpack** | **4.9%** | `RecvAndUnpackCC` 3.73 (30), `PackAndSendCC` 0.85 (30), FC + BCs 0.32 |
| **One-time init** | **3.9%** | `TwoTemperature::Initialize` 3.51 (1) |
| **Biermann operator arithmetic** | **1.4%** | AddFluxes/EMF/CT/Poynting ~1.1 (20 each), `NewTimeStep` 0.31 (17) |
| **Laser** | **0.4%** | `TraceStraightRays` 0.35 (46; min 1.66 µs, max 3.50 ms) |

Per-interior-cell cost, the number that frames the device side:

| Kernel | ns/cell |
|---|---:|
| `Sync` | 37.7 |
| `CloseBiermannStage` (reduced) | 24.8 |
| `RefreshMaterialThermodynamics` | 18.1 |
| `mhd_2t_dual_energy_sync` | 12.8 |
| `Exchange` | 12.3 |
| `Couple` | 9.0 |
| `CalculateFluxes` x1 — **reference** | 5.6 |
| `RKUpdate` — **reference** | 4.8 |

**One EOS closure costs 5–8× a complete PLM+LLF flux computation on the same cell.**

### 1.1 The host side is larger than the device side — read this before prioritising

The table above accounts for **9.62 s of GPU activity** (derived: 2.996 s = 31.15%).
Wall is **23.30 s**. **59% of the run is not GPU work.** From the same capture's
`API calls` section:

| CUDA API call | Share of API time | Total | Calls | Avg |
|---|---:|---:|---:|---:|
| `cuMemHostRegister` | **29.98%** | **5.565 s** | 10 058 | 553 µs |
| `cudaEventSynchronize` | 27.88% | 5.175 s | 419 | 12.4 ms |
| `cuMemHostUnregister` | **14.87%** | **2.760 s** | 10 058 | 274 µs |
| `cudaDeviceSynchronize` | 11.91% | 2.211 s | 1 864 | 1.19 ms |
| `cudaStreamSynchronize` | 5.63% | 1.045 s | 1 316 | 794 µs |
| `cuMemcpyAsync` | 1.11% | 206 ms | 88 264 | 2.3 µs |
| `cuEventQuery` | 1.29% | 240 ms | 653 966 | 366 ns |

- **Host-memory pinning is 44.85% of API time — 8.33 s in 10 058 register/unregister
  pairs**, ≈3 350 pairs per cycle. **This is the single largest cost in the run, larger
  than `CloseBiermannStage`.**
- **Host waiting on the device is 45.4% of API time** across the three synchronise calls —
  the dt readbacks, reduction results and queue-compaction scans that cards A3, A5 and E1
  target.
- The 88 264 device transfers themselves cost only 206 ms of API time. **The transfers are
  the cheap half; the pinning around them is the expensive half.**

Consequence for this plan: **card A0 is now the highest-priority item**, ahead of A1.

Derived facts an agent should keep in mind:

- `ConsToPrim` = 30 calls = 6 RK stages + **24 Biermann microstages** → 4 microsteps/cycle,
  2 per Strang half-step. **80% of all C2P and CC halo exchanges are microsteps.**
- Only *one* `CloseBiermannStage` kernel variant is in the profile — the reduced
  electron-only branch. The reduced closure already landed and the kernel is still #1.
- Per cycle: **24 blocking halo rounds for Biermann vs 6 for all of MHD.**
- 16 ranks (14.1 s) is **slower** than 8 ranks (10.6 s). Already communication-bound.
- The profile is a **cold 3-cycle start**: it over-weights init (3.9%) and *under*-weights
  the Planck path, whose cost rises with plasma temperature (§6 C3 note).

---

## 2. Hard constraints

### 2.1 GPU memory — a sizing parameter, not a design constraint

The production gate `gpu_memory_60_80_all` requires every one of the eight V100-16GB GPUs
to sit between **60% and 80%** of 16 384 MiB. The measured peak is **11 876 MiB = 72.49%**
(`PROFILE_REPORT.md:691`).

**Do not treat this as a ceiling on algorithm design.** `DCI_3D/README.md` gives the remedy
for missing the band directly: *"Tune the uniform mesh and repeat if it misses."* The grid
resolution is the free parameter. An optimisation that needs more memory is acceptable — it
shifts the calibrated resolution, which is a deck change, not a code constraint.

Two practical rules follow — reporting requirements, not prohibitions:

- **Report the memory delta** in the ledger for any card that allocates, and re-run
  `--mode calibrate` at each checkpoint rather than after every card, so the mesh is
  retuned once instead of drifting out of band unnoticed.
- **Retuning the mesh invalidates every frozen-reference comparison.** Neither the
  byte-exact nor the convergence gate can span a resolution change. If a wave contains
  memory-growing cards, land them together and rebuild the reference once at the checkpoint.

Reference figure for sizing arithmetic: one scalar component costs **47.8 MB/rank** on the
current deck (128 blocks × 36³ × 8 B).

### 2.2 Production gate checks that performance work can break

From `DCI_3D/README.md`. Any candidate must still pass:
`compact_20group_50step`, `compact_output_and_restart`, `finite_nonnegative_3t`,
`causal_timestep_no_collapse`, `laser_and_boundary_energy_closure`, `ch_mass_conservation`,
`restart_continuity`, `resolution_or_opacity_sensitivity`,
`reduced_light_speed_sensitivity`, `physical_light_speed_sensitivity`,
`gpu_memory_60_80_all`.

Most at risk from this plan: `finite_nonnegative_3t` and `causal_timestep_no_collapse`
(cards A1, A3), `laser_and_boundary_energy_closure` (cards E1–E4),
`gpu_memory_60_80_all` (card C5).

### 2.3 The validation gate is the binding constraint on progress

The accepted campaign used a **byte-exact frozen-field gate**. That gate has exhausted its
headroom. Several of the inefficiencies below exist *specifically* to preserve bit-identical
trajectories — e.g. `material_mixture.hpp:286-288`: *"preserving the legacy exp-then-log
inverse trajectory"*.

Every remaining large win (A1, A3, C3, C5) is by construction roundoff-perturbing and
**cannot** pass bit-identity. These are algorithm changes, not bugs.

→ **T0 (§6.0) stands up the convergence gate and must complete first.** It gates roughly
80% of the available speedup. Budget it up front.

---

## 3. Orientation

### 3.1 The two-sentence diagnosis

**Device side:** the mixed CH/He energy→temperature inverse is a **fixed 48-iteration
bisection with no convergence test** (`material_mixture.hpp:592`) forming a **serial
dependency chain** — so it is latency-bound, not FLOP-bound — and the Biermann subcycle
re-derives it, plus a full MPI halo exchange, at *microstep* rather than *macro-step*
cadence.

**Host side:** the run spends more wall time off the GPU than on it, and the dominant host
cost is not kernels or transfers but **10 058 host-memory pin/unpin pairs (8.33 s)** plus
**host waiting on device synchronisation (8.43 s)** — see §1.1. Fix the host side first;
it is cheaper, byte-exact-gateable, and larger.

### 3.2 Why the secant change should over-deliver

Each bisection iteration's `exp` depends on the previous comparison: zero ILP across ~200
transcendentals. Cutting 48 → ~6 iterations cuts *latency* ~8×, not just operation count.
This is why `Sync` at 37.7 ns/cell is ~7× a PLM+LLF kernel despite a FLOP count that does
not justify it. Expect the measured gain to exceed a naive op-count estimate.

### 3.3 Do NOT regress these — they are correct and easy to break

- **Log-mean edge integral** (`biermann_battery.cpp:825-836`, `InverseLogMean` `:126-139`).
  Path-conservative; telescopes exactly for constant `n_e`; reduces to the Graziani shock
  integral for an isothermal jump; cancellation-safe series. Also cheap.
- **Ion energy stays out of the SSPRK recurrence** (`mhd_biermann_subcycle.cpp:94-101`).
  Blending a projected ion energy as a zero-RHS variable silently drops the split update to
  first order. Any fusion of the closure into the update must preserve this.
- **FLD stability from the true flux Jacobian** (`thermal_radiation.cpp:964-972`), not the
  optically-thick bound. C5 removes *duplication*, never the estimate itself.
- **The reduced Biermann closure** + `is-1..ie+1` extent
  (`mhd_biermann_subcycle.cpp:262-288`, `driver.cpp:406-411`). Already landed, correct.
- **The IONMIX interval-energy cache** (`ionmix_two_temperature_table.hpp:413-444`).

### 3.4 Already done since rev 1 — do not redo

| rev-1 item | Status | Evidence |
|---|---|---|
| C1 `Locate` recomputing `log(unit_scale)` | **DONE** | `log_density_to_cgs` / `log_temperature_to_kelvin` are stored fields, `ionmix_two_temperature_table.hpp:157-158`, used at `:389`, `:408` |
| C1 `Min/MaxDensity/TemperatureCode` `exp`/divide per call | **DONE** | stored at `:159-162`, returned directly `:362-379` |
| A2 restrict Biermann closure extent | **DONE** | `mhd_biermann_subcycle.cpp:268-282` |
| A2 per-stage `full_thermodynamics` toggle | **DONE** | `driver.cpp:406-411`; only the reduced kernel appears in the profile |
| IONMIX inverse interval-energy cache | **DONE** | commit `6efa603e`, `ionmix_two_temperature_table.hpp:413-444` |
| Reduced tabular Biermann closure | **DONE** | commit `733bfbb7`, `two_temperature.cpp:773-803` |

---

## 4. Protocols

### 4.1 Validation gates

Every card is tagged **BE** or **CG**.

**BE — byte-exact field gate.** For pure data-movement, scheduling, launch-shape, memory
and diagnostics-gating changes. Compare against a frozen executable at a representative
later-time restart: all fields byte-identical, plus identical energy accounting, laser
remaining-ray/conservation counters, and timestep-limiter sequence. Any bit difference is a
failure, not a tolerance question.

**CG — convergence / conservation / physics-equivalence gate.** For roundoff-perturbing
algorithm changes. Requires T0 (§6.0). Must demonstrate:
- solution converges to the same physical answer under refinement;
- energy, `div B`, and CH mass conservation match the reference within stated tolerance;
- the timestep-limiter sequence does not collapse (`causal_timestep_no_collapse`);
- ion/electron/group energies stay finite and non-negative, with no new EOS-table clamps
  (`finite_nonnegative_3t`);
- laser power closure preserved with ≥2× margin to reflection and wave caps.

**Noise rule.** Run-to-run noise on sustained runs is ≈3%. Any card whose profiled kernel
gain is below that must be A/B-bracketed in **opposite orders** and, where the card says so,
bundled with its siblings so the aggregate clears noise. This rule is what rejected
fused-`RKUpdate` and standalone rolling-Planck in rev 1.

### 4.2 Measurement protocol

Kernel-level profile (the baseline in §1 was produced this way):

```bash
mpirun -n 8 nvprof --log-file <out>/nvprof.world-%q{OMPI_COMM_WORLD_RANK}.log \
  DCI_3D/build/src/athena --kokkos-map-device-id-by=mpi_rank \
  -d <out>/run -i DCI_3D/dci_3d.athinput \
  time/nlim=3 time/ndiag=1 \
  output1/dt=-1.0 output2/dt=-1.0 output3/dt=-1.0 \
  output4/dt=-1.0 output5/dt=-1.0 output6/dt=-1.0
```

**Always read BOTH sections of the nvprof output.** The `GPU activities` table is only
41% of wall time on this configuration; the `API calls` table below it holds the largest
single cost (§1.1). A card that improves `GPU activities` while inflating
`cuMemHostRegister` or `cudaEventSynchronize` is a regression.

Wall-clock A/B: sustained run, opposite orders, ≥3 repetitions each.

Memory check **at each checkpoint** (not after every card — §2.1):
```bash
python3 DCI_3D/run_case.py --clean --mode calibrate    # returns 2 unless every GPU is 60–80%
```
If it returns 2, retune the mesh and rebuild the frozen reference before continuing.

**Re-profile at a later-time restart before ranking radiation against EOS work.** The
3-cycle cold profile systematically under-weights the Planck path (C3).

### 4.3 Ledger

Append one entry per card to `DCI_3D/perf_ledger.md`:
`card id | commit | gate used | before/after kernel % | before/after wall | memory % | verdict (accepted / rejected+reverted)`.
Negative results are as valuable as positive ones — record and move on.

---

## 5. Execution order

```
A0  Host-pinning diagnosis            [DO THIS FIRST — largest single cost, pure diagnosis]
T0  Convergence gate                  [BLOCKING for every CG card; start in parallel with A0]
     │
     ├─ WAVE 1 (BE, no gate dependency)
     │    A4  gpu_aware_mpi + persistent host buffers      ← acts on A0's finding
     │    A5  fold the two Biermann MPI_Allreduce into one
     │    B1  drop the 6 per-microstage memsets
     │    B2  overlap the U and B halos in the Biermann stage list
     │    C4b opacity axes stored in log space
     │    E1  launch the laser DDA over the active count
     │    E2  gate laser diagnostics (frees 478 MB)
     │    E3  O(1) block lookup
     │    E4  one-time global block info
     │    F1b fold Spitzer constants
     │       → A4 is bracketed ALONE (it should be large enough to see);
     │         bundle the other nine into ONE A/B bracket
     │
     ├─ CHECKPOINT 1: re-profile BOTH the GPU-activity and API-call sections;
     │                re-run calibrate
     │
     ├─ WAVE 2 (CG — requires T0)
     │    A1  safeguarded secant inverse            ← LARGEST DEVICE-SIDE WIN
     │    A1b remove the exp↔log round-trips (do together with A1)
     │    A2b remove the redundant forward eval after each electron inverse
     │    C4a opacity log-value pre-storage
     │       → A1+A1b+A2b are one commit; C4a is separate
     │
     ├─ CHECKPOINT 2: re-profile. The §1 ranking will have changed materially.
     │
     ├─ WAVE 3 (CG + BE, re-ranked after checkpoint 2)
     │    A3  amortise the Biermann dt limit         ← also cuts cudaEventSynchronize
     │    C5  radiation limiter: merge reducers, then stride (read the state-ordering
     │        constraint in the card — it is NOT fusible into AddFluxes)
     │    C3  Planck recurrence + early exit + rolling bound
     │    C2  sound-speed density-location cache
     │    A6  remove redundant log(ne) in biermann_newdt
     │    B3  ghost-extent audit for Exchange / Couple / Refresh
     │
     ├─ CHECKPOINT 3: re-profile at a LATER-TIME RESTART, not a cold start.
     │                Re-measure rank imbalance (D1) — its absolute weight has fallen.
     │
     └─ WAVE 4 (structural — schedule last, own study)
          A7  block-local multirate Biermann substepping
          D1  laser-aware dynamic balancing (only if checkpoint 3 still shows a large spread)
```

---

## 6. Task cards

### 6.0a A0 — Diagnose the 10 058 host pin/unpin pairs

- **Tier:** highest priority. **Pure diagnosis — no source change in this card.**
- **Gate:** none (measurement only) · **Depends:** —
- **Target:** 8.33 s of 18.6 s API time; the largest single cost in the run (§1.1)
- **Problem:** `cuMemHostRegister` (10 058 calls, 5.565 s) + `cuMemHostUnregister` (10 058,
  2.760 s) = **44.85% of CUDA API time**, ≈3 350 register/unregister pairs per cycle at
  ~830 µs per pair. By comparison the 88 264 device transfers those pairs accompany cost
  only 206 ms.
- **What is NOT the cause:** the only explicit pinned allocation in the codebase is
  `src/laser/laser.hpp:236-237` (`Kokkos::SharedHostPinnedSpace`), allocated once. Every
  `Kokkos::create_mirror_view` in `src/laser/` mirrors into plain `HostSpace`, which does
  not pin. **No application code calls `cudaHostRegister`.** Do not go looking for one.
- **Two credible causes — determine which:**
  1. **UCX/hpcx registering MPI send/receive buffers for RDMA**, with its registration
     cache being defeated. Supporting evidence: 221 `cudaMallocAsync` and 277 `cudaFree`
     calls, which can invalidate a registration cache; and `gpu_aware_mpi = false` means
     every halo packet transits a pageable host buffer that MPI must register.
  2. **Kokkos staging pageable host buffers** for device↔host `deep_copy`.
- **Method:**
  1. Open the existing `.nsys-rep` traces in `DCI_3D/profile_20260729/` and attribute
     `cuMemHostRegister` to a caller stack. This is the fastest route and needs no new run.
  2. If inconclusive, re-run with UCX registration-cache diagnostics
     (`UCX_LOG_LEVEL=info`, and check `UCX_RCACHE_*` / `UCX_MEMTYPE_CACHE` behaviour) and
     with `nsys profile --cuda-memory-usage` to correlate against buffer lifetimes.
  3. Correlate the 10 058 count against per-cycle halo-round counts (30 CC rounds/cycle ×
     128 blocks) and against laser MPI wave counts to identify which subsystem dominates.
- **Acceptance:** a written attribution in `DCI_3D/perf_ledger.md` naming the subsystem and
  mechanism, plus a recommendation for whether A4 alone fixes it or a separate buffer-reuse
  change is needed. **Do not proceed to A4 blind** — A4's design depends on this answer.
- **Note:** some fraction of this may be `nvprof` interaction. The `nsys` traces
  discriminate; if profiler overhead is implicated, confirm with a wall-clock A/B of
  `gpu_aware_mpi` on/off *without* a profiler attached before investing further.

### 6.0b T0 — Stand up the convergence gate

- **Tier:** blocking prerequisite
- **Why:** §2.3. Without it, no CG card can be validated, and CG cards are ~80% of the
  available speedup.
- **Build:** a reproducible harness that, given a frozen reference executable and a
  representative **later-time restart** (not `t=0`), reports:
  1. field convergence under refinement (reuse `--compact-scale 2` from
     `DCI_3D/run_case.py`);
  2. total energy, `div B`, and CH-mass conservation drift vs reference, with tolerances;
  3. the timestep-limiter sequence (macro dt, Biermann substep count, radiation transport
     and source limits) vs reference;
  4. laser launched / deposited / escaped / remaining power closure and cap margins;
  5. `finite_nonnegative_3t` clamp counts from `eos_query_flags`.
- **Reuse:** `DCI_3D/verify_production_gate.py` already computes most of these metrics —
  extend it or factor out its metric layer rather than writing a parallel implementation.
- **Acceptance:** running the harness with the reference against *itself* reports pass on
  all five with zero drift; running it against a deliberately perturbed build (e.g. change
  the bisection to 47 iterations) reports the perturbation with a sane magnitude.
- **Deliverable:** `DCI_3D/convergence_gate.py` + a documented tolerance table, committed.

---

### WAVE 1 — byte-exact, no gate dependency

#### A4 — GPU-aware MPI, persistent host buffers, on-device dt
- **Gate:** BE · **Depends:** **A0** · **Target:** the 8.33 s pinning cost + the 9.6% GPU memcpy group
- **Where:** `DCI_3D/dci_3d.athinput:188` (`gpu_aware_mpi = false`);
  `src/laser/laser_mpi.cpp:37-38, 46-50`; `src/driver/driver.cpp:351`; plus whatever A0
  identifies in `src/bvals/`
- **Problem:** 45 379 HtoD + 44 256 DtoH per 3 cycles ≈ 30 000/cycle at 7–9 µs each — fixed
  per-transfer overhead, not bandwidth — **and** the 10 058 host pin/unpin pairs that cost
  25× more than the transfers themselves (§1.1). Sources: host-staged MPI packets, the
  per-microstep dt readback, and `create_mirror_view` + `deep_copy` on *every*
  `PrepareOutgoingRays` call for the send counts/offsets.
- **Change:**
  1. **Act on A0's attribution first.** If UCX registration is the cause, the fix may be
     persistent registered buffers or a UCX cache setting rather than — or in addition to —
     the flag.
  2. Qualify and enable `gpu_aware_mpi = true`. The laser path already branches on the flag
     (`laser_mpi.cpp:258-269, 283-284`); verify the bvals path does too.
  3. Hold persistent host mirrors for `mpi_send_counts_` / `mpi_send_offsets_`, as is
     already done for the packet buffers (`laser.hpp:236-237`).
  4. Keep the Biermann dt limit on-device where the control flow allows (see also A3/A5).
- **Acceptance:** ≥50% reduction in HtoD+DtoH call count **and** a large reduction in
  `cuMemHostRegister`/`Unregister` call count; byte-exact fields; no memory regression.
  Bracket this card **alone** — it should be big enough to clear noise on its own.
- **Risk:** GPU-aware MPI can be silently broken in the local hpcx build — validate with a
  small case first and keep the flag switchable.

#### A5 — One `MPI_Allreduce` per microstep instead of two
- **Gate:** BE · **Depends:** — · **Target:** 4 barriers/cycle
- **Where:** `src/driver/driver.cpp:359` (invalid-flag MAX), `:371` (limit MIN)
- **Problem:** two blocking collectives per microstep where one suffices.
- **Change:** encode "invalid" as a sentinel that MIN preserves (e.g. a negative or NaN-free
  sentinel below any legal limit), or reduce a 2-element buffer with one call. Preserve the
  property that **every rank reaches the same collective even when one rank's state is
  invalid** — the current code documents this at `:352-356` and it must not regress.
- **Acceptance:** one collective per microstep; the invalid-limit abort path still fires
  correctly under an injected fault; byte-exact fields.

#### B1 — Drop the six per-microstage memsets
- **Gate:** BE · **Depends:** — · **Target:** part of the 1.18% / 693 memsets
- **Where:** `src/mhd/mhd_biermann_subcycle.cpp:67-69` (uflx), `:149-151` (efld)
- **Problem:** six `deep_copy(…, 0.0)` per microstage purely because
  `BiermannBattery::AddFluxes` / `AddEMFs` use `+=`. 48 memsets/cycle.
- **Change:** have the first writer in each array use `=` on first touch; keep `+=` for the
  legacy stage-coupled path, which genuinely adds to ideal-MHD fluxes.
- **Watch:** the subcycle path skips `biermann_face_e1/e2/e3` entirely
  (`biermann_battery.cpp:396`, `:462`, `:528` are guarded by `!edge_integral_subcycle`), so
  the "first writer" differs between the two paths. Handle both.
- **Acceptance:** memset count drops by ≈48/cycle; byte-exact fields.

#### B2 — Overlap the U and B halos in the Biermann stage
- **Gate:** BE · **Depends:** — · **Target:** one latency round × 8 microstages/cycle
- **Where:** `src/mhd/mhd_tasks.cpp:136` — `RestrictB` takes `b_recvu` as its dependency
- **Problem:** the B-field halo is serialised behind the U halo although B's restriction
  does not need U's ghosts. On the macro list this is inherited from upstream and amortised;
  at microstep cadence it is paid 8× per cycle.
- **Change:** branch `b_restb` from `b_ct` directly and let the scheduler overlap rounds 2
  and 3. Keep `b_bcs` dependent on both `b_recvu` and `b_recvb`.
- **Acceptance:** byte-exact fields; measurable drop in stage wall time at 8 and 16 ranks.
- **Risk:** verify no hidden read of `w0` ghosts inside `RestrictB` for this configuration.

#### C4b — Store opacity table axes in log space
- **Gate:** BE · **Depends:** — · **Target:** 12 `log`/cell/lookup
- **Where:** `src/two_temperature/opacity_table.hpp:73-78, 98-103`
- **Problem:** `Locate` with `log_coordinates` computes `log(d)`, `log(density(lower))`,
  `log(density(upper))` — 3 `log` per axis, 6 per material, **12 per mixed lookup** — and
  the two axis values are compile-time constant per table.
- **Change:** store the axes pre-logged at load (`opacity_table.cpp`), leaving only `log(d)`
  and `log(t)` at runtime.
- **Acceptance:** byte-exact (the arithmetic is a reordering of exactly-representable
  loads); if it is not byte-exact, reclassify as CG and move to Wave 2.

#### E1 — Launch the laser DDA over the active count
- **Gate:** BE · **Depends:** — · **Target:** near-empty tail launches
- **Where:** `src/laser/laser_trace.cpp:627` (and `:1439` region in
  `TraceRefractiveRays`); active count already returned at `:1061`
- **Problem:** the queue is compacted (`CompactActiveQueue`, `:486-501`) but the kernel is
  then launched over the full `nrays_` = 4096 anyway, so nearly every thread exits at `:630`
  in the tail iterations. Profile evidence: min kernel time 1.66 µs vs max 3.50 ms.
- **Change:** launch `RangePolicy(0, active_count)`. Same for `PrepareOutgoingRays`
  (`laser_mpi.cpp:68`) and `BookRemainingRays` (`laser_trace.cpp:544`) where an active count
  is available.
- **Acceptance:** byte-exact laser diagnostics and deposition field; reduced launch time in
  the trace loop.

#### E2 — Gate laser diagnostics
- **Gate:** BE · **Depends:** — · **Target:** ~8 of ~11 atomics/segment, ~400 MB/stage of
  clearing traffic, and 478 MB/rank of footprint
- **Where:** `src/laser/laser_trace.cpp:858-876` (straight), `:1405-1423` (refractive);
  allocation `src/laser/laser.cpp:528`; flag `src/laser/laser.hpp:206`
- **Problem:** `cell_data` is 12 components × 47.8 MB = **573 MB/rank**, and 10 of the 12
  are pure diagnostics (segment count, tau, path, direction×path, dispersion×path,
  midpoint×path — `laser.hpp:140-142`). Production needs components 0 and 1.
  `report_diagnostics_` gates neither the allocation nor the atomic writes.
  `ClearInstantaneousData` (`laser_tasks.cpp:71-78`) additionally rewrites all 12 every
  stage (~400 MB of writes).
- **Primary motivation is the atomics and the clearing traffic, not the footprint.** Per
  §2.1 memory is a sizing parameter; do this card because ~8 of ~11 `atomic_add`s per
  segment and a 400 MB/stage memset are wasted work, and take the 478 MB as a bonus.
- **Change:** allocate `ncell_data = 2` and skip diagnostic atomics when
  `report_diagnostics_ == false`; keep the full 12 when it is true. Update the `laser` output
  variable path to handle both widths.
- **Constraint:** the production gate requires *"every laser record must contain an integral
  `waves` diagnostic"* — that is a `LaserDiagnostics` scalar, not a `cell_data` component;
  confirm before gating. **Do not** break `laser_and_boundary_energy_closure`.
- **Acceptance:** byte-exact deposition and power closure; measurable drop in
  `laser_clear_stage` and atomic traffic. Record the memory delta (≈72.5% → 69.6%) in the
  ledger for the checkpoint retune; it is not itself an acceptance criterion.

#### E3 — O(1) block lookup
- **Gate:** BE · **Depends:** — · **Target:** worst case 1024 → 1 per block crossing
- **Where:** `src/laser/laser_trace.cpp:49-55` (`FindLocalBlock`), `:66-73`
  (`FindGlobalBlock`); call sites `:363, 365, 941, 945, 1000, 1016, 1026, 1439, 1454, 1468`
- **Problem:** O(N_blocks) linear scans executed inside the ray kernel on every block
  crossing. The production mesh is static, uniform, single-level, 1024 blocks.
- **Change:** compute the logical block index by arithmetic from the mesh origin and block
  size; **retain the linear scan as a fallback** for `multilevel` meshes and select at
  runtime. Preserve the exact half-open `[min, max)` containment semantics of
  `ContainsPoint` (`:40-46`, `:57-64`) — the laser transport has known sub-ULP face-stall
  sensitivity (commits `cb597851`, `fe0b738c`).
- **Acceptance:** byte-exact ray paths, segment counts, and conservation counters. This is
  the highest-risk BE card; if bit-exactness fails, revert rather than loosen.

#### E4 — Build global block info once
- **Gate:** BE · **Depends:** — · **Target:** 1 H2D + O(N_blocks) host work per stage
- **Where:** `src/laser/laser.cpp:679-720`, called from `laser_tasks.cpp:52` every stage
- **Problem:** all `nmb_total` block descriptors are rebuilt on the host and copied H2D
  every RK stage, on a static mesh.
- **Change:** build once at construction; rebuild only when the mesh changes (`multilevel`
  remesh / load rebalance). Keep the byte-identical `LeftEdgeX` construction at `:697-716` —
  the comment there records that algebraically-equivalent interpolation differs by ulps at a
  shared rank face.
- **Acceptance:** byte-exact; one fewer H2D per stage.

#### F1b — Fold the Spitzer constants
- **Gate:** BE for the constants · **Depends:** — · **Target:** trivial but free
- **Where:** `src/two_temperature/two_temperature.cpp:46-49`
- **Change:** fold `8.0*sqrt(2.0*pi)` and `pow(electron_charge_cgs, 4)` into a single
  `constexpr`. **Leave `pow(thermal_speed_squared, 1.5)` alone in this card** — replacing it
  with `v*sqrt(v)` is roundoff-changing; that goes in Wave 2 as F1a.
- **Acceptance:** byte-exact (verify — the compiler may already fold these, in which case
  record "no-op" and move on).

**WAVE 1 bundling:** A4 is bracketed **alone** — per §1.1 it targets the largest single
cost in the run and should clear noise by itself. The other nine cards have individual
gains below the 3% noise floor; land them as one bracket and A/B in opposite orders.

---

### WAVE 2 — the main event (CG, requires T0)

#### A1 — Safeguarded secant replacing the fixed-48 bisections
- **Gate:** CG · **Depends:** T0 · **Target:** the 58.8% EOS group — **largest single win**
- **Where:** `src/materials/material_mixture.hpp:592-601` (mixed inverse, the hot one);
  also `:1536`, `:1590`, `:1643`
- **Problem:** fixed 48 iterations, no convergence test, no early exit, plain bisection.
  48 bisections of a ~6-decade log-T bracket resolve far below the table's own interpolation
  error — the last ~30 iterations resolve nothing physical. And it is a **serial dependency
  chain**, so it is latency-bound (§3.2).
- **The correct method already exists in this file:** `material_mixture.hpp:1183-1218` — a
  safeguarded regula-falsi with a real tolerance test (`:1168-1173`), best-residual
  tracking, and a bisection fallback at `:1220-1246` if it stalls. **Copy that structure.**
- **Change:** replace the four fixed-48 loops with the safeguarded secant. Tolerance must be
  tied to the table's interpolation error, not to machine epsilon — reuse the
  `relative_tolerance` construction at `:1168-1171`. Keep the bracket-preserving fallback.
- **Compounding context:** `Sync` (`two_temperature.cpp:376-455`) runs up to **four** of
  these per cell — `MinimumPressureEnergyState`, then
  `PressureEnergyFromRhoSpecificEnergies` (ion+electron), then
  `StateFromRhoSpecificEnergies` (ion+electron) — ≥400 dependent transcendentals per cell.
- **Acceptance:** iteration count drops to ≤8 typical; full CG pass; expect the wall-time
  gain to **exceed** a naive op-count estimate (§3.2). Instrument the iteration histogram
  once and record it in the ledger.
- **Risk:** secant can stall on a flat residual near table edges. The fallback must be
  exercised — inject an edge case and confirm it converges.

#### A1b — Remove the `exp`↔`log` round-trips (land with A1)
- **Gate:** CG · **Depends:** A1 · **Target:** 2 transcendentals × 48 iters × 2 components
- **Where:** `src/materials/material_mixture.hpp:593-595` + `:286-288`; same pattern at
  `:1643-1652`
- **Problem:** the loop computes `exp(log_trial)` to get `T`, then
  `MixtureComponentEnergyFromCachedDensity` immediately does `log(temperature)` to recover
  `log_trial`. The comment at `:286-288` states this exists to preserve *"the legacy
  exp-then-log inverse trajectory"* — i.e. purely for the byte-exact gate that T0 replaces.
- **Change:** thread `log_trial` through directly.
- **Acceptance:** same CG pass as A1; commit together so the bracket measures both.

#### A2b — Remove the redundant forward evaluation after each electron inverse
- **Gate:** CG · **Depends:** T0 · **Target:** directly inside the 31% kernel
- **Where:** `src/materials/material_mixture.hpp:1389-1392`
- **Problem:** `ElectronStateFromRhoSpecificEnergy` runs the inverse, then calls
  `MixtureComponentFromRhoTemperature` at the returned temperature — a full **uncached**
  forward evaluation that re-locates density in *both* tables. The inverse already returns
  pressure and energy at that temperature (`:602-603`).
- **Change:** return the inverse's own state. If a forward pass is genuinely needed for
  query-flag fidelity, use the **cached** variant (`MixtureComponentFromCachedDensity`) so
  the density location is not recomputed.
- **Acceptance:** `CloseBiermannStage` ns/cell drops measurably; `eos_query_flags` lifetime
  diagnostics unchanged in kind (verify the flag union is preserved).

#### C4a — Pre-store `log(value)` for geometric opacity interpolation
- **Gate:** BE if bit-preserving, else CG · **Depends:** T0 (fallback) · **Target:** large
  fraction of the 12.9% radiation group
- **Where:** `src/two_temperature/opacity_table.hpp:118-122`; loader
  `src/two_temperature/opacity_table.cpp`
- **Problem:** geometric interpolation recomputes **4 `log` + 1 `exp` per group per lookup**.
  The deck uses `geometric` opacity interpolation with a **mixed** (two-material) table and
  20 groups; `Couple` performs two kinds (absorption, emission) → ~320 `log` + 80 `exp` per
  cell. `AddFluxes` and `NewTimeStep` pay the transport kind on every face.
- **Precedent:** the IONMIX table already does exactly this — `log_values` pre-stored and
  consumed at `ionmix_two_temperature_table.hpp:288-290`.
- **Change:** pre-store `log(value)` at load with a zero-safe sentinel; the runtime path
  becomes linear-in-log plus one `exp`. Mirror the IONMIX `FieldAllowsGeometricInterpolation`
  guard so non-positive entries fall back to linear.
- **Acceptance:** ≥3× fewer transcendentals per opacity lookup; radiation group share drops;
  CG pass (or BE if it happens to be bit-preserving).

---

### WAVE 3 — re-ranked after checkpoint 2

#### A3 — Amortise the Biermann dt limit
- **Gate:** CG (tight tolerance) · **Depends:** T0, A5 · **Target:** `BiermannBattery::NewTimeStep` + readbacks + barriers
- **Where:** `src/driver/driver.cpp:344-431`, esp. `:351`
- **Problem:** every microstep runs a full-grid reduction + host readback + collectives, but
  the limit is a hyperbolic wave speed (electron drift + thermal-magnetic,
  `biermann_battery.cpp:1280-1490`) that varies on the **macro**-step timescale.
- **Change:** compute the limit once per half-step (or every N microsteps) and reuse it for
  equal substeps. Retain `biermann_max_stability_ratio_last_cycle` (`driver.cpp:420`) as the
  safety monitor and **fail loudly** if the realised ratio exceeds 1.
- **Acceptance:** (N−1)/N fewer `NewTimeStep` calls, readbacks and barriers; stability ratio
  stays ≤1 across a sustained run; `causal_timestep_no_collapse` passes.
- **Note:** super-time-stepping does **not** apply here — the limit is hyperbolic, not
  parabolic. Correctly rejected in rev 1. Do not revisit.

#### C5 — Reduce the radiation limiter's cost without changing the state it sees
- **Gate:** BE for step 1, CG for step 2 · **Depends:** C4a (primary lever), T0 (for step 2)
- **Where:** `src/two_temperature/thermal_radiation.cpp:974-1237` vs `:688-845`
- **Problem:** `NewTimeStep` evaluates, per face per group, the same
  `X{1,2,3}FaceMaterialState` / `mixed_opacity.Locate` / `opacity.Get` / `FLDProperties`
  that `AddFluxes` evaluates; only `FLDFaceStabilityRate` is new. Measured **8.08% for the
  limiter vs 2.66% for the fluxes** — 3× as long deciding the step size as taking it.
- **READ THIS BEFORE CHANGING ANYTHING — the obvious fixes are invalid.**
  `AddFluxes` runs mid-RK-stage. `NewTimeStep` is reached from `MHD::NewTimeStep`
  (`mhd_newdt.cpp:204`) via `MHD::TwoTempExchange` (`mhd_tasks.cpp:676`), i.e. **after**
  `Exchange`, `Couple` and `RefreshMaterialThermodynamics`. Those sources change T_e and
  the group energies, so the opacity and every FLD face state genuinely differ between the
  two calls. `mhd_tasks.cpp:84-90` states the intent explicitly: *"compute the next
  timestep from the post-source state in TwoTempExchange instead of caching a stale
  pre-source limit here."*
  Therefore **do not fuse the `Max` reduction into `AddFluxes`**, and **do not cache the
  per-face rate during `AddFluxes`** — both reintroduce precisely the stale pre-source
  limit that comment describes removing. (Memory is not the objection; correctness is.)
  This is code duplication, not computational redundancy.
- **Change — what is actually available, in order of value:**
  1. **C4a is the primary lever.** The dominant per-face cost in *both* kernels is the
     geometric opacity interpolation. Making that cheap once benefits the limiter and the
     fluxes together with no correctness question. Land C4a first, then re-measure this
     card — it may largely dissolve.
  2. **Merge the three directional `Max` reductions into one multi-reducer** (pattern
     already at `biermann_battery.cpp:1318-1321`). Removes two launches and two host
     syncs per call. **BE**, pure win.
  3. **Optionally stride the limiter** — recompute the transport dt every N steps with a
     safety margin. This is *explicit, monitored* staleness rather than accidental, and is
     defensible where fusion is not. **CG**, and must hold `causal_timestep_no_collapse`
     with a monitor that fails loudly if the realised rate exceeds the assumed limit.
- **Acceptance:** step 2 byte-exact with 2 fewer launches/syncs per call; after C4a,
  re-measure and only pursue step 3 if the limiter is still a large share.

#### C3 — Planck: recurrence, early exit, rolling bound
- **Gate:** CG · **Depends:** T0 · **Target:** `Couple`, `Initialize`, source dt reducer
- **Where:** `src/two_temperature/thermal_radiation.cpp:54-77` (`PlanckIntegral`), `:80-86`
  (`PlanckGroupFraction`), `:910` (`Couple` call site)
- **Three separate problems:**
  1. **`:67-74`** — 64 independent `exp(-n*x)` calls. `exp(-n·x) = q^n` with `q = exp(-x)`:
     one `exp` plus a running multiply. ~64× fewer transcendentals in the tail branch.
  2. **`:66-74`** — no early exit. Terms decay as `e^(−nx)`; add
     `if (term < eps*tail) break;`. `x` is spatially smooth so warp divergence is small.
  3. **`:80-86`** — `PlanckGroupFraction` evaluates *both* boundaries, so a 20-group cell
     makes 40 calls where 21 suffice. **The rolling-bound version is already correct in the
     same file** at `:1203, 1214, 1217` (source dt reducer) but is not used by `Couple` or
     `Initialize`. Two implementations of one quantity in one file is a maintenance hazard
     as well as a performance bug.
- **Important scaling note:** `PlanckIntegral` cost is **state-dependent** — early-out for
  `x ≤ 0` and `x ≥ 50`, cheap polynomial for `x < 0.5`, expensive loop only in between. The
  3-cycle cold profile has most boundaries in the `x ≥ 50` branch, so **`Couple`'s 1.69% is
  a floor, not a steady-state value.** Measure this card at a later-time restart
  (checkpoint 3), not against the cold baseline.
- **rev-1 history:** rolling-Planck reuse was rejected **as a standalone application-level
  change** because it did not clear noise. Land items 1–3 together, after A1/C4a, so the
  aggregate signal is measurable.
- **Acceptance:** measured at a later-time restart; group-emission fractions match reference
  within CG tolerance; `finite_nonnegative_3t` passes.

#### C2 — Sound-speed density-location cache
- **Gate:** BE or CG · **Depends:** — · **Target:** `Sync`, `Refresh`, `Initialize`
- **Where:** `src/materials/material_mixture.hpp:803-866`
- **Problem:** `TabularSoundSpeedSquared` performs **16 full mixed forward evaluations** per
  cell (2 components × 4 finite-difference points × 2 materials), re-`Locate`-ing density
  each time although only **3 distinct densities** occur (`density_low`, `density`,
  `density_high`).
- **Change:** pass a shared `MixedDensityCache` per distinct density, as the inverse already
  does (`:530-532`). The same fix applies to `ElectronHeatCapacityFraction` (`:941`) on the
  exchange path.
- **Acceptance:** roughly halved table traffic in the sound-speed path; BE preferred.

#### A6 — Remove redundant `log(ne)` in `biermann_newdt`
- **Gate:** BE (hoist only) · **Depends:** — · **Target:** ~3 `log`/cell/microstep
- **Where:** `src/mhd/biermann_battery.cpp:1351, 1367, 1372, 1390` and the x2/x3 blocks
- **Problem:** `const Real log_ne = log(ne)` for the **centre** cell is redeclared once per
  direction (x1, x2, x3). Neighbour `log(ne)` values are additionally recomputed by each
  adjacent cell.
- **Change:** hoist the centre `log_ne` above the directional blocks (pure BE hoist).
  Optionally, precompute `log(ne)` into a 47.8 MB scratch array in one pass and read the 7
  stencil values from it. Memory is not the objection (§2.1) — judge it on the real
  trade: one extra global write + 7 reads per cell versus 7 `log` evaluations. On V100 a
  double-precision `log` is expensive enough that the array version should win; measure it.
- **Acceptance:** byte-exact for the hoist; the scratch variant is BE too if the values are
  identical, and must report its memory delta.

#### B3 — Ghost-extent audit for `Exchange` / `Couple` / `Refresh`
- **Gate:** BE if the extents are provably unread, else CG · **Depends:** — · **Target:** up to 30% off three kernels
- **Where:** `src/mhd/mhd_tasks.cpp:669-677` (all three run `0..n1m1`);
  `src/two_temperature/two_temperature.cpp:688-696`
- **Problem:** these operator-split sources run over the **ghost-inclusive 36³** (46 656
  cells = 1.42× the interior) so the next stage's reconstruction sees consistent ghosts —
  but the halo exchange that follows would supply exactly those values.
- **Change:** determine per kernel which ghost layers are genuinely consumed before the next
  halo exchange overwrites them, and narrow the extent accordingly. The Biermann path has
  already been narrowed correctly (`mhd_biermann_subcycle.cpp:262-288`) — use that analysis
  as the template.
- **Acceptance:** byte-exact if the narrowed cells are provably overwritten before use;
  otherwise CG. Each 1.42× → 1.0× is a 30% cut on a top-6 kernel.

#### F1a — `pow(v, 1.5)` → `v*sqrt(v)`
- **Gate:** CG · **Depends:** T0 · **Where:** `src/two_temperature/two_temperature.cpp:47`
- Bundle with any other Wave-3 CG card; too small to bracket alone.

---

### WAVE 4 — structural

#### A7 — Block-local multirate Biermann substepping
- **Gate:** CG + dedicated stability/conservation study · **Depends:** checkpoint 3
- **Target:** the largest remaining structural win
- **Where:** `src/driver/driver.cpp:344-431`; `src/mhd/mhd_tasks.cpp:102-152`
- **Problem:** the half-interval is tiled with **globally-minimum uniform** microsteps
  (`driver.cpp:371`), so all 33.5 M cells march at the worst cell's rate — and the limit is
  spatially localised near steep `∇p_e/n_e`. Additionally, each microstage costs **three
  serialised blocking halo rounds** (`mhd_tasks.cpp:109-141`), giving **24 Biermann halo
  rounds/cycle vs 6 for all of MHD**. This is the primary suspect for 16 ranks being slower
  than 8.
- **Change:** let blocks advance at their own stable rate with flux correction at block
  faces, instead of a global barrier per microstep.
- **Preconditions:** B2 (halo overlap) and A3 (dt amortisation) should land first — they are
  cheap and they reduce the surface area of this change.
- **Must preserve:** ion energy stays out of the SSPRK recurrence (§3.3).
- **Acceptance:** full CG plus a convergence study and `div B` / conservation diagnostics vs
  a frozen reference; document in its own report section as rev 1 did for accepted stages.

#### D1 — Load balance, only if still warranted
- **Gate:** BE for pure repartition · **Depends:** checkpoint 3
- **Context:** `material_work_imbalance/DIAGNOSIS.md` records ranks 0–3 owning 3.8× more
  mixed cells than ranks 4–7 (2.1× `Sync` spread). A **material-only** static partition was
  tried and **rejected**: it raised laser communication faces 12.8% and slowed sustained
  runs 3%.
- **Rule:** any repartition must put **laser ray residence in the objective**, or preserve
  the current laser-aware `x1_rank_map` cuts (deck lines 34-44). Re-measure the spread after
  Wave 2 — the imbalance's absolute weight falls with the material cost. Do not invest
  unless checkpoint 3 still shows a large spread.

---

## 7. Do NOT revisit — profiled and rejected

Evidence in `DCI_3D/profile_20260729/PROFILE_REPORT.md`, §"Rejected …".

| Item | Why rejected |
|---|---|
| Fused `MHD::RKUpdate` | Gain below noise as an application-level change |
| Rolling-Planck coupling **as a standalone change** | Below noise; **may be re-attempted only bundled inside C3, after A1/C4a** |
| Static communication-aware composition partition | +12.81% laser comm faces, −3% sustained |
| Scalar density continuation | Rejected on the byte-exact gate |
| Exchange-wide density-cache reuse | Rejected |
| Strict mixed-inverse invariant hoisting | Register pressure |
| Normal-face shock-mask limiter relaxation | Rejected |
| **Super-time-stepping for Biermann** | The limit is a **hyperbolic** wave speed, not diffusive. STS does not apply. |

**Config note, no code change:** `biermann_shock_suppression = true` (deck line 84) is
effectively inert on the subcycle path — `AddFluxes` skips `ComputeShockMask` when
subcycling (`biermann_battery.cpp:354-355`) and suppression is instead the smooth
`activation` smoothstep computed in `CachedElectronDensityCode` (`:64-86`) and folded into
the vertex pressure coordinate (`:947-953`, 3-D path). It costs nothing. It is a
correctness/config flag, not a performance lever.

---

## 8. Expected outcome

If A0/A4 and Waves 1–3 land as specified:

**Host side (the bigger half — 59% of wall today):**

| Cost | Now | Expected | Driver |
|---|---:|---:|---|
| `cuMemHostRegister` + `Unregister` | 8.33 s / 44.9% API | largely eliminated | A0 → A4 |
| Host waits (`cudaEventSynchronize` + `Device` + `Stream`) | 8.43 s / 45.4% API | substantially reduced | A3, A5, E1 |

**Device side:**

| Group | Now | Expected | Driver |
|---|---:|---:|---|
| Tabular EOS closure | 58.8% | ~20–30% | A1, A1b, A2b, C2, B3 |
| Radiation | 12.9% | ~7–8% | C4a primarily; C3; C5 steps 2–3 |
| Data movement | 9.6% | ~3–4% | A4, A3, A5, B1 |
| Laser | 0.4% + launch gaps | ~0.2%, gaps largely gone | E1–E4 |

Memory is informational, not a target: E2 frees ≈478 MB (72.5% → ≈69.6%), and any card
that grows memory is fine provided the mesh is retuned once at the checkpoint (§2.1).

Wave 4 (A7) then attacks what remains of the communication tax, which is also the most
likely fix for 16 ranks being slower than 8.

Re-derive both tables at each checkpoint from a fresh profile — **capture the API-call
section as well as GPU activities** (§4.2), and **take checkpoint 3 from a later-time
restart, not a cold start**.
