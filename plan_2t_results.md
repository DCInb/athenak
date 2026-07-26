# plan_2t results log

Execution record for `plan_2t.md`. Newest entries appended per task.

## Phase 0 — Environment & build matrix

### B0 — Toolchain sanity (2026-07-26, HEAD `56e8cee5`)

| Component | Version / state |
|---|---|
| CPU / RAM / disk | 24 cores, 61 GB RAM, 816 GB free |
| gcc / g++ | 12.4.0 (Ubuntu, `/usr/bin/gcc`) |
| cmake | 3.28.3 |
| CUDA (nvcc) | 12.9.86 |
| MPI | OpenMPI 4.1.7rc1 (nvhpc 25.7 hpcx, `/home/mengqi/Research/nvhpc_257/...`) |
| Kokkos | 4.4.0 via sibling fallback `/home/mengqi/Research/athenak/kokkos` |
| NVIDIA driver | **MISMATCH — GPU runs blocked** (see below) |

Matches the toolchain recorded in `test-2t/build.log` (GCC 12.4.0, CUDA 12.9, Kokkos 4.4.0, hpcx MPI) — prior GPU results remain comparable.

**Decisions:**
- **Kokkos submodule left uninitialized** (`kokkos/` empty, pointer `6739bc62`). All prior verified results used the sibling `../athenak/kokkos` (4.4.0); keeping it preserves comparability. Testing against the submodule's Kokkos is deferred as an optional follow-up.
- **Skill script limitation found**: `athenak_case.sh` resolves problems only as `src/pgen/<name>.cpp`; on this branch all pgens live in `src/pgen/tests/` (dispatch via `pgen.cpp`), so the script cannot configure this repo (even `--problem built_in_pgens` dies, unlike the invocation recorded in `test-2t/build.log`). Builds replicate its cmake recipe manually: source `bashrc_athenaK`, `-DAthena_ENABLE_MPI=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DCMAKE_CXX_COMPILER=.../nvcc_wrapper` + MPI paths from `mpicxx --showme`, no `-DPROBLEM` (→ `built_in_pgens`).

**BLOCKER for GPU runs:** `nvidia-smi` → "Failed to initialize NVML: Driver/library version mismatch". Kernel module **580.159.03** loaded; userspace **580.173.02** installed (`nvidia-driver-580` upgraded without reboot). GPU builds are unaffected; **all GPU runs (Phase 2 `--gpu`, Phase 5 harnesses) blocked until reboot** (user action).

### B1 — Serial CPU build — **PASS**
- `build-cpu/src/athena` built clean (exit 0), first-ever CPU compile of this branch.
- Warnings: 4× benign `-Wformat-truncation` (`%02d` into 4-byte buffer for `eradNN` output names, `basetype_output.cpp:370/403/537/572`); safe while group index ≤ 99 (guaranteed by `n_groups ≤ 100`).
- Smoke test: `inputs/hydro/two_temperature_relax.athinput` runs to tlim (6 cycles), clean exit.

### B2 — MPI CPU build — **PASS**
- `build-mpicpu/src/athena` built clean (exit 0); first non-nvcc compile of `laser_mpi.cpp`. Same 4 benign format-truncation warnings as B1.
- Smoke test: 2-rank `mpirun` on `two_temperature_relax.athinput` with `meshblock/nx1=8` runs to tlim, clean exit. (Deck default is a single 16-cell block → correct fatal "Fewer MeshBlocks than MPI ranks" at 2 ranks without the override.)

### B3 — CUDA+MPI build — **PASS** (and GPU runs work!)
- `build-gpu/src/athena` built clean (Kokkos 4.4.0, SERIAL+CUDA, VOLTA70, CUDA 12.9) — same config as prior `test-2t/build.log`.
- Warnings (393 lines) all triaged benign: bulk = repeated nvcc "sm_70 deprecated" notice; 4× implicit-`this` lambda capture in **upstream** `bvals_part.cpp:354,473` (particles, untouched by branch); 2× unused variable in upstream `conduction.cpp:330` / `gauss_legendre.cpp:40`; 4× the same format-truncation as B1. **Zero warnings in new 2T/laser modules.**
- Smoke: `mpirun -n 1` on 2T relax deck runs to tlim on GPU. Verified genuinely on-device: `CUDA_VISIBLE_DEVICES=""` aborts with runtime_error.
- **Driver-mismatch blocker downgraded to caveat**: CUDA compute works despite kernel 580.159.03 vs userspace 580.173.02; only NVML (`nvidia-smi` monitoring) is broken. Reboot still recommended before performance measurements (P3/P4); functional GPU testing can proceed.
- Note: MPI-linked binaries must be launched via `mpirun` with `bashrc_athenaK` sourced (direct exec fails in opal_init due to relocated hpcx prefix).

### B4 — OpenMP CPU build — **PASS**
- `build-omp/src/athena` built clean; same 4 benign warnings. Smoke: 2T relax deck with `OMP_NUM_THREADS=4` runs to tlim, clean exit.

### B5 — laser-target custom pgen build — **PASS**
- `build-laser-target/src/athena` built clean (CUDA+MPI + `-DPROBLEM=../../laser-target/laser_target`).
- Smoke: full-coupling run (laser + 2T + 3-group radiation + Biermann, 2D 96×64) for 5 cycles on GPU, clean exit.

### B6 — Warning sweep + smoke tests — **PASS**
- New 2T/laser modules are **warning-free on all four backends** (gcc serial, gcc+MPI, gcc+OpenMP, nvcc CUDA).
- Only branch-attributable warning: 4× benign `-Wformat-truncation` in `basetype_output.cpp:370/403/537/572` (`%02d` for `eradNN` names; safe under the `n_groups ≤ 100` cap). Optional cleanup: widen `number[4]` buffer.
- Smoke matrix: B1 serial ✓, B2 `mpirun -n 2` ✓, B3 GPU ✓ (verified on-device), B4 OMP×4 ✓, B5 full-coupling GPU ✓ — all on 2T decks, clean exits.

## Phase 0 summary — COMPLETE (2026-07-26)

All five builds pass at HEAD `56e8cee5`; every smoke test clean. Deviations from plan assumptions:
1. `athenak_case.sh` cannot configure this repo (pgens in `src/pgen/tests/`); its cmake recipe is replicated manually (documented in B0).
2. NVIDIA driver mismatch (kernel 580.159.03 / userspace 580.173.02) breaks `nvidia-smi` but **not CUDA compute** — GPU testing may proceed; reboot before Phase 5 performance numbers.
3. Kokkos submodule left uninitialized; sibling `../athenak/kokkos` (4.4.0) in use for continuity.

Ready for Phase 1 (defect fixes F1–F7) and Phase 2 (regression baseline `tst/run_test_suite.py --cpu/--mpicpu`).

## Phase 1 — Defect fixes (2026-07-26)

| Fix | Change | Verified |
|---|---|---|
| F1 | `basetype_output.cpp`: laser null-guard range `<171` → `<178` (covers `laser_dir1`…`laser_x3_moment`) | Negative test: `laser_dir1` without `<laser>` now clean FATAL (was segfault) |
| F2 | New fatal guards: `<hydro>/two_temperature`+FOFC (`hydro.cpp`) and `<mhd>/two_temperature`+FOFC (`mhd.cpp`); scalar fluxes are not FOFC-replaced. MHD `fofc_scal` confirmed dead scaffolding (allocated, never used) — left in place, noted | Both guards fire with clear messages (MHD one tested with `dual_energy=false` override; dual-energy guard otherwise fires first) |
| F3 | Removed dead `<laser>/electron_number_per_density` (read + member); no deck/doc/test referenced it | grep clean |
| F4 | Rank-local fatal paths now `MPI_Abort(MPI_COMM_WORLD,1)` (io_wrapper convention): 8 sites in `laser_mpi.cpp`, reduction-failure in `laser_trace.cpp`, `OpacityTableError` (MPI-guarded). Conservation-failure exit at `laser_trace.cpp:~1166` intentionally left `std::exit` — it runs after a successful `Allreduce`, so all ranks exit together. Input-validation exits (deterministic on all ranks) unchanged, matching upstream style | Compiles on all 5 configs incl. serial (guards correct) |
| F5 | Deferred to T-B4 (quantify Biermann `vd*_` AMR inconsistency first) | — |
| F6 | `docs/laser_algorithm.md`: documented `cell_data`/`laser_energy` restart reset (physical state continues; diagnostic resets) | T-X2 will verify the continuity claim |
| F7 | Cross-link comments on the 3 duplicated internal-energy floor helpers, noting the intentional dfloor difference in `mhd_dual_energy.cpp` | Claims checked by review workflow |

**Verification**: all five builds rebuild clean with the fixes; full smoke matrix re-passes (serial/OMP/2-rank MPI/GPU on 2T relax + laser-target coupled 5 cycles on GPU).

**Adversarial 3-lens review (semantics / MPI-build / claims-docs): no blockers.** Confirmed: laser var table is exactly 165–177 with `NOUTPUT_CHOICES=178`; F2 guards reachable and correctly ordered (`ptwo_temp` null-init, assigned before `use_fofc` read); GR-excision FOFC path can't coexist with 2T; no deck/test pairs fofc+2T (nothing newly rejected); all `MPI_Abort` sites MPI-only-compiled, serial builds unaffected; aborting `MPI_COMM_WORLD` correct (laser's `mpi_comm_` is a dup of it); retained `std::exit` at the conservation check is genuinely collective (predicate built solely from Allreduce results). Review findings applied:
1. `mhd_dual_energy.cpp` F7 comment was factually wrong (both call sites already pre-clamp with dfloor; internal clamp is redundant defense) — comment corrected.
2. Reduction-failure message in `laser_trace.cpp` printed only on rank 0 though the abort can be rank-local — now prints on every rank.
3. io_wrapper convention includes `std::exit` fallback after `MPI_Abort` (MPI_Abort is not `[[noreturn]]`) — fallback added at all 9 converted sites.
4. Inherited gap closed: 2T constructor checks now also exclude `is_dynamical_relativistic` (previously a `<z4c>`+2T deck would construct 2T unsupported and dyn-GR excision flux replacement would bypass the F2 guard).

**Phase 1 COMPLETE** — all five configs rebuilt clean with the review-driven amendments (rebuild2-*.log).

## Phase 2 — Regression baseline (2026-07-26)

### R1 first run: 5/6 CPU tests passed; refraction test caught a REAL physics regression

First-ever execution of the branch's own suite. `test_nr_laser_refraction_cpu` failed its convergence sweep — and diagnosis showed it was **not** a tolerance issue:

- **Symptom**: refractive `laser_path` error 3.2e-4 vs 2e-4 rtol at `refractive_cell_fraction=1.0`; per-point errors halved with each step halving; measured self-convergence order **0.93** (test requires >1.7).
- **Root cause**: commit `56e8cee5` ("quadratic ray-position reconstruction", part of the FLASH-tube benchmark fixes) made the refractive force position-dependent within a cell (`grad + hess·δ`), but the KDK stepper still evaluated the entire kick at the segment **start** point — first-order for a spatially varying force. The refraction test was authored at `6d503569` and never re-run after the tracer change (the suite had never been run at all).
- **Proof**: worktree build at `6d503569` reproduces order **2.018** (flat fixed-grid error 2.5e-5); HEAD-before-fix gives 0.93.
- **Fix** (`laser_trace.cpp`): proper kick–drift–kick — recompute the half-kick with the final face-limited step, drift with the half-kicked wave vector, apply the second half-kick with the cell's quadratic force model evaluated at the drift **endpoint**. Reduces exactly to the previous update for constant force (hess=0), so straight-tracer and constant-gradient behavior is unchanged.
- **Result after fix**: path error **3.1e-8** (was 3.2e-4 broken / 2.5e-5 pre-regression), moment error 2.7e-5→4.8e-7 at order **1.88**, dispersion residuals ~100× smaller. Strictly better than both prior states.
- **Follow-up required**: FLASH-tube exit-power spot-check (the tracer change shifts refractive results slightly; expect within the 1e-3 threshold, likely improved) — done below; full harness rerun in Phase 5.

Other CPU results (first run): two_temperature, thermal_radiation, biermann_battery, laser (DDA/attenuation/reflection), laser_amr — **all passed**. The `--mpicpu` phase did not run in the first invocation (suite aborts on CPU-phase failure); included in the rerun.

Test infrastructure: created `.venv-tst/` (venv over the `athenak-vis` conda interpreter + pytest 9.1.1) since no environment had pytest.

### R2 first run: MPI laser test caught a SECOND real defect — decomposition-dependent work caps

`test_nr_laser_mpicpu` (first-ever run) failed comparing `laser_q` across rank counts. Diagnosis:

- **Symptom**: with tight caps (`max_segments_per_launch=3 × max_transport_iterations=8` = 24-segment budget on a 64-cell slab), np=1 and np=2 truncated deposition at segment 24 (booking 47% of beam power as `remaining`), while np=4/np=8 completed the full path.
- **Root cause**: the work caps bound device work **per transport wave**, but new waves were triggered only by MPI ray migration. Fine decompositions (few cells per rank) migrate before exhausting the budget and get a fresh budget each wave; serial and coarse decompositions get exactly one wave. Same failure mode as the archived `test-2t/logs/laser_limited_gpu*.log` stall (312 segments, `remaining=0.47`), which had been worked around rather than fixed.
- **Fix** (`laser.hpp`, `laser_trace.cpp`, `laser_mpi.cpp`): uniform wave semantics. (1) Capped rays stay `active` at trace end instead of being booked `remaining` (booking now happens only on the final wave via new `BookRemainingRays()`); (2) wave termination counts all still-active rays (`CountActiveRays()`), not just received migrants, so ranks with capped local rays keep waving; (3) the serial and nranks==1 paths run the same wave loop up to `max_mpi_waves`; (4) the active queue is rebuilt from ray status at every trace entry (`SeedActiveQueue()`) — this also removes a latent queue-slot clobbering hazard where an unpacked migrant's scatter write could evict a surviving capped ray from the compacted queue. A ray's total budget is now `max_mpi_waves × max_transport_iterations × max_segments_per_launch` for every decomposition; default-cap runs behave as before (single wave).
- **Result**: capped 1/2/4-rank runs all trace the full 832 segments, `remaining=0`, `deposited=1−e⁻²` exact, cross-rank deposited agreement ~2e-15. Docs updated (`docs/laser_algorithm.md`).

### Phase 2 final state — COMPLETE (2026-07-27)

- Full suite green: **6/6 CPU tests + MPI-CPU migration test** (1/2/4/8 ranks, boundary-parallel stress beam, direct-buffer/gpu-aware branch) pass with both tracer fixes in.
- Fixes propagated to all five build dirs; GPU spot checks with final binaries:
  - FLASH tube r64: exit-power rel err 5.8e-4 (<1e-3 threshold, improved from 6.2e-4), dispersion 8.2e-7, residual 2.4e-15.
  - Capped-laser wave semantics on GPU: np=1 and np=2 both complete (832 segments, remaining=0), deposited agreement ~2e-16.
  - laser-target coupled deck: 5 cycles clean.
- R3 (porting `_gpu` variants of the branch tests into tst/) remains open — deferred to Phase 6/CI work; the `test-2t` harness covers GPU interim.
- Net Phase 2 outcome: the never-before-run suite caught **two real physics/parallelism defects** (first-order refractive convergence; decomposition-dependent laser work caps), both root-caused, fixed, and verified on CPU and GPU.

## Phase 3a+3b — 2T core + thermal radiation physics validation (2026-07-27)

11 test agents + adversarial critic (who re-ran the two most suspicious claims and confirmed every figure; bitwise-reproducible artifacts under scratchpad `phase3/`). **All physics checks PASS.** Highlights:

| Test | Result |
|---|---|
| T-2T1 shock partition | 18/18 solver×recon×integrator combos on 2T Brio–Wu: positivity, partition closure ≤5.4e-12, energy drift ≤3e-14; Ti/Te ratio invariant through shocks to ≤5e-13 (proportional Sync, f_e∈{0.1,0.9}) |
| T-2T2 exchange limits | Exact decay `exp(−t/(f_i·t_ei))` to ≤2e-12 at every output; t_ei=0 instant, t_ei<0 frozen, rate scaling with f_e∈{0.1,0.9}, ratio∈{0,10} all exact; (eion+eele) conserved ≤1e-12 |
| T-2T3 dual-energy selectors | Baseline retention exact; **demonstrations**: eta1=0 → 13/48 cells corrupted (rel err 4.1e6) in magnetically dominated regime, eta2=0 byte-identical corruption, dual_energy=false → 100% eint loss — quantifies why the scheme + both switches are load-bearing. Well-conditioned case: conservative branch, total energy to machine eps |
| T-2T4 guard matrix | 15/15 clean fatals with correct messages. Found: Biermann `ng<2` guard (`mhd.cpp:143-147`) is unreachable dead code (mesh.cpp exits first); `mesh.cpp:231` message said "More than 2" for condition `ng<2` (fixed → "At least 2") |
| T-2T5 hydro multi-D/MPI | 2D 2T shock serial vs 2/4 ranks identical ≤1e-12; reflect/outflow BCs clean, reflect conserves ≤1e-10 |
| T-R1 diffusion rate | FLD transport coefficient correct to **+0.065%** (σ²-vs-t regression) with quantified fixed offset −0.438·dx²; 2D/3D runs reproduce 1D exactly (≤1e-12); conservation to machine eps |
| T-R2 limiter matrix | All five limiters run + conserve (larsen/none/minmax first-ever execution); limiter ordering correct; free-streaming: LP caps half-height front at 0.87·c_hat (leading-edge O(Δx) precursor converging 1.14→1.06·c_hat with resolution — inherent explicit-FLD artifact, documented); `none` violates causality by ~10³ (expected); dt∝κ collapse measured exactly |
| T-R3 coupling stress | Exact one-step lagged-emission/implicit-absorption formula reproduced ≤1e-12; Te→Trad equilibrium ≤1e-6; near-zero-eele clamp keeps positivity + conservation (limit cycle under extreme stiffness characterized); source_cfl behavior documented |
| T-R4 opacity tables | Linear vs log interpolation exact on analytic table; scale parameters round-trip to round-off (note: value scales are *divisors* of table units — doc convention); out-of-range clamps to edge exactly; 7/7 malformed tables → clean fatals |
| T-R5 MHD radiation | MHD hook path matches hydro twin to 1e-19 (llf/llf, B=0). Initial hlld-vs-hllc comparison differed 1.14e-2 in shock cells only — truncation-level solver algebra difference at B=0, not a radiation bug |
| T-R6 SMR+MPI radiation | Diffusive-flux correction at refinement boundary conserves to machine eps (4.4e-16); 1/2/4 ranks identical ≤1e-13; SMR vs uniform L1 9.6e-4 (<1e-3) |

Open MED items: superluminal FLD leading-edge precursor (inherent; document as limitation in D1), dead Biermann guard (cleanup candidate), hlld-B=0 shock-cell delta (characterized, no action).

## Phase 3c+3d — laser + Biermann/dual-energy physics validation (2026-07-27)

11 test agents + adversarial critic (independent re-runs reproduced every spot-checked number to the printed digit; no fabrication, no physics contradictions). **10/11 PASS; one HIGH feature defect found.**

| Test | Result |
|---|---|
| T-L2 oblique reflection | n_c·cos²θ turning law verified to sub-cell (Δx ≤ 4e-9) at θ={0°,30°,60°}; oblique_turning=false moves to n_c depth exactly; power accounting machine-precision every stage |
| T-L4 IB kappa | Fixed and auto-Coulomb-log branches match Python transcription of `laser_physics.hpp` to ≤1.5e-13 over 100× ranges in ρ,Te; lnΛ clamp verified; Beer–Lambert to 1e-16. Findings: uniform supercritical plasma silently absorbs 100% in cell 0 despite critical_reflection (needs resolved gradient — doc trap); laser samples post-`duale` electron energy → O(dt) κ state-timing (fine for scheme order, comment-worthy) |
| T-L5 beam geometry | Multi-beam power exact (Σ=7 to 1e-14); apertures (2D line, 3D Fibonacci disk) geometrically correct; gaussian/uniform renormalization exact; time gating quantized to step-start as documented |
| T-L6 work caps | New wave semantics verified: extreme caps (1 seg/wave) still complete identically serial vs MPI; max_mpi_waves as true cap books partial Beer–Lambert exactly; periodic κ=0 terminates at wave cap without hang; reflection-cap booking matches code intent |
| T-L8 deposition targets | total vs electron: identical optics, correct partition behavior (Ti/Te pinned in total mode; Te/Ti grows in electron mode); conservation 2.9e-11 (E_tot-normalized; outflow-flux attribution plausible but not directly integrated — MED note) |
| T-L9 MPI stress | 1/3/8 ranks: fields bitwise, reductions ~1e-14; 861 transfers (boundary-graze, 64 rays); many-wave forcing physics-identical; 8-rank wall-time overhead documented |
| T-B1 Biermann MPI | Analytic rate at 1/2/4 ranks: L2 3.32e-3 (=serial to 1e-10); B1=B2=0 ≤2e-12; bcc3 ≤1e-13 across rank counts; 3D flux-CT variant clean |
| **T-B2 shock mask** | **FAIL (feature defect, HIGH)**: suppression ON yields ~4.7× larger max\|B3\| (6.5× rms) at the discontinuity than OFF at 64²; non-monotone in threshold; cause = hard 0/1 mask edge injects spurious curl (E→0 across one face). Positivity/energy clean in all 12 runs — a quality defect of the mask, not an instability. Critic reproduced bitwise. **Recommendation: smooth the mask ramp and/or re-evaluate at higher resolution before relying on suppression; defaults unchanged pending user decision** |
| T-B3 inertness | coefficient=0 bitwise-identical to battery-off (restart-seeded; fresh biermann-pgen with battery off hard-aborts — guard could be relaxed); dt sequences identical; 2-rank bitwise too |
| T-B4 Biermann SMR (**F5 decision**) | Uncorrected `vd*_`: eele-partition perturbation ≤6.3e-6 over 205 steps, strictly inside the 9.7e-6 coarse–fine truncation band; total-E drift 0.0 at hst resolution; closure ≤1.7e-16; SMR lands between coarse/fine references; 2-rank bitwise. **F5 CLOSED: no code fix; comment added to `ApplyElectronWork`** (critic endorsed) |
| T-B5 dynamic SMR dual-energy | 2D Brio–Wu shock crossing a level-1 boundary: conservation ≤1e-10 pre-boundary-exit, closure ≤1e-9 including the CF band at every output, positivity, 1/2/4-rank agreement ≤1e-12, SMR vs uniform truncation-level. Highest-risk dual_vf machinery clean |

Additional MED/LOW notes: max_mpi_waves-bound truncation is decomposition-dependent when it binds (benign for converged runs — documented); `.bin` writer is hard-coded float32 (repeatedly caused false 1e-7 "noise" during analysis — fp64 bin option would help); tb3 fresh-init guard.

## Phase 5 — P1+P2 benchmark harness refresh (2026-07-27)

### P1 — test-2t 5-feature × {1,2,4}-GPU matrix: **15/15 PASS, zero drift**
- `test-2t/build` rebuilt at current HEAD (all archives preserved as `*-pre-refresh`).
- Every metric in all 15 cells **bit-identical** to the archived 6d503569-era values; cross-rank scaled error exactly 0.0 everywhere. The 4-rank laser diagnostics line is byte-identical old vs new — the archived runs never engaged the work caps, so the wave-semantics rewrite is confirmed behavior-neutral for converged transport, and the Phase-1 fixes perturb nothing at these tolerances. Strongest possible non-regression statement for the fix campaign.

### P2 — FLASH laser-tube uniform + SMR: **7/7 PASS, improvements across the board** (KDK fix)
| Metric | old → new |
|---|---|
| Exit-power rel err r32 / r64 / SMR | 4.07e-4 → **3.27e-4** / 6.21e-4 → **5.81e-4** / 2.80e-4 → **2.31e-4** |
| Max dispersion r32 / r64 | 1.13e-3 → **3.29e-6** (÷345) / 5.65e-4 → **8.18e-7** (÷690) |
| SMR track L1 | 3.63e-3 → **1.96e-3** cm (÷1.85); field-vs-log dep diff 1.38e-8 → 9.0e-9 |
| Multi-GPU field diffs | exactly 0.0 (unchanged); transfers 6/8 unchanged; conservation ~1e-16 |
- One benign uptick: SMR max dispersion ×1.26 (7.3e-4, still 7× under threshold).
- Convergence checks (r64 < 0.75×r32) still PASS; fine-level deposited fraction 0.4084 matches archived.

### Phase 6 progress (concurrent)
- D1 main pass done: shock-mask caveat + F5 note (`BIERMANN_BATTERY.md`), explicit-FLD dt/precursor limitations (`THERMAL_RADIATION.md`), supercritical-uniform trap + wave-cap decomposition caveat (`laser_algorithm.md`). Units worked example pending T-X4.
- D2: CI structurally covers the 7 branch tests (unfiltered `--cpu`/`--mpicpu` jobs; fork CI triggers only on PRs to main, upstream runners). New `tst/test_suite/nr/test_nr_guards_cpu.py` codifies the 11-case negative matrix (validated green; includes a permanent regression test for the F1 segfault fix). Six `_gpu` wrapper tests added closing R3 (validation run pending).
- Fixed while validating guards: `mesh.cpp:231` misleading ghost-zone message ("More than 2" → "At least 2").

## Phase 4 — cross-cutting integration tests (2026-07-27)

7 agents + adversarial critic (2 independent re-runs; all spot-checks upheld). **All seven risk areas retired**:

| Test | Result |
|---|---|
| T-X1 AMR+MPI+laser | 9/9 PASS. Adaptive refinement + derefinement churn (30 created/31 deleted) + load balancing (up to 60 block migrations) at 1/2/4 ranks: **bitwise rank-invariant in double** (critic-verified), residuals ≤4.4e-16 every stage, zero failed rays; `laser_energy` survives derefinement via exact conservative restriction averaging (volume integral preserved to 12 digits) |
| T-X2 restarts | All five round-trips bitwise-exact in physical state (incl. rank-count change 2→4 and 2→1); `laser_energy` resets exactly as A_final−A_seam (documented behavior; restart notice added to driver) |
| T-X3 radiation+AMR dt | **Theorized hazard already mitigated in code**: `AdaptiveMeshRefinement` re-runs `NewTimeStep` post-regrid; measured exact 4× dt drop at every refinement event, zero instability. (Residual upstream gap: particles dt not re-called — out of scope) |
| T-X4 units audit | Consistency relations derived + verified by scale-transformation invariance (5e-13). laser-target deck confirmed internally inconsistent (beam 299× weaker than labeled, ĉ ÷930, C_B ×7.4, f_e↔zeff contradiction) — by design (topology benchmark); recipe published as `docs/UNITS_2T_LASER.md`, README labeled |
| T-X5 laser-target rerun | 6/6 PASS, zero drift vs archived results (≤6e-12 all metrics); 2-rank invariance improved to ~1e-13; `max_reflected=250` mystery resolved (per-event counter, budget 258) |
| T-X6 BC surface | Reflect/outflow/vacuum stable and conserving; **FLD `outflow` = insulated wall** (documented); inflow+2T needs pgen support (documented); hydro 2T+shearing-box scalar transport **validated working**; guard gap found (MHD 2T+shear+dual_energy=false) |
| T-X7 CPU↔GPU | All ≤1.23e-12 (FMA-characterized); laser transport/deposition/reflection/refraction **bit-reproducible** across backends, devices, and repeated GPU runs |

**Fixes applied from Phase 4 + 3c/3d findings** (all builds green, suite gate rerun):
1. Biermann shock-suppression mask: hard 0/1 edge → linear ramp (threshold/2 → threshold). Halves the spurious-B3 artifact at every threshold (ON 1.54e-5→9.3e-6; thr=0.1 now at unsuppressed noise floor); smooth flow bit-unchanged; docs updated with post-fix guidance.
2. Dead Biermann `ng<2` guard removed (unreachable; mesh-level check subsumes it).
3. MHD 2T + `<shearing_box>` now guarded (any dual_energy setting); guard-tested.
4. Hydro 2T + `<thermal_radiation>` + `<shearing_box>` now guarded (scalar advection validated, FLD-through-shear-remap not).
5. Restart notice added: rank-0 prints that cumulative laser diagnostics reset (physical state unaffected).
6. `test_nr_guards_cpu.py` extended with the shearing-box case (12 cases total).
7. FLD boundary-condition semantics documented (`THERMAL_RADIATION.md`); units recipe published (`docs/UNITS_2T_LASER.md`); laser-target README unit note.

**Critic's remaining open items**: commit the tested tree to pin the validated state (user action — everything above is uncommitted); benchmark-deck opacities unverified against physical tables (inherent to demo deck, labeled); T-X2 "bitwise" certified at ~5e-13 output resolution (low risk).

## Phase 5 — P3+P4 performance (2026-07-27; **confirmed canonical post-reboot** — see addendum below)

### P4 — feature-cost baselines (1× V100, 64³, 32³ blocks, 200 cycles, median of 3; rep spread <1%)

| Config | zone-cycles/s | cost vs plain |
|---|---|---|
| Plain MHD 2T (dual energy) | 3.61e7 | 1.00× |
| + laser, straight, 16 rays | 2.05e7 | 1.77× |
| + laser, straight, 64 rays | 2.03e7 | 1.78× (ray count ~free) |
| + 3-group FLD radiation | 2.21e7 | 1.64× |
| + laser + radiation | 1.61e7 | 2.25× |
| + refractive laser, 64 rays | 1.55e7 | 2.32× |
| CPU serial, laser16 config | 7.57e5 | GPU/CPU ≈ **27×** |

(P4's agent returned prematurely; GPU data was complete on disk and the missing CPU rep was rerun directly — archived in scratchpad `phase5/p4/results_raw.csv`.)

### P3 — 3D SMR scale probe (128³ root + level-1 tube, 120 blocks, 3.9M cells, 64 rays)

- Laser share of runtime: **9.2%** (axis beam, 1 GPU) → 12.5% (4 GPU) → **25.7%** (4 GPU with real ray migration, 49 transfers/trace at ~0.1 ms/transfer). The 64³ P4 numbers (1.77×) reflect fixed per-stage overhead that amortizes away at production scale.
- 4-GPU strong scaling ≈ **32% efficiency with or without laser** — the limit is MHD comm at ~1M cells/rank, not the laser.
- The prior 2D pathology (2-rank 2.5× slower) does **not** reproduce: worst 3D case is still 1.11× faster at 4 GPUs than 1.
- O(nmb_total) block scans measurable but not dominant at 120 blocks (0.33–0.68 µs/segment, <10% effect from 4× scan-length change); the concern stands for O(10³⁺) blocks — profile after reboot.
- Conservation at scale: every stage residual ≤9.2e-15; deposited = 1−e⁻¹ analytic, 1-vs-4-GPU agreement 8e-15.

---

# D3 — Final campaign report (2026-07-27)

## Coverage matrix

| Axis | Covered |
|---|---|
| Builds | serial CPU, MPI CPU, OpenMP, CUDA+MPI, serial+CUDA (suite), custom-pgen — all warning-free in new modules |
| Modules | 2T core, dual-energy MHD, multigroup FLD + opacity tables, Biermann battery, laser (straight/refractive/reflection/IB) — each validated against analytic references |
| Parallelism | 1–8 MPI ranks, 1–4 GPUs, odd rank counts, rank-count-changing restarts; fields bitwise rank-invariant everywhere tested |
| Mesh | uniform 1D/2D/3D, SMR, adaptive AMR with derefinement churn and load balancing |
| Cross-cutting | restarts, CPU↔GPU (bit-reproducible laser transport), units scaling, BCs, guard matrix (12 negative tests), performance baselines |

## Defects found and fixed (all validated, suite green)

1. **Refractive tracer order regression** (first-order since `56e8cee5`) → endpoint-force KDK; order 1.88 restored, FLASH-tube metrics improved, dispersion ÷345–690.
2. **Decomposition-dependent laser work caps** (serial truncated where MPI completed; root cause of the archived `laser_limited` stall) → uniform wave semantics + queue rebuild (also removed a latent queue-clobbering hazard).
3. **Laser-output null-pointer segfault** (vars 171–177 unguarded) → guard widened; regression-tested.
4. **FOFC × 2T scalar-flux inconsistency** → fatal guards (hydro + MHD).
5. **Biermann shock-suppression mask counterproductive** (hard edge injected 4.7× the noise it suppressed) → linear ramp; artifact halved at every threshold, smooth flow bit-unchanged.
6. **Guard gaps**: MHD 2T+shearing-box (any dual_energy), hydro 2T+FLD+shearing-box, 2T+dynamical-GR — all now fatal.
7. **MPI robustness**: 9 rank-local `std::exit` paths → `MPI_Abort`+fallback; rank-local error printing fixed.
8. Cosmetics: dead Biermann guard removed, `mesh.cpp` message wording, dead laser parameter removed, restart notice for laser diagnostics, triplicated floor helpers cross-linked.

## Accepted limitations (documented in docs/)

- No radiation momentum/pressure; comoving FLD only; explicit diffusion (dt ∝ κ·dx²) with an O(Δx) superluminal precursor in free-streaming; FLD `outflow` BC is an insulated wall.
- n_e ∝ ρ fixed-ionization; single gamma; `t_ei` spatially constant.
- Laser: cumulative diagnostics reset on restart (physical state exact); uniform supercritical medium absorbs without reflecting (needs resolved ramp); `max_mpi_waves`-truncated runs are decomposition-dependent; O(dt) electron-energy state-timing in IB κ; O(nmb_total) block scans (fine ≤120 blocks, profile at 10³⁺).
- Biermann: `vd*_` work term not AMR-flux-corrected (partition perturbation bounded by truncation — measured, commented); suppression cannot beat the unsuppressed noise floor at marginal resolution (use threshold ~0.1).
- laser-target benchmark is a coupling-topology demo, not a calibrated scenario (see `docs/UNITS_2T_LASER.md` for building one).

## Sign-off state

- `tst/` suite: **8 CPU test files + MPI migration test green** (incl. new guard matrix); 6 new `_gpu` wrappers green on serial+CUDA.
- GPU harnesses: test-2t 15/15 bit-identical to archived; FLASH tubes 7/7 improved; laser-target 6/6 zero-drift.
- **Outstanding user actions**: (1) commit the working tree to pin the validated state (~25 modified/new src+tst+docs files, all uncommitted); (2) optional follow-ups: physical-opacity benchmark deck, FOFC scalar support, fp64 bin output option, Marshak/vacuum radiation BC.

## Post-reboot addendum (2026-07-27, after user reboot)

- Driver healthy: kernel module and userspace both 580.173.02, NVML restored. Full inventory: **8× Tesla V100-SXM2-16GB** (devices 0–7; the pre-reboot campaign used only 2–5).
- **P4 baselines confirmed canonical**: all six configs reproduce within ±1.1% (laser 1.79×, radiation 1.65×, laser+radiation 2.27×, refractive 2.36× vs plain 2T MHD at 3.65e7 z-c/s; GPU/CPU 27×).
- **P3 1/4-GPU confirmed** (4-GPU efficiency 30.5%/32.6% laser/no-laser). **New 8-GPU measurement: strong-scaling ceiling found** — 8 ranks is slower than 4 (0.76–0.81× vs 1 GPU, ~10% efficiency), identically with and without laser: at 15 blocks (0.49M cells)/rank the 3.9M-cell problem is over-decomposed and MHD communication dominates. For this problem size, 4 GPUs is the sweet spot; larger problems are needed to exercise 8. Raw data: scratchpad `phase5/{p4,p3-scaleup}/results_postreboot.csv`.

### B3 — CUDA+MPI build
- Status: running (`build-gpu/`, logs `build-gpu-config.log`, `build-gpu-make.log`)
