# plan_2t.md — Build & Test Plan for the 3T (ion/electron/radiation) + Laser Modules

Branch `2T` on top of upstream `40453f7a` (~12k insertions, 14 commits). New subsystems:

| Module | Files | Commits |
|---|---|---|
| Ion–electron two-temperature core | `src/two_temperature/two_temperature.{cpp,hpp}` | `f9e54eaf` |
| Multigroup FLD thermal radiation + tabulated opacities | `src/two_temperature/thermal_radiation.*`, `opacity_table.*` | `4ab845c4`, `5e95c34b` |
| Dual-energy MHD (2T-backed) | `src/mhd/mhd_dual_energy.cpp`, `src/mhd/rsolvers/dual_eint_mhd.hpp`, `src/eos/ideal_mhd.cpp`, `src/bvals/*` | `3b0de559` |
| Flux-form Biermann battery | `src/mhd/biermann_battery.{cpp,hpp}` | `d174045d` |
| Laser ray transport (straight/refractive, reflection, AMR, MPI) | `src/laser/*` (~2400 lines) | `761e4180`…`6d503569` |
| Benchmark harnesses | `test-2t/`, `laser-target/`, `tst/test_suite/nr/test_nr_{two_temperature,thermal_radiation,biermann_battery,laser*}_*.py` | `56e8cee5` |

**Current verified state** (from `test-2t/diagnostics.md`, `laser-target/diagnostics.md`): all ad-hoc GPU harnesses PASS, but only on **one config** — CUDA (V100, Kokkos 4.4.0, nvcc 12.9) + OpenMPI at 1/2/4 ranks, at HEAD `6d503569`. The `tst/` regression suite (7 test files) **exists but has never been run on this branch/machine** (no artifacts). Serial CPU, MPI-CPU, OpenMP builds: never compiled. That asymmetry drives the phase ordering below.

**Build tooling**: use `/home/mengqi/.codex/skills/athenak-build-run/scripts/athenak_case.sh {build|run|build-run}` for GPU builds/runs (sources `/home/mengqi/Research/bashrc_athenaK`, configures MPI+CUDA+VOLTA70, `make -j 40`, `mpirun`). CPU-only builds need direct cmake (the script hardcodes CUDA flags). Note: `kokkos/` submodule is **empty** in this checkout — CMake falls back to `../athenak/kokkos` (`CMakeLists.txt:113-120`), which is also where the default `nvcc_wrapper` lives, so the script defaults are correct as-is.

Task status legend: `[ ]` todo · `[x]` done · `[!]` blocked/failed. Fill in as executed.

---

## Phase 0 — Environment & build matrix

Goal: prove the branch compiles everywhere it claims to run. All 2T/laser sources are compiled **unconditionally** (no CMake option), so any build exercises them.

- [x] **B0 — Toolchain sanity.** *(done 2026-07-26 — see plan_2t_results.md; GPU runs blocked by NVIDIA driver mismatch until reboot; kokkos sibling fallback kept; athenak_case.sh can't configure this repo → cmake recipe replicated manually)* `source /home/mengqi/Research/bashrc_athenaK`; record `nvcc --version`, `mpicxx --version`, `gcc --version`, GPU inventory (`nvidia-smi`). Decide submodule policy: either `git submodule update --init kokkos` or keep relying on the `../athenak/kokkos` fallback (document which in `test-2t/environment.txt`).
- [x] **B1 — Serial CPU build** (PASS — see plan_2t_results.md): `cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release && make -j 40 -C build-cpu`. Watch for: `Kokkos::SharedHostPinnedSpace` usage in `src/laser/laser.hpp` (guarded by `MPI_PARALLEL_ENABLED`, should alias HostSpace), extended-lambda-isms that nvcc tolerates but gcc doesn't.
- [x] **B2 — MPI CPU build** (PASS — see plan_2t_results.md): as B1 plus `-DAthena_ENABLE_MPI=ON` → `build-mpicpu`. This is the first-ever compile of `laser_mpi.cpp` without nvcc.
- [x] **B3 — CUDA+MPI build** (PASS; GPU compute works despite NVML mismatch — see plan_2t_results.md): `athenak_case.sh build --problem laser_profile --repo /home/mengqi/Research/athenak-2t --build-dir build-gpu`. Confirm reproducibility of `test-2t/build.log` at current HEAD `56e8cee5`.
- [x] **B4 — OpenMP CPU build** (PASS — see plan_2t_results.md): B1 plus `-DAthena_ENABLE_OPENMP=ON`. Laser atomics + `parallel_scan` on the OpenMP backend is a distinct code path.
- [x] **B5 — Custom-pgen build**: PASS — builds and runs 5 coupled cycles on GPU (see plan_2t_results.md).
- [x] **B6 — Warning sweep**: PASS — new modules warning-free on all backends; 4 benign format-truncation warnings in `basetype_output.cpp` (see plan_2t_results.md).

Pass criteria: all builds complete; smoke run on each binary. **Phase 0 COMPLETE 2026-07-26.**

## Phase 1 — Known defects to fix or consciously defer (found by code inspection)

These were confirmed by direct source inspection during planning. Fix before the test phases that would trip over them, or log as accepted limitations.

- [x] **F1 — Laser output null-deref (real bug, fix first).** *(fixed + negative-tested)* `src/outputs/basetype_output.cpp:211` guards missing `<laser>` block only for output indices 165–170, but laser variables span 165–177 (`src/outputs/outputs.hpp:104-113`). Requesting `laser_dir1`…`laser_x3_moment` without a `<laser>` block segfaults on null `plaser`. Fix the range; add negative test (T-L10).
- [x] **F2 — FOFC × 2T scalar-flux inconsistency.** *(fatal guards added hydro+MHD, negative-tested; full scalar-aware FOFC left as future work)* `src/hydro/hydro_fofc.cpp:68` replaces only the base hydro fluxes with first-order LLF; 2T component energies / radiation-group fluxes (added *before* FOFC, `hydro_tasks.cpp:197-206`) keep high-order values built from the old mass flux → inconsistent scalar transport exactly in troubled cells. MHD blocks only `dual_energy`+FOFC (`mhd.cpp:287-291`); hydro 2T+FOFC and MHD 2T (`dual_energy=false`)+FOFC are unguarded. Minimum: add a fatal guard like the dual-energy one. Better: extend FOFC to scalars.
- [x] **F3 — Dead parameter.** *(removed)* `<laser> electron_number_per_density` is read (`laser.cpp:152`) and never used; only `electron_number_per_gram` feeds n_e. Remove or wire up.
- [x] **F4 — `std::exit` on error paths under MPI.** *(rank-local sites → MPI_Abort+exit; collective/input-validation sites left per upstream convention)* Laser conservation/dispersion aborts (`laser_trace.cpp:1148-1161`) and opacity-table parse errors (`opacity_table.cpp:112-118`) exit without `MPI_Abort` → mpirun negative tests can hang. Prefer `MPI_Abort`; until then run all MPI negative tests under `timeout`.
- [x] **F5 — Biermann drift velocities not AMR-flux-corrected.** `vd*_` used by `ApplyElectronWork` are per-level local (unlike `dual_vf`, which was explicitly corrected for this reason, `dual_eint_mhd.hpp:13-15`) → e_ele work term is coarse/fine-inconsistent at refinement boundaries. Quantify in T-B4 before deciding whether to fix or document.
- [x] **F6 — Documented restart gap.** *(documented in docs/laser_algorithm.md; review verified cell_data never feeds back into u0/w0)* Laser `cell_data` (incl. cumulative `laser_energy`) is not in restart files; it silently resets. Either persist it or document the reset (docs + T-X2 verifies 2T/radiation state does restart correctly).
- [x] **F7 — Consolidate triplicated floor helper.** *(cross-link comments added, corrected per review; unification deferred)* `MHDInternalEnergyFloor` exists in `ideal_mhd.cpp:17`, `mhd_dual_energy.cpp:22`, `prolong_prims.cpp:31` with one behavioral difference (dfloor clamp). Unify or add a comment cross-linking them.

## Phase 2 — Regression suite baseline (the branch's own tests, never yet run)

Runner: `cd tst && python run_test_suite.py <mode>` (each mode does a clean cmake+make, then pytest filtered by suffix). The 7 branch tests: `test_nr_two_temperature_cpu`, `test_nr_thermal_radiation_cpu`, `test_nr_biermann_battery_cpu`, `test_nr_laser_cpu` (DDA geometry, attenuation, reflection round-trips), `test_nr_laser_amr_cpu` (SMR sweep + adaptive), `test_nr_laser_refraction_cpu`, `test_nr_laser_mpicpu` (2/4/8 ranks, serial-vs-MPI field comparison).

- [x] **R1 — CPU suite**: PASS after fixing a real defect it caught — the refractive tracer had regressed to first-order convergence (commit `56e8cee5`); fixed with endpoint-force KDK half-kick. 6/6 CPU tests green. (See plan_2t_results.md.)
- [x] **R2 — MPI-CPU suite**: PASS after fixing a second real defect it caught — laser work caps were decomposition-dependent (per-wave budget refreshed only on MPI migration); fixed with uniform wave semantics in serial and MPI paths. (See plan_2t_results.md.)
- [x] **R3 — GPU suite gap**: no `_gpu` variant of any 2T/laser test exists. Port the highest-value ones (`test_nr_laser`, `test_nr_two_temperature`, `test_nr_thermal_radiation`) to `_gpu` so `run_test_suite.py --gpu` covers this branch; interim substitute is the `test-2t/` harness (Phase 5).
- [x] **R4 — Record baseline**: recorded in `plan_2t_results.md` (working tree = `56e8cee5` + Phase-1 fixes + two Phase-2 tracer fixes; both suite phases green 2026-07-27).

## Phase 3 — Module-level physics tests (extend beyond current coverage)

Existing analytic validations (all 1D/uniform/serial or single-config GPU): exact e–i relaxation, 0-D radiation relaxation + 1D group diffusion decay, early-time linear Biermann dB3/dt, Beer–Lambert laser attenuation, FLASH laser-tube exit power (7×10⁻⁴ rel. err), laser-target coupled closure (9.4×10⁻⁵). The tasks below cover what those *don't*.

### 3a. Two-temperature core (`src/two_temperature/two_temperature.cpp`)

- [x] **T-2T1 — Shock partition test.** Sod + Brio–Wu with `two_temperature=true` (`inputs/mhd/two_temperature_bw.athinput` exists, is referenced by no test). Check: `e_ion+e_ele == eint` to round-off after Sync, positivity of all four T/e fields across the shock, FLASH-proportional partition behavior. Sweep riemann solvers (hlle/hllc/hlld/llf) × reconstruction (plm/ppm4/wenoz) × rk2/rk3 — scalar upwinding and dual-energy interplay differ per combination; only plm+hlle+rk2 has ever run.
- [x] **T-2T2 — Exchange limit cases.** `t_ei=0` (instant equilibration), `t_ei<0` (off), asymmetric `electron_heat_capacity_fraction` (0.1, 0.9), `initial_electron_temperature_ratio` ∈ {0, 10}. Verify analytic decay `ΔT ∝ exp[-(1+cv_e/cv_i)t/t_ei]` for each cv split.
- [x] **T-2T3 — Dual-energy selector sweep.** `dual_energy_eta1/eta2` ∈ {0, defaults, large}; magnetically dominated case (exists) *plus* a well-conditioned case verifying the conservative branch preserves total energy to machine precision; `dual_energy=false` with 2T (Sync-only path).
- [x] **T-2T4 — Error-guard matrix.** Confirm clean fatal exits: 2T+isothermal EOS, 2T+SR/GR, 2T with `<ion-neutral>`/`<radiation>`, `dual_energy` without 2T, `dual_energy`+FOFC, `dual_energy`+viscosity/conduction/srcterms, `<thermal_radiation>` without 2T, Biermann without 2T / in 1D / with ng<2, laser without MHD, laser without 2T, refractive+critical_reflection. One athinput each; expect nonzero exit + sensible message (wrap MPI cases in `timeout`, see F4).
- [x] **T-2T5 — Hydro-side multi-D/MPI.** All hydro 2T tests are 1D serial. Run 2D hydro 2T relax + advection at 1/2/4 ranks; ghost-zone Sync/Exchange with reflect and outflow BCs (only periodic ever tested).

### 3b. Thermal radiation + opacity tables (`thermal_radiation.cpp`, `opacity_table.cpp`)

- [x] **T-R1 — Quantitative diffusion-rate validation.** Existing diffusion test checks variance *decay*, not the rate. Add a 1D Gaussian-pulse spread vs analytic linear-diffusion solution (optically thick, limiter≈1/3 regime); repeat 2D/3D to exercise transverse-gradient branches in all three `AddFluxes` kernels (`thermal_radiation.cpp:341-414`).
- [x] **T-R2 — Limiter matrix.** All five limiters (`none`, `harmonic`, `larsen`, `minmax`, `levermore-pomraning`) — `larsen`/`none`/`minmax` have never executed anywhere. Include a free-streaming (κ→small) case: check flux capped at c_hat·E and the expected dt collapse (document the explicit-diffusion dt cost).
- [x] **T-R3 — Coupling stress.** Relaxation to T_e=T_rad equilibrium (exact lagged-emission/implicit-absorption formula already in the CPU test — reuse); asymmetric `kappa_emission≠kappa_absorption`; near-zero e_ele to hit the emission-rescale clamp (`thermal_radiation.cpp:480-505`); `source_cfl` on/off.
- [x] **T-R4 — Opacity tables.** Linear vs log interpolation equivalence on a log-generated table; all `opacity_*_scale` params vs a pre-scaled table; out-of-range (ρ,T) clamping; malformed-file negative tests (wrong group count, mismatched `group_bound`, trailing tokens) — expect clean errors (F4 caveat).
- [x] **T-R5 — MHD radiation deck.** `inputs/mhd/two_temperature_mgfld.athinput` (larsen limiter) is referenced by no test — run it; the MHD hook path (`mhd_tasks.cpp:216-218`) is parallel code to the tested hydro path.
- [x] **T-R6 — MPI/SMR conservation.** Diffusion step-front (`initial_profile=step`) with a static refinement boundary through the front, `couple_matter=false`: total `Σ rho·erad_g` conserved through fine/coarse flux correction; 1 vs 2 vs 4 ranks identical to round-off. Hydro scalar flux correction of diffusive fluxes at level boundaries has **zero** existing coverage.

### 3c. Laser (`src/laser/`)

- [x] **T-L1 — CPU re-validation of GPU-only results.** Rerun `test-2t/inputs/two_temperature_laser_gpu.athinput` analytics (Beer–Lambert per-cell deposition, electron-only heating, ray-count/τ/path checks) on B1/B2 builds; compare to GPU within 2e-11.
- [x] **T-L2 — Oblique reflection quantitative test.** Reflection is only round-trip-tested. Add oblique incidence onto a linear ramp: measured turning depth vs n_c·cos²θ for several θ; `oblique_turning=false` control turning at n_c. (`inputs/mhd/two_temperature_laser_reflection.athinput` as base.)
- [x] **T-L3 — Refraction convergence.** Analytic parabolic trajectory in constant gradient; sweep `refractive_cell_fraction` {0.5, 0.25, 0.1} for convergence order; `laser_dispersion_error` < tolerance throughout.
- [x] **T-L4 — Inverse bremsstrahlung κ unit test.** Hand-compute FLASH κ at fixed (n_e, T_e, Z, λ) and compare a 1-cell deposition; test **both** fixed and auto Coulomb log — the local Debye-log branch is completely unvalidated (FLASH tube forced `coulomb_log=1`).
- [x] **T-L5 — Beam geometry.** `nbeams≥2`; 3D Fibonacci-disk and 2D line apertures; `gaussian` vs `uniform` (Σ ray powers == beam power exactly); `start_time`/`end_time` gating mid-run.
- [x] **T-L6 — Work-cap behavior.** Deliberately undersized `max_transport_iterations`/`max_segments_per_launch`/`max_reflections_per_ray`: power must land in `remaining` (conservation intact), diagnostics must flag it. This reproduces the failure mode seen in the archived `logs/laser_limited_gpu*.log` stall.
- [x] **T-L7 — Periodic transport.** `periodic_transport=true` on a fully periodic mesh with κ>0 (ray wraps and extinguishes) and κ=0 (runs to iteration cap → `remaining`, must not hang); error out on non-periodic mesh.
- [x] **T-L8 — `deposition_target=total`** (never run): verify IEN-only deposition, ions+electrons heated via Sync proportionality instead of electron-only.
- [x] **T-L9 — MPI stress.** ≥8 ranks, many blocks/rank, beams crossing ranks repeatedly (target ≫99 transfers, the current max observed); odd rank count (3); `gpu_aware_mpi=true` on B3 (never run — pinned-host staging is the only exercised path).
- [x] **T-L10 — Negative/output tests.** Post-F1-fix: request each laser output var without `<laser>` → clean error, not segfault. Beam through collapsed dimension, refractive launch above n_c, IB without cgs scales → clean fatals.

### 3d. Biermann battery + dual energy (`biermann_battery.cpp`, `mhd_dual_energy.cpp`)

- [x] **T-B1 — MPI invariance.** The 2D/3D analytic Biermann tests at 2/4 ranks (Biermann EMFs cross MPI block boundaries through `SendE/RecvE` — untested); B1=B2=0 symmetry and rate error match serial.
- [x] **T-B2 — Shock robustness.** Brio–Wu / blast with `biermann_battery=true`: exercise `ComputeShockMask` (suppression on/off, threshold sweep 0.1–0.5); positivity of all energies; the regression test is smooth-flow only.
- [x] **T-B3 — Zero-coefficient inertness.** `biermann_coefficient=0` bitwise-identical to `biermann_battery=false` (already true in laser-target control — make it a standalone check).
- [x] **T-B4 — SMR/AMR Biermann.** Static refined region straddling the gradient: monitor div(B)=0 (CT across levels), total energy, and e_ele conservation at level boundaries — this quantifies the uncorrected `vd*_` drift issue (F5). Then adaptive + MPI.
- [x] **T-B5 — Dynamic multi-D SMR dual-energy.** A shock crossing a refinement boundary in 2D with `dual_energy=true`: `dual_vf` flux-correction machinery (`flux_correct_cc.cpp:242-360`) has only ever moved zero-velocity or 1D data. Check conservation + no aux/total divergence at the interface.

## Phase 4 — Cross-cutting integration tests (zero current coverage)

Ranked by risk (from the code-audit critique):

- [x] **T-X1 — Adaptive AMR + MPI + laser.** `two_temperature_laser_amr.athinput` with adaptive refinement, ≥2 ranks, load balancing enabled: `cell_data` pack/unpack across ranks (`load_balance.cpp:555-558, 851-854`) and post-regrid `RefreshGlobalBlockInfo` have never executed together. Verify no failed rays, conservation each stage, `laser_energy` survives refine/derefine/migration.
- [x] **T-X2 — Restart round-trips.** (a) 2T relaxation: stop at t/2, restart, compare to uninterrupted (verifies `InitializeTwoTemperature` skip on restart, `driver.cpp:313-321`); (b) radiation diffusion mid-front; (c) dual-energy+Biermann run; (d) laser run — expect `laser_energy` reset (F6) but physical state continuous; (e) restart with different rank count.
- [x] **T-X3 — Radiation + AMR dt stability.** Refinement halves dx → explicit diffusion dt shrinks 4×, but `NewTimeStep` ran pre-refinement (`driver.cpp:452-454`). Construct a case where refinement triggers in an optically thick region and check for post-refinement instability/overshoot in E_g. Outcome decides whether a post-regrid dt recompute is needed.
- [x] **T-X4 — Units consistency audit.** Only the laser reads `<units>`; `arad`, `c_light`, `biermann_coefficient` are bare code-unit knobs. Write a worked example (docs) deriving mutually consistent `<units>` + `<thermal_radiation>` + `<laser>` scales for one physical scenario (e.g. the laser-target setup in real units); add an input-echo check that flags `c_light` inconsistent with laser cgs scales.
- [x] **T-X5 — Full-coupling benchmark rerun.** `laser-target/` (laser→electron→radiation→Biermann) at current HEAD on B3; extend to 4 ranks; confirm closure metrics match `laser-target/results.json` (field diff ≤1e-9-level, not bitwise — documented).
- [x] **T-X6 — BC surface.** Reflect + inflow + user BCs with 2T/radiation (ghost-zone Sync/Exchange correctness); shearing-box guard check for hydro 2T (only MHD dual-energy is guarded, `mhd.cpp:219-226` — decide guard vs support).
- [x] **T-X7 — CPU↔GPU cross-check.** Same decks on B1 vs B3 (rk1 to minimize reordering): field agreement to ~1e-11 rtol for 2T/radiation; laser deposition agreement (atomics ordering may give round-off-level scatter — record actual tolerance).

## Phase 5 — Benchmark harness refresh & scale

- [x] **P1 — Rerun `test-2t/run_all.py`** (5 features × 1/2/4 GPUs) at current HEAD; requires clean `runs/` dirs; GPU IDs default 2–5 (`run_all.py:81`). All 15 cells PASS, cross-rank scaled error 0.
- [x] **P2 — Rerun FLASH laser tube + SMR variants** (`test-2t/flash-laser-tube{,-smr}/`): exit-power error <1e-3 vs Kaiser (2000) analytic 0.7017811; note documented non-monotonic resolution behavior (r64 worse than r32) — investigate if time permits, else keep documented.
- [x] **P3 — Scale-up probe.** One 128³+ multi-level laser+2T run at 4 GPUs: watch the O(nmb_total) per-crossing block searches (`laser_trace.cpp:40-64`) and per-stage host mirror copies (`laser_trace.cpp:1049-1058`) for scaling pathology; record stage-time breakdown (laser-target already showed 2.5× slowdown at 2 ranks).
- [x] **P4 — Performance baseline.** zone-cycles/sec with laser on/off and radiation on/off for a fixed deck; archive in diagnostics for future regression.

## Phase 6 — Documentation & CI closure

- [x] **D1 — Module docs.** Ensure `docs/laser_algorithm.md` matches final behavior (esp. restart caveat F6, work-cap semantics T-L6, unit conventions T-X4); add equivalent short docs for 2T/thermal-radiation parameters (the full block/param tables exist in this plan's source reports).
- [x] **D2 — Wire branch tests into CI** (`.github/workflows/main.yml` runs `--cpu`/`--mpicpu` upstream): confirm the 7 new tests run there or add them; add the negative-test matrix (T-2T4) as a fast job.
- [x] **D3 — Final report.** Summarize the full matrix (build × module × parallelism × mesh) with pass/fail + measured errors; list accepted limitations explicitly (no radiation momentum/pressure, n_e∝ρ fixed ionization, laser restart reset if F6 deferred, explicit-diffusion dt cost, `vd*_` AMR inconsistency if F5 deferred).

---

## Execution notes

- **Order**: Phase 0 → F1/F2/F4 fixes → Phase 2 (baseline) → Phases 3–4 (parallelizable by module) → Phase 5 → Phase 6. Phases 3a–3d are independent of each other; within each, serial before MPI before GPU.
- **Every MPI negative test under `timeout 120`** until F4 is fixed (`std::exit` without `MPI_Abort` can hang mpirun).
- **Never overwrite `runs/`**: the harness runners refuse; use fresh out-dirs or clean explicitly.
- Conservation thresholds proven achievable on this branch: 2e-11 (2T/radiation drift), 1e-10 (laser power residual), 2e-12 (field symmetry) — reuse them for new tests rather than inventing looser ones.
- Reference analytic setups already in-tree: exact relaxation formulas (`tst/test_suite/nr/test_nr_two_temperature_cpu.py`), lagged-emission update formulas (`test_nr_thermal_radiation_cpu.py:111-127`), DDA geometry checks (`test_nr_laser_cpu.py`), Kaiser tube (`test-2t/flash-laser-tube/SOURCES.md`).
