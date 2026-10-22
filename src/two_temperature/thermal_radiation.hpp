#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.hpp
//! \brief Multigroup flux-limited diffusion coupled to two-temperature electrons.

#include <cstddef>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "materials/material_mixture.hpp"
#include "two_temperature/thermal_radiation_timers.hpp"

namespace diffusion { class NativeDiffusionSolver; }

class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace two_temperature {

class OpacityTable;
class MixedOpacityTable;
struct ImplicitAMRAggregateState;
class ImplicitRadiationHalo;
struct ImplicitRadiationGroupWorkers;

//----------------------------------------------------------------------------------------
//! \class ThermalRadiation
//! \brief Evolves comoving radiation-group energy densities with FLD.
//!
//! Radiation groups are stored as advected fluid scalars after the ion and electron
//! energies.  The default explicit integrator adds diffusive group fluxes to the
//! finite-volume fluid fluxes.  The optional FLASH-like integrator advances diffusion
//! with a time-lagged backward-Euler solve, including conservative SMR/AMR face
//! coupling; it does not use the explicit asymptotic-preserving upwind correction.
//! Absorption and Planck emission are operator split and exchange energy only with
//! the electron component.

class ThermalRadiation {
 public:
  ThermalRadiation(MeshBlockPack *ppack, ParameterInput *pin, int first_group_index,
                   int electron_index, Real gamma_minus_one,
                   Real electron_heat_capacity_fraction,
                   materials::MaterialMixture *material_mixture = nullptr);
  ~ThermalRadiation();

  int ngroups;
  int ifirst;
  Real dtnew;
  int implicit_iterations_last_solve = 0;
  Real implicit_residual_last_solve = 0.0;
  int implicit_residual_replacements_last_solve = 0;
  Real implicit_backward_error_last_solve = 0.0;
  int implicit_amr_fallbacks_last_solve = 0;
  int source_iterations_last_solve = 0;
  int source_fallbacks_last_solve = 0;
  Real source_residual_last_solve = 0.0;
  // Rank-local, step-averaged outward diffusive power through physical boundaries.
  Real implicit_boundary_power = 0.0;

  // component 0 = total specific radiation energy, component 1 = radiation temperature
  DvceArray5D<Real> diagnostics;

  void Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  int il, int iu, int jl, int ju, int kl, int ku);
  void UpdateDiagnostics(const DvceArray5D<Real> &cons, const DvceArray5D<Real> &prim,
                         int il, int iu, int jl, int ju, int kl, int ku);
  void AddFluxes(const DvceArray5D<Real> &prim,
                 const DvceArray5D<Real> &temperature,
                 DvceFaceFld5D<Real> &flx, FaceFldOrigin origin = {});
  void Couple(Real dt, DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
              DvceArray5D<Real> &temperature,
              Real material_pressure_floor, Real material_temperature_floor,
              int il, int iu, int jl, int ju, int kl, int ku);
  bool IsImplicit() const { return implicit_transport_; }
  void SolveImplicitTransport(Real dt, DvceArray5D<Real> &cons,
                              DvceArray5D<Real> &prim,
                              const DvceArray5D<Real> &temperature,
                              MeshBoundaryValuesCC *pbval);
  void NewTimeStep(const DvceArray5D<Real> &prim,
                   const DvceArray5D<Real> &temperature);

 private:
  // A worker shares immutable material/opacity data and owns separate solver storage.
  ThermalRadiation(const ThermalRadiation &) = default;
  int implicit_group_batch_size_ = 1;
  bool implicit_batch_worker_ = false;
  bool implicit_batch_retry_ = false;
  // A converged high-order iterate is retained for a deferred monotone retry.
  bool implicit_batch_retry_low_order_ = false;
  bool borrowed_opacity_ = false;
  int implicit_group_begin_ = 0, implicit_group_end_ = -1;
  ImplicitRadiationGroupWorkers *implicit_group_workers_ = nullptr;
  void SolveImplicitTransportBatched(Real dt, DvceArray5D<Real> &cons,
      DvceArray5D<Real> &prim, const DvceArray5D<Real> &temperature,
      MeshBoundaryValuesCC *pbval);
  void DestroyImplicitGroupWorkers();
  MeshBlockPack *pmy_pack_;
  int iele_;
  int limiter_mode_;
  Real gamma_minus_one_;
  Real cv_e_fraction_;
  Real arad_;
  Real chat_;
  Real flux_limit_coefficient_;
  Real initial_radiation_temperature_;
  Real initial_radiation_temperature_right_;
  Real initial_radiation_x1_;
  Real energy_floor_;
  Real source_cfl_;
  bool nonlinear_source_ = true;
  bool source_report_ = false;
  Real source_nonlinear_tolerance_ = 1.0e-10;
  Real source_nonlinear_absolute_tolerance_ = 0.0;
  int source_max_iterations_ = 80;
  int source_fallback_substeps_ = 8;
  Real ap_streaming_threshold_;
  Real ap_optical_depth_threshold_;
  int initial_profile_mode_;
  bool couple_matter_;
  bool use_ap_transport_;
  bool implicit_transport_ = false;
  bool implicit_report_ = false;
  bool solver_summary_ = false;
  mutable RadiationSolverTimers solver_timers_;
  Real implicit_tolerance_ = 1.0e-10;
  int implicit_max_iterations_ = 400;
  int implicit_residual_check_interval_ = 50;
  // Composite interface order: 1=monotone two-point/CG, 2=quadratic FV/GMRES.
  int implicit_amr_order_ = 2;
  int implicit_gmres_restart_ = 30;
  int implicit_gmres_restart_max_ = 30;
  // CGS with two full passes batches the Arnoldi projections into one reduction/pass.
  bool implicit_gmres_cgs2_ = false;
  // Optional fixed SPD preconditioner metric for the outer AMR Arnoldi basis.
  bool implicit_gmres_preconditioner_metric_ = false;
  // Optional truncated two-point CG preconditioner for flexible AMR GMRES.
  int implicit_amr_inner_iterations_ = 0;
  bool implicit_amr_inner_preconditioner_norm_ = false;
  bool implicit_amr_high_order_ = false;
  // 0=point Jacobi, 1=factor-three global Galerkin V-cycle with symmetric red/black
  // Gauss-Seidel smoothing and an exact global MeshBlock-root solve. Refined meshes
  // use an additive diagonal plus volume-weighted block-constant coarse correction.
  // 2=Jacobi/local aggregate diagonals plus a global factor-nine CHOLMOD solve.
  // 3=Jacobi/local aggregate diagonals plus a factor-three BoomerAMG V-cycle.
  // 4=cell-level BoomerAMG on a single-level mesh (no geometric aggregation).
  // 5=native Kokkos aggregation multigrid on the conservative leaf-cell graph.
  int implicit_preconditioner_mode_ = 0;
  diffusion::NativeDiffusionSolver *implicit_native_solver_ = nullptr;
  bool implicit_native_device_mpi_ = false;
  std::size_t implicit_native_max_bytes_ = 1024ULL*1024*1024;
  // 0=automatic device/host selection, 1=host CHOLMOD, 2=device sparse factor solve.
  int implicit_amr_coarse_solver_ = 0;
  // AMG setup estimate budget; available GPU memory is checked separately.
  std::size_t implicit_amr_amg_max_bytes_ = 1024ULL*1024ULL*1024ULL;
  // Optional cell-AMG transfer reuse; numerical matrices stay current each solve.
  bool implicit_amg_reuse_ = false;
  int implicit_amg_reuse_steps_ = 4, implicit_amg_reuse_slots_ = 5;
  Real implicit_amg_reuse_drift_ = 64.0;
  ImplicitAMRAggregateState *implicit_amr_aggregate_state_ = nullptr;
  bool implicit_multilevel_enabled_ = false;
  bool implicit_multilevel_has_remote_neighbors_ = false;
  int implicit_multilevel_nx1_[2] = {0, 0};
  int implicit_multilevel_nx2_[2] = {0, 0};
  int implicit_multilevel_nx3_[2] = {0, 0};
  int implicit_multilevel_offset_[2] = {0, 0};
  // 0=zero-gradient (Neumann), 1=fixed face value (Dirichlet), 2=zero ghost
  // value (the vacuum convention used by the existing explicit DCI boundary).
  int implicit_boundary_type_[6] = {0, 0, 0, 0, 0, 0};
  Real implicit_boundary_value_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Immutable face/group spectra; each scalar/batched worker selects its own values.
  std::vector<Real> implicit_boundary_group_values_;
  bool use_material_mixture_ = false;
  materials::MaterialMixtureDevice material_mixture_;
  bool use_opacity_table_ = false;
  OpacityTable *opacity_table_ = nullptr;
  bool use_mixed_opacity_table_ = false;
  MixedOpacityTable *mixed_opacity_table_ = nullptr;

  DualArray1D<Real> group_bounds_;
  DualArray1D<Real> kappa_transport_;
  DualArray1D<Real> kappa_absorption_;
  DualArray1D<Real> kappa_emission_;
  DualArray1D<int> source_integer_stats_;
  DualArray1D<Real> source_real_stats_;

  // One component per worker; storage scales with the bounded batch size.
  DvceArray5D<Real> implicit_old_;
  DvceArray5D<Real> implicit_solution_;
  DvceArray5D<Real> implicit_coefficient_;
  DvceArray5D<Real> implicit_residual_;
  DvceArray5D<Real> implicit_direction_;
  DvceArray5D<Real> implicit_preconditioned_;
  DvceArray5D<Real> implicit_operator_;
  DvceArray5D<Real> implicit_coarse_scratch_;

  // Composite AMR operator: fine-face conductances are frozen per group.  Refluxed
  // fluxes give the coarse cell exactly the opposite integrated fine-cell flux.
  // Only active faces are stored; cell-centered stencil donors retain their halos.
  DvceFaceFld5D<Real> implicit_amr_conductance_{"rad_amr_conductance", 0, 0, 0, 0, 0};
  DvceFaceFld5D<Real> implicit_amr_flux_{"rad_amr_flux", 0, 0, 0, 0, 0};
  FaceFldOrigin implicit_amr_face_origin_;
  BandView5D<Real> ImplicitAMRFaceBand(const DvceArray5D<Real> &face) const {
    return MakeBandView(face, implicit_amr_face_origin_);
  }
  // Diagonals and coefficient gradients are local to active cells. Their origin
  // matches the active face registers; solution/coefficient donor halos are separate.
  BandView5D<Real> ImplicitAMRCellBand(const DvceArray5D<Real> &cell) const {
    return MakeBandView(cell, implicit_amr_face_origin_);
  }
  DvceArray5D<Real> implicit_amr_diagonal_;
  // Volume-weighted Galerkin matrix for one constant basis per active MeshBlock.
  // Uses the monotone AMR face graph as an SPD coarse correction for either order.
  DualArray2D<Real> implicit_amr_block_matrix_;
  DvceArray5D<Real> implicit_amr_gradient_;
  // Store only active cells of the Arnoldi vectors; matvec scratch includes halos.
  DvceArray2D<Real> implicit_gmres_basis_;
  DvceArray2D<Real> implicit_gmres_corrections_;
  DvceArray5D<Real> implicit_amr_inner_z_;
  MeshBoundaryValuesCC *implicit_bvals_ = nullptr;
  // 0=legacy block messages, 1=packed device buffers, 2=packed host staging.
  int implicit_halo_exchange_ = 1;
  bool implicit_halo_faces_ = true;
  bool implicit_halo_compact_ = true;
  ImplicitRadiationHalo *implicit_halo_ = nullptr;

  // The multilevel preconditioner packs the 15^3 and 5^3 Galerkin diagonals into one
  // allocation.  For a 45^3 MeshBlock this is (15^3+5^3)/45^3 = 3.84% of one fine
  // field.  Existing fine solver arrays are reused for coarse right-hand sides/solutions.
  DvceArray2D<Real> implicit_multilevel_vector_;
  // Six frozen face conductances per aggregate, rebuilt once per group alongside
  // the diagonals.  Keeping them avoids fine-face summations in every V-cycle.
  DvceArray3D<Real> implicit_multilevel_conductance_;
  DvceArray3D<Real> implicit_multilevel_send_faces_;
  DvceArray3D<Real> implicit_multilevel_recv_faces_;
  DualArray2D<int> implicit_coarse_neighbor_gid_device_;
  DualArray2D<int> implicit_coarse_neighbor_rank_device_;
  DualArray2D<int> implicit_multilevel_block_parity_;
#if MPI_PARALLEL_ENABLED
  MPI_Comm implicit_multilevel_comm_ = MPI_COMM_NULL;
#endif

  // Six root-face sums and one scalar are stored per global MeshBlock.  The dense root
  // Cholesky factor is host-only and is rebuilt for each frozen group operator.
  DualArray2D<Real> implicit_coarse_faces_;
  DualArray1D<Real> implicit_coarse_vector_;
  std::vector<int> implicit_coarse_neighbor_gid_;
  std::vector<Real> implicit_coarse_scaling_;
  std::vector<Real> implicit_coarse_cholesky_;

 public:
  // Public because nvcc requires any member enclosing an extended device lambda to have
  // public access.  These remain implementation details of SolveImplicitTransport().
  void ApplyImplicitPhysicalBoundaries(DvceArray5D<Real> &field,
                                       bool homogeneous_boundary,
                                       bool coefficient_field = false,
                                       bool coarse_grid = false);
  void ExchangeImplicitField(DvceArray5D<Real> &field,
                             MeshBoundaryValuesCC *pbval,
                             bool homogeneous_boundary,
                             bool coefficient_field = false,
                             bool inner_halo = false);
  void ApplyImplicitOperator(const DvceArray5D<Real> &field,
                             DvceArray5D<Real> &result, Real dt);
  Real ImplicitGlobalDot(const DvceArray5D<Real> &lhs,
                         const DvceArray5D<Real> &rhs) const;
  Real ImplicitComponentwiseBackwardError(
      const DvceArray5D<Real> &field, const DvceArray5D<Real> &rhs,
      const DvceArray5D<Real> &residual, Real dt);
  void BuildImplicitAMROperator(Real dt);
  void PrepareImplicitAMRGradient(const DvceArray5D<Real> &field);
  void BuildImplicitAMRBlockCoarsePreconditioner(Real dt);
  void ApplyImplicitAMRBlockCoarseCorrection(const DvceArray5D<Real> &residual,
                                            DvceArray5D<Real> &preconditioned);
  static bool HasImplicitAMRAggregateSupport();
  static bool HasImplicitAMRAggregateDeviceSupport();
  static bool HasImplicitAMRAMGSupport();
  void DestroyImplicitAMRAggregateState();
  void BuildImplicitAMRAggregatePreconditioner(Real dt);
  void ApplyImplicitAMRAggregatePreconditioner(const DvceArray5D<Real> &residual,
                                              DvceArray5D<Real> &preconditioned);
  bool SolveImplicitAMRGMRES(Real dt, MeshBoundaryValuesCC *pbval, Real scale,
                            int &iterations, int &restarts,
                            Real &relative_residual, Real &backward_error);
  void ApplyImplicitAMROperator(const DvceArray5D<Real> &field,
                                DvceArray5D<Real> &result, Real dt,
                                bool absolute_values = false);
  void RefluxImplicitAMRFluxes();
  void BuildImplicitBlockCoarsePreconditioner(Real dt);
  void ExchangeImplicitMultilevelFaces(const DvceArray5D<Real> &field, int level);
  void SolveImplicitBlockRootSystem();
  void ApplyImplicitPreconditioner(const DvceArray5D<Real> &residual,
                                   DvceArray5D<Real> &preconditioned, Real dt);
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
