#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.hpp
//! \brief Multigroup flux-limited diffusion coupled to two-temperature electrons.

#include <vector>

#include "athena.hpp"
#include "materials/material_mixture.hpp"

class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace two_temperature {

class OpacityTable;
class MixedOpacityTable;

//----------------------------------------------------------------------------------------
//! \class ThermalRadiation
//! \brief Evolves comoving radiation-group energy densities with FLD.
//!
//! Radiation groups are stored as advected fluid scalars after the ion and electron
//! energies.  The default explicit integrator adds diffusive group fluxes to the
//! finite-volume fluid fluxes.  The optional FLASH-like integrator advances diffusion
//! with a centered, time-lagged backward-Euler solve; it does not use the explicit
//! asymptotic-preserving upwind correction.  Absorption and Planck emission are operator
//! split and exchange energy only with the electron component.

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
                 DvceFaceFld5D<Real> &flx);
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
  Real implicit_tolerance_ = 1.0e-10;
  int implicit_max_iterations_ = 400;
  int implicit_residual_check_interval_ = 50;
  // 0=point Jacobi, 1=factor-three global Galerkin V-cycle with symmetric red/black
  // Gauss-Seidel smoothing and an exact global MeshBlock-root solve.  Incompatible
  // MeshBlock sizes fall back to point Jacobi.
  int implicit_preconditioner_mode_ = 0;
  bool implicit_multilevel_enabled_ = false;
  int implicit_multilevel_nx1_[2] = {0, 0};
  int implicit_multilevel_nx2_[2] = {0, 0};
  int implicit_multilevel_nx3_[2] = {0, 0};
  int implicit_multilevel_offset_[2] = {0, 0};
  // 0=zero-gradient (Neumann), 1=fixed face value (Dirichlet), 2=zero ghost
  // value (the vacuum convention used by the existing explicit DCI boundary).
  int implicit_boundary_type_[6] = {0, 0, 0, 0, 0, 0};
  Real implicit_boundary_value_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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

  // One component is solved at a time so the storage cost does not scale with ngroups.
  DvceArray5D<Real> implicit_old_;
  DvceArray5D<Real> implicit_solution_;
  DvceArray5D<Real> implicit_coefficient_;
  DvceArray5D<Real> implicit_residual_;
  DvceArray5D<Real> implicit_direction_;
  DvceArray5D<Real> implicit_preconditioned_;
  DvceArray5D<Real> implicit_operator_;
  DvceArray5D<Real> implicit_coarse_scratch_;

  // The multilevel preconditioner packs the 15^3 and 5^3 Galerkin diagonals into one
  // allocation.  For a 45^3 MeshBlock this is (15^3+5^3)/45^3 = 3.84% of one fine
  // field.  Existing fine solver arrays are reused for coarse right-hand sides/solutions.
  DvceArray2D<Real> implicit_multilevel_vector_;
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
                                       bool coefficient_field = false);
  void ExchangeImplicitField(DvceArray5D<Real> &field,
                             MeshBoundaryValuesCC *pbval,
                             bool homogeneous_boundary,
                             bool coefficient_field = false);
  void ApplyImplicitOperator(const DvceArray5D<Real> &field,
                             DvceArray5D<Real> &result, Real dt);
  Real ImplicitGlobalDot(const DvceArray5D<Real> &lhs,
                         const DvceArray5D<Real> &rhs) const;
  Real ImplicitComponentwiseBackwardError(
      const DvceArray5D<Real> &field, const DvceArray5D<Real> &rhs,
      const DvceArray5D<Real> &residual, Real dt) const;
  void BuildImplicitBlockCoarsePreconditioner(Real dt);
  void ExchangeImplicitMultilevelFaces(const DvceArray5D<Real> &field, int level);
  void SolveImplicitBlockRootSystem();
  void ApplyImplicitPreconditioner(const DvceArray5D<Real> &residual,
                                   DvceArray5D<Real> &preconditioned, Real dt);
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
