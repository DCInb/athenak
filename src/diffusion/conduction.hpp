#ifndef DIFFUSION_CONDUCTION_HPP_
#define DIFFUSION_CONDUCTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.hpp
//! \brief Contains data and functions that implement various formulations for conduction.
//  Currently only isotropic conduction implemented

#include <cstddef>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

namespace diffusion { class NativeDiffusionSolver; }

class MeshBlockPack;
class MeshBoundaryValuesCC;
struct EOS_Data;
namespace materials {
class MaterialMixture;
}
namespace two_temperature {
class TwoTemperature;
class ImplicitRadiationHalo;
}

//----------------------------------------------------------------------------------------
//! \class Conduction
//! \brief data and functions that implement thermal conduction in Hydro and MHD

class Conduction {
 public:
  Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin,
             two_temperature::TwoTemperature *ptwo_temp = nullptr,
             materials::MaterialMixture *pmaterials = nullptr);
  ~Conduction();

  // data
  Real dtnew;
  Real alpha_iso;       // isotropic thermal diffusivity
  Real alpha_aniso;     // anisotropic thermal diffusivity
  bool alpha_spitzer;   // switch to turn on Spitzer conductivity
  Real q_limit;         // saturated heat flux limit

  // The legacy flux-integrated path remains the default.  The implicit path is an
  // operator-split electron-temperature solve used by two-temperature Hydro/MHD.
  bool IsImplicit() const { return implicit_; }
  int iterations_last_solve = 0;
  int nonlinear_iterations_last_solve = 0;
  Real residual_last_solve = 0.0;

  // functions
  void AddHeatFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                     DvceFaceFld5D<Real> &f, FaceFldOrigin origin = {});
  void AddHeatFluxIso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                      DvceFaceFld5D<Real> &f, FaceFldOrigin origin = {});
  void AddHeatFluxAniso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f, FaceFldOrigin origin = {});
  void AddHeatFluxSpitzer(const DvceArray5D<Real> &w, const EOS_Data &eos,
                          DvceFaceFld5D<Real> &f, FaceFldOrigin origin = {});
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
  void SolveImplicit(Real dt, DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                     MeshBoundaryValuesCC *pbval,
                     two_temperature::TwoTemperature *ptwo_temp);

 private:
  enum class FluxLimiter : int { none = 0, harmonic = 1, minmax = 2, larsen = 3 };
  enum class BoundaryType : int { neumann = 0, dirichlet = 1 };

  MeshBlockPack* pmy_pack;
  materials::MaterialMixture *pmaterials_ = nullptr;
  bool implicit_ = false;
  bool report_ = false;
  bool native_mg_ = false, native_device_mpi_ = false;
  std::size_t native_max_bytes_ = 1024ULL*1024*1024;
  diffusion::NativeDiffusionSolver *native_solver_ = nullptr;
  // Shared scalar halo: 0=legacy, 1=packed device, 2=packed host staging.
  int halo_exchange_ = 1;
  bool compact_halo_ = true;
  two_temperature::ImplicitRadiationHalo *implicit_halo_ = nullptr;
  Real theta_ = 1.0;
  Real linear_tolerance_ = 1.0e-10;
  Real nonlinear_tolerance_ = 1.0e-8;
  int max_iterations_ = 400;
  int max_nonlinear_iterations_ = 8;
  Real coulomb_log_ = 10.0;
  Real spitzer_multiplier_ = 1.0;
  Real spitzer_temperature_floor_kelvin_ = 1.0;
  Real flux_limit_coefficient_ = 0.06;
  FluxLimiter flux_limiter_ = FluxLimiter::harmonic;
  BoundaryType boundary_type_[6] = {
      BoundaryType::neumann, BoundaryType::neumann,
      BoundaryType::neumann, BoundaryType::neumann,
      BoundaryType::neumann, BoundaryType::neumann};
  Real boundary_value_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real gamma_minus_one_ = 0.0;
  Real electron_heat_capacity_fraction_ = 0.0;

  // Single-component solver fields.  All have the ordinary AthenaK cell-centered
  // layout so the existing MeshBlock/MPI halo exchange can be reused.
  DvceArray5D<Real> temperature_old_;
  DvceArray5D<Real> temperature_new_;
  DvceArray5D<Real> conductivity_;
  DvceArray5D<Real> capacity_;
  DvceArray5D<Real> energy_old_;
  DvceArray5D<Real> explicit_laplacian_;
  DvceArray5D<Real> residual_;
  DvceArray5D<Real> direction_;
  DvceArray5D<Real> preconditioned_;
  DvceArray5D<Real> operator_direction_;
  DvceArray5D<Real> correction_;
  DvceArray5D<Real> coarse_scratch_;
  // Frozen positive face graph and its refluxed diagonal on a composite leaf mesh.
  MeshBoundaryValuesCC *implicit_bvals_ = nullptr;
  DvceFaceFld5D<Real> amr_conductance_{"cond_amr_conductance", 0, 0, 0, 0, 0};
  DvceFaceFld5D<Real> amr_flux_{"cond_amr_flux", 0, 0, 0, 0, 0};
  // Only faces surrounding active cells are used by the operator and reflux.
  FaceFldOrigin amr_face_origin_;
  BandView5D<Real> AMRFaceBand(const DvceArray5D<Real> &face) const {
    return MakeBandView(face, amr_face_origin_);
  }
  DvceArray5D<Real> amr_diagonal_;

 public:
  // Public because nvcc requires any member enclosing an extended device lambda to have
  // public access.  These remain implementation details of SolveImplicit().
  void ExchangeSolverField(DvceArray5D<Real> &field,
                           MeshBoundaryValuesCC *pbval,
                           bool homogeneous_boundary, bool coefficient_field = false,
                           bool full_halo = false);
  void ApplyPhysicalBoundaries(DvceArray5D<Real> &field,
                               bool homogeneous_boundary, bool coefficient_field = false);
  void BuildAMROperator();
  void ApplyAMRDiffusionOperator(const DvceArray5D<Real> &field,
                                DvceArray5D<Real> &result);
  void RefluxAMRFluxes();
  void ApplyDiffusionOperator(const DvceArray5D<Real> &field,
                              DvceArray5D<Real> &result);
  void ApplyJacobian(const DvceArray5D<Real> &field,
                     DvceArray5D<Real> &result, Real dt);
  Real GlobalDot(const DvceArray5D<Real> &lhs,
                 const DvceArray5D<Real> &rhs) const;
};
#endif // DIFFUSION_CONDUCTION_HPP_
