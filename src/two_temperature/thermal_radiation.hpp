#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.hpp
//! \brief Multigroup flux-limited diffusion coupled to two-temperature electrons.

#include "athena.hpp"
#include "materials/material_mixture.hpp"

class MeshBlockPack;
class ParameterInput;

namespace two_temperature {

class OpacityTable;
class MixedOpacityTable;

//----------------------------------------------------------------------------------------
//! \class ThermalRadiation
//! \brief Evolves comoving radiation-group energy densities with explicit FLD.
//!
//! Radiation groups are stored as advected fluid scalars after the ion and electron
//! energies.  Diffusive group fluxes are added to the finite-volume fluid fluxes, so
//! existing boundary communication, flux correction, AMR, and restart support apply.
//! Absorption and Planck emission are operator split and exchange energy only with the
//! electron component.

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
              int il, int iu, int jl, int ju, int kl, int ku);
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
  Real ap_streaming_threshold_;
  Real ap_optical_depth_threshold_;
  int initial_profile_mode_;
  bool couple_matter_;
  bool use_ap_transport_;
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
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_THERMAL_RADIATION_HPP_
