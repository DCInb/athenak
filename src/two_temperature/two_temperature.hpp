#ifndef TWO_TEMPERATURE_TWO_TEMPERATURE_HPP_
#define TWO_TEMPERATURE_TWO_TEMPERATURE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file two_temperature.hpp
//! \brief Two-temperature ion/electron internal-energy model for Newtonian ideal gases.

#include <string>

#include "athena.hpp"
#include "materials/material_mixture.hpp"

class MeshBlockPack;
class ParameterInput;

namespace two_temperature {

class ThermalRadiation;

//----------------------------------------------------------------------------------------
//! \class TwoTemperature
//! \brief Evolves redundant ion and electron internal energies alongside total energy.
//!
//! The two component energies are stored after the user passive scalars.  Their fluxes
//! are therefore the usual mass flux multiplied by component specific internal energy.
//! After every fluid update, Sync() assigns the difference between total internal energy
//! and the two advected component energies in proportion to their partial pressures.  For
//! the common-gamma ideal gas implemented here this is the RAGE-like pressure partition
//! described in the FLASH multitemperature hydrodynamics documentation.

class TwoTemperature {
 public:
  enum ThermodynamicField : int {
    ion_pressure = 0,
    electron_pressure = 1,
    electron_number_density_cgs = 2,
    mean_ionization = 3,
    sound_speed_squared = 4,
    effective_charge = 5,
    // Bitwise OR of every IONMIX query flag seen by this cell in this process segment.
    // The thermodynamic cache is reconstructed on restart, so lifetime gating must OR
    // eos_flags from all pre- and post-restart output segments.
    eos_query_flags = 6
  };
  static constexpr int nthermodynamic_fields = 7;
  TwoTemperature(const std::string &block, MeshBlockPack *ppack, ParameterInput *pin,
                 int first_component_index,
                 materials::MaterialMixture *material_mixture = nullptr);
  ~TwoTemperature();

  int iion;  // conserved rho*e_i / primitive e_i index
  int iele;  // conserved rho*e_e / primitive e_e index

  // component 0 = ion temperature, component 1 = electron temperature
  DvceArray5D<Real> temperature;
  DvceArray5D<Real> thermodynamics;

  // Optional FLASH-like thermal multigroup radiation model.
  ThermalRadiation *pradiation = nullptr;

  int NumberOfRadiationGroups() const;
  Real InitialElectronEnergyFraction() const { return initial_e_fraction_; }
  Real InitialElectronTemperatureRatio() const { return initial_temperature_ratio_; }
  Real ElectronHeatCapacityFraction() const { return cv_e_fraction_; }

  // Set initial component energies from the total internal energy and requested Te/Ti.
  void Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  int il, int iu, int jl, int ju, int kl, int ku);

  // Reconcile advected component energies with the conservative total internal energy.
  void Sync(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
            int il, int iu, int jl, int ju, int kl, int ku);

  // Close a Biermann RK stage while preserving its independently evolved electron
  // energy: ion energy is total internal minus electron energy.  Intermediate tabular
  // stages may refresh only the electron state consumed by the Biermann operator; a
  // half-step endpoint always requests the complete canonical thermodynamic cache.
  void CloseBiermannStage(
      DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
      int il, int iu, int jl, int ju, int kl, int ku,
      bool full_thermodynamics = true);

  // Apply exact, energy-conserving ion/electron temperature relaxation over dt.
  void Exchange(Real dt, DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                int il, int iu, int jl, int ju, int kl, int ku);

  // Add multigroup FLD fluxes and compute their explicit stability limit.
  void AddRadiationFluxes(const DvceArray5D<Real> &prim, DvceFaceFld5D<Real> &flx);
  void RadiationNewTimeStep(const DvceArray5D<Real> &prim);

  // Public because nvcc requires a member enclosing an extended device lambda to have
  // public access. This is an internal cache operation despite its access level.
  void RefreshMaterialThermodynamics(
      const DvceArray5D<Real> &cons, int il, int iu, int jl, int ju,
      int kl, int ku);

 private:
  MeshBlockPack *pmy_pack_;
  Real gamma_minus_one_;
  Real cv_i_fraction_;
  Real cv_e_fraction_;
  Real initial_e_fraction_;
  Real initial_temperature_ratio_;
  Real t_ei_;
  bool use_material_mixture_;
  materials::MaterialMixtureDevice material_mixture_;
  bool use_spitzer_exchange_ = false;
  Real spitzer_coulomb_log_ = 10.0;
  Real spitzer_multiplier_ = 1.0;
  Real spitzer_temperature_floor_ = 1.0;
  Real density_scale_cgs_ = 1.0;
  Real velocity_squared_cgs_ = 1.0;
  Real time_scale_cgs_ = 1.0;
  Real density_floor_ = 0.0;
  Real pressure_floor_ = 0.0;
  Real temperature_floor_ = 0.0;
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_TWO_TEMPERATURE_HPP_
