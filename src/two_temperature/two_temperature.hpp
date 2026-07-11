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

class MeshBlockPack;
class ParameterInput;

namespace two_temperature {

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
  TwoTemperature(const std::string &block, MeshBlockPack *ppack, ParameterInput *pin,
                 int first_component_index);
  ~TwoTemperature() = default;

  int iion;  // conserved rho*e_i / primitive e_i index
  int iele;  // conserved rho*e_e / primitive e_e index

  // component 0 = ion temperature, component 1 = electron temperature
  DvceArray5D<Real> temperature;

  // Set initial component energies from the total internal energy and requested Te/Ti.
  void Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  int il, int iu, int jl, int ju, int kl, int ku);

  // Reconcile advected component energies with the conservative total internal energy.
  void Sync(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
            int il, int iu, int jl, int ju, int kl, int ku);

  // Apply exact, energy-conserving ion/electron temperature relaxation over dt.
  void Exchange(Real dt, DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                int il, int iu, int jl, int ju, int kl, int ku);

 private:
  MeshBlockPack *pmy_pack_;
  Real gamma_minus_one_;
  Real cv_i_fraction_;
  Real cv_e_fraction_;
  Real initial_e_fraction_;
  Real t_ei_;
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_TWO_TEMPERATURE_HPP_
