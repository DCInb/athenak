//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file two_temperature.cpp
//! \brief Implements the Newtonian ideal-gas ion/electron two-temperature model.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/two_temperature.hpp"

namespace two_temperature {

//----------------------------------------------------------------------------------------
// Constructor.  Heat capacities are normalized so cv_i + cv_e = 1/(gamma - 1).

TwoTemperature::TwoTemperature(const std::string &block, MeshBlockPack *ppack,
                               ParameterInput *pin, int first_component_index) :
    iion(first_component_index),
    iele(first_component_index + 1),
    temperature("two-temperature", 1, 1, 1, 1, 1),
    pmy_pack_(ppack) {
  Real gamma = pin->GetReal(block, "gamma");
  gamma_minus_one_ = gamma - 1.0;
  cv_e_fraction_ = pin->GetOrAddReal(block, "electron_heat_capacity_fraction", 0.5);
  cv_i_fraction_ = 1.0 - cv_e_fraction_;
  Real initial_temperature_ratio =
      pin->GetOrAddReal(block, "initial_electron_temperature_ratio", 1.0);
  t_ei_ = pin->GetOrAddReal(block, "t_ei", -1.0);

  if (cv_e_fraction_ <= 0.0 || cv_e_fraction_ >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block
              << "> electron_heat_capacity_fraction must lie strictly between 0 and 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (initial_temperature_ratio < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block
              << "> initial_electron_temperature_ratio must be non-negative"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // At fixed total internal energy, e_e/e_tot follows from cv_e*Te and cv_i*Ti.
  Real denominator = cv_i_fraction_ + cv_e_fraction_*initial_temperature_ratio;
  initial_e_fraction_ = cv_e_fraction_*initial_temperature_ratio/denominator;

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(temperature, nmb, 2, ncells3, ncells2, ncells1);

  if (pin->DoesBlockExist("thermal_radiation") &&
      pin->GetOrAddBoolean("thermal_radiation", "enabled", true)) {
    pradiation = new ThermalRadiation(ppack, pin, iele + 1, iele,
        gamma_minus_one_, cv_e_fraction_);
  }
}

//----------------------------------------------------------------------------------------

TwoTemperature::~TwoTemperature() {
  if (pradiation != nullptr) delete pradiation;
}

//----------------------------------------------------------------------------------------

int TwoTemperature::NumberOfRadiationGroups() const {
  return (pradiation == nullptr) ? 0 : pradiation->ngroups;
}

//----------------------------------------------------------------------------------------
//! Initialize the redundant component energies from the total internal energy.

void TwoTemperature::Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int iion_ = iion;
  int iele_ = iele;
  Real gm1 = gamma_minus_one_;
  Real fi = cv_i_fraction_;
  Real fe = cv_e_fraction_;
  Real fe0 = initial_e_fraction_;
  auto temp = temperature;

  par_for("two_temp_init", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = prim(m, IDN, k, j, i);
    Real eint = prim(m, IEN, k, j, i);
    Real eele = fe0*eint;
    Real eion = eint - eele;

    cons(m, iion_, k, j, i) = eion;
    cons(m, iele_, k, j, i) = eele;
    prim(m, iion_, k, j, i) = eion/density;
    prim(m, iele_, k, j, i) = eele/density;
    temp(m, 0, k, j, i) = gm1*eion/(density*fi);
    temp(m, 1, k, j, i) = gm1*eele/(density*fe);
  });
  if (pradiation != nullptr) {
    pradiation->Initialize(cons, prim, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------
//! Apply the FLASH/RAGE-like pressure partition to hydrodynamic work and shock heating.

void TwoTemperature::Sync(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                          int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int iion_ = iion;
  int iele_ = iele;
  Real gm1 = gamma_minus_one_;
  Real fi = cv_i_fraction_;
  Real fe = cv_e_fraction_;
  Real fe0 = initial_e_fraction_;
  auto temp = temperature;

  par_for("two_temp_sync", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = prim(m, IDN, k, j, i);
    Real eint = prim(m, IEN, k, j, i);
    Real eion_adv = fmax(cons(m, iion_, k, j, i), 0.0);
    Real eele_adv = fmax(cons(m, iele_, k, j, i), 0.0);
    Real component_sum = eion_adv + eele_adv;

    Real eele;
    if (component_sum > 0.0) {
      // With a common gamma, partial pressure is proportional to energy density.
      eele = eint*(eele_adv/component_sum);
    } else {
      eele = fe0*eint;
    }
    eele = fmin(fmax(eele, 0.0), eint);
    Real eion = eint - eele;  // assign the remainder for round-off-level conservation

    cons(m, iion_, k, j, i) = eion;
    cons(m, iele_, k, j, i) = eele;
    prim(m, iion_, k, j, i) = eion/density;
    prim(m, iele_, k, j, i) = eele/density;
    temp(m, 0, k, j, i) = gm1*eion/(density*fi);
    temp(m, 1, k, j, i) = gm1*eele/(density*fe);
  });
  if (pradiation != nullptr) {
    pradiation->UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------
//! Exact solution of the constant-t_ei FLASH ion/electron heat-exchange equations.

void TwoTemperature::Exchange(Real dt, DvceArray5D<Real> &cons,
                              DvceArray5D<Real> &prim,
                              int il, int iu, int jl, int ju, int kl, int ku) {
  if (t_ei_ >= 0.0) {
    int nmb1 = pmy_pack_->nmb_thispack - 1;
    int iion_ = iion;
    int iele_ = iele;
    Real gm1 = gamma_minus_one_;
    Real fi = cv_i_fraction_;
    Real fe = cv_e_fraction_;
    Real decay = 0.0;
    if (t_ei_ > 0.0) {
      // FLASH: Delta T decays as exp[-(1 + cv_e/cv_i) dt/t_ei].
      decay = exp(-(1.0 + fe/fi)*dt/t_ei_);
    }
    auto temp = temperature;

    par_for("two_temp_exchange", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real density = cons(m, IDN, k, j, i);
      Real eion = cons(m, iion_, k, j, i);
      Real eele = cons(m, iele_, k, j, i);
      Real eint = eion + eele;

      Real tion = gm1*eion/(density*fi);
      Real tele = gm1*eele/(density*fe);
      Real teq = fi*tion + fe*tele;
      Real delta_t = (tion - tele)*decay;
      Real tion_new = teq + fe*delta_t;

      Real eion_new = density*fi*tion_new/gm1;
      eion_new = fmin(fmax(eion_new, 0.0), eint);
      Real eele_new = eint - eion_new;

      cons(m, iion_, k, j, i) = eion_new;
      cons(m, iele_, k, j, i) = eele_new;
      prim(m, iion_, k, j, i) = eion_new/density;
      prim(m, iele_, k, j, i) = eele_new/density;
      temp(m, 0, k, j, i) = gm1*eion_new/(density*fi);
      temp(m, 1, k, j, i) = gm1*eele_new/(density*fe);
    });
  }

  if (pradiation != nullptr) {
    pradiation->Couple(dt, cons, prim, temperature, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------

void TwoTemperature::AddRadiationFluxes(const DvceArray5D<Real> &prim,
                                        DvceFaceFld5D<Real> &flx) {
  if (pradiation != nullptr) pradiation->AddFluxes(prim, flx);
}

//----------------------------------------------------------------------------------------

void TwoTemperature::RadiationNewTimeStep(const DvceArray5D<Real> &prim) {
  if (pradiation != nullptr) pradiation->NewTimeStep(prim);
}

} // namespace two_temperature
