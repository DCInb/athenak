//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "conduction.hpp"
#include "materials/material_mixture.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {

[[noreturn]] void ConductionError(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

// VanLeer Limiter which takes 2 slopes
KOKKOS_INLINE_FUNCTION
Real VLL2State(const Real a, const Real b) {
  if (a*b > 0) {
    return 2.0*a*b/(a+b);
  } else {
    return 0.0;
  }
}

// VanLeer Limiter which takes 4 slopes
KOKKOS_INLINE_FUNCTION
Real VLL4State(const Real a, const Real b, const Real c, const Real d) {
  return VLL2State(VLL2State(a,b), VLL2State(c,d));
}

//----------------------------------------------------------------------------------------
//! \fn Real TempDepKappa()
//! \brief Temperature-dependent conductivity given by Parker (1953) and Spitzer (1962)

KOKKOS_INLINE_FUNCTION
Real TempDepKappa(Real temp, Real limit) {
  if (temp < 6.5e4) {
    return 2.5e3 * pow(temp, 0.5);
  } else {
    return fmin(6.0e-7*pow(temp, 2.5), limit);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.
// Note that the coefficients of thermal conduction, alpha_iso, etc., correspond to
// diffusivities. The conductivity is kappa = (dens)*alpha, and the energy flux
// q = -kappa * (dT/dx) = - alpha * d * *dT/dx)

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin,
                       two_temperature::TwoTemperature *ptwo_temp,
                       materials::MaterialMixture *pmaterials) :
    pmy_pack(pp),
    pmaterials_(pmaterials) {
  // Read parameters for thermal diffusivity (if any)
  alpha_iso = pin->GetOrAddReal(block,"alpha_iso", 0.0);
  alpha_aniso = pin->GetOrAddReal(block,"alpha_aniso", 0.0);
  alpha_spitzer = pin->GetOrAddBoolean(block,"alpha_spitzer", false);
  // Limit on thermal heat flux (saturated conduction)
  q_limit = pin->GetOrAddReal(block,"q_limit",
                     static_cast<Real>(std::numeric_limits<float>::max()));

  const std::string integrator =
      pin->GetOrAddString(block, "conduction_integrator", "explicit");
  if (integrator == "explicit") {
    implicit_ = false;
    return;
  }
  if (integrator != "implicit") {
    ConductionError("<"+block+">/conduction_integrator must be 'explicit' or "
                    "'implicit'");
  }
  implicit_ = true;

  if (ptwo_temp == nullptr) {
    ConductionError("Implicit thermal conduction requires <"+block+
                    ">/two_temperature=true");
  }
  if (pp->pmesh->multilevel) {
    ConductionError("Implicit thermal conduction does not yet support SMR/AMR; "
                    "the matrix operator currently requires a uniform-level mesh");
  }
  if (alpha_aniso != 0.0) {
    ConductionError("Implicit thermal conduction currently supports isotropic "
                    "conductivity only");
  }
  if (!std::isfinite(alpha_iso) || alpha_iso < 0.0) {
    ConductionError("<"+block+">/alpha_iso must be finite and non-negative");
  }
  if (alpha_iso == 0.0 && !alpha_spitzer) {
    ConductionError("Implicit thermal conduction requires alpha_iso > 0 or "
                    "alpha_spitzer=true");
  }
  if (alpha_spitzer && (pmaterials_ == nullptr || pp->punit == nullptr)) {
    ConductionError("Implicit Spitzer conduction requires <materials> and <units>");
  }

  theta_ = pin->GetOrAddReal(block, "conduction_theta", 1.0);
  linear_tolerance_ =
      pin->GetOrAddReal(block, "conduction_linear_tolerance", 1.0e-10);
  nonlinear_tolerance_ =
      pin->GetOrAddReal(block, "conduction_nonlinear_tolerance", 1.0e-8);
  max_iterations_ =
      pin->GetOrAddInteger(block, "conduction_max_iterations", 400);
  max_nonlinear_iterations_ = pin->GetOrAddInteger(
      block, "conduction_max_nonlinear_iterations", 8);
  report_ = pin->GetOrAddBoolean(block, "conduction_report", false);
  coulomb_log_ =
      pin->GetOrAddReal(block, "conduction_coulomb_log", 10.0);
  spitzer_multiplier_ =
      pin->GetOrAddReal(block, "conduction_spitzer_multiplier", 1.0);
  spitzer_temperature_floor_kelvin_ = pin->GetOrAddReal(
      block, "conduction_temperature_floor_kelvin", 1.0);
  flux_limit_coefficient_ =
      pin->GetOrAddReal(block, "conduction_flux_limit_coefficient", 0.06);
  gamma_minus_one_ = pin->GetReal(block, "gamma")-1.0;
  electron_heat_capacity_fraction_ = ptwo_temp->ElectronHeatCapacityFraction();

  if (!std::isfinite(theta_) || theta_ <= 0.0 || theta_ > 1.0) {
    ConductionError("<"+block+">/conduction_theta must be finite and in (0,1]");
  }
  if (!std::isfinite(linear_tolerance_) || linear_tolerance_ <= 0.0 ||
      !std::isfinite(nonlinear_tolerance_) || nonlinear_tolerance_ <= 0.0 ||
      max_iterations_ <= 0 || max_nonlinear_iterations_ <= 0) {
    ConductionError("Implicit-conduction tolerances and iteration limits must be "
                    "positive");
  }
  if (!std::isfinite(coulomb_log_) || coulomb_log_ <= 0.0 ||
      !std::isfinite(spitzer_multiplier_) || spitzer_multiplier_ <= 0.0 ||
      !std::isfinite(spitzer_temperature_floor_kelvin_) ||
      spitzer_temperature_floor_kelvin_ < 0.0 ||
      !std::isfinite(flux_limit_coefficient_) ||
      flux_limit_coefficient_ <= 0.0) {
    ConductionError("Implicit Spitzer and flux-limiter coefficients must be finite "
                    "and positive (the temperature floor may be zero)");
  }

  const std::string limiter = pin->GetOrAddString(
      block, "conduction_flux_limiter", alpha_spitzer ? "harmonic" : "none");
  if (limiter == "none") {
    flux_limiter_ = FluxLimiter::none;
  } else if (limiter == "harmonic") {
    flux_limiter_ = FluxLimiter::harmonic;
  } else if (limiter == "minmax" || limiter == "min/max") {
    flux_limiter_ = FluxLimiter::minmax;
  } else if (limiter == "larsen") {
    flux_limiter_ = FluxLimiter::larsen;
  } else {
    ConductionError("Unknown <"+block+">/conduction_flux_limiter='"+limiter+
                    "'; expected none, harmonic, minmax, or larsen");
  }

  const char *boundary_names[6] = {
      "conduction_x1_inner_boundary", "conduction_x1_outer_boundary",
      "conduction_x2_inner_boundary", "conduction_x2_outer_boundary",
      "conduction_x3_inner_boundary", "conduction_x3_outer_boundary"};
  const char *value_names[6] = {
      "conduction_x1_inner_value", "conduction_x1_outer_value",
      "conduction_x2_inner_value", "conduction_x2_outer_value",
      "conduction_x3_inner_value", "conduction_x3_outer_value"};
  for (int face = 0; face < 6; ++face) {
    const std::string type =
        pin->GetOrAddString(block, boundary_names[face], "neumann");
    if (type == "neumann" || type == "zero-gradient") {
      boundary_type_[face] = BoundaryType::neumann;
    } else if (type == "dirichlet") {
      boundary_type_[face] = BoundaryType::dirichlet;
    } else {
      ConductionError("Unknown <"+block+">/"+boundary_names[face]+"='"+type+
                      "'; expected neumann or dirichlet");
    }
    boundary_value_[face] = pin->GetOrAddReal(block, value_names[face], 0.0);
    if (!std::isfinite(boundary_value_[face])) {
      ConductionError("Implicit-conduction boundary values must be finite");
    }
  }

  const int nmb = std::max(pp->nmb_thispack, pp->pmesh->nmb_maxperrank);
  auto &indcs = pp->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1+2*indcs.ng;
  const int ncells2 = (indcs.nx2 > 1) ? indcs.nx2+2*indcs.ng : 1;
  const int ncells3 = (indcs.nx3 > 1) ? indcs.nx3+2*indcs.ng : 1;
  auto allocate = [&](DvceArray5D<Real> &view) {
    Kokkos::realloc(view, nmb, 1, ncells3, ncells2, ncells1);
  };
  allocate(temperature_old_);
  allocate(temperature_new_);
  allocate(conductivity_);
  allocate(capacity_);
  allocate(energy_old_);
  allocate(explicit_laplacian_);
  allocate(residual_);
  allocate(direction_);
  allocate(preconditioned_);
  allocate(operator_direction_);
  allocate(correction_);
  allocate(coarse_scratch_);
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxes()
//! \brief Wrapper function that adds heat fluxes for different types of thermal
//! conduction to face-centered fluxes of conserved variables

void Conduction::AddHeatFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (alpha_iso != 0) {
    AddHeatFluxIso(w0, eos, flx);
  }
  if (alpha_aniso != 0) {
    AddHeatFluxAniso(w0, eos, flx);
  }
  if (alpha_spitzer) {
    AddHeatFluxSpitzer(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxIso()
//! \brief Adds isotropic heat flux computed using constant conductivity to face-centered
//! fluxes of conserved variables

void Conduction::AddHeatFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  Real gm1 = eos.gamma-1.0;
  Real &alpha_ = alpha_iso;

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j,i  )/w0(m,IDN,k,j,i  );
    Real templ = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx1;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
    flx1(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j  ,i)/w0(m,IDN,k,j  ,i);
    Real templ = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx2;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
    flx2(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k  ,j,i)/w0(m,IDN,k  ,j,i);
    Real templ = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx3;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
    flx3(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxAniso()
//! \brief Current a no-op function, to be added later

void Conduction::AddHeatFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TempDependentHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables with
//! temperature-dependent conductivity

void Conduction::AddHeatFluxSpitzer(const DvceArray5D<Real> &w0, const EOS_Data &eos,
   DvceFaceFld5D<Real> &flx) {
/*
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  const bool &sat_hflux_ = sat_hflux;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real gm1 = eos.gamma-1.0;
  Real kappaceil = kappa_ceiling;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                    pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
    Real temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real pres_l = w0(m,IEN,k,j,i-1)*gm1;
    Real pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx1 = (temp_r-temp_l)/size.d_view(m).dx1;
    Real hflx = kappaf*dtempdx1;
    // Saturation of thermal conduction by harmonic mean
    if (sat_hflux_) {
      Real dtempdx2 = 0.0, dtempdx3 = 0.0;
      if (multi_d) {
        temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
        temp_lr = w0(m,IEN,k,j+1,i-1)/w0(m,IDN,k,j+1,i-1)*gm1;
        temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
        temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
        dtempdx2 = VanLeerLimiter4State(temp_rr-temp_r,temp_r-temp_rl,
                                        temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      }
      if (three_d) {
        temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
        temp_lr = w0(m,IEN,k+1,j,i-1)/w0(m,IDN,k+1,j,i-1)*gm1;
        temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
        temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
        dtempdx3 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                              temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx3;
      }
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx1(m,IEN,k,j,i) -= hflx;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    temp_l = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
    temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    pres_l = w0(m,IEN,k,j-1,i)*gm1;
    pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx2 = (temp_r-temp_l)/size.d_view(m).dx2;
    Real hflx = kappaf*dtempdx2;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx3 = 0.0;
      temp_ll = w0(m,IEN,k,j-1,i-1)/w0(m,IDN,k,j-1,i-1)*gm1;
      temp_lr = w0(m,IEN,k,j-1,i+1)/w0(m,IDN,k,j-1,i+1)*gm1;
      temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
      temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      if (three_d) {
        temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
        temp_lr = w0(m,IEN,k+1,j-1,i)/w0(m,IDN,k+1,j-1,i)*gm1;
        temp_rl = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
        temp_rr = w0(m,IEN,k+1,j,i)/w0(m,IDN,k+1,j,i)*gm1;
        dtempdx3 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                              temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx3;
      }
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx2(m,IEN,k,j,i) -= hflx;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Add heat fluxes into fluxes of conserved variables: energy
    Real temp_l = 0.0, temp_r = 0.0, pres_l = 0.0, pres_r = 0.0;
    Real temp_ll = 0.0, temp_lr = 0.0, temp_rl = 0.0, temp_rr = 0.0;
    temp_l = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i)*gm1;
    temp_r = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    pres_l = w0(m,IEN,k-1,j,i)*gm1;
    pres_r = w0(m,IEN,k,j,i)*gm1;
    Real kappaf = 0.5*(TempDepKappa(temp_unit*temp_l,kappaceil)+
                  TempDepKappa(temp_unit*temp_r,kappaceil))/kappa_unit;
    Real dtempdx3 = (temp_r-temp_l)/size.d_view(m).dx3;
    Real hflx = kappaf*dtempdx3;
    // Saturation of thermal conduction
    if (sat_hflux_) {
      Real dtempdx1 = 0.0, dtempdx2 = 0.0;
      temp_ll = w0(m,IEN,k-1,j,i-1)/w0(m,IDN,k-1,j,i-1)*gm1;
      temp_lr = w0(m,IEN,k-1,j,i+1)/w0(m,IDN,k-1,j,i+1)*gm1;
      temp_rl = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1)*gm1;
      temp_rr = w0(m,IEN,k,j,i+1)/w0(m,IDN,k,j,i+1)*gm1;
      dtempdx1 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx1;
      temp_ll = w0(m,IEN,k-1,j-1,i)/w0(m,IDN,k-1,j-1,i)*gm1;
      temp_lr = w0(m,IEN,k-1,j+1,i)/w0(m,IDN,k-1,j+1,i)*gm1;
      temp_rl = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i)*gm1;
      temp_rr = w0(m,IEN,k,j+1,i)/w0(m,IDN,k,j+1,i)*gm1;
      dtempdx2 = VL4Limiter(temp_rr-temp_r,temp_r-temp_rl,
                            temp_lr-temp_l,temp_l-temp_ll)/size.d_view(m).dx2;
      Real tempgrad = sqrt(SQR(dtempdx1)+SQR(dtempdx2)+SQR(dtempdx3));
      Real pres_cs = 0.5*(pres_l*sqrt(temp_l)+pres_r*sqrt(temp_r));
      Real sat_fac = 1.0/(1.0+kappaf*tempgrad/(1.5*pres_cs));
      hflx *= sat_fac;
    }
    flx3(m,IEN,k,j,i) -= hflx;
  });

*/
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
//  if (sat_hflux == true) {
//    dtnew = static_cast<Real>(std::numeric_limits<float>::max());
//    return;
//  }

  // set flag for Spitzer conductivity
  bool spitzer_ = alpha_spitzer;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real kappa_unit = pmy_pack->punit->pressure_cgs()*pmy_pack->punit->velocity_cgs()*
                      pmy_pack->punit->length_cgs()/pmy_pack->punit->temperature_cgs();

  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &w0_ = w0;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  Real gm1 = eos_data.gamma-1.0;
  Real alpha0 = alpha_iso;

  // find smallest timestep for thermal conduction in each cell
  // Note loop over all cells needed even for constant conductivity
  Kokkos::parallel_reduce("cond_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real alpha_ = alpha0;
//    if (spitzer_) {
//      Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
//      kappa_ = TempDepKappa(temp*temp_unit, limit_)/kappa_unit;
//    }

    min_dt = fmin(min_dt, SQR(size.d_view(m).dx1)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    if (multi_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx2)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    }
    if (three_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx3)/alpha_*w0_(m,IDN,k,j,i)/gm1);
    }
  }, Kokkos::Min<Real>(dtnew));
  dtnew *= fac;

  return;
}
