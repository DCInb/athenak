//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation.cpp
//! \brief Explicit multigroup FLD and electron-radiation energy exchange.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "two_temperature/opacity_table.hpp"
#include "two_temperature/thermal_radiation.hpp"

namespace two_temperature {
namespace {

constexpr Real kPlanckIntegralInfinity = 6.4939394022668291491;  // pi^4/15

// Integral_0^x t^3/(exp(t)-1) dt.  The small-x expansion avoids cancellation, while
// the exponentially convergent complementary series is accurate over the rest of the
// range and is suitable for device execution.
KOKKOS_INLINE_FUNCTION
Real PlanckIntegral(Real x) {
  if (x <= 0.0) return 0.0;
  if (x >= 50.0) return kPlanckIntegralInfinity;
  if (x < 0.5) {
    Real x2 = x*x;
    Real x3 = x2*x;
    return x3/3.0 - x3*x/8.0 + x3*x2/60.0
           - x3*x2*x2/5040.0 + x3*x2*x2*x2/272160.0
           - x3*x2*x2*x2*x2/13305600.0;
  }

  Real tail = 0.0;
  for (int n = 1; n <= 64; ++n) {
    Real rn = static_cast<Real>(n);
    Real invn = 1.0/rn;
    Real invn2 = invn*invn;
    Real term = exp(-rn*x)*(x*x*x*invn + 3.0*x*x*invn2
                            + 6.0*x*invn2*invn + 6.0*invn2*invn2);
    tail += term;
  }
  return fmin(fmax(kPlanckIntegralInfinity - tail, 0.0),
              kPlanckIntegralInfinity);
}

KOKKOS_INLINE_FUNCTION
Real PlanckGroupFraction(Real lower_bound, Real upper_bound, Real temperature) {
  if (temperature <= 0.0) return 0.0;
  Real fraction = (PlanckIntegral(upper_bound/temperature)
                   - PlanckIntegral(lower_bound/temperature))
                  /kPlanckIntegralInfinity;
  return fmin(fmax(fraction, 0.0), 1.0);
}

// mode: 0=none, 1=FLASH harmonic, 2=FLASH Larsen, 3=FLASH min/max,
// 4=Levermore-Pomraning.  D has units of length and the physical diffusion coefficient
// multiplying grad(E) is c_hat*D.
KOKKOS_INLINE_FUNCTION
Real FLDCoefficient(Real sigma, Real energy, Real grad, Real alpha,
                    Real energy_floor, int mode) {
  sigma = fmax(sigma, 1.0e-30);
  if (mode == 0) return 1.0/(3.0*sigma);

  Real r = grad/(sigma*fmax(energy, energy_floor));
  Real ra = r/alpha;
  Real lambda;
  if (mode == 1) {
    lambda = 1.0/(3.0 + ra);
  } else if (mode == 2) {
    lambda = 1.0/sqrt(9.0 + ra*ra);
  } else if (mode == 3) {
    lambda = (ra > 0.0) ? fmin(ONE_3RD, 1.0/ra) : ONE_3RD;
  } else {
    lambda = (2.0 + ra)/(6.0 + 3.0*ra + ra*ra);
  }
  return lambda/sigma;
}

} // namespace

//----------------------------------------------------------------------------------------
// Constructor.  Group boundaries are photon energies h*nu/k_B in code-temperature units;
// constant and tabulated models both return mass opacities, so sigma=rho*kappa.

ThermalRadiation::ThermalRadiation(MeshBlockPack *ppack, ParameterInput *pin,
    int first_group_index, int electron_index, Real gamma_minus_one,
    Real electron_heat_capacity_fraction) :
    ngroups(pin->GetInteger("thermal_radiation", "n_groups")),
    ifirst(first_group_index),
    dtnew(FLT_MAX),
    diagnostics("thermal-radiation-diagnostics", 1, 1, 1, 1, 1),
    pmy_pack_(ppack),
    iele_(electron_index),
    gamma_minus_one_(gamma_minus_one),
    cv_e_fraction_(electron_heat_capacity_fraction),
    group_bounds_("thermal-radiation-bounds", 1),
    kappa_transport_("thermal-radiation-kappa-transport", 1),
    kappa_absorption_("thermal-radiation-kappa-absorption", 1),
    kappa_emission_("thermal-radiation-kappa-emission", 1) {
  if (ngroups < 1 || ngroups > 100) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<thermal_radiation>/n_groups must be between 1 and 100"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  arad_ = pin->GetReal("thermal_radiation", "arad");
  chat_ = pin->GetReal("thermal_radiation", "c_light");
  flux_limit_coefficient_ =
      pin->GetOrAddReal("thermal_radiation", "flux_limit_coefficient", 1.0);
  initial_radiation_temperature_ =
      pin->GetOrAddReal("thermal_radiation", "initial_radiation_temperature", 0.0);
  initial_radiation_temperature_right_ = initial_radiation_temperature_;
  initial_radiation_x1_ = 0.0;
  energy_floor_ = pin->GetOrAddReal("thermal_radiation", "energy_floor", 1.0e-30);
  source_cfl_ = pin->GetOrAddReal("thermal_radiation", "source_cfl", 0.1);
  couple_matter_ = pin->GetOrAddBoolean("thermal_radiation", "couple_matter", true);

  std::string initial_profile =
      pin->GetOrAddString("thermal_radiation", "initial_profile", "uniform");
  if (initial_profile == "uniform") {
    initial_profile_mode_ = 0;
  } else if (initial_profile == "step") {
    initial_profile_mode_ = 1;
    initial_radiation_temperature_right_ = pin->GetReal(
        "thermal_radiation", "initial_radiation_temperature_right");
    initial_radiation_x1_ =
        pin->GetOrAddReal("thermal_radiation", "initial_radiation_x1", 0.0);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/initial_profile='"
              << initial_profile << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (arad_ <= 0.0 || chat_ <= 0.0 || flux_limit_coefficient_ <= 0.0 ||
      initial_radiation_temperature_ < 0.0 ||
      initial_radiation_temperature_right_ < 0.0 || energy_floor_ <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Thermal-radiation constants must be positive and the "
              << "initial radiation temperature must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string limiter =
      pin->GetOrAddString("thermal_radiation", "flux_limiter", "levermore-pomraning");
  if (limiter == "none") {
    limiter_mode_ = 0;
  } else if (limiter == "harmonic") {
    limiter_mode_ = 1;
  } else if (limiter == "larsen") {
    limiter_mode_ = 2;
  } else if (limiter == "minmax" || limiter == "min/max") {
    limiter_mode_ = 3;
  } else if (limiter == "levermore-pomraning" || limiter == "levermore") {
    limiter_mode_ = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/flux_limiter='" << limiter
              << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Kokkos::realloc(group_bounds_, ngroups + 1);
  Kokkos::realloc(kappa_transport_, ngroups);
  Kokkos::realloc(kappa_absorption_, ngroups);
  Kokkos::realloc(kappa_emission_, ngroups);

  for (int g = 0; g <= ngroups; ++g) {
    group_bounds_.h_view(g) = pin->GetReal(
        "thermal_radiation", "group_bound_" + std::to_string(g));
    if (group_bounds_.h_view(g) < 0.0 ||
        (g > 0 && group_bounds_.h_view(g) <= group_bounds_.h_view(g-1))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Thermal-radiation group boundaries must be "
                << "non-negative and strictly increasing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::string opacity_model =
      pin->GetOrAddString("thermal_radiation", "opacity_model", "constant");
  if (opacity_model == "constant") {
    for (int g = 0; g < ngroups; ++g) {
      std::string suffix = std::to_string(g);
      kappa_transport_.h_view(g) = pin->GetReal(
          "thermal_radiation", "kappa_transport_" + suffix);
      kappa_absorption_.h_view(g) = pin->GetOrAddReal(
          "thermal_radiation", "kappa_absorption_" + suffix, 0.0);
      kappa_emission_.h_view(g) = pin->GetOrAddReal(
          "thermal_radiation", "kappa_emission_" + suffix,
          kappa_absorption_.h_view(g));
      if (kappa_transport_.h_view(g) <= 0.0 ||
          kappa_absorption_.h_view(g) < 0.0 || kappa_emission_.h_view(g) < 0.0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Transport opacities must be positive and absorption/"
                  << "emission opacities must be non-negative" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  } else if (opacity_model == "table" || opacity_model == "tabulated") {
    use_opacity_table_ = true;
    opacity_table_ = new OpacityTable(pin, ngroups, group_bounds_);
    for (int g = 0; g < ngroups; ++g) {
      kappa_transport_.h_view(g) = 1.0;
      kappa_absorption_.h_view(g) = 0.0;
      kappa_emission_.h_view(g) = 0.0;
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unknown <thermal_radiation>/opacity_model='"
              << opacity_model << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  group_bounds_.modify_host();
  kappa_transport_.modify_host();
  kappa_absorption_.modify_host();
  kappa_emission_.modify_host();
  group_bounds_.sync_device();
  kappa_transport_.sync_device();
  kappa_absorption_.sync_device();
  kappa_emission_.sync_device();

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(diagnostics, nmb, 2, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------

ThermalRadiation::~ThermalRadiation() {
  if (opacity_table_ != nullptr) delete opacity_table_;
}

//----------------------------------------------------------------------------------------
//! Initialize every group from a Planck spectrum at the requested radiation temperature.

void ThermalRadiation::Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                  int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  Real trad_left = initial_radiation_temperature_;
  Real trad_right = initial_radiation_temperature_right_;
  Real xsplit = initial_radiation_x1_;
  int profile = initial_profile_mode_;
  Real arad = arad_;
  auto bounds = group_bounds_.d_view;
  auto diag = diagnostics;
  auto size = pmy_pack_->pmb->mb_size;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is;
  int nx1 = indcs.nx1;

  par_for("thermal_rad_init", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = prim(m, IDN, k, j, i);
    Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real trad = (profile == 1 && x1v >= xsplit) ? trad_right : trad_left;
    Real total = 0.0;
    Real blackbody = arad*trad*trad*trad*trad;
    for (int g = 0; g < ng; ++g) {
      Real eg = blackbody*PlanckGroupFraction(bounds(g), bounds(g+1), trad);
      cons(m, i0+g, k, j, i) = eg;
      prim(m, i0+g, k, j, i) = eg/density;
      total += eg;
    }
    diag(m, 0, k, j, i) = total/density;
    diag(m, 1, k, j, i) = pow(total/arad, 0.25);
  });
}

//----------------------------------------------------------------------------------------
//! Recompute total radiation energy and radiation temperature diagnostics.

void ThermalRadiation::UpdateDiagnostics(const DvceArray5D<Real> &cons,
    const DvceArray5D<Real> &prim, int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  Real arad = arad_;
  auto diag = diagnostics;
  par_for("thermal_rad_diagnostics", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real total = 0.0;
    for (int g = 0; g < ng; ++g) total += fmax(cons(m, i0+g, k, j, i), 0.0);
    diag(m, 0, k, j, i) = total/prim(m, IDN, k, j, i);
    diag(m, 1, k, j, i) = pow(total/arad, 0.25);
  });
}

//----------------------------------------------------------------------------------------
//! Add q_g=-c_hat*D_g*grad(E_g) to each radiation-group finite-volume flux.

void ThermalRadiation::AddFluxes(const DvceArray5D<Real> &w0,
                                 DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto kt = kappa_transport_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  int iele = iele_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real chat = chat_;
  Real alpha = flux_limit_coefficient_;
  Real floor = energy_floor_;
  int mode = limiter_mode_;

  auto flx1 = flx.x1f;
  par_for("thermal_rad_flux1", DevExeSpace(), 0, nmb1, 0, ng-1,
          ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int g, int k, int j, int i) {
    int n = i0 + g;
    Real el = w0(m, IDN, k, j, i-1)*w0(m, n, k, j, i-1);
    Real er = w0(m, IDN, k, j, i)*w0(m, n, k, j, i);
    Real grad1 = (er-el)/size.d_view(m).dx1;
    Real grad2 = 0.0;
    Real grad3 = 0.0;
    if (multi_d) {
      Real ell = w0(m, IDN, k, j-1, i-1)*w0(m, n, k, j-1, i-1);
      Real elu = w0(m, IDN, k, j+1, i-1)*w0(m, n, k, j+1, i-1);
      Real erl = w0(m, IDN, k, j-1, i)*w0(m, n, k, j-1, i);
      Real eru = w0(m, IDN, k, j+1, i)*w0(m, n, k, j+1, i);
      grad2 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx2);
    }
    if (three_d) {
      Real ell = w0(m, IDN, k-1, j, i-1)*w0(m, n, k-1, j, i-1);
      Real elu = w0(m, IDN, k+1, j, i-1)*w0(m, n, k+1, j, i-1);
      Real erl = w0(m, IDN, k-1, j, i)*w0(m, n, k-1, j, i);
      Real eru = w0(m, IDN, k+1, j, i)*w0(m, n, k+1, j, i);
      grad3 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx3);
    }
    Real energy = 0.5*(el+er);
    Real density = 0.5*(w0(m, IDN, k, j, i-1)+w0(m, IDN, k, j, i));
    Real tele = 0.5*gm1*(w0(m, iele, k, j, i-1)+w0(m, iele, k, j, i))/fe;
    Real grad = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
    Real kappa = use_table ? opacity.Get(opacity_transport, g, density, tele) : kt(g);
    Real dcoef = FLDCoefficient(density*kappa, energy, grad, alpha, floor, mode);
    flx1(m, n, k, j, i) -= chat*dcoef*grad1;
  });
  if (pmy_pack_->pmesh->one_d) return;

  auto flx2 = flx.x2f;
  par_for("thermal_rad_flux2", DevExeSpace(), 0, nmb1, 0, ng-1,
          ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int g, int k, int j, int i) {
    int n = i0 + g;
    Real el = w0(m, IDN, k, j-1, i)*w0(m, n, k, j-1, i);
    Real er = w0(m, IDN, k, j, i)*w0(m, n, k, j, i);
    Real grad1;
    Real ell = w0(m, IDN, k, j-1, i-1)*w0(m, n, k, j-1, i-1);
    Real elu = w0(m, IDN, k, j-1, i+1)*w0(m, n, k, j-1, i+1);
    Real erl = w0(m, IDN, k, j, i-1)*w0(m, n, k, j, i-1);
    Real eru = w0(m, IDN, k, j, i+1)*w0(m, n, k, j, i+1);
    grad1 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx1);
    Real grad2 = (er-el)/size.d_view(m).dx2;
    Real grad3 = 0.0;
    if (three_d) {
      ell = w0(m, IDN, k-1, j-1, i)*w0(m, n, k-1, j-1, i);
      elu = w0(m, IDN, k+1, j-1, i)*w0(m, n, k+1, j-1, i);
      erl = w0(m, IDN, k-1, j, i)*w0(m, n, k-1, j, i);
      eru = w0(m, IDN, k+1, j, i)*w0(m, n, k+1, j, i);
      grad3 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx3);
    }
    Real energy = 0.5*(el+er);
    Real density = 0.5*(w0(m, IDN, k, j-1, i)+w0(m, IDN, k, j, i));
    Real tele = 0.5*gm1*(w0(m, iele, k, j-1, i)+w0(m, iele, k, j, i))/fe;
    Real grad = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
    Real kappa = use_table ? opacity.Get(opacity_transport, g, density, tele) : kt(g);
    Real dcoef = FLDCoefficient(density*kappa, energy, grad, alpha, floor, mode);
    flx2(m, n, k, j, i) -= chat*dcoef*grad2;
  });
  if (pmy_pack_->pmesh->two_d) return;

  auto flx3 = flx.x3f;
  par_for("thermal_rad_flux3", DevExeSpace(), 0, nmb1, 0, ng-1,
          ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int g, int k, int j, int i) {
    int n = i0 + g;
    Real el = w0(m, IDN, k-1, j, i)*w0(m, n, k-1, j, i);
    Real er = w0(m, IDN, k, j, i)*w0(m, n, k, j, i);
    Real ell = w0(m, IDN, k-1, j, i-1)*w0(m, n, k-1, j, i-1);
    Real elu = w0(m, IDN, k-1, j, i+1)*w0(m, n, k-1, j, i+1);
    Real erl = w0(m, IDN, k, j, i-1)*w0(m, n, k, j, i-1);
    Real eru = w0(m, IDN, k, j, i+1)*w0(m, n, k, j, i+1);
    Real grad1 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx1);
    ell = w0(m, IDN, k-1, j-1, i)*w0(m, n, k-1, j-1, i);
    elu = w0(m, IDN, k-1, j+1, i)*w0(m, n, k-1, j+1, i);
    erl = w0(m, IDN, k, j-1, i)*w0(m, n, k, j-1, i);
    eru = w0(m, IDN, k, j+1, i)*w0(m, n, k, j+1, i);
    Real grad2 = (elu-ell+eru-erl)/(4.0*size.d_view(m).dx2);
    Real grad3 = (er-el)/size.d_view(m).dx3;
    Real energy = 0.5*(el+er);
    Real density = 0.5*(w0(m, IDN, k-1, j, i)+w0(m, IDN, k, j, i));
    Real tele = 0.5*gm1*(w0(m, iele, k-1, j, i)+w0(m, iele, k, j, i))/fe;
    Real grad = sqrt(grad1*grad1+grad2*grad2+grad3*grad3);
    Real kappa = use_table ? opacity.Get(opacity_transport, g, density, tele) : kt(g);
    Real dcoef = FLDCoefficient(density*kappa, energy, grad, alpha, floor, mode);
    flx3(m, n, k, j, i) -= chat*dcoef*grad3;
  });
}

//----------------------------------------------------------------------------------------
//! Apply FLASH-style time-lagged Planck emission and implicit group absorption.
//!
//! The sum of radiation changes is removed from the electron and material total energies.
//! Positive emission is scaled only when necessary to prevent a negative electron energy.

void ThermalRadiation::Couple(Real dt, DvceArray5D<Real> &cons,
    DvceArray5D<Real> &prim, DvceArray5D<Real> &temperature,
    int il, int iu, int jl, int ju, int kl, int ku) {
  if (!couple_matter_ || dt <= 0.0) {
    UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
    return;
  }

  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int ng = ngroups;
  int i0 = ifirst;
  int ie = iele_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real arad = arad_;
  Real chat = chat_;
  auto bounds = group_bounds_.d_view;
  auto ka = kappa_absorption_.d_view;
  auto ke = kappa_emission_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  auto diag = diagnostics;

  par_for("thermal_rad_couple", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real density = cons(m, IDN, k, j, i);
    Real eele_old = fmax(cons(m, ie, k, j, i), 0.0);
    Real tele = gm1*eele_old/(density*fe);
    Real blackbody = arad*tele*tele*tele*tele;
    Real positive = 0.0;
    Real negative = 0.0;

    for (int g = 0; g < ng; ++g) {
      Real old = fmax(cons(m, i0+g, k, j, i), 0.0);
      Real kappaa = use_table ? opacity.Get(
          opacity_absorption, g, density, tele) : ka(g);
      Real kappae = use_table ? opacity.Get(
          opacity_emission, g, density, tele) : ke(g);
      Real siga = density*kappaa;
      Real sige = density*kappae;
      Real source = sige*blackbody*
          PlanckGroupFraction(bounds(g), bounds(g+1), tele);
      Real updated = (old + dt*chat*source)/(1.0 + dt*chat*siga);
      Real delta = updated-old;
      if (delta > 0.0) positive += delta;
      if (delta < 0.0) negative += delta;
    }

    Real available = eele_old-negative;  // absorbed radiation is immediately available
    Real emission_scale = (positive > available && positive > 0.0)
        ? available/positive : 1.0;
    Real total_delta = 0.0;
    Real total_radiation = 0.0;
    for (int g = 0; g < ng; ++g) {
      Real old = fmax(cons(m, i0+g, k, j, i), 0.0);
      Real kappaa = use_table ? opacity.Get(
          opacity_absorption, g, density, tele) : ka(g);
      Real kappae = use_table ? opacity.Get(
          opacity_emission, g, density, tele) : ke(g);
      Real siga = density*kappaa;
      Real sige = density*kappae;
      Real source = sige*blackbody*
          PlanckGroupFraction(bounds(g), bounds(g+1), tele);
      Real updated = (old + dt*chat*source)/(1.0 + dt*chat*siga);
      Real delta = updated-old;
      if (delta > 0.0) delta *= emission_scale;
      Real value = old+delta;
      cons(m, i0+g, k, j, i) = value;
      prim(m, i0+g, k, j, i) = value/density;
      total_delta += delta;
      total_radiation += value;
    }

    Real eele_new = fmax(eele_old-total_delta, 0.0);
    Real matter_delta = eele_new-eele_old;
    cons(m, ie, k, j, i) = eele_new;
    prim(m, ie, k, j, i) = eele_new/density;
    cons(m, IEN, k, j, i) += matter_delta;
    prim(m, IEN, k, j, i) += matter_delta;
    temperature(m, 1, k, j, i) = gm1*eele_new/(density*fe);
    diag(m, 0, k, j, i) = total_radiation/density;
    diag(m, 1, k, j, i) = pow(total_radiation/arad, 0.25);
  });
}

//----------------------------------------------------------------------------------------
//! Compute the explicit FLD stability limit and an optional source-accuracy limit.

void ThermalRadiation::NewTimeStep(const DvceArray5D<Real> &w0) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int ng = ngroups;
  int i0 = ifirst;
  int ie = iele_;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size;
  auto kt = kappa_transport_.d_view;
  auto ka = kappa_absorption_.d_view;
  auto kem = kappa_emission_.d_view;
  bool use_table = use_opacity_table_;
  OpacityTableDevice opacity;
  if (use_table) opacity = opacity_table_->DeviceData();
  auto bounds = group_bounds_.d_view;
  Real chat = chat_;
  Real floor = energy_floor_;
  Real arad = arad_;
  Real gm1 = gamma_minus_one_;
  Real fe = cv_e_fraction_;
  Real source_cfl = source_cfl_;
  bool couple = couple_matter_;

  int nmb = pmy_pack_->nmb_thispack;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int ncell = nmb*nkji;
  Real minimum = FLT_MAX;
  Kokkos::parallel_reduce("thermal_rad_newdt",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int idx, Real &min_dt) {
    int m = idx/nkji;
    int k = (idx-m*nkji)/nji;
    int j = (idx-m*nkji-k*nji)/nx1;
    int i = idx-m*nkji-k*nji-j*nx1;
    i += is;
    j += js;
    k += ks;
    Real density = w0(m, IDN, k, j, i);
    Real inv_dx2 = 1.0/SQR(size.d_view(m).dx1);
    if (multi_d) inv_dx2 += 1.0/SQR(size.d_view(m).dx2);
    if (three_d) inv_dx2 += 1.0/SQR(size.d_view(m).dx3);
    Real cell_dt = FLT_MAX;
    Real source_rate = 0.0;
    Real tele = gm1*w0(m, ie, k, j, i)/fe;
    Real blackbody = arad*tele*tele*tele*tele;

    for (int g = 0; g < ng; ++g) {
      int n = i0+g;
      Real energy = density*w0(m, n, k, j, i);
      // Every limiter implemented above satisfies D_fl <= 1/(3*sigma_t).  Using the
      // un-limited coefficient here is conservative even when the face gradient differs
      // from the cell-centered gradient used to evaluate the limiter.
      Real kappat = use_table ? opacity.Get(
          opacity_transport, g, density, tele) : kt(g);
      Real dcoef = 1.0/(3.0*density*kappat);
      cell_dt = fmin(cell_dt, 0.5/(chat*dcoef*inv_dx2));

      if (couple && source_cfl > 0.0) {
        Real equilibrium = blackbody*
            PlanckGroupFraction(bounds(g), bounds(g+1), tele);
        Real kappaa = use_table ? opacity.Get(
            opacity_absorption, g, density, tele) : ka(g);
        Real kappae = use_table ? opacity.Get(
            opacity_emission, g, density, tele) : kem(g);
        source_rate += chat*fabs(density*kappae*equilibrium
                                 - density*kappaa*energy);
      }
    }
    if (couple && source_cfl > 0.0 && source_rate > 0.0) {
      Real eele = density*w0(m, ie, k, j, i);
      cell_dt = fmin(cell_dt, source_cfl*fmax(eele, floor)/source_rate);
    }
    min_dt = fmin(min_dt, cell_dt);
  }, Kokkos::Min<Real>(minimum));
  dtnew = minimum;
}

} // namespace two_temperature
