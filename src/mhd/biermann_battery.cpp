//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file biermann_battery.cpp
//! \brief FLASH-style flux formulation of the Biermann battery for 2T MHD.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/biermann_battery.hpp"
#include "parameter_input.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real ElectronPressure(const DvceArray5D<Real> &w, const int iele,
                      const Real gm1, const int m, const int k, const int j,
                      const int i) {
  return gm1 * fmax(w(m, IDN, k, j, i) * w(m, iele, k, j, i), 0.0);
}

} // namespace

namespace mhd {

//----------------------------------------------------------------------------------------
// Constructor.  AthenaK uses mu0=k_B=1 normalized units here.  The coefficient
// is the normalized inverse electron charge in E_B=-C_B grad(p_e)/n_e and
// v_e-v=-C_B curl(B)/n_e.  The electron number density is n_e=f_e*rho, where
// f_e is the electron heat-capacity fraction of the common-gamma 2T model.

BiermannBattery::BiermannBattery(MeshBlockPack *ppack, ParameterInput *pin,
                                 int electron_index,
                                 Real electron_heat_capacity_fraction,
                                 Real gamma, Real density_floor,
                                 Real pressure_floor)
    : coefficient(pin->GetOrAddReal("mhd", "biermann_coefficient", 1.0)),
      dtnew(std::numeric_limits<float>::max()),
      suppress_in_shocks(
          pin->GetOrAddBoolean("mhd", "biermann_shock_suppression", true)),
      shock_threshold(
          pin->GetOrAddReal("mhd", "biermann_shock_threshold", 0.25)),
      pmy_pack_(ppack), iele_(electron_index),
      electron_fraction_(electron_heat_capacity_fraction), gamma_(gamma),
      gamma_minus_one_(gamma - 1.0), density_floor_(density_floor),
      pressure_floor_(pressure_floor),
      smooth_cell_("biermann-smooth", 1, 1, 1, 1),
      e3x1_("biermann-e3x1", 1, 1, 1, 1), e2x1_("biermann-e2x1", 1, 1, 1, 1),
      e1x2_("biermann-e1x2", 1, 1, 1, 1), e3x2_("biermann-e3x2", 1, 1, 1, 1),
      e2x3_("biermann-e2x3", 1, 1, 1, 1), e1x3_("biermann-e1x3", 1, 1, 1, 1),
      vd1_("biermann-vd1", 1, 1, 1, 1), vd2_("biermann-vd2", 1, 1, 1, 1),
      vd3_("biermann-vd3", 1, 1, 1, 1) {
  if (!std::isfinite(coefficient) || coefficient < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<mhd>/biermann_coefficient must be finite and "
              << "non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(shock_threshold) || shock_threshold <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<mhd>/biermann_shock_threshold must be finite and "
              << "positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng : 1;
  Kokkos::realloc(smooth_cell_, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(e3x1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(e2x1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(vd1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(e1x2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(e3x2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(vd2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(e2x3_, nmb, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(e1x3_, nmb, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(vd3_, nmb, ncells3 + 1, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
//! \brief Build a symmetric FLASH-like shock indicator.  Raw grad(p_e)/n_e is
//! not a convergent Biermann discretization inside a discontinuity, so all
//! faces adjacent to a cell with a large centered gas-pressure jump are
//! disabled when requested.

void BiermannBattery::ComputeShockMask(const DvceArray5D<Real> &prim) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int il = is - 1, iu = ie + 1;
  int jl = multi_d ? js - 1 : js;
  int ju = multi_d ? je + 1 : je;
  int kl = three_d ? ks - 1 : ks;
  int ku = three_d ? ke + 1 : ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto smooth = smooth_cell_;

  if (!suppress_in_shocks) {
    par_for(
        "biermann_smooth_all", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          smooth(m, k, j, i) = 1.0;
        });
    return;
  }

  Real gm1 = gamma_minus_one_;
  Real pfloor = fmax(pressure_floor_, 1.0e-30);
  Real threshold = shock_threshold;
  auto w = prim;
  par_for(
      "biermann_shock_mask", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real mask = 1.0;
        Real pm = fmax(gm1 * w(m, IEN, k, j, i - 1), pfloor);
        Real pp = fmax(gm1 * w(m, IEN, k, j, i + 1), pfloor);
        Real jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
        if (jump > threshold)
          mask = 0.0;
        if (multi_d) {
          pm = fmax(gm1 * w(m, IEN, k, j - 1, i), pfloor);
          pp = fmax(gm1 * w(m, IEN, k, j + 1, i), pfloor);
          jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
          if (jump > threshold)
            mask = 0.0;
        }
        if (three_d) {
          pm = fmax(gm1 * w(m, IEN, k - 1, j, i), pfloor);
          pp = fmax(gm1 * w(m, IEN, k + 1, j, i), pfloor);
          jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
          if (jump > threshold)
            mask = 0.0;
        }
        smooth(m, k, j, i) = mask;
      });
}

//----------------------------------------------------------------------------------------
//! \brief Compute Cartesian face fluxes.  Magnetic fluxes are represented by
//! the transverse face electric fields and are later passed through flux-CT in
//! AddEMFs().

void BiermannBattery::AddFluxes(const DvceArray5D<Real> &prim,
                                const DvceArray5D<Real> &bcc,
                                DvceFaceFld5D<Real> &flx) {
  ComputeShockMask(prim);

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto size = pmy_pack_->pmb->mb_size;
  auto w = prim;
  auto smooth = smooth_cell_;
  auto e21 = e2x1_;
  auto e31 = e3x1_;
  auto e12 = e1x2_;
  auto e32 = e3x2_;
  auto e13 = e1x3_;
  auto e23 = e2x3_;
  Real coeff = coefficient;
  Real gm1 = gamma_minus_one_;
  Real fe = electron_fraction_;
  Real dfloor = density_floor_;
  int iele = iele_;

  // x1-face electric fields, including the transverse halo needed by flux-CT.
  int jl = multi_d ? js - 1 : js;
  int ju = multi_d ? je + 1 : je;
  int kl = three_d ? ks - 1 : ks;
  int ku = three_d ? ke + 1 : ke;
  par_for(
      "biermann_face_e1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is, ie + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real dp2 = 0.0;
        Real dp3 = 0.0;
        if (multi_d) {
          dp2 = 0.25 *
                (ElectronPressure(w, iele, gm1, m, k, j + 1, i - 1) +
                 ElectronPressure(w, iele, gm1, m, k, j + 1, i) -
                 ElectronPressure(w, iele, gm1, m, k, j - 1, i - 1) -
                 ElectronPressure(w, iele, gm1, m, k, j - 1, i)) /
                size.d_view(m).dx2;
        }
        if (three_d) {
          dp3 = 0.25 *
                (ElectronPressure(w, iele, gm1, m, k + 1, j, i - 1) +
                 ElectronPressure(w, iele, gm1, m, k + 1, j, i) -
                 ElectronPressure(w, iele, gm1, m, k - 1, j, i - 1) -
                 ElectronPressure(w, iele, gm1, m, k - 1, j, i)) /
                size.d_view(m).dx3;
        }
        Real rho =
            fmax(0.5 * (w(m, IDN, k, j, i - 1) + w(m, IDN, k, j, i)), dfloor);
        Real mask = fmin(smooth(m, k, j, i - 1), smooth(m, k, j, i));
        Real inv_ne = 1.0 / (fe * rho);
        e21(m, k, j, i) = -coeff * mask * dp2 * inv_ne;
        e31(m, k, j, i) = -coeff * mask * dp3 * inv_ne;
      });

  // x2-face electric fields.
  if (multi_d) {
    kl = three_d ? ks - 1 : ks;
    ku = three_d ? ke + 1 : ke;
    par_for(
        "biermann_face_e2", DevExeSpace(), 0, nmb1, kl, ku, js, je + 1, is - 1,
        ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real dp1 = 0.25 *
                     (ElectronPressure(w, iele, gm1, m, k, j - 1, i + 1) +
                      ElectronPressure(w, iele, gm1, m, k, j, i + 1) -
                      ElectronPressure(w, iele, gm1, m, k, j - 1, i - 1) -
                      ElectronPressure(w, iele, gm1, m, k, j, i - 1)) /
                     size.d_view(m).dx1;
          Real dp3 = 0.0;
          if (three_d) {
            dp3 = 0.25 *
                  (ElectronPressure(w, iele, gm1, m, k + 1, j - 1, i) +
                   ElectronPressure(w, iele, gm1, m, k + 1, j, i) -
                   ElectronPressure(w, iele, gm1, m, k - 1, j - 1, i) -
                   ElectronPressure(w, iele, gm1, m, k - 1, j, i)) /
                  size.d_view(m).dx3;
          }
          Real rho =
              fmax(0.5 * (w(m, IDN, k, j - 1, i) + w(m, IDN, k, j, i)), dfloor);
          Real mask = fmin(smooth(m, k, j - 1, i), smooth(m, k, j, i));
          Real inv_ne = 1.0 / (fe * rho);
          e12(m, k, j, i) = -coeff * mask * dp1 * inv_ne;
          e32(m, k, j, i) = -coeff * mask * dp3 * inv_ne;
        });
  }

  // x3-face electric fields.
  if (three_d) {
    par_for(
        "biermann_face_e3", DevExeSpace(), 0, nmb1, ks, ke + 1, js - 1, je + 1,
        is - 1, ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real dp1 = 0.25 *
                     (ElectronPressure(w, iele, gm1, m, k - 1, j, i + 1) +
                      ElectronPressure(w, iele, gm1, m, k, j, i + 1) -
                      ElectronPressure(w, iele, gm1, m, k - 1, j, i - 1) -
                      ElectronPressure(w, iele, gm1, m, k, j, i - 1)) /
                     size.d_view(m).dx1;
          Real dp2 = 0.25 *
                     (ElectronPressure(w, iele, gm1, m, k - 1, j + 1, i) +
                      ElectronPressure(w, iele, gm1, m, k, j + 1, i) -
                      ElectronPressure(w, iele, gm1, m, k - 1, j - 1, i) -
                      ElectronPressure(w, iele, gm1, m, k, j - 1, i)) /
                     size.d_view(m).dx2;
          Real rho =
              fmax(0.5 * (w(m, IDN, k - 1, j, i) + w(m, IDN, k, j, i)), dfloor);
          Real mask = fmin(smooth(m, k - 1, j, i), smooth(m, k, j, i));
          Real inv_ne = 1.0 / (fe * rho);
          e13(m, k, j, i) = -coeff * mask * dp1 * inv_ne;
          e23(m, k, j, i) = -coeff * mask * dp2 * inv_ne;
        });
  }

  // Add Poynting and electron enthalpy fluxes on active x1 faces.  The
  // component electron equation carries epsilon_e*v_d; p_e*div(v_d) is applied
  // after the update.
  auto b = bcc;
  auto flx1 = flx.x1f;
  auto vd1 = vd1_;
  Real gamma = gamma_;
  par_for(
      "biermann_energy_flux1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rho_l = fmax(w(m, IDN, k, j, i - 1), dfloor);
        Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
        Real rho = fmax(0.5 * (rho_l + rho_r), dfloor);
        Real mask = fmin(smooth(m, k, j, i - 1), smooth(m, k, j, i));
        Real j1 = 0.0;
        if (multi_d) {
          j1 += 0.25 *
                (b(m, IBZ, k, j + 1, i - 1) + b(m, IBZ, k, j + 1, i) -
                 b(m, IBZ, k, j - 1, i - 1) - b(m, IBZ, k, j - 1, i)) /
                size.d_view(m).dx2;
        }
        if (three_d) {
          j1 -= 0.25 *
                (b(m, IBY, k + 1, j, i - 1) + b(m, IBY, k + 1, j, i) -
                 b(m, IBY, k - 1, j, i - 1) - b(m, IBY, k - 1, j, i)) /
                size.d_view(m).dx3;
        }
        Real drift = -coeff * mask * j1 / (fe * rho);
        vd1(m, k, j, i) = drift;
        Real eps_l = fmax(rho_l * w(m, iele, k, j, i - 1), 0.0);
        Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
        Real eps = (drift >= 0.0) ? eps_l : eps_r;
        flx1(m, iele, k, j, i) += eps * drift;
        Real by = 0.5 * (b(m, IBY, k, j, i - 1) + b(m, IBY, k, j, i));
        Real bz = 0.5 * (b(m, IBZ, k, j, i - 1) + b(m, IBZ, k, j, i));
        flx1(m, IEN, k, j, i) += e21(m, k, j, i) * bz - e31(m, k, j, i) * by;
        flx1(m, IEN, k, j, i) += gamma * eps * drift;
      });

  // Active x2 faces.
  if (multi_d) {
    auto flx2 = flx.x2f;
    auto vd2 = vd2_;
    par_for(
        "biermann_energy_flux2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is,
        ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real rho_l = fmax(w(m, IDN, k, j - 1, i), dfloor);
          Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
          Real rho = fmax(0.5 * (rho_l + rho_r), dfloor);
          Real mask = fmin(smooth(m, k, j - 1, i), smooth(m, k, j, i));
          Real j2 = -0.25 *
                    (b(m, IBZ, k, j - 1, i + 1) + b(m, IBZ, k, j, i + 1) -
                     b(m, IBZ, k, j - 1, i - 1) - b(m, IBZ, k, j, i - 1)) /
                    size.d_view(m).dx1;
          if (three_d) {
            j2 += 0.25 *
                  (b(m, IBX, k + 1, j - 1, i) + b(m, IBX, k + 1, j, i) -
                   b(m, IBX, k - 1, j - 1, i) - b(m, IBX, k - 1, j, i)) /
                  size.d_view(m).dx3;
          }
          Real drift = -coeff * mask * j2 / (fe * rho);
          vd2(m, k, j, i) = drift;
          Real eps_l = fmax(rho_l * w(m, iele, k, j - 1, i), 0.0);
          Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
          Real eps = (drift >= 0.0) ? eps_l : eps_r;
          flx2(m, iele, k, j, i) += eps * drift;
          Real bx = 0.5 * (b(m, IBX, k, j - 1, i) + b(m, IBX, k, j, i));
          Real bz = 0.5 * (b(m, IBZ, k, j - 1, i) + b(m, IBZ, k, j, i));
          flx2(m, IEN, k, j, i) += e32(m, k, j, i) * bx - e12(m, k, j, i) * bz;
          flx2(m, IEN, k, j, i) += gamma * eps * drift;
        });
  }

  // Active x3 faces.
  if (three_d) {
    auto flx3 = flx.x3f;
    auto vd3 = vd3_;
    par_for(
        "biermann_energy_flux3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je, is,
        ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real rho_l = fmax(w(m, IDN, k - 1, j, i), dfloor);
          Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
          Real rho = fmax(0.5 * (rho_l + rho_r), dfloor);
          Real mask = fmin(smooth(m, k - 1, j, i), smooth(m, k, j, i));
          Real j3 = 0.25 *
                    (b(m, IBY, k - 1, j, i + 1) + b(m, IBY, k, j, i + 1) -
                     b(m, IBY, k - 1, j, i - 1) - b(m, IBY, k, j, i - 1)) /
                    size.d_view(m).dx1;
          j3 -= 0.25 *
                (b(m, IBX, k - 1, j + 1, i) + b(m, IBX, k, j + 1, i) -
                 b(m, IBX, k - 1, j - 1, i) - b(m, IBX, k, j - 1, i)) /
                size.d_view(m).dx2;
          Real drift = -coeff * mask * j3 / (fe * rho);
          vd3(m, k, j, i) = drift;
          Real eps_l = fmax(rho_l * w(m, iele, k - 1, j, i), 0.0);
          Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
          Real eps = (drift >= 0.0) ? eps_l : eps_r;
          flx3(m, iele, k, j, i) += eps * drift;
          Real bx = 0.5 * (b(m, IBX, k - 1, j, i) + b(m, IBX, k, j, i));
          Real by = 0.5 * (b(m, IBY, k - 1, j, i) + b(m, IBY, k, j, i));
          flx3(m, IEN, k, j, i) += e13(m, k, j, i) * by - e23(m, k, j, i) * bx;
          flx3(m, IEN, k, j, i) += gamma * eps * drift;
        });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Arithmetic flux-CT construction, matching the face-flux form
//! documented for FLASH's staggered-mesh Biermann solver.  Communication later
//! reconciles these edge fields across blocks and refinement levels exactly
//! like the ideal-MHD EMF.

void BiermannBattery::AddEMFs(DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e21 = e2x1_;
  auto e31 = e3x1_;
  auto e12 = e1x2_;
  auto e32 = e3x2_;
  auto e13 = e1x3_;
  auto e23 = e2x3_;

  if (pmy_pack_->pmesh->two_d) {
    par_for(
        "biermann_fluxct2", DevExeSpace(), 0, nmb1, js, je + 1, is, ie + 1,
        KOKKOS_LAMBDA(const int m, const int j, const int i) {
          e1(m, ks, j, i) += e12(m, ks, j, i);
          e1(m, ke + 1, j, i) += e12(m, ks, j, i);
          e2(m, ks, j, i) += e21(m, ks, j, i);
          e2(m, ke + 1, j, i) += e21(m, ks, j, i);
          e3(m, ks, j, i) += 0.25 * (e31(m, ks, j - 1, i) + e31(m, ks, j, i) +
                                     e32(m, ks, j, i - 1) + e32(m, ks, j, i));
        });
    return;
  }

  if (pmy_pack_->pmesh->three_d) {
    par_for(
        "biermann_fluxct3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je + 1, is,
        ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          e1(m, k, j, i) += 0.25 * (e12(m, k - 1, j, i) + e12(m, k, j, i) +
                                    e13(m, k, j - 1, i) + e13(m, k, j, i));
          e2(m, k, j, i) += 0.25 * (e23(m, k, j, i - 1) + e23(m, k, j, i) +
                                    e21(m, k - 1, j, i) + e21(m, k, j, i));
          e3(m, k, j, i) += 0.25 * (e32(m, k, j, i - 1) + e32(m, k, j, i) +
                                    e31(m, k, j - 1, i) + e31(m, k, j, i));
        });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Apply the p_e div(v_e-v) term omitted by conservative advection of
//! epsilon_e.

void BiermannBattery::ApplyElectronWork(Real dt, DvceArray5D<Real> &cons,
                                        DvceArray5D<Real> &prim) {
  if (coefficient == 0.0)
    return;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto size = pmy_pack_->pmb->mb_size;
  auto u = cons;
  auto w = prim;
  auto vd1 = vd1_;
  auto vd2 = vd2_;
  auto vd3 = vd3_;
  int iele = iele_;
  Real gm1 = gamma_minus_one_;
  Real dfloor = density_floor_;

  par_for(
      "biermann_electron_work", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real divvd =
            (vd1(m, k, j, i + 1) - vd1(m, k, j, i)) / size.d_view(m).dx1;
        if (multi_d) {
          divvd += (vd2(m, k, j + 1, i) - vd2(m, k, j, i)) / size.d_view(m).dx2;
        }
        if (three_d) {
          divvd += (vd3(m, k + 1, j, i) - vd3(m, k, j, i)) / size.d_view(m).dx3;
        }
        Real eele = fmax(u(m, iele, k, j, i), 0.0) * exp(-gm1 * divvd * dt);
        Real density = fmax(u(m, IDN, k, j, i), dfloor);
        u(m, iele, k, j, i) = eele;
        w(m, iele, k, j, i) = eele / density;
      });
}

//----------------------------------------------------------------------------------------
//! \brief Add the electron drift and thermal-magnetic characteristic speeds.
//! The latter follows FLASH's v_TM^2=(gamma-1) T_e |grad ln(n_e)|^2/(e^2 n_e)
//! in normalized mu0=k_B=1 units.

void BiermannBattery::NewTimeStep(const DvceArray5D<Real> &prim,
                                  const DvceArray5D<Real> &bcc) {
  dtnew = std::numeric_limits<float>::max();
  if (coefficient == 0.0)
    return;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int nmkji = pmy_pack_->nmb_thispack * nx3 * nx2 * nx1;
  int nkji = nx3 * nx2 * nx1;
  int nji = nx2 * nx1;
  auto size = pmy_pack_->pmb->mb_size;
  auto w = prim;
  auto b = bcc;
  Real coeff = coefficient;
  Real gm1 = gamma_minus_one_;
  Real fe = electron_fraction_;
  Real dfloor = density_floor_;
  int iele = iele_;
  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  Kokkos::parallel_reduce(
      "biermann_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2,
                    Real &min_dt3) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = idx - m * nkji - k * nji - j * nx1 + is;
        k += ks;
        j += js;

        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real rho = fmax(w(m, IDN, k, j, i), dfloor);
        Real ne = fe * rho;
        Real dln1 = (log(fmax(w(m, IDN, k, j, i + 1), dfloor)) -
                     log(fmax(w(m, IDN, k, j, i - 1), dfloor))) /
                    (2.0 * dx1);
        Real dln2 = 0.0;
        Real dln3 = 0.0;
        if (multi_d) {
          dln2 = (log(fmax(w(m, IDN, k, j + 1, i), dfloor)) -
                  log(fmax(w(m, IDN, k, j - 1, i), dfloor))) /
                 (2.0 * dx2);
        }
        if (three_d) {
          dln3 = (log(fmax(w(m, IDN, k + 1, j, i), dfloor)) -
                  log(fmax(w(m, IDN, k - 1, j, i), dfloor))) /
                 (2.0 * dx3);
        }

        Real j1 = 0.0;
        Real j2 =
            -(b(m, IBZ, k, j, i + 1) - b(m, IBZ, k, j, i - 1)) / (2.0 * dx1);
        Real j3 =
            (b(m, IBY, k, j, i + 1) - b(m, IBY, k, j, i - 1)) / (2.0 * dx1);
        if (multi_d) {
          j1 += (b(m, IBZ, k, j + 1, i) - b(m, IBZ, k, j - 1, i)) / (2.0 * dx2);
          j3 -= (b(m, IBX, k, j + 1, i) - b(m, IBX, k, j - 1, i)) / (2.0 * dx2);
        }
        if (three_d) {
          j1 -= (b(m, IBY, k + 1, j, i) - b(m, IBY, k - 1, j, i)) / (2.0 * dx3);
          j2 += (b(m, IBX, k + 1, j, i) - b(m, IBX, k - 1, j, i)) / (2.0 * dx3);
        }

        Real tele = gm1 * fmax(w(m, iele, k, j, i), 0.0) / fe;
        Real gradln = sqrt(SQR(dln1) + SQR(dln2) + SQR(dln3));
        Real vtm = coeff * sqrt(gm1 * tele / ne) * gradln;
        Real vd1 = -coeff * j1 / ne;
        Real vd2 = -coeff * j2 / ne;
        Real vd3 = -coeff * j3 / ne;
        Real speed = fabs(vd1) + vtm;
        if (speed > 0.0)
          min_dt1 = fmin(min_dt1, dx1 / speed);
        if (multi_d) {
          speed = fabs(vd2) + vtm;
          if (speed > 0.0)
            min_dt2 = fmin(min_dt2, dx2 / speed);
        }
        if (three_d) {
          speed = fabs(vd3) + vtm;
          if (speed > 0.0)
            min_dt3 = fmin(min_dt3, dx3 / speed);
        }
      },
      Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2), Kokkos::Min<Real>(dt3));

  dtnew = dt1;
  if (multi_d)
    dtnew = std::min(dtnew, dt2);
  if (three_d)
    dtnew = std::min(dtnew, dt3);
}

} // namespace mhd
