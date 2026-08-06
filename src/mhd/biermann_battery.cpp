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
#include "bvals/bvals.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "mhd/biermann_battery.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real ElectronPressure(const DvceArray5D<Real> &w, const int iele,
                      const DvceArray5D<Real> &thermodynamics,
                      const bool use_tabular_materials,
                      const Real gm1, const int m, const int k, const int j,
                      const int i) {
  if (use_tabular_materials) {
    return fmax(thermodynamics(
        m, two_temperature::TwoTemperature::electron_pressure, k, j, i), 0.0);
  }
  return gm1 * fmax(w(m, IDN, k, j, i) * w(m, iele, k, j, i), 0.0);
}

KOKKOS_INLINE_FUNCTION
Real TotalPressure(const DvceArray5D<Real> &w,
                   const DvceArray5D<Real> &thermodynamics,
                   const bool use_tabular_materials, const Real gm1,
                   const int m, const int k, const int j, const int i) {
  if (use_tabular_materials) {
    return fmax(
        thermodynamics(m, two_temperature::TwoTemperature::ion_pressure, k, j, i)+
        thermodynamics(m, two_temperature::TwoTemperature::electron_pressure,
                       k, j, i), 0.0);
  }
  return gm1*w(m, IEN, k, j, i);
}

struct BoundedElectronDensity {
  Real value;
  Real activation;
};

// Convert the shared physical number-density cache back to the normalization used by
// the Biermann coefficient: ne_code=rho*sum_s(Y_s Z_s/A_s).  In a nearly neutral table
// cell, regularize the denominator with a minimum q_e=ne/rho and smoothly switch the
// plasma source off between q_min and 2*q_min.  This prevents a nonzero table pressure
// floor from manufacturing an arbitrarily large battery in matter with no free electrons.
// The physical cached density is retained whenever the source is active.
KOKKOS_INLINE_FUNCTION
BoundedElectronDensity CachedElectronDensityCode(
    const DvceArray5D<Real> &thermodynamics, const Real cgs_to_code,
    const Real mass_density, const Real minimum_electron_fraction,
    const Real minimum_positive,
    const int m, const int k, const int j, const int i) {
  const Real physical = fmax(thermodynamics(
      m, two_temperature::TwoTemperature::electron_number_density_cgs,
      k, j, i)*cgs_to_code, 0.0);
  const Real threshold = fmax(
      minimum_electron_fraction*fmax(mass_density, 0.0), minimum_positive);
  BoundedElectronDensity result;
  result.value = fmax(physical, threshold);
  if (physical <= threshold) {
    result.activation = 0.0;
  } else if (physical >= 2.0*threshold) {
    result.activation = 1.0;
  } else {
    const Real x = physical/threshold-1.0;
    result.activation = x*x*(3.0-2.0*x);
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
Real FaceInverseElectronDensity(const BoundedElectronDensity &left,
                                const BoundedElectronDensity &right) {
  const Real activation = fmin(left.activation, right.activation);
  if (!(activation > 0.0)) return 0.0;
  const Real density = 0.5*(left.value+right.value);
  return activation/density;
}

KOKKOS_INLINE_FUNCTION
Real OneSidedSlopeBound(const Real vm, const Real vc, const Real vp,
                        const Real dx) {
  return fmax(fabs(vp-vc), fabs(vc-vm))/dx;
}

KOKKOS_INLINE_FUNCTION
Real ShockMaskFromJump(const Real jump, const Real compression,
                       const Real compression_threshold,
                       const Real threshold_low, const Real threshold_high) {
  if (compression <= compression_threshold || jump <= threshold_low) return 1.0;

  Real jump_activation = 1.0;
  if (jump < threshold_high) {
    jump_activation = (jump-threshold_low)/(threshold_high-threshold_low);
  }

  // Smoothly turn on compression gating between M_c and 2 M_c.  A hard switch
  // at M_c makes the mask discontinuous wherever sound speed varies transverse
  // to a shock, which is itself an artificial Biermann source.  M_c=0 retains
  // the useful sign-only limit for controlled tests.
  Real compression_activation = 1.0;
  if (compression_threshold > 0.0 && compression < 2.0*compression_threshold) {
    const Real x = compression/compression_threshold-1.0;
    compression_activation = x*x*(3.0-2.0*x);
  }
  return 1.0-jump_activation*compression_activation;
}

// Inverse of the logarithmic mean L(a,b)=(b-a)/(ln(b)-ln(a)).  Written in
// terms of the centered relative difference to avoid cancellation for nearly
// equal positive states.  The series is atanh(r)/r through O(r^6).
KOKKOS_INLINE_FUNCTION
Real InverseLogMean(const Real a, const Real b) {
  const Real mean = 0.5*(a+b);
  const Real r = (b-a)/(a+b);
  const Real ar = fabs(r);
  if (ar < 1.0e-4) {
    const Real r2 = r*r;
    return (1.0+r2*(1.0/3.0+r2*(1.0/5.0+r2/7.0)))/mean;
  }
  return log(b/a)/(b-a);
}

} // namespace

namespace mhd {

//----------------------------------------------------------------------------------------
// Constructor.  AthenaK uses mu0=k_B=1 normalized units here.  The coefficient
// is the normalized inverse electron charge in E_B=-C_B grad(p_e)/n_e and
// v_e-v=-C_B curl(B)/n_e.  Legacy decks use n_e=f_e*rho.  With <materials>, n_e
// is instead evaluated from rho*sum(Y_s Z_s/A_s); the local heat-capacity fraction
// remains a separate quantity used only to convert electron energy to temperature.

BiermannBattery::BiermannBattery(MeshBlockPack *ppack, ParameterInput *pin,
                                 int electron_index,
                                 Real electron_heat_capacity_fraction,
                                 Real gamma, Real density_floor,
                                 Real pressure_floor,
                                 materials::MaterialMixture *material_mixture)
    : coefficient(pin->GetOrAddReal("mhd", "biermann_coefficient", 1.0)),
      dtnew(std::numeric_limits<Real>::max()),
      suppress_in_shocks(
          pin->GetOrAddBoolean("mhd", "biermann_shock_suppression", true)),
      shock_threshold(
          pin->GetOrAddReal("mhd", "biermann_shock_threshold", 0.25)),
      shock_compression_threshold(pin->GetOrAddReal(
          "mhd", "biermann_shock_compression_threshold", 0.02)),
      pmy_pack_(ppack), iele_(electron_index),
      electron_fraction_(electron_heat_capacity_fraction), gamma_(gamma),
      gamma_minus_one_(gamma - 1.0), density_floor_(density_floor),
      pressure_floor_(pressure_floor),
      temperature_floor_(pin->GetOrAddReal("mhd", "tfloor", 0.0)),
      minimum_electron_fraction_(pin->GetOrAddReal(
          "mhd", "biermann_minimum_electron_fraction", 1.0e-12)),
      use_material_mixture_(material_mixture != nullptr),
      smooth_x1_("biermann-smooth-x1", 1, 1, 1, 1),
      smooth_x2_("biermann-smooth-x2", 1, 1, 1, 1),
      smooth_x3_("biermann-smooth-x3", 1, 1, 1, 1),
      e3x1_("biermann-e3x1", 1, 1, 1, 1), e2x1_("biermann-e2x1", 1, 1, 1, 1),
      e1x2_("biermann-e1x2", 1, 1, 1, 1), e3x2_("biermann-e3x2", 1, 1, 1, 1),
      e2x3_("biermann-e2x3", 1, 1, 1, 1), e1x3_("biermann-e1x3", 1, 1, 1, 1),
      vd1_("biermann-vd1", 1, 1, 1, 1), vd2_("biermann-vd2", 1, 1, 1, 1),
      vd3_("biermann-vd3", 1, 1, 1, 1),
      vd_flux_("biermann-vd-flux", 1, 1, 1, 1, 1),
      pressure_vertex_("biermann-pressure-vertex", 1, 1, 1, 1),
      electron_density_vertex_("biermann-electron-density-vertex", 1, 1, 1, 1) {
  if (use_material_mixture_) material_mixture_ = material_mixture->DeviceData();
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
  if (!std::isfinite(shock_compression_threshold) ||
      shock_compression_threshold < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<mhd>/biermann_shock_compression_threshold must be finite and "
              << "non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(minimum_electron_fraction_) ||
      minimum_electron_fraction_ <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<mhd>/biermann_minimum_electron_fraction must be finite and "
              << "positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2 * indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng : 1;
  Kokkos::realloc(smooth_x1_, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(smooth_x2_, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(smooth_x3_, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(e3x1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(e2x1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(vd1_, nmb, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(e1x2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(e3x2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(vd2_, nmb, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(e2x3_, nmb, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(e1x3_, nmb, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(vd3_, nmb, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(vd_flux_.x1f, nmb, 1, ncells3, ncells2, ncells1 + 1);
  Kokkos::realloc(vd_flux_.x2f, nmb, 1, ncells3, ncells2 + 1, ncells1);
  Kokkos::realloc(vd_flux_.x3f, nmb, 1, ncells3 + 1, ncells2, ncells1);
  Kokkos::realloc(pressure_vertex_, nmb, ncells3 + 1, ncells2 + 1,
                  ncells1 + 1);
  Kokkos::realloc(electron_density_vertex_, nmb, ncells3 + 1,
                  ncells2 + 1, ncells1 + 1);
}

//----------------------------------------------------------------------------------------
//! \brief Build a symmetric FLASH-like shock indicator.  Raw grad(p_e)/n_e is
//! not a convergent Biermann discretization inside a discontinuity, so faces
//! adjacent to cells with a large centered gas-pressure jump are attenuated
//! when requested.  In resolved compressive flow each directional mask ramps
//! linearly from 1 at jump = threshold/5 down to 0 at jump = threshold.
//! Compression is normalized by the local acoustic speed and ramps on smoothly
//! from the configured threshold to twice that value; requiring a finite
//! converging velocity jump prevents most thermal-magnetic grid oscillations from
//! being misclassified as shocks without introducing a second hard mask edge.
//! Applying the masks as a diagonal operator to matching components of grad(p_e)
//! and curl(B), rather than multiplying the full electric/drift vectors by one
//! scalar minimum, reduces unrelated transverse mask coupling.  The same operator
//! on both terms retains their discrete energy-exchange pairing; curved or oblique
//! shock surfaces still require the gauge-form treatment for exact curl consistency.
//! Smooth flow (jump <= threshold/5) keeps every component exactly 1.

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
  auto smooth1 = smooth_x1_;
  auto smooth2 = smooth_x2_;
  auto smooth3 = smooth_x3_;

  if (!suppress_in_shocks) {
    par_for(
        "biermann_smooth_all", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          smooth1(m, k, j, i) = 1.0;
          smooth2(m, k, j, i) = 1.0;
          smooth3(m, k, j, i) = 1.0;
        });
    return;
  }

  Real gm1 = gamma_minus_one_;
  Real pfloor = fmax(pressure_floor_, 1.0e-30);
  Real thr_hi = shock_threshold;
  Real thr_lo = 0.2 * shock_threshold;
  Real compression_threshold = shock_compression_threshold;
  Real gamma = gamma_;
  Real dfloor = density_floor_;
  auto w = prim;
  auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;
  bool use_tabular_materials =
      use_material_mixture_ && material_mixture_.UsesTabularEOS();
  par_for(
      "biermann_shock_mask", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real pm = fmax(TotalPressure(
            w, thermodynamics, use_tabular_materials, gm1, m, k, j, i-1), pfloor);
        Real pp = fmax(TotalPressure(
            w, thermodynamics, use_tabular_materials, gm1, m, k, j, i+1), pfloor);
        Real jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
        Real rho_m = fmax(w(m, IDN, k, j, i-1), dfloor);
        Real rho_p = fmax(w(m, IDN, k, j, i+1), dfloor);
        Real speed = sqrt(gamma*(pp+pm)/(rho_p+rho_m));
        Real compression = (w(m, IVX, k, j, i-1)-w(m, IVX, k, j, i+1))/speed;
        smooth1(m, k, j, i) = ShockMaskFromJump(
            jump, compression, compression_threshold, thr_lo, thr_hi);
        if (multi_d) {
          pm = fmax(TotalPressure(
              w, thermodynamics, use_tabular_materials, gm1, m, k, j-1, i), pfloor);
          pp = fmax(TotalPressure(
              w, thermodynamics, use_tabular_materials, gm1, m, k, j+1, i), pfloor);
          jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
          rho_m = fmax(w(m, IDN, k, j-1, i), dfloor);
          rho_p = fmax(w(m, IDN, k, j+1, i), dfloor);
          speed = sqrt(gamma*(pp+pm)/(rho_p+rho_m));
          compression = (w(m, IVY, k, j-1, i)-w(m, IVY, k, j+1, i))/speed;
          smooth2(m, k, j, i) = ShockMaskFromJump(
              jump, compression, compression_threshold, thr_lo, thr_hi);
        } else {
          smooth2(m, k, j, i) = 1.0;
        }
        if (three_d) {
          pm = fmax(TotalPressure(
              w, thermodynamics, use_tabular_materials, gm1, m, k-1, j, i), pfloor);
          pp = fmax(TotalPressure(
              w, thermodynamics, use_tabular_materials, gm1, m, k+1, j, i), pfloor);
          jump = 2.0 * fabs(pp - pm) / fmax(pp + pm, 2.0 * pfloor);
          rho_m = fmax(w(m, IDN, k-1, j, i), dfloor);
          rho_p = fmax(w(m, IDN, k+1, j, i), dfloor);
          speed = sqrt(gamma*(pp+pm)/(rho_p+rho_m));
          compression = (w(m, IVZ, k-1, j, i)-w(m, IVZ, k+1, j, i))/speed;
          smooth3(m, k, j, i) = ShockMaskFromJump(
              jump, compression, compression_threshold, thr_lo, thr_hi);
        } else {
          smooth3(m, k, j, i) = 1.0;
        }
      });
}

//----------------------------------------------------------------------------------------
//! \brief Compute Cartesian face fluxes.  Magnetic fluxes are represented by
//! the transverse face electric fields and are later passed through flux-CT in
//! AddEMFs().

void BiermannBattery::AddFluxes(const DvceArray5D<Real> &prim,
                                const DvceArray5D<Real> &bcc,
                                DvceFaceFld5D<Real> &flx) {
  const bool edge_integral_subcycle = pmy_pack_->pmhd->biermann_subcycle;
  if (!edge_integral_subcycle) ComputeShockMask(prim);

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto size = pmy_pack_->pmb->mb_size;
  auto w = prim;
  auto smooth1 = smooth_x1_;
  auto smooth2 = smooth_x2_;
  auto smooth3 = smooth_x3_;
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
  bool use_materials = use_material_mixture_;
  auto material_mixture = material_mixture_;
  bool use_tabular_materials =
      use_materials && material_mixture.UsesTabularEOS();
  auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;
  Real electron_density_cgs_to_code = use_tabular_materials
      ? materials::MaterialMixtureDevice::atomic_mass_unit_cgs/
        material_mixture.density_to_cgs
      : 1.0;
  Real minimum_electron_fraction = minimum_electron_fraction_;
  Real minimum_positive = std::numeric_limits<Real>::min();
  // x1-face electric fields, including the transverse halo needed by flux-CT.
  int jl = multi_d ? js - 1 : js;
  int ju = multi_d ? je + 1 : je;
  int kl = three_d ? ks - 1 : ks;
  int ku = three_d ? ke + 1 : ke;
  if (!edge_integral_subcycle) {
    par_for(
        "biermann_face_e1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, is, ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real dp2 = 0.0;
        Real dp3 = 0.0;
        if (multi_d) {
          Real pp = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j+1, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j+1, i);
          Real pm = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j-1, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j-1, i);
          dp2 = 0.25 * (pp - pm) / size.d_view(m).dx2;
        }
        if (three_d) {
          Real pp = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k+1, j, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k+1, j, i);
          Real pm = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j, i);
          dp3 = 0.25 * (pp - pm) / size.d_view(m).dx3;
        }
        Real mask2 = fmin(smooth2(m, k, j, i - 1), smooth2(m, k, j, i));
        Real mask3 = fmin(smooth3(m, k, j, i - 1), smooth3(m, k, j, i));
        Real inv_ne;
        if (use_tabular_materials) {
          const Real rho_l = fmax(w(m, IDN, k, j, i-1), dfloor);
          const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
          const BoundedElectronDensity ne_l = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_l,
              minimum_electron_fraction, minimum_positive, m, k, j, i-1);
          const BoundedElectronDensity ne_r = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_r,
              minimum_electron_fraction, minimum_positive, m, k, j, i);
          inv_ne = FaceInverseElectronDensity(ne_l, ne_r);
        } else if (use_materials) {
          const Real rho_l = fmax(w(m, IDN, k, j, i-1), dfloor);
          const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
              const auto mix_l = material_mixture.CompositionFromPrimitive(
                  w, m, k, j, i-1);
              const auto mix_r = material_mixture.CompositionFromPrimitive(
                  w, m, k, j, i);
          const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                               material_mixture.ElectronNumberDensity(rho_r, mix_r));
          inv_ne = 1.0/ne;
        } else {
          const Real rho = fmax(
              0.5*(w(m, IDN, k, j, i-1)+w(m, IDN, k, j, i)), dfloor);
          inv_ne = 1.0/(fe*rho);
        }
        e21(m, k, j, i) = -coeff * mask2 * dp2 * inv_ne;
        e31(m, k, j, i) = -coeff * mask3 * dp3 * inv_ne;
        });
  }

  // x2-face electric fields.
  if (!edge_integral_subcycle && multi_d) {
    kl = three_d ? ks - 1 : ks;
    ku = three_d ? ke + 1 : ke;
    par_for(
        "biermann_face_e2", DevExeSpace(), 0, nmb1, kl, ku, js, je + 1, is - 1,
        ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real pp = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j-1, i+1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j, i+1);
          Real pm = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j-1, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j, i-1);
          Real dp1 = 0.25 * (pp - pm) / size.d_view(m).dx1;
          Real dp3 = 0.0;
          if (three_d) {
            pp = ElectronPressure(
                w, iele, thermodynamics, use_tabular_materials,
                gm1, m, k+1, j-1, i)+ElectronPressure(
                w, iele, thermodynamics, use_tabular_materials,
                gm1, m, k+1, j, i);
            pm = ElectronPressure(
                w, iele, thermodynamics, use_tabular_materials,
                gm1, m, k-1, j-1, i)+ElectronPressure(
                w, iele, thermodynamics, use_tabular_materials,
                gm1, m, k-1, j, i);
            dp3 = 0.25 * (pp - pm) / size.d_view(m).dx3;
          }
          Real mask1 = fmin(smooth1(m, k, j - 1, i), smooth1(m, k, j, i));
          Real mask3 = fmin(smooth3(m, k, j - 1, i), smooth3(m, k, j, i));
          Real inv_ne;
          if (use_tabular_materials) {
            const Real rho_l = fmax(w(m, IDN, k, j-1, i), dfloor);
            const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
            const BoundedElectronDensity ne_l = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_l,
                minimum_electron_fraction, minimum_positive, m, k, j-1, i);
            const BoundedElectronDensity ne_r = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_r,
                minimum_electron_fraction, minimum_positive, m, k, j, i);
            inv_ne = FaceInverseElectronDensity(ne_l, ne_r);
          } else if (use_materials) {
            const Real rho_l = fmax(w(m, IDN, k, j-1, i), dfloor);
            const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
                const auto mix_l = material_mixture.CompositionFromPrimitive(
                    w, m, k, j-1, i);
                const auto mix_r = material_mixture.CompositionFromPrimitive(
                    w, m, k, j, i);
            const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                                 material_mixture.ElectronNumberDensity(rho_r, mix_r));
            inv_ne = 1.0/ne;
          } else {
            const Real rho = fmax(
                0.5*(w(m, IDN, k, j-1, i)+w(m, IDN, k, j, i)), dfloor);
            inv_ne = 1.0/(fe*rho);
          }
          e12(m, k, j, i) = -coeff * mask1 * dp1 * inv_ne;
          e32(m, k, j, i) = -coeff * mask3 * dp3 * inv_ne;
        });
  }

  // x3-face electric fields.
  if (!edge_integral_subcycle && three_d) {
    par_for(
        "biermann_face_e3", DevExeSpace(), 0, nmb1, ks, ke + 1, js - 1, je + 1,
        is - 1, ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real pp = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j, i+1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j, i+1);
          Real pm = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j, i-1)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j, i-1);
          Real dp1 = 0.25 * (pp - pm) / size.d_view(m).dx1;
          pp = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j+1, i)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j+1, i);
          pm = ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k-1, j-1, i)+ElectronPressure(
              w, iele, thermodynamics, use_tabular_materials,
              gm1, m, k, j-1, i);
          Real dp2 = 0.25 * (pp - pm) / size.d_view(m).dx2;
          Real mask1 = fmin(smooth1(m, k - 1, j, i), smooth1(m, k, j, i));
          Real mask2 = fmin(smooth2(m, k - 1, j, i), smooth2(m, k, j, i));
          Real inv_ne;
          if (use_tabular_materials) {
            const Real rho_l = fmax(w(m, IDN, k-1, j, i), dfloor);
            const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
            const BoundedElectronDensity ne_l = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_l,
                minimum_electron_fraction, minimum_positive, m, k-1, j, i);
            const BoundedElectronDensity ne_r = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_r,
                minimum_electron_fraction, minimum_positive, m, k, j, i);
            inv_ne = FaceInverseElectronDensity(ne_l, ne_r);
          } else if (use_materials) {
            const Real rho_l = fmax(w(m, IDN, k-1, j, i), dfloor);
            const Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
                const auto mix_l = material_mixture.CompositionFromPrimitive(
                    w, m, k-1, j, i);
                const auto mix_r = material_mixture.CompositionFromPrimitive(
                    w, m, k, j, i);
            const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                                 material_mixture.ElectronNumberDensity(rho_r, mix_r));
            inv_ne = 1.0/ne;
          } else {
            const Real rho = fmax(
                0.5*(w(m, IDN, k-1, j, i)+w(m, IDN, k, j, i)), dfloor);
            inv_ne = 1.0/(fe*rho);
          }
          e13(m, k, j, i) = -coeff * mask1 * dp1 * inv_ne;
          e23(m, k, j, i) = -coeff * mask2 * dp2 * inv_ne;
        });
  }

  // Add Poynting and electron enthalpy fluxes on active x1 faces.  The
  // component electron equation carries epsilon_e*v_d; p_e*div(v_d) is applied
  // after the update.
  auto b = bcc;
  auto flx1 = flx.x1f;
  auto vd1 = vd1_;
  auto vd_flux1 = vd_flux_.x1f;
  Real gamma = gamma_;
  par_for(
      "biermann_energy_flux1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real rho_l = fmax(w(m, IDN, k, j, i - 1), dfloor);
        Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
        Real mask = edge_integral_subcycle ? 1.0 :
            fmin(smooth1(m, k, j, i - 1), smooth1(m, k, j, i));
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
        Real drift;
        if (use_tabular_materials) {
          const BoundedElectronDensity ne_l = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_l,
              minimum_electron_fraction, minimum_positive, m, k, j, i-1);
          const BoundedElectronDensity ne_r = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_r,
              minimum_electron_fraction, minimum_positive, m, k, j, i);
          drift = -coeff*mask*j1*FaceInverseElectronDensity(ne_l, ne_r);
        } else if (use_materials) {
              const auto mix_l = material_mixture.CompositionFromPrimitive(
                  w, m, k, j, i-1);
              const auto mix_r = material_mixture.CompositionFromPrimitive(
                  w, m, k, j, i);
          const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                               material_mixture.ElectronNumberDensity(rho_r, mix_r));
          drift = -coeff*mask*j1/ne;
        } else {
          const Real rho = fmax(0.5*(rho_l+rho_r), dfloor);
          drift = -coeff*mask*j1/(fe*rho);
        }
        vd1(m, k, j, i) = drift;
        vd_flux1(m, 0, k, j, i) = drift;
        Real eps_l = fmax(rho_l * w(m, iele, k, j, i - 1), 0.0);
        Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
        Real eps = (drift >= 0.0) ? eps_l : eps_r;
        flx1(m, iele, k, j, i) += eps * drift;
        Real by = 0.5 * (b(m, IBY, k, j, i - 1) + b(m, IBY, k, j, i));
        Real bz = 0.5 * (b(m, IBZ, k, j, i - 1) + b(m, IBZ, k, j, i));
        if (!edge_integral_subcycle) {
          flx1(m, IEN, k, j, i) +=
              e21(m, k, j, i) * bz - e31(m, k, j, i) * by;
        }
        if (use_tabular_materials) {
          const Real pe_l = fmax(thermodynamics(
              m, two_temperature::TwoTemperature::electron_pressure,
              k, j, i-1), 0.0);
          const Real pe_r = fmax(thermodynamics(
              m, two_temperature::TwoTemperature::electron_pressure,
              k, j, i), 0.0);
          flx1(m, IEN, k, j, i) +=
              (eps+((drift >= 0.0) ? pe_l : pe_r))*drift;
        } else {
          flx1(m, IEN, k, j, i) += gamma*eps*drift;
        }
      });

  // Active x2 faces.
  if (multi_d) {
    auto flx2 = flx.x2f;
    auto vd2 = vd2_;
    auto vd_flux2 = vd_flux_.x2f;
    par_for(
        "biermann_energy_flux2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is,
        ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real rho_l = fmax(w(m, IDN, k, j - 1, i), dfloor);
          Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
          Real mask = edge_integral_subcycle ? 1.0 :
              fmin(smooth2(m, k, j - 1, i), smooth2(m, k, j, i));
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
          Real drift;
          if (use_tabular_materials) {
            const BoundedElectronDensity ne_l = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_l,
                minimum_electron_fraction, minimum_positive, m, k, j-1, i);
            const BoundedElectronDensity ne_r = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_r,
                minimum_electron_fraction, minimum_positive, m, k, j, i);
            drift = -coeff*mask*j2*FaceInverseElectronDensity(ne_l, ne_r);
          } else if (use_materials) {
                const auto mix_l = material_mixture.CompositionFromPrimitive(
                    w, m, k, j-1, i);
                const auto mix_r = material_mixture.CompositionFromPrimitive(
                    w, m, k, j, i);
            const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                                 material_mixture.ElectronNumberDensity(rho_r, mix_r));
            drift = -coeff*mask*j2/ne;
          } else {
            const Real rho = fmax(0.5*(rho_l+rho_r), dfloor);
            drift = -coeff*mask*j2/(fe*rho);
          }
          vd2(m, k, j, i) = drift;
          vd_flux2(m, 0, k, j, i) = drift;
          Real eps_l = fmax(rho_l * w(m, iele, k, j - 1, i), 0.0);
          Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
          Real eps = (drift >= 0.0) ? eps_l : eps_r;
          flx2(m, iele, k, j, i) += eps * drift;
          Real bx = 0.5 * (b(m, IBX, k, j - 1, i) + b(m, IBX, k, j, i));
          Real bz = 0.5 * (b(m, IBZ, k, j - 1, i) + b(m, IBZ, k, j, i));
          if (!edge_integral_subcycle) {
            flx2(m, IEN, k, j, i) +=
                e32(m, k, j, i) * bx - e12(m, k, j, i) * bz;
          }
          if (use_tabular_materials) {
            const Real pe_l = fmax(thermodynamics(
                m, two_temperature::TwoTemperature::electron_pressure,
                k, j-1, i), 0.0);
            const Real pe_r = fmax(thermodynamics(
                m, two_temperature::TwoTemperature::electron_pressure,
                k, j, i), 0.0);
            flx2(m, IEN, k, j, i) +=
                (eps+((drift >= 0.0) ? pe_l : pe_r))*drift;
          } else {
            flx2(m, IEN, k, j, i) += gamma*eps*drift;
          }
        });
  }

  // Active x3 faces.
  if (three_d) {
    auto flx3 = flx.x3f;
    auto vd3 = vd3_;
    auto vd_flux3 = vd_flux_.x3f;
    par_for(
        "biermann_energy_flux3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je, is,
        ie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real rho_l = fmax(w(m, IDN, k - 1, j, i), dfloor);
          Real rho_r = fmax(w(m, IDN, k, j, i), dfloor);
          Real mask = edge_integral_subcycle ? 1.0 :
              fmin(smooth3(m, k - 1, j, i), smooth3(m, k, j, i));
          Real j3 = 0.25 *
                    (b(m, IBY, k - 1, j, i + 1) + b(m, IBY, k, j, i + 1) -
                     b(m, IBY, k - 1, j, i - 1) - b(m, IBY, k, j, i - 1)) /
                    size.d_view(m).dx1;
          j3 -= 0.25 *
                (b(m, IBX, k - 1, j + 1, i) + b(m, IBX, k, j + 1, i) -
                 b(m, IBX, k - 1, j - 1, i) - b(m, IBX, k, j - 1, i)) /
                size.d_view(m).dx2;
          Real drift;
          if (use_tabular_materials) {
            const BoundedElectronDensity ne_l = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_l,
                minimum_electron_fraction, minimum_positive, m, k-1, j, i);
            const BoundedElectronDensity ne_r = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_r,
                minimum_electron_fraction, minimum_positive, m, k, j, i);
            drift = -coeff*mask*j3*FaceInverseElectronDensity(ne_l, ne_r);
          } else if (use_materials) {
                const auto mix_l = material_mixture.CompositionFromPrimitive(
                    w, m, k-1, j, i);
                const auto mix_r = material_mixture.CompositionFromPrimitive(
                    w, m, k, j, i);
            const Real ne = 0.5*(material_mixture.ElectronNumberDensity(rho_l, mix_l)+
                                 material_mixture.ElectronNumberDensity(rho_r, mix_r));
            drift = -coeff*mask*j3/ne;
          } else {
            const Real rho = fmax(0.5*(rho_l+rho_r), dfloor);
            drift = -coeff*mask*j3/(fe*rho);
          }
          vd3(m, k, j, i) = drift;
          vd_flux3(m, 0, k, j, i) = drift;
          Real eps_l = fmax(rho_l * w(m, iele, k - 1, j, i), 0.0);
          Real eps_r = fmax(rho_r * w(m, iele, k, j, i), 0.0);
          Real eps = (drift >= 0.0) ? eps_l : eps_r;
          flx3(m, iele, k, j, i) += eps * drift;
          Real bx = 0.5 * (b(m, IBX, k - 1, j, i) + b(m, IBX, k, j, i));
          Real by = 0.5 * (b(m, IBY, k - 1, j, i) + b(m, IBY, k, j, i));
          if (!edge_integral_subcycle) {
            flx3(m, IEN, k, j, i) +=
                e13(m, k, j, i) * by - e23(m, k, j, i) * bx;
          }
          if (use_tabular_materials) {
            const Real pe_l = fmax(thermodynamics(
                m, two_temperature::TwoTemperature::electron_pressure,
                k-1, j, i), 0.0);
            const Real pe_r = fmax(thermodynamics(
                m, two_temperature::TwoTemperature::electron_pressure,
                k, j, i), 0.0);
            flx3(m, IEN, k, j, i) +=
                (eps+((drift >= 0.0) ? pe_l : pe_r))*drift;
          } else {
            flx3(m, IEN, k, j, i) += gamma*eps*drift;
          }
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

  // Path-conservative endpoint form of the physical one-form
  // -C_B*int(dp_e/n_e).  The logarithmic mean of n_e is the unique two-point
  // mean used here because it has both required limiting properties:
  //
  //  * constant n_e: I_edge=-C_B*(p_b-p_a)/n_e, so the circulation
  //    telescopes to roundoff for an arbitrary pressure field;
  //  * ideal isothermal jump p_e=n_e*T_e: I_edge=-C_B*T_e*ln(p_b/p_a),
  //    exactly the gauge-equivalent Graziani shock integral.
  //
  // It is second-order equivalent to -C_B*dp_e/n_e in smooth flow.  The
  // communicated edge value is therefore finite at a discontinuity without
  // introducing a spatial shock mask.  This is the production subcycle operator.
  if (pmy_pack_->pmhd->biermann_subcycle) {
    auto w = pmy_pack_->pmhd->w0;
    auto pressure = pressure_vertex_;
    auto electron_density = electron_density_vertex_;
    auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;
    auto material_mixture = material_mixture_;
    const int iele = iele_;
    const Real gm1 = gamma_minus_one_;
    const Real fe = electron_fraction_;
    const Real coeff = coefficient;
    const Real dfloor = density_floor_;
    const Real minimum_electron_fraction = minimum_electron_fraction_;
    const Real minimum_positive = std::numeric_limits<Real>::min();
    const bool use_materials = use_material_mixture_;
    const bool use_tabular_materials =
        use_materials && material_mixture.UsesTabularEOS();
    const Real electron_density_cgs_to_code = use_tabular_materials
        ? materials::MaterialMixtureDevice::atomic_mass_unit_cgs/
          material_mixture.density_to_cgs
        : 1.0;
    auto size = pmy_pack_->pmb->mb_size;

    if (pmy_pack_->pmesh->two_d) {
      par_for(
          "biermann_edge_integral_vertex_state_2d", DevExeSpace(), 0, nmb1,
          js, je+1, is, ie+1,
          KOKKOS_LAMBDA(const int m, const int j, const int i) {
            Real vertex_pressure = 0.0;
            Real vertex_density = 0.0;
            for (int dj=-1; dj<=0; ++dj) {
              for (int di=-1; di<=0; ++di) {
                const int jc = j+dj;
                const int ic = i+di;
                const Real rho = fmax(w(m, IDN, ks, jc, ic), dfloor);
                Real pe = ElectronPressure(
                    w, iele, thermodynamics, use_tabular_materials,
                    gm1, m, ks, jc, ic);
                Real ne;
                if (use_tabular_materials) {
                  const BoundedElectronDensity bounded = CachedElectronDensityCode(
                      thermodynamics, electron_density_cgs_to_code, rho,
                      minimum_electron_fraction, minimum_positive,
                      m, ks, jc, ic);
                  // Treat activation as part of the pressure coordinate rather
                  // than multiplying an edge by a spatial mask.  This preserves
                  // exact constant-density telescoping through neutral regions.
                  pe *= bounded.activation;
                  ne = bounded.value;
                } else if (use_materials) {
                  const auto mix_y = material_mixture.CompositionFromPrimitive(
                      w, m, ks, jc, ic);
                  ne = material_mixture.ElectronNumberDensity(rho, mix_y);
                } else {
                  ne = fe*rho;
                }
                vertex_pressure += 0.25*pe;
                vertex_density += 0.25*ne;
              }
            }
            pressure(m, ks, j, i) = vertex_pressure;
            electron_density(m, ks, j, i) =
                fmax(vertex_density, minimum_positive);
          });

      par_for(
          "biermann_edge_integral_e1_2d", DevExeSpace(), 0, nmb1,
          js, je+1, is, ie,
          KOKKOS_LAMBDA(const int m, const int j, const int i) {
            const Real inv_ne = InverseLogMean(
                electron_density(m, ks, j, i),
                electron_density(m, ks, j, i+1));
            const Real value = -coeff*(pressure(m, ks, j, i+1)-
                pressure(m, ks, j, i))*inv_ne/size.d_view(m).dx1;
            e1(m, ks, j, i) += value;
            e1(m, ke+1, j, i) += value;
          });

      par_for(
          "biermann_edge_integral_e2_2d", DevExeSpace(), 0, nmb1,
          js, je, is, ie+1,
          KOKKOS_LAMBDA(const int m, const int j, const int i) {
            const Real inv_ne = InverseLogMean(
                electron_density(m, ks, j, i),
                electron_density(m, ks, j+1, i));
            const Real value = -coeff*(pressure(m, ks, j+1, i)-
                pressure(m, ks, j, i))*inv_ne/size.d_view(m).dx2;
            e2(m, ks, j, i) += value;
            e2(m, ke+1, j, i) += value;
          });
      return;
    }

    if (pmy_pack_->pmesh->three_d) {
      par_for(
          "biermann_edge_integral_vertex_state_3d", DevExeSpace(), 0, nmb1,
          ks, ke+1, js, je+1, is, ie+1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            Real vertex_pressure = 0.0;
            Real vertex_density = 0.0;
            for (int dk=-1; dk<=0; ++dk) {
              for (int dj=-1; dj<=0; ++dj) {
                for (int di=-1; di<=0; ++di) {
                  const int kc = k+dk;
                  const int jc = j+dj;
                  const int ic = i+di;
                  const Real rho = fmax(w(m, IDN, kc, jc, ic), dfloor);
                  Real pe = ElectronPressure(
                      w, iele, thermodynamics, use_tabular_materials,
                      gm1, m, kc, jc, ic);
                  Real ne;
                  if (use_tabular_materials) {
                    const BoundedElectronDensity bounded = CachedElectronDensityCode(
                        thermodynamics, electron_density_cgs_to_code, rho,
                        minimum_electron_fraction, minimum_positive,
                        m, kc, jc, ic);
                    pe *= bounded.activation;
                    ne = bounded.value;
                  } else if (use_materials) {
                    const auto mix_y =
                        material_mixture.CompositionFromPrimitive(
                            w, m, kc, jc, ic);
                    ne = material_mixture.ElectronNumberDensity(rho, mix_y);
                  } else {
                    ne = fe*rho;
                  }
                  vertex_pressure += 0.125*pe;
                  vertex_density += 0.125*ne;
                }
              }
            }
            pressure(m, k, j, i) = vertex_pressure;
            electron_density(m, k, j, i) =
                fmax(vertex_density, minimum_positive);
          });

      par_for(
          "biermann_edge_integral_e1_3d", DevExeSpace(), 0, nmb1,
          ks, ke+1, js, je+1, is, ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const Real inv_ne = InverseLogMean(
                electron_density(m, k, j, i),
                electron_density(m, k, j, i+1));
            e1(m, k, j, i) += -coeff*(pressure(m, k, j, i+1)-
                pressure(m, k, j, i))*inv_ne/size.d_view(m).dx1;
          });

      par_for(
          "biermann_edge_integral_e2_3d", DevExeSpace(), 0, nmb1,
          ks, ke+1, js, je, is, ie+1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const Real inv_ne = InverseLogMean(
                electron_density(m, k, j, i),
                electron_density(m, k, j+1, i));
            e2(m, k, j, i) += -coeff*(pressure(m, k, j+1, i)-
                pressure(m, k, j, i))*inv_ne/size.d_view(m).dx2;
          });

      par_for(
          "biermann_edge_integral_e3_3d", DevExeSpace(), 0, nmb1,
          ks, ke, js, je+1, is, ie+1,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const Real inv_ne = InverseLogMean(
                electron_density(m, k, j, i),
                electron_density(m, k+1, j, i));
            e3(m, k, j, i) += -coeff*(pressure(m, k+1, j, i)-
                pressure(m, k, j, i))*inv_ne/size.d_view(m).dx3;
          });
      return;
    }
  }

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
//! \brief Add E_B x B on every active face from the final communicated edge EMF.
//!
//! AddFluxes has already added the electron drift/enthalpy terms.  Poynting flux
//! is delayed until after edge reconciliation and communication so the magnetic
//! and total-energy updates use the same discrete electric field.  Ordinary CC
//! flux correction then restricts this complete fine-face flux at AMR interfaces.

void BiermannBattery::AddPoyntingFluxFromEdgeEMF(
    const DvceArray5D<Real> &bcc, const DvceEdgeFld4D<Real> &efld,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto b = bcc;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto flx1 = flx.x1f;
  auto flx2 = flx.x2f;
  auto flx3 = flx.x3f;

  par_for(
      "biermann_edge_poynting_flux1", DevExeSpace(), 0, nmb1,
      ks, ke, js, je, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real ey = 0.5*(e2(m, k, j, i)+e2(m, k+1, j, i));
        const Real ez = 0.5*(e3(m, k, j, i)+e3(m, k, j+1, i));
        const Real by = 0.5*(b(m, IBY, k, j, i-1)+b(m, IBY, k, j, i));
        const Real bz = 0.5*(b(m, IBZ, k, j, i-1)+b(m, IBZ, k, j, i));
        flx1(m, IEN, k, j, i) += ey*bz-ez*by;
      });

  if (multi_d) {
    par_for(
        "biermann_edge_poynting_flux2", DevExeSpace(), 0, nmb1,
        ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          const Real ex = 0.5*(e1(m, k, j, i)+e1(m, k+1, j, i));
          const Real ez = 0.5*(e3(m, k, j, i)+e3(m, k, j, i+1));
          const Real bx = 0.5*(b(m, IBX, k, j-1, i)+b(m, IBX, k, j, i));
          const Real bz = 0.5*(b(m, IBZ, k, j-1, i)+b(m, IBZ, k, j, i));
          flx2(m, IEN, k, j, i) += ez*bx-ex*bz;
        });
  }

  if (three_d) {
    par_for(
        "biermann_edge_poynting_flux3", DevExeSpace(), 0, nmb1,
        ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          const Real ex = 0.5*(e1(m, k, j, i)+e1(m, k, j+1, i));
          const Real ey = 0.5*(e2(m, k, j, i)+e2(m, k, j, i+1));
          const Real bx = 0.5*(b(m, IBX, k-1, j, i)+b(m, IBX, k, j, i));
          const Real by = 0.5*(b(m, IBY, k-1, j, i)+b(m, IBY, k, j, i));
          flx3(m, IEN, k, j, i) += ex*by-ey*bx;
        });
  }
}

bool BiermannBattery::DirectDriftCorrectionEnabled() const {
  return pmy_pack_->pmesh->multilevel;
}

DvceFaceFld5D<Real> *BiermannBattery::DriftCorrectionFlux() {
  return &vd_flux_;
}

void BiermannBattery::UseCorrectedDriftFlux() {
  if (!pmy_pack_->pmesh->multilevel) return;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack - 1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  auto vd1 = vd1_;
  auto vd2 = vd2_;
  auto vd3 = vd3_;
  auto vd_flux1 = vd_flux_.x1f;
  auto vd_flux2 = vd_flux_.x2f;
  auto vd_flux3 = vd_flux_.x3f;

  par_for(
      "biermann_use_corrected_vd1", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        vd1(m, k, j, i) = vd_flux1(m, 0, k, j, i);
      });
  if (multi_d) {
    par_for(
        "biermann_use_corrected_vd2", DevExeSpace(), 0, nmb1, ks, ke, js,
        je + 1, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          vd2(m, k, j, i) = vd_flux2(m, 0, k, j, i);
        });
  }
  if (three_d) {
    par_for(
        "biermann_use_corrected_vd3", DevExeSpace(), 0, nmb1, ks, ke + 1, js,
        je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          vd3(m, k, j, i) = vd_flux3(m, 0, k, j, i);
        });
  }
}

//----------------------------------------------------------------------------------------
//! \brief Apply the p_e div(v_e-v) term omitted by conservative advection of
//! epsilon_e.

// This exponential map is used only by the legacy stage-coupled path.  Its face drift
// velocities vd*_ are computed per level and are NOT SMR/AMR flux-corrected (unlike
// dual_vf).  The dedicated subcycle instead refluxes vd_flux_ and installs it through
// UseCorrectedDriftFlux before applying AddElectronWorkRHS.  Measured legacy
// consequence: the ion/electron partition at coarse/fine boundaries is perturbed only
// within the coarse-fine
// truncation envelope (~6e-6 vs ~1e-5 band over 200 steps on the 2D battery test),
// with machine-level total-energy conservation and dual-energy closure unaffected
// (the per-stage sync absorbs the mismatch into the partition). Accepted as a
// characterized approximation; revisit only if sub-truncation partition accuracy at
// refinement boundaries becomes a requirement.
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
  bool use_tabular_materials =
      use_material_mixture_ && material_mixture_.UsesTabularEOS();
  auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;

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
        Real eele = fmax(u(m, iele, k, j, i), 0.0);
        if (use_tabular_materials) {
          const Real pe = fmax(thermodynamics(
              m, two_temperature::TwoTemperature::electron_pressure,
              k, j, i), 0.0);
          if (eele > 0.0) eele *= exp(-(pe/eele)*divvd*dt);
        } else {
          eele *= exp(-gm1*divvd*dt);
        }
        Real density = fmax(u(m, IDN, k, j, i), dfloor);
        u(m, iele, k, j, i) = eele;
        w(m, iele, k, j, i) = eele / density;
      });
}

//----------------------------------------------------------------------------------------
//! \brief Add the electron pressure-work source as an explicit stage RHS.
//!
//! The face drift velocities were built by AddFluxes from the same stage-start state.
//! Reading electron energy/pressure from prim (rather than the already blended cons)
//! makes this exactly the additive source required by Heun/SSPRK2.

void BiermannBattery::AddElectronWorkRHS(
    Real dt, DvceArray5D<Real> &cons, const DvceArray5D<Real> &prim) {
  if (coefficient == 0.0 || dt == 0.0)
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
  bool use_tabular_materials =
      use_material_mixture_ && material_mixture_.UsesTabularEOS();
  auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;

  par_for(
      "biermann_electron_work_rhs", DevExeSpace(), 0, nmb1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real divvd =
            (vd1(m, k, j, i + 1) - vd1(m, k, j, i)) / size.d_view(m).dx1;
        if (multi_d) {
          divvd +=
              (vd2(m, k, j + 1, i) - vd2(m, k, j, i)) / size.d_view(m).dx2;
        }
        if (three_d) {
          divvd +=
              (vd3(m, k + 1, j, i) - vd3(m, k, j, i)) / size.d_view(m).dx3;
        }
        Real electron_pressure;
        if (use_tabular_materials) {
          electron_pressure = fmax(thermodynamics(
              m, two_temperature::TwoTemperature::electron_pressure,
              k, j, i), 0.0);
        } else {
          const Real electron_energy = fmax(
              w(m, IDN, k, j, i) * w(m, iele, k, j, i), 0.0);
          electron_pressure = gm1 * electron_energy;
        }
        u(m, iele, k, j, i) -= dt * electron_pressure * divvd;
      });
}

//----------------------------------------------------------------------------------------
//! \brief Add the electron drift and thermal-magnetic characteristic speeds.
//! The latter follows FLASH's v_TM^2=(gamma-1) T_e |grad ln(n_e)|^2/(e^2 n_e)
//! in normalized mu0=k_B=1 units.

void BiermannBattery::NewTimeStep(const DvceArray5D<Real> &prim,
                                  const DvceArray5D<Real> &bcc) {
  dtnew = std::numeric_limits<Real>::max();
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
  bool use_materials = use_material_mixture_;
  auto material_mixture = material_mixture_;
  bool use_tabular_materials =
      use_materials && material_mixture.UsesTabularEOS();
  auto thermodynamics = pmy_pack_->pmhd->ptwo_temp->thermodynamics;
  Real electron_density_cgs_to_code = use_tabular_materials
      ? materials::MaterialMixtureDevice::atomic_mass_unit_cgs/
        material_mixture.density_to_cgs
      : 1.0;
  Real minimum_electron_fraction = minimum_electron_fraction_;
  Real minimum_positive = std::numeric_limits<Real>::min();
  Real dt1 = std::numeric_limits<Real>::max();
  Real dt2 = std::numeric_limits<Real>::max();
  Real dt3 = std::numeric_limits<Real>::max();

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
        Real local_fe = fe;
        Real ne;
        // The centre cell's log(ne) was redeclared once per direction.  It is the same
        // value in all three: hoist it, and let x2/x3 read what x1 already evaluated.
        Real log_ne = 0.0;
        Real electron_activation = 1.0;
        Real dln1;
        if (use_tabular_materials) {
          const BoundedElectronDensity ne_center = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho,
              minimum_electron_fraction, minimum_positive, m, k, j, i);
          const Real rho_p = fmax(w(m, IDN, k, j, i+1), dfloor);
          const Real rho_m = fmax(w(m, IDN, k, j, i-1), dfloor);
          const BoundedElectronDensity ne_p = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_p,
              minimum_electron_fraction, minimum_positive, m, k, j, i+1);
          const BoundedElectronDensity ne_m = CachedElectronDensityCode(
              thermodynamics, electron_density_cgs_to_code, rho_m,
              minimum_electron_fraction, minimum_positive, m, k, j, i-1);
          ne = ne_center.value;
          electron_activation = ne_center.activation;
          log_ne = log(ne);
          dln1 = fmax(fabs(log(ne_p.value)-log_ne),
                      fabs(log_ne-log(ne_m.value)))/dx1;
        } else if (use_materials) {
          const auto mix_y0 = material_mixture.CompositionFromPrimitive(
              w, m, k, j, i);
          local_fe = material_mixture.ElectronHeatCapacityFraction(mix_y0);
          ne = material_mixture.ElectronNumberDensity(rho, mix_y0);
          const Real rho_p = fmax(w(m, IDN, k, j, i+1), dfloor);
          const Real rho_m = fmax(w(m, IDN, k, j, i-1), dfloor);
          const auto mix_p = material_mixture.CompositionFromPrimitive(
              w, m, k, j, i+1);
          const auto mix_m = material_mixture.CompositionFromPrimitive(
              w, m, k, j, i-1);
          const Real ne_p = material_mixture.ElectronNumberDensity(rho_p, mix_p);
          const Real ne_m = material_mixture.ElectronNumberDensity(rho_m, mix_m);
          log_ne = log(ne);
          dln1 = fmax(fabs(log(ne_p)-log_ne),
                      fabs(log_ne-log(ne_m)))/dx1;
        } else {
          ne = fe*rho;
          log_ne = log(ne);
          const Real ne_p = fe*fmax(w(m, IDN, k, j, i+1), dfloor);
          const Real ne_m = fe*fmax(w(m, IDN, k, j, i-1), dfloor);
          dln1 = fmax(fabs(log(ne_p)-log_ne),
                      fabs(log_ne-log(ne_m)))/dx1;
        }
        Real dln2 = 0.0;
        Real dln3 = 0.0;
        if (multi_d) {
          if (use_tabular_materials) {
            const Real rho_p = fmax(w(m, IDN, k, j+1, i), dfloor);
            const Real rho_m = fmax(w(m, IDN, k, j-1, i), dfloor);
            const BoundedElectronDensity ne_p = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_p,
                minimum_electron_fraction, minimum_positive, m, k, j+1, i);
            const BoundedElectronDensity ne_m = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_m,
                minimum_electron_fraction, minimum_positive, m, k, j-1, i);
            dln2 = fmax(fabs(log(ne_p.value)-log_ne),
                        fabs(log_ne-log(ne_m.value)))/dx2;
          } else if (use_materials) {
            const Real rho_p = fmax(w(m, IDN, k, j+1, i), dfloor);
            const Real rho_m = fmax(w(m, IDN, k, j-1, i), dfloor);
            const auto mix_p = material_mixture.CompositionFromPrimitive(
                w, m, k, j+1, i);
            const auto mix_m = material_mixture.CompositionFromPrimitive(
                w, m, k, j-1, i);
            const Real ne_p = material_mixture.ElectronNumberDensity(rho_p, mix_p);
            const Real ne_m = material_mixture.ElectronNumberDensity(rho_m, mix_m);
            dln2 = fmax(fabs(log(ne_p)-log_ne),
                        fabs(log_ne-log(ne_m)))/dx2;
          } else {
            const Real ne_p = fe*fmax(w(m, IDN, k, j+1, i), dfloor);
            const Real ne_m = fe*fmax(w(m, IDN, k, j-1, i), dfloor);
            dln2 = fmax(fabs(log(ne_p)-log_ne),
                        fabs(log_ne-log(ne_m)))/dx2;
          }
        }
        if (three_d) {
          if (use_tabular_materials) {
            const Real rho_p = fmax(w(m, IDN, k+1, j, i), dfloor);
            const Real rho_m = fmax(w(m, IDN, k-1, j, i), dfloor);
            const BoundedElectronDensity ne_p = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_p,
                minimum_electron_fraction, minimum_positive, m, k+1, j, i);
            const BoundedElectronDensity ne_m = CachedElectronDensityCode(
                thermodynamics, electron_density_cgs_to_code, rho_m,
                minimum_electron_fraction, minimum_positive, m, k-1, j, i);
            dln3 = fmax(fabs(log(ne_p.value)-log_ne),
                        fabs(log_ne-log(ne_m.value)))/dx3;
          } else if (use_materials) {
            const Real rho_p = fmax(w(m, IDN, k+1, j, i), dfloor);
            const Real rho_m = fmax(w(m, IDN, k-1, j, i), dfloor);
            const auto mix_p = material_mixture.CompositionFromPrimitive(
                w, m, k+1, j, i);
            const auto mix_m = material_mixture.CompositionFromPrimitive(
                w, m, k-1, j, i);
            const Real ne_p = material_mixture.ElectronNumberDensity(rho_p, mix_p);
            const Real ne_m = material_mixture.ElectronNumberDensity(rho_m, mix_m);
            dln3 = fmax(fabs(log(ne_p)-log_ne),
                        fabs(log_ne-log(ne_m)))/dx3;
          } else {
            const Real ne_p = fe*fmax(w(m, IDN, k+1, j, i), dfloor);
            const Real ne_m = fe*fmax(w(m, IDN, k-1, j, i), dfloor);
            dln3 = fmax(fabs(log(ne_p)-log_ne),
                        fabs(log_ne-log(ne_m)))/dx3;
          }
        }

        // Centered two-cell differences vanish for a Nyquist checkerboard, exactly
        // where the explicit drift update needs its tightest limit.  Bound each curl
        // contribution by the larger adjacent one-sided slope instead.  The triangle
        // inequality deliberately gives a conservative speed when curl terms cancel.
        const Real dbz_dx = OneSidedSlopeBound(b(m, IBZ, k, j, i-1),
                                               b(m, IBZ, k, j, i),
                                               b(m, IBZ, k, j, i+1), dx1);
        const Real dby_dx = OneSidedSlopeBound(b(m, IBY, k, j, i-1),
                                               b(m, IBY, k, j, i),
                                               b(m, IBY, k, j, i+1), dx1);
        Real j1_bound = 0.0;
        Real j2_bound = dbz_dx;
        Real j3_bound = dby_dx;
        if (multi_d) {
          j1_bound += OneSidedSlopeBound(b(m, IBZ, k, j-1, i),
                                         b(m, IBZ, k, j, i),
                                         b(m, IBZ, k, j+1, i), dx2);
          j3_bound += OneSidedSlopeBound(b(m, IBX, k, j-1, i),
                                         b(m, IBX, k, j, i),
                                         b(m, IBX, k, j+1, i), dx2);
        }
        if (three_d) {
          j1_bound += OneSidedSlopeBound(b(m, IBY, k-1, j, i),
                                         b(m, IBY, k, j, i),
                                         b(m, IBY, k+1, j, i), dx3);
          j2_bound += OneSidedSlopeBound(b(m, IBX, k-1, j, i),
                                         b(m, IBX, k, j, i),
                                         b(m, IBX, k+1, j, i), dx3);
        }

        Real gradln = sqrt(SQR(dln1) + SQR(dln2) + SQR(dln3));
        Real vtm;
        if (use_tabular_materials) {
          const Real pe = fmax(thermodynamics(
              m, two_temperature::TwoTemperature::electron_pressure,
              k, j, i), 0.0);
          vtm = (electron_activation > 0.0 && pe > 0.0)
              ? coeff*electron_activation*(sqrt(gm1*pe)/ne)*gradln
              : 0.0;
        } else {
          Real tele = gm1*fmax(w(m, iele, k, j, i), 0.0)/local_fe;
          vtm = coeff*sqrt(gm1*tele/ne)*gradln;
        }
        Real vd1;
        Real vd2;
        Real vd3;
        if (use_tabular_materials) {
          vd1 = coeff*electron_activation*j1_bound/ne;
          vd2 = coeff*electron_activation*j2_bound/ne;
          vd3 = coeff*electron_activation*j3_bound/ne;
        } else {
          vd1 = coeff*j1_bound/ne;
          vd2 = coeff*j2_bound/ne;
          vd3 = coeff*j3_bound/ne;
        }
        Real speed = vd1 + vtm;
        if (speed > 0.0)
          min_dt1 = fmin(min_dt1, dx1 / speed);
        if (multi_d) {
          speed = vd2 + vtm;
          if (speed > 0.0)
            min_dt2 = fmin(min_dt2, dx2 / speed);
        }
        if (three_d) {
          speed = vd3 + vtm;
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
