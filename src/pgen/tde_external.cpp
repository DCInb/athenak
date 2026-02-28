//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_external.cpp
//! \brief Problem generator for a Lane-Emden star in an external SMBH potential.
//!
//! The stellar profile is a Lane-Emden polytrope with fixed total stellar mass M*=1.
//! An external Newtonian BH gravity is added through a user source term:
//!   Phi_BH = -G M_BH / sqrt(|r-r_BH|^2 + eps^2).
//! Here G is inferred from gravity/four_pi_G as G = four_pi_G/(4*pi), consistent with
//! the unit system used by self-gravity setups in this code.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

namespace {

// External BH gravity source-term parameters
Real bh_mass_global;
Real bh_x_global, bh_y_global, bh_z_global;
Real bh_soft_global;
Real newton_g_global;

// Time-dependent BH orbit in star-centered coordinates.
bool bh_orbit_enabled_global;
Real orbit_center_x_global, orbit_center_y_global, orbit_center_z_global;
Real frame_vx_global, frame_vy_global, frame_vz_global;
Real orbit_mu_global;
Real orbit_r0x_global, orbit_r0y_global, orbit_r0z_global;
Real orbit_v0x_global, orbit_v0y_global, orbit_v0z_global;
Real orbit_ecc_global, orbit_rp_global;
Real orbit_beta_global, orbit_mass_ratio_global;
Real orbit_theta_deg_global, orbit_sep_initial_global;
bool orbit_provide_params_global;

struct LaneEmdenProfile {
  std::vector<Real> xi;
  std::vector<Real> theta;
  std::vector<Real> dtheta;
  Real xi1;
  Real dtheta_xi1;
};

inline Real ThetaPow(const Real theta, const Real n) {
  return (theta > 0.0) ? std::pow(theta, n) : 0.0;
}

inline Real DegToRad(const Real deg) {
  return deg * (M_PI/180.0);
}

inline void FrameCenterAtTime(const Real t, Real &cx, Real &cy, Real &cz) {
  cx = orbit_center_x_global + frame_vx_global*t;
  cy = orbit_center_y_global + frame_vy_global*t;
  cz = orbit_center_z_global + frame_vz_global*t;
}

void RotateAboutY(Real &x, Real &y, Real &z, const Real angle) {
  Real c = std::cos(angle);
  Real s = std::sin(angle);
  Real x_new = c*x + s*z;
  Real z_new = -s*x + c*z;
  x = x_new;
  z = z_new;
}

inline Real Norm3(const Real x, const Real y, const Real z) {
  return std::sqrt(x*x + y*y + z*z);
}

void Stumpff(const Real z, Real &c2, Real &c3) {
  if (z > 1.0e-8) {
    Real sz = std::sqrt(z);
    c2 = (1.0 - std::cos(sz)) / z;
    c3 = (sz - std::sin(sz)) / (sz*sz*sz);
  } else if (z < -1.0e-8) {
    Real sz = std::sqrt(-z);
    c2 = (std::cosh(sz) - 1.0) / (-z);
    c3 = (std::sinh(sz) - sz) / (sz*sz*sz);
  } else {
    Real z2 = z*z;
    Real z3 = z2*z;
    c2 = 0.5 - z/24.0 + z2/720.0 - z3/40320.0;
    c3 = 1.0/6.0 - z/120.0 + z2/5040.0 - z3/362880.0;
  }
}

bool PropagateTwoBodyUniversal(const Real dt, const Real mu,
                               const Real r0x, const Real r0y, const Real r0z,
                               const Real v0x, const Real v0y, const Real v0z,
                               Real &rx, Real &ry, Real &rz,
                               Real &vx, Real &vy, Real &vz) {
  if (mu <= 0.0) return false;

  Real r0 = Norm3(r0x, r0y, r0z);
  if (r0 <= 0.0) return false;

  if (dt == 0.0) {
    rx = r0x;
    ry = r0y;
    rz = r0z;
    vx = v0x;
    vy = v0y;
    vz = v0z;
    return true;
  }

  Real v0sq = v0x*v0x + v0y*v0y + v0z*v0z;
  Real sqrt_mu = std::sqrt(mu);
  Real rv0 = (r0x*v0x + r0y*v0y + r0z*v0z) / r0;
  Real alpha = 2.0/r0 - v0sq/mu;

  Real chi;
  if (std::abs(alpha) > 1.0e-12) {
    chi = sqrt_mu*std::abs(alpha)*dt;
  } else {
    chi = sqrt_mu*dt/r0;
  }
  if (!std::isfinite(chi)) chi = 0.0;

  bool converged = false;
  for (int it = 0; it < 100; ++it) {
    Real z = alpha*chi*chi;
    Real c2, c3;
    Stumpff(z, c2, c3);

    Real chi2 = chi*chi;
    Real chi3 = chi2*chi;
    Real f = (r0*rv0/sqrt_mu)*chi2*c2
           + (1.0 - alpha*r0)*chi3*c3
           + r0*chi - sqrt_mu*dt;
    Real fp = (r0*rv0/sqrt_mu)*chi*(1.0 - z*c3)
            + (1.0 - alpha*r0)*chi2*c2 + r0;
    if (!std::isfinite(f) || !std::isfinite(fp) || std::abs(fp) < 1.0e-14) {
      return false;
    }

    Real dchi = -f/fp;
    chi += dchi;
    if (!std::isfinite(chi)) return false;
    if (std::abs(dchi) < 1.0e-13*std::fmax(1.0, std::abs(chi))) {
      converged = true;
      break;
    }
  }
  if (!converged) return false;

  Real z = alpha*chi*chi;
  Real c2, c3;
  Stumpff(z, c2, c3);
  Real chi2 = chi*chi;
  Real chi3 = chi2*chi;
  Real f = 1.0 - (chi2/r0)*c2;
  Real g = dt - (chi3/sqrt_mu)*c3;

  rx = f*r0x + g*v0x;
  ry = f*r0y + g*v0y;
  rz = f*r0z + g*v0z;

  Real r = Norm3(rx, ry, rz);
  if (r <= 0.0) return false;
  Real fdot = (sqrt_mu/(r*r0))*(alpha*chi3*c3 - chi);
  Real gdot = 1.0 - (chi2/r)*c2;

  vx = fdot*r0x + gdot*v0x;
  vy = fdot*r0y + gdot*v0y;
  vz = fdot*r0z + gdot*v0z;
  return std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz);
}

void ComputeBHOrbitState(const Real t, Real &x, Real &y, Real &z,
                         Real &vx, Real &vy, Real &vz) {
  if (!bh_orbit_enabled_global) {
    // Fixed BH in the star-centered frame: keep initial relative offset constant.
    Real cx, cy, cz;
    FrameCenterAtTime(t, cx, cy, cz);
    x = cx + orbit_r0x_global;
    y = cy + orbit_r0y_global;
    z = cz + orbit_r0z_global;
    vx = frame_vx_global;
    vy = frame_vy_global;
    vz = frame_vz_global;
    return;
  }

  Real rx, ry, rz;
  bool ok = PropagateTwoBodyUniversal(
      t, orbit_mu_global,
      orbit_r0x_global, orbit_r0y_global, orbit_r0z_global,
      orbit_v0x_global, orbit_v0y_global, orbit_v0z_global,
      rx, ry, rz, vx, vy, vz);

  if (!ok) {
    rx = orbit_r0x_global;
    ry = orbit_r0y_global;
    rz = orbit_r0z_global;
    vx = orbit_v0x_global;
    vy = orbit_v0y_global;
    vz = orbit_v0z_global;
  }

  Real cx, cy, cz;
  FrameCenterAtTime(t, cx, cy, cz);
  x = cx + rx;
  y = cy + ry;
  z = cz + rz;
}

inline void LaneEmdenRHS(const Real xi, const Real theta, const Real dtheta,
                         const Real n, Real &dtheta_dxi, Real &ddtheta_dxi) {
  dtheta_dxi = dtheta;
  if (xi <= 0.0) {
    ddtheta_dxi = -1.0/3.0;
  } else {
    ddtheta_dxi = -2.0*dtheta/xi - ThetaPow(theta, n);
  }
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteValue(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  Real t2 = t*t;
  Real t3 = t2*t;
  Real h00 = (2.0*t3 - 3.0*t2 + 1.0);
  Real h10 = (t3 - 2.0*t2 + t);
  Real h01 = (-2.0*t3 + 3.0*t2);
  Real h11 = (t3 - t2);
  return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1;
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteSlope(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  if (h <= 0.0) return m0;
  Real t2 = t*t;
  Real dh00 = (6.0*t2 - 6.0*t);
  Real dh10 = (3.0*t2 - 4.0*t + 1.0);
  Real dh01 = (-6.0*t2 + 6.0*t);
  Real dh11 = (3.0*t2 - 2.0*t);
  Real dtheta_dt = dh00*y0 + dh10*h*m0 + dh01*y1 + dh11*h*m1;
  return dtheta_dt / h;
}

inline void RK4Step(const Real xi, const Real h, const Real theta, const Real dtheta,
                    const Real n, Real &theta_out, Real &dtheta_out) {
  Real k1_t, k1_dt;
  LaneEmdenRHS(xi, theta, dtheta, n, k1_t, k1_dt);

  Real t2 = theta + 0.5*h*k1_t;
  Real dt2 = dtheta + 0.5*h*k1_dt;
  Real k2_t, k2_dt;
  LaneEmdenRHS(xi + 0.5*h, t2, dt2, n, k2_t, k2_dt);

  Real t3 = theta + 0.5*h*k2_t;
  Real dt3 = dtheta + 0.5*h*k2_dt;
  Real k3_t, k3_dt;
  LaneEmdenRHS(xi + 0.5*h, t3, dt3, n, k3_t, k3_dt);

  Real t4 = theta + h*k3_t;
  Real dt4 = dtheta + h*k3_dt;
  Real k4_t, k4_dt;
  LaneEmdenRHS(xi + h, t4, dt4, n, k4_t, k4_dt);

  theta_out = theta + (h/6.0)*(k1_t + 2.0*k2_t + 2.0*k3_t + k4_t);
  dtheta_out = dtheta + (h/6.0)*(k1_dt + 2.0*k2_dt + 2.0*k3_dt + k4_dt);
}

bool SolveLaneEmden(const Real n, LaneEmdenProfile &profile) {
  constexpr Real xi_max = 2.0e3;
  constexpr int max_steps = 5000000;
  constexpr Real atol = 1.0e-12;
  constexpr Real rtol = 1.0e-10;
  constexpr Real safety = 0.9;
  constexpr Real fac_min = 0.2;
  constexpr Real fac_max = 2.0;
  constexpr Real h_min = 1.0e-10;
  constexpr Real h_max = 5.0e-2;
  constexpr Real xi0 = 1.0e-6;

  profile.xi.clear();
  profile.theta.clear();
  profile.dtheta.clear();
  profile.xi.push_back(0.0);
  profile.theta.push_back(1.0);
  profile.dtheta.push_back(0.0);

  Real xi = xi0;
  Real theta = 1.0 - SQR(xi)/6.0 + n*SQR(SQR(xi))/120.0;
  Real dtheta = -xi/3.0 + n*xi*SQR(xi)/30.0;
  profile.xi.push_back(xi);
  profile.theta.push_back(theta);
  profile.dtheta.push_back(dtheta);

  Real h = 1.0e-4;

  for (int step = 0; step < max_steps; ++step) {
    if (xi >= xi_max) break;
    if (h < h_min) h = h_min;
    if (h > h_max) h = h_max;
    if (xi + h > xi_max) h = xi_max - xi;
    if (h <= 0.0) break;

    Real theta_big, dtheta_big;
    RK4Step(xi, h, theta, dtheta, n, theta_big, dtheta_big);

    Real theta_half, dtheta_half;
    RK4Step(xi, 0.5*h, theta, dtheta, n, theta_half, dtheta_half);
    Real theta_half2, dtheta_half2;
    RK4Step(xi + 0.5*h, 0.5*h, theta_half, dtheta_half, n, theta_half2, dtheta_half2);

    if (!std::isfinite(theta_big) || !std::isfinite(dtheta_big) ||
        !std::isfinite(theta_half2) || !std::isfinite(dtheta_half2)) {
      return false;
    }

    Real sc_t = atol + rtol*std::fmax(std::abs(theta_half2), std::abs(theta_big));
    Real sc_dt = atol + rtol*std::fmax(std::abs(dtheta_half2), std::abs(dtheta_big));
    Real err_t = std::abs(theta_half2 - theta_big) / (15.0*sc_t);
    Real err_dt = std::abs(dtheta_half2 - dtheta_big) / (15.0*sc_dt);
    Real err = std::fmax(err_t, err_dt);

    if (err <= 1.0) {
      Real xi_next = xi + h;
      Real theta_next = theta_half2;
      Real dtheta_next = dtheta_half2;

      if (theta_next <= 0.0) {
        Real tlo = 0.0;
        Real thi = 1.0;
        for (int it = 0; it < 80; ++it) {
          Real tm = 0.5*(tlo + thi);
          Real thm = CubicHermiteValue(theta, dtheta, theta_next, dtheta_next, h, tm);
          if (thm > 0.0) {
            tlo = tm;
          } else {
            thi = tm;
          }
        }
        Real troot = 0.5*(tlo + thi);
        profile.xi1 = xi + troot*h;
        profile.dtheta_xi1 =
            CubicHermiteSlope(theta, dtheta, theta_next, dtheta_next, h, troot);
        profile.xi.push_back(profile.xi1);
        profile.theta.push_back(0.0);
        profile.dtheta.push_back(profile.dtheta_xi1);
        return true;
      }

      profile.xi.push_back(xi_next);
      profile.theta.push_back(theta_next);
      profile.dtheta.push_back(dtheta_next);
      xi = xi_next;
      theta = theta_next;
      dtheta = dtheta_next;
    }

    Real fac = fac_max;
    if (err > 1.0e-30) {
      fac = safety*std::pow(1.0/err, 0.2);
      fac = std::fmax(fac_min, std::fmin(fac_max, fac));
    }
    h = std::fmax(h_min, std::fmin(h_max, h*fac));
  }

  return false;
}

KOKKOS_INLINE_FUNCTION
Real LaneEmdenTheta(const Real xi, const DvceArray1D<Real> xi_tab,
                    const DvceArray1D<Real> theta_tab,
                    const DvceArray1D<Real> dtheta_tab, const int npts) {
  if (xi <= 0.0) return 1.0;
  if (xi >= xi_tab(npts - 1)) return 0.0;

  int lo = 0;
  int hi = npts - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (xi_tab(mid) <= xi) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  Real x0 = xi_tab(lo);
  Real x1 = xi_tab(hi);
  if (x1 <= x0) return Kokkos::fmax(theta_tab(lo), 0.0);

  Real frac = (xi - x0) / (x1 - x0);
  Real theta = CubicHermiteValue(theta_tab(lo), dtheta_tab(lo),
                                 theta_tab(hi), dtheta_tab(hi), x1 - x0, frac);
  return Kokkos::fmax(theta, 0.0);
}

void TDEExternalGravitySource(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  bool eos_is_ideal = pmbp->phydro->peos->eos_data.is_ideal;

  Real stage_fraction = 0.0;
  if (pm->dt > 0.0) {
    stage_fraction = bdt/pm->dt;
  }
  if (stage_fraction < 0.0) stage_fraction = 0.0;
  Real t_src = pm->time + stage_fraction * pm->dt;
  Real bhvx = 0.0, bhvy = 0.0, bhvz = 0.0;
  Real bhx, bhy, bhz;
  ComputeBHOrbitState(t_src, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  Real cx, cy, cz;
  FrameCenterAtTime(t_src, cx, cy, cz);

  Real gm = newton_g_global * bh_mass_global;
  Real eps2 = bh_soft_global * bh_soft_global;

  // Non-inertial translational correction for star-centered coordinates:
  // subtract frame acceleration at the moving frame origin.
  Real dx0 = cx - bhx;
  Real dy0 = cy - bhy;
  Real dz0 = cz - bhz;
  Real r20 = dx0*dx0 + dy0*dy0 + dz0*dz0 + eps2;
  r20 = std::fmax(r20, static_cast<Real>(1.0e-30));
  Real invr0 = 1.0 / std::sqrt(r20);
  Real invr30 = invr0*invr0*invr0;
  Real ax_frame = -gm * dx0 * invr30;
  Real ay_frame = -gm * dy0 * invr30;
  Real az_frame = -gm * dz0 * invr30;

  par_for("tde_bh_src", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    Real dx = x - bhx;
    Real dy = y - bhy;
    Real dz = z - bhz;
    Real r2 = dx*dx + dy*dy + dz*dz + eps2;
    r2 = Kokkos::fmax(r2, static_cast<Real>(1.0e-30));
    Real invr = 1.0 / Kokkos::sqrt(r2);
    Real invr3 = invr*invr*invr;

    Real ax = -gm * dx * invr3 - ax_frame;
    Real ay = -gm * dy * invr3 - ay_frame;
    Real az = -gm * dz * invr3 - az_frame;

    Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0) return;

    Real ke_old = 0.0;
    if (eos_is_ideal) {
      Real vx_old = u0(m, IM1, k, j, i) / rho;
      Real vy_old = u0(m, IM2, k, j, i) / rho;
      Real vz_old = u0(m, IM3, k, j, i) / rho;
      ke_old = 0.5 * rho * (vx_old*vx_old + vy_old*vy_old + vz_old*vz_old);
    }

    u0(m, IM1, k, j, i) += bdt * rho * ax;
    u0(m, IM2, k, j, i) += bdt * rho * ay;
    u0(m, IM3, k, j, i) += bdt * rho * az;

    if (eos_is_ideal) {
      Real vx_new = u0(m, IM1, k, j, i) / rho;
      Real vy_new = u0(m, IM2, k, j, i) / rho;
      Real vz_new = u0(m, IM3, k, j, i) / rho;
      Real ke_new = 0.5 * rho * (vx_new*vx_new + vy_new*vy_new + vz_new*vz_new);
      u0(m, IEN, k, j, i) += (ke_new - ke_old);
    }
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Lane-Emden star (M*=1) with external Newtonian BH gravity source term.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Follow disk.cpp behavior: always enable and enroll user source terms here.
  user_srcs = true;
  user_srcs_func = TDEExternalGravitySource;

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (four_pi_G <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "gravity/four_pi_G must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  newton_g_global = four_pi_G / (4.0*M_PI);

  // Star parameters
  constexpr Real star_mass = 1.0;
  Real r_star = pin->GetOrAddReal("problem", "star_radius", 0.5);
  Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-8);
  Real poly_n = pin->GetOrAddReal("problem", "poly_n", 1.5);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);
  Real x_center = pin->GetOrAddReal("problem", "x_center", -1.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  Real vx_star = pin->GetOrAddReal("problem", "vx_star", 0.0);
  Real vy_star = pin->GetOrAddReal("problem", "vy_star", 0.0);
  Real vz_star = pin->GetOrAddReal("problem", "vz_star", 0.0);

  if (poly_n <= 0.0 || poly_n >= 5.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/poly_n must satisfy 0 < n < 5." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (r_star <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/star_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rho_floor < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/rho_floor must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // External BH setup: Phantom-style orbit controls, but in star-centered coordinates.
  bh_soft_global = pin->GetOrAddReal("problem", "bh_softening", 1.0e-2);
  if (bh_soft_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_softening must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bool provide_params = pin->GetOrAddBoolean("problem", "provide_params", false);
  bool orbit_bh_enable = pin->GetOrAddBoolean("problem", "orbit_bh_enable", true);
  Real mass_ratio = pin->GetOrAddReal("problem", "mass_ratio", 1000.0);
  Real beta = pin->GetOrAddReal("problem", "beta", 1.0);
  Real ecc_bh = pin->GetOrAddReal("problem", "ecc_bh", 1.0);
  Real theta_bh_deg = pin->GetOrAddReal("problem", "theta_bh", 0.0);
  Real sep_initial = pin->GetOrAddReal("problem", "sep_initial", 10.0);

  Real x1 = pin->GetOrAddReal("problem", "x1", 0.0);
  Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  Real vx1 = pin->GetOrAddReal("problem", "vx1", 0.0);
  Real vy1 = pin->GetOrAddReal("problem", "vy1", 0.0);
  Real vz1 = pin->GetOrAddReal("problem", "vz1", 0.0);

  if (mass_ratio <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/mass_ratio must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (beta <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/beta must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ecc_bh <= 0.0 || ecc_bh > 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/ecc_bh must satisfy 0 < ecc_bh <= 1." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (sep_initial <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/sep_initial must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_orbit_enabled_global = orbit_bh_enable;
  bh_mass_global = mass_ratio * star_mass;
  orbit_center_x_global = x_center;
  orbit_center_y_global = y_center;
  orbit_center_z_global = z_center;
  frame_vx_global = vx_star;
  frame_vy_global = vy_star;
  frame_vz_global = vz_star;
  orbit_mass_ratio_global = mass_ratio;
  orbit_beta_global = beta;
  orbit_ecc_global = ecc_bh;
  orbit_theta_deg_global = theta_bh_deg;
  orbit_sep_initial_global = sep_initial;
  orbit_provide_params_global = provide_params;

  Real r_tidal = r_star * std::cbrt(mass_ratio);
  Real r_peri = r_tidal / beta;
  Real mu_orbit = newton_g_global * (star_mass + bh_mass_global);
  orbit_rp_global = r_peri;
  orbit_mu_global = mu_orbit;

  Real star_x = 0.0, star_y = 0.0, star_z = 0.0;
  Real star_vx = 0.0, star_vy = 0.0, star_vz = 0.0;

  if (provide_params) {
    // Same parameter names as Phantom setup: user gives the star state relative to BH.
    // Star-centered frame here uses the opposite sign for the BH state.
    star_x = x1;
    star_y = y1;
    star_z = z1;
    star_vx = vx1;
    star_vy = vy1;
    star_vz = vz1;
  } else {
    Real theta_bh = DegToRad(theta_bh_deg);
    if (ecc_bh < (1.0 - 1.0e-12)) {
      Real semia = r_peri/(1.0 - ecc_bh);
      Real r_apo = semia*(1.0 + ecc_bh);
      Real v_apo = std::sqrt(mu_orbit*(2.0/r_apo - 1.0/semia));

      // Start at apoapsis with pericenter along +y, then apply inclination.
      star_x = 0.0;
      star_y = -r_apo;
      star_z = 0.0;
      star_vx = v_apo;
      star_vy = 0.0;
      star_vz = 0.0;
      RotateAboutY(star_x, star_y, star_z, -theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, -theta_bh);
    } else {
      // Parabolic branch follows the same construction used in Phantom's TDE setup.
      Real r0 = sep_initial * r_tidal;
      Real y0 = -2.0*r_peri + r0;
      Real x2 = r0*r0 - y0*y0;
      if (x2 < -1.0e-12*std::fmax(1.0, r0*r0)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "problem/sep_initial is too small for the chosen beta." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real x0 = std::sqrt(std::fmax(x2, 0.0));
      Real denom = std::sqrt(4.0*r_peri*r_peri + x0*x0);
      if (denom <= 0.0 || r0 <= 0.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "Invalid parabolic initial condition geometry." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real vel = std::sqrt(2.0*mu_orbit/r0);

      star_x = -x0;
      star_y = y0;
      star_z = 0.0;
      star_vx = vel*(2.0*r_peri/denom);
      star_vy = vel*(-x0/denom);
      star_vz = 0.0;
      RotateAboutY(star_x, star_y, star_z, theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, theta_bh);
    }
  }

  orbit_r0x_global = -star_x;
  orbit_r0y_global = -star_y;
  orbit_r0z_global = -star_z;
  orbit_v0x_global = -star_vx;
  orbit_v0y_global = -star_vy;
  orbit_v0z_global = -star_vz;

  if (!std::isfinite(orbit_r0x_global) || !std::isfinite(orbit_r0y_global) ||
      !std::isfinite(orbit_r0z_global) || !std::isfinite(orbit_v0x_global) ||
      !std::isfinite(orbit_v0y_global) || !std::isfinite(orbit_v0z_global)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Non-finite BH orbit initial state." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bh_x_global = orbit_center_x_global + orbit_r0x_global;
  bh_y_global = orbit_center_y_global + orbit_r0y_global;
  bh_z_global = orbit_center_z_global + orbit_r0z_global;

  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires a <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pmbp->phydro->peos->eos_data.is_ideal)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires hydro/eos = ideal." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  LaneEmdenProfile lane_profile;
  if (!SolveLaneEmden(poly_n, lane_profile)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Failed to solve Lane-Emden equation for n=" << poly_n << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real xi1 = lane_profile.xi1;
  Real qn = -SQR(xi1) * lane_profile.dtheta_xi1;
  if (qn <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Lane-Emden mass factor is non-positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real a_scale = r_star / xi1;
  Real rho_central = star_mass / (4.0*M_PI*std::pow(a_scale, 3.0)*qn);
  Real poly_k = (four_pi_G / (poly_n + 1.0)) * SQR(a_scale)
              * std::pow(rho_central, 1.0 - 1.0/poly_n);
  Real gamma_poly = 1.0 + 1.0/poly_n;

  int lane_npts = static_cast<int>(lane_profile.xi.size());
  DvceArray1D<Real> lane_xi_d("lane_xi", lane_npts);
  DvceArray1D<Real> lane_theta_d("lane_theta", lane_npts);
  DvceArray1D<Real> lane_dtheta_d("lane_dtheta", lane_npts);
  HostArray1D<Real> lane_xi_h = Kokkos::create_mirror_view(lane_xi_d);
  HostArray1D<Real> lane_theta_h = Kokkos::create_mirror_view(lane_theta_d);
  HostArray1D<Real> lane_dtheta_h = Kokkos::create_mirror_view(lane_dtheta_d);
  for (int n = 0; n < lane_npts; ++n) {
    lane_xi_h(n) = lane_profile.xi[n];
    lane_theta_h(n) = lane_profile.theta[n];
    lane_dtheta_h(n) = lane_profile.dtheta[n];
  }
  Kokkos::deep_copy(lane_xi_d, lane_xi_h);
  Kokkos::deep_copy(lane_theta_d, lane_theta_h);
  Kokkos::deep_copy(lane_dtheta_d, lane_dtheta_h);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  bool eos_is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  Real gamma_eos = pmbp->phydro->peos->eos_data.gamma;
  if (gamma_eos <= 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "hydro/gamma must be > 1 for ideal EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real xi_scale = xi1 / r_star;
  int nmb = pmbp->nmb_thispack;

  if (eos_is_ideal && global_variable::my_rank == 0) {
    if (std::abs(gamma_eos - gamma_poly) > 1.0e-8) {
      std::cout << "### WARNING in ProblemGenerator::UserProblem" << std::endl
                << "For consistent polytropic initialization, set hydro/gamma = "
                << gamma_poly << " (1 + 1/n)." << std::endl;
    }
  }

  par_for("tde_external_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center) + SQR(z - z_center));
    Real theta = 0.0;
    if (r < r_star) {
      Real xi = r * xi_scale;
      theta = LaneEmdenTheta(xi, lane_xi_d, lane_theta_d, lane_dtheta_d, lane_npts);
    }

    Real rho_star = rho_central * Kokkos::pow(theta, poly_n);
    if (amp > 0.0 && r < r_star) {
      rho_star *= (1.0 + amp * (r * r) / (r_star * r_star)
                   * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
    }

    Real rho = rho_floor + rho_star;
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = rho * vx_star;
    u0(m, IM2, k, j, i) = rho * vy_star;
    u0(m, IM3, k, j, i) = rho * vz_star;

    if (eos_is_ideal) {
      Real rho_eos = Kokkos::fmax(rho, rho_floor);
      Real pgas = poly_k * Kokkos::pow(rho_eos, gamma_poly);
      Real ekin = 0.5 * rho * (vx_star*vx_star + vy_star*vy_star + vz_star*vz_star);
      u0(m, IEN, k, j, i) = pgas / (gamma_eos - 1.0) + ekin;
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- TDE External Potential ---" << std::endl
      << "star_mass (fixed)      = " << star_mass << std::endl
      << "star_radius            = " << r_star << std::endl
      << "rho_central (derived)  = " << rho_central << std::endl
      << "rho_floor              = " << rho_floor << std::endl
      << "poly_n                 = " << poly_n << std::endl
      << "poly_gamma             = " << gamma_poly << std::endl
      << "poly_K (derived)       = " << poly_k << std::endl
      << "Lane-Emden xi1         = " << xi1 << std::endl
      << "BH mass                = " << bh_mass_global << std::endl
      << "BH orbit enabled       = " << (bh_orbit_enabled_global ? "true" : "false")
      << std::endl
      << "BH position            = (" << bh_x_global << ", "
      << bh_y_global << ", " << bh_z_global << ")" << std::endl
      << "BH frame center        = (" << orbit_center_x_global << ", "
      << orbit_center_y_global << ", " << orbit_center_z_global << ")" << std::endl
      << "Frame velocity         = (" << frame_vx_global << ", "
      << frame_vy_global << ", " << frame_vz_global << ")" << std::endl
      << "BH frame note          = star-centered coordinates" << std::endl
      << "BH softening           = " << bh_soft_global << std::endl
      << "orbit_bh_enable        = "
      << (bh_orbit_enabled_global ? "true" : "false") << std::endl
      << "Newtonian G            = " << newton_g_global << std::endl
      << "four_pi_G              = " << four_pi_G << std::endl
      << std::endl;
    if (bh_orbit_enabled_global) {
      std::cout
        << "provide_params         = "
        << (orbit_provide_params_global ? "true" : "false") << std::endl
        << "mass_ratio             = " << orbit_mass_ratio_global << std::endl
        << "beta                   = " << orbit_beta_global << std::endl
        << "ecc_bh                 = " << orbit_ecc_global << std::endl
        << "theta_bh (deg)         = " << orbit_theta_deg_global << std::endl
        << "sep_initial            = " << orbit_sep_initial_global << std::endl
        << "r_tidal                = " << r_tidal << std::endl
        << "r_peri                 = " << orbit_rp_global << std::endl
        << "BH init rel pos        = (" << orbit_r0x_global << ", "
        << orbit_r0y_global << ", " << orbit_r0z_global << ")" << std::endl
        << "BH init rel vel        = (" << orbit_v0x_global << ", "
        << orbit_v0y_global << ", " << orbit_v0z_global << ")" << std::endl
        << std::endl;
    }
  }
}
