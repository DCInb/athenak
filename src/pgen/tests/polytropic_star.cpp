//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file polytropic_star.cpp
//! \brief Problem generator for collapse of a Lane-Emden polytropic sphere with AMR.
//!
//! Follows the same structure as be_collapse.cpp: self-gravity via multigrid Poisson,
//! Jeans-style user AMR criterion, and zero initial velocity field. The only physics
//! difference is the initial density profile: here it is a numerically solved
//! Lane-Emden polytrope with index n and finite surface at the first zero xi_1.

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
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

namespace {

// AMR parameter (set from input in pgen, read in refinement function)
Real njeans_threshold;
Real iso_cs_global;
Real poly_n_global;
Real poly_k_global;
Real rho_floor_global;
bool eos_is_ideal_global;

struct LaneEmdenProfile {
  std::vector<Real> xi;
  std::vector<Real> theta;
  std::vector<Real> dtheta;
  Real xi1;
};

inline Real ThetaPow(const Real theta, const Real n) {
  return (theta > 0.0) ? std::pow(theta, n) : 0.0;
}

inline void LaneEmdenRHS(const Real xi, const Real theta, const Real dtheta,
                         const Real n, Real &dtheta_dxi, Real &ddtheta_dxi) {
  dtheta_dxi = dtheta;
  // Start integration away from xi=0, so this guard is only for safety.
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
  // Regular center expansion of Lane-Emden solution.
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
        profile.xi.push_back(profile.xi1);
        profile.theta.push_back(0.0);
        profile.dtheta.push_back(
            CubicHermiteSlope(theta, dtheta, theta_next, dtheta_next, h, troot));
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
                                 theta_tab(hi), dtheta_tab(hi),
                                 x1 - x0, frac);
  return Kokkos::fmax(theta, 0.0);
}

}  // namespace

// Forward declaration
void PolyStarRefinement(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::PolytropicStar()
//! \brief Sets up a Lane-Emden polytropic sphere for gravitational collapse with Jeans AMR.

void ProblemGenerator::PolytropicStar(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // --- gravity coupling ---
  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
    pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
    if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  // --- problem parameters ---
  Real r_star = pin->GetOrAddReal("problem", "star_radius", 0.5);
  Real rho_central = pin->GetOrAddReal("problem", "rho_central", 1.0);
  Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-8);
  Real poly_n = pin->GetOrAddReal("problem", "poly_n", 1.0);
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);  // m=2 perturbation amplitude
  Real x_center = pin->GetOrAddReal("problem", "x_center", 0.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  if (poly_n <= 0.0 || poly_n >= 5.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/poly_n must satisfy 0 < n < 5 for a finite-radius "
              << "Lane-Emden sphere." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (r_star <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/star_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rho_central <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "problem/rho_central must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (four_pi_G <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "gravity/four_pi_G must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  LaneEmdenProfile lane_profile;
  if (!SolveLaneEmden(poly_n, lane_profile)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::PolytropicStar" << std::endl
              << "Failed to solve Lane-Emden equation for n=" << poly_n << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real lane_xi1 = lane_profile.xi1;
  Real a_scale = r_star / lane_xi1;
  Real poly_k = (four_pi_G / (poly_n + 1.0)) * SQR(a_scale)
              * std::pow(rho_central, 1.0 - 1.0/poly_n);
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

  // --- AMR Jeans criterion ---
  njeans_threshold = pin->GetOrAddReal("problem", "njeans", 16.0);

  // Determine sound speed for Jeans criterion.
  // - isothermal: use EOS iso sound speed (same method as be_collapse)
  // - ideal: use local polytropic sound speed from P=K rho^(1+1/n)
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro != nullptr) {
    iso_cs_global = pmbp->phydro->peos->eos_data.iso_cs;
    eos_is_ideal_global = pmbp->phydro->peos->eos_data.is_ideal;
  } else {
    iso_cs_global = 1.0;
    eos_is_ideal_global = false;
  }
  poly_n_global = poly_n;
  poly_k_global = poly_k;
  rho_floor_global = rho_floor;

  if (global_variable::my_rank == 0 && pmbp->phydro != nullptr && eos_is_ideal_global) {
    Real gamma_target = 1.0 + 1.0/poly_n;
    Real gamma_eos = pmbp->phydro->peos->eos_data.gamma;
    if (std::abs(gamma_eos - gamma_target) > 1.0e-8) {
      std::cout << "### WARNING in ProblemGenerator::PolytropicStar" << std::endl
                << "For consistency with poly_n=" << poly_n
                << ", set hydro/gamma = 1 + 1/n = " << gamma_target
                << " (current gamma = " << gamma_eos << ")." << std::endl;
    }
  }

  // Register Jeans refinement condition
  user_ref_func = PolyStarRefinement;

  // --- initialize density ---
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->phydro == nullptr) return;
  auto &u0 = pmbp->phydro->u0;
  int nmb = pmbp->nmb_thispack;
  bool eos_is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  Real gamma_eos = pmbp->phydro->peos->eos_data.gamma;
  Real gamma_poly = 1.0 + 1.0/poly_n;
  Real xi_scale = lane_xi1 / r_star;

  par_for("polytropic_star_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
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

    Real rho = rho_floor + rho_central * Kokkos::pow(theta, poly_n);
    if (amp > 0.0 && r < r_star) {
      rho *= (1.0 + amp * (r * r) / (r_star * r_star)
              * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
    }

    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (eos_is_ideal) {
      Real pgas = poly_k * Kokkos::pow(rho, gamma_poly);
      u0(m, IEN, k, j, i) = pgas / (gamma_eos - 1.0);
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- Polytropic Star ---" << std::endl
      << "rho_central            = " << rho_central << std::endl
      << "star_radius            = " << r_star << std::endl
      << "rho_floor              = " << rho_floor << std::endl
      << "Perturbation amplitude = " << amp << std::endl
      << "poly_n                 = " << poly_n << std::endl
      << "poly_gamma             = " << gamma_poly << std::endl
      << "Lane-Emden xi1         = " << lane_xi1 << std::endl
      << "poly_K (derived)       = " << poly_k << std::endl
      << "Jeans AMR threshold    = " << njeans_threshold << std::endl
      << "four_pi_G              = " << four_pi_G << std::endl
      << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PolyStarRefinement()
//! \brief Jeans-length AMR criterion for self-gravitating gas.
//!
//! For each meshblock, computes the minimum Jeans number:
//!   nJ = cs / sqrt(rho_max) * (2*pi / dx)
//! and sets the refinement flag accordingly.

void PolyStarRefinement(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  int ng = indcs.ng;
  const int nkji = (nx3 + 2 * ng) * (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int nji  = (nx2 + 2 * ng) * (nx1 + 2 * ng);
  const int ni   = (nx1 + 2 * ng);
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real cs_iso = iso_cs_global;
  Real poly_n = poly_n_global;
  Real poly_k = poly_k_global;
  Real rho_floor = rho_floor_global;
  bool eos_is_ideal = eos_is_ideal_global;
  Real gamma_poly = 1.0 + 1.0/poly_n;
  Real njeans = njeans_threshold;

  par_for_outer("PolyStarAMR", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_rhomax;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [&](const int idx, Real &rhomax) {
        int k = idx / nji;
        int j = (idx - k * nji) / ni;
        int i = (idx - k * nji - j * ni);
        rhomax = Kokkos::fmax(u0(m, IDN, k, j, i), rhomax);
      },
      Kokkos::Max<Real>(team_rhomax));

    Real rho_eval = Kokkos::fmax(team_rhomax, rho_floor);
    Real cs = cs_iso;
    if (eos_is_ideal) {
      cs = Kokkos::sqrt(gamma_poly * poly_k * Kokkos::pow(rho_eval, 1.0/poly_n));
    }
    Real dx = size.d_view(m).dx1;
    Real nj_min = cs / Kokkos::sqrt(rho_eval) * (2.0 * M_PI / dx);

    if (nj_min < njeans) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (nj_min > njeans * 2.5) {
      refine_flag.d_view(m + mbs) = -1;
    } else {
      refine_flag.d_view(m + mbs) = 0;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}
