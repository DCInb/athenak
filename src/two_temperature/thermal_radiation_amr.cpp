//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_amr.cpp
//! \brief Conservative composite-mesh operator for implicit radiation diffusion.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/thermal_radiation_halo.hpp"

namespace two_temperature {
namespace {

[[noreturn]] void ImplicitAMRError(const std::string &message) {
  if (global_variable::my_rank == 0) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << std::endl
              << message << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
bool IsPhysicalAMRBoundary(const BoundaryFlag flag) {
  return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
         flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef;
}

KOKKOS_INLINE_FUNCTION
Real AMRDiagonalWeight(const bool physical_face, const int boundary_type) {
  if (!physical_face) return 1.0;
  if (boundary_type == 0) return 0.0;
  if (boundary_type == 1) return 2.0;
  return 1.0;
}

// Sum one aggregate boundary from the refluxed positive face graph. Internal
// fine faces cancel in P^T A P; retaining only boundary faces avoids subtracting
// large, nearly equal diagonal and off-diagonal sums in optically thin cells.
KOKKOS_INLINE_FUNCTION
Real AMRAggregateFaceSum(const BandView5D<Real> &faces, const int m,
                        const int dir, const bool upper, const int i0,
                        const int j0, const int k0, const int ni,
                        const int nj, const int nk) {
  const int na = dir == 0 ? nj : ni;
  const int nb = dir == 2 ? nj : nk;
  Real sum = 0.0;
  for (int b = 0; b < nb; ++b) {
    for (int a = 0; a < na; ++a) {
      const int i = i0+(dir == 0 ? (upper ? ni : 0) : a);
      const int j = j0+(dir == 1 ? (upper ? nj : 0) : (dir == 0 ? a : b));
      const int k = k0+(dir == 2 ? (upper ? nk : 0) : b);
      sum += faces(m, 0, k, j, i);
    }
  }
  return sum;
}

// Coarse cells are twice as wide as fine cells.  Their two half-cell resistances
// are added in series; use a scaled harmonic mean to avoid division by zero or
// overflow when either frozen diffusion coefficient is very small.
KOKKOS_INLINE_FUNCTION
Real CoarseFineConductance(const Real left, const Real right, const Real dx,
                           const bool coarse_left) {
  if (!(left > 0.0) || !(right > 0.0)) return 0.0;
  const Real scale = fmin(left, right);
  const Real left_distance = coarse_left ? 1.0 : 0.5;
  const Real right_distance = coarse_left ? 0.5 : 1.0;
  return (scale/dx)/(left_distance*(scale/left)+right_distance*(scale/right));
}

// A coarse cell average restricted to either transverse half has weights
// [-s/8, 1, s/8], where s is the child's sign.  The second-moment correction
// vanishes: (H/4)^2+(H/2)^2/12-H^2/12=0.  Taking the tensor product also
// reproduces mixed transverse quadratics in three dimensions.  Keep the
// normal average over the full coarse cell; the normal derivative below is
// derived for cell averages, rather than point samples at cell centers.
// Evaluate only the correction to the central cell, using donor differences.
// This is the same tensor stencil but avoids cancellation of a common energy
// offset, which a large frozen diffusion coefficient would otherwise amplify.
KOKKOS_INLINE_FUNCTION
Real CoarseSubfaceCorrection(const DvceArray5D<Real> &coarse, const int m,
                            const int ck, const int cj, const int ci, const int dir,
                            const int sk, const int sj, const int si,
                            const bool multi_d, const bool three_d) {
  const bool ti = dir != 0;
  const bool tj = multi_d && dir != 1;
  const bool tk = three_d && dir != 2;
  Real correction = 0.0;
  if (ti) {
    correction += 0.125*si*(coarse(m, 0, ck, cj, ci+1)-
                            coarse(m, 0, ck, cj, ci-1));
  }
  if (tj) {
    correction += 0.125*sj*(coarse(m, 0, ck, cj+1, ci)-
                            coarse(m, 0, ck, cj-1, ci));
  }
  if (tk) {
    correction += 0.125*sk*(coarse(m, 0, ck+1, cj, ci)-
                            coarse(m, 0, ck-1, cj, ci));
  }
  if (ti && tj) {
    correction += 0.015625*si*sj*
        ((coarse(m, 0, ck, cj+1, ci+1)-coarse(m, 0, ck, cj+1, ci-1))-
         (coarse(m, 0, ck, cj-1, ci+1)-coarse(m, 0, ck, cj-1, ci-1)));
  }
  if (ti && tk) {
    correction += 0.015625*si*sk*
        ((coarse(m, 0, ck+1, cj, ci+1)-coarse(m, 0, ck+1, cj, ci-1))-
         (coarse(m, 0, ck-1, cj, ci+1)-coarse(m, 0, ck-1, cj, ci-1)));
  }
  if (tj && tk) {
    correction += 0.015625*sj*sk*
        ((coarse(m, 0, ck+1, cj+1, ci)-coarse(m, 0, ck+1, cj-1, ci))-
         (coarse(m, 0, ck-1, cj+1, ci)-coarse(m, 0, ck-1, cj-1, ci)));
  }
  return correction;
}

KOKKOS_INLINE_FUNCTION
Real CoarseSubfaceTrace(const DvceArray5D<Real> &coarse, const int m,
                       const int ck, const int cj, const int ci, const int dir,
                       const int sk, const int sj, const int si,
                       const bool multi_d, const bool three_d,
                       const bool positive, const bool absolute_values = false) {
  const int ni = dir == 0 ? 0 : 1;
  const int nj = multi_d && dir != 1 ? 1 : 0;
  const int nk = three_d && dir != 2 ? 1 : 0;
  const Real center = coarse(m, 0, ck, cj, ci);
  if (positive && !(center > 0.0)) return 0.0;
  Real trace = absolute_values ? 0.0 : center+CoarseSubfaceCorrection(
      coarse, m, ck, cj, ci, dir, sk, sj, si, multi_d, three_d);
  if (!positive && !absolute_values) return trace;
  Real minimum = center, maximum = center;
  for (int dk = -nk; dk <= nk; ++dk) {
    const Real wk = dk == 0 ? 1.0 : 0.125*dk*sk;
    for (int dj = -nj; dj <= nj; ++dj) {
      const Real wj = dj == 0 ? 1.0 : 0.125*dj*sj;
      for (int di = -ni; di <= ni; ++di) {
        const Real wi = di == 0 ? 1.0 : 0.125*di*si;
        const Real value = coarse(m, 0, ck+dk, cj+dj, ci+di);
        if (absolute_values) trace += fabs(wi*wj*wk*value);
        minimum = fmin(minimum, value);
        maximum = fmax(maximum, value);
      }
    }
  }
  // Limiting the frozen coefficient, unlike limiting a Krylov vector, leaves
  // the matrix application linear.  Smooth extrema change only by O(H^2).
  return positive ? fmax(0.0, fmin(maximum, fmax(minimum, trace))) : trace;
}

// The fine pair and coarse cell are both width-2h slabs.  Their harmonic
// resistance is second order for smooth K and gives the exact flux for a
// continuous, piecewise-linear energy across a piecewise-constant K jump.
KOKKOS_INLINE_FUNCTION
Real QuadraticCoarseFineConductance(const Real coarse, const Real fine0,
                                   const Real fine1, const Real dx) {
  if (!(coarse > 0.0) || !(fine0 > 0.0) || !(fine1 > 0.0)) return 0.0;
  const Real scale = fmin(coarse, fmin(fine0, fine1));
  return (scale/dx)/(scale/coarse+0.5*(scale/fine0+scale/fine1));
}

}  // namespace

//----------------------------------------------------------------------------------------
//! Freeze face conductances and a positive Jacobi preconditioner for one group.
//! The low-order diagonal is exact.  In the quadratic case it approximates the
//! diagonal using positive face conductances, without transverse signed weights.

void ThermalRadiation::BuildImplicitAMROperator(const Real dt) {
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  const bool high_order = implicit_amr_high_order_;
  const int cis = indcs.cis, cie = indcs.cie;
  const int cjs = indcs.cjs, cje = indcs.cje;
  const int cks = indcs.cks, cke = indcs.cke;
  auto coefficient = implicit_coefficient_;
  auto coarse = implicit_coarse_scratch_;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto neighbors = pmy_pack_->pmb->nghbr.d_view;
  auto levels = pmy_pack_->pmb->mb_lev.d_view;
  auto boundaries = pmy_pack_->pmb->mb_bcs.d_view;
  const Real vacuum_cap = 0.5*chat_*flux_limit_coefficient_;

  for (int dir = 0; dir < ndim; ++dir) {
    const int di = dir == 0 ? 1 : 0;
    const int dj = dir == 1 ? 1 : 0;
    const int dk = dir == 2 ? 1 : 0;
    const int first = dir == 0 ? is : (dir == 1 ? js : ks);
    const int last = dir == 0 ? ie+1 : (dir == 1 ? je+1 : ke+1);
    const int lower_face = 2*dir;
    const int neighbor_first = dir == 2 ? 24 : 8*dir;
    const int bminus = implicit_boundary_type_[lower_face];
    const int bplus = implicit_boundary_type_[lower_face+1];
    auto conductance = ImplicitAMRFaceBand(dir == 0 ? implicit_amr_conductance_.x1f :
        (dir == 1 ? implicit_amr_conductance_.x2f : implicit_amr_conductance_.x3f));
    par_for("thermal_rad_impl_amr_conductance", DevExeSpace(), 0, nmb1,
            ks, ke+dk, js, je+dj, is, ie+di,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int face_index = dir == 0 ? i : (dir == 1 ? j : k);
      const bool lower = face_index == first;
      const bool upper = face_index == last;
      bool coarse_neighbor = false;
      if (lower || upper) {
        const int begin = neighbor_first+(upper ? 4 : 0);
        // A fine block's coarse neighbor occupies the slot for its child parity,
        // which need not be the first slot of this face.
        for (int n = begin; n < begin+4; ++n) {
          if (neighbors(m, n).gid >= 0 && neighbors(m, n).lev < levels(m)) {
            coarse_neighbor = true;
          }
        }
      }
      const Real dx = dir == 0 ? size(m).dx1 :
          (dir == 1 ? size(m).dx2 : size(m).dx3);
      const Real left = coefficient(m, 0, k-dk, j-dj, i-di);
      const Real right = coefficient(m, 0, k, j, i);
      // Retain the uniform-grid stencil away from refinement interfaces.
      Real value = coarse_neighbor ? CoarseFineConductance(left, right, dx, lower) :
          0.5*(left+right)/dx;
      if (coarse_neighbor && high_order) {
        const int ci = dir == 0 ? (lower ? cis-1 : cie+1) : cis+(i-is)/2;
        const int cj = dir == 1 ? (lower ? cjs-1 : cje+1) :
            (multi_d ? cjs+(j-js)/2 : 0);
        const int ck = dir == 2 ? (lower ? cks-1 : cke+1) :
            (three_d ? cks+(k-ks)/2 : 0);
        const Real coarse_k = CoarseSubfaceTrace(coarse, m, ck, cj, ci, dir,
            2*((k-ks)%2)-1, 2*((j-js)%2)-1, 2*((i-is)%2)-1,
            multi_d, three_d, true);
        const int fi = i-(upper ? di : 0);
        const int fj = j-(upper ? dj : 0);
        const int fk = k-(upper ? dk : 0);
        const int inward = lower ? 1 : -1;
        value = QuadraticCoarseFineConductance(coarse_k,
            coefficient(m, 0, fk, fj, fi),
            coefficient(m, 0, fk+inward*dk, fj+inward*dj, fi+inward*di), dx);
      }
      const bool vacuum =
          (lower && bminus == 2 &&
           IsPhysicalAMRBoundary(boundaries(m, lower_face))) ||
          (upper && bplus == 2 &&
           IsPhysicalAMRBoundary(boundaries(m, lower_face+1)));
      if (vacuum) value = fmin(value, vacuum_cap);
      conductance(m, 0, k, j, i) = value;
    });
  }

  // Reflux a copy: the unrefluxed cache remains the source of every matvec's
  // fine-face fluxes.  Coarse-face provisional fluxes are replaced after each
  // evaluation because a coarse face can couple to several different fine cells.
  Kokkos::deep_copy(DevExeSpace(), implicit_amr_flux_.x1f,
                   implicit_amr_conductance_.x1f);
  if (multi_d) {
    Kokkos::deep_copy(DevExeSpace(), implicit_amr_flux_.x2f,
                     implicit_amr_conductance_.x2f);
  }
  if (three_d) {
    Kokkos::deep_copy(DevExeSpace(), implicit_amr_flux_.x3f,
                     implicit_amr_conductance_.x3f);
  }
  RefluxImplicitAMRFluxes();

  auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
  auto face1 = ImplicitAMRFaceBand(implicit_amr_flux_.x1f);
  auto face2 = ImplicitAMRFaceBand(implicit_amr_flux_.x2f);
  auto face3 = ImplicitAMRFaceBand(implicit_amr_flux_.x3f);
  const int b0 = implicit_boundary_type_[0], b1 = implicit_boundary_type_[1];
  const int b2 = implicit_boundary_type_[2], b3 = implicit_boundary_type_[3];
  const int b4 = implicit_boundary_type_[4], b5 = implicit_boundary_type_[5];
  par_for("thermal_rad_impl_amr_diagonal", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real wm1 = AMRDiagonalWeight(i == is &&
        IsPhysicalAMRBoundary(boundaries(m, 0)), b0);
    const Real wp1 = AMRDiagonalWeight(i == ie &&
        IsPhysicalAMRBoundary(boundaries(m, 1)), b1);
    Real sum = (wm1*face1(m, 0, k, j, i)+
                wp1*face1(m, 0, k, j, i+1))/size(m).dx1;
    if (multi_d) {
      const Real wm2 = AMRDiagonalWeight(j == js &&
          IsPhysicalAMRBoundary(boundaries(m, 2)), b2);
      const Real wp2 = AMRDiagonalWeight(j == je &&
          IsPhysicalAMRBoundary(boundaries(m, 3)), b3);
      sum += (wm2*face2(m, 0, k, j, i)+
              wp2*face2(m, 0, k, j+1, i))/size(m).dx2;
    }
    if (three_d) {
      const Real wm3 = AMRDiagonalWeight(k == ks &&
          IsPhysicalAMRBoundary(boundaries(m, 4)), b4);
      const Real wp3 = AMRDiagonalWeight(k == ke &&
          IsPhysicalAMRBoundary(boundaries(m, 5)), b5);
      sum += (wm3*face3(m, 0, k, j, i)+
              wp3*face3(m, 0, k+1, j, i))/size(m).dx3;
    }
    diagonal(m, 0, k, j, i) = 1.0+dt*sum;
  });

  if (implicit_preconditioner_mode_ >= 1 &&
      implicit_multilevel_enabled_) {
    auto aggregate_diagonal = implicit_multilevel_vector_;
    // These face arrays still contain the refluxed conductance copy here.
    // Matvecs overwrite them with fluxes, so cache the aggregate diagonals now.
    // High-order groups use the same positive graph as their inner PCG solve;
    // a low-order fallback rebuilds this cache with its own exact graph.
    for (int level = 0; level < 2; ++level) {
      const int cnx1 = implicit_multilevel_nx1_[level];
      const int cnx2 = implicit_multilevel_nx2_[level];
      const int cnx3 = implicit_multilevel_nx3_[level];
      const int ni = indcs.nx1/cnx1;
      const int nj = indcs.nx2/cnx2;
      const int nk = indcs.nx3/cnx3;
      const int level_offset = implicit_multilevel_offset_[level];
      par_for("thermal_rad_amr_aggregate_diagonal", DevExeSpace(), 0, nmb1,
              0, cnx3-1, 0, cnx2-1, 0, cnx1-1,
      KOKKOS_LAMBDA(int m, int ck, int cj, int ci) {
        const int i0 = is+ci*ni, j0 = js+cj*nj, k0 = ks+ck*nk;
        const Real wm1 = AMRDiagonalWeight(ci == 0 &&
            IsPhysicalAMRBoundary(boundaries(m, 0)), b0);
        const Real wp1 = AMRDiagonalWeight(ci == cnx1-1 &&
            IsPhysicalAMRBoundary(boundaries(m, 1)), b1);
        Real sum = (wm1*AMRAggregateFaceSum(
            face1, m, 0, false, i0, j0, k0, ni, nj, nk)+
            wp1*AMRAggregateFaceSum(
            face1, m, 0, true, i0, j0, k0, ni, nj, nk))/size(m).dx1;
        if (multi_d) {
          const Real wm2 = AMRDiagonalWeight(cj == 0 &&
              IsPhysicalAMRBoundary(boundaries(m, 2)), b2);
          const Real wp2 = AMRDiagonalWeight(cj == cnx2-1 &&
              IsPhysicalAMRBoundary(boundaries(m, 3)), b3);
          sum += (wm2*AMRAggregateFaceSum(
              face2, m, 1, false, i0, j0, k0, ni, nj, nk)+
              wp2*AMRAggregateFaceSum(
              face2, m, 1, true, i0, j0, k0, ni, nj, nk))/size(m).dx2;
        }
        if (three_d) {
          const Real wm3 = AMRDiagonalWeight(ck == 0 &&
              IsPhysicalAMRBoundary(boundaries(m, 4)), b4);
          const Real wp3 = AMRDiagonalWeight(ck == cnx3-1 &&
              IsPhysicalAMRBoundary(boundaries(m, 5)), b5);
          sum += (wm3*AMRAggregateFaceSum(
              face3, m, 2, false, i0, j0, k0, ni, nj, nk)+
              wp3*AMRAggregateFaceSum(
              face3, m, 2, true, i0, j0, k0, ni, nj, nk))/size(m).dx3;
        }
        const int packed = level_offset+(ck*cnx2+cj)*cnx1+ci;
        aggregate_diagonal(m, packed) = static_cast<Real>(ni*nj*nk)+dt*sum;
      });
    }
  }
}

//----------------------------------------------------------------------------------------
//! Build E=P^T V A_low P for one constant basis function per active MeshBlock.
//! The monotone interface operator supplies an SPD coarse graph even when the
//! Krylov operator uses signed quadratic reconstruction. Each edge is assembled
//! once, on the smaller GID for equal levels or the fine side for unequal levels.

void ThermalRadiation::BuildImplicitAMRBlockCoarsePreconditioner(const Real dt) {
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmy_pack_->nmb_thispack;
  const int gid0 = pmy_pack_->gids;
  const int ncoarse = pmy_pack_->pmesh->nmb_total;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  const std::size_t matrix_size = static_cast<std::size_t>(ncoarse)*ncoarse;
  if (matrix_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    ImplicitAMRError("AMR block-coarse matrix exceeds the MPI count limit; use jacobi");
  }
  if (implicit_amr_block_matrix_.extent_int(0) != ncoarse) {
    Kokkos::realloc(implicit_amr_block_matrix_, ncoarse, ncoarse);
    Kokkos::realloc(implicit_coarse_vector_, ncoarse);
    implicit_coarse_cholesky_.resize(matrix_size);
    implicit_coarse_scaling_.resize(ncoarse);
  }
  auto matrix = implicit_amr_block_matrix_.d_view;
  Kokkos::deep_copy(DevExeSpace(), matrix, 0.0);
  auto size = pmy_pack_->pmb->mb_size.d_view;
  const Real cells = static_cast<Real>(indcs.nx1*indcs.nx2*indcs.nx3);
  par_for("thermal_rad_amr_block_mass", DevExeSpace(), 0, nmb-1,
  KOKKOS_LAMBDA(int m) {
    matrix(gid0+m, gid0+m) = cells*size(m).dx1*size(m).dx2*size(m).dx3;
  });
  auto coefficient = implicit_coefficient_;
  auto neighbors = pmy_pack_->pmb->nghbr.d_view;
  auto levels = pmy_pack_->pmb->mb_lev.d_view;
  auto boundaries = pmy_pack_->pmb->mb_bcs.d_view;
  for (int face = 0; face < 2*ndim; ++face) {
    const int dir = face/2;
    const bool lower = face%2 == 0;
    const int di = dir == 0 ? 1 : 0;
    const int dj = dir == 1 ? 1 : 0;
    const int dk = dir == 2 ? 1 : 0;
    const int il = dir == 0 ? (lower ? is : ie) : is;
    const int iu = dir == 0 ? il : ie;
    const int jl = dir == 1 ? (lower ? js : je) : js;
    const int ju = dir == 1 ? jl : je;
    const int kl = dir == 2 ? (lower ? ks : ke) : ks;
    const int ku = dir == 2 ? kl : ke;
    const int begin = (dir == 2 ? 24 : 8*dir)+(lower ? 0 : 4);
    const int boundary_type = implicit_boundary_type_[face];
    auto conductance = ImplicitAMRFaceBand(dir == 0 ? implicit_amr_conductance_.x1f :
        (dir == 1 ? implicit_amr_conductance_.x2f : implicit_amr_conductance_.x3f));
    par_for("thermal_rad_amr_block_edges", DevExeSpace(), 0, nmb-1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int gid = gid0+m;
      int peer = -1, peer_level = levels(m);
      for (int n = begin; n < begin+4; ++n) {
        if (neighbors(m, n).gid >= 0 && neighbors(m, n).lev <= levels(m)) {
          peer = neighbors(m, n).gid;
          peer_level = neighbors(m, n).lev;
        }
      }
      if (peer >= 0 && peer_level == levels(m) && gid >= peer) return;
      const bool physical = IsPhysicalAMRBoundary(boundaries(m, face));
      // Finer neighbors contribute their own edges; there is no provisional
      // coarse-face contribution to this graph. Periodic self-neighbors vanish.
      if (peer < 0 && !physical) return;
      const Real dx = dir == 0 ? size(m).dx1 :
          (dir == 1 ? size(m).dx2 : size(m).dx3);
      const Real area = size(m).dx1*size(m).dx2*size(m).dx3/dx;
      const int fi = i+(lower ? 0 : di);
      const int fj = j+(lower ? 0 : dj);
      const int fk = k+(lower ? 0 : dk);
      Real value = conductance(m, 0, fk, fj, fi);
      if (peer >= 0 && peer_level < levels(m)) {
        // Coefficient ghosts still hold their actual coarse-cell values. Use
        // half-cell resistance, independent of the high-order flux cache.
        const Real left = coefficient(m, 0, fk-dk, fj-dj, fi-di);
        const Real right = coefficient(m, 0, fk, fj, fi);
        value = CoarseFineConductance(left, right, dx, lower);
      }
      const Real weight = dt*area*value;
      if (peer >= 0) {
        Kokkos::atomic_add(&matrix(gid, gid), weight);
        Kokkos::atomic_add(&matrix(peer, peer), weight);
        Kokkos::atomic_add(&matrix(gid, peer), -weight);
        Kokkos::atomic_add(&matrix(peer, gid), -weight);
      } else {
        Kokkos::atomic_add(&matrix(gid, gid),
                          AMRDiagonalWeight(true, boundary_type)*weight);
      }
    });
  }
  implicit_amr_block_matrix_.template modify<DevExeSpace>();
  implicit_amr_block_matrix_.template sync<HostMemSpace>();
  auto host_matrix = implicit_amr_block_matrix_.h_view;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, host_matrix.data(), static_cast<int>(matrix_size),
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  auto &factor = implicit_coarse_cholesky_;
  auto &scaling = implicit_coarse_scaling_;
  for (int row = 0; row < ncoarse; ++row) {
    const Real diagonal = host_matrix(row, row);
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
      ImplicitAMRError("Invalid AMR block-coarse diagonal");
    }
    scaling[row] = 1.0/std::sqrt(diagonal);
  }
  for (int row = 0; row < ncoarse; ++row) {
    for (int col = 0; col < ncoarse; ++col) {
      factor[static_cast<std::size_t>(row)*ncoarse+col] =
          host_matrix(row, col)*scaling[row]*scaling[col];
    }
  }
  // Dense, diagonally equilibrated Cholesky is shared by all ranks. The identity
  // in the fine operator supplies a positive mass for every block, including a
  // disconnected block with zero diffusion coefficient or reflecting boundaries.
  for (int row = 0; row < ncoarse; ++row) {
    for (int col = 0; col <= row; ++col) {
      Real value = factor[static_cast<std::size_t>(row)*ncoarse+col];
      for (int q = 0; q < col; ++q) {
        value -= factor[static_cast<std::size_t>(row)*ncoarse+q]*
                 factor[static_cast<std::size_t>(col)*ncoarse+q];
      }
      if (row == col) {
        if (!(value > 0.0) || !std::isfinite(value)) {
          ImplicitAMRError("AMR block-coarse Cholesky failed at pivot "+
                           std::to_string(row));
        }
        factor[static_cast<std::size_t>(row)*ncoarse+col] = std::sqrt(value);
      } else {
        factor[static_cast<std::size_t>(row)*ncoarse+col] =
            value/factor[static_cast<std::size_t>(col)*ncoarse+col];
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! Add P E^-1 P^T V r to the positive diagonal correction. This is fixed and
//! linear throughout a Krylov solve, and is SPD in the volume inner product.

void ThermalRadiation::ApplyImplicitAMRBlockCoarseCorrection(
    const DvceArray5D<Real> &residual, DvceArray5D<Real> &preconditioned) {
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int gid0 = pmy_pack_->gids;
  auto coarse = implicit_coarse_vector_.d_view;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto r = residual;
  auto z = preconditioned;
  Kokkos::deep_copy(DevExeSpace(), coarse, 0.0);
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int plane = nx1*nx2, cells = plane*indcs.nx3;
  constexpr int tile_size = 256;
  const int tiles = (cells+tile_size-1)/tile_size;
  // A single MeshBlock may contain tens of thousands of fine cells. Reduce
  // bounded tiles before their atomic additions to avoid contending on one
  // coarse entry per cell, while exposing multiple GPU teams per MeshBlock.
  par_for_outer("thermal_rad_amr_block_restrict", DevExeSpace(), 0, 0,
                0, nmb1, 0, tiles-1,
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m, const int tile) {
    const int begin = tile*tile_size;
    const int count = cells-begin < tile_size ? cells-begin : tile_size;
    Real tile_sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, count),
    [=](const int offset, Real &sum) {
      const int cell = begin+offset;
      const int k = cell/plane+ks;
      const int j = (cell%plane)/nx1+js;
      const int i = cell%nx1+is;
      sum += r(m, 0, k, j, i);
    }, tile_sum);
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      Kokkos::atomic_add(&coarse(gid0+m),
          size(m).dx1*size(m).dx2*size(m).dx3*tile_sum);
    });
  });
  SolveImplicitBlockRootSystem();
  par_for("thermal_rad_amr_block_correct", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    z(m, 0, k, j, i) += coarse(gid0+m);
  });
}

//----------------------------------------------------------------------------------------
//! Differentiate the exchanged energy using its actual coarse/fine geometry.
//! At a fine cell touching an interface, fit the same cell-average quadratic used
//! by the face flux.  Its derivative at the fine-cell center is second order,
//! including fields with transverse variation.  Other cells retain centered
//! differences; restricted fine neighbors already represent full coarse cells.

void ThermalRadiation::PrepareImplicitAMRGradient(const DvceArray5D<Real> &field) {
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int cis = indcs.cis, cie = indcs.cie;
  const int cjs = indcs.cjs, cje = indcs.cje;
  const int cks = indcs.cks, cke = indcs.cke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  auto f = field;
  auto coarse = implicit_coarse_scratch_;
  auto gradient = ImplicitAMRCellBand(implicit_amr_gradient_);
  auto neighbors = pmy_pack_->pmb->nghbr.d_view;
  auto levels = pmy_pack_->pmb->mb_lev.d_view;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  par_for("thermal_rad_impl_amr_gradient", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int dir = 0; dir < 3; ++dir) {
      if (dir >= ndim) {
        gradient(m, dir, k, j, i) = 0.0;
        continue;
      }
      const int di = dir == 0 ? 1 : 0;
      const int dj = dir == 1 ? 1 : 0;
      const int dk = dir == 2 ? 1 : 0;
      const Real dx = dir == 0 ? size(m).dx1 :
          (dir == 1 ? size(m).dx2 : size(m).dx3);
      Real value = (f(m, 0, k+dk, j+dj, i+di)-
                    f(m, 0, k-dk, j-dj, i-di))/(2.0*dx);
      const int index = dir == 0 ? i : (dir == 1 ? j : k);
      const int first = dir == 0 ? is : (dir == 1 ? js : ks);
      const int last = dir == 0 ? ie : (dir == 1 ? je : ke);
      const bool lower = index == first;
      const bool upper = index == last;
      bool coarse_neighbor = false;
      if (lower || upper) {
        const int begin = (dir == 2 ? 24 : 8*dir)+(upper ? 4 : 0);
        for (int n = begin; n < begin+4; ++n) {
          if (neighbors(m, n).gid >= 0 && neighbors(m, n).lev < levels(m)) {
            coarse_neighbor = true;
          }
        }
      }
      if (coarse_neighbor) {
        const int ci = dir == 0 ? (lower ? cis-1 : cie+1) : cis+(i-is)/2;
        const int cj = dir == 1 ? (lower ? cjs-1 : cje+1) :
            (multi_d ? cjs+(j-js)/2 : 0);
        const int ck = dir == 2 ? (lower ? cks-1 : cke+1) :
            (three_d ? cks+(k-ks)/2 : 0);
        const Real correction = CoarseSubfaceCorrection(coarse, m, ck, cj, ci, dir,
            2*((k-ks)%2)-1, 2*((j-js)%2)-1, 2*((i-is)%2)-1,
            multi_d, three_d);
        const int inward = lower ? 1 : -1;
        const Real f0 = f(m, 0, k, j, i);
        const Real f1 = f(m, 0, k+inward*dk, j+inward*dj, i+inward*di);
        const Real center = coarse(m, 0, ck, cj, ci);
        value = inward*(2.0*(f0-center)+5.0*(f1-f0)-2.0*correction)/(8.0*dx);
      }
      gradient(m, dir, k, j, i) = value;
    }
  });
}

//----------------------------------------------------------------------------------------
//! Apply A=I-dt*div(K*grad), using a single conservative flux at each AMR face.
//! In absolute mode accumulate a bound on magnitudes of the stencil contributions.
//! This bound is used for the low-order componentwise backward error only: signed
//! high-order interpolation requires a separate true-residual convergence check.

void ThermalRadiation::ApplyImplicitAMROperator(
    const DvceArray5D<Real> &field, DvceArray5D<Real> &result, const Real dt,
    const bool absolute_values) {
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const bool multi_d = pmy_pack_->pmesh->multi_d;
  const bool three_d = pmy_pack_->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  const bool high_order = implicit_amr_high_order_;
  const int cis = indcs.cis, cie = indcs.cie;
  const int cjs = indcs.cjs, cje = indcs.cje;
  const int cks = indcs.cks, cke = indcs.cke;
  auto coarse = implicit_coarse_scratch_;
  auto neighbors = pmy_pack_->pmb->nghbr.d_view;
  auto levels = pmy_pack_->pmb->mb_lev.d_view;
  auto f = field;
  for (int dir = 0; dir < ndim; ++dir) {
    const int di = dir == 0 ? 1 : 0;
    const int dj = dir == 1 ? 1 : 0;
    const int dk = dir == 2 ? 1 : 0;
    const int first = dir == 0 ? is : (dir == 1 ? js : ks);
    const int last = dir == 0 ? ie+1 : (dir == 1 ? je+1 : ke+1);
    const int neighbor_first = dir == 2 ? 24 : 8*dir;
    auto conductance = ImplicitAMRFaceBand(dir == 0 ? implicit_amr_conductance_.x1f :
        (dir == 1 ? implicit_amr_conductance_.x2f : implicit_amr_conductance_.x3f));
    auto flux = ImplicitAMRFaceBand(dir == 0 ? implicit_amr_flux_.x1f :
        (dir == 1 ? implicit_amr_flux_.x2f : implicit_amr_flux_.x3f));
    par_for("thermal_rad_impl_amr_flux", DevExeSpace(), 0, nmb1,
            ks, ke+dk, js, je+dj, is, ie+di,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real left = f(m, 0, k-dk, j-dj, i-di);
      const Real right = f(m, 0, k, j, i);
      Real difference = absolute_values ? fabs(left)+fabs(right) : left-right;
      const int face_index = dir == 0 ? i : (dir == 1 ? j : k);
      const bool lower = face_index == first;
      const bool upper = face_index == last;
      bool coarse_neighbor = false;
      if (high_order && (lower || upper)) {
        const int begin = neighbor_first+(upper ? 4 : 0);
        for (int n = begin; n < begin+4; ++n) {
          if (neighbors(m, n).gid >= 0 && neighbors(m, n).lev < levels(m)) {
            coarse_neighbor = true;
          }
        }
      }
      if (coarse_neighbor) {
        const int ci = dir == 0 ? (lower ? cis-1 : cie+1) : cis+(i-is)/2;
        const int cj = dir == 1 ? (lower ? cjs-1 : cje+1) :
            (multi_d ? cjs+(j-js)/2 : 0);
        const int ck = dir == 2 ? (lower ? cks-1 : cke+1) :
            (three_d ? cks+(k-ks)/2 : 0);
        const int fi = i-(upper ? di : 0);
        const int fj = j-(upper ? dj : 0);
        const int fk = k-(upper ? dk : 0);
        const int inward = lower ? 1 : -1;
        const Real f0 = f(m, 0, fk, fj, fi);
        const Real f1 = f(m, 0, fk+inward*dk, fj+inward*dj, fi+inward*di);
        // For constant K this is K*(2*C-F0-F1)/(4*h), the quadratic-exact
        // normal derivative of one coarse and two fine cell averages.
        if (absolute_values) {
          const Real trace = CoarseSubfaceTrace(coarse, m, ck, cj, ci, dir,
              2*((k-ks)%2)-1, 2*((j-js)%2)-1, 2*((i-is)%2)-1,
              multi_d, three_d, false, true);
          difference = trace+0.5*(fabs(f0)+fabs(f1));
        } else {
          const Real correction = CoarseSubfaceCorrection(coarse, m, ck, cj, ci, dir,
              2*((k-ks)%2)-1, 2*((j-js)%2)-1, 2*((i-is)%2)-1,
              multi_d, three_d);
          const Real center = coarse(m, 0, ck, cj, ci);
          difference = (lower ? 1.0 : -1.0)*((center-f0)-0.5*(f1-f0)+correction);
        }
      }
      flux(m, 0, k, j, i) = conductance(m, 0, k, j, i)*difference;
    });
  }
  RefluxImplicitAMRFluxes();

  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto face1 = ImplicitAMRFaceBand(implicit_amr_flux_.x1f);
  auto face2 = ImplicitAMRFaceBand(implicit_amr_flux_.x2f);
  auto face3 = ImplicitAMRFaceBand(implicit_amr_flux_.x3f);
  auto out = result;
  const Real lower_sign = absolute_values ? 1.0 : -1.0;
  par_for("thermal_rad_impl_amr_operator", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real divergence = (face1(m, 0, k, j, i+1)+
                       lower_sign*face1(m, 0, k, j, i))/size(m).dx1;
    if (multi_d) {
      divergence += (face2(m, 0, k, j+1, i)+
                     lower_sign*face2(m, 0, k, j, i))/size(m).dx2;
    }
    if (three_d) {
      divergence += (face3(m, 0, k+1, j, i)+
                     lower_sign*face3(m, 0, k, j, i))/size(m).dx3;
    }
    const Real center = f(m, 0, k, j, i);
    out(m, 0, k, j, i) = (absolute_values ? fabs(center) : center)+dt*divergence;
  });
}

//----------------------------------------------------------------------------------------
//! Area-average fine fluxes onto their neighboring coarse faces, including MPI.

void ThermalRadiation::RefluxImplicitAMRFluxes() {
  if (implicit_bvals_ == nullptr) {
    ImplicitAMRError("Missing boundary communicator for implicit-radiation AMR fluxes");
  }
  if (implicit_halo_ != nullptr) {
    implicit_halo_->Reflux(implicit_amr_flux_, implicit_amr_face_origin_);
    return;
  }
  if (implicit_bvals_->InitFluxRecv(1) != TaskStatus::complete) {
    ImplicitAMRError("Could not initialize implicit-radiation AMR flux receives");
  }
  if (implicit_bvals_->PackAndSendFluxCC(
          implicit_amr_flux_, nullptr, implicit_amr_face_origin_) !=
      TaskStatus::complete) {
    ImplicitAMRError("Could not send implicit-radiation AMR fluxes");
  }
  TaskStatus status = TaskStatus::incomplete;
  while (status == TaskStatus::incomplete) {
    status = implicit_bvals_->RecvAndUnpackFluxCC(
        implicit_amr_flux_, nullptr, implicit_amr_face_origin_);
  }
  if (status != TaskStatus::complete) {
    ImplicitAMRError("Could not receive implicit-radiation AMR fluxes");
  }
  if (implicit_bvals_->ClearFluxSend() != TaskStatus::complete ||
      implicit_bvals_->ClearFluxRecv() != TaskStatus::complete) {
    ImplicitAMRError("Could not clear implicit-radiation AMR flux communication");
  }
}

}  // namespace two_temperature
