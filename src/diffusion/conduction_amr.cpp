//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file conduction_amr.cpp
//! \brief Conservative two-point electron diffusion on the composite leaf mesh.

#include <cmath>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "diffusion/conduction.hpp"
#include "two_temperature/thermal_radiation_halo.hpp"
#include "mesh/mesh.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool PhysicalFace(const BoundaryFlag flag) {
  return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
         flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef;
}

KOKKOS_INLINE_FUNCTION
Real DiagonalWeight(const bool physical, const int boundary) {
  return physical ? (boundary == 0 ? 0.0 : 2.0) : 1.0;
}

// The coarse center is one fine-cell width from the face, the fine center half
// a fine-cell width. Add these two thermal resistances in series. Scaling by
// the smaller conductivity avoids overflow and handles insulating cells exactly.
KOKKOS_INLINE_FUNCTION
Real InterfaceConductance(const Real left, const Real right,
                          const Real dx, const bool coarse_left) {
  if (!(left > 0.0) || !(right > 0.0)) return 0.0;
  const Real scale = fmin(left, right);
  return (scale/dx)/((coarse_left ? 1.0 : 0.5)*(scale/left)+
                     (coarse_left ? 0.5 : 1.0)*(scale/right));
}

}  // namespace

//----------------------------------------------------------------------------------------
//! Rebuild the frozen face graph every solve, including after refinement/redistribution.

void Conduction::BuildAMROperator() {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  auto size = pmy_pack->pmb->mb_size.d_view;
  auto neighbors = pmy_pack->pmb->nghbr.d_view;
  auto levels = pmy_pack->pmb->mb_lev.d_view;
  auto boundaries = pmy_pack->pmb->mb_bcs.d_view;
  auto kappa = conductivity_;

  for (int dir = 0; dir < ndim; ++dir) {
    const int di = dir == 0 ? 1 : 0;
    const int dj = dir == 1 ? 1 : 0;
    const int dk = dir == 2 ? 1 : 0;
    const int first = dir == 0 ? is : (dir == 1 ? js : ks);
    const int last = dir == 0 ? ie+1 : (dir == 1 ? je+1 : ke+1);
    const int neighbor_first = dir == 2 ? 24 : 8*dir;
    auto face = AMRFaceBand(dir == 0 ? amr_conductance_.x1f :
        (dir == 1 ? amr_conductance_.x2f : amr_conductance_.x3f));
    par_for("cond_amr_conductance", DevExeSpace(), 0, nmb1,
            ks, ke+dk, js, je+dj, is, ie+di,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const int index = dir == 0 ? i : (dir == 1 ? j : k);
      const bool lower = index == first;
      const bool upper = index == last;
      bool coarse_neighbor = false;
      if (lower || upper) {
        const int begin = neighbor_first+(upper ? 4 : 0);
        // The coarse neighbor's slot depends on the fine block's child parity.
        for (int n = begin; n < begin+4; ++n) {
          if (neighbors(m, n).gid >= 0 && neighbors(m, n).lev < levels(m)) {
            coarse_neighbor = true;
          }
        }
      }
      const Real dx = dir == 0 ? size(m).dx1 :
          (dir == 1 ? size(m).dx2 : size(m).dx3);
      const Real left = kappa(m, 0, k-dk, j-dj, i-di);
      const Real right = kappa(m, 0, k, j, i);
      face(m, 0, k, j, i) = coarse_neighbor ?
          InterfaceConductance(left, right, dx, lower) : 0.5*(left+right)/dx;
    });
  }

  // Reflux a copy to build the exact diagonal. Every matvec subsequently
  // evaluates fine-face fluxes and replaces provisional coarse-face fluxes.
  Kokkos::deep_copy(DevExeSpace(), amr_flux_.x1f, amr_conductance_.x1f);
  if (multi_d) Kokkos::deep_copy(DevExeSpace(), amr_flux_.x2f, amr_conductance_.x2f);
  if (three_d) Kokkos::deep_copy(DevExeSpace(), amr_flux_.x3f, amr_conductance_.x3f);
  RefluxAMRFluxes();

  auto diagonal = amr_diagonal_;
  auto face1 = AMRFaceBand(amr_flux_.x1f);
  auto face2 = AMRFaceBand(amr_flux_.x2f);
  auto face3 = AMRFaceBand(amr_flux_.x3f);
  const int b0 = static_cast<int>(boundary_type_[0]);
  const int b1 = static_cast<int>(boundary_type_[1]);
  const int b2 = static_cast<int>(boundary_type_[2]);
  const int b3 = static_cast<int>(boundary_type_[3]);
  const int b4 = static_cast<int>(boundary_type_[4]);
  const int b5 = static_cast<int>(boundary_type_[5]);
  par_for("cond_amr_diagonal", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real wm1 = DiagonalWeight(i == is && PhysicalFace(boundaries(m, 0)), b0);
    const Real wp1 = DiagonalWeight(i == ie && PhysicalFace(boundaries(m, 1)), b1);
    Real sum = (wm1*face1(m, 0, k, j, i)+wp1*face1(m, 0, k, j, i+1))/size(m).dx1;
    if (multi_d) {
      const Real wm2 = DiagonalWeight(j == js && PhysicalFace(boundaries(m, 2)), b2);
      const Real wp2 = DiagonalWeight(j == je && PhysicalFace(boundaries(m, 3)), b3);
      sum += (wm2*face2(m, 0, k, j, i)+wp2*face2(m, 0, k, j+1, i))/size(m).dx2;
    }
    if (three_d) {
      const Real wm3 = DiagonalWeight(k == ks && PhysicalFace(boundaries(m, 4)), b4);
      const Real wp3 = DiagonalWeight(k == ke && PhysicalFace(boundaries(m, 5)), b5);
      sum += (wm3*face3(m, 0, k, j, i)+wp3*face3(m, 0, k+1, j, i))/size(m).dx3;
    }
    diagonal(m, 0, k, j, i) = sum;
  });
}

//----------------------------------------------------------------------------------------
//! Apply div(K grad(T)) with one flux shared by the two sides of every leaf face.

void Conduction::ApplyAMRDiffusionOperator(const DvceArray5D<Real> &field,
                                          DvceArray5D<Real> &result) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack-1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  auto f = field;
  for (int dir = 0; dir < ndim; ++dir) {
    const int di = dir == 0 ? 1 : 0;
    const int dj = dir == 1 ? 1 : 0;
    const int dk = dir == 2 ? 1 : 0;
    auto conductance = AMRFaceBand(dir == 0 ? amr_conductance_.x1f :
        (dir == 1 ? amr_conductance_.x2f : amr_conductance_.x3f));
    auto flux = AMRFaceBand(dir == 0 ? amr_flux_.x1f :
        (dir == 1 ? amr_flux_.x2f : amr_flux_.x3f));
    par_for("cond_amr_flux", DevExeSpace(), 0, nmb1,
            ks, ke+dk, js, je+dj, is, ie+di,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      flux(m, 0, k, j, i) = conductance(m, 0, k, j, i)*
          (f(m, 0, k, j, i)-f(m, 0, k-dk, j-dj, i-di));
    });
  }
  RefluxAMRFluxes();

  auto size = pmy_pack->pmb->mb_size.d_view;
  auto face1 = AMRFaceBand(amr_flux_.x1f);
  auto face2 = AMRFaceBand(amr_flux_.x2f);
  auto face3 = AMRFaceBand(amr_flux_.x3f);
  auto out = result;
  par_for("cond_amr_laplacian", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real lap = (face1(m, 0, k, j, i+1)-face1(m, 0, k, j, i))/size(m).dx1;
    if (multi_d) lap += (face2(m, 0, k, j+1, i)-face2(m, 0, k, j, i))/size(m).dx2;
    if (three_d) lap += (face3(m, 0, k+1, j, i)-face3(m, 0, k, j, i))/size(m).dx3;
    out(m, 0, k, j, i) = lap;
  });
}

//----------------------------------------------------------------------------------------
//! Area-average fine fluxes onto neighboring coarse faces, locally and across MPI.

void Conduction::RefluxAMRFluxes() {
  if (implicit_halo_ != nullptr) {
    implicit_halo_->Reflux(amr_flux_, amr_face_origin_);
    return;
  }
  if (implicit_bvals_ == nullptr ||
      implicit_bvals_->InitFluxRecv(1) != TaskStatus::complete ||
      implicit_bvals_->PackAndSendFluxCC(amr_flux_, nullptr, amr_face_origin_) !=
          TaskStatus::complete) {
    Kokkos::abort("Could not initialize/send implicit-conduction AMR flux exchange");
  }
  TaskStatus status = TaskStatus::incomplete;
  while (status == TaskStatus::incomplete) {
    status = implicit_bvals_->RecvAndUnpackFluxCC(amr_flux_, nullptr, amr_face_origin_);
  }
  if (status != TaskStatus::complete ||
      implicit_bvals_->ClearFluxSend() != TaskStatus::complete ||
      implicit_bvals_->ClearFluxRecv() != TaskStatus::complete) {
    Kokkos::abort("Could not complete implicit-conduction AMR flux exchange");
  }
}
