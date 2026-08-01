//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_biermann_subcycle.cpp
//! \brief Dedicated SSPRK2 stage operations for multirate Biermann integration.

#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "mhd/biermann_battery.hpp"
#include "mhd/mhd.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "two_temperature/two_temperature.hpp"

namespace mhd {

bool MHD::BiermannSubcycleActive() const {
  return biermann_subcycle && pbiermann != nullptr && pbiermann->coefficient != 0.0;
}

Real MHD::BiermannSubcycleTimeStepLimit() {
  if (!BiermannSubcycleActive()) {
    return std::numeric_limits<Real>::max();
  }
  pbiermann->NewTimeStep(w0, bcc0);
  return biermann_subcycle_cfl * pbiermann->dtnew;
}

//! Post receives needed by one self-contained Biermann stage.  Orbital and shearing
//! communications deliberately do not participate in microsteps.

TaskStatus MHD::BiermannInitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nmhd + nscalars);
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->InitFluxRecv(
        nmhd + nscalars + (pbiermann->DirectDriftCorrectionEnabled() ? 1 : 0));
    if (tstat != TaskStatus::complete) return tstat;
  }
  return pbval_b->InitFluxRecv(3);
}

//----------------------------------------------------------------------------------------
//! Save the beginning of an SSPRK2 microstep in the existing low-storage registers.

TaskStatus MHD::BiermannCopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Build only Biermann face fluxes.  AddFluxes uses += because the legacy path adds to
//! ideal-MHD fluxes, so the dedicated arrays must be cleared first.

TaskStatus MHD::BiermannFluxes(Driver *pdrive, int stage) {
  Kokkos::deep_copy(uflx.x1f, 0.0);
  Kokkos::deep_copy(uflx.x2f, 0.0);
  Kokkos::deep_copy(uflx.x3f, 0.0);
  pbiermann->AddFluxes(w0, bcc0, uflx);
  return TaskStatus::complete;
}

TaskStatus MHD::BiermannSendFlux(Driver *pdrive, int stage) {
  if (!pmy_pack->pmesh->multilevel) return TaskStatus::complete;
  return pbval_u->PackAndSendFluxCC(
      uflx, pbiermann->DirectDriftCorrectionEnabled()
                ? pbiermann->DriftCorrectionFlux() : nullptr);
}

TaskStatus MHD::BiermannRecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(
        uflx, pbiermann->DirectDriftCorrectionEnabled()
                  ? pbiermann->DriftCorrectionFlux() : nullptr);
  }
  if (tstat == TaskStatus::complete) {
    pbiermann->UseCorrectedDriftFlux();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! Conservative total/electron-energy SSPRK2 update plus the matching additive
//! electron pressure-work RHS.  Ion energy is an algebraic auxiliary for the Biermann
//! operator: after CT it is reconstructed from total internal minus electron energy.
//! Keeping it out of this recurrence is essential.  Blending a projected ion energy as
//! though it had a zero RHS leaves the projection outside Heun's method and reduces the
//! full split update to first order.

TaskStatus MHD::BiermannRKUpdate(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int iele = ptwo_temp->iele;
  Real gam_current = (stage == 1) ? 0.0 : 0.5;
  Real gam_start = (stage == 1) ? 1.0 : 0.5;
  Real beta_dt = ((stage == 1) ? 1.0 : 0.5) * biermann_substep_dt;
  auto u = u0;
  auto u_start = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto size = pmy_pack->pmb->mb_size;

  par_for(
      "biermann_energy_update", DevExeSpace(), 0, nmb1, 0, 1, ks, ke, js, je,
      is, ie,
      KOKKOS_LAMBDA(const int m, const int v, const int k, const int j,
                    const int i) {
        const int n = (v == 0) ? IEN : iele;
        Real divf = (flx1(m, n, k, j, i + 1) - flx1(m, n, k, j, i)) /
                    size.d_view(m).dx1;
        if (multi_d) {
          divf += (flx2(m, n, k, j + 1, i) - flx2(m, n, k, j, i)) /
                  size.d_view(m).dx2;
        }
        if (three_d) {
          divf += (flx3(m, n, k + 1, j, i) - flx3(m, n, k, j, i)) /
                  size.d_view(m).dx3;
        }
        u(m, n, k, j, i) = gam_current * u(m, n, k, j, i) +
                            gam_start * u_start(m, n, k, j, i) -
                            beta_dt * divf;
      });
  pbiermann->AddElectronWorkRHS(beta_dt, u0, w0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Construct Biermann-only CT edge fields from the face fields cached by AddFluxes.

TaskStatus MHD::BiermannEField(Driver *pdrive, int stage) {
  Kokkos::deep_copy(efld.x1e, 0.0);
  Kokkos::deep_copy(efld.x2e, 0.0);
  Kokkos::deep_copy(efld.x3e, 0.0);
  pbiermann->AddEMFs(efld);
  pbiermann->ReconcileCompositeAMREMFs(efld);
  return TaskStatus::complete;
}

TaskStatus MHD::BiermannCompositeEnergyFlux(Driver *pdrive, int stage) {
  pbiermann->AddPoyntingFluxFromEdgeEMF(bcc0, efld, uflx);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Constrained-transport SSPRK2 update with the microstep coefficients.

TaskStatus MHD::BiermannCT(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real gam_current = (stage == 1) ? 0.0 : 0.5;
  Real gam_start = (stage == 1) ? 1.0 : 0.5;
  Real beta_dt = ((stage == 1) ? 1.0 : 0.5) * biermann_substep_dt;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto size = pmy_pack->pmb->mb_size;

  if (multi_d) {
    auto bx = b0.x1f;
    auto bx_start = b1.x1f;
    par_for(
        "biermann_ct_b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie + 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real value = gam_current * bx(m, k, j, i) +
                       gam_start * bx_start(m, k, j, i);
          value -= beta_dt * (e3(m, k, j + 1, i) - e3(m, k, j, i)) /
                   size.d_view(m).dx2;
          if (three_d) {
            value += beta_dt * (e2(m, k + 1, j, i) - e2(m, k, j, i)) /
                     size.d_view(m).dx3;
          }
          bx(m, k, j, i) = value;
        });
  }

  auto by = b0.x2f;
  auto by_start = b1.x2f;
  par_for(
      "biermann_ct_b2", DevExeSpace(), 0, nmb1, ks, ke, js, je + 1, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real value = gam_current * by(m, k, j, i) +
                     gam_start * by_start(m, k, j, i);
        value += beta_dt * (e3(m, k, j, i + 1) - e3(m, k, j, i)) /
                 size.d_view(m).dx1;
        if (three_d) {
          value -= beta_dt * (e1(m, k + 1, j, i) - e1(m, k, j, i)) /
                   size.d_view(m).dx3;
        }
        by(m, k, j, i) = value;
      });

  auto bz = b0.x3f;
  auto bz_start = b1.x3f;
  par_for(
      "biermann_ct_b3", DevExeSpace(), 0, nmb1, ks, ke + 1, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real value = gam_current * bz(m, k, j, i) +
                     gam_start * bz_start(m, k, j, i);
        value -= beta_dt * (e2(m, k, j, i + 1) - e2(m, k, j, i)) /
                 size.d_view(m).dx1;
        if (multi_d) {
          value += beta_dt * (e1(m, k, j + 1, i) - e1(m, k, j, i)) /
                   size.d_view(m).dx2;
        }
        bz(m, k, j, i) = value;
      });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Close active fine cells before restriction.  C2P/floors are nonlinear, so restricting
//! the pre-closure auxiliary ion energy cannot reproduce the accepted coarse dual-energy
//! state.  The final full-domain closure below remains necessary after communication,
//! physical boundaries, and prolongation refresh the ghost zones.

TaskStatus MHD::BiermannCloseInterior(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  peos->ConsToPrim(
      u0, b0, w0, bcc0, false,
      indcs.is, indcs.ie, indcs.js, indcs.je, indcs.ks, indcs.ke);
  ptwo_temp->CloseBiermannStage(
      u0, w0, indcs.is, indcs.ie, indcs.js, indcs.je, indcs.ks, indcs.ke,
      !biermann_reduced_closure);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Reconstruct stage primitives, cell-centred B, and the 2T/material state on the
//! Biermann constraint manifold.  Total energy, face B, and electron energy are the
//! independent SSPRK variables; ion energy is the redundant algebraic component.

TaskStatus MHD::BiermannConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n1m1 = indcs.nx1 + 2*indcs.ng - 1;
  int n2m1 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng - 1 : 0;
  int n3m1 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng - 1 : 0;

  // An intermediate subcycle stage is consumed only by the Biermann face/edge
  // stencil.  Its widest access is one cell beyond the active domain, so closing
  // the second ghost layer repeats the expensive mixed-material inverse without
  // supplying a value that the next microstage can read.  The closing stage of
  // every Strang half-step still refreshes the complete ghost domain: the regular
  // MHD reconstruction that follows is allowed to consume both ghost layers.
  int il = 0, iu = n1m1;
  int jl = 0, ju = n2m1;
  int kl = 0, ku = n3m1;
  if (!biermann_stage_full_thermodynamics) {
    il = indcs.is - 1;
    iu = indcs.ie + 1;
    if (indcs.nx2 > 1) {
      jl = indcs.js - 1;
      ju = indcs.je + 1;
    }
    if (indcs.nx3 > 1) {
      kl = indcs.ks - 1;
      ku = indcs.ke + 1;
    }
  }
  peos->ConsToPrim(u0, b0, w0, bcc0, false, il, iu, jl, ju, kl, ku);
  ptwo_temp->CloseBiermannStage(
      u0, w0, il, iu, jl, ju, kl, ku,
      biermann_stage_full_thermodynamics);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! Clear exactly the communications posted by BiermannInitRecv.

TaskStatus MHD::BiermannClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_b->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }
  return pbval_b->ClearFluxSend();
}

TaskStatus MHD::BiermannClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_b->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }
  return pbval_b->ClearFluxRecv();
}

} // namespace mhd
