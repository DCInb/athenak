//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Quasineutral Ohm-Faraday system: E+u x B = eta*(curl B-J_fast,relative).
//! eta is magnetic diffusivity in code units. CT and refluxed Poynting flux use the
//! same reconciled edge electric field. See docs/FAST_ELECTRONS.md for the closure.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_physics.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace fast_electrons {
namespace {
void OutflowCurrent(MeshBlockPack *pack, DvceArray5D<Real> a, bool coarse) {
  const auto ind = pack->pmesh->mb_indcs;
  const int is = coarse ? ind.cis : ind.is, ie = coarse ? ind.cie : ind.ie;
  const int js = coarse ? ind.cjs : ind.js, je = coarse ? ind.cje : ind.je;
  const int ks = coarse ? ind.cks : ind.ks, ke = coarse ? ind.cke : ind.ke;
  const auto bcs = pack->pmb->mb_bcs.d_view;
  par_for("fe_current_outflow", DevExeSpace(), 0, pack->nmb_thispack-1,
          0, a.extent_int(2)-1, 0, a.extent_int(3)-1, 0, a.extent_int(4)-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    int ii = i, jj = j, kk = k;
    if (i < is && bcs(m, 0) == BoundaryFlag::outflow) ii = is;
    if (i > ie && bcs(m, 1) == BoundaryFlag::outflow) ii = ie;
    if (j < js && bcs(m, 2) == BoundaryFlag::outflow) jj = js;
    if (j > je && bcs(m, 3) == BoundaryFlag::outflow) jj = je;
    if (k < ks && bcs(m, 4) == BoundaryFlag::outflow) kk = ks;
    if (k > ke && bcs(m, 5) == BoundaryFlag::outflow) kk = ke;
    if (ii != i || jj != j || kk != k) {
      for (int d = 0; d < 3; ++d) a(m, d, k, j, i) = a(m, d, kk, jj, ii);
    }
  });
}
}  // namespace

void FastElectrons::FillCurrentBoundaries() {
  const bool multilevel = pmy_pack_->pmesh->multilevel;
  OutflowCurrent(pmy_pack_, current_, false);
  if (multilevel) {
    Kokkos::deep_copy(coarse_current_, 0.0);
    pmy_pack_->pmesh->pmr->RestrictCC(current_, coarse_current_);
    OutflowCurrent(pmy_pack_, coarse_current_, true);
  }
  current_bvals_->InitRecv(3);
  current_bvals_->PackAndSendCC(current_, coarse_current_);
  current_bvals_->ClearSend();
  current_bvals_->ClearRecv();
  current_bvals_->RecvAndUnpackCC(current_, coarse_current_);
  OutflowCurrent(pmy_pack_, current_, false);
  if (multilevel) {
    current_bvals_->FillCoarseInBndryCC(current_, coarse_current_);
    OutflowCurrent(pmy_pack_, coarse_current_, true);
    current_bvals_->ProlongateCC(current_, coarse_current_);
    OutflowCurrent(pmy_pack_, current_, false);
  }
}

void FastElectrons::BuildFields() {
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  const bool multi = pmy_pack_->pmesh->multi_d, three = pmy_pack_->pmesh->three_d;
  auto b = pmy_pack_->pmhd->b0;
  auto bcc = pmy_pack_->pmhd->bcc0;
  auto w = pmy_pack_->pmhd->w0;
  auto jfast = current_;
  auto e = emf_;
  auto data = cell_data;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const Real eta = diffusivity_;
  if (eta == 0.0) {
    // NGP particles gather only active cells. Ideal induction belongs to MHD,
    // so this local force needs neither current halos nor edge-EMF exchanges.
    par_for("fe_ideal_electric", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
            ind.js, ind.je, ind.is, ind.ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int d = 0; d < 3; ++d) {
        const int a = (d+1)%3, c = (d+2)%3;
        data(m, 8+d, k, j, i) = -w(m, IVX+a, k, j, i)*bcc(m, c, k, j, i)+
                                w(m, IVX+c, k, j, i)*bcc(m, a, k, j, i);
      }
    });
    return;
  }
  FillCurrentBoundaries();
  par_for("fe_ohm1", DevExeSpace(), 0, nmb1, ind.ks, ind.ke+1,
          ind.js, ind.je+1, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int jm = multi ? j-1 : 0, jp = multi ? j : 0;
    const int km = three ? k-1 : 0, kp = three ? k : 0;
    Real curl = 0.0;
    if (multi) curl += (b.x3f(m, k, j, i)-b.x3f(m, k, j-1, i))/size(m).dx2;
    if (three) curl -= (b.x2f(m, k, j, i)-b.x2f(m, k-1, j, i))/size(m).dx3;
    const Real fast = 0.25*(jfast(m, 0, km, jm, i)+jfast(m, 0, km, jp, i)+
                             jfast(m, 0, kp, jm, i)+jfast(m, 0, kp, jp, i));
    e.x1e(m, k, j, i) = eta*(curl-fast);
  });
  par_for("fe_ohm2", DevExeSpace(), 0, nmb1, ind.ks, ind.ke+1,
          ind.js, ind.je, ind.is, ind.ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int km = three ? k-1 : 0, kp = three ? k : 0;
    Real curl = -(b.x3f(m, k, j, i)-b.x3f(m, k, j, i-1))/size(m).dx1;
    if (three) curl += (b.x1f(m, k, j, i)-b.x1f(m, k-1, j, i))/size(m).dx3;
    const Real fast = 0.25*(jfast(m, 1, km, j, i-1)+jfast(m, 1, km, j, i)+
                             jfast(m, 1, kp, j, i-1)+jfast(m, 1, kp, j, i));
    e.x2e(m, k, j, i) = eta*(curl-fast);
  });
  par_for("fe_ohm3", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
          ind.js, ind.je+1, ind.is, ind.ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int jm = multi ? j-1 : 0, jp = multi ? j : 0;
    Real curl = (b.x2f(m, k, j, i)-b.x2f(m, k, j, i-1))/size(m).dx1;
    if (multi) curl -= (b.x1f(m, k, j, i)-b.x1f(m, k, j-1, i))/size(m).dx2;
    const Real fast = 0.25*(jfast(m, 2, k, jm, i-1)+jfast(m, 2, k, jm, i)+
                             jfast(m, 2, k, jp, i-1)+jfast(m, 2, k, jp, i));
    e.x3e(m, k, j, i) = eta*(curl-fast);
  });
  auto boundary = pmy_pack_->pmhd->pbval_b;
  boundary->InitFluxRecv(3);
  boundary->PackAndSendFluxFC(emf_);
  boundary->ClearFluxSend();
  boundary->ClearFluxRecv();
  while (boundary->RecvAndUnpackFluxFC(emf_) == TaskStatus::incomplete) {}
  par_for("fe_gather_electric", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
          ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real electric[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 2; ++a) {
      for (int c = 0; c < 2; ++c) {
        electric[0] += 0.25*e.x1e(m, k+a, j+c, i);
        electric[1] += 0.25*e.x2e(m, k+a, j, i+c);
        electric[2] += 0.25*e.x3e(m, k, j+a, i+c);
      }
    }
    for (int d = 0; d < 3; ++d) {
      const int a = (d+1)%3, c = (d+2)%3;
      // Ideal induction is already evolved by the MHD integrator; only the particle
      // force needs this ideal term during the split hybrid operator.
      data(m, 8+d, k, j, i) = electric[d]-w(m, IVX+a, k, j, i)*bcc(m, c, k, j, i)+
                                          w(m, IVX+c, k, j, i)*bcc(m, a, k, j, i);
    }
  });
}

Real FastElectrons::FieldTimeStep(bool include_electric) {
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  const bool multi = pmy_pack_->pmesh->multi_d, three = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto b = pmy_pack_->pmhd->bcc0;
  auto data = cell_data;
  const Real eta = diffusivity_, c = kLightSpeedCgs*time_cgs_/length_cgs_;
  const Real gyro = kElectronChargeCgs*bunit_*time_cgs_/(kElectronMassCgs*kLightSpeedCgs);
  const Real q = cutoff_/kElectronRestEnergyCgs;
  const Real pmin = std::sqrt(q)*std::sqrt(q+2.0);
  Real diffusion_rate = 0.0, particle_rate = 0.0;
  Kokkos::parallel_reduce("fe_field_dt", Kokkos::RangePolicy<DevExeSpace>(
      0, pmy_pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3),
  KOKKOS_LAMBDA(int idx, Real &diffusion_max, Real &particle_max) {
    const int i = idx%ind.nx1+ind.is; idx /= ind.nx1;
    const int j = idx%ind.nx2+ind.js; idx /= ind.nx2;
    const int k = idx%ind.nx3+ind.ks, m = idx/ind.nx3;
    const Real ix = 1.0/size(m).dx1, iy = multi ? 1.0/size(m).dx2 : 0.0;
    const Real iz = three ? 1.0/size(m).dx3 : 0.0;
    const Real diffusion = 2.0*eta*(ix*ix+iy*iy+iz*iz);
    Real r = fmax(diffusion, c*(ix+iy+iz));
    Real b2 = 0.0, e2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      b2 += b(m, d, k, j, i)*b(m, d, k, j, i);
      if (include_electric) e2 += data(m, 8+d, k, j, i)*data(m, 8+d, k, j, i);
    }
    r = fmax(r, gyro*sqrt(b2));
    if (include_electric) r = fmax(r, gyro*sqrt(e2)/(c*pmin));
    if (!Kokkos::isfinite(r)) Kokkos::abort("Nonfinite hybrid field timestep");
    diffusion_max = fmax(diffusion_max, diffusion);
    particle_max = fmax(particle_max, r);
  }, Kokkos::Max<Real>(diffusion_rate), Kokkos::Max<Real>(particle_rate));
  // Reduce both limits and particle presence together. Empty ranks must still
  // contribute their mesh/field limits: particles may enter them this substep.
  Real rates[3] = {diffusion_rate, particle_rate,
                   device_packets_.extent(0) == 0 ? 0.0 : 1.0};
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, rates, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  const Real rate = rates[2] > 0.0 ? rates[1] : rates[0];
  return rate > 0.0 ? field_cfl_/rate : std::numeric_limits<Real>::max();
}

void FastElectrons::AdvanceFields(Real dt) {
  if (diffusivity_ == 0.0) return;
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  const bool multi = pmy_pack_->pmesh->multi_d, three = pmy_pack_->pmesh->three_d;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto b = pmy_pack_->pmhd->b0;
  auto bc = pmy_pack_->pmhd->bcc0;
  auto e = emf_;
  auto flux = energy_flux_;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  par_for("fe_poynting1", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
          ind.js, ind.je, ind.is, ind.ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real ey = 0.5*(e.x2e(m, k, j, i)+e.x2e(m, k+1, j, i));
    const Real ez = 0.5*(e.x3e(m, k, j, i)+e.x3e(m, k, j+1, i));
    const Real by = 0.5*(bc(m, 1, k, j, i-1)+bc(m, 1, k, j, i));
    const Real bz = 0.5*(bc(m, 2, k, j, i-1)+bc(m, 2, k, j, i));
    flux.x1f(m, 0, k, j, i) = ey*bz-ez*by;
  });
  if (multi) {
    par_for("fe_poynting2", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
            ind.js, ind.je+1, ind.is, ind.ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real ex = 0.5*(e.x1e(m, k, j, i)+e.x1e(m, k+1, j, i));
      const Real ez = 0.5*(e.x3e(m, k, j, i)+e.x3e(m, k, j, i+1));
      const Real bx = 0.5*(bc(m, 0, k, j-1, i)+bc(m, 0, k, j, i));
      const Real bz = 0.5*(bc(m, 2, k, j-1, i)+bc(m, 2, k, j, i));
      flux.x2f(m, 0, k, j, i) = ez*bx-ex*bz;
    });
    par_for("fe_ct1", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
            ind.js, ind.je, ind.is, ind.ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b.x1f(m, k, j, i) -= dt*(e.x3e(m, k, j+1, i)-e.x3e(m, k, j, i))/size(m).dx2;
      if (three) b.x1f(m, k, j, i) +=
          dt*(e.x2e(m, k+1, j, i)-e.x2e(m, k, j, i))/size(m).dx3;
    });
  }
  if (three) {
    par_for("fe_poynting3", DevExeSpace(), 0, nmb1, ind.ks, ind.ke+1,
            ind.js, ind.je, ind.is, ind.ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real ex = 0.5*(e.x1e(m, k, j, i)+e.x1e(m, k, j+1, i));
      const Real ey = 0.5*(e.x2e(m, k, j, i)+e.x2e(m, k, j, i+1));
      const Real bx = 0.5*(bc(m, 0, k-1, j, i)+bc(m, 0, k, j, i));
      const Real by = 0.5*(bc(m, 1, k-1, j, i)+bc(m, 1, k, j, i));
      flux.x3f(m, 0, k, j, i) = ex*by-ey*bx;
    });
  }
  par_for("fe_ct2", DevExeSpace(), 0, nmb1, ind.ks, ind.ke,
          ind.js, ind.je+1, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b.x2f(m, k, j, i) += dt*(e.x3e(m, k, j, i+1)-e.x3e(m, k, j, i))/size(m).dx1;
    if (three) b.x2f(m, k, j, i) -=
        dt*(e.x1e(m, k+1, j, i)-e.x1e(m, k, j, i))/size(m).dx3;
  });
  par_for("fe_ct3", DevExeSpace(), 0, nmb1, ind.ks, ind.ke+1,
          ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b.x3f(m, k, j, i) -= dt*(e.x2e(m, k, j, i+1)-e.x2e(m, k, j, i))/size(m).dx1;
    if (multi) b.x3f(m, k, j, i) +=
        dt*(e.x1e(m, k, j+1, i)-e.x1e(m, k, j, i))/size(m).dx2;
  });
  if (pmy_pack_->pmesh->multilevel) {
    current_bvals_->InitFluxRecv(1);
    current_bvals_->PackAndSendFluxCC(energy_flux_);
    current_bvals_->ClearFluxSend();
    current_bvals_->ClearFluxRecv();
    while (current_bvals_->RecvAndUnpackFluxCC(energy_flux_) == TaskStatus::incomplete) {}
  }
}
}  // namespace fast_electrons
