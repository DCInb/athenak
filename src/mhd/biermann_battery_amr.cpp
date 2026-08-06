//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file biermann_battery_amr.cpp
//! \brief Composite-AMR edge EMFs and matching Biermann Poynting reflux state.

#include <cmath>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "mhd/biermann_battery.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "two_temperature/biermann_closure.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

struct CompositeCellState {
  Real density;
  Real electron_pressure;
  Real electron_density;
};

KOKKOS_INLINE_FUNCTION
Real CompositeInverseLogMean(const Real a, const Real b) {
  const Real mean = 0.5*(a+b);
  const Real r = (b-a)/(a+b);
  const Real ar = fabs(r);
  if (ar < 1.0e-4) {
    const Real r2 = r*r;
    return (1.0+r2*(1.0/3.0+r2*(1.0/5.0+r2/7.0)))/mean;
  }
  return log(b/a)/(b-a);
}

struct CompositeVertexState {
  Real electron_pressure;
  Real electron_density;
};

//! Device-copyable evaluator for the coarse representation owned by one fine block.
//! Coarse data are authoritative in the interior.  At a physical-domain face, build
//! the missing coarse ghost state with the scalar semantics of that boundary.  Simple
//! extrapolative boundaries use the adjacent coarse cell; other boundary traces are
//! conservatively coarsened from the already boundary-filled fine register.
struct CompositeAMREvaluator {
  DvceArray5D<Real> coarse_conserved;
  DvceArray5D<Real> fine_conserved;
  DvceFaceFld4D<Real> coarse_magnetic;
  DvceFaceFld4D<Real> fine_magnetic;
  DvceArray2D<BoundaryFlag> mb_bcs;
  two_temperature::BiermannEndpointClosure closure;
  int ion_index;
  int electron_index;
  Real electron_fraction;
  Real minimum_electron_fraction;
  Real minimum_positive;
  bool two_d;
  int is, js, ks;
  int cis, cie, cjs, cje, cks, cke;

  KOKKOS_INLINE_FUNCTION
  static void Shift(const int direction, const int offset,
                    int &k, int &j, int &i) {
    if (direction == 0) {
      i += offset;
    } else if (direction == 1) {
      j += offset;
    } else {
      k += offset;
    }
  }

  KOKKOS_INLINE_FUNCTION
  static bool IsPhysicalBoundary(const BoundaryFlag flag) {
    return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
           flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef;
  }

  KOKKOS_INLINE_FUNCTION
  static bool IsScalarExtrapolation(const BoundaryFlag flag) {
    return flag == BoundaryFlag::reflect || flag == BoundaryFlag::outflow ||
           flag == BoundaryFlag::diode;
  }

  KOKKOS_INLINE_FUNCTION
  CompositeCellState CellFromValues(
      const Real raw_density,
      const Real momentum1, const Real momentum2, const Real momentum3,
      const Real total_energy, const Real raw_ion_energy,
      const Real raw_electron_energy, const Real *raw_material_densities,
      const Real bcc1, const Real bcc2, const Real bcc3) const {
    const two_temperature::BiermannClosedState closed = closure.CloseConserved(
        raw_density, momentum1, momentum2, momentum3, total_energy,
        raw_ion_energy, raw_electron_energy, raw_material_densities,
        bcc1, bcc2, bcc3);
    CompositeCellState result;
    result.density = closed.density;

    if (closure.use_tabular) {
      const materials::MaterialThermodynamicState state =
          closure.mixture.StateFromRhoSpecificEnergiesNoSound(
              result.density, closed.ion_energy/result.density,
              closed.electron_energy/result.density,
              closed.composition);
      result.electron_pressure = fmax(state.electron_pressure, 0.0);
      const Real conversion =
          materials::MaterialMixtureDevice::atomic_mass_unit_cgs/
          closure.mixture.density_to_cgs;
      const Real physical = fmax(
          state.electron_number_density_cgs*conversion, 0.0);
      const Real threshold = fmax(
          minimum_electron_fraction*result.density, minimum_positive);
      result.electron_density = fmax(physical, threshold);
      Real activation = 1.0;
      if (physical <= threshold) {
        activation = 0.0;
      } else if (physical < 2.0*threshold) {
        const Real x = physical/threshold-1.0;
        activation = x*x*(3.0-2.0*x);
      }
      // This is the same neutral-state treatment as the production edge cochain:
      // activation is part of the pressure coordinate, not an edge mask.
      result.electron_pressure *= activation;
    } else {
      result.electron_pressure =
          closure.gamma_minus_one*closed.electron_energy;
      if (closure.use_materials) {
        result.electron_density = closure.mixture.ElectronNumberDensity(
            result.density, closed.composition);
      } else {
        result.electron_density = electron_fraction*result.density;
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  CompositeCellState CellFrom(const DvceArray5D<Real> &conserved,
                              const DvceFaceFld4D<Real> &magnetic,
                              const int m, const int k,
                              const int j, const int i) const {
    Real material_densities[materials::kMaxMaterials-1] = {};
    if (closure.use_materials) {
      for (int n = 0; n < closure.mixture.nmaterials-1; ++n) {
        material_densities[n] =
            conserved(m, closure.mixture.scalar_indices[n], k, j, i);
      }
    }
    const Real bcc1 = 0.5*(
        magnetic.x1f(m, k, j, i)+magnetic.x1f(m, k, j, i+1));
    const Real bcc2 = 0.5*(
        magnetic.x2f(m, k, j, i)+magnetic.x2f(m, k, j+1, i));
    const Real bcc3 = 0.5*(
        magnetic.x3f(m, k, j, i)+magnetic.x3f(m, k+1, j, i));
    return CellFromValues(
        conserved(m, IDN, k, j, i),
        conserved(m, IM1, k, j, i), conserved(m, IM2, k, j, i),
        conserved(m, IM3, k, j, i), conserved(m, IEN, k, j, i),
        conserved(m, ion_index, k, j, i),
        conserved(m, electron_index, k, j, i), material_densities,
        bcc1, bcc2, bcc3);
  }

  KOKKOS_INLINE_FUNCTION
  CompositeCellState CellFromRestrictedFine(
      const int m, const int k, const int j, const int i) const {
    const int fi = 2*(i-cis)+is;
    const int fj = 2*(j-cjs)+js;
    const int fk = two_d ? ks : 2*(k-cks)+ks;
    const int nk = two_d ? 1 : 2;
    const Real weight = two_d ? 0.25 : 0.125;
    Real density = 0.0;
    Real momentum1 = 0.0;
    Real momentum2 = 0.0;
    Real momentum3 = 0.0;
    Real total_energy = 0.0;
    Real ion_energy = 0.0;
    Real electron_energy = 0.0;
    Real material_densities[materials::kMaxMaterials-1] = {};
    for (int dk=0; dk<nk; ++dk) {
      for (int dj=0; dj<2; ++dj) {
        for (int di=0; di<2; ++di) {
          density += weight*fine_conserved(m, IDN, fk+dk, fj+dj, fi+di);
          momentum1 += weight*fine_conserved(m, IM1, fk+dk, fj+dj, fi+di);
          momentum2 += weight*fine_conserved(m, IM2, fk+dk, fj+dj, fi+di);
          momentum3 += weight*fine_conserved(m, IM3, fk+dk, fj+dj, fi+di);
          total_energy += weight*fine_conserved(m, IEN, fk+dk, fj+dj, fi+di);
          ion_energy +=
              weight*fine_conserved(m, ion_index, fk+dk, fj+dj, fi+di);
          electron_energy +=
              weight*fine_conserved(m, electron_index, fk+dk, fj+dj, fi+di);
          if (closure.use_materials) {
            for (int n = 0; n < closure.mixture.nmaterials-1; ++n) {
              material_densities[n] += weight*fine_conserved(
                  m, closure.mixture.scalar_indices[n], fk+dk, fj+dj, fi+di);
            }
          }
        }
      }
    }
    Real bcc1 = 0.0;
    Real bcc2 = 0.0;
    Real bcc3 = 0.0;
    if (two_d) {
      bcc1 = 0.25*(
          fine_magnetic.x1f(m, ks, fj, fi)+
          fine_magnetic.x1f(m, ks, fj+1, fi)+
          fine_magnetic.x1f(m, ks, fj, fi+2)+
          fine_magnetic.x1f(m, ks, fj+1, fi+2));
      bcc2 = 0.25*(
          fine_magnetic.x2f(m, ks, fj, fi)+
          fine_magnetic.x2f(m, ks, fj, fi+1)+
          fine_magnetic.x2f(m, ks, fj+2, fi)+
          fine_magnetic.x2f(m, ks, fj+2, fi+1));
      bcc3 = 0.25*(
          fine_magnetic.x3f(m, ks, fj, fi)+
          fine_magnetic.x3f(m, ks, fj, fi+1)+
          fine_magnetic.x3f(m, ks, fj+1, fi)+
          fine_magnetic.x3f(m, ks, fj+1, fi+1));
    } else {
      for (int dk=0; dk<2; ++dk) {
        for (int dj=0; dj<2; ++dj) {
          bcc1 += 0.125*(
              fine_magnetic.x1f(m, fk+dk, fj+dj, fi)+
              fine_magnetic.x1f(m, fk+dk, fj+dj, fi+2));
        }
      }
      for (int dk=0; dk<2; ++dk) {
        for (int di=0; di<2; ++di) {
          bcc2 += 0.125*(
              fine_magnetic.x2f(m, fk+dk, fj, fi+di)+
              fine_magnetic.x2f(m, fk+dk, fj+2, fi+di));
        }
      }
      for (int dj=0; dj<2; ++dj) {
        for (int di=0; di<2; ++di) {
          bcc3 += 0.125*(
              fine_magnetic.x3f(m, fk, fj+dj, fi+di)+
              fine_magnetic.x3f(m, fk+2, fj+dj, fi+di));
        }
      }
    }
    return CellFromValues(
        density, momentum1, momentum2, momentum3, total_energy,
        ion_energy, electron_energy, material_densities, bcc1, bcc2, bcc3);
  }

  KOKKOS_INLINE_FUNCTION
  CompositeVertexState VertexFrom(const DvceArray5D<Real> &conserved,
                                  const int m, const int k,
                                  const int j, const int i) const {
    CompositeVertexState result{0.0, 0.0};
    const Real weight = two_d ? 0.25 : 0.125;
    const int dk_min = two_d ? 0 : -1;
    for (int dk=dk_min; dk<=0; ++dk) {
      for (int dj=-1; dj<=0; ++dj) {
        for (int di=-1; di<=0; ++di) {
          const CompositeCellState cell =
              CellFrom(conserved, coarse_magnetic, m, k+dk, j+dj, i+di);
          result.electron_pressure += weight*cell.electron_pressure;
          result.electron_density += weight*cell.electron_density;
        }
      }
    }
    result.electron_density = fmax(result.electron_density, minimum_positive);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  CompositeVertexState ExtrapolatedCoarseVertex(
      const int m, const int k, const int j, const int i) const {
    CompositeVertexState result{0.0, 0.0};
    const Real weight = two_d ? 0.25 : 0.125;
    const int dk_min = two_d ? 0 : -1;
    const bool inner_x1 = i == cis &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x1));
    const bool outer_x1 = i == cie+1 &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x1));
    const bool inner_x2 = j == cjs &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x2));
    const bool outer_x2 = j == cje+1 &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x2));
    const bool inner_x3 = !two_d && k == cks &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x3));
    const bool outer_x3 = !two_d && k == cke+1 &&
        IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x3));
    for (int dk=dk_min; dk<=0; ++dk) {
      for (int dj=-1; dj<=0; ++dj) {
        for (int di=-1; di<=0; ++di) {
          int kc = k+dk;
          int jc = j+dj;
          int ic = i+di;
          bool use_fine_trace = false;
          if (inner_x1 && ic < cis) {
            if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::inner_x1))) {
              ic = cis;
            } else {
              use_fine_trace = true;
            }
          }
          if (outer_x1 && ic > cie) {
            if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::outer_x1))) {
              ic = cie;
            } else {
              use_fine_trace = true;
            }
          }
          if (inner_x2 && jc < cjs) {
            if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::inner_x2))) {
              jc = cjs;
            } else {
              use_fine_trace = true;
            }
          }
          if (outer_x2 && jc > cje) {
            if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::outer_x2))) {
              jc = cje;
            } else {
              use_fine_trace = true;
            }
          }
          if (!two_d) {
            if (inner_x3 && kc < cks) {
              if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::inner_x3))) {
                kc = cks;
              } else {
                use_fine_trace = true;
              }
            }
            if (outer_x3 && kc > cke) {
              if (IsScalarExtrapolation(mb_bcs(m, BoundaryFace::outer_x3))) {
                kc = cke;
              } else {
                use_fine_trace = true;
              }
            }
          }
          const CompositeCellState cell = use_fine_trace
              ? CellFromRestrictedFine(m, k+dk, j+dj, i+di)
              : CellFrom(coarse_conserved, coarse_magnetic, m, kc, jc, ic);
          result.electron_pressure += weight*cell.electron_pressure;
          result.electron_density += weight*cell.electron_density;
        }
      }
    }
    result.electron_density = fmax(result.electron_density, minimum_positive);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  CompositeVertexState Vertex(const int m, const int k,
                              const int j, const int i) const {
    const bool physical_x1 =
        (i == cis && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x1))) ||
        (i == cie+1 && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x1)));
    const bool physical_x2 =
        (j == cjs && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x2))) ||
        (j == cje+1 && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x2)));
    const bool physical_x3 = !two_d && (
        (k == cks && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::inner_x3))) ||
        (k == cke+1 && IsPhysicalBoundary(mb_bcs(m, BoundaryFace::outer_x3))));
    if (physical_x1 || physical_x2 || physical_x3) {
      return ExtrapolatedCoarseVertex(m, k, j, i);
    }
    return VertexFrom(coarse_conserved, m, k, j, i);
  }
};

} // namespace

namespace mhd {

//----------------------------------------------------------------------------------------
//! \brief Reconcile every fine/coarse Biermann edge mortar before FC flux exchange.

void BiermannBattery::ReconcileCompositeAMREMFs(DvceEdgeFld4D<Real> &efld) {
  if (!pmy_pack_->pmesh->multilevel) return;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks, ke = indcs.ke;
  const int cis = indcs.cis, cie = indcs.cie;
  const int cjs = indcs.cjs, cje = indcs.cje;
  const int cks = indcs.cks, cke = indcs.cke;
  const int nmb = pmy_pack_->nmb_thispack;
  const int nnghbr = pmy_pack_->pmb->nnghbr;
  const bool two_d = pmy_pack_->pmesh->two_d;
  auto nghbr = pmy_pack_->pmb->nghbr.d_view;
  auto mblev = pmy_pack_->pmb->mb_lev.d_view;
  auto &sbuf = pmy_pack_->pmhd->pbval_b->sendbuf;
  auto size = pmy_pack_->pmb->mb_size;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;

  const auto &eos = pmy_pack_->pmhd->peos->eos_data;
  const two_temperature::BiermannEndpointClosure closure{
      material_mixture_, gamma_minus_one_, eos.dfloor, eos.pfloor, eos.tfloor,
      eos.sfloor, eos.sigma_max, pmy_pack_->pmhd->dual_energy_eta1,
      pmy_pack_->pmhd->use_dual_energy, use_material_mixture_,
      use_material_mixture_ && material_mixture_.UsesTabularEOS()};
  CompositeAMREvaluator evaluator{
      pmy_pack_->pmhd->coarse_u0, pmy_pack_->pmhd->u0,
      pmy_pack_->pmhd->coarse_b0, pmy_pack_->pmhd->b0,
      pmy_pack_->pmb->mb_bcs.d_view, closure,
      pmy_pack_->pmhd->ptwo_temp->iion, iele_, electron_fraction_,
      minimum_electron_fraction_, std::numeric_limits<Real>::min(),
      two_d, is, js, ks, cis, cie, cjs, cje, cks, cke};

  if (two_d) {
    auto pressure = pressure_vertex_;
    auto electron_density = electron_density_vertex_;
    const Real coeff = coefficient;

    // Build one state at every fine vertex on a coarse/fine mortar.  Coincident
    // vertices use the synchronized coarse representation.  The midpoint is linear
    // in both p_e and n_e, so the two endpoint/log-mean segment integrals add exactly
    // to the corresponding coarse edge integral.  Rebuilding the incident normal
    // edges below is essential: shifting only the tangential pair creates circulation
    // in the first fine cell, especially where orthogonal refinement faces meet.
    Kokkos::TeamPolicy<> vertex_policy(DevExeSpace(), nmb, Kokkos::AUTO);
    Kokkos::parallel_for(
        "biermann_composite_amr_vertex_2d", vertex_policy,
        KOKKOS_LAMBDA(TeamMember_t tmember) {
          const int m = tmember.league_rank();
          for (int n=0; n<nnghbr && n<16; ++n) {
            const bool coarse_neighbor =
                nghbr(m, n).gid >= 0 && nghbr(m, n).lev < mblev(m);
            if (!coarse_neighbor) {
              tmember.team_barrier();
              continue;
            }

            if (n < 8) {
              // x1 face: intervals and midpoint vertices run in x2.
              const auto q = sbuf[n].iflux_coar[1];
              const int ni = q.bie-q.bis+1;
              const int nj = q.bje-q.bjs+1;
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(tmember, nj*ni),
                  [&](const int idx) {
                    const int cj = q.bjs+idx/ni;
                    const int ci = q.bis+idx%ni;
                    const int fj = 2*cj-cjs;
                    const int fi = 2*ci-cis;
                    const CompositeVertexState left =
                        evaluator.Vertex(m, q.bks, cj, ci);
                    const CompositeVertexState right =
                        evaluator.Vertex(m, q.bks, cj+1, ci);
                    pressure(m, ks, fj, fi) = left.electron_pressure;
                    electron_density(m, ks, fj, fi) = left.electron_density;
                    pressure(m, ks, fj+1, fi) = 0.5*(
                        left.electron_pressure+right.electron_pressure);
                    electron_density(m, ks, fj+1, fi) = 0.5*(
                        left.electron_density+right.electron_density);
                  });
              tmember.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(tmember, ni),
                  [&](const int idx) {
                    const int ci = q.bis+idx;
                    const int fj = 2*(q.bje+1)-cjs;
                    const int fi = 2*ci-cis;
                    const CompositeVertexState right =
                        evaluator.Vertex(m, q.bks, q.bje+1, ci);
                    pressure(m, ks, fj, fi) = right.electron_pressure;
                    electron_density(m, ks, fj, fi) = right.electron_density;
                  });
            } else {
              // x2 face: intervals and midpoint vertices run in x1.
              const auto q = sbuf[n].iflux_coar[0];
              const int ni = q.bie-q.bis+1;
              const int nj = q.bje-q.bjs+1;
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(tmember, nj*ni),
                  [&](const int idx) {
                    const int cj = q.bjs+idx/ni;
                    const int ci = q.bis+idx%ni;
                    const int fj = 2*cj-cjs;
                    const int fi = 2*ci-cis;
                    const CompositeVertexState left =
                        evaluator.Vertex(m, q.bks, cj, ci);
                    const CompositeVertexState right =
                        evaluator.Vertex(m, q.bks, cj, ci+1);
                    pressure(m, ks, fj, fi) = left.electron_pressure;
                    electron_density(m, ks, fj, fi) = left.electron_density;
                    pressure(m, ks, fj, fi+1) = 0.5*(
                        left.electron_pressure+right.electron_pressure);
                    electron_density(m, ks, fj, fi+1) = 0.5*(
                        left.electron_density+right.electron_density);
                  });
              tmember.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange<>(tmember, nj),
                  [&](const int idx) {
                    const int cj = q.bjs+idx;
                    const int fj = 2*cj-cjs;
                    const int fi = 2*(q.bie+1)-cis;
                    const CompositeVertexState right =
                        evaluator.Vertex(m, q.bks, cj, q.bie+1);
                    pressure(m, ks, fj, fi) = right.electron_pressure;
                    electron_density(m, ks, fj, fi) = right.electron_density;
                  });
            }
            tmember.team_barrier();
          }
        });

    par_for(
        "biermann_composite_amr_edge_e1_2d", DevExeSpace(), 0, nmb-1,
        indcs.js, indcs.je+1, indcs.is, indcs.ie,
        KOKKOS_LAMBDA(const int m, const int j, const int i) {
          const Real inv_ne = CompositeInverseLogMean(
              electron_density(m, ks, j, i),
              electron_density(m, ks, j, i+1));
          const Real value = -coeff*(pressure(m, ks, j, i+1)-
              pressure(m, ks, j, i))*inv_ne/size.d_view(m).dx1;
          e1(m, ks, j, i) = value;
          e1(m, ke+1, j, i) = value;
        });
    par_for(
        "biermann_composite_amr_edge_e2_2d", DevExeSpace(), 0, nmb-1,
        indcs.js, indcs.je, indcs.is, indcs.ie+1,
        KOKKOS_LAMBDA(const int m, const int j, const int i) {
          const Real inv_ne = CompositeInverseLogMean(
              electron_density(m, ks, j, i),
              electron_density(m, ks, j+1, i));
          const Real value = -coeff*(pressure(m, ks, j+1, i)-
              pressure(m, ks, j, i))*inv_ne/size.d_view(m).dx2;
          e2(m, ks, j, i) = value;
          e2(m, ke+1, j, i) = value;
        });
    return;
  }

  auto pressure = pressure_vertex_;
  auto electron_density = electron_density_vertex_;
  const Real coeff = coefficient;

  // Faces can meet at edges and corners, so one team owns every mortar neighbor of a
  // block and processes them in a deterministic order.  Each coarse edge writes its
  // starting endpoint and midpoint; a separate pass writes only the terminal endpoint,
  // avoiding duplicate writes between adjacent intervals on CUDA.
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb, Kokkos::AUTO);
  Kokkos::parallel_for(
      "biermann_composite_amr_vertex_3d", policy,
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int m = tmember.league_rank();
        const auto set_edge_vertices = [&](const int component,
                                           const MeshBufferIndcs q) {
          const int ni = q.bie-q.bis+1;
          const int nj = q.bje-q.bjs+1;
          const int nk = q.bke-q.bks+1;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(tmember, nk*nj*ni),
              [&](const int idx) {
                const int ck = q.bks+idx/(nj*ni);
                const int rem = idx%(nj*ni);
                const int cj = q.bjs+rem/ni;
                const int ci = q.bis+rem%ni;
                int ckr = ck, cjr = cj, cir = ci;
                CompositeAMREvaluator::Shift(component, 1, ckr, cjr, cir);
                const CompositeVertexState left =
                    evaluator.Vertex(m, ck, cj, ci);
                const CompositeVertexState right =
                    evaluator.Vertex(m, ckr, cjr, cir);
                const int fk = 2*ck-cks;
                const int fj = 2*cj-cjs;
                const int fi = 2*ci-cis;
                int fkm = fk, fjm = fj, fim = fi;
                CompositeAMREvaluator::Shift(component, 1, fkm, fjm, fim);
                pressure(m, fk, fj, fi) = left.electron_pressure;
                electron_density(m, fk, fj, fi) = left.electron_density;
                pressure(m, fkm, fjm, fim) = 0.5*(
                    left.electron_pressure+right.electron_pressure);
                electron_density(m, fkm, fjm, fim) = 0.5*(
                    left.electron_density+right.electron_density);
              });
          tmember.team_barrier();

          const int nt = (component == 0) ? nk*nj :
                         ((component == 1) ? nk*ni : nj*ni);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(tmember, nt),
              [&](const int idx) {
                int ck, cj, ci;
                if (component == 0) {
                  ck = q.bks+idx/nj;
                  cj = q.bjs+idx%nj;
                  ci = q.bie+1;
                } else if (component == 1) {
                  ck = q.bks+idx/ni;
                  cj = q.bje+1;
                  ci = q.bis+idx%ni;
                } else {
                  ck = q.bke+1;
                  cj = q.bjs+idx/ni;
                  ci = q.bis+idx%ni;
                }
                const int fk = 2*ck-cks;
                const int fj = 2*cj-cjs;
                const int fi = 2*ci-cis;
                const CompositeVertexState right =
                    evaluator.Vertex(m, ck, cj, ci);
                pressure(m, fk, fj, fi) = right.electron_pressure;
                electron_density(m, fk, fj, fi) = right.electron_density;
              });
          tmember.team_barrier();
        };

        for (int n=0; n<nnghbr && n<48; ++n) {
          const bool coarse_neighbor =
              nghbr(m, n).gid >= 0 && nghbr(m, n).lev < mblev(m);
          if (coarse_neighbor && n < 8) {
            set_edge_vertices(1, sbuf[n].iflux_coar[1]);
            set_edge_vertices(2, sbuf[n].iflux_coar[2]);
          } else if (coarse_neighbor && n < 16) {
            set_edge_vertices(0, sbuf[n].iflux_coar[0]);
            set_edge_vertices(2, sbuf[n].iflux_coar[2]);
          } else if (coarse_neighbor && n < 24) {
            set_edge_vertices(2, sbuf[n].iflux_coar[2]);
          } else if (coarse_neighbor && n < 32) {
            set_edge_vertices(0, sbuf[n].iflux_coar[0]);
            set_edge_vertices(1, sbuf[n].iflux_coar[1]);
          } else if (coarse_neighbor && n < 40) {
            set_edge_vertices(1, sbuf[n].iflux_coar[1]);
          } else if (coarse_neighbor) {
            set_edge_vertices(0, sbuf[n].iflux_coar[0]);
          }
          tmember.team_barrier();
        }
      });

  par_for(
      "biermann_composite_amr_edge_e1_3d", DevExeSpace(), 0, nmb-1,
      indcs.ks, indcs.ke+1, indcs.js, indcs.je+1, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real inv_ne = CompositeInverseLogMean(
            electron_density(m, k, j, i), electron_density(m, k, j, i+1));
        e1(m, k, j, i) = -coeff*(pressure(m, k, j, i+1)-
            pressure(m, k, j, i))*inv_ne/size.d_view(m).dx1;
      });
  par_for(
      "biermann_composite_amr_edge_e2_3d", DevExeSpace(), 0, nmb-1,
      indcs.ks, indcs.ke+1, indcs.js, indcs.je, indcs.is, indcs.ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real inv_ne = CompositeInverseLogMean(
            electron_density(m, k, j, i), electron_density(m, k, j+1, i));
        e2(m, k, j, i) = -coeff*(pressure(m, k, j+1, i)-
            pressure(m, k, j, i))*inv_ne/size.d_view(m).dx2;
      });
  par_for(
      "biermann_composite_amr_edge_e3_3d", DevExeSpace(), 0, nmb-1,
      indcs.ks, indcs.ke, indcs.js, indcs.je+1, indcs.is, indcs.ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real inv_ne = CompositeInverseLogMean(
            electron_density(m, k, j, i), electron_density(m, k+1, j, i));
        e3(m, k, j, i) = -coeff*(pressure(m, k+1, j, i)-
            pressure(m, k, j, i))*inv_ne/size.d_view(m).dx3;
      });
}

} // namespace mhd
