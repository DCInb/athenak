//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//! \brief Calculate 3D fluxes for hydro

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "hydro/rsolvers/advect_hyd.hpp"
#include "hydro/rsolvers/llf_hyd.hpp"
#include "hydro/rsolvers/hlle_hyd.hpp"
#include "hydro/rsolvers/hllc_hyd.hpp"
#include "hydro/rsolvers/roe_hyd.hpp"
#include "hydro/rsolvers/dual_eint_hyd.hpp"
#include "hydro/rsolvers/llf_srhyd.hpp"
#include "hydro/rsolvers/hlle_srhyd.hpp"
#include "hydro/rsolvers/hllc_srhyd.hpp"
#include "hydro/rsolvers/llf_grhyd.hpp"
#include "hydro/rsolvers/hlle_grhyd.hpp"
#include "two_temperature/two_temperature.hpp"

namespace hydro {

// The following three helpers mirror the ones in mhd/mhd_fluxes.cpp; keep them in sync.
// They supply the LLF flux with a face pressure and sound speed drawn from the tabulated
// material closure instead of from the gamma-law carrier EOS.

KOKKOS_INLINE_FUNCTION
Real CachedMaterialPressure(const DvceArray5D<Real> &thermodynamics,
                            const int m, const int k, const int j, const int i) {
  return thermodynamics(
             m, two_temperature::TwoTemperature::ion_pressure, k, j, i)+
         thermodynamics(
             m, two_temperature::TwoTemperature::electron_pressure, k, j, i);
}

KOKKOS_INLINE_FUNCTION
void ReconstructMaterialThermodynamicsX1(
    TeamMember_t const &member, const bool donor_cell,
    const int m, const int k, const int j, const int il, const int iu,
    const DvceArray5D<Real> &thermodynamics,
    ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  par_for_inner(member, il, iu, [&](const int i) {
    const Real pressure = CachedMaterialPressure(thermodynamics, m, k, j, i);
    const Real cs2 = thermodynamics(
        m, two_temperature::TwoTemperature::sound_speed_squared, k, j, i);
    if (donor_cell) {
      ql(material_total_pressure, i+1) = pressure;
      qr(material_total_pressure, i) = pressure;
      ql(material_sound_speed_squared, i+1) = cs2;
      qr(material_sound_speed_squared, i) = cs2;
    } else {
      PLM(CachedMaterialPressure(thermodynamics, m, k, j, i-1), pressure,
          CachedMaterialPressure(thermodynamics, m, k, j, i+1),
          ql(material_total_pressure, i+1), qr(material_total_pressure, i));
      PLM(thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k, j, i-1), cs2,
          thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k, j, i+1),
          ql(material_sound_speed_squared, i+1),
          qr(material_sound_speed_squared, i));
    }
  });
}

KOKKOS_INLINE_FUNCTION
void ReconstructMaterialThermodynamicsX2(
    TeamMember_t const &member, const bool donor_cell,
    const int m, const int k, const int j, const int il, const int iu,
    const DvceArray5D<Real> &thermodynamics,
    ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  par_for_inner(member, il, iu, [&](const int i) {
    const Real pressure = CachedMaterialPressure(thermodynamics, m, k, j, i);
    const Real cs2 = thermodynamics(
        m, two_temperature::TwoTemperature::sound_speed_squared, k, j, i);
    if (donor_cell) {
      ql_jp1(material_total_pressure, i) = pressure;
      qr_j(material_total_pressure, i) = pressure;
      ql_jp1(material_sound_speed_squared, i) = cs2;
      qr_j(material_sound_speed_squared, i) = cs2;
    } else {
      PLM(CachedMaterialPressure(thermodynamics, m, k, j-1, i), pressure,
          CachedMaterialPressure(thermodynamics, m, k, j+1, i),
          ql_jp1(material_total_pressure, i), qr_j(material_total_pressure, i));
      PLM(thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k, j-1, i), cs2,
          thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k, j+1, i),
          ql_jp1(material_sound_speed_squared, i),
          qr_j(material_sound_speed_squared, i));
    }
  });
}

KOKKOS_INLINE_FUNCTION
void ReconstructMaterialThermodynamicsX3(
    TeamMember_t const &member, const bool donor_cell,
    const int m, const int k, const int j, const int il, const int iu,
    const DvceArray5D<Real> &thermodynamics,
    ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  par_for_inner(member, il, iu, [&](const int i) {
    const Real pressure = CachedMaterialPressure(thermodynamics, m, k, j, i);
    const Real cs2 = thermodynamics(
        m, two_temperature::TwoTemperature::sound_speed_squared, k, j, i);
    if (donor_cell) {
      ql_kp1(material_total_pressure, i) = pressure;
      qr_k(material_total_pressure, i) = pressure;
      ql_kp1(material_sound_speed_squared, i) = cs2;
      qr_k(material_sound_speed_squared, i) = cs2;
    } else {
      PLM(CachedMaterialPressure(thermodynamics, m, k-1, j, i), pressure,
          CachedMaterialPressure(thermodynamics, m, k+1, j, i),
          ql_kp1(material_total_pressure, i), qr_k(material_total_pressure, i));
      PLM(thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k-1, j, i), cs2,
          thermodynamics(m, two_temperature::TwoTemperature::sound_speed_squared,
                         k+1, j, i),
          ql_kp1(material_sound_speed_squared, i),
          qr_k(material_sound_speed_squared, i));
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.

template <Hydro_RSolver rsolver_method_>
void Hydro::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*(indcs_.ng);

  int &nhyd_  = nhydro;
  int nvars = nhydro + nscalars;
  const bool use_dual_energy_ = use_dual_energy;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  const bool use_tabular_material_eos_ = use_tabular_material_eos;
  DvceArray5D<Real> material_thermodynamics_;
  if (use_tabular_material_eos_) {
    material_thermodynamics_ = ptwo_temp->thermodynamics;
  }
  const int nmaterial_fields_ = use_tabular_material_eos_
      ? nmaterial_face_fields : 0;
  const bool material_donor_cell_ =
      recon_method_ == ReconstructionMethod::dc;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                     ScrArray2D<Real>::shmem_size(nmaterial_fields_, ncells1)) * 2;
  int scr_level = 0;
  auto &flx1_ = uflx.x1f;
  auto &vf1_ = dual_vf.x1f;

  // set the loop limits for 1D/2D/3D problems
  int il = is, iu = ie+1, jl = js, ju = je, kl = ks, ku = ke;
  if (use_fofc) {
    il = is-1, iu = ie+2;
    if (pmy_pack->pmesh->two_d) {
      jl = js-1, ju = je+1, kl = ks, ku = ke;
    } else {
      jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  }

  par_for_outer("hflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku, jl, ju,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> material_left(
        member.team_scratch(scr_level), nmaterial_fields_, ncells1);
    ScrArray2D<Real> material_right(
        member.team_scratch(scr_level), nmaterial_fields_, ncells1);

    // Reconstruct qR[i] and qL[i+1]
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, il-1, iu, w0_, wl, wr);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member,eos_,extrema,true, m, k, j, il-1, iu, w0_, wl, wr);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos_, true, m, k, j, il-1, iu, w0_, wl, wr);
        break;
      default:
        break;
    }
    if (use_tabular_material_eos_) {
      ReconstructMaterialThermodynamicsX1(
          member, material_donor_cell_, m, k, j, il-1, iu,
          material_thermodynamics_, material_left, material_right);
    }
    // Sync all threads in the team so that scratch memory is consistent
    member.team_barrier();

    // compute fluxes over [is,ie+1]
    // NOTE(@pdmullen): Capture variables prior to if constexpr.  Required for cuda 11.6+.
    auto eos = eos_;
    auto indcs = indcs_;
    auto size = size_;
    auto coord = coord_;
    auto flx1 = flx1_;
    auto vf1 = vf1_;
    if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
      Advect(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
      LLF(member, eos, use_tabular_material_eos_, indcs, size, coord,
          m, k, j, il, iu, IVX, wl, wr, material_left, material_right, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
      HLLE(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
      HLLC(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
      Roe(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
      LLF_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
      HLLE_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
      HLLC_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
      LLF_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
      HLLE_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVX, wl, wr, flx1);
    }
    member.team_barrier();
    if (use_dual_energy_) {
      UpwindDualEnergyVelocity(member, eos, m, k, j, il, iu, wl, wr, flx1, vf1);
      member.team_barrier();
    }

    // calculate fluxes of scalars (if any)
    if (nvars > nhyd_) {
      for (int n=nhyd_; n<nvars; ++n) {
        par_for_inner(member, is, ie+1, [&](const int i) {
          if (flx1_(m,IDN,k,j,i) >= 0.0) {
            flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wl(n,i);
          } else {
            flx1_(m,n,k,j,i) = flx1_(m,IDN,k,j,i)*wr(n,i);
          }
        });
      }
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                ScrArray2D<Real>::shmem_size(nmaterial_fields_, ncells1)) * 3;
    auto &flx2_ = uflx.x2f;
    auto &vf2_ = dual_vf.x2f;

    // set the loop limits for 1D/2D/3D problems
    il = is, iu = ie, jl = js-1, ju = je+1, kl = ks, ku = ke;
    if (use_fofc) {
      jl = js-2, ju = je+2;
      if (pmy_pack->pmesh->two_d) {
        il = is-1, iu = ie+1, kl = ks, ku = ke;
      } else {
        il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
      }
    }

    par_for_outer("hflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);
      ScrArray2D<Real> scr5(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);
      ScrArray2D<Real> scr6(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);

      for (int j=jl; j<=ju; ++j) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_jp1 = scr2;
        auto wr     = scr3;
        auto material_left = scr4;
        auto material_left_jp1 = scr5;
        auto material_right = scr6;
        if ((j%2) == 0) {
          wl     = scr2;
          wl_jp1 = scr1;
          material_left = scr5;
          material_left_jp1 = scr4;
        }

        // Reconstruct qR[j] and qL[j+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, il, iu, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, il, iu, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX2(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos_, true, m, k, j, il, iu, w0_, wl_jp1, wr);
            break;
          default:
            break;
        }
        if (use_tabular_material_eos_) {
          ReconstructMaterialThermodynamicsX2(
              member, material_donor_cell_, m, k, j, il, iu,
              material_thermodynamics_, material_left_jp1, material_right);
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].  RS returns flux in input wr array
        if (j>jl) {
          // NOTE(@pdmullen): Capture variables prior to if constexpr.
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto flx2 = flx2_;
          auto vf2 = vf2_;
          if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
            Advect(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
            LLF(member, eos, use_tabular_material_eos_, indcs, size, coord,
                m, k, j, il, iu, IVY, wl, wr, material_left, material_right, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
            HLLE(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
            HLLC(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
            Roe(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
            LLF_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
            HLLE_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
            HLLC_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
            LLF_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
            HLLE_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
          }
          member.team_barrier();
          if (use_dual_energy_) {
            UpwindDualEnergyVelocity(member, eos, m, k, j, il, iu, wl, wr, flx2, vf2);
            member.team_barrier();
          }
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd_) {
          for (int n=nhyd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx2_(m,IDN,k,j,i) >= 0.0) {
                flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end of loop over j
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = (ScrArray2D<Real>::shmem_size(nvars, ncells1) +
                ScrArray2D<Real>::shmem_size(nmaterial_fields_, ncells1)) * 3;
    auto &flx3_ = uflx.x3f;
    auto &vf3_ = dual_vf.x3f;

    // set the loop limits
    il = is, iu = ie, jl = js, ju = je, kl = ks-1, ku = ke+1;
    if (use_fofc) { il = is-1, iu = ie+1, jl = js-1, ju = je+1, kl = ks-2, ku = ke+2; }

    par_for_outer("hflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, jl, ju,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr4(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);
      ScrArray2D<Real> scr5(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);
      ScrArray2D<Real> scr6(
          member.team_scratch(scr_level), nmaterial_fields_, ncells1);

      for (int k=kl; k<=ku; ++k) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_kp1 = scr2;
        auto wr     = scr3;
        auto material_left = scr4;
        auto material_left_kp1 = scr5;
        auto material_right = scr6;
        if ((k%2) == 0) {
          wl     = scr2;
          wl_kp1 = scr1;
          material_left = scr5;
          material_left_kp1 = scr4;
        }

        // Reconstruct qR[k] and qL[k+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, il, iu, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, il, iu, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX3(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos_, true, m, k, j, il, iu, w0_, wl_kp1, wr);
            break;
          default:
            break;
        }
        if (use_tabular_material_eos_) {
          ReconstructMaterialThermodynamicsX3(
              member, material_donor_cell_, m, k, j, il, iu,
              material_thermodynamics_, material_left_kp1, material_right);
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].  RS returns flux in input wr array
        if (k>kl) {
          // NOTE(@pdmullen): Capture variables prior to if constexpr.
          auto eos = eos_;
          auto indcs = indcs_;
          auto size = size_;
          auto coord = coord_;
          auto flx3 = flx3_;
          auto vf3 = vf3_;
          if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
            Advect(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
            LLF(member, eos, use_tabular_material_eos_, indcs, size, coord,
                m, k, j, il, iu, IVZ, wl, wr, material_left, material_right, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
            HLLE(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
            HLLC(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
            Roe(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
            LLF_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
            HLLE_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
            HLLC_SR(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
            LLF_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
            HLLE_GR(member, eos, indcs, size, coord, m, k, j, il, iu, IVZ, wl, wr, flx3);
          }
          member.team_barrier();
          if (use_dual_energy_) {
            UpwindDualEnergyVelocity(member, eos, m, k, j, il, iu, wl, wr, flx3, vf3);
            member.team_barrier();
          }
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd_) {
          for (int n=nhyd_; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx3_(m,IDN,k,j,i) >= 0.0) {
                flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3_(m,n,k,j,i) = flx3_(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end loop over k
    });
  }

  return;
}

// function definitions for each template parameter
template void Hydro::CalculateFluxes<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
