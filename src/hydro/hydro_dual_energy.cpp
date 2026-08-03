//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_dual_energy.cpp
//! \brief Two-temperature-backed dual-energy formalism for Newtonian ideal-gas hydro.
//!
//! This is the B=0 counterpart of mhd/mhd_dual_energy.cpp; keep the two in sync.  Without
//! a magnetic field the conservative internal energy is total minus kinetic, but a
//! laser-driven corona still reaches states where that difference is pure cancellation
//! noise, so the advected ion+electron sum remains the fallback pressure source.

#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool DualEnergySyncEligible(const Real eint_cons, const Real local_etot_max,
                            const Real eta2) {
  if (eint_cons <= 0.0) return false;
  if (eta2 <= 0.0) return true;
  return eint_cons > eta2*fmax(local_etot_max, 1.0e-18);
}

} // namespace

namespace hydro {

TaskStatus Hydro::DualEnergyStep(Driver *pdrive, int stage) {
  if (use_dual_energy) {
    const Real beta_dt = pdrive->beta[stage-1]*pmy_pack->pmesh->dt;
    ApplyDualEnergyFormalism(beta_dt);
  }
  return TaskStatus::complete;
}

// Add the gamma-law p div(v) work omitted by passive-scalar advection of rho*e_i and
// rho*e_e.  Both components have the same gamma, so a common multiplicative update
// preserves their temperature partition while evolving their sum as internal energy.
void Hydro::ApplyDualEnergyFormalism(const Real dt) {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  auto u0_ = u0;
  auto w0_ = w0;
  auto vf1_ = dual_vf.x1f;
  auto vf2_ = dual_vf.x2f;
  auto vf3_ = dual_vf.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int iion = ptwo_temp->iion;
  const int iele = ptwo_temp->iele;
  const Real fe0 = ptwo_temp->InitialElectronEnergyFraction();
  const Real gm1 = eos.gamma - 1.0;
  const bool use_materials = pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) material_mixture = pmaterials->DeviceData();
  const Real initial_temperature_ratio =
      ptwo_temp->InitialElectronTemperatureRatio();

  par_for("hyd_2t_dual_energy_compress", DevExeSpace(),
          0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real divv = (vf1_(m, 0, k, j, i+1) - vf1_(m, 0, k, j, i))/
                mbsize.d_view(m).dx1;
    if (multi_d) {
      divv += (vf2_(m, 0, k, j+1, i) - vf2_(m, 0, k, j, i))/
              mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divv += (vf3_(m, 0, k+1, j, i) - vf3_(m, 0, k, j, i))/
              mbsize.d_view(m).dx3;
    }

    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    Real eion = fmax(u0_(m, iion, k, j, i), 0.0);
    Real eele = fmax(u0_(m, iele, k, j, i), 0.0);
    const Real component_sum = eion + eele;
    Real initial_fraction = fe0;
    if (use_materials) {
      const Real y0 = material_mixture.Material0MassFractionFromConserved(
          u0_, m, k, j, i, eos.dfloor);
      if (material_mixture.UsesTabularEOS()) {
        const materials::MaterialPressureEnergyState state =
            material_mixture.PressureEnergyFromRhoSpecificEnergies(
                dens, eion/dens, eele/dens, y0);
        const materials::MaterialPressureEnergyState floor =
            material_mixture.MinimumPressureEnergyState(
                dens, y0, eos.pfloor, eos.tfloor);
        if (eion > 0.0) {
          eion *= exp(-(state.ion_pressure/eion)*divv*dt);
        }
        if (eele > 0.0) {
          eele *= exp(-(state.electron_pressure/eele)*divv*dt);
        }
        eion = fmax(eion, dens*floor.ion_specific_internal_energy);
        eele = fmax(eele, dens*floor.electron_specific_internal_energy);
        u0_(m, iion, k, j, i) = eion;
        u0_(m, iele, k, j, i) = eele;
        w0_(m, iion, k, j, i) = eion/dens;
        w0_(m, iele, k, j, i) = eele/dens;
        return;
      }
      initial_fraction = material_mixture.InitialElectronEnergyFraction(
          y0, initial_temperature_ratio);
    }
    const Real eele_fraction =
        (component_sum > 0.0) ? eele/component_sum : initial_fraction;
    Real eint = component_sum*exp(-gm1*divv*dt);
    eint = fmax(eint, eos.HydroInternalEnergyDensityFloor(dens));
    eele = fmin(fmax(eele_fraction*eint, 0.0), eint);
    eion = eint - eele;

    u0_(m, iion, k, j, i) = eion;
    u0_(m, iele, k, j, i) = eele;
    w0_(m, iion, k, j, i) = eion/dens;
    w0_(m, iele, k, j, i) = eele/dens;
  });
}

// Refresh the auxiliary component-energy sum from conservative total energy only where
// subtraction is well conditioned.  Else retain the independently evolved 2T energy.
void Hydro::SynchronizeDualEnergyFromTotal() {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int gis = is - ng;
  const int gie = ie + ng;
  const int gjs = multi_d ? js - ng : js;
  const int gje = multi_d ? je + ng : je;
  const int gks = three_d ? ks - ng : ks;
  const int gke = three_d ? ke + ng : ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto etot_max_ = dual_etot_max;
  auto &eos = peos->eos_data;
  const int iion = ptwo_temp->iion;
  const int iele = ptwo_temp->iele;
  const Real fe0 = ptwo_temp->InitialElectronEnergyFraction();
  const Real eta2 = dual_energy_eta2;
  const bool use_materials = pmaterials != nullptr;
  materials::MaterialMixtureDevice material_mixture;
  if (use_materials) material_mixture = pmaterials->DeviceData();
  const Real initial_temperature_ratio =
      ptwo_temp->InitialElectronTemperatureRatio();

  if (eta2 > 0.0) {
    par_for("hyd_2t_dual_energy_etot_max", DevExeSpace(),
            0, nmb1, gks, gke, gjs, gje, gis, gie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real emax = 0.0;
      const int kmin = three_d ? ((k - 1 > gks) ? k - 1 : gks) : k;
      const int kmax = three_d ? ((k + 1 < gke) ? k + 1 : gke) : k;
      const int jmin = multi_d ? ((j - 1 > gjs) ? j - 1 : gjs) : j;
      const int jmax = multi_d ? ((j + 1 < gje) ? j + 1 : gje) : j;
      const int imin = (i - 1 > gis) ? i - 1 : gis;
      const int imax = (i + 1 < gie) ? i + 1 : gie;
      for (int kk = kmin; kk <= kmax; ++kk) {
        for (int jj = jmin; jj <= jmax; ++jj) {
          for (int ii = imin; ii <= imax; ++ii) {
            emax = fmax(emax, u0_(m, IEN, kk, jj, ii));
          }
        }
      }
      etot_max_(m, k, j, i) = emax;
    });
  }

  par_for("hyd_2t_dual_energy_sync", DevExeSpace(),
          0, nmb1, gks, gke, gjs, gje, gis, gie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) +
                          SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real eint_cons = u0_(m, IEN, k, j, i) - e_k;
    const Real local_etot_max = (eta2 > 0.0) ? etot_max_(m, k, j, i) : 0.0;

    Real eion = fmax(u0_(m, iion, k, j, i), 0.0);
    Real eele = fmax(u0_(m, iele, k, j, i), 0.0);
    const Real component_sum = eion + eele;
    Real initial_fraction = fe0;
    if (use_materials) {
      const Real y0 = material_mixture.Material0MassFractionFromConserved(
          u0_, m, k, j, i, eos.dfloor);
      if (material_mixture.UsesTabularEOS()) {
        const materials::MaterialPressureEnergyState floor =
            material_mixture.MinimumPressureEnergyState(
                dens, y0, eos.pfloor, eos.tfloor);
        const Real eion_floor = dens*floor.ion_specific_internal_energy;
        const Real eele_floor = dens*floor.electron_specific_internal_energy;
        const Real minimum_sum = eion_floor+eele_floor;
        Real eint_aux = component_sum;
        if (DualEnergySyncEligible(eint_cons, local_etot_max, eta2)) {
          eint_aux = eint_cons;
        }
        eint_aux = fmax(eint_aux, minimum_sum);
        if (eint_cons < minimum_sum) {
          u0_(m, IEN, k, j, i) += minimum_sum-eint_cons;
        }
        Real ion_fraction;
        const bool zero_residual_fast_path =
            y0 > 0.0 && y0 < 1.0 && eion > 0.0 && eele > 0.0 &&
            Kokkos::isfinite(eint_aux) && eint_aux == component_sum &&
            material_mixture.TabularPressureSumsAreSafelyFinite();
        if (zero_residual_fast_path) {
          // Retain strict-bounds failures even though no diagnostic flags are stored by
          // this dual-energy path.  With an exact zero residual, the finite pressure
          // partition cannot affect any arithmetic below.
          static_cast<void>(material_mixture.SpecificEnergiesQueryFlags(
              dens, fmax(eion, eion_floor)/dens,
              fmax(eele, eele_floor)/dens, y0));
          ion_fraction = eion/component_sum;
        } else {
          const materials::MaterialPressureEnergyState state =
              material_mixture.PressureEnergyFromRhoSpecificEnergies(
                  dens, fmax(eion, eion_floor)/dens,
                  fmax(eele, eele_floor)/dens, y0);
          const Real pressure_sum = state.ion_pressure+state.electron_pressure;
          ion_fraction = (pressure_sum > 0.0)
              ? state.ion_pressure/pressure_sum
              : ((component_sum > 0.0) ? eion/component_sum : 0.5);
        }
        const Real residual = eint_aux-component_sum;
        Real ion_extra = fmax(eion+ion_fraction*residual-eion_floor, 0.0);
        Real electron_extra = fmax(
            eele+(1.0-ion_fraction)*residual-eele_floor, 0.0);
        const Real available = eint_aux-minimum_sum;
        const Real extra_sum = ion_extra+electron_extra;
        if (extra_sum > 0.0) {
          ion_extra *= available/extra_sum;
          electron_extra *= available/extra_sum;
        } else {
          ion_extra = ion_fraction*available;
          electron_extra = (1.0-ion_fraction)*available;
        }
        u0_(m, iion, k, j, i) = eion_floor+ion_extra;
        u0_(m, iele, k, j, i) = eele_floor+electron_extra;
        return;
      }
      initial_fraction = material_mixture.InitialElectronEnergyFraction(
          y0, initial_temperature_ratio);
    }
    const Real eele_fraction =
        (component_sum > 0.0) ? eele/component_sum : initial_fraction;
    Real eint_aux = component_sum;
    if (DualEnergySyncEligible(eint_cons, local_etot_max, eta2)) {
      eint_aux = eint_cons;
    }
    eint_aux = fmax(eint_aux, eos.HydroInternalEnergyDensityFloor(dens));
    eele = fmin(fmax(eele_fraction*eint_aux, 0.0), eint_aux);
    eion = eint_aux - eele;
    u0_(m, iion, k, j, i) = eion;
    u0_(m, iele, k, j, i) = eele;
  });
}

} // namespace hydro
