//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_target.cpp
//! \brief Laser-heated solid/corona target for radiation--Biermann communication tests.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "diffusion/conduction.hpp"
#include "laser/laser.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/biermann_battery.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

constexpr int kHistoryFields = 19;

void LaserTargetHistory(HistoryData *pdata, Mesh *pm);
void VacuumRadiationBoundary(Mesh *pm);

[[noreturn]] void ProblemError(const char *message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool NearlyEqual(const Real lhs, const Real rhs, const Real scale = 1.0) {
  return std::abs(lhs-rhs) <= 1.0e-12*std::fmax(scale, 1.0);
}

} // namespace

//----------------------------------------------------------------------------------------
//! Initialize a smooth planar solid/corona interface at constant temperature.
//!
//! Initially grad(p_e) is parallel to grad(n_e), so the continuum Biermann source is
//! zero.  The finite-radius laser subsequently produces a transverse electron-pressure
//! gradient.  Its cross product with the target-normal density gradient generates an
//! azimuthal magnetic field in the x2--x3 plane.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  static_assert(kHistoryFields <= NHISTORY_VARIABLES,
                "laser_target history exceeds AthenaK's reduction capacity");
  user_hist_func = LaserTargetHistory;
  user_bcs_func = VacuumRadiationBoundary;
  // Ordinary outflow remains the fluid boundary.  The callback replaces only the
  // embedded FLD group ghosts with a vacuum state after physical boundaries are filled.
  user_bcs = true;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->plaser == nullptr) {
    ProblemError("laser_target requires <mhd> and <laser> blocks");
  }
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;
  if (ptwo == nullptr || ptwo->pradiation == nullptr) {
    ProblemError("laser_target requires two-temperature MHD and <thermal_radiation>");
  }
  if (pmhd->pcond == nullptr || !pmhd->pcond->IsImplicit()) {
    ProblemError("laser_target requires implicit electron thermal conduction");
  }
  if (pmhd->pbiermann == nullptr || !pmhd->use_dual_energy) {
    ProblemError("laser_target requires Biermann battery and dual energy");
  }
  if (!pmhd->biermann_subcycle || !pmhd->biermann_reduced_closure) {
    ProblemError("laser_target requires reduced-closure Biermann subcycling");
  }
  if (!pmy_mesh_->three_d) {
    ProblemError("laser_target requires a three-dimensional Cartesian mesh");
  }
  if (pmhd->nuser_scalars != 1) {
    ProblemError("laser_target requires one user scalar for rho*Y_CH");
  }
  if (pmhd->pmaterials == nullptr || !pmhd->pmaterials->UsesTabularEOS()) {
    ProblemError("laser_target requires tabulated CH/He materials");
  }
  if (ptwo->pradiation->ngroups != 20) {
    ProblemError("laser_target requires the 20 reference radiation groups");
  }
  if (!pmhd->peos->eos_data.is_gamma_law) {
    ProblemError("laser_target tabular material mode requires a gamma-law MHD carrier");
  }
  if (!NearlyEqual(pmhd->pbiermann->coefficient,
                   2.9236304219444733e-3,
                   2.9236304219444733e-3)) {
    ProblemError("laser_target requires the material-number-density Biermann normalization");
  }

  const Real ambient_density =
      pin->GetOrAddReal("problem", "ambient_density", 1.0e-4);
  const Real solid_density = pin->GetOrAddReal("problem", "solid_density", 1.0);
  const Real temperature_kelvin =
      pin->GetOrAddReal("problem", "temperature_kelvin", 2.320903624310016e5);
  const Real target_surface =
      pin->GetOrAddReal("problem", "target_surface", 5.0e-2);
  const Real transition_width =
      pin->GetOrAddReal("problem", "transition_width", 6.0e-2);
  const Real corrugation_amplitude =
      pin->GetOrAddReal("problem", "corrugation_amplitude", 0.0);
  const Real corrugation_wavelength =
      pin->GetOrAddReal("problem", "corrugation_wavelength", 1.0);
  if (!(ambient_density > 0.0) || !(solid_density > ambient_density) ||
      !(temperature_kelvin > 0.0) || !(transition_width > 0.0) ||
      !(corrugation_wavelength > 0.0)) {
    ProblemError("laser_target densities, temperature, and length scales are invalid");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int scalar_index = pmhd->pmaterials->ScalarIndex();
  const auto material_mixture = pmhd->pmaterials->DeviceData();
  const Real code_temperature =
      material_mixture.CodeTemperatureFromKelvin(temperature_kelvin);
  auto size = pmbp->pmb->mb_size;
  auto w0 = pmhd->w0;
  auto bcc0 = pmhd->bcc0;
  const Real wave_number = 2.0*std::acos(-1.0)/corrugation_wavelength;

  Kokkos::deep_copy(w0, 0.0);
  Kokkos::deep_copy(bcc0, 0.0);
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  par_for("pgen_laser_target", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    const Real surface =
        target_surface+corrugation_amplitude*cos(wave_number*x2)*
                       cos(wave_number*x3);
    const Real alpha = 0.5*(1.0+tanh((x1-surface)/transition_width));
    const Real ch_partial_density = alpha*solid_density;
    const Real he_partial_density = (1.0-alpha)*ambient_density;
    const Real density = ch_partial_density+he_partial_density;
    const Real ch_mass_fraction = ch_partial_density/density;
    const auto thermo = material_mixture.StateFromRhoTemperatures(
        density, code_temperature, code_temperature, ch_mass_fraction);
    const Real internal_energy = density*(thermo.ion_specific_internal_energy+
                                          thermo.electron_specific_internal_energy);

    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = internal_energy;
    w0(m, scalar_index, k, j, i) = ch_mass_fraction;
  });

  pmhd->peos->PrimToCons(w0, bcc0, pmhd->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! Apply a zero-energy (Dirichlet vacuum) boundary to embedded FLD groups only.

void VacuumRadiationBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *prad = pmbp->pmhd->ptwo_temp->pradiation;
  auto u0 = pmbp->pmhd->u0;
  auto mb_bcs = pmbp->pmb->mb_bcs;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1+2*ng;
  const int n2 = pm->multi_d ? indcs.nx2+2*ng : 1;
  const int n3 = pm->three_d ? indcs.nx3+2*ng : 1;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack-1;
  const int ifirst = prad->ifirst;
  const int ngroups = prad->ngroups;

  par_for("laser_target_vacuum_rad_x1", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int g, int k, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, is-n-1) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, ie+n+1) = 0.0;
    }
  });

  if (pm->multi_d) {
    par_for("laser_target_vacuum_rad_x2", DevExeSpace(), 0, nmb1, 0, ngroups-1,
            0, n3-1, 0, n1-1,
    KOKKOS_LAMBDA(int m, int g, int k, int i) {
      if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
        for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, js-n-1, i) = 0.0;
      }
      if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
        for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, je+n+1, i) = 0.0;
      }
    });
  }

  if (pm->three_d) {
    par_for("laser_target_vacuum_rad_x3", DevExeSpace(), 0, nmb1, 0, ngroups-1,
            0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(int m, int g, int j, int i) {
      if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
        for (int n = 0; n < ng; ++n) u0(m, ifirst+g, ks-n-1, j, i) = 0.0;
      }
      if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
        for (int n = 0; n < ng; ++n) u0(m, ifirst+g, ke+n+1, j, i) = 0.0;
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! Integrated communication diagnostics.  All entries are volume integrals except the
//! three first moments.  The ordinary history writer performs the MPI reduction.

void LaserTargetHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;
  auto *prad = ptwo->pradiation;

  pdata->nhist = kHistoryFields;
  pdata->label[0] = "laser_E";
  pdata->label[1] = "laser_P";
  pdata->label[2] = "eion_E";
  pdata->label[3] = "eele_E";
  pdata->label[4] = "erad_E";
  pdata->label[5] = "mat_E";
  pdata->label[6] = "chain_E";
  pdata->label[7] = "abs_B";
  pdata->label[8] = "mag_E";
  pdata->label[9] = "bier_S";
  pdata->label[10] = "laser_x";
  pdata->label[11] = "erad_x";
  pdata->label[12] = "eele_x";
  pdata->label[13] = "volume";
  pdata->label[14] = "mass";
  pdata->label[15] = "CH_mass";
  pdata->label[16] = "eos_floor";
  pdata->label[17] = "eos_bad";
  pdata->label[18] = "divB";

  auto u0 = pmhd->u0;
  auto w0 = pmhd->w0;
  auto bcc0 = pmhd->bcc0;
  auto b0 = pmhd->b0;
  auto laser_data = pmbp->plaser->cell_data;
  auto thermodynamics = ptwo->thermodynamics;
  auto size = pmbp->pmb->mb_size;
  const auto material_mixture = pmhd->pmaterials->DeviceData();
  const int scalar_index = pmhd->pmaterials->ScalarIndex();
  int iion = ptwo->iion;
  int iele = ptwo->iele;
  int ifirst = prad->ifirst;
  int ngroups = prad->ngroups;
  Real biermann_coefficient = pmhd->pbiermann->coefficient;
  const Real electron_density_cgs_to_code =
      materials::MaterialMixtureDevice::atomic_mass_unit_cgs/
      material_mixture.density_to_cgs;
  constexpr int disallowed_eos_flags =
      materials::ionmix_density_above_table |
      materials::ionmix_temperature_below_table |
      materials::ionmix_temperature_above_table |
      materials::ionmix_energy_above_table;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int nhist = pdata->nhist;
  array_sum::GlobalSum sum_this_rank;

  Kokkos::parallel_reduce(
      "laser_target_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &sum) {
        int m = idx/nkji;
        int k = (idx-m*nkji)/nji + ks;
        int j = (idx-m*nkji-(k-ks)*nji)/nx1 + js;
        int i = idx-m*nkji-(k-ks)*nji-(j-js)*nx1 + is;
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real volume = dx1*dx2*dx3;
        Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);

        Real radiation_energy = 0.0;
        for (int g = 0; g < ngroups; ++g) {
          radiation_energy += u0(m, ifirst+g, k, j, i);
        }
        Real material_energy = u0(m, IEN, k, j, i);
        Real b1 = bcc0(m, IBX, k, j, i);
        Real b2 = bcc0(m, IBY, k, j, i);
        Real b3 = bcc0(m, IBZ, k, j, i);

        const Real dne_dx = electron_density_cgs_to_code*
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k, j, i+1)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k, j, i-1))/(2.0*dx1);
        const Real dne_dy = electron_density_cgs_to_code*
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k, j+1, i)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k, j-1, i))/(2.0*dx2);
        const Real dne_dz = electron_density_cgs_to_code*
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k+1, j, i)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_number_density_cgs, k-1, j, i))/(2.0*dx3);
        const Real dpe_dx =
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k, j, i+1)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k, j, i-1))/(2.0*dx1);
        const Real dpe_dy =
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k, j+1, i)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k, j-1, i))/(2.0*dx2);
        const Real dpe_dz =
            (thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k+1, j, i)-
             thermodynamics(m, two_temperature::TwoTemperature::
                 electron_pressure, k-1, j, i))/(2.0*dx3);
        const Real ne = fmax(electron_density_cgs_to_code*thermodynamics(
            m, two_temperature::TwoTemperature::electron_number_density_cgs,
            k, j, i), 1.0e-30);
        const Real source_x = dne_dy*dpe_dz-dne_dz*dpe_dy;
        const Real source_y = dne_dz*dpe_dx-dne_dx*dpe_dz;
        const Real source_z = dne_dx*dpe_dy-dne_dy*dpe_dx;
        Real source = biermann_coefficient*
            sqrt(SQR(source_x)+SQR(source_y)+SQR(source_z))/
            fmax(ne*ne, 1.0e-30);

        array_sum::GlobalSum local;
        local.the_array[0] = volume*laser_data(m, 1, k, j, i);
        local.the_array[1] = volume*laser_data(m, 0, k, j, i);
        local.the_array[2] = volume*u0(m, iion, k, j, i);
        local.the_array[3] = volume*u0(m, iele, k, j, i);
        local.the_array[4] = volume*radiation_energy;
        local.the_array[5] = volume*material_energy;
        local.the_array[6] = volume*(material_energy+radiation_energy);
        local.the_array[7] = volume*sqrt(b1*b1+b2*b2+b3*b3);
        local.the_array[8] = 0.5*volume*(b1*b1+b2*b2+b3*b3);
        local.the_array[9] = volume*source;
        local.the_array[10] = volume*x1*laser_data(m, 1, k, j, i);
        local.the_array[11] = volume*x1*radiation_energy;
        local.the_array[12] = volume*x1*u0(m, iele, k, j, i);
        local.the_array[13] = volume;
        local.the_array[14] = volume*u0(m, IDN, k, j, i);
        local.the_array[15] = volume*u0(m, scalar_index, k, j, i);
        const int eos_flags = static_cast<int>(thermodynamics(
            m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
        local.the_array[16] =
            ((eos_flags & materials::ionmix_energy_below_table) != 0) ? 1.0 : 0.0;
        local.the_array[17] =
            ((eos_flags & disallowed_eos_flags) != 0) ? 1.0 : 0.0;
        const Real divb =
            (b0.x1f(m, k, j, i+1)-b0.x1f(m, k, j, i))/dx1+
            (b0.x2f(m, k, j+1, i)-b0.x2f(m, k, j, i))/dx2+
            (b0.x3f(m, k+1, j, i)-b0.x3f(m, k, j, i))/dx3;
        local.the_array[18] = volume*fabs(divb);
        for (int n = nhist; n < NHISTORY_VARIABLES; ++n) {
          local.the_array[n] = 0.0;
        }
        sum += local;
      }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_rank.the_array[n];
  }
}

} // namespace
