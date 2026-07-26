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
#include "laser/laser.hpp"
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

void LaserTargetHistory(HistoryData *pdata, Mesh *pm);

[[noreturn]] void ProblemError(const char *message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
//! Initialize a smooth planar solid/corona interface at constant temperature.
//!
//! Initially grad(p_e) is parallel to grad(n_e), so the continuum Biermann source is
//! zero.  The finite-radius laser subsequently produces a transverse electron-pressure
//! gradient.  Its cross product with the target-normal density gradient generates B3.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = LaserTargetHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->plaser == nullptr) {
    ProblemError("laser_target requires <mhd> and <laser> blocks");
  }
  if (pmbp->pmhd->ptwo_temp == nullptr ||
      pmbp->pmhd->ptwo_temp->pradiation == nullptr) {
    ProblemError("laser_target requires two-temperature MHD and <thermal_radiation>");
  }
  if (pmbp->pmhd->pbiermann == nullptr) {
    ProblemError("laser_target requires <mhd>/biermann_battery=true");
  }
  if (!(pmy_mesh_->multi_d) || pmy_mesh_->three_d) {
    ProblemError("laser_target is a two-dimensional Cartesian benchmark");
  }

  Real ambient_density =
      pin->GetOrAddReal("problem", "ambient_density", 1.0e-4);
  Real solid_density = pin->GetOrAddReal("problem", "solid_density", 1.0);
  Real temperature = pin->GetOrAddReal("problem", "temperature", 2.0e-2);
  Real target_surface = pin->GetOrAddReal("problem", "target_surface", 5.0e-2);
  Real transition_width =
      pin->GetOrAddReal("problem", "transition_width", 6.0e-2);
  Real corrugation_amplitude =
      pin->GetOrAddReal("problem", "corrugation_amplitude", 0.0);
  Real corrugation_wavelength =
      pin->GetOrAddReal("problem", "corrugation_wavelength", 1.0);
  if (!(ambient_density > 0.0) || !(solid_density > ambient_density) ||
      !(temperature > 0.0) || !(transition_width > 0.0) ||
      !(corrugation_wavelength > 0.0)) {
    ProblemError("laser_target densities, temperature, and length scales are invalid");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  auto w0 = pmbp->pmhd->w0;
  auto bcc0 = pmbp->pmhd->bcc0;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real wave_number = 2.0*std::acos(-1.0)/corrugation_wavelength;

  Kokkos::deep_copy(w0, 0.0);
  Kokkos::deep_copy(bcc0, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);

  par_for("pgen_laser_target", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max);
    Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max);
    Real surface = target_surface + corrugation_amplitude*cos(wave_number*x2);
    Real fraction = 0.5*(1.0 + tanh((x1-surface)/transition_width));
    Real density = ambient_density + (solid_density-ambient_density)*fraction;
    Real pressure = density*temperature;

    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = pressure/gm1;
  });

  pmbp->pmhd->peos->PrimToCons(
      w0, bcc0, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! Integrated communication diagnostics.  All entries are volume integrals except the
//! three first moments.  The ordinary history writer performs the MPI reduction.

void LaserTargetHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;
  auto *prad = ptwo->pradiation;

  pdata->nhist = 14;
  pdata->label[0] = "laser_E";
  pdata->label[1] = "laser_P";
  pdata->label[2] = "eion_E";
  pdata->label[3] = "eele_E";
  pdata->label[4] = "erad_E";
  pdata->label[5] = "mat_E";
  pdata->label[6] = "chain_E";
  pdata->label[7] = "abs_Bz";
  pdata->label[8] = "mag_E";
  pdata->label[9] = "bier_S";
  pdata->label[10] = "laser_x";
  pdata->label[11] = "erad_x";
  pdata->label[12] = "eele_x";
  pdata->label[13] = "volume";

  auto u0 = pmhd->u0;
  auto w0 = pmhd->w0;
  auto bcc0 = pmhd->bcc0;
  auto laser_data = pmbp->plaser->cell_data;
  auto size = pmbp->pmb->mb_size;
  int iion = ptwo->iion;
  int iele = ptwo->iele;
  int ifirst = prad->ifirst;
  int ngroups = prad->ngroups;
  Real gm1 = pmhd->peos->eos_data.gamma - 1.0;
  Real electron_fraction = ptwo->ElectronHeatCapacityFraction();
  Real biermann_coefficient = pmhd->pbiermann->coefficient;

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
        Real volume = dx1*dx2*size.d_view(m).dx3;
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

        Real dne_dx = electron_fraction*
            (w0(m, IDN, k, j, i+1)-w0(m, IDN, k, j, i-1))/(2.0*dx1);
        Real dne_dy = electron_fraction*
            (w0(m, IDN, k, j+1, i)-w0(m, IDN, k, j-1, i))/(2.0*dx2);
        Real dpe_dx = gm1*
            (u0(m, iele, k, j, i+1)-u0(m, iele, k, j, i-1))/(2.0*dx1);
        Real dpe_dy = gm1*
            (u0(m, iele, k, j+1, i)-u0(m, iele, k, j-1, i))/(2.0*dx2);
        Real ne = electron_fraction*w0(m, IDN, k, j, i);
        Real source = biermann_coefficient*
            fabs(dne_dx*dpe_dy-dne_dy*dpe_dx)/fmax(ne*ne, 1.0e-30);

        array_sum::GlobalSum local;
        local.the_array[0] = volume*laser_data(m, 1, k, j, i);
        local.the_array[1] = volume*laser_data(m, 0, k, j, i);
        local.the_array[2] = volume*u0(m, iion, k, j, i);
        local.the_array[3] = volume*u0(m, iele, k, j, i);
        local.the_array[4] = volume*radiation_energy;
        local.the_array[5] = volume*material_energy;
        local.the_array[6] = volume*(material_energy+radiation_energy);
        local.the_array[7] = volume*fabs(b3);
        local.the_array[8] = 0.5*volume*(b1*b1+b2*b2+b3*b3);
        local.the_array[9] = volume*source;
        local.the_array[10] = volume*x1*laser_data(m, 1, k, j, i);
        local.the_array[11] = volume*x1*radiation_energy;
        local.the_array[12] = volume*x1*u0(m, iele, k, j, i);
        local.the_array[13] = volume;
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
