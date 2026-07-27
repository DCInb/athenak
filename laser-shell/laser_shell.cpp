//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_shell.cpp
//! \brief Laser-driven, open spherical-cap CH shell in three Cartesian dimensions.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "laser/laser.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "two_temperature/two_temperature.hpp"

namespace {

void LaserShellHistory(HistoryData *pdata, Mesh *pm);

[[noreturn]] void ProblemError(const char *message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
//! Initialize a spherical shell cap centered on the -x1 axis. The laser enters from
//! positive x1, crosses the open side, and illuminates the concave inner shell surface.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = LaserShellHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->plaser == nullptr) {
    ProblemError("laser_shell requires <mhd> and <laser> blocks");
  }
  if (pmbp->pmhd->ptwo_temp == nullptr) {
    ProblemError("laser_shell requires <mhd>/two_temperature=true");
  }
  if (!pmy_mesh_->three_d) {
    ProblemError("laser_shell requires a three-dimensional Cartesian mesh");
  }

  const Real ambient_density =
      pin->GetOrAddReal("problem", "ambient_density", 1.0e-8);
  const Real solid_density =
      pin->GetOrAddReal("problem", "solid_density", 1.0);
  const Real temperature =
      pin->GetOrAddReal("problem", "temperature", 1.7268498302462577e-6);
  const Real inner_radius =
      pin->GetOrAddReal("problem", "inner_radius", 0.8);
  const Real outer_radius =
      pin->GetOrAddReal("problem", "outer_radius", 1.0);
  const Real opening_half_angle_deg =
      pin->GetOrAddReal("problem", "opening_half_angle_deg", 25.0);
  const Real transition_width =
      pin->GetOrAddReal("problem", "transition_width", 2.0e-2);
  const Real angular_transition =
      pin->GetOrAddReal("problem", "angular_transition", 1.0e-2);
  if (!(ambient_density > 0.0) || !(solid_density > ambient_density) ||
      !(temperature > 0.0) || !(inner_radius > 0.0) ||
      !(outer_radius > inner_radius) || !(opening_half_angle_deg > 0.0) ||
      !(opening_half_angle_deg < 90.0) || !(transition_width > 0.0) ||
      !(angular_transition > 0.0)) {
    ProblemError("laser_shell geometry and thermodynamic parameters are invalid");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  auto w0 = pmbp->pmhd->w0;
  auto bcc0 = pmbp->pmhd->bcc0;
  const Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  const Real pi = std::acos(-1.0);
  const Real cos_half_angle = std::cos(opening_half_angle_deg*pi/180.0);
  const Real beam_origin_x1 = pin->GetReal("laser", "beam0_origin_x1");
  const Real beam_direction_x1 = pin->GetReal("laser", "beam0_direction_x1");
  const Real beam_direction_x2 = pin->GetReal("laser", "beam0_direction_x2");
  const Real beam_direction_x3 = pin->GetReal("laser", "beam0_direction_x3");
  const Real beam_radius = pin->GetReal("laser", "beam0_radius");
  const Real origin_tolerance = 1.0e-12*
      std::fmax(std::abs(pmy_mesh_->mesh_size.x1max), 1.0);
  const Real projected_inner_radius =
      inner_radius*std::sin(opening_half_angle_deg*pi/180.0);
  if (std::abs(beam_origin_x1-pmy_mesh_->mesh_size.x1max) > origin_tolerance ||
      std::abs(beam_direction_x1+1.0) > origin_tolerance ||
      std::abs(beam_direction_x2) > origin_tolerance ||
      std::abs(beam_direction_x3) > origin_tolerance ||
      !(beam_radius > 0.0) || beam_radius > projected_inner_radius) {
    ProblemError("laser_shell requires a -x1 beam launched from the right boundary "
                 "with an aperture no larger than the projected inner cap");
  }

  Kokkos::deep_copy(w0, 0.0);
  Kokkos::deep_copy(bcc0, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);

  par_for("pgen_laser_shell", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
    const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
    const Real radius = sqrt(x1*x1 + x2*x2 + x3*x3);
    const Real axis_cosine = (radius > 0.0) ? -x1/radius : -1.0;

    const Real inside_outer =
        0.5*(1.0 - tanh((radius-outer_radius)/transition_width));
    const Real outside_inner =
        0.5*(1.0 + tanh((radius-inner_radius)/transition_width));
    const Real inside_cap = 0.5*(1.0 +
        tanh((axis_cosine-cos_half_angle)/angular_transition));
    const Real material_fraction = inside_outer*outside_inner*inside_cap;
    const Real density = ambient_density +
        (solid_density-ambient_density)*material_fraction;
    const Real pressure = density*temperature;

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
//! Volume-integrated run diagnostics. The history writer performs the MPI reduction.

void LaserShellHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;

  pdata->nhist = 12;
  pdata->label[0] = "laser_E";
  pdata->label[1] = "laser_P";
  pdata->label[2] = "mass";
  pdata->label[3] = "mom1";
  pdata->label[4] = "mat_E";
  pdata->label[5] = "kin_E";
  pdata->label[6] = "eion_E";
  pdata->label[7] = "eele_E";
  pdata->label[8] = "laser_x";
  pdata->label[9] = "laser_r2";
  pdata->label[10] = "abs_B";
  pdata->label[11] = "volume";

  auto u0 = pmhd->u0;
  auto bcc0 = pmhd->bcc0;
  auto laser_data = pmbp->plaser->cell_data;
  auto size = pmbp->pmb->mb_size;
  const int iion = ptwo->iion;
  const int iele = ptwo->iele;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nhist = pdata->nhist;
  array_sum::GlobalSum sum_this_rank;

  Kokkos::parallel_reduce(
      "laser_shell_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k = (idx-m*nkji)/nji + ks;
        const int j = (idx-m*nkji-(k-ks)*nji)/nx1 + js;
        const int i = idx-m*nkji-(k-ks)*nji-(j-js)*nx1 + is;
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2*
                            size.d_view(m).dx3;
        const Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                                    size.d_view(m).x1max);
        const Real x2 = CellCenterX(j-js, nx2, size.d_view(m).x2min,
                                    size.d_view(m).x2max);
        const Real x3 = CellCenterX(k-ks, nx3, size.d_view(m).x3min,
                                    size.d_view(m).x3max);
        const Real density = u0(m, IDN, k, j, i);
        const Real momentum_squared = SQR(u0(m, IM1, k, j, i)) +
                                      SQR(u0(m, IM2, k, j, i)) +
                                      SQR(u0(m, IM3, k, j, i));
        const Real bmag = sqrt(SQR(bcc0(m, IBX, k, j, i)) +
                               SQR(bcc0(m, IBY, k, j, i)) +
                               SQR(bcc0(m, IBZ, k, j, i)));

        array_sum::GlobalSum local;
        local.the_array[0] = volume*laser_data(m, 1, k, j, i);
        local.the_array[1] = volume*laser_data(m, 0, k, j, i);
        local.the_array[2] = volume*density;
        local.the_array[3] = volume*u0(m, IM1, k, j, i);
        local.the_array[4] = volume*u0(m, IEN, k, j, i);
        local.the_array[5] = volume*0.5*momentum_squared/density;
        local.the_array[6] = volume*u0(m, iion, k, j, i);
        local.the_array[7] = volume*u0(m, iele, k, j, i);
        local.the_array[8] = volume*x1*laser_data(m, 1, k, j, i);
        local.the_array[9] = volume*(x2*x2+x3*x3)*laser_data(m, 1, k, j, i);
        local.the_array[10] = volume*bmag;
        local.the_array[11] = volume;
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
