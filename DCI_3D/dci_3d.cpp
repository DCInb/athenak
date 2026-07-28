//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dci_3d.cpp
//! \brief Provisional 3D laser drive of an open CH spherical cap with 3T radiation.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

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

constexpr int kHistoryFields = 20;

void DCIHistory(HistoryData *pdata, Mesh *pm);
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
//! Initialize the provisional CH spherical cap and its conservative material tracer.
//!
//! The smooth geometry variable alpha is a volume fraction.  The passive scalar is a
//! mass fraction: rho*X_CH = alpha*rho_CH.  This distinction is important across the
//! smoothed solid/ambient interface where the component densities differ by eight orders
//! of magnitude.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  static_assert(kHistoryFields <= NHISTORY_VARIABLES,
                "DCI history exceeds AthenaK's reduction capacity");
  user_hist_func = DCIHistory;
  user_bcs_func = VacuumRadiationBoundary;
  // Fluid faces use ordinary outflow flags.  Force the post-physical-BC callback so it
  // can replace only the embedded FLD group ghost cells with a vacuum state.
  user_bcs = true;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->plaser == nullptr) {
    ProblemError("dci_3d requires <mhd> and <laser> blocks");
  }
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;
  if (ptwo == nullptr || ptwo->pradiation == nullptr) {
    ProblemError("dci_3d requires two-temperature MHD and thermal radiation");
  }
  if (pmhd->pbiermann == nullptr || !pmhd->use_dual_energy) {
    ProblemError("dci_3d requires Biermann battery and dual energy");
  }
  if (!pmy_mesh_->three_d) {
    ProblemError("dci_3d requires a three-dimensional Cartesian mesh");
  }
  if (pmhd->nuser_scalars != 1) {
    ProblemError("dci_3d requires exactly one user scalar for rho*X_CH");
  }
  if (ptwo->pradiation->ngroups != 3) {
    ProblemError("dci_3d provisional radiation model requires exactly three groups");
  }
  if (!pmhd->peos->eos_data.is_gamma_law) {
    ProblemError("dci_3d currently requires the documented gamma-law CH surrogate");
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
    ProblemError("dci_3d geometry and thermodynamic parameters are invalid");
  }

  const std::string geometry =
      pin->GetString("laser", "beam0_geometry");
  const std::string profile = pin->GetString("laser", "beam0_profile");
  const Real lens_x1 = pin->GetReal("laser", "beam0_lens_x1");
  const Real target_x1 = pin->GetReal("laser", "beam0_target_x1");
  const Real aperture_radius =
      pin->GetReal("laser", "beam0_aperture_radius");
  const Real target_radius = pin->GetReal("laser", "beam0_target_radius");
  const Real beam_power = pin->GetReal("laser", "beam0_power");
  const Real wavelength = pin->GetReal("laser", "beam0_wavelength");
  const Real beam_start = pin->GetReal("laser", "beam0_start_time");
  const Real beam_end = pin->GetReal("laser", "beam0_end_time");
  const Real pi = std::acos(-1.0);
  const Real projected_inner_radius =
      inner_radius*std::sin(opening_half_angle_deg*pi/180.0);
  if (geometry != "lens" || profile != "gaussian" ||
      !NearlyEqual(lens_x1, pmy_mesh_->mesh_size.x1max,
                   std::abs(pmy_mesh_->mesh_size.x1max)) ||
      !NearlyEqual(target_x1, -inner_radius, inner_radius) ||
      !(aperture_radius > target_radius) || !(target_radius > 0.0) ||
      target_radius > projected_inner_radius ||
      !NearlyEqual(beam_power, 2.0e19, 2.0e19) ||
      !NearlyEqual(wavelength, 1.053e-4, 1.053e-4) ||
      !NearlyEqual(beam_start, 0.0) || !NearlyEqual(beam_end, 5.0, 5.0)) {
    ProblemError("dci_3d requires the documented focused Gaussian 10 kJ beam");
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int scalar_index = pmhd->nmhd;
  auto size = pmbp->pmb->mb_size;
  auto w0 = pmhd->w0;
  auto bcc0 = pmhd->bcc0;
  const Real gm1 = pmhd->peos->eos_data.gamma - 1.0;
  const Real cos_half_angle =
      std::cos(opening_half_angle_deg*pi/180.0);

  Kokkos::deep_copy(w0, 0.0);
  Kokkos::deep_copy(bcc0, 0.0);
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  par_for("pgen_dci_3d", DevExeSpace(), 0, nmb1,
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
    const Real alpha = inside_outer*outside_inner*inside_cap;

    const Real ch_partial_density = alpha*solid_density;
    const Real ambient_partial_density = (1.0-alpha)*ambient_density;
    const Real density = ch_partial_density + ambient_partial_density;
    const Real ch_mass_fraction = ch_partial_density/density;
    const Real pressure = density*temperature;

    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = pressure/gm1;
    w0(m, scalar_index, k, j, i) = ch_mass_fraction;
  });

  pmhd->peos->PrimToCons(
      w0, bcc0, pmhd->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! Apply a zero-energy (Dirichlet vacuum) boundary to embedded FLD groups only.
//!
//! AthenaK first applies the ordinary outflow condition to all MHD variables.  This
//! callback then replaces radiation group ghost cells at global faces.  A zero-gradient
//! outflow condition would instead be an insulated FLD wall.

void VacuumRadiationBoundary(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *prad = pmbp->pmhd->ptwo_temp->pradiation;
  auto u0 = pmbp->pmhd->u0;
  auto mb_bcs = pmbp->pmb->mb_bcs;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = indcs.nx2 + 2*ng;
  const int n3 = indcs.nx3 + 2*ng;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int ifirst = prad->ifirst;
  const int ngroups = prad->ngroups;

  par_for("dci_vacuum_rad_x1", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n3-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int g, int k, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, is-n-1) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, j, ie+n+1) = 0.0;
    }
  });

  par_for("dci_vacuum_rad_x2", DevExeSpace(), 0, nmb1, 0, ngroups-1,
          0, n3-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int g, int k, int i) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, js-n-1, i) = 0.0;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
      for (int n = 0; n < ng; ++n) u0(m, ifirst+g, k, je+n+1, i) = 0.0;
    }
  });

  par_for("dci_vacuum_rad_x3", DevExeSpace(), 0, nmb1, 0, ngroups-1,
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

//----------------------------------------------------------------------------------------
//! Integrated physics and conservation diagnostics (MPI reduction is done by writer).

void DCIHistory(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto *ptwo = pmhd->ptwo_temp;
  auto *prad = ptwo->pradiation;

  pdata->nhist = kHistoryFields;
  pdata->label[0] = "laser_Edep";
  pdata->label[1] = "laser_Pdep";
  pdata->label[2] = "mass";
  pdata->label[3] = "CH_mass";
  pdata->label[4] = "mat_E";
  pdata->label[5] = "kin_E";
  pdata->label[6] = "mag_E";
  pdata->label[7] = "eion_E";
  pdata->label[8] = "eele_E";
  pdata->label[9] = "erad00_E";
  pdata->label[10] = "erad01_E";
  pdata->label[11] = "erad02_E";
  pdata->label[12] = "chain_E";
  pdata->label[13] = "abs_B";
  pdata->label[14] = "bier_S";
  pdata->label[15] = "laser_x";
  pdata->label[16] = "rad_x";
  pdata->label[17] = "mix_mass";
  pdata->label[18] = "mom1";
  pdata->label[19] = "volume";

  auto u0 = pmhd->u0;
  auto bcc0 = pmhd->bcc0;
  auto laser_data = pmbp->plaser->cell_data;
  auto size = pmbp->pmb->mb_size;
  const int scalar_index = pmhd->nmhd;
  const int iion = ptwo->iion;
  const int iele = ptwo->iele;
  const int ifirst = prad->ifirst;
  const Real gm1 = pmhd->peos->eos_data.gamma - 1.0;
  const Real fe = ptwo->ElectronHeatCapacityFraction();
  const Real biermann_coefficient = pmhd->pbiermann->coefficient;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  array_sum::GlobalSum sum_this_rank;

  Kokkos::parallel_reduce(
      "dci_3d_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k = (idx-m*nkji)/nji + ks;
        const int j = (idx-m*nkji-(k-ks)*nji)/nx1 + js;
        const int i = idx-m*nkji-(k-ks)*nji-(j-js)*nx1 + is;
        const Real dx1 = size.d_view(m).dx1;
        const Real dx2 = size.d_view(m).dx2;
        const Real dx3 = size.d_view(m).dx3;
        const Real volume = dx1*dx2*dx3;
        const Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min,
                                    size.d_view(m).x1max);
        const Real density = u0(m, IDN, k, j, i);
        const Real xch = u0(m, scalar_index, k, j, i)/density;
        const Real momentum_squared = SQR(u0(m, IM1, k, j, i)) +
                                      SQR(u0(m, IM2, k, j, i)) +
                                      SQR(u0(m, IM3, k, j, i));
        const Real b1 = bcc0(m, IBX, k, j, i);
        const Real b2 = bcc0(m, IBY, k, j, i);
        const Real b3 = bcc0(m, IBZ, k, j, i);
        const Real b2sum = b1*b1 + b2*b2 + b3*b3;
        const Real erad0 = u0(m, ifirst, k, j, i);
        const Real erad1 = u0(m, ifirst+1, k, j, i);
        const Real erad2 = u0(m, ifirst+2, k, j, i);
        const Real erad = erad0 + erad1 + erad2;

        const Real dne_dx1 = fe*(u0(m, IDN, k, j, i+1) -
                                 u0(m, IDN, k, j, i-1))/(2.0*dx1);
        const Real dne_dx2 = fe*(u0(m, IDN, k, j+1, i) -
                                 u0(m, IDN, k, j-1, i))/(2.0*dx2);
        const Real dne_dx3 = fe*(u0(m, IDN, k+1, j, i) -
                                 u0(m, IDN, k-1, j, i))/(2.0*dx3);
        const Real dpe_dx1 = gm1*(u0(m, iele, k, j, i+1) -
                                  u0(m, iele, k, j, i-1))/(2.0*dx1);
        const Real dpe_dx2 = gm1*(u0(m, iele, k, j+1, i) -
                                  u0(m, iele, k, j-1, i))/(2.0*dx2);
        const Real dpe_dx3 = gm1*(u0(m, iele, k+1, j, i) -
                                  u0(m, iele, k-1, j, i))/(2.0*dx3);
        const Real cross1 = dne_dx2*dpe_dx3 - dne_dx3*dpe_dx2;
        const Real cross2 = dne_dx3*dpe_dx1 - dne_dx1*dpe_dx3;
        const Real cross3 = dne_dx1*dpe_dx2 - dne_dx2*dpe_dx1;
        const Real ne = fe*density;
        const Real biermann_source = biermann_coefficient*
            sqrt(cross1*cross1 + cross2*cross2 + cross3*cross3)/
            fmax(ne*ne, 1.0e-60);
        const Real bounded_xch = fmin(fmax(xch, 0.0), 1.0);

        array_sum::GlobalSum local;
        local.the_array[0] = volume*laser_data(m, 1, k, j, i);
        local.the_array[1] = volume*laser_data(m, 0, k, j, i);
        local.the_array[2] = volume*density;
        local.the_array[3] = volume*u0(m, scalar_index, k, j, i);
        local.the_array[4] = volume*u0(m, IEN, k, j, i);
        local.the_array[5] = volume*0.5*momentum_squared/density;
        local.the_array[6] = volume*0.5*b2sum;
        local.the_array[7] = volume*u0(m, iion, k, j, i);
        local.the_array[8] = volume*u0(m, iele, k, j, i);
        local.the_array[9] = volume*erad0;
        local.the_array[10] = volume*erad1;
        local.the_array[11] = volume*erad2;
        local.the_array[12] = volume*(u0(m, IEN, k, j, i)+erad);
        local.the_array[13] = volume*sqrt(b2sum);
        local.the_array[14] = volume*biermann_source;
        local.the_array[15] = volume*x1*laser_data(m, 1, k, j, i);
        local.the_array[16] = volume*x1*erad;
        local.the_array[17] = volume*density*4.0*bounded_xch*(1.0-bounded_xch);
        local.the_array[18] = volume*u0(m, IM1, k, j, i);
        local.the_array[19] = volume;
        sum += local;
      }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));
  Kokkos::fence();

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_rank.the_array[n];
  }
}

} // namespace
