//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser.cpp
//! \brief Static density profiles for laser reflection and refraction reference tests.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "laser/laser.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! Initialize a zero-velocity, zero-field medium with an axial profile plus optional
//! transverse linear and quadratic terms. These profiles provide analytic reference
//! solutions for planar reflection, constant-gradient bending, and symmetric lenses.

void ProblemGenerator::LaserProfile(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->plaser == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Laser profile test requires <mhd> and <laser> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string profile =
      pin->GetOrAddString("problem", "density_profile", "linear");
  int profile_mode = 0;
  if (profile == "linear") {
    profile_mode = 0;
  } else if (profile == "exponential") {
    profile_mode = 1;
  } else if (profile == "two_wall") {
    profile_mode = 2;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Laser density_profile must be linear, exponential, or two_wall"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 0.5);
  Real density_gradient =
      pin->GetOrAddReal("problem", "density_gradient", 1.0);
  Real density_exponent =
      pin->GetOrAddReal("problem", "density_exponent", std::log(3.0));
  Real density_wall_center =
      pin->GetOrAddReal("problem", "density_wall_center", 0.5);
  Real density_wall_half_gap =
      pin->GetOrAddReal("problem", "density_wall_half_gap", 0.2);
  Real density_gradient_x2 =
      pin->GetOrAddReal("problem", "density_gradient_x2", 0.0);
  Real density_gradient_x3 =
      pin->GetOrAddReal("problem", "density_gradient_x3", 0.0);
  Real density_curvature_x2 =
      pin->GetOrAddReal("problem", "density_curvature_x2", 0.0);
  Real density_curvature_x3 =
      pin->GetOrAddReal("problem", "density_curvature_x3", 0.0);
  Real temperature = pin->GetOrAddReal("problem", "temperature", 1.0);
  Real temperature_density_power =
      pin->GetOrAddReal("problem", "temperature_density_power", 0.0);
  Real temperature_reference_density =
      pin->GetOrAddReal("problem", "temperature_reference_density", rho0);
  if (!(rho0 > 0.0) || !(temperature > 0.0) ||
      !std::isfinite(density_wall_center) ||
      !std::isfinite(density_wall_half_gap) || density_wall_half_gap < 0.0 ||
      !(temperature_reference_density > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Laser profile density, temperature, reference density, and wall "
                 "geometry are invalid"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  auto w = pmbp->pmhd->w0;
  auto bcc = pmbp->pmhd->bcc0;
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real x1min = pmy_mesh_->mesh_size.x1min;

  Kokkos::deep_copy(w, 0.0);
  Kokkos::deep_copy(bcc, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);

  par_for("pgen_laser_profile", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max);
    Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max);
    Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max);
    Real distance = x1-x1min;
    Real density;
    if (profile_mode == 0) {
      density = rho0+density_gradient*distance;
    } else if (profile_mode == 1) {
      density = rho0*exp(density_exponent*distance);
    } else {
      Real wall_distance = fmax(fabs(x1-density_wall_center)-
                                density_wall_half_gap, 0.0);
      density = rho0+density_gradient*wall_distance;
    }
    density += density_gradient_x2*x2 + density_gradient_x3*x3 +
               density_curvature_x2*x2*x2 + density_curvature_x3*x3*x3;
    density = fmax(density, 1.0e-20);
    Real local_temperature = temperature*
        pow(density/temperature_reference_density, temperature_density_power);
    w(m, IDN, k, j, i) = density;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = density*local_temperature/gm1;
  });

  pmbp->pmhd->peos->PrimToCons(
      w, bcc, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
}
