//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser.cpp
//! \brief Static planar density profiles for laser reflection reference tests.

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
//! Initialize a zero-velocity, zero-field medium with a density profile along x1 and
//! constant material temperature. The profile is deliberately simple enough to provide
//! analytic critical-surface locations for laser regression tests.

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
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Laser density_profile must be linear or exponential"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 0.5);
  Real density_gradient =
      pin->GetOrAddReal("problem", "density_gradient", 1.0);
  Real density_exponent =
      pin->GetOrAddReal("problem", "density_exponent", std::log(3.0));
  Real temperature = pin->GetOrAddReal("problem", "temperature", 1.0);
  if (!(rho0 > 0.0) || !(temperature > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Laser profile density and temperature must be positive"
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
    Real distance = x1-x1min;
    Real density = (profile_mode == 0)
        ? rho0 + density_gradient*distance
        : rho0*exp(density_exponent*distance);
    density = fmax(density, 1.0e-20);
    w(m, IDN, k, j, i) = density;
    w(m, IVX, k, j, i) = 0.0;
    w(m, IVY, k, j, i) = 0.0;
    w(m, IVZ, k, j, i) = 0.0;
    w(m, IEN, k, j, i) = density*temperature/gm1;
  });

  pmbp->pmhd->peos->PrimToCons(
      w, bcc, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
}
