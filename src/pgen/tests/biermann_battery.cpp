//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file biermann_battery.cpp
//! \brief Smooth crossed-gradient problem for the 2T Biermann battery.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \brief Initialize p=p0 exp[a_p sin(k x)] and rho=rho0 exp[a_rho sin(k y)].
//!
//! The electron pressure varies along x while electron number density varies
//! along y, giving an analytic early-time B3 source proportional to
//! cos(kx)cos(ky).  The problem remains smooth and periodic, so it also checks
//! flux synchronization across blocks.

void ProblemGenerator::BiermannBattery(ParameterInput *pin,
                                       const bool restart) {
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Biermann battery test requires <mhd> (the initial state does not "
              << "depend on the battery, so <mhd>/biermann_battery=false is allowed "
              << "as a zero-field control)" << std::endl;
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
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);
  Real density_amplitude =
      pin->GetOrAddReal("problem", "density_amplitude", 0.2);
  Real pressure_amplitude =
      pin->GetOrAddReal("problem", "pressure_amplitude", 0.2);
  Real wave_number = 2.0 * std::acos(-1.0);
  int material_scalar = -1;
  Real material0_fraction = 1.0;
  if (pmbp->pmhd->pmaterials != nullptr) {
    material_scalar = pmbp->pmhd->pmaterials->DeviceData().scalar_index;
    material0_fraction = pin->GetOrAddReal(
        "problem", "material0_fraction", 1.0);
    if (!std::isfinite(material0_fraction) || material0_fraction < 0.0 ||
        material0_fraction > 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/material0_fraction must be finite and in [0,1]"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  Kokkos::deep_copy(w, 0.0);
  Kokkos::deep_copy(bcc, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);

  par_for(
      "pgen_biermann", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real x1 = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real x2 = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
        Real density = rho0 * exp(density_amplitude * sin(wave_number * x2));
        Real pressure = p0 * exp(pressure_amplitude * sin(wave_number * x1));
        w(m, IDN, k, j, i) = density;
        w(m, IVX, k, j, i) = 0.0;
        w(m, IVY, k, j, i) = 0.0;
        w(m, IVZ, k, j, i) = 0.0;
        w(m, IEN, k, j, i) = pressure / gm1;
        if (material_scalar >= 0) {
          w(m, material_scalar, k, j, i) = material0_fraction;
        }
      });

  pmbp->pmhd->peos->PrimToCons(w, bcc, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
}
