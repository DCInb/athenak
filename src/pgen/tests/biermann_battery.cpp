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
//! \brief Initialize smooth periodic pressure and density gradients.
//!
//! The electron pressure varies along x while electron number density varies
//! along y, giving an analytic early-time B3 source proportional to
//! cos(kx)cos(ky).  Optional, default-zero pressure-y/pressure-z and density-z
//! amplitudes extend the same problem to three dimensions.  They also permit a
//! mixed pressure field at exactly uniform electron density, for which the
//! Biermann curl must vanish.  The problem remains smooth and periodic, so it
//! also checks flux synchronization across blocks.

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
  Real pressure_x2_amplitude =
      pin->GetOrAddReal("problem", "pressure_x2_amplitude", 0.0);
  Real pressure_x3_amplitude =
      pin->GetOrAddReal("problem", "pressure_x3_amplitude", 0.0);
  Real density_x3_amplitude =
      pin->GetOrAddReal("problem", "density_x3_amplitude", 0.0);
  Real checkerboard_b3_amplitude =
      pin->GetOrAddReal("problem", "checkerboard_b3_amplitude", 0.0);
  Real compression_rate =
      pin->GetOrAddReal("problem", "compression_rate", 0.0);
  Real compression_rate_x1 = pin->GetOrAddReal(
      "problem", "compression_rate_x1", compression_rate);
  Real compression_rate_x2 = pin->GetOrAddReal(
      "problem", "compression_rate_x2", compression_rate);
  Real compression_rate_x3 = pin->GetOrAddReal(
      "problem", "compression_rate_x3", compression_rate);
  if (!std::isfinite(checkerboard_b3_amplitude)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem/checkerboard_b3_amplitude must be finite" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(compression_rate) ||
      !std::isfinite(compression_rate_x1) ||
      !std::isfinite(compression_rate_x2) ||
      !std::isfinite(compression_rate_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "problem compression rates must be finite" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real wave_number = 2.0 * std::acos(-1.0);
  int material_scalar = -1;
  Real material0_fraction = 1.0;
  Real material0_fraction_x1_amplitude = 0.0;
  Real material0_fraction_x2_amplitude = 0.0;
  Real material0_fraction_x3_amplitude = 0.0;
  materials::MaterialMixtureDevice material_mixture;
  if (pmbp->pmhd->pmaterials != nullptr) {
    material_mixture = pmbp->pmhd->pmaterials->DeviceData();
    material_scalar = material_mixture.scalar_index;
    material0_fraction = pin->GetOrAddReal(
        "problem", "material0_fraction", 1.0);
    material0_fraction_x1_amplitude = pin->GetOrAddReal(
        "problem", "material0_fraction_x1_amplitude", 0.0);
    material0_fraction_x2_amplitude = pin->GetOrAddReal(
        "problem", "material0_fraction_x2_amplitude", 0.0);
    material0_fraction_x3_amplitude = pin->GetOrAddReal(
        "problem", "material0_fraction_x3_amplitude", 0.0);
    const Real total_material_amplitude =
        std::abs(material0_fraction_x1_amplitude) +
        std::abs(material0_fraction_x2_amplitude) +
        std::abs(material0_fraction_x3_amplitude);
    if (!std::isfinite(material0_fraction) ||
        !std::isfinite(total_material_amplitude) ||
        material0_fraction-total_material_amplitude < 0.0 ||
        material0_fraction+total_material_amplitude > 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "problem/material0_fraction and its sinusoidal amplitudes must "
                << "remain finite and in [0,1]"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  Kokkos::deep_copy(w, 0.0);
  Kokkos::deep_copy(bcc, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmbp->pmhd->b0.x3f, 0.0);
  auto b3f = pmbp->pmhd->b0.x3f;
  const Real mesh_x1min = pmy_mesh_->mesh_size.x1min;
  const Real mesh_x2min = pmy_mesh_->mesh_size.x2min;

  par_for(
      "pgen_biermann", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real x1 = CellCenterX(i - is, indcs.nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real x2 = CellCenterX(j - js, indcs.nx2, size.d_view(m).x2min,
                              size.d_view(m).x2max);
        Real x3 = CellCenterX(k - ks, indcs.nx3, size.d_view(m).x3min,
                              size.d_view(m).x3max);
        Real density = rho0 * exp(
            density_amplitude * sin(wave_number * x2) +
            density_x3_amplitude * sin(wave_number * x3));
        Real pressure = p0 * exp(
            pressure_amplitude * sin(wave_number * x1) +
            pressure_x2_amplitude * sin(wave_number * x2) +
            pressure_x3_amplitude * sin(wave_number * x3));
        // Optional Nyquist mode for the stability-limiter regression.  B3 may vary
        // arbitrarily in x1/x2 without contributing to div(B), and is held constant
        // in x3.  Deriving the parity from physical location keeps the pattern
        // continuous across MeshBlock boundaries instead of resetting it per block.
        const int gi = static_cast<int>(floor(
            (x1-mesh_x1min)/size.d_view(m).dx1));
        const int gj = static_cast<int>(floor(
            (x2-mesh_x2min)/size.d_view(m).dx2));
        const Real b3 = ((gi+gj) & 1) == 0
            ? checkerboard_b3_amplitude : -checkerboard_b3_amplitude;
        w(m, IDN, k, j, i) = density;
        w(m, IVX, k, j, i) = -compression_rate_x1*x1;
        w(m, IVY, k, j, i) = -compression_rate_x2*x2;
        w(m, IVZ, k, j, i) = -compression_rate_x3*x3;
        w(m, IEN, k, j, i) = pressure / gm1;
        bcc(m, IBZ, k, j, i) = b3;
        b3f(m, k, j, i) = b3;
        if (k == ke) b3f(m, k+1, j, i) = b3;
        if (material_scalar >= 0) {
          const Real y0 = material0_fraction +
              material0_fraction_x1_amplitude*sin(wave_number*x1) +
              material0_fraction_x2_amplitude*sin(wave_number*x2) +
              material0_fraction_x3_amplitude*sin(wave_number*x3);
          for (int n=0; n<material_mixture.nmaterials; ++n) {
            w(m, material_mixture.scalar_indices(n), k, j, i) = 0.0;
          }
          if (material_mixture.nmaterials == 1) {
            w(m, material_mixture.scalar_indices(0), k, j, i) = 1.0;
          } else {
            w(m, material_mixture.scalar_indices(0), k, j, i) = y0;
            w(m, material_mixture.scalar_indices(
                material_mixture.nmaterials-1), k, j, i) = 1.0-y0;
          }
        }
      });

  pmbp->pmhd->peos->PrimToCons(w, bcc, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
}
