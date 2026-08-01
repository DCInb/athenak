#ifndef TWO_TEMPERATURE_BIERMANN_CLOSURE_HPP_
#define TWO_TEMPERATURE_BIERMANN_CLOSURE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file biermann_closure.hpp
//! \brief Device-callable thermodynamic closure for accepted Biermann stage states.

#include "athena.hpp"
#include "materials/material_mixture.hpp"

namespace two_temperature {

struct BiermannClosedState {
  Real density;
  Real internal_energy;
  Real ion_energy;
  Real electron_energy;
  Real material0_mass_fraction;
  int query_flags;
};

//! Reproduce the Newtonian MHD C2P energy selection followed by the Biermann-specific
//! two-temperature closure.  Composite-AMR endpoint states must use the same nonlinear
//! floors and dual-energy branch as accepted fine-grid RK stages.
struct BiermannEndpointClosure {
  materials::MaterialMixtureDevice mixture;
  Real gamma_minus_one;
  Real density_floor;
  Real pressure_floor;
  Real temperature_floor;
  Real entropy_floor;
  Real sigma_max;
  Real dual_energy_eta1;
  bool use_dual_energy;
  bool use_materials;
  bool use_tabular;

  KOKKOS_INLINE_FUNCTION
  Real InternalEnergyFloor(const Real density) const {
    Real floor = pressure_floor/gamma_minus_one;
    if (temperature_floor > 0.0) {
      floor = fmax(floor, density*temperature_floor/gamma_minus_one);
    }
    if (entropy_floor > 0.0) {
      floor = fmax(floor, density*entropy_floor*
                           pow(density, gamma_minus_one)/gamma_minus_one);
    }
    return floor;
  }

  KOKKOS_INLINE_FUNCTION
  BiermannClosedState CloseSelected(
      const Real density, const Real selected_internal_energy,
      const Real raw_electron_energy, const Real raw_material0_fraction) const {
    BiermannClosedState result;
    result.density = density;
    result.material0_mass_fraction = use_materials
        ? mixture.ClampMassFraction(raw_material0_fraction) : 0.0;
    result.query_flags = 0;

    Real internal_energy = fmax(selected_internal_energy, 0.0);
    Real ion_energy_floor = 0.0;
    Real electron_energy_floor = 0.0;
    if (use_materials) {
      const materials::MaterialPressureEnergyState floor =
          mixture.MinimumPressureEnergyState(
              density, result.material0_mass_fraction,
              pressure_floor, temperature_floor);
      ion_energy_floor = density*floor.ion_specific_internal_energy;
      electron_energy_floor = density*floor.electron_specific_internal_energy;
      const bool total_below_floor =
          internal_energy < ion_energy_floor+electron_energy_floor;
      internal_energy = fmax(
          internal_energy, ion_energy_floor+electron_energy_floor);
      result.query_flags = floor.query_flags;
      if (use_tabular &&
          (total_below_floor || raw_electron_energy < electron_energy_floor ||
           internal_energy-raw_electron_energy < ion_energy_floor)) {
        result.query_flags |= materials::ionmix_energy_below_table;
      }
    }

    result.electron_energy = fmin(
        fmax(raw_electron_energy, electron_energy_floor),
        internal_energy-ion_energy_floor);
    result.ion_energy = internal_energy-result.electron_energy;
    result.internal_energy = internal_energy;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  BiermannClosedState CloseConserved(
      const Real raw_density,
      const Real momentum1, const Real momentum2, const Real momentum3,
      const Real total_energy, const Real raw_ion_energy,
      const Real raw_electron_energy, const Real raw_material0_density,
      const Real bcc1, const Real bcc2, const Real bcc3) const {
    const Real magnetic_squared = SQR(bcc1)+SQR(bcc2)+SQR(bcc3);
    const Real effective_density = fmax(
        raw_density, fmax(density_floor, magnetic_squared/sigma_max));
    const Real kinetic_energy = 0.5*(
        SQR(momentum1)+SQR(momentum2)+SQR(momentum3))/effective_density;
    const Real conservative_internal =
        total_energy-kinetic_energy-0.5*magnetic_squared;

    Real selected_internal = conservative_internal;
    if (use_dual_energy) {
      const Real auxiliary_internal =
          fmax(raw_ion_energy, 0.0)+fmax(raw_electron_energy, 0.0);
      const bool use_conservative =
          conservative_internal > 0.0 &&
          (dual_energy_eta1 <= 0.0 ||
           conservative_internal >
               dual_energy_eta1*fmax(total_energy, 1.0e-18));
      selected_internal = use_conservative
          ? conservative_internal : auxiliary_internal;
    }
    selected_internal = fmax(
        selected_internal, InternalEnergyFloor(effective_density));

    Real material0_fraction = 0.0;
    if (use_materials) {
      material0_fraction = mixture.ClampMassFraction(
          raw_material0_density/effective_density);
    }
    return CloseSelected(
        effective_density, selected_internal, raw_electron_energy,
        material0_fraction);
  }
};

} // namespace two_temperature

#endif // TWO_TEMPERATURE_BIERMANN_CLOSURE_HPP_
