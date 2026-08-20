#ifndef MATERIALS_MATERIAL_MIXTURE_HPP_
#define MATERIALS_MATERIAL_MIXTURE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.hpp
//! \brief Multi-material ideal or separate-ion/electron tabular plasma closure.

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"

class ParameterInput;
namespace units {class Units;}

namespace materials {

struct SpeciesProperties {
  Real abar = 1.0;  //!< Mean ion mass in atomic-mass units.
  Real zbar = 1.0;  //!< Ideal-path mean number of free electrons per ion.
  Real zeff = 1.0;  //!< Separate collision model; IONMIX CN4 does not contain Zeff.
  Real t_ei = -1.0; //!< Ion-electron exchange time; negative disables exchange.
};

//----------------------------------------------------------------------------------------
//! \struct MaterialComposition
//! \brief Mass fractions of the active components, normalized to sum to one.
//!
//! A composition is a lazy accessor over either an explicitly supplied fraction array or
//! the passive-scalar fields of one or two cells.  It therefore has a fixed device-side
//! footprint while the number of materials remains a runtime value.  Negative and
//! over-unity inputs are clamped, then all explicit fractions are normalized.  If every
//! input is zero, the final material is selected as the deterministic fallback.

struct MaterialComposition {
  enum Source : int {
    fallback = 0,
    direct_fractions = 1,
    legacy_two_material = 2,
    primitive_cell = 3,
    conserved_cell = 4,
    primitive_pair = 5,
    conserved_restriction = 6
  };

  DvceArray5D<Real> state;
  DvceArray1D<int> scalar_indices;
  const Real *direct = nullptr;
  Real inverse_sum = 0.0;
  Real direct_scale = 1.0;
  Real legacy_y0 = 1.0;
  Real density = 0.0;
  Real weight0 = 0.5;
  Real weight1 = 0.5;
  Real source0_inverse_sum = 0.0;
  Real source1_inverse_sum = 0.0;
  int count = 1;
  int source = fallback;
  int m0 = 0;
  int k0 = 0;
  int j0 = 0;
  int i0 = 0;
  int m1 = 0;
  int k1 = 0;
  int j1 = 0;
  int i1 = 0;
  int restriction_nk = 1;
  Real restriction_weight = 1.0;

  KOKKOS_INLINE_FUNCTION
  static Real Clamp(const Real value) {
    return fmin(fmax(value, 0.0), 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  Real PrimitiveFraction(const int material, const int m, const int k,
                         const int j, const int i,
                         const Real source_inverse_sum) const {
    if (!(source_inverse_sum > 0.0)) {
      return (material == count-1) ? 1.0 : 0.0;
    }
    return Clamp(state(m, scalar_indices(material), k, j, i))*
           source_inverse_sum;
  }

  KOKKOS_INLINE_FUNCTION
  Real RawFraction(const int material) const {
    if (source == direct_fractions) {
      return Clamp(direct[material]*direct_scale);
    }
    if (source == legacy_two_material) {
      return (material == 0) ? legacy_y0 : 1.0-legacy_y0;
    }
    if (source == primitive_cell) {
      return Clamp(state(m0, scalar_indices(material), k0, j0, i0));
    }
    if (source == conserved_cell) {
      return (density > 0.0)
          ? Clamp(state(m0, scalar_indices(material), k0, j0, i0)/density)
          : 0.0;
    }
    if (source == primitive_pair) {
      return Clamp(
          weight0*PrimitiveFraction(
              material, m0, k0, j0, i0, source0_inverse_sum)+
          weight1*PrimitiveFraction(
              material, m1, k1, j1, i1, source1_inverse_sum));
    }
    if (source == conserved_restriction) {
      if (!(density > 0.0)) return 0.0;
      Real partial_density = 0.0;
      for (int dk = 0; dk < restriction_nk; ++dk) {
        for (int dj = 0; dj < 2; ++dj) {
          for (int di = 0; di < 2; ++di) {
            partial_density += restriction_weight*state(
                m0, scalar_indices(material), k0+dk, j0+dj, i0+di);
          }
        }
      }
      return Clamp(partial_density/density);
    }
    return 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator[](const int material) const {
    if (material < 0 || material >= count) return 0.0;
    if (!(inverse_sum > 0.0)) return (material == count-1) ? 1.0 : 0.0;
    return RawFraction(material)*inverse_sum;
  }
};

struct MaterialThermodynamicState {
  Real ion_temperature = 0.0;
  Real electron_temperature = 0.0;
  Real ion_pressure = 0.0;
  Real electron_pressure = 0.0;
  Real ion_specific_internal_energy = 0.0;
  Real electron_specific_internal_energy = 0.0;
  Real mean_ionization = 0.0;
  Real electron_number_density_cgs = 0.0;
  Real effective_charge = 0.0;
  Real sound_speed_squared = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Electron closure needed by operators that evolve only electron energy while density
// and composition remain fixed.  Keeping this result separate from the complete state
// avoids an ion-energy inverse, ion/electron collision properties, and sound speed.
struct MaterialElectronState {
  Real electron_temperature = 0.0;
  Real electron_pressure = 0.0;
  Real electron_number_density_cgs = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Pressure and component energies needed by reduced update paths. Keeping this result
// distinct from MaterialThermodynamicState prevents pressure-only callers from paying
// for ionization, electron-density, effective-charge, or sound-speed queries.
struct MaterialPressureEnergyState {
  Real ion_pressure = 0.0;
  Real electron_pressure = 0.0;
  Real ion_specific_internal_energy = 0.0;
  Real electron_specific_internal_energy = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Canonical component temperatures and bounds diagnostics for transient caches.
struct MaterialTemperatureState {
  Real ion_temperature = 0.0;
  Real electron_temperature = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

struct MaterialExchangeState {
  MaterialThermodynamicState thermodynamics;
  Real ion_specific_internal_energy = 0.0;
  Real electron_specific_internal_energy = 0.0;
  Real energy_residual = 0.0;
  Real temperature_difference_residual = 0.0;
  int iterations = 0;
  int used_fallback = 0;
};

// Exchange result used when a later operator reconstructs the complete material cache.
struct MaterialTransientExchangeState {
  MaterialTemperatureState temperatures;
  Real ion_specific_internal_energy = 0.0;
  Real electron_specific_internal_energy = 0.0;
  Real energy_residual = 0.0;
  Real temperature_difference_residual = 0.0;
  int iterations = 0;
  int used_fallback = 0;
};

//----------------------------------------------------------------------------------------
//! \struct MaterialMixtureDevice
//! \brief Device-copyable closure represented by nmaterials advected mass fractions.
//!
//! In tabular mode a mixed cell has one bulk velocity, one shared ion temperature, and
//! one shared electron temperature.  The selected, explicit mixing rule is
//!
//! rho_0=Y rho, rho_1=(1-Y)rho,
//! e_a=Y e_{a,0}(rho_0,T_a)+(1-Y)e_{a,1}(rho_1,T_a),
//! P_a=P_{a,0}(rho_0,T_a)+P_{a,1}(rho_1,T_a).
//!
//! Below a pure table's minimum density, its pressure is linearly extrapolated to zero
//! while its specific energy and ionization use the minimum-density surface.  This makes
//! the trace-material pressure continuous as Y approaches zero without inventing a
//! finite pressure for an absent material.

struct MaterialMixtureDevice {
  DvceArray1D<SpeciesProperties> species;
  // Table-device structs themselves contain Kokkos views. Store their bytewise device
  // representation rather than nesting them in another typed view, which would invoke
  // host reference-count operations while copying the outer collection.
  DvceArray1D<unsigned char> material_table_storage;
  const IonmixTwoTemperatureTableDevice *material_tables = nullptr;
  DvceArray1D<int> scalar_indices;
  //! Number of active components. Every component owns one passive scalar.
  int nmaterials = 1;
  int scalar_index = -1;  //!< Absolute primitive/conserved variable index.
  bool use_tabular_eos = false;
  Real gamma_minus_one = 2.0/3.0;
  Real density_to_cgs = 1.0;
  Real temperature_to_kelvin = 1.0;
  Real wave_speed_safety = 1.05;

  static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;

  //! Uniform indexed access to the named first two components and the array tail.
  KOKKOS_INLINE_FUNCTION
  const SpeciesProperties &Species(const int index) const {
    return species(index);
  }

  KOKKOS_INLINE_FUNCTION
  SpeciesProperties &Species(const int index) {
    return species(index);
  }

  KOKKOS_INLINE_FUNCTION
  const IonmixTwoTemperatureTableDevice &SpeciesTable(const int index) const {
    return material_tables[index];
  }

  //! Composition from all explicitly advected fractions.  The caller-owned array must
  //! remain live while the returned accessor is used.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromFractions(const Real *fractions) const {
    MaterialComposition result;
    result.count = nmaterials;
    result.source = MaterialComposition::direct_fractions;
    result.direct = fractions;
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  //! Composition from partial material densities, used by small local AMR stencils.
  //! The density array follows the same lifetime rule as CompositionFromFractions.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromPartialDensities(
      const Real *partial_densities, const Real density) const {
    MaterialComposition result;
    result.count = nmaterials;
    result.source = MaterialComposition::direct_fractions;
    result.direct = partial_densities;
    result.direct_scale = (density > 0.0) ? 1.0/density : 0.0;
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  //! Two-material composition preserving the original y0/(1-y0) arithmetic exactly.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromY0(const Real y0_in) const {
    MaterialComposition result;
    result.count = 2;
    result.source = MaterialComposition::legacy_two_material;
    result.legacy_y0 = ClampMassFraction(y0_in);
    result.inverse_sum = 1.0;
    return result;
  }

 private:
  struct ComponentTemperatureState {
    Real temperature = 0.0;
    int query_flags = ionmix_query_in_bounds;
  };

  struct ComponentAtTemperature {
    Real temperature = 0.0;
    Real pressure = 0.0;
    Real specific_internal_energy = 0.0;
    int query_flags = ionmix_query_in_bounds;
  };

  struct MixedDensityCache {
    Real density = 0.0;
  };

  struct MixedEnergyIntervalCache {};

  struct ComponentPairAtTemperature {
    ComponentAtTemperature ion;
    ComponentAtTemperature electron;
  };

  struct NativeMinimumTemperatureState {
    Real temperature = 0.0;
  };

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature SpeciesComponent(
      const IonmixTwoTemperatureTableDevice &table,
      const IonmixComponent component, const Real partial_density,
      const Real temperature) const {
    ComponentAtTemperature result;
    if (!(partial_density > 0.0)) {
      result.temperature = temperature;
      return result;
    }
    const Real minimum_density = table.MinimumDensityCode();
    const Real query_density = fmax(partial_density, minimum_density);
    const IonmixComponentState state = table.ComponentFromRhoTemperature(
        component, query_density, temperature);
    result.temperature = state.temperature;
    result.pressure = state.pressure;
    result.specific_internal_energy = state.specific_internal_energy;
    result.query_flags = state.query_flags;
    if (partial_density < minimum_density) {
      result.pressure *= partial_density/minimum_density;
      result.query_flags |= ionmix_density_below_table;
    }
    return result;
  }

  //! Additive partial-density mixing rule for any component count.  Accumulating left to
  //! right reproduces the original two-material association exactly when count==2.
  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoTemperature(
      const IonmixComponent component, const Real density,
      const Real temperature, const MaterialComposition &mix) const {
    ComponentAtTemperature result;
    Real pressure = 0.0;
    Real specific_internal_energy = 0.0;
    int query_flags = ionmix_query_in_bounds;
    Real representative_temperature = 0.0;
    bool have_representative = false;
    for (int n = 0; n < mix.count; ++n) {
      const ComponentAtTemperature state = SpeciesComponent(
          SpeciesTable(n), component, density*mix[n], temperature);
      pressure += state.pressure;
      specific_internal_energy += mix[n]*state.specific_internal_energy;
      query_flags |= state.query_flags;
      // The canonical temperature is that of the first present component; a component
      // with zero mass fraction reports the requested temperature unchanged.
      if (!have_representative && (mix[n] > 0.0 || n == mix.count-1)) {
        representative_temperature = state.temperature;
        have_representative = true;
      }
    }
    result.temperature = representative_temperature;
    result.pressure = pressure;
    result.specific_internal_energy = specific_internal_energy;
    result.query_flags = query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoTemperature(
      const IonmixComponent component, const Real density,
      const Real temperature, const Real y0_in) const {
    return MixtureComponentFromRhoTemperature(
        component, density, temperature, CompositionFromY0(y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromCachedDensity(
      const IonmixComponent component, const Real density,
      const Real temperature, const MaterialComposition &mix,
      MixedDensityCache &cache) const {
    cache.density = density;
    return MixtureComponentFromRhoTemperature(
        component, density, temperature, mix);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromCachedDensity(
      const IonmixComponent component, const Real density,
      const Real temperature, const Real y0_in, MixedDensityCache &cache) const {
    return MixtureComponentFromCachedDensity(
        component, density, temperature, CompositionFromY0(y0_in), cache);
  }

  KOKKOS_INLINE_FUNCTION
  Real MixtureComponentEnergyFromCachedDensity(
      const IonmixComponent component, const Real temperature,
      const MaterialComposition &mix, const MixedDensityCache &density_cache,
      MixedEnergyIntervalCache &energy_cache) const {
    (void)energy_cache;
    Real energy = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      energy += mix[n]*SpeciesComponent(
          SpeciesTable(n), component, density_cache.density*mix[n],
          temperature).specific_internal_energy;
    }
    return energy;
  }

  KOKKOS_INLINE_FUNCTION
  ComponentTemperatureState SpeciesTemperatureFromRhoTemperature(
      const IonmixTwoTemperatureTableDevice &table,
      const Real partial_density, const Real temperature) const {
    ComponentTemperatureState result;
    if (!(partial_density > 0.0)) {
      result.temperature = temperature;
      return result;
    }
    const Real minimum_density = table.MinimumDensityCode();
    const Real query_density = fmax(partial_density, minimum_density);
    const IonmixTemperatureState state = table.TemperatureFromRhoTemperature(
        query_density, temperature);
    result.temperature = state.temperature;
    result.query_flags = state.query_flags;
    if (partial_density < minimum_density) {
      result.query_flags |= ionmix_density_below_table;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  ComponentTemperatureState MixtureTemperatureFromRhoTemperature(
      const Real density, const Real temperature,
      const MaterialComposition &mix) const {
    ComponentTemperatureState result;
    int query_flags = ionmix_query_in_bounds;
    Real representative_temperature = 0.0;
    bool have_representative = false;
    for (int n = 0; n < mix.count; ++n) {
      const ComponentTemperatureState state =
          SpeciesTemperatureFromRhoTemperature(
              SpeciesTable(n), density*mix[n], temperature);
      query_flags |= state.query_flags;
      if (!have_representative && (mix[n] > 0.0 || n == mix.count-1)) {
        representative_temperature = state.temperature;
        have_representative = true;
      }
    }
    result.temperature = representative_temperature;
    result.query_flags = query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  ComponentTemperatureState MixtureTemperatureFromRhoTemperature(
      const Real density, const Real temperature, const Real y0_in) const {
    return MixtureTemperatureFromRhoTemperature(
        density, temperature, CompositionFromY0(y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState SpeciesPressureEnergyFromRhoTemperature(
      const IonmixTwoTemperatureTableDevice &table,
      const Real partial_density, const Real temperature) const {
    MaterialPressureEnergyState result;
    if (!(partial_density > 0.0)) return result;
    const Real minimum_density = table.MinimumDensityCode();
    const Real query_density = fmax(partial_density, minimum_density);
    const IonmixPressureEnergyState state =
        table.PressureEnergyFromRhoTemperature(query_density, temperature);
    result.ion_pressure = state.ion_pressure;
    result.electron_pressure = state.electron_pressure;
    if (partial_density < minimum_density) {
      const Real pressure_scale = partial_density/minimum_density;
      result.ion_pressure *= pressure_scale;
      result.electron_pressure *= pressure_scale;
    }
    result.ion_specific_internal_energy = state.ion_specific_internal_energy;
    result.electron_specific_internal_energy =
        state.electron_specific_internal_energy;
    result.query_flags = state.query_flags;
    if (partial_density < minimum_density) {
      result.query_flags |= ionmix_density_below_table;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperature(
      const Real density, const Real temperature,
      const MaterialComposition &mix) const {
    MaterialPressureEnergyState result;
    for (int n = 0; n < mix.count; ++n) {
      const MaterialPressureEnergyState state =
          SpeciesPressureEnergyFromRhoTemperature(
              SpeciesTable(n), density*mix[n], temperature);
      result.ion_pressure += state.ion_pressure;
      result.electron_pressure += state.electron_pressure;
      result.ion_specific_internal_energy +=
          mix[n]*state.ion_specific_internal_energy;
      result.electron_specific_internal_energy +=
          mix[n]*state.electron_specific_internal_energy;
      result.query_flags |= state.query_flags;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperature(
      const Real density, const Real temperature, const Real y0_in) const {
    return TabularPressureEnergyFromRhoTemperature(
        density, temperature, CompositionFromY0(y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ion_temperature, y0);
    const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, electron_temperature, y0);
    MaterialPressureEnergyState result;
    result.ion_pressure = ion.pressure;
    result.electron_pressure = electron.pressure;
    result.ion_specific_internal_energy = ion.specific_internal_energy;
    result.electron_specific_internal_energy = electron.specific_internal_energy;
    result.query_flags = ion.query_flags | electron.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState SpeciesPressureEnergyFromRhoMinimumTemperature(
      const IonmixTwoTemperatureTableDevice &table,
      const Real partial_density) const {
    MaterialPressureEnergyState result;
    if (!(partial_density > 0.0)) return result;
    const Real minimum_density = table.MinimumDensityCode();
    const Real query_density = fmax(partial_density, minimum_density);
    const IonmixPressureEnergyState state =
        table.PressureEnergyFromRhoMinimumTemperature(query_density);
    result.ion_pressure = state.ion_pressure;
    result.electron_pressure = state.electron_pressure;
    if (partial_density < minimum_density) {
      const Real pressure_scale = partial_density/minimum_density;
      result.ion_pressure *= pressure_scale;
      result.electron_pressure *= pressure_scale;
    }
    result.ion_specific_internal_energy = state.ion_specific_internal_energy;
    result.electron_specific_internal_energy =
        state.electron_specific_internal_energy;
    result.query_flags = state.query_flags;
    if (partial_density < minimum_density) {
      result.query_flags |= ionmix_density_below_table;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoNativeMinimum(
      const Real density, const Real temperature, const MaterialComposition &mix,
      const bool use_native_minimum) const {
    MaterialPressureEnergyState result;
    for (int n = 0; n < mix.count; ++n) {
      const MaterialPressureEnergyState state =
          (use_native_minimum && mix[n] > 0.0 &&
           SpeciesTable(n).MinimumTemperatureCode() == temperature)
          ? SpeciesPressureEnergyFromRhoMinimumTemperature(
                SpeciesTable(n), density*mix[n])
          : SpeciesPressureEnergyFromRhoTemperature(
                SpeciesTable(n), density*mix[n], temperature);
      result.ion_pressure += state.ion_pressure;
      result.electron_pressure += state.electron_pressure;
      result.ion_specific_internal_energy +=
          mix[n]*state.ion_specific_internal_energy;
      result.electron_specific_internal_energy +=
          mix[n]*state.electron_specific_internal_energy;
      result.query_flags |= state.query_flags;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoNativeMinimum(
      const Real density, const Real temperature, const Real y0_in,
      const bool use_native_minimum) const {
    return TabularPressureEnergyFromRhoNativeMinimum(
        density, temperature, CompositionFromY0(y0_in), use_native_minimum);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState IdealPressureEnergyFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const MaterialComposition &mix) const {
    const Real fe = ElectronHeatCapacityFraction(mix);
    const Real fi = 1.0-fe;
    MaterialPressureEnergyState result;
    result.ion_specific_internal_energy = fi*ion_temperature/gamma_minus_one;
    result.electron_specific_internal_energy =
        fe*electron_temperature/gamma_minus_one;
    result.ion_pressure = density*fi*ion_temperature;
    result.electron_pressure = density*fe*electron_temperature;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState TabularPressureEnergyFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const MaterialComposition &mix) const {
    const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ion_temperature, mix);
    const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, electron_temperature, mix);
    MaterialPressureEnergyState result;
    result.ion_pressure = ion.pressure;
    result.electron_pressure = electron.pressure;
    result.ion_specific_internal_energy = ion.specific_internal_energy;
    result.electron_specific_internal_energy = electron.specific_internal_energy;
    result.query_flags = ion.query_flags | electron.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState IdealPressureEnergyFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    MaterialPressureEnergyState result;
    result.ion_specific_internal_energy =
        fi*ion_temperature/gamma_minus_one;
    result.electron_specific_internal_energy =
        fe*electron_temperature/gamma_minus_one;
    result.ion_pressure = density*fi*ion_temperature;
    result.electron_pressure = density*fe*electron_temperature;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real CommonMinimumTemperature() const {
    Real result = SpeciesTable(0).MinimumTemperatureCode();
    for (int n = 1; n < nmaterials; ++n) {
      result = fmax(result, SpeciesTable(n).MinimumTemperatureCode());
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real CommonMaximumTemperature() const {
    Real result = SpeciesTable(0).MaximumTemperatureCode();
    for (int n = 1; n < nmaterials; ++n) {
      result = fmin(result, SpeciesTable(n).MaximumTemperatureCode());
    }
    return result;
  }

  //! The valid shared-temperature window is set only by the components actually present,
  //! so a pure cell keeps its own table's full range.
  KOKKOS_INLINE_FUNCTION
  Real MinimumTemperatureForComposition(const MaterialComposition &mix) const {
    Real result = 0.0;
    bool have_any = false;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      const Real value = SpeciesTable(n).MinimumTemperatureCode();
      result = have_any ? fmax(result, value) : value;
      have_any = true;
    }
    return have_any ? result : CommonMinimumTemperature();
  }

  KOKKOS_INLINE_FUNCTION
  Real MinimumTemperatureForComposition(const Real y0_in) const {
    return MinimumTemperatureForComposition(CompositionFromY0(y0_in));
  }

  //! Return the shared minimum; native-table ownership is recomputed per material when
  //! evaluating the floor, avoiding a fixed-width bit mask.
  KOKKOS_INLINE_FUNCTION
  NativeMinimumTemperatureState NativeMinimumTemperatureForComposition(
      const MaterialComposition &mix) const {
    NativeMinimumTemperatureState result;
    result.temperature = MinimumTemperatureForComposition(mix);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  NativeMinimumTemperatureState NativeMinimumTemperatureForComposition(
      const Real y0_in) const {
    return NativeMinimumTemperatureForComposition(CompositionFromY0(y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumTemperatureForComposition(const MaterialComposition &mix) const {
    Real result = 0.0;
    bool have_any = false;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      const Real value = SpeciesTable(n).MaximumTemperatureCode();
      result = have_any ? fmin(result, value) : value;
      have_any = true;
    }
    return have_any ? result : CommonMaximumTemperature();
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumTemperatureForComposition(const Real y0_in) const {
    return MaximumTemperatureForComposition(CompositionFromY0(y0_in));
  }

  //! Index of the single present component, or -1 when the cell is genuinely mixed.
  KOKKOS_INLINE_FUNCTION
  int PureComponentIndex(const MaterialComposition &mix) const {
    int found = -1;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      if (found >= 0) return -1;
      found = n;
    }
    // All fractions zero cannot happen for a normalized composition, but treat it as the
    // last component so callers still have a well-defined pure table.
    return (found >= 0) ? found : mix.count-1;
  }

  KOKKOS_INLINE_FUNCTION
  bool AnyTableUsesErrorBounds(const MaterialComposition &mix) const {
    for (int n = 0; n < mix.count; ++n) {
      if (SpeciesTable(n).bounds_error != 0) return true;
    }
    return false;
  }

  KOKKOS_INLINE_FUNCTION
  bool AllPresentTablesExtrapolateHighTemperature(
      const MaterialComposition &mix) const {
    bool have_any = false;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      have_any = true;
      if (SpeciesTable(n).extrapolate_high_temperature == 0) return false;
    }
    return have_any;
  }

  KOKKOS_INLINE_FUNCTION
  bool AllPresentTablesExtrapolateHighTemperature(const Real y0) const {
    return AllPresentTablesExtrapolateHighTemperature(CompositionFromY0(y0));
  }

  KOKKOS_INLINE_FUNCTION
  Real InitialElectronTemperature(
      const Real ion_temperature, const Real electron_to_ion_temperature,
      const Real minimum_temperature, const Real maximum_temperature,
      const bool extrapolate_high_temperature) const {
    const Real scaled_temperature =
        electron_to_ion_temperature*ion_temperature;
    if (!Kokkos::isfinite(scaled_temperature) ||
        scaled_temperature < 0.0) {
      Kokkos::abort("Initial electron temperature is not finite and non-negative.");
    }
    const Real lower_bounded = fmax(scaled_temperature, minimum_temperature);
    return extrapolate_high_temperature
        ? lower_bounded : fmin(lower_bounded, maximum_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoSpecificEnergyCached(
      const IonmixComponent component, const Real density,
      const Real target_energy, const MaterialComposition &mix,
      MixedDensityCache &cache) const {
    if (!Kokkos::isfinite(target_energy)) {
      Kokkos::abort("Mixed IONMIX inverse energy must be finite.");
    }
    const int energy_low_flag = ionmix_energy_below_table;
    const int energy_high_flag = ionmix_energy_above_table;
    // A cell of one pure material uses that table's own logarithmic inverse directly.
    const int pure_index = PureComponentIndex(mix);
    if (pure_index >= 0) {
      const IonmixComponentState state =
          SpeciesTable(pure_index).ComponentFromRhoSpecificEnergy(
              component, density, target_energy);
      ComponentAtTemperature result;
      result.temperature = state.temperature;
      result.pressure = state.pressure;
      result.specific_internal_energy = state.specific_internal_energy;
      result.query_flags = state.query_flags;
      return result;
    }
    const bool error_bounds = AnyTableUsesErrorBounds(mix);
    const Real minimum_temperature = MinimumTemperatureForComposition(mix);
    const Real maximum_temperature = MaximumTemperatureForComposition(mix);
    Real upper_temperature = maximum_temperature;
    ComponentAtTemperature minimum = MixtureComponentFromCachedDensity(
        component, density, minimum_temperature, mix, cache);
    ComponentAtTemperature maximum = MixtureComponentFromCachedDensity(
        component, density, maximum_temperature, mix, cache);
    if (target_energy < minimum.specific_internal_energy) {
      if (error_bounds) {
        Kokkos::abort("Mixed IONMIX inverse energy is below the table range.");
      }
      minimum.query_flags |= energy_low_flag;
      return minimum;
    }
    if (target_energy > maximum.specific_internal_energy) {
      if (error_bounds) {
        Kokkos::abort("Mixed IONMIX inverse energy is above the table range.");
      }
      if (!AllPresentTablesExtrapolateHighTemperature(mix)) {
        maximum.query_flags |= energy_high_flag;
        return maximum;
      }
      // FLASH permits its caloric Newton solve to range above the last IONMIX node.
      // Grow a logarithmic bracket on the continued mixed surface; unlike a pure table,
      // a weighted sum of different endpoint slopes has no analytic inverse.
      constexpr Real log_two = 0.693147180559945309417232121458176568;
      Real log_upper = log(maximum_temperature);
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        log_upper += log_two;
        upper_temperature = exp(log_upper);
        if (!Kokkos::isfinite(upper_temperature)) break;
        maximum = MixtureComponentFromCachedDensity(
            component, density, upper_temperature, mix, cache);
        if (maximum.specific_internal_energy >= target_energy) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Mixed IONMIX high-temperature continuation could not bracket energy.");
      }
    }

    // A mass-weighted sum of geometric surfaces is not itself geometric.  A short
    // safeguarded log-temperature bisection preserves the exact forward rule and also
    // supports component tables with different native temperature grids.
    //
    // The fixed iteration count with no convergence test looks wasteful and was the
    // target of card A1.  It was replaced with a bracketed false-position solver, gated,
    // and reverted: capping this loop at a single iteration does not make the run any
    // faster, so its cost is below measurement here.  See DCI_3D/perf_ledger.md, A1.
    if (target_energy == minimum.specific_internal_energy) return minimum;
    if (target_energy == maximum.specific_internal_energy) return maximum;
    MixedEnergyIntervalCache energy_cache;
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real log_trial = 0.5*(log_low+log_high);
      const Real trial_energy = MixtureComponentEnergyFromCachedDensity(
          component, exp(log_trial), mix, cache, energy_cache);
      if (trial_energy < target_energy) {
        log_low = log_trial;
      } else {
        log_high = log_trial;
      }
    }
    return MixtureComponentFromCachedDensity(
        component, density, exp(0.5*(log_low+log_high)), mix, cache);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoSpecificEnergyCached(
      const IonmixComponent component, const Real density,
      const Real target_energy, const Real y0, MixedDensityCache &cache) const {
    return MixtureComponentFromRhoSpecificEnergyCached(
        component, density, target_energy, CompositionFromY0(y0), cache);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(
      const IonmixComponent component, const Real density,
      const Real target_energy, const MaterialComposition &mix) const {
    MixedDensityCache cache;
    return MixtureComponentFromRhoSpecificEnergyCached(
        component, density, target_energy, mix, cache);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(
      const IonmixComponent component, const Real density,
      const Real target_energy, const Real y0) const {
    MixedDensityCache cache;
    return MixtureComponentFromRhoSpecificEnergyCached(
        component, density, target_energy, CompositionFromY0(y0), cache);
  }

  // Recover the bounds flags of an inverse query without solving for its in-range
  // temperature.  Pure-material table inverses are already logarithmic and inexpensive;
  // a mixed state needs only its two endpoint energies instead of 48 bisection probes.
  KOKKOS_INLINE_FUNCTION
  int MixtureComponentSpecificEnergyQueryFlagsCached(
      const IonmixComponent component, const Real density,
      const Real target_energy, const MaterialComposition &mix,
      MixedDensityCache &cache) const {
    if (!Kokkos::isfinite(target_energy)) {
      Kokkos::abort("Mixed IONMIX inverse energy must be finite.");
    }
    const int pure_index = PureComponentIndex(mix);
    if (pure_index >= 0) {
      const IonmixComponentState inverse =
          SpeciesTable(pure_index).ComponentFromRhoSpecificEnergy(
              component, density, target_energy);
      const ComponentAtTemperature forward = MixtureComponentFromRhoTemperature(
          component, density, inverse.temperature, mix);
      return inverse.query_flags | forward.query_flags;
    }
    const ComponentAtTemperature minimum = MixtureComponentFromCachedDensity(
        component, density, MinimumTemperatureForComposition(mix), mix, cache);
    const ComponentAtTemperature maximum = MixtureComponentFromCachedDensity(
        component, density, MaximumTemperatureForComposition(mix), mix, cache);
    const int query_flags = minimum.query_flags | maximum.query_flags;
    const bool error_bounds = AnyTableUsesErrorBounds(mix);
    if (target_energy < minimum.specific_internal_energy) {
      if (error_bounds) {
        Kokkos::abort("Mixed IONMIX inverse energy is below the table range.");
      }
      const int forward_query_flags = MixtureTemperatureFromRhoTemperature(
          density, minimum.temperature, mix).query_flags;
      return minimum.query_flags | forward_query_flags |
             ionmix_energy_below_table;
    } else if (target_energy > maximum.specific_internal_energy) {
      if (error_bounds) {
        Kokkos::abort("Mixed IONMIX inverse energy is above the table range.");
      }
      if (AllPresentTablesExtrapolateHighTemperature(mix)) {
        constexpr int density_query_flags =
            ionmix_density_below_table | ionmix_density_above_table;
        return query_flags & density_query_flags;
      }
      const int forward_query_flags = MixtureTemperatureFromRhoTemperature(
          density, maximum.temperature, mix).query_flags;
      return maximum.query_flags | forward_query_flags |
             ionmix_energy_above_table;
    }
    if (target_energy == minimum.specific_internal_energy) {
      const int forward_query_flags = MixtureTemperatureFromRhoTemperature(
          density, minimum.temperature, mix).query_flags;
      return minimum.query_flags | forward_query_flags;
    }
    if (target_energy == maximum.specific_internal_energy) {
      const int forward_query_flags = MixtureTemperatureFromRhoTemperature(
          density, maximum.temperature, mix).query_flags;
      return maximum.query_flags | forward_query_flags;
    }
    constexpr int density_query_flags =
        ionmix_density_below_table | ionmix_density_above_table;
    return query_flags & density_query_flags;
  }

  KOKKOS_INLINE_FUNCTION
  int MixtureComponentSpecificEnergyQueryFlagsCached(
      const IonmixComponent component, const Real density,
      const Real target_energy, const Real y0, MixedDensityCache &cache) const {
    return MixtureComponentSpecificEnergyQueryFlagsCached(
        component, density, target_energy, CompositionFromY0(y0), cache);
  }

  KOKKOS_INLINE_FUNCTION
  int MixtureComponentSpecificEnergyQueryFlags(
      const IonmixComponent component, const Real density,
      const Real target_energy, const MaterialComposition &mix) const {
    MixedDensityCache cache;
    return MixtureComponentSpecificEnergyQueryFlagsCached(
        component, density, target_energy, mix, cache);
  }

  KOKKOS_INLINE_FUNCTION
  int MixtureComponentSpecificEnergyQueryFlags(
      const IonmixComponent component, const Real density,
      const Real target_energy, const Real y0) const {
    MixedDensityCache cache;
    return MixtureComponentSpecificEnergyQueryFlagsCached(
        component, density, target_energy, CompositionFromY0(y0), cache);
  }

  KOKKOS_INLINE_FUNCTION
  ComponentPairAtTemperature MixtureComponentsFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    MixedDensityCache cache;
    ComponentPairAtTemperature result;
    result.ion = MixtureComponentFromRhoSpecificEnergyCached(
        IonmixComponent::ion, density, ion_specific_energy, mix, cache);
    result.electron = MixtureComponentFromRhoSpecificEnergyCached(
        IonmixComponent::electron, density, electron_specific_energy, mix, cache);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  ComponentPairAtTemperature MixtureComponentsFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    return MixtureComponentsFromRhoSpecificEnergies(
        density, ion_specific_energy, electron_specific_energy,
        CompositionFromY0(y0));
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularElectronNumberPerAtomicMass(
      const Real density, const Real electron_temperature,
      const MaterialComposition &mix) const {
    Real result = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      const IonmixTwoTemperatureTableDevice &table = SpeciesTable(n);
      const Real rho = fmax(density*mix[n], table.MinimumDensityCode());
      result += mix[n]*table.MeanIonizationFromRhoTemperature(
                    rho, electron_temperature)/Species(n).abar;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularElectronNumberPerAtomicMass(
      const Real density, const Real electron_temperature,
      const Real y0_in) const {
    return TabularElectronNumberPerAtomicMass(
        density, electron_temperature, CompositionFromY0(y0_in));
  }

  //! Electron-density-weighted mean of each component's ionization-scaled Zeff.
  KOKKOS_INLINE_FUNCTION
  Real TabularEffectiveCharge(const Real density, const Real electron_temperature,
                              const MaterialComposition &mix) const {
    Real electron_weight = 0.0;
    Real charge_weight = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      const IonmixTwoTemperatureTableDevice &table = SpeciesTable(n);
      const SpeciesProperties &species = Species(n);
      const Real rho = fmax(density*mix[n], table.MinimumDensityCode());
      const Real zbar = table.MeanIonizationFromRhoTemperature(
          rho, electron_temperature);
      const Real ne = mix[n]*zbar/species.abar;
      electron_weight += ne;
      charge_weight += ne*(species.zeff/species.zbar)*zbar;
    }
    return (electron_weight > 0.0) ? charge_weight/electron_weight : 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularEffectiveCharge(const Real density, const Real electron_temperature,
                              const Real y0_in) const {
    return TabularEffectiveCharge(
        density, electron_temperature, CompositionFromY0(y0_in));
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState TabularStateNoSound(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const MaterialComposition &mix) const {
    const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ion_temperature, mix);
    const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, electron_temperature, mix);
    MaterialThermodynamicState result;
    result.ion_temperature = ion.temperature;
    result.electron_temperature = electron.temperature;
    result.ion_pressure = ion.pressure;
    result.electron_pressure = electron.pressure;
    result.ion_specific_internal_energy = ion.specific_internal_energy;
    result.electron_specific_internal_energy = electron.specific_internal_energy;
    Real electron_weight = 0.0;
    Real ion_weight = 0.0;
    Real effective_charge_weight = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      const IonmixTwoTemperatureTableDevice &table = SpeciesTable(n);
      const SpeciesProperties &species = Species(n);
      const Real rho = fmax(density*mix[n], table.MinimumDensityCode());
      const Real zbar = table.MeanIonizationFromRhoTemperature(
          rho, electron.temperature);
      const Real component_electron_weight = mix[n]*zbar/species.abar;
      electron_weight += component_electron_weight;
      ion_weight += mix[n]/species.abar;
      effective_charge_weight += component_electron_weight*
          (species.zeff/species.zbar)*zbar;
    }
    result.mean_ionization = (ion_weight > 0.0)
        ? electron_weight/ion_weight : 0.0;
    result.electron_number_density_cgs = density*density_to_cgs*
        electron_weight/atomic_mass_unit_cgs;
    result.effective_charge = (electron_weight > 0.0)
        ? effective_charge_weight/electron_weight : 0.0;
    result.query_flags = ion.query_flags | electron.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState TabularStateNoSound(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    return TabularStateNoSound(density, ion_temperature, electron_temperature,
                               CompositionFromY0(y0));
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularSoundSpeedSquared(const Real density, const Real ion_temperature,
                                const Real electron_temperature,
                                const Real ion_pressure,
                                const Real electron_pressure,
                                const MaterialComposition &mix) const {
    constexpr Real log_step = 1.0e-3;
    const Real density_low = density*exp(-log_step);
    Real density_high = density*exp(log_step);
    Real maximum_density = Kokkos::Experimental::infinity<Real>::value;
    for (int n = 0; n < mix.count; ++n) {
      if (!(mix[n] > 0.0)) continue;
      maximum_density = fmin(
          maximum_density, SpeciesTable(n).MaximumDensityCode()/mix[n]);
    }
    density_high = fmin(density_high, maximum_density);

    // Caching the three distinct density locations here (card C2) removes ten of the
    // sixteen density searches and is a natural-looking win.  It is not: measured twice,
    // on a 3-cycle window and again on a 20-cycle steady state, it is slower both times.
    // See DCI_3D/perf_ledger.md, C2 -- these kernels are bound by the dependent chain of
    // scattered table reads, not by the number of searches.
    Real sound_speed_squared = 0.0;
    for (int component_index = 0; component_index < 2; ++component_index) {
      const IonmixComponent component = (component_index == 0)
          ? IonmixComponent::ion : IonmixComponent::electron;
      const Real temperature = (component_index == 0)
          ? ion_temperature : electron_temperature;
      const Real center_pressure = (component_index == 0)
          ? ion_pressure : electron_pressure;
      const Real temperature_low = fmax(
          temperature*exp(-log_step), MinimumTemperatureForComposition(mix));
      const Real candidate_temperature_high = temperature*exp(log_step);
      const Real temperature_high =
          AllPresentTablesExtrapolateHighTemperature(mix)
          ? candidate_temperature_high
          : fmin(candidate_temperature_high,
                 MaximumTemperatureForComposition(mix));
      const ComponentAtTemperature rho_low = MixtureComponentFromRhoTemperature(
          component, density_low, temperature, mix);
      const ComponentAtTemperature rho_high = MixtureComponentFromRhoTemperature(
          component, density_high, temperature, mix);
      const ComponentAtTemperature temp_low = MixtureComponentFromRhoTemperature(
          component, density, temperature_low, mix);
      const ComponentAtTemperature temp_high = MixtureComponentFromRhoTemperature(
          component, density, temperature_high, mix);
      const Real density_span = density_high-density_low;
      const Real temperature_span = temperature_high-temperature_low;
      if (!(density_span > 0.0) || !(temperature_span > 0.0)) continue;
      const Real de_drho =
          (rho_high.specific_internal_energy-rho_low.specific_internal_energy)/
          density_span;
      const Real dp_drho = (rho_high.pressure-rho_low.pressure)/density_span;
      const Real de_dtemperature =
          (temp_high.specific_internal_energy-temp_low.specific_internal_energy)/
          temperature_span;
      const Real dp_dtemperature =
          (temp_high.pressure-temp_low.pressure)/temperature_span;
      if (!(de_dtemperature > 0.0)) continue;
      const Real dtemperature_drho =
          (center_pressure/(density*density)-de_drho)/de_dtemperature;
      sound_speed_squared += dp_drho+dp_dtemperature*dtemperature_drho;
    }
    const Real pressure_scale =
        fmax((ion_pressure+electron_pressure)/density, 0.0);
    sound_speed_squared = fmax(sound_speed_squared, pressure_scale);
    return SQR(wave_speed_safety)*sound_speed_squared;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularSoundSpeedSquared(const Real density, const Real ion_temperature,
                                const Real electron_temperature,
                                const Real ion_pressure,
                                const Real electron_pressure,
                                const Real y0) const {
    return TabularSoundSpeedSquared(density, ion_temperature, electron_temperature,
                                    ion_pressure, electron_pressure,
                                    CompositionFromY0(y0));
  }

 public:
  KOKKOS_INLINE_FUNCTION
  bool UsesTabularEOS() const { return use_tabular_eos; }

  // Bounds used by nonlinear electron-temperature transport operators.  The underlying
  // composition-aware helpers are private because they also serve the table inverses;
  // these narrow wrappers expose only the valid shared-temperature interval.
  KOKKOS_INLINE_FUNCTION
  Real MinimumTransportTemperature(const Real y0) const {
    return MinimumTemperatureForComposition(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumTransportTemperature(const Real y0) const {
    return MaximumTemperatureForComposition(y0);
  }

  // Each table loader reserves enough exponent headroom for all interpolated
  // non-negative pressure contributions (up to three materials times ion/electron). This
  // capability permits exact zero-residual shortcuts that would otherwise have to
  // preserve a legacy inf/inf pressure ratio.
  KOKKOS_INLINE_FUNCTION
  bool TabularPressureSumsAreSafelyFinite() const {
    if (!use_tabular_eos) return false;
    for (int n = 0; n < nmaterials; ++n) {
      if (!SpeciesTable(n).pressure_interpolation_is_safely_finite) return false;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  Real ClampMassFraction(const Real y0) const {
    return fmin(fmax(y0, 0.0), 1.0);
  }

  //! True when more than one component is present, i.e. the cell is genuinely mixed and
  //! no pure-material table inverse applies.  Replaces the two-material `0<y0<1` test.
  KOKKOS_INLINE_FUNCTION
  bool IsMixed(const MaterialComposition &mix) const {
    int present = 0;
    for (int n = 0; n < mix.count; ++n) {
      if (mix[n] > 0.0) ++present;
    }
    return present > 1;
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromPrimitive(const DvceArray5D<Real> &prim,
                                           const int m, const int k,
                                           const int j, const int i) const {
    return CompositionFromPrimitive(prim, m, k, j, i)[0];
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromConserved(const DvceArray5D<Real> &cons,
                                          const int m, const int k,
                                          const int j, const int i,
                                          const Real density_floor = 0.0) const {
    return CompositionFromConserved(cons, m, k, j, i, density_floor)[0];
  }

  //! Full composition from all advected mass fractions.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromPrimitive(const DvceArray5D<Real> &prim,
                                               const int m, const int k,
                                               const int j, const int i) const {
    MaterialComposition result;
    result.state = prim;
    result.scalar_indices = scalar_indices;
    result.count = nmaterials;
    result.source = MaterialComposition::primitive_cell;
    result.m0 = m;
    result.k0 = k;
    result.j0 = j;
    result.i0 = i;
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromConserved(const DvceArray5D<Real> &cons,
                                               const int m, const int k,
                                               const int j, const int i,
                                               const Real density_floor = 0.0) const {
    MaterialComposition result;
    result.state = cons;
    result.scalar_indices = scalar_indices;
    result.count = nmaterials;
    result.source = MaterialComposition::conserved_cell;
    result.m0 = m;
    result.k0 = k;
    result.j0 = j;
    result.i0 = i;
    result.density = fmax(cons(m, IDN, k, j, i), density_floor);
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  //! Composition of a regular 2x2x(nk) conservative restriction stencil.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromConservedRestriction(
      const DvceArray5D<Real> &cons, const int m,
      const int k, const int j, const int i, const int nk,
      const Real weight, const Real density) const {
    MaterialComposition result;
    result.state = cons;
    result.scalar_indices = scalar_indices;
    result.count = nmaterials;
    result.source = MaterialComposition::conserved_restriction;
    result.m0 = m;
    result.k0 = k;
    result.j0 = j;
    result.i0 = i;
    result.restriction_nk = nk;
    result.restriction_weight = weight;
    result.density = density;
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  //! Density-weighted composition of two primitive cells, used at radiation faces.
  KOKKOS_INLINE_FUNCTION
  MaterialComposition CompositionFromPrimitivePair(
      const DvceArray5D<Real> &prim,
      const int m0, const int k0, const int j0, const int i0,
      const int m1, const int k1, const int j1, const int i1,
      const Real raw_weight0, const Real raw_weight1) const {
    MaterialComposition result;
    result.state = prim;
    result.scalar_indices = scalar_indices;
    result.count = nmaterials;
    result.source = MaterialComposition::primitive_pair;
    result.m0 = m0;
    result.k0 = k0;
    result.j0 = j0;
    result.i0 = i0;
    result.m1 = m1;
    result.k1 = k1;
    result.j1 = j1;
    result.i1 = i1;
    Real source0_sum = 0.0;
    Real source1_sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) {
      source0_sum += MaterialComposition::Clamp(
          prim(m0, scalar_indices(n), k0, j0, i0));
      source1_sum += MaterialComposition::Clamp(
          prim(m1, scalar_indices(n), k1, j1, i1));
    }
    result.source0_inverse_sum =
        (source0_sum > 0.0) ? 1.0/source0_sum : 0.0;
    result.source1_inverse_sum =
        (source1_sum > 0.0) ? 1.0/source1_sum : 0.0;
    const Real weight_sum = raw_weight0+raw_weight1;
    if (weight_sum > 0.0) {
      result.weight0 = raw_weight0/weight_sum;
      result.weight1 = raw_weight1/weight_sum;
    }
    Real sum = 0.0;
    for (int n = 0; n < nmaterials; ++n) sum += result.RawFraction(n);
    result.inverse_sum = (sum > 0.0) ? 1.0/sum : 0.0;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real IonNumberPerAtomicMass(const MaterialComposition &mix) const {
    Real result = 0.0;
    for (int n = 0; n < mix.count; ++n) result += mix[n]/Species(n).abar;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real IonNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0/Species(0).abar+(1.0-y0)/Species(1).abar;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerAtomicMass(const MaterialComposition &mix) const {
    Real result = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      result += mix[n]*Species(n).zbar/Species(n).abar;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0*Species(0).zbar/Species(0).abar+
           (1.0-y0)*Species(1).zbar/Species(1).abar;
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanAtomicMass(const Real y0) const {
    return 1.0/IonNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanIonization(const Real y0) const {
    return ElectronNumberPerAtomicMass(y0)/IonNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanParticleMass(const Real y0) const {
    return 1.0/(IonNumberPerAtomicMass(y0)+ElectronNumberPerAtomicMass(y0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacityFraction(const Real y0) const {
    const Real ni = IonNumberPerAtomicMass(y0);
    const Real ne = ElectronNumberPerAtomicMass(y0);
    return ne/(ni+ne);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacityFraction(const Real density,
                                    const Real ion_temperature,
                                    const Real electron_temperature,
                                    const Real y0) const {
    if (!use_tabular_eos) return ElectronHeatCapacityFraction(y0);
    constexpr Real log_step = 1.0e-3;
    const Real ti_low = fmax(
        ion_temperature*exp(-log_step), MinimumTemperatureForComposition(y0));
    const Real candidate_ti_high = ion_temperature*exp(log_step);
    const Real ti_high = AllPresentTablesExtrapolateHighTemperature(y0)
        ? candidate_ti_high
        : fmin(candidate_ti_high, MaximumTemperatureForComposition(y0));
    const Real te_low = fmax(
        electron_temperature*exp(-log_step), MinimumTemperatureForComposition(y0));
    const Real candidate_te_high = electron_temperature*exp(log_step);
    const Real te_high = AllPresentTablesExtrapolateHighTemperature(y0)
        ? candidate_te_high
        : fmin(candidate_te_high, MaximumTemperatureForComposition(y0));
    const Real cvi = (MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ti_high, y0).specific_internal_energy-
        MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ti_low, y0).specific_internal_energy)/
        fmax(ti_high-ti_low, 1.0e-30);
    const Real cve = (MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, te_high, y0).specific_internal_energy-
        MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, te_low, y0).specific_internal_energy)/
        fmax(te_high-te_low, 1.0e-30);
    return (cvi+cve > 0.0) ? cve/(cvi+cve) : ElectronHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonHeatCapacityFraction(const Real y0) const {
    return 1.0-ElectronHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacity(const Real gm1, const Real y0) const {
    return ElectronHeatCapacityFraction(y0)/gm1;
  }

  KOKKOS_INLINE_FUNCTION
  Real IonHeatCapacity(const Real gm1, const Real y0) const {
    return IonHeatCapacityFraction(y0)/gm1;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerGram(const Real y0) const {
    return ElectronNumberPerAtomicMass(y0)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensity(const Real density, const Real y0) const {
    return density*ElectronNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensity(const Real density, const Real y0,
                             const Real electron_temperature) const {
    if (!use_tabular_eos) return ElectronNumberDensity(density, y0);
    return density*TabularElectronNumberPerAtomicMass(
        density, electron_temperature, y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensityCgs(const Real code_density,
                                const Real density_scale_cgs,
                                const Real y0) const {
    return code_density*density_scale_cgs*ElectronNumberPerGram(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensityCgsFromTemperature(
      const Real code_density, const Real y0,
      const Real electron_temperature) const {
    if (!use_tabular_eos) {
      return ElectronNumberDensityCgs(code_density, density_to_cgs, y0);
    }
    return code_density*density_to_cgs*TabularElectronNumberPerAtomicMass(
        code_density, electron_temperature, y0)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real EffectiveCharge(const MaterialComposition &mix) const {
    Real electron_weight = 0.0;
    Real charge_weight = 0.0;
    Real largest_zeff = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      const SpeciesProperties &species = Species(n);
      const Real ne = mix[n]*species.zbar/species.abar;
      electron_weight += ne;
      charge_weight += ne*species.zeff;
      largest_zeff = fmax(largest_zeff, species.zeff);
    }
    return (electron_weight > 0.0) ? charge_weight/electron_weight : largest_zeff;
  }

  KOKKOS_INLINE_FUNCTION
  Real EffectiveCharge(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*Species(0).zbar/Species(0).abar;
    const Real ne1 = (1.0-y0)*Species(1).zbar/Species(1).abar;
    return (ne0+ne1 > 0.0)
               ? (ne0*Species(0).zeff+ne1*Species(1).zeff)/(ne0+ne1)
               : fmax(Species(0).zeff, Species(1).zeff);
  }

  KOKKOS_INLINE_FUNCTION
  Real EffectiveCharge(const Real density, const Real y0,
                       const Real electron_temperature) const {
    return use_tabular_eos
               ? TabularEffectiveCharge(density, electron_temperature, y0)
               : EffectiveCharge(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real CodeTemperatureFromKelvin(const Real kelvin) const {
    return kelvin/temperature_to_kelvin;
  }

  KOKKOS_INLINE_FUNCTION
  Real InitialElectronEnergyFraction(const Real y0,
                                     const Real electron_to_ion_temperature) const {
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    return fe*electron_to_ion_temperature/
           (fi+fe*electron_to_ion_temperature);
  }

  //! Composition-aware entry points.  These are what an N-material problem generator and
  //! its physics operators call; the scalar y0 forms below remain for two-material code.
  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoTemperaturesNoSound(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const MaterialComposition &mix) const {
    if (use_tabular_eos) return TabularStateNoSound(
        density, ion_temperature, electron_temperature, mix);
    const Real fe = ElectronHeatCapacityFraction(mix);
    const Real fi = 1.0-fe;
    MaterialThermodynamicState result;
    result.ion_temperature = ion_temperature;
    result.electron_temperature = electron_temperature;
    result.ion_specific_internal_energy = fi*ion_temperature/gamma_minus_one;
    result.electron_specific_internal_energy =
        fe*electron_temperature/gamma_minus_one;
    result.ion_pressure = density*fi*ion_temperature;
    result.electron_pressure = density*fe*electron_temperature;
    result.mean_ionization =
        ElectronNumberPerAtomicMass(mix)/IonNumberPerAtomicMass(mix);
    result.electron_number_density_cgs = density*density_to_cgs*
        ElectronNumberPerAtomicMass(mix)/atomic_mass_unit_cgs;
    result.effective_charge = EffectiveCharge(mix);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const MaterialComposition &mix) const {
    MaterialThermodynamicState result = StateFromRhoTemperaturesNoSound(
        density, ion_temperature, electron_temperature, mix);
    if (use_tabular_eos) {
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature,
          result.ion_pressure, result.electron_pressure, mix);
      return result;
    }
    result.sound_speed_squared = (1.0+gamma_minus_one)*
        (result.ion_pressure+result.electron_pressure)/density;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacityFraction(const MaterialComposition &mix) const {
    const Real ni = IonNumberPerAtomicMass(mix);
    const Real ne = ElectronNumberPerAtomicMass(mix);
    return ne/(ni+ne);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronTemperature(const Real density,
                           const Real electron_specific_energy,
                           const MaterialComposition &mix) const {
    if (use_tabular_eos) {
      return MixtureComponentFromRhoSpecificEnergy(
          IonmixComponent::electron, density, electron_specific_energy,
          mix).temperature;
    }
    return gamma_minus_one*electron_specific_energy/
           ElectronHeatCapacityFraction(mix);
  }

  //! Electron caloric EOS in the forward (rho, Te) direction.  Source operators that
  //! root-find directly in electron temperature use this reduced query to avoid also
  //! interpolating the unchanged ion component on every nonlinear iteration.
  KOKKOS_INLINE_FUNCTION
  Real ElectronSpecificEnergyFromRhoTemperature(
      const Real density, const Real electron_temperature,
      const MaterialComposition &mix) const {
    if (use_tabular_eos) {
      return MixtureComponentFromRhoTemperature(
          IonmixComponent::electron, density, electron_temperature,
          mix).specific_internal_energy;
    }
    return ElectronHeatCapacityFraction(mix)*electron_temperature/
           gamma_minus_one;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensity(const Real density,
                             const MaterialComposition &mix) const {
    return density*ElectronNumberPerAtomicMass(mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensity(const Real density, const MaterialComposition &mix,
                             const Real electron_temperature) const {
    if (!use_tabular_eos) return ElectronNumberDensity(density, mix);
    return density*TabularElectronNumberPerAtomicMass(
        density, electron_temperature, mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensityCgs(const Real code_density,
                                const Real density_scale_cgs,
                                const MaterialComposition &mix) const {
    return code_density*density_scale_cgs*
           ElectronNumberPerAtomicMass(mix)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real EffectiveCharge(const Real density, const MaterialComposition &mix,
                       const Real electron_temperature) const {
    return use_tabular_eos
               ? TabularEffectiveCharge(density, electron_temperature, mix)
               : EffectiveCharge(mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoTemperaturesNoSound(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    if (use_tabular_eos) return TabularStateNoSound(
        density, ion_temperature, electron_temperature, y0);
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    MaterialThermodynamicState result;
    result.ion_temperature = ion_temperature;
    result.electron_temperature = electron_temperature;
    result.ion_specific_internal_energy = fi*ion_temperature/gamma_minus_one;
    result.electron_specific_internal_energy =
        fe*electron_temperature/gamma_minus_one;
    result.ion_pressure = density*fi*ion_temperature;
    result.electron_pressure = density*fe*electron_temperature;
    result.mean_ionization = MeanIonization(y0);
    result.electron_number_density_cgs =
        ElectronNumberDensityCgs(density, density_to_cgs, y0);
    result.effective_charge = EffectiveCharge(y0);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    MaterialThermodynamicState result = StateFromRhoTemperaturesNoSound(
        density, ion_temperature, electron_temperature, y0);
    if (use_tabular_eos) {
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature,
          result.ion_pressure, result.electron_pressure, y0);
      return result;
    }
    result.sound_speed_squared = (1.0+gamma_minus_one)*
        (result.ion_pressure+result.electron_pressure)/density;
    return result;
  }

 private:
  // Solve the tabular heat-exchange constraint at fixed total specific energy without
  // repeatedly inverting both material tables.  For positive heat capacities, the ion
  // temperature satisfying Te-Ti=target_difference is bracketed by the old Ti and by
  // old Te-target_difference.  A safeguarded secant normally converges in a few forward
  // table evaluations; bisection is retained only for difficult mixed-table segments.
  template <bool transient_state>
  using ExchangeResult = std::conditional_t<
      transient_state, MaterialTransientExchangeState, MaterialExchangeState>;

  template <bool transient_state>
  KOKKOS_INLINE_FUNCTION
  ExchangeResult<transient_state>
  ExchangeStateFromRhoTotalEnergyTemperatureDifference(
      const Real density, const Real old_ion_specific_energy,
      const Real old_electron_specific_energy,
      const Real old_ion_temperature, const Real old_electron_temperature,
      const Real target_difference, const MaterialComposition &mix) const {
    ExchangeResult<transient_state> result;
    const Real total_specific_energy =
        old_ion_specific_energy+old_electron_specific_energy;
    if (!use_tabular_eos) {
      const Real fe = ElectronHeatCapacityFraction(mix);
      const Real ti = gamma_minus_one*total_specific_energy-
                      fe*target_difference;
      if constexpr (transient_state) {
        const Real fi = 1.0-fe;
        result.temperatures.ion_temperature = ti;
        result.temperatures.electron_temperature = ti+target_difference;
        result.ion_specific_internal_energy =
            fi*ti/gamma_minus_one;
      } else {
        result.thermodynamics = StateFromRhoTemperatures(
            density, ti, ti+target_difference, mix);
        result.ion_specific_internal_energy =
            result.thermodynamics.ion_specific_internal_energy;
      }
      result.electron_specific_internal_energy =
          total_specific_energy-result.ion_specific_internal_energy;
      return result;
    }

    Real low_temperature = old_ion_temperature;
    Real high_temperature = old_electron_temperature-target_difference;
    if (low_temperature > high_temperature) {
      const Real swap = low_temperature;
      low_temperature = high_temperature;
      high_temperature = swap;
    }

    ComponentAtTemperature ion_low = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, low_temperature, mix);
    ComponentAtTemperature electron_low = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density,
        low_temperature+target_difference, mix);
    ComponentAtTemperature ion_high = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, high_temperature, mix);
    ComponentAtTemperature electron_high = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density,
        high_temperature+target_difference, mix);
    int exchange_query_flags =
        ion_low.query_flags | electron_low.query_flags |
        ion_high.query_flags | electron_high.query_flags;
    Real residual_low = ion_low.specific_internal_energy+
                        electron_low.specific_internal_energy-
                        total_specific_energy;
    Real residual_high = ion_high.specific_internal_energy+
                         electron_high.specific_internal_energy-
                         total_specific_energy;
    const Real energy_scale = fmax(
        fabs(total_specific_energy),
        fmax(fabs(ion_low.specific_internal_energy)+
             fabs(electron_low.specific_internal_energy),
             fabs(ion_high.specific_internal_energy)+
             fabs(electron_high.specific_internal_energy)));
    // The relative target is intentionally looser than roundoff for double precision,
    // while the epsilon term keeps the same convergence test meaningful in float builds.
    const Real relative_tolerance = fmax(
        static_cast<Real>(1.0e-11),
        static_cast<Real>(64.0)*
            Kokkos::Experimental::epsilon<Real>::value);
    const Real tolerance = relative_tolerance*fmax(
        energy_scale, Kokkos::Experimental::norm_min<Real>::value);
    Real best_temperature = (fabs(residual_low) <= fabs(residual_high))
        ? low_temperature : high_temperature;
    Real best_residual = (fabs(residual_low) <= fabs(residual_high))
        ? residual_low : residual_high;

    const bool zero_width = !(high_temperature > low_temperature);
    bool converged = fabs(best_residual) <= tolerance;
    bool bracketed = converged ||
        (!zero_width && residual_low <= 0.0 && residual_high >= 0.0);
    for (int iteration = 0; iteration < 6 && !converged && bracketed;
         ++iteration) {
      const Real span = high_temperature-low_temperature;
      const Real denominator = residual_high-residual_low;
      Real trial_temperature = (denominator > 0.0)
          ? low_temperature-residual_low*span/denominator
          : 0.5*(low_temperature+high_temperature);
      if (!Kokkos::isfinite(trial_temperature) ||
          !(trial_temperature > low_temperature) ||
          !(trial_temperature < high_temperature)) {
        trial_temperature = 0.5*(low_temperature+high_temperature);
      }
      const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
          IonmixComponent::ion, density, trial_temperature, mix);
      const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
          IonmixComponent::electron, density,
          trial_temperature+target_difference, mix);
      exchange_query_flags |= ion.query_flags | electron.query_flags;
      const Real residual = ion.specific_internal_energy+
                            electron.specific_internal_energy-
                            total_specific_energy;
      ++result.iterations;
      if (fabs(residual) < fabs(best_residual)) {
        best_temperature = trial_temperature;
        best_residual = residual;
      }
      if (fabs(residual) <= tolerance) {
        converged = true;
      } else if (residual < 0.0) {
        low_temperature = trial_temperature;
        residual_low = residual;
      } else {
        high_temperature = trial_temperature;
        residual_high = residual;
      }
    }

    if (!converged && bracketed) {
      result.used_fallback = 1;
      for (int iteration = 0; iteration < 48 && !converged; ++iteration) {
        const Real trial_temperature = 0.5*(low_temperature+high_temperature);
        const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
            IonmixComponent::ion, density, trial_temperature, mix);
        const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
            IonmixComponent::electron, density,
            trial_temperature+target_difference, mix);
        exchange_query_flags |= ion.query_flags | electron.query_flags;
        const Real residual = ion.specific_internal_energy+
                              electron.specific_internal_energy-
                              total_specific_energy;
        ++result.iterations;
        if (fabs(residual) < fabs(best_residual)) {
          best_temperature = trial_temperature;
          best_residual = residual;
        }
        if (fabs(residual) <= tolerance) {
          converged = true;
        } else if (residual < 0.0) {
          low_temperature = trial_temperature;
        } else {
          high_temperature = trial_temperature;
        }
      }
    }

    if (!converged || !bracketed) {
      // Decline the exchange and recover the original conservative component split.  The
      // inverse query makes the returned temperatures and pressures describe those exact
      // component energies (up to the table inverse tolerance), rather than an endpoint
      // selected from a failed temperature bracket.
      result.used_fallback = 2;
      result.ion_specific_internal_energy = old_ion_specific_energy;
      result.electron_specific_internal_energy = old_electron_specific_energy;
      if constexpr (transient_state) {
        const MaterialThermodynamicState recovered =
            StateFromRhoSpecificEnergiesNoSound(
                density, old_ion_specific_energy,
                old_electron_specific_energy, mix);
        result.temperatures.ion_temperature = recovered.ion_temperature;
        result.temperatures.electron_temperature = recovered.electron_temperature;
        result.temperatures.query_flags =
            recovered.query_flags | exchange_query_flags;
        result.energy_residual =
            recovered.ion_specific_internal_energy+
            recovered.electron_specific_internal_energy-total_specific_energy;
        result.temperature_difference_residual =
            recovered.electron_temperature-recovered.ion_temperature-
            target_difference;
      } else {
        result.thermodynamics = StateFromRhoSpecificEnergiesNoSound(
            density, old_ion_specific_energy, old_electron_specific_energy, mix);
        result.thermodynamics.query_flags |= exchange_query_flags;
        result.energy_residual =
            result.thermodynamics.ion_specific_internal_energy+
            result.thermodynamics.electron_specific_internal_energy-
            total_specific_energy;
        result.temperature_difference_residual =
            result.thermodynamics.electron_temperature-
            result.thermodynamics.ion_temperature-target_difference;
      }
      return result;
    }

    const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, best_temperature, mix);
    result.ion_specific_internal_energy =
        ion.specific_internal_energy;
    // Assign the tolerance-bounded residual to the electron component so the conservative
    // sum is exact, then invert that exact electron energy before constructing the cache.
    result.electron_specific_internal_energy =
        total_specific_energy-result.ion_specific_internal_energy;
    const ComponentAtTemperature electron = MixtureComponentFromRhoSpecificEnergy(
        IonmixComponent::electron, density,
        result.electron_specific_internal_energy, mix);
    if constexpr (transient_state) {
      // Retain the same canonical forward round trip and query order as
      // TabularStateNoSound.  Reusing the raw inverse temperature can differ by ulps
      // after its exp/log coordinate round trip, and Te is consumed by radiation.
      const ComponentTemperatureState canonical_ion =
          MixtureTemperatureFromRhoTemperature(
              density, ion.temperature, mix);
      const ComponentTemperatureState canonical_electron =
          MixtureTemperatureFromRhoTemperature(
              density, electron.temperature, mix);
      result.temperatures.ion_temperature = canonical_ion.temperature;
      result.temperatures.electron_temperature = canonical_electron.temperature;
      result.temperatures.query_flags =
          canonical_ion.query_flags | canonical_electron.query_flags;
      result.temperatures.query_flags |=
          exchange_query_flags | ion.query_flags | electron.query_flags;
      result.temperature_difference_residual =
          result.temperatures.electron_temperature-
          result.temperatures.ion_temperature-target_difference;
    } else {
      result.thermodynamics = TabularStateNoSound(
          density, ion.temperature, electron.temperature, mix);
      result.thermodynamics.query_flags |=
          exchange_query_flags | ion.query_flags | electron.query_flags;
      result.temperature_difference_residual =
          result.thermodynamics.electron_temperature-
          result.thermodynamics.ion_temperature-target_difference;
    }
    result.energy_residual = best_residual;
    return result;
  }

 public:
  KOKKOS_INLINE_FUNCTION
  MaterialExchangeState StateFromRhoTotalEnergyTemperatureDifference(
      const Real density, const Real old_ion_specific_energy,
      const Real old_electron_specific_energy,
      const Real old_ion_temperature, const Real old_electron_temperature,
      const Real target_difference, const Real y0) const {
    return ExchangeStateFromRhoTotalEnergyTemperatureDifference<false>(
        density, old_ion_specific_energy, old_electron_specific_energy,
        old_ion_temperature, old_electron_temperature, target_difference,
        CompositionFromY0(y0));
  }

  // Reduced exchange result for a transient cache. Canonical temperatures and query
  // flags remain exact, while solver metadata and authoritative conservative component
  // energies remain complete. No pressure, field-energy, or ionization interpolation
  // is performed for the final transient state.
  KOKKOS_INLINE_FUNCTION
  MaterialTransientExchangeState
  StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
      const Real density, const Real old_ion_specific_energy,
      const Real old_electron_specific_energy,
      const Real old_ion_temperature, const Real old_electron_temperature,
      const Real target_difference, const Real y0) const {
    return ExchangeStateFromRhoTotalEnergyTemperatureDifference<true>(
        density, old_ion_specific_energy, old_electron_specific_energy,
        old_ion_temperature, old_electron_temperature, target_difference,
        CompositionFromY0(y0));
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoSpecificEnergiesNoSound(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    if (use_tabular_eos) {
      const ComponentPairAtTemperature components =
          MixtureComponentsFromRhoSpecificEnergies(
              density, ion_specific_energy, electron_specific_energy, y0);
      const ComponentAtTemperature &ion = components.ion;
      const ComponentAtTemperature &electron = components.electron;
      MaterialThermodynamicState result = TabularStateNoSound(
          density, ion.temperature, electron.temperature, y0);
      result.query_flags |= ion.query_flags | electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    const Real ti = gamma_minus_one*ion_specific_energy/fi;
    const Real te = gamma_minus_one*electron_specific_energy/fe;
    return StateFromRhoTemperaturesNoSound(density, ti, te, y0);
  }

  // Canonical electron temperature, pressure, and free-electron density from electron
  // specific energy.  The tabular branch deliberately repeats the forward query used by
  // the full-state API after the inverse: this retains trace-material pressure scaling
  // below a pure table's minimum density.  The repeat reuses the inverse's own density
  // cache, so it costs one interpolation rather than re-locating the density in both
  // material tables.
  KOKKOS_INLINE_FUNCTION
  MaterialElectronState ElectronStateFromRhoSpecificEnergy(
      const Real density, const Real electron_specific_energy,
      const Real y0) const {
    MaterialElectronState result;
    if (use_tabular_eos) {
      MixedDensityCache cache;
      const ComponentAtTemperature inverse =
          MixtureComponentFromRhoSpecificEnergyCached(
              IonmixComponent::electron, density, electron_specific_energy,
              y0, cache);
      const ComponentAtTemperature electron = MixtureComponentFromCachedDensity(
          IonmixComponent::electron, density, inverse.temperature, y0, cache);
      result.electron_temperature = electron.temperature;
      result.electron_pressure = electron.pressure;
      result.electron_number_density_cgs = ElectronNumberDensityCgsFromTemperature(
          density, y0, electron.temperature);
      result.query_flags = inverse.query_flags | electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(y0);
    result.electron_temperature =
        gamma_minus_one*electron_specific_energy/fe;
    result.electron_pressure = density*fe*result.electron_temperature;
    result.electron_number_density_cgs =
        ElectronNumberDensityCgs(density, density_to_cgs, y0);
    return result;
  }

  // Bounds diagnostics for the ion inverse omitted by electron-only transient closure.
  // This preserves lifetime flags and strict table-bounds failures without paying for
  // the mixed-table root solve whose temperature/pressure result is not consumed.
  KOKKOS_INLINE_FUNCTION
  int IonSpecificEnergyQueryFlags(
      const Real density, const Real ion_specific_energy,
      const Real y0) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    return MixtureComponentSpecificEnergyQueryFlags(
        IonmixComponent::ion, density, ion_specific_energy, y0);
  }

  KOKKOS_INLINE_FUNCTION
  int ElectronSpecificEnergyQueryFlags(
      const Real density, const Real electron_specific_energy,
      const Real y0) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    return MixtureComponentSpecificEnergyQueryFlags(
        IonmixComponent::electron, density, electron_specific_energy, y0);
  }

  // Paired form reuses the two prepared partial-density locations, matching the
  // ion-then-electron evaluation order of the full pressure/energy closure.
  KOKKOS_INLINE_FUNCTION
  int SpecificEnergiesQueryFlags(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    MixedDensityCache cache;
    int query_flags = MixtureComponentSpecificEnergyQueryFlagsCached(
        IonmixComponent::ion, density, ion_specific_energy, y0, cache);
    query_flags |= MixtureComponentSpecificEnergyQueryFlagsCached(
        IonmixComponent::electron, density, electron_specific_energy, y0, cache);
    return query_flags;
  }

  // Reduced state for pressure/floor-only paths.  The tabular branch intentionally
  // retains the inverse-to-forward reconstruction used by the full-state API: at pure
  // endpoints below a table's minimum density, the forward SpeciesComponent query
  // applies the documented pressure scaling to zero density.
  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState PressureEnergyFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    if (use_tabular_eos) {
      const ComponentPairAtTemperature components =
          MixtureComponentsFromRhoSpecificEnergies(
              density, ion_specific_energy, electron_specific_energy, y0);
      const ComponentAtTemperature &ion = components.ion;
      const ComponentAtTemperature &electron = components.electron;
      MaterialPressureEnergyState result =
          TabularPressureEnergyFromRhoTemperatures(
              density, ion.temperature, electron.temperature, y0);
      result.query_flags |= ion.query_flags | electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    const Real ti = gamma_minus_one*ion_specific_energy/fi;
    const Real te = gamma_minus_one*electron_specific_energy/fe;
    return IdealPressureEnergyFromRhoTemperatures(density, ti, te, y0);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    MaterialThermodynamicState result = StateFromRhoSpecificEnergiesNoSound(
        density, ion_specific_energy, electron_specific_energy, y0);
    if (use_tabular_eos) {
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature,
          result.ion_pressure, result.electron_pressure, y0);
    } else {
      result.sound_speed_squared = (1.0+gamma_minus_one)*
          (result.ion_pressure+result.electron_pressure)/density;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState InitialStateFromTotalSpecificEnergy(
      const Real density, const Real total_specific_energy,
      const Real y0, const Real electron_to_ion_temperature) const {
    if (!use_tabular_eos) {
      const Real electron_fraction = InitialElectronEnergyFraction(
          y0, electron_to_ion_temperature);
      return StateFromRhoSpecificEnergies(
          density, (1.0-electron_fraction)*total_specific_energy,
          electron_fraction*total_specific_energy, y0);
    }
    const bool error_bounds = SpeciesTable(0).bounds_error != 0 ||
                              SpeciesTable(1).bounds_error != 0;
    const bool extrapolate_high_temperature =
        AllPresentTablesExtrapolateHighTemperature(y0);
    const Real minimum_temperature = MinimumTemperatureForComposition(y0);
    const Real maximum_temperature = MaximumTemperatureForComposition(y0);
    Real upper_temperature = maximum_temperature;
    const Real low_electron_temperature = InitialElectronTemperature(
        minimum_temperature, electron_to_ion_temperature,
        minimum_temperature, maximum_temperature,
        extrapolate_high_temperature);
    Real high_electron_temperature = InitialElectronTemperature(
        upper_temperature, electron_to_ion_temperature,
        minimum_temperature, maximum_temperature,
        extrapolate_high_temperature);
    MaterialThermodynamicState low = TabularStateNoSound(
        density, minimum_temperature, low_electron_temperature, y0);
    MaterialThermodynamicState high = TabularStateNoSound(
        density, upper_temperature, high_electron_temperature, y0);
    const Real low_energy = low.ion_specific_internal_energy+
                            low.electron_specific_internal_energy;
    const Real high_energy = high.ion_specific_internal_energy+
                             high.electron_specific_internal_energy;
    if (total_specific_energy < low_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is below range.");
      low.query_flags |= ionmix_energy_below_table;
      low.sound_speed_squared = TabularSoundSpeedSquared(
          density, low.ion_temperature, low.electron_temperature,
          low.ion_pressure, low.electron_pressure, y0);
      return low;
    }
    if (total_specific_energy > high_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is above range.");
      if (!extrapolate_high_temperature) {
        high.query_flags |= ionmix_energy_above_table;
        high.sound_speed_squared = TabularSoundSpeedSquared(
            density, high.ion_temperature, high.electron_temperature,
            high.ion_pressure, high.electron_pressure, y0);
        return high;
      }
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        upper_temperature *= 2.0;
        if (!Kokkos::isfinite(upper_temperature)) break;
        high_electron_temperature = InitialElectronTemperature(
            upper_temperature, electron_to_ion_temperature,
            minimum_temperature, maximum_temperature, true);
        high = TabularStateNoSound(
            density, upper_temperature, high_electron_temperature, y0);
        if (high.ion_specific_internal_energy+
                high.electron_specific_internal_energy >=
            total_specific_energy) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Initial mixed IONMIX continuation could not bracket energy.");
      }
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real ti = exp(0.5*(log_low+log_high));
      const Real te = InitialElectronTemperature(
          ti, electron_to_ion_temperature, minimum_temperature,
          maximum_temperature, extrapolate_high_temperature);
      const MaterialThermodynamicState trial = TabularStateNoSound(
          density, ti, te, y0);
      const Real energy = trial.ion_specific_internal_energy+
                          trial.electron_specific_internal_energy;
      if (energy < total_specific_energy) {
        log_low = log(ti);
      } else {
        log_high = log(ti);
      }
    }
    const Real ti = exp(0.5*(log_low+log_high));
    const Real te = InitialElectronTemperature(
        ti, electron_to_ion_temperature, minimum_temperature,
        maximum_temperature, extrapolate_high_temperature);
    return StateFromRhoTemperatures(density, ti, te, y0);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState MinimumStateNoSound(
      const Real density, const Real y0, const Real pressure_floor = 0.0,
      const Real temperature_floor = 0.0) const {
    if (!use_tabular_eos) {
      Real temperature = fmax(temperature_floor, 0.0);
      MaterialThermodynamicState state = StateFromRhoTemperaturesNoSound(
          density, temperature, temperature, y0);
      if (state.ion_pressure+state.electron_pressure < pressure_floor) {
        temperature = pressure_floor/density;
        state = StateFromRhoTemperaturesNoSound(
            density, temperature, temperature, y0);
      }
      return state;
    }
    const Real minimum_temperature =
        fmax(MinimumTemperatureForComposition(y0), temperature_floor);
    const Real maximum_temperature = MaximumTemperatureForComposition(y0);
    const bool extrapolate_high_temperature =
        AllPresentTablesExtrapolateHighTemperature(y0);
    Real upper_temperature = extrapolate_high_temperature
        ? fmax(minimum_temperature, maximum_temperature)
        : maximum_temperature;
    MaterialThermodynamicState state = StateFromRhoTemperaturesNoSound(
        density, minimum_temperature, minimum_temperature, y0);
    if (state.ion_pressure+state.electron_pressure >= pressure_floor) return state;
    MaterialThermodynamicState maximum = StateFromRhoTemperaturesNoSound(
        density, upper_temperature, upper_temperature, y0);
    if (maximum.ion_pressure+maximum.electron_pressure < pressure_floor) {
      if (SpeciesTable(0).bounds_error != 0 || SpeciesTable(1).bounds_error != 0) {
        Kokkos::abort("Mixed IONMIX pressure floor is above the table range.");
      }
      if (!extrapolate_high_temperature) {
        maximum.query_flags |= ionmix_temperature_above_table;
        return maximum;
      }
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        upper_temperature *= 2.0;
        if (!Kokkos::isfinite(upper_temperature)) break;
        maximum = StateFromRhoTemperaturesNoSound(
            density, upper_temperature, upper_temperature, y0);
        if (maximum.ion_pressure+maximum.electron_pressure >= pressure_floor) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Mixed IONMIX continuation could not bracket pressure floor.");
      }
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real temperature = exp(0.5*(log_low+log_high));
      const MaterialThermodynamicState trial = TabularStateNoSound(
          density, temperature, temperature, y0);
      if (trial.ion_pressure+trial.electron_pressure < pressure_floor) {
        log_low = log(temperature);
      } else {
        log_high = log(temperature);
      }
    }
    const Real temperature = exp(0.5*(log_low+log_high));
    return StateFromRhoTemperaturesNoSound(
        density, temperature, temperature, y0);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState MinimumPressureEnergyState(
      const Real density, const Real y0, const Real pressure_floor = 0.0,
      const Real temperature_floor = 0.0) const {
    if (!use_tabular_eos) {
      Real temperature = fmax(temperature_floor, 0.0);
      MaterialPressureEnergyState state =
          IdealPressureEnergyFromRhoTemperatures(
              density, temperature, temperature, y0);
      if (state.ion_pressure+state.electron_pressure < pressure_floor) {
        temperature = pressure_floor/density;
        state = IdealPressureEnergyFromRhoTemperatures(
            density, temperature, temperature, y0);
      }
      return state;
    }
    const NativeMinimumTemperatureState native_minimum =
        NativeMinimumTemperatureForComposition(y0);
    const Real minimum_temperature =
        fmax(native_minimum.temperature, temperature_floor);
    const bool use_native_minimum =
        minimum_temperature == native_minimum.temperature;
    const Real maximum_temperature = MaximumTemperatureForComposition(y0);
    const bool extrapolate_high_temperature =
        AllPresentTablesExtrapolateHighTemperature(y0);
    Real upper_temperature = extrapolate_high_temperature
        ? fmax(minimum_temperature, maximum_temperature)
        : maximum_temperature;
    MaterialPressureEnergyState state = TabularPressureEnergyFromRhoNativeMinimum(
        density, minimum_temperature, y0, use_native_minimum);
    if (state.ion_pressure+state.electron_pressure >= pressure_floor) return state;
    MaterialPressureEnergyState maximum = TabularPressureEnergyFromRhoTemperature(
        density, upper_temperature, y0);
    if (maximum.ion_pressure+maximum.electron_pressure < pressure_floor) {
      if (SpeciesTable(0).bounds_error != 0 || SpeciesTable(1).bounds_error != 0) {
        Kokkos::abort("Mixed IONMIX pressure floor is above the table range.");
      }
      if (!extrapolate_high_temperature) {
        maximum.query_flags |= ionmix_temperature_above_table;
        return maximum;
      }
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        upper_temperature *= 2.0;
        if (!Kokkos::isfinite(upper_temperature)) break;
        maximum = TabularPressureEnergyFromRhoTemperature(
            density, upper_temperature, y0);
        if (maximum.ion_pressure+maximum.electron_pressure >= pressure_floor) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Mixed IONMIX continuation could not bracket pressure floor.");
      }
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real temperature = exp(0.5*(log_low+log_high));
      const MaterialPressureEnergyState trial =
          TabularPressureEnergyFromRhoTemperature(density, temperature, y0);
      if (trial.ion_pressure+trial.electron_pressure < pressure_floor) {
        log_low = log(temperature);
      } else {
        log_high = log(temperature);
      }
    }
    const Real temperature = exp(0.5*(log_low+log_high));
    return TabularPressureEnergyFromRhoTemperature(density, temperature, y0);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState MinimumState(
      const Real density, const Real y0, const Real pressure_floor = 0.0,
      const Real temperature_floor = 0.0) const {
    MaterialThermodynamicState result = MinimumStateNoSound(
        density, y0, pressure_floor, temperature_floor);
    if (use_tabular_eos) {
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature,
          result.ion_pressure, result.electron_pressure, y0);
    } else {
      result.sound_speed_squared = (1.0+gamma_minus_one)*
          (result.ion_pressure+result.electron_pressure)/density;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronTemperature(const Real density,
                           const Real electron_specific_energy,
                           const Real y0) const {
    if (!use_tabular_eos) {
      return gamma_minus_one*electron_specific_energy/
             ElectronHeatCapacityFraction(y0);
    }
    return MixtureComponentFromRhoSpecificEnergy(
        IonmixComponent::electron, density, electron_specific_energy, y0).temperature;
  }

  KOKKOS_INLINE_FUNCTION
  Real IonTemperature(const Real gamma_minus_one_in,
                      const Real ion_specific_energy, const Real y0) const {
    return gamma_minus_one_in*ion_specific_energy/IonHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonTemperatureFromRhoSpecificEnergy(
      const Real density, const Real ion_specific_energy, const Real y0) const {
    if (!use_tabular_eos) {
      return IonTemperature(gamma_minus_one, ion_specific_energy, y0);
    }
    return MixtureComponentFromRhoSpecificEnergy(
        IonmixComponent::ion, density, ion_specific_energy, y0).temperature;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronPressure(const Real density,
                        const Real electron_specific_energy,
                        const Real y0) const {
    if (!use_tabular_eos) return gamma_minus_one*density*electron_specific_energy;
    return MixtureComponentFromRhoSpecificEnergy(
        IonmixComponent::electron, density, electron_specific_energy, y0).pressure;
  }

  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(const Real density,
                             const Real ion_specific_energy,
                             const Real electron_specific_energy,
                             const Real y0, const Real bx,
                             const Real by, const Real bz) const {
    const MaterialThermodynamicState state = StateFromRhoSpecificEnergies(
        density, ion_specific_energy, electron_specific_energy, y0);
    const Real va2 = (SQR(bx)+SQR(by)+SQR(bz))/density;
    const Real vax2 = SQR(bx)/density;
    const Real sum = state.sound_speed_squared+va2;
    const Real discriminant = fmax(
        sum*sum-4.0*state.sound_speed_squared*vax2, 0.0);
    return sqrt(0.5*(sum+sqrt(discriminant)));
  }

  KOKKOS_INLINE_FUNCTION
  Real ExchangeTime(const MaterialComposition &mix) const {
    Real electron_weight = 0.0;
    Real rate = 0.0;
    for (int n = 0; n < mix.count; ++n) {
      const SpeciesProperties &species = Species(n);
      const Real ne = mix[n]*species.zbar/species.abar;
      // A present component asking for instantaneous exchange wins outright.
      if (ne > 0.0 && species.t_ei == 0.0) return 0.0;
      electron_weight += ne;
      if (species.t_ei > 0.0) rate += ne/species.t_ei;
    }
    return (rate > 0.0) ? electron_weight/rate : -1.0;
  }

  KOKKOS_INLINE_FUNCTION
  Real ExchangeTime(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*Species(0).zbar/Species(0).abar;
    const Real ne1 = (1.0-y0)*Species(1).zbar/Species(1).abar;
    if ((ne0 > 0.0 && Species(0).t_ei == 0.0) ||
        (ne1 > 0.0 && Species(1).t_ei == 0.0)) return 0.0;
    Real rate = 0.0;
    if (Species(0).t_ei > 0.0) rate += ne0/Species(0).t_ei;
    if (Species(1).t_ei > 0.0) rate += ne1/Species(1).t_ei;
    return (rate > 0.0) ? (ne0+ne1)/rate : -1.0;
  }

  //--------------------------------------------------------------------------------------
  // Composition forms of the remaining public API.  Each one forwards to the same
  // generalized private helper the scalar version uses, so a two-material deck that
  // passes a composition gets bit-identical results to one passing y0.

  KOKKOS_INLINE_FUNCTION
  Real MeanAtomicMass(const MaterialComposition &mix) const {
    return 1.0/IonNumberPerAtomicMass(mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanParticleMass(const MaterialComposition &mix) const {
    return 1.0/(IonNumberPerAtomicMass(mix)+ElectronNumberPerAtomicMass(mix));
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerGram(const MaterialComposition &mix) const {
    return ElectronNumberPerAtomicMass(mix)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensityCgsFromTemperature(
      const Real code_density, const MaterialComposition &mix,
      const Real electron_temperature) const {
    if (!use_tabular_eos) {
      return ElectronNumberDensityCgs(code_density, density_to_cgs, mix);
    }
    return code_density*density_to_cgs*TabularElectronNumberPerAtomicMass(
        code_density, electron_temperature, mix)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacityFraction(const Real density,
                                    const Real ion_temperature,
                                    const Real electron_temperature,
                                    const MaterialComposition &mix) const {
    if (!use_tabular_eos) return ElectronHeatCapacityFraction(mix);
    constexpr Real log_step = 1.0e-3;
    const Real minimum = MinimumTemperatureForComposition(mix);
    const Real maximum = MaximumTemperatureForComposition(mix);
    const Real ti_low = fmax(ion_temperature*exp(-log_step), minimum);
    const Real candidate_ti_high = ion_temperature*exp(log_step);
    const Real ti_high = AllPresentTablesExtrapolateHighTemperature(mix)
        ? candidate_ti_high : fmin(candidate_ti_high, maximum);
    const Real te_low = fmax(electron_temperature*exp(-log_step), minimum);
    const Real candidate_te_high = electron_temperature*exp(log_step);
    const Real te_high = AllPresentTablesExtrapolateHighTemperature(mix)
        ? candidate_te_high : fmin(candidate_te_high, maximum);
    const Real cvi = (MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ti_high, mix).specific_internal_energy-
        MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ti_low, mix).specific_internal_energy)/
        fmax(ti_high-ti_low, 1.0e-30);
    const Real cve = (MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, te_high, mix).specific_internal_energy-
        MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, te_low, mix).specific_internal_energy)/
        fmax(te_high-te_low, 1.0e-30);
    return (cvi+cve > 0.0) ? cve/(cvi+cve) : ElectronHeatCapacityFraction(mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real MinimumTransportTemperature(const MaterialComposition &mix) const {
    return MinimumTemperatureForComposition(mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumTransportTemperature(const MaterialComposition &mix) const {
    return MaximumTemperatureForComposition(mix);
  }

  KOKKOS_INLINE_FUNCTION
  Real InitialElectronEnergyFraction(
      const MaterialComposition &mix,
      const Real electron_to_ion_temperature) const {
    const Real fe = ElectronHeatCapacityFraction(mix);
    const Real fi = 1.0-fe;
    return fe*electron_to_ion_temperature/
           (fi+fe*electron_to_ion_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoSpecificEnergiesNoSound(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    if (use_tabular_eos) {
      const ComponentPairAtTemperature components =
          MixtureComponentsFromRhoSpecificEnergies(
              density, ion_specific_energy, electron_specific_energy, mix);
      MaterialThermodynamicState result = TabularStateNoSound(
          density, components.ion.temperature,
          components.electron.temperature, mix);
      result.query_flags |=
          components.ion.query_flags | components.electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(mix);
    const Real fi = 1.0-fe;
    return StateFromRhoTemperaturesNoSound(
        density, gamma_minus_one*ion_specific_energy/fi,
        gamma_minus_one*electron_specific_energy/fe, mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    MaterialThermodynamicState result = StateFromRhoSpecificEnergiesNoSound(
        density, ion_specific_energy, electron_specific_energy, mix);
    if (use_tabular_eos) {
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature,
          result.ion_pressure, result.electron_pressure, mix);
    } else {
      result.sound_speed_squared = (1.0+gamma_minus_one)*
          (result.ion_pressure+result.electron_pressure)/density;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState PressureEnergyFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    if (use_tabular_eos) {
      const ComponentPairAtTemperature components =
          MixtureComponentsFromRhoSpecificEnergies(
              density, ion_specific_energy, electron_specific_energy, mix);
      MaterialPressureEnergyState result =
          TabularPressureEnergyFromRhoTemperatures(
              density, components.ion.temperature,
              components.electron.temperature, mix);
      result.query_flags |=
          components.ion.query_flags | components.electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(mix);
    const Real fi = 1.0-fe;
    return IdealPressureEnergyFromRhoTemperatures(
        density, gamma_minus_one*ion_specific_energy/fi,
        gamma_minus_one*electron_specific_energy/fe, mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialElectronState ElectronStateFromRhoSpecificEnergy(
      const Real density, const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    MaterialElectronState result;
    if (use_tabular_eos) {
      MixedDensityCache cache;
      const ComponentAtTemperature inverse =
          MixtureComponentFromRhoSpecificEnergyCached(
              IonmixComponent::electron, density, electron_specific_energy,
              mix, cache);
      const ComponentAtTemperature electron = MixtureComponentFromCachedDensity(
          IonmixComponent::electron, density, inverse.temperature, mix, cache);
      result.electron_temperature = electron.temperature;
      result.electron_pressure = electron.pressure;
      result.electron_number_density_cgs = ElectronNumberDensityCgsFromTemperature(
          density, mix, electron.temperature);
      result.query_flags = inverse.query_flags | electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(mix);
    result.electron_temperature = gamma_minus_one*electron_specific_energy/fe;
    result.electron_pressure = density*fe*result.electron_temperature;
    result.electron_number_density_cgs =
        ElectronNumberDensityCgs(density, density_to_cgs, mix);
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  int IonSpecificEnergyQueryFlags(
      const Real density, const Real ion_specific_energy,
      const MaterialComposition &mix) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    return MixtureComponentSpecificEnergyQueryFlags(
        IonmixComponent::ion, density, ion_specific_energy, mix);
  }

  KOKKOS_INLINE_FUNCTION
  int ElectronSpecificEnergyQueryFlags(
      const Real density, const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    return MixtureComponentSpecificEnergyQueryFlags(
        IonmixComponent::electron, density, electron_specific_energy, mix);
  }

  KOKKOS_INLINE_FUNCTION
  int SpecificEnergiesQueryFlags(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy,
      const MaterialComposition &mix) const {
    if (!use_tabular_eos) return ionmix_query_in_bounds;
    MixedDensityCache cache;
    int query_flags = MixtureComponentSpecificEnergyQueryFlagsCached(
        IonmixComponent::ion, density, ion_specific_energy, mix, cache);
    query_flags |= MixtureComponentSpecificEnergyQueryFlagsCached(
        IonmixComponent::electron, density, electron_specific_energy, mix, cache);
    return query_flags;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState InitialStateFromTotalSpecificEnergy(
      const Real density, const Real total_specific_energy,
      const MaterialComposition &mix,
      const Real electron_to_ion_temperature) const {
    if (!use_tabular_eos) {
      const Real electron_fraction = InitialElectronEnergyFraction(
          mix, electron_to_ion_temperature);
      return StateFromRhoSpecificEnergies(
          density, (1.0-electron_fraction)*total_specific_energy,
          electron_fraction*total_specific_energy, mix);
    }
    const bool error_bounds = AnyTableUsesErrorBounds(mix);
    const bool extrapolate_high_temperature =
        AllPresentTablesExtrapolateHighTemperature(mix);
    const Real minimum_temperature = MinimumTemperatureForComposition(mix);
    const Real maximum_temperature = MaximumTemperatureForComposition(mix);
    Real upper_temperature = maximum_temperature;
    const Real low_electron_temperature = InitialElectronTemperature(
        minimum_temperature, electron_to_ion_temperature,
        minimum_temperature, maximum_temperature,
        extrapolate_high_temperature);
    Real high_electron_temperature = InitialElectronTemperature(
        upper_temperature, electron_to_ion_temperature,
        minimum_temperature, maximum_temperature,
        extrapolate_high_temperature);
    MaterialThermodynamicState low = TabularStateNoSound(
        density, minimum_temperature, low_electron_temperature, mix);
    MaterialThermodynamicState high = TabularStateNoSound(
        density, upper_temperature, high_electron_temperature, mix);
    const Real low_energy = low.ion_specific_internal_energy+
                            low.electron_specific_internal_energy;
    const Real high_energy = high.ion_specific_internal_energy+
                             high.electron_specific_internal_energy;
    if (total_specific_energy < low_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is below range.");
      low.query_flags |= ionmix_energy_below_table;
      low.sound_speed_squared = TabularSoundSpeedSquared(
          density, low.ion_temperature, low.electron_temperature,
          low.ion_pressure, low.electron_pressure, mix);
      return low;
    }
    if (total_specific_energy > high_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is above range.");
      if (!extrapolate_high_temperature) {
        high.query_flags |= ionmix_energy_above_table;
        high.sound_speed_squared = TabularSoundSpeedSquared(
            density, high.ion_temperature, high.electron_temperature,
            high.ion_pressure, high.electron_pressure, mix);
        return high;
      }
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        upper_temperature *= 2.0;
        if (!Kokkos::isfinite(upper_temperature)) break;
        high_electron_temperature = InitialElectronTemperature(
            upper_temperature, electron_to_ion_temperature,
            minimum_temperature, maximum_temperature, true);
        high = TabularStateNoSound(
            density, upper_temperature, high_electron_temperature, mix);
        if (high.ion_specific_internal_energy+
                high.electron_specific_internal_energy >=
            total_specific_energy) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Initial mixed IONMIX continuation could not bracket energy.");
      }
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real ti = exp(0.5*(log_low+log_high));
      const Real te = InitialElectronTemperature(
          ti, electron_to_ion_temperature, minimum_temperature,
          maximum_temperature, extrapolate_high_temperature);
      const MaterialThermodynamicState trial = TabularStateNoSound(
          density, ti, te, mix);
      const Real energy = trial.ion_specific_internal_energy+
                          trial.electron_specific_internal_energy;
      if (energy < total_specific_energy) {
        log_low = log(ti);
      } else {
        log_high = log(ti);
      }
    }
    const Real ti = exp(0.5*(log_low+log_high));
    const Real te = InitialElectronTemperature(
        ti, electron_to_ion_temperature, minimum_temperature,
        maximum_temperature, extrapolate_high_temperature);
    return StateFromRhoTemperatures(density, ti, te, mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialPressureEnergyState MinimumPressureEnergyState(
      const Real density, const MaterialComposition &mix,
      const Real pressure_floor = 0.0,
      const Real temperature_floor = 0.0) const {
    if (!use_tabular_eos) {
      Real temperature = fmax(temperature_floor, 0.0);
      MaterialPressureEnergyState state =
          IdealPressureEnergyFromRhoTemperatures(
              density, temperature, temperature, mix);
      if (state.ion_pressure+state.electron_pressure < pressure_floor) {
        temperature = pressure_floor/density;
        state = IdealPressureEnergyFromRhoTemperatures(
            density, temperature, temperature, mix);
      }
      return state;
    }
    const NativeMinimumTemperatureState native_minimum =
        NativeMinimumTemperatureForComposition(mix);
    const Real minimum_temperature =
        fmax(native_minimum.temperature, temperature_floor);
    const bool use_native_minimum =
        minimum_temperature == native_minimum.temperature;
    const Real maximum_temperature = MaximumTemperatureForComposition(mix);
    const bool extrapolate_high_temperature =
        AllPresentTablesExtrapolateHighTemperature(mix);
    Real upper_temperature = extrapolate_high_temperature
        ? fmax(minimum_temperature, maximum_temperature)
        : maximum_temperature;
    MaterialPressureEnergyState state = TabularPressureEnergyFromRhoNativeMinimum(
        density, minimum_temperature, mix, use_native_minimum);
    if (state.ion_pressure+state.electron_pressure >= pressure_floor) return state;
    MaterialPressureEnergyState maximum = TabularPressureEnergyFromRhoTemperature(
        density, upper_temperature, mix);
    if (maximum.ion_pressure+maximum.electron_pressure < pressure_floor) {
      if (AnyTableUsesErrorBounds(mix)) {
        Kokkos::abort("Mixed IONMIX pressure floor is above the table range.");
      }
      if (!extrapolate_high_temperature) {
        maximum.query_flags |= ionmix_temperature_above_table;
        return maximum;
      }
      bool bracketed = false;
      for (int iteration = 0; iteration < 64; ++iteration) {
        upper_temperature *= 2.0;
        if (!Kokkos::isfinite(upper_temperature)) break;
        maximum = TabularPressureEnergyFromRhoTemperature(
            density, upper_temperature, mix);
        if (maximum.ion_pressure+maximum.electron_pressure >= pressure_floor) {
          bracketed = true;
          break;
        }
      }
      if (!bracketed) {
        Kokkos::abort(
            "Mixed IONMIX continuation could not bracket pressure floor.");
      }
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(upper_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real temperature = exp(0.5*(log_low+log_high));
      const MaterialPressureEnergyState trial =
          TabularPressureEnergyFromRhoTemperature(density, temperature, mix);
      if (trial.ion_pressure+trial.electron_pressure < pressure_floor) {
        log_low = log(temperature);
      } else {
        log_high = log(temperature);
      }
    }
    const Real temperature = exp(0.5*(log_low+log_high));
    return TabularPressureEnergyFromRhoTemperature(density, temperature, mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialTransientExchangeState
  StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
      const Real density, const Real old_ion_specific_energy,
      const Real old_electron_specific_energy,
      const Real old_ion_temperature, const Real old_electron_temperature,
      const Real target_difference, const MaterialComposition &mix) const {
    return ExchangeStateFromRhoTotalEnergyTemperatureDifference<true>(
        density, old_ion_specific_energy, old_electron_specific_energy,
        old_ion_temperature, old_electron_temperature, target_difference, mix);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialExchangeState StateFromRhoTotalEnergyTemperatureDifference(
      const Real density, const Real old_ion_specific_energy,
      const Real old_electron_specific_energy,
      const Real old_ion_temperature, const Real old_electron_temperature,
      const Real target_difference, const MaterialComposition &mix) const {
    return ExchangeStateFromRhoTotalEnergyTemperatureDifference<false>(
        density, old_ion_specific_energy, old_electron_specific_energy,
        old_ion_temperature, old_electron_temperature, target_difference, mix);
  }
};

class MaterialMixture {
 public:
  // 'block' names the fluid block that owns this closure ("hydro" or "mhd"); it supplies
  // the default ion/electron exchange time.  Reading it from a fixed block name would
  // silently create the other fluid's input block.
  MaterialMixture(ParameterInput *pin, const std::string &block, int first_user_scalar,
                  int nuser_scalars, Real gamma, units::Units *unit_system = nullptr);
  ~MaterialMixture() = default;

  MaterialMixtureDevice DeviceData() const { return data_; }
  int ScalarIndex() const { return data_.scalar_index; }
  int NumberOfMaterials() const { return data_.nmaterials; }
  bool UsesTabularEOS() const { return data_.use_tabular_eos; }

 private:
  MaterialMixtureDevice data_;
  std::vector<std::unique_ptr<IonmixTwoTemperatureTable>> tables_;
};

} // namespace materials

#endif // MATERIALS_MATERIAL_MIXTURE_HPP_
