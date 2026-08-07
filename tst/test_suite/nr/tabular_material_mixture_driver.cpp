//========================================================================================
//! \file tabular_material_mixture_driver.cpp
//! \brief Unequal-grid mixed IONMIX closure and material-LLF checks.

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/rsolvers/material_llf_mhd.hpp"

namespace {

materials::MaterialMixtureDevice CloneMixture(
    const materials::MaterialMixtureDevice &source, const std::string &label) {
  materials::MaterialMixtureDevice result = source;
  result.species = DvceArray1D<materials::SpeciesProperties>(
      label+"-species", source.nmaterials);
  Kokkos::deep_copy(result.species, source.species);
  if (source.scalar_indices.extent_int(0) > 0) {
    result.scalar_indices = DvceArray1D<int>(
        label+"-scalar-indices", source.scalar_indices.extent_int(0));
    Kokkos::deep_copy(result.scalar_indices, source.scalar_indices);
  }
  return result;
}

void SetMaterialTables(
    materials::MaterialMixtureDevice &mixture,
    const std::vector<materials::IonmixTwoTemperatureTableDevice> &tables,
    const std::string &label) {
  const std::size_t bytes =
      tables.size()*sizeof(materials::IonmixTwoTemperatureTableDevice);
  HostArray1D<unsigned char> host_storage(label+"-host", bytes);
  std::memcpy(host_storage.data(), tables.data(), bytes);
  mixture.material_table_storage =
      DvceArray1D<unsigned char>(label+"-device", bytes);
  Kokkos::deep_copy(mixture.material_table_storage, host_storage);
  mixture.material_tables =
      reinterpret_cast<const materials::IonmixTwoTemperatureTableDevice *>(
          mixture.material_table_storage.data());
}

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(const Real actual, const Real expected,
                 const Real tolerance = 3.0e-10) {
  return Kokkos::isfinite(actual) &&
         fabs(actual-expected) <= tolerance*fmax(1.0, fabs(expected));
}

// The radiation-coupled exchange path carries only canonical temperatures, query flags,
// authoritative component energies, and solver/conservative metadata. Those values feed
// radiation or conserved-energy updates and must remain exactly equal.
KOKKOS_INLINE_FUNCTION
bool ExactReducedExchangeMatch(
    const materials::MaterialExchangeState &full,
    const materials::MaterialTransientExchangeState &reduced) {
  return reduced.temperatures.ion_temperature ==
             full.thermodynamics.ion_temperature &&
         reduced.temperatures.electron_temperature ==
             full.thermodynamics.electron_temperature &&
         reduced.temperatures.query_flags ==
             full.thermodynamics.query_flags &&
         reduced.ion_specific_internal_energy ==
             full.ion_specific_internal_energy &&
         reduced.electron_specific_internal_energy ==
             full.electron_specific_internal_energy &&
         reduced.energy_residual == full.energy_residual &&
         reduced.temperature_difference_residual ==
             full.temperature_difference_residual &&
         reduced.iterations == full.iterations &&
         reduced.used_fallback == full.used_fallback;
}

KOKKOS_INLINE_FUNCTION
bool ExactPressureEnergyMatch(
    const materials::MaterialPressureEnergyState &actual,
    const materials::MaterialThermodynamicState &expected) {
  return actual.ion_pressure == expected.ion_pressure &&
         actual.electron_pressure == expected.electron_pressure &&
         actual.ion_specific_internal_energy ==
             expected.ion_specific_internal_energy &&
         actual.electron_specific_internal_energy ==
             expected.electron_specific_internal_energy &&
         actual.query_flags == expected.query_flags;
}

struct UncachedComponentAtTemperature {
  Real temperature = 0.0;
  Real pressure = 0.0;
  Real specific_internal_energy = 0.0;
  int query_flags = materials::ionmix_query_in_bounds;
};

struct UncachedMaterialStates {
  materials::MaterialThermodynamicState no_sound;
  materials::MaterialThermodynamicState full;
};

KOKKOS_INLINE_FUNCTION
UncachedComponentAtTemperature UncachedSpeciesComponent(
    const materials::IonmixTwoTemperatureTableDevice &table,
    const materials::IonmixComponent component, const Real partial_density,
    const Real temperature) {
  UncachedComponentAtTemperature result;
  if (!(partial_density > 0.0)) {
    result.temperature = temperature;
    return result;
  }
  const Real minimum_density = table.MinimumDensityCode();
  const Real query_density = fmax(partial_density, minimum_density);
  const auto state = table.ComponentFromRhoTemperature(
      component, query_density, temperature);
  result.temperature = state.temperature;
  result.pressure = state.pressure;
  result.specific_internal_energy = state.specific_internal_energy;
  result.query_flags = state.query_flags;
  if (partial_density < minimum_density) {
    result.pressure *= partial_density/minimum_density;
    result.query_flags |= materials::ionmix_density_below_table;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
UncachedComponentAtTemperature UncachedMixtureComponentFromRhoTemperature(
    const materials::MaterialMixtureDevice &mixture,
    const materials::IonmixComponent component, const Real density,
    const Real temperature, const Real y0_in) {
  const Real y0 = mixture.ClampMassFraction(y0_in);
  const Real y1 = 1.0-y0;
  const auto state0 = UncachedSpeciesComponent(
      mixture.SpeciesTable(0), component, density*y0, temperature);
  const auto state1 = UncachedSpeciesComponent(
      mixture.SpeciesTable(1), component, density*y1, temperature);
  UncachedComponentAtTemperature result;
  result.temperature = (y0 > 0.0) ? state0.temperature : state1.temperature;
  result.pressure = state0.pressure+state1.pressure;
  result.specific_internal_energy =
      y0*state0.specific_internal_energy+y1*state1.specific_internal_energy;
  result.query_flags = state0.query_flags | state1.query_flags;
  return result;
}

KOKKOS_INLINE_FUNCTION
UncachedComponentAtTemperature CopyTableComponent(
    const materials::IonmixComponentState &state) {
  UncachedComponentAtTemperature result;
  result.temperature = state.temperature;
  result.pressure = state.pressure;
  result.specific_internal_energy = state.specific_internal_energy;
  result.query_flags = state.query_flags;
  return result;
}

KOKKOS_INLINE_FUNCTION
UncachedComponentAtTemperature UncachedMixtureComponentFromRhoSpecificEnergy(
    const materials::MaterialMixtureDevice &mixture,
    const materials::IonmixComponent component, const Real density,
    const Real target_energy, const Real y0_in) {
  if (!Kokkos::isfinite(target_energy)) {
    Kokkos::abort("Uncached mixed inverse energy must be finite.");
  }
  const Real y0 = mixture.ClampMassFraction(y0_in);
  if (y0 >= 1.0) {
    return CopyTableComponent(
        mixture.SpeciesTable(0).ComponentFromRhoSpecificEnergy(
            component, density, target_energy));
  }
  if (y0 <= 0.0) {
    return CopyTableComponent(
        mixture.SpeciesTable(1).ComponentFromRhoSpecificEnergy(
            component, density, target_energy));
  }

  const bool error_bounds = mixture.SpeciesTable(0).bounds_error != 0 ||
                            mixture.SpeciesTable(1).bounds_error != 0;
  const Real minimum_temperature = fmax(
      mixture.SpeciesTable(0).MinimumTemperatureCode(),
      mixture.SpeciesTable(1).MinimumTemperatureCode());
  const Real maximum_temperature = fmin(
      mixture.SpeciesTable(0).MaximumTemperatureCode(),
      mixture.SpeciesTable(1).MaximumTemperatureCode());
  auto minimum = UncachedMixtureComponentFromRhoTemperature(
      mixture, component, density, minimum_temperature, y0);
  auto maximum = UncachedMixtureComponentFromRhoTemperature(
      mixture, component, density, maximum_temperature, y0);
  if (target_energy < minimum.specific_internal_energy) {
    if (error_bounds) {
      Kokkos::abort("Uncached mixed inverse energy is below the table range.");
    }
    minimum.query_flags |= materials::ionmix_energy_below_table;
    return minimum;
  }
  if (target_energy > maximum.specific_internal_energy) {
    if (error_bounds) {
      Kokkos::abort("Uncached mixed inverse energy is above the table range.");
    }
    maximum.query_flags |= materials::ionmix_energy_above_table;
    return maximum;
  }
  if (target_energy == minimum.specific_internal_energy) return minimum;
  if (target_energy == maximum.specific_internal_energy) return maximum;

  Real log_low = log(minimum_temperature);
  Real log_high = log(maximum_temperature);
  for (int iteration = 0; iteration < 48; ++iteration) {
    const Real log_trial = 0.5*(log_low+log_high);
    const auto trial = UncachedMixtureComponentFromRhoTemperature(
        mixture, component, density, exp(log_trial), y0);
    if (trial.specific_internal_energy < target_energy) {
      log_low = log_trial;
    } else {
      log_high = log_trial;
    }
  }
  return UncachedMixtureComponentFromRhoTemperature(
      mixture, component, density, exp(0.5*(log_low+log_high)), y0);
}

KOKKOS_INLINE_FUNCTION
UncachedMaterialStates UncachedStatesFromRhoSpecificEnergies(
    const materials::MaterialMixtureDevice &mixture, const Real density,
    const Real ion_specific_energy, const Real electron_specific_energy,
    const Real y0) {
  const auto ion = UncachedMixtureComponentFromRhoSpecificEnergy(
      mixture, materials::IonmixComponent::ion, density,
      ion_specific_energy, y0);
  const auto electron = UncachedMixtureComponentFromRhoSpecificEnergy(
      mixture, materials::IonmixComponent::electron, density,
      electron_specific_energy, y0);
  UncachedMaterialStates result;
  result.no_sound = mixture.StateFromRhoTemperaturesNoSound(
      density, ion.temperature, electron.temperature, y0);
  result.no_sound.query_flags |= ion.query_flags | electron.query_flags;
  result.full = mixture.StateFromRhoTemperatures(
      density, ion.temperature, electron.temperature, y0);
  result.full.query_flags |= ion.query_flags | electron.query_flags;
  return result;
}

KOKKOS_INLINE_FUNCTION
materials::MaterialThermodynamicState
UncachedStateFromRhoSpecificEnergiesNoSound(
    const materials::MaterialMixtureDevice &mixture, const Real density,
    const Real ion_specific_energy, const Real electron_specific_energy,
    const Real y0) {
  // Exchange fallback 2 stops after this canonical no-sound reconstruction.
  const auto ion = UncachedMixtureComponentFromRhoSpecificEnergy(
      mixture, materials::IonmixComponent::ion, density,
      ion_specific_energy, y0);
  const auto electron = UncachedMixtureComponentFromRhoSpecificEnergy(
      mixture, materials::IonmixComponent::electron, density,
      electron_specific_energy, y0);
  auto result = mixture.StateFromRhoTemperaturesNoSound(
      density, ion.temperature, electron.temperature, y0);
  result.query_flags |= ion.query_flags | electron.query_flags;
  return result;
}

// Independent legacy-order Exchange oracle. All fixed-density probes and the final
// conservative electron inverse use the ordinary, unprepared table queries above. The
// final public state reconstruction intentionally remains the canonical fresh query.
KOKKOS_INLINE_FUNCTION
materials::MaterialExchangeState
UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
    const materials::MaterialMixtureDevice &mixture, const Real density,
    const Real old_ion_specific_energy, const Real old_electron_specific_energy,
    const Real old_ion_temperature, const Real old_electron_temperature,
    const Real target_difference, const Real y0) {
  materials::MaterialExchangeState result;
  const Real total_specific_energy =
      old_ion_specific_energy+old_electron_specific_energy;
  if (!mixture.UsesTabularEOS()) {
    const Real fe = mixture.ElectronHeatCapacityFraction(y0);
    const Real ti = mixture.gamma_minus_one*total_specific_energy-
                    fe*target_difference;
    result.thermodynamics = mixture.StateFromRhoTemperatures(
        density, ti, ti+target_difference, y0);
    result.ion_specific_internal_energy =
        result.thermodynamics.ion_specific_internal_energy;
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

  auto ion_low = UncachedMixtureComponentFromRhoTemperature(
      mixture, materials::IonmixComponent::ion, density,
      low_temperature, y0);
  auto electron_low = UncachedMixtureComponentFromRhoTemperature(
      mixture, materials::IonmixComponent::electron, density,
      low_temperature+target_difference, y0);
  auto ion_high = UncachedMixtureComponentFromRhoTemperature(
      mixture, materials::IonmixComponent::ion, density,
      high_temperature, y0);
  auto electron_high = UncachedMixtureComponentFromRhoTemperature(
      mixture, materials::IonmixComponent::electron, density,
      high_temperature+target_difference, y0);
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
  const Real relative_tolerance = fmax(
      static_cast<Real>(1.0e-11),
      static_cast<Real>(64.0)*Kokkos::Experimental::epsilon<Real>::value);
  const Real tolerance = relative_tolerance*fmax(
      energy_scale, Kokkos::Experimental::norm_min<Real>::value);
  Real best_temperature = (fabs(residual_low) <= fabs(residual_high))
      ? low_temperature : high_temperature;
  Real best_residual = (fabs(residual_low) <= fabs(residual_high))
      ? residual_low : residual_high;

  const bool zero_width = !(high_temperature > low_temperature);
  bool converged = fabs(best_residual) <= tolerance;
  const bool bracketed = converged ||
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
    const auto ion = UncachedMixtureComponentFromRhoTemperature(
        mixture, materials::IonmixComponent::ion, density,
        trial_temperature, y0);
    const auto electron = UncachedMixtureComponentFromRhoTemperature(
        mixture, materials::IonmixComponent::electron, density,
        trial_temperature+target_difference, y0);
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
      const auto ion = UncachedMixtureComponentFromRhoTemperature(
          mixture, materials::IonmixComponent::ion, density,
          trial_temperature, y0);
      const auto electron = UncachedMixtureComponentFromRhoTemperature(
          mixture, materials::IonmixComponent::electron, density,
          trial_temperature+target_difference, y0);
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
    result.used_fallback = 2;
    result.ion_specific_internal_energy = old_ion_specific_energy;
    result.electron_specific_internal_energy = old_electron_specific_energy;
    result.thermodynamics = UncachedStateFromRhoSpecificEnergiesNoSound(
        mixture, density, old_ion_specific_energy,
        old_electron_specific_energy, y0);
    result.thermodynamics.query_flags |= exchange_query_flags;
    result.energy_residual =
        result.thermodynamics.ion_specific_internal_energy+
        result.thermodynamics.electron_specific_internal_energy-
        total_specific_energy;
    result.temperature_difference_residual =
        result.thermodynamics.electron_temperature-
        result.thermodynamics.ion_temperature-target_difference;
    return result;
  }

  const auto ion = UncachedMixtureComponentFromRhoTemperature(
      mixture, materials::IonmixComponent::ion, density,
      best_temperature, y0);
  result.ion_specific_internal_energy = ion.specific_internal_energy;
  result.electron_specific_internal_energy =
      total_specific_energy-result.ion_specific_internal_energy;
  const auto electron = UncachedMixtureComponentFromRhoSpecificEnergy(
      mixture, materials::IonmixComponent::electron, density,
      result.electron_specific_internal_energy, y0);
  result.thermodynamics = mixture.StateFromRhoTemperaturesNoSound(
      density, ion.temperature, electron.temperature, y0);
  result.thermodynamics.query_flags |=
      exchange_query_flags | ion.query_flags | electron.query_flags;
  result.energy_residual = best_residual;
  result.temperature_difference_residual =
      result.thermodynamics.electron_temperature-
      result.thermodynamics.ion_temperature-target_difference;
  return result;
}

KOKKOS_INLINE_FUNCTION
bool ExactMaterialStateMatch(
    const materials::MaterialThermodynamicState &actual,
    const materials::MaterialThermodynamicState &expected) {
  return actual.ion_temperature == expected.ion_temperature &&
         actual.electron_temperature == expected.electron_temperature &&
         actual.ion_pressure == expected.ion_pressure &&
         actual.electron_pressure == expected.electron_pressure &&
         actual.ion_specific_internal_energy ==
             expected.ion_specific_internal_energy &&
         actual.electron_specific_internal_energy ==
             expected.electron_specific_internal_energy &&
         actual.mean_ionization == expected.mean_ionization &&
         actual.electron_number_density_cgs ==
             expected.electron_number_density_cgs &&
         actual.effective_charge == expected.effective_charge &&
         actual.sound_speed_squared == expected.sound_speed_squared &&
         actual.query_flags == expected.query_flags;
}

KOKKOS_INLINE_FUNCTION
bool ExactMaterialExchangeMatch(
    const materials::MaterialExchangeState &actual,
    const materials::MaterialExchangeState &expected) {
  return ExactMaterialStateMatch(actual.thermodynamics, expected.thermodynamics) &&
         actual.ion_specific_internal_energy ==
             expected.ion_specific_internal_energy &&
         actual.electron_specific_internal_energy ==
             expected.electron_specific_internal_energy &&
         actual.energy_residual == expected.energy_residual &&
         actual.temperature_difference_residual ==
             expected.temperature_difference_residual &&
         actual.iterations == expected.iterations &&
         actual.used_fallback == expected.used_fallback;
}

KOKKOS_INLINE_FUNCTION
bool ExactPreparedAndUncachedInverseMatch(
    const materials::MaterialMixtureDevice &mixture, const Real density,
    const Real ion_specific_energy, const Real electron_specific_energy,
    const Real y0) {
  const UncachedMaterialStates expected =
      UncachedStatesFromRhoSpecificEnergies(
          mixture, density, ion_specific_energy, electron_specific_energy, y0);
  const auto no_sound = mixture.StateFromRhoSpecificEnergiesNoSound(
      density, ion_specific_energy, electron_specific_energy, y0);
  const auto full = mixture.StateFromRhoSpecificEnergies(
      density, ion_specific_energy, electron_specific_energy, y0);
  return ExactMaterialStateMatch(no_sound, expected.no_sound) &&
         ExactMaterialStateMatch(full, expected.full);
}

KOKKOS_INLINE_FUNCTION
bool ExactSingleComponentInverseMatch(
    const materials::MaterialMixtureDevice &mixture, const Real density,
    const Real ion_specific_energy, const Real electron_specific_energy,
    const Real y0) {
  const UncachedComponentAtTemperature ion =
      UncachedMixtureComponentFromRhoSpecificEnergy(
          mixture, materials::IonmixComponent::ion, density,
          ion_specific_energy, y0);
  const UncachedComponentAtTemperature electron =
      UncachedMixtureComponentFromRhoSpecificEnergy(
          mixture, materials::IonmixComponent::electron, density,
          electron_specific_energy, y0);
  return mixture.IonTemperatureFromRhoSpecificEnergy(
             density, ion_specific_energy, y0) == ion.temperature &&
         mixture.ElectronTemperature(
             density, electron_specific_energy, y0) == electron.temperature &&
         mixture.ElectronPressure(
             density, electron_specific_energy, y0) == electron.pressure;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: tabular_material_mixture_driver CH_TABLE HE_TABLE\n";
    return EXIT_FAILURE;
  }
  Kokkos::initialize(argc, argv);
  int return_code = EXIT_SUCCESS;
  {
    materials::IonmixTwoTemperatureTableOptions options;
    options.bounds_policy = materials::IonmixBoundsPolicy::error;
    materials::IonmixTwoTemperatureTable ch(argv[1], options);
    materials::IonmixTwoTemperatureTable he(argv[2], options);
    options.geometric_interpolation = false;
    materials::IonmixTwoTemperatureTable nonlinear_ch(argv[1], options);
    materials::IonmixTwoTemperatureTable nonlinear_he(argv[2], options);

    materials::MaterialMixtureDevice mixture;
    mixture.nmaterials = 2;
    mixture.species = DvceArray1D<materials::SpeciesProperties>(
        "base-mixture-species", 2);
    mixture.Species(0).abar = 6.5;
    mixture.Species(0).zbar = 2.0;
    mixture.Species(0).zeff = 6.0;
    mixture.Species(1).abar = 4.0;
    mixture.Species(1).zbar = 4.0;
    mixture.Species(1).zeff = 8.0;
    const auto ch_table = ch.DeviceData();
    const auto he_table = he.DeviceData();
    SetMaterialTables(mixture, {ch_table, he_table}, "base-mixture-tables");
    mixture.use_tabular_eos = true;
    mixture.density_to_cgs = 1.0;
    mixture.temperature_to_kelvin = 1.0;
    mixture.wave_speed_safety = 1.05;
    materials::MaterialMixtureDevice single_mixture = mixture;
    single_mixture.nmaterials = 1;
    single_mixture.species = DvceArray1D<materials::SpeciesProperties>(
        "single-mixture-species", 1);
    single_mixture.Species(0) = mixture.Species(0);
    SetMaterialTables(single_mixture, {ch_table}, "single-mixture-tables");
    materials::MaterialMixtureDevice many_mixture = mixture;
    many_mixture.nmaterials = 4;
    many_mixture.species = DvceArray1D<materials::SpeciesProperties>(
        "many-mixture-species", 4);
    many_mixture.Species(0) = mixture.Species(0);
    many_mixture.Species(1) = mixture.Species(1);
    many_mixture.Species(2).abar = 12.0;
    many_mixture.Species(2).zbar = 1.5;
    many_mixture.Species(2).zeff = 3.0;
    many_mixture.Species(3).abar = 20.0;
    many_mixture.Species(3).zbar = 5.0;
    many_mixture.Species(3).zeff = 7.0;
    SetMaterialTables(
        many_mixture, {ch_table, he_table, ch_table, he_table},
        "many-mixture-tables");
    materials::MaterialMixtureDevice nonlinear_mixture =
        CloneMixture(mixture, "nonlinear-mixture");
    SetMaterialTables(
        nonlinear_mixture,
        {nonlinear_ch.DeviceData(), nonlinear_he.DeviceData()},
        "nonlinear-mixture-tables");
    materials::MaterialMixtureDevice clamped_mixture =
        CloneMixture(mixture, "clamped-mixture");
    auto clamped_ch_table = ch_table;
    auto clamped_he_table = he_table;
    clamped_ch_table.bounds_error = 0;
    clamped_he_table.bounds_error = 0;
    SetMaterialTables(
        clamped_mixture, {clamped_ch_table, clamped_he_table},
        "clamped-mixture-tables");
    materials::MaterialMixtureDevice ideal_mixture =
        CloneMixture(mixture, "ideal-mixture");
    ideal_mixture.use_tabular_eos = false;
    ideal_mixture.gamma_minus_one = 2.0/3.0;
    materials::MaterialMixtureDevice ideal_many_mixture =
        CloneMixture(many_mixture, "ideal-many-mixture");
    ideal_many_mixture.use_tabular_eos = false;
    ideal_many_mixture.gamma_minus_one = 2.0/3.0;

    // Clone the paired value/log-value views, then make both component-energy surfaces
    // vary with density. The primary fixtures intentionally use density-independent
    // energies, which cannot expose a wrong prepared-density interpolation token in
    // Exchange.
    materials::MaterialMixtureDevice density_dependent_mixture =
        CloneMixture(mixture, "density-dependent-mixture");
    const int ch_density_count = mixture.SpeciesTable(0).ndensity;
    const int ch_temperature_count = mixture.SpeciesTable(0).ntemperature;
    const int he_density_count = mixture.SpeciesTable(1).ndensity;
    const int he_temperature_count = mixture.SpeciesTable(1).ntemperature;
    DvceArray3D<Real> density_dependent_ch_values(
        "density-dependent-ch-values",
        materials::IonmixTwoTemperatureTableDevice::nfields,
        ch_density_count, ch_temperature_count);
    DvceArray3D<Real> density_dependent_he_values(
        "density-dependent-he-values",
        materials::IonmixTwoTemperatureTableDevice::nfields,
        he_density_count, he_temperature_count);
    DvceArray3D<Real> density_dependent_ch_log_values(
        "density-dependent-ch-log-values",
        materials::IonmixTwoTemperatureTableDevice::nfields,
        ch_density_count, ch_temperature_count);
    DvceArray3D<Real> density_dependent_he_log_values(
        "density-dependent-he-log-values",
        materials::IonmixTwoTemperatureTableDevice::nfields,
        he_density_count, he_temperature_count);
    Kokkos::deep_copy(
        density_dependent_ch_values, mixture.SpeciesTable(0).values);
    Kokkos::deep_copy(
        density_dependent_he_values, mixture.SpeciesTable(1).values);
    const int ch_value_count =
        materials::IonmixTwoTemperatureTableDevice::nfields*
        ch_density_count*ch_temperature_count;
    const int he_value_count =
        materials::IonmixTwoTemperatureTableDevice::nfields*
        he_density_count*he_temperature_count;
    Kokkos::parallel_for(
        "make-density-dependent-material-energies",
        Kokkos::RangePolicy<>(0, ch_value_count+he_value_count),
        KOKKOS_LAMBDA(const int index) {
          if (index < ch_value_count) {
            const int temperature_index = index%ch_temperature_count;
            const int row = index/ch_temperature_count;
            const int density_index = row%ch_density_count;
            const int field = row/ch_density_count;
            if (field == materials::IonmixTwoTemperatureTableDevice::
                             ion_specific_internal_energy ||
                field == materials::IonmixTwoTemperatureTableDevice::
                             electron_specific_internal_energy) {
              density_dependent_ch_values(
                  field, density_index, temperature_index) *=
                  1.0+static_cast<Real>(density_index);
            }
            const Real value = density_dependent_ch_values(
                field, density_index, temperature_index);
            density_dependent_ch_log_values(
                field, density_index, temperature_index) =
                (value > 0.0) ? log(value) : 0.0;
            return;
          }
          const int he_index = index-ch_value_count;
          const int temperature_index = he_index%he_temperature_count;
          const int row = he_index/he_temperature_count;
          const int density_index = row%he_density_count;
          const int field = row/he_density_count;
          if (field == materials::IonmixTwoTemperatureTableDevice::
                           ion_specific_internal_energy ||
              field == materials::IonmixTwoTemperatureTableDevice::
                           electron_specific_internal_energy) {
            density_dependent_he_values(
                field, density_index, temperature_index) *=
                1.0+static_cast<Real>(density_index);
          }
          const Real value = density_dependent_he_values(
              field, density_index, temperature_index);
          density_dependent_he_log_values(
              field, density_index, temperature_index) =
              (value > 0.0) ? log(value) : 0.0;
        });
    Kokkos::fence();
    auto density_dependent_ch_table = ch_table;
    auto density_dependent_he_table = he_table;
    density_dependent_ch_table.values = density_dependent_ch_values;
    density_dependent_he_table.values = density_dependent_he_values;
    density_dependent_ch_table.log_values = density_dependent_ch_log_values;
    density_dependent_he_table.log_values = density_dependent_he_log_values;
    SetMaterialTables(
        density_dependent_mixture,
        {density_dependent_ch_table, density_dependent_he_table},
        "density-dependent-mixture-tables");

    int failures = 0;
    Kokkos::parallel_reduce(
        "unequal_grid_material_closure", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int, int &local_failures) {
          const Real density = 2.0;
          const Real ych = 0.5;
          const auto mixed = mixture.StateFromRhoTemperatures(
              density, 100.0, 100.0, ych);
          if (!NearlyEqual(mixed.ion_specific_internal_energy, 250.0) ||
              !NearlyEqual(mixed.electron_specific_internal_energy, 450.0) ||
              !NearlyEqual(mixed.ion_pressure, 150.0) ||
              !NearlyEqual(mixed.electron_pressure, 350.0) ||
              !NearlyEqual(mixed.ion_temperature, 100.0) ||
              !NearlyEqual(mixed.electron_temperature, 100.0) ||
              !NearlyEqual(mixed.mean_ionization, 2.0) ||
              !(mixed.sound_speed_squared > 0.0)) {
            ++local_failures;
          }

          const Real ne_ch = 0.5*2.0/6.5;
          const Real ne_he = 0.5*2.0/4.0;
          const Real expected_zeff = (ne_ch*6.0+ne_he*4.0)/(ne_ch+ne_he);
          if (!NearlyEqual(mixed.effective_charge, expected_zeff)) {
            ++local_failures;
          }

          // A single material is a valid explicit composition. Any positive fraction
          // and the deterministic all-zero fallback both normalize to that material.
          Real single_fraction[1] = {0.25};
          Real single_zero_fraction[1] = {0.0};
          const auto mix1 =
              single_mixture.CompositionFromFractions(single_fraction);
          const auto zero1 =
              single_mixture.CompositionFromFractions(single_zero_fraction);
          const auto state1 = single_mixture.StateFromRhoTemperatures(
              density, 100.0, 100.0, mix1);
          const auto inverse1 = single_mixture.StateFromRhoSpecificEnergies(
              density, state1.ion_specific_internal_energy,
              state1.electron_specific_internal_energy, mix1);
          const auto pure_ch = mixture.StateFromRhoTemperatures(
              density, 100.0, 100.0, 1.0);
          if (mix1.count != 1 || mix1[0] != 1.0 || zero1[0] != 1.0 ||
              !NearlyEqual(state1.ion_specific_internal_energy,
                           pure_ch.ion_specific_internal_energy) ||
              !NearlyEqual(state1.electron_specific_internal_energy,
                           pure_ch.electron_specific_internal_energy) ||
              !NearlyEqual(state1.ion_pressure, pure_ch.ion_pressure) ||
              !NearlyEqual(state1.electron_pressure,
                           pure_ch.electron_pressure) ||
              !NearlyEqual(inverse1.ion_temperature, 100.0) ||
              !NearlyEqual(inverse1.electron_temperature, 100.0)) {
            ++local_failures;
          }

          // Four components exercise a runtime count above the former compile-time cap.
          // The tail components reuse the two tables with distinct species weights so
          // both table and ideal mixing are covered.
          Real many_fractions[4] = {0.1, 0.2, 0.3, 0.4};
          const materials::MaterialComposition mix4 =
              many_mixture.CompositionFromFractions(many_fractions);
          Real pure_fractions[4][4] = {};
          materials::MaterialComposition pure[4];
          for (int n = 0; n < 4; ++n) {
            pure_fractions[n][n] = 1.0;
            pure[n] = many_mixture.CompositionFromFractions(pure_fractions[n]);
          }
          materials::MaterialThermodynamicState component[4];
          for (int n = 0; n < 4; ++n) {
            component[n] = many_mixture.StateFromRhoTemperaturesNoSound(
                density*mix4[n], 100.0, 100.0, pure[n]);
          }
          const auto mixed4 = many_mixture.StateFromRhoTemperatures(
              density, 100.0, 100.0, mix4);
          Real expected_ion_energy = 0.0;
          Real expected_electron_energy = 0.0;
          Real expected_ion_pressure = 0.0;
          Real expected_electron_pressure = 0.0;
          Real expected_electron_density = 0.0;
          int expected_flags = materials::ionmix_query_in_bounds;
          for (int n = 0; n < 4; ++n) {
            expected_ion_energy +=
                mix4[n]*component[n].ion_specific_internal_energy;
            expected_electron_energy +=
                mix4[n]*component[n].electron_specific_internal_energy;
            expected_ion_pressure += component[n].ion_pressure;
            expected_electron_pressure += component[n].electron_pressure;
            expected_electron_density += component[n].electron_number_density_cgs;
            expected_flags |= component[n].query_flags;
          }
          const auto inverse4 = many_mixture.StateFromRhoSpecificEnergies(
              density, mixed4.ion_specific_internal_energy,
              mixed4.electron_specific_internal_energy, mix4);
          const auto electron4 = many_mixture.ElectronStateFromRhoSpecificEnergy(
              density, mixed4.electron_specific_internal_energy, mix4);
          const auto ideal4 = ideal_many_mixture.StateFromRhoTemperaturesNoSound(
              density, 2.5, 1.5, mix4);
          Real overshoot_fractions[4] = {0.8, 0.7, 0.3, 0.2};
          const auto overshoot =
              many_mixture.CompositionFromFractions(overshoot_fractions);
          Real zero_fractions[4] = {};
          const auto all_zero =
              many_mixture.CompositionFromFractions(zero_fractions);
          if (!NearlyEqual(mix4[0], 0.1) ||
              !NearlyEqual(mix4[1], 0.2) ||
              !NearlyEqual(mix4[2], 0.3) ||
              !NearlyEqual(mix4[3], 0.4) ||
              !NearlyEqual(mixed4.ion_specific_internal_energy,
                           expected_ion_energy) ||
              !NearlyEqual(mixed4.electron_specific_internal_energy,
                           expected_electron_energy) ||
              !NearlyEqual(mixed4.ion_pressure, expected_ion_pressure) ||
              !NearlyEqual(mixed4.electron_pressure,
                           expected_electron_pressure) ||
              !NearlyEqual(mixed4.electron_number_density_cgs,
                           expected_electron_density) ||
              mixed4.query_flags != expected_flags ||
              !(mixed4.sound_speed_squared > 0.0) ||
              !NearlyEqual(inverse4.ion_temperature, 100.0) ||
              !NearlyEqual(inverse4.electron_temperature, 100.0) ||
              !NearlyEqual(electron4.electron_temperature, 100.0) ||
              !NearlyEqual(electron4.electron_pressure,
                           mixed4.electron_pressure) ||
              !NearlyEqual(electron4.electron_number_density_cgs,
                           mixed4.electron_number_density_cgs) ||
              !NearlyEqual(ideal_many_mixture.ElectronTemperature(
                               density,
                               ideal4.electron_specific_internal_energy,
                               mix4),
                           1.5) ||
              !NearlyEqual(overshoot[0], 0.4) ||
              !NearlyEqual(overshoot[1], 0.35) ||
              !NearlyEqual(overshoot[2], 0.15) ||
              !NearlyEqual(overshoot[3], 0.1) ||
              all_zero[0] != 0.0 || all_zero[1] != 0.0 ||
              all_zero[2] != 0.0 || all_zero[3] != 1.0) {
            ++local_failures;
          }

          const auto inverse = mixture.StateFromRhoSpecificEnergies(
              density, 250.0, 450.0, ych);
          const auto mixed_no_sound = mixture.StateFromRhoTemperaturesNoSound(
              density, 100.0, 100.0, ych);
          const auto inverse_no_sound = mixture.StateFromRhoSpecificEnergiesNoSound(
              density, 250.0, 450.0, ych);
          const auto inverse_pressure_energy =
              mixture.PressureEnergyFromRhoSpecificEnergies(
                  density, 250.0, 450.0, ych);
          // These are the fields consumed by the tabular dual-energy pressure partition.
          // Omitting sound speed must leave them and their bounds diagnostics unchanged.
          if (!NearlyEqual(inverse.ion_temperature, 100.0) ||
              !NearlyEqual(inverse.electron_temperature, 100.0) ||
              !NearlyEqual(inverse.ion_pressure, 150.0) ||
              !NearlyEqual(inverse.electron_pressure, 350.0) ||
              !NearlyEqual(inverse_no_sound.ion_specific_internal_energy,
                           inverse.ion_specific_internal_energy) ||
              !NearlyEqual(inverse_no_sound.electron_specific_internal_energy,
                           inverse.electron_specific_internal_energy) ||
              !NearlyEqual(inverse_no_sound.ion_pressure,
                           inverse.ion_pressure) ||
              !NearlyEqual(inverse_no_sound.electron_pressure,
                           inverse.electron_pressure) ||
              inverse_pressure_energy.ion_specific_internal_energy !=
                  inverse_no_sound.ion_specific_internal_energy ||
              inverse_pressure_energy.electron_specific_internal_energy !=
                  inverse_no_sound.electron_specific_internal_energy ||
              inverse_pressure_energy.ion_pressure !=
                  inverse_no_sound.ion_pressure ||
              inverse_pressure_energy.electron_pressure !=
                  inverse_no_sound.electron_pressure ||
              inverse_pressure_energy.query_flags != inverse_no_sound.query_flags ||
              inverse_no_sound.query_flags != inverse.query_flags ||
              mixed_no_sound.sound_speed_squared != 0.0 ||
              inverse_no_sound.sound_speed_squared != 0.0 ||
              !(inverse.sound_speed_squared > 0.0)) {
            ++local_failures;
          }

          // The oracle above is the ordinary-forward 48-step mixed inverse. Every state
          // field remains authoritative, including final forward reconstruction metadata.
          if (!ExactPreparedAndUncachedInverseMatch(
                  mixture, density, 250.0, 450.0, ych) ||
              !ExactSingleComponentInverseMatch(
                  mixture, density, 250.0, 450.0, ych)) {
            ++local_failures;
          }

          // Fractions one machine epsilon inside each endpoint remain genuinely mixed.
          // They exercise scalar preparation and below-minimum pressure scaling on
          // opposite materials while sharing the density locations between components.
          const Real inverse_tiny_fraction =
              Kokkos::Experimental::epsilon<Real>::value;
          for (int near_endpoint = 0; near_endpoint < 2; ++near_endpoint) {
            const Real fraction = (near_endpoint == 0)
                ? inverse_tiny_fraction : 1.0-inverse_tiny_fraction;
            const auto seed = mixture.StateFromRhoTemperaturesNoSound(
                density, 50.0, 200.0, fraction);
            const auto prepared_state = mixture.StateFromRhoSpecificEnergiesNoSound(
                density, seed.ion_specific_internal_energy,
                seed.electron_specific_internal_energy, fraction);
            if (!ExactPreparedAndUncachedInverseMatch(
                    mixture, density, seed.ion_specific_internal_energy,
                    seed.electron_specific_internal_energy, fraction) ||
                !ExactSingleComponentInverseMatch(
                    mixture, density, seed.ion_specific_internal_energy,
                    seed.electron_specific_internal_energy, fraction) ||
                (prepared_state.query_flags &
                 materials::ionmix_density_below_table) == 0) {
              ++local_failures;
            }
          }

          // Both partial densities are below their tables. The ordinary and prepared
          // inverses must apply the same minimum-density lookup, pressure scaling, and
          // manual density flag before constructing both public state variants.
          const Real below_minimum_density = 0.25;
          const auto below_minimum_seed = mixture.StateFromRhoTemperaturesNoSound(
              below_minimum_density, 100.0, 200.0, ych);
          const auto below_minimum_prepared =
              mixture.StateFromRhoSpecificEnergiesNoSound(
                  below_minimum_density,
                  below_minimum_seed.ion_specific_internal_energy,
                  below_minimum_seed.electron_specific_internal_energy, ych);
          if (!ExactPreparedAndUncachedInverseMatch(
                  mixture, below_minimum_density,
                  below_minimum_seed.ion_specific_internal_energy,
                  below_minimum_seed.electron_specific_internal_energy, ych) ||
              !ExactSingleComponentInverseMatch(
                  mixture, below_minimum_density,
                  below_minimum_seed.ion_specific_internal_energy,
                  below_minimum_seed.electron_specific_internal_energy, ych) ||
              (below_minimum_prepared.query_flags &
               materials::ionmix_density_below_table) == 0) {
            ++local_failures;
          }

          // Clamp both energy directions on both components. Energy flags from inverse
          // endpoints must survive the same exact full/no-sound forward reconstruction.
          for (int clamp_order = 0; clamp_order < 2; ++clamp_order) {
            const Real ion_energy = (clamp_order == 0) ? 0.0 : 1.0e30;
            const Real electron_energy = (clamp_order == 0) ? 1.0e30 : 0.0;
            const auto prepared_state =
                clamped_mixture.StateFromRhoSpecificEnergiesNoSound(
                    density, ion_energy, electron_energy, ych);
            const int expected_energy_flags =
                materials::ionmix_energy_below_table |
                materials::ionmix_energy_above_table;
            if (!ExactPreparedAndUncachedInverseMatch(
                    clamped_mixture, density, ion_energy, electron_energy, ych) ||
                (prepared_state.query_flags & expected_energy_flags) !=
                    expected_energy_flags) {
              ++local_failures;
            }
          }

          const auto pure_ch_floor = mixture.MinimumState(density, 1.0);
          const auto pure_he_floor = mixture.MinimumState(density, 0.0);
          const auto mixed_floor = mixture.MinimumState(density, ych);
          const auto mixed_floor_no_sound = mixture.MinimumStateNoSound(density, ych);
          const auto mixed_floor_pressure_energy =
              mixture.MinimumPressureEnergyState(density, ych);
          if (!NearlyEqual(pure_ch_floor.ion_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.electron_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.ion_specific_internal_energy, 30.0) ||
              !NearlyEqual(pure_he_floor.ion_temperature, 20.0) ||
              !NearlyEqual(pure_he_floor.ion_specific_internal_energy, 40.0) ||
              !NearlyEqual(mixed_floor.ion_temperature, 20.0) ||
              !NearlyEqual(mixed_floor.ion_specific_internal_energy, 50.0) ||
              !NearlyEqual(mixed_floor.electron_specific_internal_energy, 90.0) ||
              !NearlyEqual(mixed_floor_no_sound.ion_specific_internal_energy,
                           mixed_floor.ion_specific_internal_energy) ||
              !NearlyEqual(mixed_floor_no_sound.electron_specific_internal_energy,
                           mixed_floor.electron_specific_internal_energy) ||
              !NearlyEqual(mixed_floor_no_sound.ion_pressure,
                           mixed_floor.ion_pressure) ||
              !NearlyEqual(mixed_floor_no_sound.electron_pressure,
                           mixed_floor.electron_pressure) ||
              mixed_floor_pressure_energy.ion_specific_internal_energy !=
                  mixed_floor_no_sound.ion_specific_internal_energy ||
              mixed_floor_pressure_energy.electron_specific_internal_energy !=
                  mixed_floor_no_sound.electron_specific_internal_energy ||
              mixed_floor_pressure_energy.ion_pressure !=
                  mixed_floor_no_sound.ion_pressure ||
              mixed_floor_pressure_energy.electron_pressure !=
                  mixed_floor_no_sound.electron_pressure ||
              mixed_floor_pressure_energy.query_flags !=
                  mixed_floor_no_sound.query_flags ||
              mixed_floor_no_sound.query_flags != mixed_floor.query_flags ||
              mixed_floor_no_sound.sound_speed_squared != 0.0 ||
              !(mixed_floor.sound_speed_squared > 0.0)) {
            ++local_failures;
          }

          // The floor-only native-minimum path must exactly reproduce the generic full
          // state for pure, epsilon-endpoint, and mixed compositions.
          const Real floor_fractions[] = {
              0.0, 1.0, inverse_tiny_fraction,
              1.0-inverse_tiny_fraction, ych};
          for (int ifraction = 0; ifraction < 5; ++ifraction) {
            const auto full_floor = mixture.MinimumStateNoSound(
                density, floor_fractions[ifraction]);
            const auto reduced_floor = mixture.MinimumPressureEnergyState(
                density, floor_fractions[ifraction]);
            if (!ExactPressureEnergyMatch(reduced_floor, full_floor)) {
              ++local_failures;
            }
          }

          // Common-temperature floor queries must remain exact for fractions one
          // machine epsilon inside each endpoint and when both partial densities are
          // below their respective tables.
          for (int near_endpoint = 0; near_endpoint < 2; ++near_endpoint) {
            const Real fraction = (near_endpoint == 0)
                ? inverse_tiny_fraction : 1.0-inverse_tiny_fraction;
            const auto full_floor = mixture.MinimumStateNoSound(density, fraction);
            const auto reduced_floor =
                mixture.MinimumPressureEnergyState(density, fraction);
            if (!ExactPressureEnergyMatch(reduced_floor, full_floor) ||
                (reduced_floor.query_flags &
                 materials::ionmix_density_below_table) == 0) {
              ++local_failures;
            }
          }
          const auto below_minimum_floor = mixture.MinimumStateNoSound(
              below_minimum_density, ych);
          const auto below_minimum_floor_pressure_energy =
              mixture.MinimumPressureEnergyState(below_minimum_density, ych);
          if (!ExactPressureEnergyMatch(
                  below_minimum_floor_pressure_energy, below_minimum_floor) ||
              (below_minimum_floor_pressure_energy.query_flags &
               materials::ionmix_density_below_table) == 0) {
            ++local_failures;
          }

          const auto pressure_floor_state = mixture.MinimumStateNoSound(
              density, ych, 750.0, 0.0);
          const auto pressure_floor_pressure_energy =
              mixture.MinimumPressureEnergyState(density, ych, 750.0, 0.0);
          const auto temperature_floor_state = mixture.MinimumStateNoSound(
              density, ych, 0.0, 100.0);
          const auto temperature_floor_pressure_energy =
              mixture.MinimumPressureEnergyState(density, ych, 0.0, 100.0);
          const auto above_range_floor_state = clamped_mixture.MinimumStateNoSound(
              density, ych, 1.0e30, 0.0);
          const auto above_range_floor_pressure_energy =
              clamped_mixture.MinimumPressureEnergyState(
                  density, ych, 1.0e30, 0.0);
          if (pressure_floor_pressure_energy.ion_pressure !=
                  pressure_floor_state.ion_pressure ||
              pressure_floor_pressure_energy.electron_pressure !=
                  pressure_floor_state.electron_pressure ||
              pressure_floor_pressure_energy.ion_specific_internal_energy !=
                  pressure_floor_state.ion_specific_internal_energy ||
              pressure_floor_pressure_energy.electron_specific_internal_energy !=
                  pressure_floor_state.electron_specific_internal_energy ||
              pressure_floor_pressure_energy.query_flags !=
                  pressure_floor_state.query_flags ||
              temperature_floor_pressure_energy.ion_pressure !=
                  temperature_floor_state.ion_pressure ||
              temperature_floor_pressure_energy.electron_pressure !=
                  temperature_floor_state.electron_pressure ||
              temperature_floor_pressure_energy.ion_specific_internal_energy !=
                  temperature_floor_state.ion_specific_internal_energy ||
              temperature_floor_pressure_energy.electron_specific_internal_energy !=
                  temperature_floor_state.electron_specific_internal_energy ||
              temperature_floor_pressure_energy.query_flags !=
                  temperature_floor_state.query_flags ||
              above_range_floor_pressure_energy.ion_pressure !=
                  above_range_floor_state.ion_pressure ||
              above_range_floor_pressure_energy.electron_pressure !=
                  above_range_floor_state.electron_pressure ||
              above_range_floor_pressure_energy.ion_specific_internal_energy !=
                  above_range_floor_state.ion_specific_internal_energy ||
              above_range_floor_pressure_energy.electron_specific_internal_energy !=
                  above_range_floor_state.electron_specific_internal_energy ||
              above_range_floor_pressure_energy.query_flags !=
                  above_range_floor_state.query_flags ||
              (above_range_floor_pressure_energy.query_flags &
               materials::ionmix_temperature_above_table) == 0) {
            ++local_failures;
          }

          const auto ideal_state = ideal_mixture.StateFromRhoSpecificEnergiesNoSound(
              density, 2.5, 1.5, ych);
          const auto ideal_pressure_energy =
              ideal_mixture.PressureEnergyFromRhoSpecificEnergies(
                  density, 2.5, 1.5, ych);
          const auto ideal_floor = ideal_mixture.MinimumStateNoSound(
              density, ych, 3.0, 2.0);
          const auto ideal_floor_pressure_energy =
              ideal_mixture.MinimumPressureEnergyState(
                  density, ych, 3.0, 2.0);
          if (ideal_pressure_energy.ion_pressure != ideal_state.ion_pressure ||
              ideal_pressure_energy.electron_pressure !=
                  ideal_state.electron_pressure ||
              ideal_pressure_energy.ion_specific_internal_energy !=
                  ideal_state.ion_specific_internal_energy ||
              ideal_pressure_energy.electron_specific_internal_energy !=
                  ideal_state.electron_specific_internal_energy ||
              ideal_pressure_energy.query_flags != ideal_state.query_flags ||
              ideal_floor_pressure_energy.ion_pressure != ideal_floor.ion_pressure ||
              ideal_floor_pressure_energy.electron_pressure !=
                  ideal_floor.electron_pressure ||
              ideal_floor_pressure_energy.ion_specific_internal_energy !=
                  ideal_floor.ion_specific_internal_energy ||
              ideal_floor_pressure_energy.electron_specific_internal_energy !=
                  ideal_floor.electron_specific_internal_energy ||
              ideal_floor_pressure_energy.query_flags != ideal_floor.query_flags) {
            ++local_failures;
          }

          // Under clamp policy, the reduced dual-energy state must preserve endpoint
          // behavior, including pressure extrapolation below a pure table's density range.
          for (int endpoint = 0; endpoint < 2; ++endpoint) {
            const Real fraction = (endpoint == 0) ? 0.0 : 1.0;
            const Real low_density = 0.25;
            const auto endpoint_state =
                clamped_mixture.StateFromRhoSpecificEnergiesNoSound(
                    low_density, 40.0, 80.0, fraction);
            const auto endpoint_pressure_energy =
                clamped_mixture.PressureEnergyFromRhoSpecificEnergies(
                    low_density, 40.0, 80.0, fraction);
            const auto endpoint_floor = clamped_mixture.MinimumStateNoSound(
                low_density, fraction);
            const auto endpoint_floor_pressure_energy =
                clamped_mixture.MinimumPressureEnergyState(low_density, fraction);
            if (endpoint_pressure_energy.ion_pressure !=
                    endpoint_state.ion_pressure ||
                endpoint_pressure_energy.electron_pressure !=
                    endpoint_state.electron_pressure ||
                endpoint_pressure_energy.ion_specific_internal_energy !=
                    endpoint_state.ion_specific_internal_energy ||
                endpoint_pressure_energy.electron_specific_internal_energy !=
                    endpoint_state.electron_specific_internal_energy ||
                endpoint_pressure_energy.query_flags != endpoint_state.query_flags ||
                endpoint_floor_pressure_energy.ion_pressure !=
                    endpoint_floor.ion_pressure ||
                endpoint_floor_pressure_energy.electron_pressure !=
                    endpoint_floor.electron_pressure ||
                endpoint_floor_pressure_energy.ion_specific_internal_energy !=
                    endpoint_floor.ion_specific_internal_energy ||
                endpoint_floor_pressure_energy.electron_specific_internal_energy !=
                    endpoint_floor.electron_specific_internal_energy ||
                endpoint_floor_pressure_energy.query_flags != endpoint_floor.query_flags) {
              ++local_failures;
            }
          }

          const auto pure_ch_low = mixture.StateFromRhoTemperatures(
              density, 10.0, 10.0, 1.0);
          if (!NearlyEqual(pure_ch_low.effective_charge, 0.6)) {
            ++local_failures;
          }
          const auto pure_ch_full = mixture.StateFromRhoTemperatures(
              density, 100.0, 100.0, 1.0);
          if (!NearlyEqual(pure_ch_full.effective_charge, 6.0)) {
            ++local_failures;
          }

          const auto old_exchange_state = mixture.StateFromRhoTemperatures(
              density, 50.0, 200.0, ych);
          const Real exchange_total =
              old_exchange_state.ion_specific_internal_energy+
              old_exchange_state.electron_specific_internal_energy;
          const Real target_difference = 75.0;
          const auto exchange =
              mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density,
                  old_exchange_state.ion_specific_internal_energy,
                  old_exchange_state.electron_specific_internal_energy,
                  old_exchange_state.ion_temperature,
                  old_exchange_state.electron_temperature,
                  target_difference, ych);
          const auto reduced_exchange =
              mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                  density,
                  old_exchange_state.ion_specific_internal_energy,
                  old_exchange_state.electron_specific_internal_energy,
                  old_exchange_state.ion_temperature,
                  old_exchange_state.electron_temperature,
                  target_difference, ych);
          const auto uncached_exchange =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  mixture, density,
                  old_exchange_state.ion_specific_internal_energy,
                  old_exchange_state.electron_specific_internal_energy,
                  old_exchange_state.ion_temperature,
                  old_exchange_state.electron_temperature,
                  target_difference, ych);
          const Real expected_tion =
              (exchange_total-4.5*target_difference)/7.0;
          if (!ExactMaterialExchangeMatch(exchange, uncached_exchange) ||
              !ExactReducedExchangeMatch(uncached_exchange, reduced_exchange) ||
              !NearlyEqual(exchange.thermodynamics.ion_temperature,
                           expected_tion) ||
              !NearlyEqual(exchange.thermodynamics.electron_temperature,
                           expected_tion+target_difference) ||
              !NearlyEqual(exchange.ion_specific_internal_energy+
                           exchange.electron_specific_internal_energy,
                           exchange_total) ||
              fabs(exchange.energy_residual) > 1.0e-9*exchange_total ||
              fabs(exchange.temperature_difference_residual) > 1.0e-10 ||
              exchange.iterations <= 0 || exchange.iterations > 6 ||
              exchange.used_fallback != 0) {
            ++local_failures;
          }

          // Repeat a zero-width, already-converged mixed Exchange with density-dependent
          // component energies. A doubled-density token is an explicit negative control:
          // it changes each endpoint energy, turning this zero-iteration success into
          // fallback 2, so the exact oracle comparison cannot accept a wrong token.
          const Real sensitive_density = 3.0;
          const Real sensitive_fraction = 0.4;
          const auto sensitive_old =
              density_dependent_mixture.StateFromRhoTemperaturesNoSound(
                  sensitive_density, 50.0, 200.0, sensitive_fraction);
          const Real sensitive_target =
              sensitive_old.electron_temperature-
              sensitive_old.ion_temperature;
          const auto sensitive_exchange =
              density_dependent_mixture.
                  StateFromRhoTotalEnergyTemperatureDifference(
                      sensitive_density,
                      sensitive_old.ion_specific_internal_energy,
                      sensitive_old.electron_specific_internal_energy,
                      sensitive_old.ion_temperature,
                      sensitive_old.electron_temperature,
                      sensitive_target, sensitive_fraction);
          const auto reduced_sensitive_exchange =
              density_dependent_mixture.
                  StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                      sensitive_density,
                      sensitive_old.ion_specific_internal_energy,
                      sensitive_old.electron_specific_internal_energy,
                      sensitive_old.ion_temperature,
                      sensitive_old.electron_temperature,
                      sensitive_target, sensitive_fraction);
          const auto uncached_sensitive_exchange =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  density_dependent_mixture, sensitive_density,
                  sensitive_old.ion_specific_internal_energy,
                  sensitive_old.electron_specific_internal_energy,
                  sensitive_old.ion_temperature,
                  sensitive_old.electron_temperature,
                  sensitive_target, sensitive_fraction);
          const Real sensitive_density0 =
              sensitive_density*sensitive_fraction;
          const Real sensitive_density1 =
              sensitive_density*(1.0-sensitive_fraction);
          const auto wrong_location0 =
              density_dependent_mixture.SpeciesTable(0).
                  PrepareDensityLocation(2.0*sensitive_density0);
          const auto wrong_location1 =
              density_dependent_mixture.SpeciesTable(1).
                  PrepareDensityLocation(2.0*sensitive_density1);
          const auto wrong_ion0 =
              density_dependent_mixture.SpeciesTable(0).
                  ComponentFromPreparedDensityTemperature(
                      materials::IonmixComponent::ion, wrong_location0,
                      sensitive_old.ion_temperature);
          const auto wrong_electron0 =
              density_dependent_mixture.SpeciesTable(0).
                  ComponentFromPreparedDensityTemperature(
                      materials::IonmixComponent::electron, wrong_location0,
                      sensitive_old.electron_temperature);
          const auto wrong_ion1 =
              density_dependent_mixture.SpeciesTable(1).
                  ComponentFromPreparedDensityTemperature(
                      materials::IonmixComponent::ion, wrong_location1,
                      sensitive_old.ion_temperature);
          const auto wrong_electron1 =
              density_dependent_mixture.SpeciesTable(1).
                  ComponentFromPreparedDensityTemperature(
                      materials::IonmixComponent::electron, wrong_location1,
                      sensitive_old.electron_temperature);
          const Real sensitive_total =
              sensitive_old.ion_specific_internal_energy+
              sensitive_old.electron_specific_internal_energy;
          const Real wrong_endpoint_energy =
              sensitive_fraction*(wrong_ion0.specific_internal_energy+
                                  wrong_electron0.specific_internal_energy)+
              (1.0-sensitive_fraction)*(
                  wrong_ion1.specific_internal_energy+
                  wrong_electron1.specific_internal_energy);
          const Real wrong_endpoint_residual =
              wrong_endpoint_energy-sensitive_total;
          const Real wrong_endpoint_scale = fmax(
              fabs(sensitive_total), fabs(wrong_endpoint_energy));
          if (!ExactMaterialExchangeMatch(
                  sensitive_exchange, uncached_sensitive_exchange) ||
              !ExactReducedExchangeMatch(
                  uncached_sensitive_exchange, reduced_sensitive_exchange) ||
              sensitive_exchange.used_fallback != 0 ||
              sensitive_exchange.iterations != 0 ||
              fabs(wrong_endpoint_residual) <=
                  1.0e-6*fmax(wrong_endpoint_scale, 1.0)) {
            ++local_failures;
          }

          // Pure-table endpoints take different native temperature grids and inverse
          // paths. Both must still publish the same canonical temperatures as the full
          // mixed-state reconstruction.
          for (int endpoint = 0; endpoint < 2; ++endpoint) {
            const Real fraction = (endpoint == 0) ? 0.0 : 1.0;
            const auto endpoint_old = mixture.StateFromRhoTemperaturesNoSound(
                density, 50.0, 200.0, fraction);
            const Real endpoint_target = 0.5*(
                endpoint_old.electron_temperature-endpoint_old.ion_temperature);
            const auto endpoint_exchange =
                mixture.StateFromRhoTotalEnergyTemperatureDifference(
                    density,
                    endpoint_old.ion_specific_internal_energy,
                    endpoint_old.electron_specific_internal_energy,
                    endpoint_old.ion_temperature,
                    endpoint_old.electron_temperature,
                    endpoint_target, fraction);
            const auto reduced_endpoint_exchange =
                mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                    density,
                    endpoint_old.ion_specific_internal_energy,
                    endpoint_old.electron_specific_internal_energy,
                    endpoint_old.ion_temperature,
                    endpoint_old.electron_temperature,
                    endpoint_target, fraction);
            const auto uncached_endpoint_exchange =
                UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                    mixture, density,
                    endpoint_old.ion_specific_internal_energy,
                    endpoint_old.electron_specific_internal_energy,
                    endpoint_old.ion_temperature,
                    endpoint_old.electron_temperature,
                    endpoint_target, fraction);
            if (!ExactMaterialExchangeMatch(
                    endpoint_exchange, uncached_endpoint_exchange) ||
                !ExactReducedExchangeMatch(
                    uncached_endpoint_exchange, reduced_endpoint_exchange) ||
                endpoint_exchange.used_fallback != 0) {
              ++local_failures;
            }
          }

          // The He-only temperatures are above the absent CH table's upper bound.
          // Under error bounds, completing this exchange proves that pure compositions
          // query only their active table, including the reduced final canonicalization.
          const Real active_only_fraction = 0.0;
          const auto active_only_old = mixture.StateFromRhoTemperaturesNoSound(
              density, 2000.0, 4000.0, active_only_fraction);
          const Real active_only_target = 0.5*(
              active_only_old.electron_temperature-
              active_only_old.ion_temperature);
          const auto active_only_exchange =
              mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density,
                  active_only_old.ion_specific_internal_energy,
                  active_only_old.electron_specific_internal_energy,
                  active_only_old.ion_temperature,
                  active_only_old.electron_temperature,
                  active_only_target, active_only_fraction);
          const auto reduced_active_only_exchange =
              mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                  density,
                  active_only_old.ion_specific_internal_energy,
                  active_only_old.electron_specific_internal_energy,
                  active_only_old.ion_temperature,
                  active_only_old.electron_temperature,
                  active_only_target, active_only_fraction);
          const auto uncached_active_only_exchange =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  mixture, density,
                  active_only_old.ion_specific_internal_energy,
                  active_only_old.electron_specific_internal_energy,
                  active_only_old.ion_temperature,
                  active_only_old.electron_temperature,
                  active_only_target, active_only_fraction);
          if (!(active_only_old.ion_temperature >
                    mixture.SpeciesTable(0).MaximumTemperatureCode()) ||
              active_only_old.query_flags != materials::ionmix_query_in_bounds ||
              !ExactMaterialExchangeMatch(
                  active_only_exchange, uncached_active_only_exchange) ||
              !ExactReducedExchangeMatch(
                  uncached_active_only_exchange, reduced_active_only_exchange) ||
              active_only_exchange.used_fallback != 0 ||
              active_only_exchange.thermodynamics.query_flags !=
                  materials::ionmix_query_in_bounds) {
            ++local_failures;
          }

          // Machine-epsilon fractions inside each pure endpoint still take both
          // species branches. The reduced temperature-only reconstruction must retain
          // the exact canonical result of the full mixed-state reconstruction.
          const Real tiny_fraction =
              Kokkos::Experimental::epsilon<Real>::value;
          for (int near_endpoint = 0; near_endpoint < 2; ++near_endpoint) {
            const Real fraction = (near_endpoint == 0)
                ? tiny_fraction : 1.0-tiny_fraction;
            const auto tiny_old = mixture.StateFromRhoTemperaturesNoSound(
                density, 50.0, 200.0, fraction);
            const Real tiny_target = 0.5*(
                tiny_old.electron_temperature-tiny_old.ion_temperature);
            const auto tiny_exchange =
                mixture.StateFromRhoTotalEnergyTemperatureDifference(
                    density,
                    tiny_old.ion_specific_internal_energy,
                    tiny_old.electron_specific_internal_energy,
                    tiny_old.ion_temperature,
                    tiny_old.electron_temperature,
                    tiny_target, fraction);
            const auto reduced_tiny_exchange =
                mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                    density,
                    tiny_old.ion_specific_internal_energy,
                    tiny_old.electron_specific_internal_energy,
                    tiny_old.ion_temperature,
                    tiny_old.electron_temperature,
                    tiny_target, fraction);
            const auto uncached_tiny_exchange =
                UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                    mixture, density,
                    tiny_old.ion_specific_internal_energy,
                    tiny_old.electron_specific_internal_energy,
                    tiny_old.ion_temperature,
                    tiny_old.electron_temperature,
                    tiny_target, fraction);
            if (!ExactMaterialExchangeMatch(
                    tiny_exchange, uncached_tiny_exchange) ||
                !ExactReducedExchangeMatch(
                    uncached_tiny_exchange, reduced_tiny_exchange) ||
                tiny_exchange.used_fallback != 0) {
              ++local_failures;
            }
          }

          // Exercise the final canonical-temperature query after both a forward old
          // state and an inverse reconstruction of that state. This catches ulp-level
          // differences that appear only on a second table-coordinate round trip.
          const auto forward_round_trip_old =
              mixture.StateFromRhoTemperaturesNoSound(
                  density, 50.0, 200.0, ych);
          const auto inverse_round_trip_old =
              mixture.StateFromRhoSpecificEnergiesNoSound(
                  density,
                  forward_round_trip_old.ion_specific_internal_energy,
                  forward_round_trip_old.electron_specific_internal_energy,
                  ych);
          for (int origin = 0; origin < 2; ++origin) {
            const materials::MaterialThermodynamicState canonical_old =
                (origin == 0) ? forward_round_trip_old : inverse_round_trip_old;
            const Real round_trip_target = 0.5*(
                canonical_old.electron_temperature-
                canonical_old.ion_temperature);
            const auto round_trip_exchange =
                mixture.StateFromRhoTotalEnergyTemperatureDifference(
                    density,
                    canonical_old.ion_specific_internal_energy,
                    canonical_old.electron_specific_internal_energy,
                    canonical_old.ion_temperature,
                    canonical_old.electron_temperature,
                    round_trip_target, ych);
            const auto reduced_round_trip_exchange =
                mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                    density,
                    canonical_old.ion_specific_internal_energy,
                    canonical_old.electron_specific_internal_energy,
                    canonical_old.ion_temperature,
                    canonical_old.electron_temperature,
                    round_trip_target, ych);
            const auto uncached_round_trip_exchange =
                UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                    mixture, density,
                    canonical_old.ion_specific_internal_energy,
                    canonical_old.electron_specific_internal_energy,
                    canonical_old.ion_temperature,
                    canonical_old.electron_temperature,
                    round_trip_target, ych);
            if (!ExactMaterialExchangeMatch(
                    round_trip_exchange, uncached_round_trip_exchange) ||
                !ExactReducedExchangeMatch(
                    uncached_round_trip_exchange, reduced_round_trip_exchange) ||
                round_trip_exchange.used_fallback != 0) {
              ++local_failures;
            }
          }

          // Start on both clamp-policy temperature bounds, then relax toward an
          // interior temperature difference. This exercises canonicalization of
          // bounded component inputs without conflating it with failed-bracket recovery.
          const Real bounded_old_tion = 19.0;
          const Real bounded_old_tele = 1001.0;
          const auto bounded_old = clamped_mixture.StateFromRhoTemperaturesNoSound(
              density, bounded_old_tion, bounded_old_tele, ych);
          const Real bounded_target =
              0.5*(bounded_old_tele-bounded_old_tion);
          const auto bounded_exchange =
              clamped_mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density,
                  bounded_old.ion_specific_internal_energy,
                  bounded_old.electron_specific_internal_energy,
                  bounded_old_tion,
                  bounded_old_tele,
                  bounded_target, ych);
          const auto reduced_bounded_exchange =
              clamped_mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                  density,
                  bounded_old.ion_specific_internal_energy,
                  bounded_old.electron_specific_internal_energy,
                  bounded_old_tion,
                  bounded_old_tele,
                  bounded_target, ych);
          const auto uncached_bounded_exchange =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  clamped_mixture, density,
                  bounded_old.ion_specific_internal_energy,
                  bounded_old.electron_specific_internal_energy,
                  bounded_old_tion, bounded_old_tele, bounded_target, ych);
          if (!ExactMaterialExchangeMatch(
                  bounded_exchange, uncached_bounded_exchange) ||
              !ExactReducedExchangeMatch(
                  uncached_bounded_exchange, reduced_bounded_exchange) ||
              bounded_exchange.used_fallback != 0 ||
              (bounded_exchange.thermodynamics.query_flags &
               materials::ionmix_temperature_below_table) == 0 ||
              (bounded_exchange.thermodynamics.query_flags &
               materials::ionmix_temperature_above_table) == 0) {
            ++local_failures;
          }

          // The ideal branch is not used by the tabular Exchange caller, but its
          // reduced/full implementations share the public API and must agree on every
          // field that the reduced result promises to populate.
          const auto ideal_exchange_old =
              ideal_mixture.StateFromRhoTemperaturesNoSound(
                  density, 0.75, 1.5, ych);
          const Real ideal_target = 0.25;
          const auto ideal_exchange =
              ideal_mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density,
                  ideal_exchange_old.ion_specific_internal_energy,
                  ideal_exchange_old.electron_specific_internal_energy,
                  ideal_exchange_old.ion_temperature,
                  ideal_exchange_old.electron_temperature,
                  ideal_target, ych);
          const auto reduced_ideal_exchange =
              ideal_mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                  density,
                  ideal_exchange_old.ion_specific_internal_energy,
                  ideal_exchange_old.electron_specific_internal_energy,
                  ideal_exchange_old.ion_temperature,
                  ideal_exchange_old.electron_temperature,
                  ideal_target, ych);
          const auto uncached_ideal_exchange =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  ideal_mixture, density,
                  ideal_exchange_old.ion_specific_internal_energy,
                  ideal_exchange_old.electron_specific_internal_energy,
                  ideal_exchange_old.ion_temperature,
                  ideal_exchange_old.electron_temperature,
                  ideal_target, ych);
          if (!ExactMaterialExchangeMatch(
                  ideal_exchange, uncached_ideal_exchange) ||
              !ExactReducedExchangeMatch(
                  uncached_ideal_exchange, reduced_ideal_exchange) ||
              ideal_exchange.used_fallback != 0 ||
              ideal_exchange.iterations != 0) {
            ++local_failures;
          }

          // A zero-width bracket with a nonzero residual is not convergence. Recovery
          // must decline the exchange and preserve the exact conservative split passed
          // by the caller, independent of the stale cached temperatures.
          const Real recovery_ion_energy = 125.0;
          const Real recovery_electron_energy = 900.0;
          const auto recovery =
              clamped_mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density, recovery_ion_energy, recovery_electron_energy,
                  2000.0, 2000.0, 0.0, ych);
          const auto reduced_recovery =
              clamped_mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                  density, recovery_ion_energy, recovery_electron_energy,
                  2000.0, 2000.0, 0.0, ych);
          const auto uncached_recovery =
              UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                  clamped_mixture, density,
                  recovery_ion_energy, recovery_electron_energy,
                  2000.0, 2000.0, 0.0, ych);
          if (!ExactMaterialExchangeMatch(recovery, uncached_recovery) ||
              !ExactReducedExchangeMatch(uncached_recovery, reduced_recovery) ||
              recovery.used_fallback != 2 || recovery.iterations != 0 ||
              recovery.ion_specific_internal_energy != recovery_ion_energy ||
              recovery.electron_specific_internal_energy != recovery_electron_energy ||
              recovery.thermodynamics.sound_speed_squared != 0.0 ||
              (recovery.thermodynamics.query_flags &
               materials::ionmix_temperature_above_table) == 0) {
            ++local_failures;
          }

          // Linear interpolation of the fixture values on logarithmic temperature axes
          // supplies a deliberately nonlinear energy curve. Find and validate a bracket
          // that exhausts the six safeguarded secant steps and converges by bisection.
          bool exercised_bisection = false;
          for (int ion_index = 0; ion_index < 6 && !exercised_bisection; ++ion_index) {
            const Real old_tion = 25.0*exp(0.45*ion_index);
            for (int electron_index = 0;
                 electron_index < 6 && !exercised_bisection; ++electron_index) {
              const Real old_tele = 70.0*exp(0.45*electron_index);
              if (!(old_tele > old_tion) || old_tele > 950.0) continue;
              const auto nonlinear_old =
                  nonlinear_mixture.StateFromRhoTemperaturesNoSound(
                      density, old_tion, old_tele, ych);
              for (int decay_index = 1;
                   decay_index < 10 && !exercised_bisection; ++decay_index) {
                const Real nonlinear_target =
                    0.1*decay_index*(old_tele-old_tion);
                const auto nonlinear_exchange =
                    nonlinear_mixture.StateFromRhoTotalEnergyTemperatureDifference(
                        density,
                        nonlinear_old.ion_specific_internal_energy,
                        nonlinear_old.electron_specific_internal_energy,
                        nonlinear_old.ion_temperature,
                        nonlinear_old.electron_temperature,
                        nonlinear_target, ych);
                if (nonlinear_exchange.used_fallback == 1) {
                  const auto uncached_nonlinear_exchange =
                      UncachedExchangeStateFromRhoTotalEnergyTemperatureDifference(
                          nonlinear_mixture, density,
                          nonlinear_old.ion_specific_internal_energy,
                          nonlinear_old.electron_specific_internal_energy,
                          nonlinear_old.ion_temperature,
                          nonlinear_old.electron_temperature,
                          nonlinear_target, ych);
                  const auto reduced_nonlinear_exchange =
                      nonlinear_mixture.
                          StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                              density,
                              nonlinear_old.ion_specific_internal_energy,
                              nonlinear_old.electron_specific_internal_energy,
                              nonlinear_old.ion_temperature,
                              nonlinear_old.electron_temperature,
                              nonlinear_target, ych);
                  const Real nonlinear_total =
                      nonlinear_old.ion_specific_internal_energy+
                      nonlinear_old.electron_specific_internal_energy;
                  if (!ExactMaterialExchangeMatch(
                          nonlinear_exchange, uncached_nonlinear_exchange) ||
                      !ExactReducedExchangeMatch(
                          uncached_nonlinear_exchange,
                          reduced_nonlinear_exchange) ||
                      nonlinear_exchange.iterations <= 6 ||
                      nonlinear_exchange.iterations > 54 ||
                      !NearlyEqual(
                          nonlinear_exchange.ion_specific_internal_energy+
                          nonlinear_exchange.electron_specific_internal_energy,
                          nonlinear_total, 3.0e-12) ||
                      nonlinear_exchange.thermodynamics.sound_speed_squared != 0.0) {
                    ++local_failures;
                  }
                  exercised_bisection = true;
                }
              }
            }
          }
          if (!exercised_bisection) ++local_failures;

          MHDPrim1D left;
          left.d = density;
          left.vx = left.vy = left.vz = 0.0;
          left.e = density*(250.0+450.0);
          left.by = left.bz = 0.0;
          const MHDPrim1D right = left;
          MHDCons1D flux;
          mhd::SingleStateLLF_MHDMaterial(
              left, right, 0.0,
              mixed.ion_pressure+mixed.electron_pressure,
              mixed.ion_pressure+mixed.electron_pressure,
              mixed.sound_speed_squared, mixed.sound_speed_squared, flux);
          if (!NearlyEqual(flux.d, 0.0) ||
              !NearlyEqual(flux.mx, 500.0) ||
              !NearlyEqual(flux.e, 0.0)) {
            ++local_failures;
          }
        }, Kokkos::Sum<int>(failures));
    if (failures != 0) {
      std::cerr << failures << " tabular material checks failed\n";
      return_code = EXIT_FAILURE;
    }
  }
  Kokkos::finalize();
  return return_code;
}
