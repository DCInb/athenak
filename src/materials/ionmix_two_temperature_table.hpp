#ifndef MATERIALS_IONMIX_TWO_TEMPERATURE_TABLE_HPP_
#define MATERIALS_IONMIX_TWO_TEMPERATURE_TABLE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ionmix_two_temperature_table.hpp
//! \brief Device-copyable separate ion/electron IONMIX EOS lookup table.

#include <cstdint>
#include <string>

#include "athena.hpp"

namespace materials {

enum class IonmixBoundsPolicy : int {
  clamp = 0,
  error = 1
};

enum class IonmixComponent : int {
  ion = 0,
  electron = 1
};

enum IonmixQueryFlag : int {
  ionmix_query_in_bounds = 0,
  ionmix_density_below_table = 1 << 0,
  ionmix_density_above_table = 1 << 1,
  ionmix_temperature_below_table = 1 << 2,
  ionmix_temperature_above_table = 1 << 3,
  ionmix_energy_below_table = 1 << 4,
  ionmix_energy_above_table = 1 << 5
};

//----------------------------------------------------------------------------------------
//! \struct IonmixTwoTemperatureTableOptions
//! \brief Host-side code-unit conversion and interpolation policy.
//!
//! The native file is always CGS: density in g/cm^3, temperature in K, pressure in
//! erg/cm^3, and specific energy in erg/g.  The first two factors convert a code query
//! into the file units; the latter two convert a file value back into code units.

struct IonmixTwoTemperatureTableOptions {
  IonmixBoundsPolicy bounds_policy = IonmixBoundsPolicy::clamp;
  bool geometric_interpolation = true;
  Real density_to_cgs = 1.0;
  Real temperature_to_kelvin = 1.0;
  Real pressure_from_cgs = 1.0;
  Real specific_energy_from_cgs = 1.0;
};

//----------------------------------------------------------------------------------------
//! \struct IonmixTwoTemperatureTableMetadata
//! \brief Host-only identity and physical-domain information for restart validation.

struct IonmixTwoTemperatureTableMetadata {
  std::string source_file;
  std::string file_fingerprint;
  std::uint64_t file_fingerprint_value = 0;
  std::uint64_t file_size = 0;
  int format_version = 0;
  int ndensity = 0;
  int ntemperature = 0;
  Real abar = 0.0;
  Real minimum_density_cgs = 0.0;
  Real maximum_density_cgs = 0.0;
  Real minimum_temperature_kelvin = 0.0;
  Real maximum_temperature_kelvin = 0.0;
  bool ion_energy_is_strictly_positive = false;
  bool electron_energy_is_strictly_positive = false;
  bool pressure_interpolation_is_safely_finite = false;
};

struct IonmixComponentState {
  Real temperature = 0.0;
  Real pressure = 0.0;
  Real specific_internal_energy = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Both component pressure/energy pairs at one density and temperature. This reduced
// state omits the canonical temperature when the caller does not consume it.
struct IonmixPressureEnergyState {
  Real ion_pressure = 0.0;
  Real electron_pressure = 0.0;
  Real ion_specific_internal_energy = 0.0;
  Real electron_specific_internal_energy = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Canonical temperature and bounds diagnostics without field interpolation.
struct IonmixTemperatureState {
  Real temperature = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

// Density interpolation state that can be reused across temperature probes. The token
// is valid only for the table device and unit conversion that prepared it.
struct IonmixDensityLocation {
  Real fraction = 0.0;
  int lower = 0;
  int query_flags = ionmix_query_in_bounds;
};

// Density-interpolated endpoint energies for one native temperature interval. The
// owning inverse query keeps one cache per material because their temperature grids
// need not be identical.
struct IonmixEnergyIntervalCache {
  Real lower_energy = 0.0;
  Real upper_energy = 0.0;
  Real log_lower_energy = 0.0;
  Real log_upper_energy = 0.0;
  int temperature_lower = -1;
};

struct IonmixTwoTemperatureState {
  IonmixComponentState ion;
  IonmixComponentState electron;
  Real mean_ionization = 0.0;
  int query_flags = ionmix_query_in_bounds;
};

//----------------------------------------------------------------------------------------
//! \struct IonmixTwoTemperatureTableDevice
//! \brief Lightweight device view supporting forward and inverse component queries.

struct IonmixTwoTemperatureTableDevice {
  enum Field : int {
    ion_pressure = 0,
    electron_pressure = 1,
    ion_specific_internal_energy = 2,
    electron_specific_internal_energy = 3,
    mean_ionization = 4
  };
  static constexpr int nfields = 5;

  DvceArray1D<Real> log_density_cgs;
  DvceArray1D<Real> log_temperature_kelvin;
  DvceArray3D<Real> values;
  DvceArray3D<Real> log_values;
  int ndensity = 0;
  int ntemperature = 0;
  int bounds_error = 0;
  bool geometric_interpolation = true;
  bool ion_energy_is_strictly_positive = false;
  bool electron_energy_is_strictly_positive = false;
  bool pressure_interpolation_is_safely_finite = false;
  int minimum_temperature_round_trips_exactly = 0;
  Real abar = 0.0;
  Real density_to_cgs = 1.0;
  Real temperature_to_kelvin = 1.0;
  Real pressure_from_cgs = 1.0;
  Real specific_energy_from_cgs = 1.0;
  Real log_density_to_cgs = 0.0;
  Real log_temperature_to_kelvin = 0.0;
  Real minimum_density_code = 0.0;
  Real maximum_density_code = 0.0;
  Real minimum_temperature_code = 0.0;
  Real maximum_temperature_code = 0.0;

 private:
  struct AxisLocation {
    int lower = 0;
    Real fraction = 0.0;
    Real bounded_log_coordinate = 0.0;
    int query_flags = ionmix_query_in_bounds;
  };

  template <bool log_coordinate_is_precomputed = false>
  KOKKOS_INLINE_FUNCTION
  AxisLocation Locate(const DvceArray1D<Real> &axis, const int size,
                      const Real code_coordinate, const Real log_unit_scale,
                      const int low_flag, const int high_flag,
                      const int lower_hint = -1,
                      const Real precomputed_log_coordinate = 0.0) const {
    if (!Kokkos::isfinite(code_coordinate) || !(code_coordinate > 0.0)) {
      Kokkos::abort("IONMIX table coordinates must be finite and positive.");
    }
    const Real log_coordinate = log_coordinate_is_precomputed
        ? precomputed_log_coordinate : log(code_coordinate);
    const Real coordinate = log_coordinate+log_unit_scale;
    if (!Kokkos::isfinite(coordinate)) {
      Kokkos::abort("IONMIX table coordinate conversion is not finite.");
    }

    AxisLocation result;
    if (coordinate < axis(0)) {
      if (bounds_error != 0) {
        Kokkos::abort("IONMIX table query is below a configured table axis.");
      }
      result.lower = 0;
      result.fraction = 0.0;
      result.bounded_log_coordinate = axis(0);
      result.query_flags = low_flag;
      return result;
    }
    if (coordinate > axis(size-1)) {
      if (bounds_error != 0) {
        Kokkos::abort("IONMIX table query is above a configured table axis.");
      }
      result.lower = size-2;
      result.fraction = 1.0;
      result.bounded_log_coordinate = axis(size-1);
      result.query_flags = high_flag;
      return result;
    }
    if (coordinate == axis(0)) {
      result.lower = 0;
      result.fraction = 0.0;
      result.bounded_log_coordinate = coordinate;
      return result;
    }
    if (coordinate == axis(size-1)) {
      result.lower = size-2;
      result.fraction = 1.0;
      result.bounded_log_coordinate = coordinate;
      return result;
    }

    // Repeated inverse probes normally remain in one native temperature interval.
    // Intervals own [axis[lower], axis[lower+1]); exact internal nodes therefore fall
    // through to the ordinary search unless the hint already names the upper interval.
    if (lower_hint >= 0 && lower_hint < size-1 &&
        coordinate >= axis(lower_hint) && coordinate < axis(lower_hint+1)) {
      result.lower = lower_hint;
      result.fraction =
          (coordinate-axis(lower_hint))/(axis(lower_hint+1)-axis(lower_hint));
      result.bounded_log_coordinate = coordinate;
      return result;
    }

    int lower = 0;
    int upper = size-1;
    while (upper-lower > 1) {
      const int middle = lower+(upper-lower)/2;
      if (axis(middle) > coordinate) {
        upper = middle;
      } else {
        lower = middle;
      }
    }
    result.lower = lower;
    result.fraction = (coordinate-axis(lower))/(axis(upper)-axis(lower));
    result.bounded_log_coordinate = coordinate;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real InterpolatePair(const Real lower, const Real upper, const Real fraction,
                       const bool allow_geometric) const {
    if (fraction <= 0.0) return lower;
    if (fraction >= 1.0) return upper;
    // Fix the contraction order so inline cache context cannot change rounded table
    // values or mixed-inverse bisection decisions on CUDA.
    if (allow_geometric && lower > 0.0 && upper > 0.0) {
      return exp(Kokkos::fma(fraction, log(upper),
                             (1.0-fraction)*log(lower)));
    }
    return Kokkos::fma(fraction, upper, (1.0-fraction)*lower);
  }

  KOKKOS_INLINE_FUNCTION
  Real InterpolatePairWithLogs(const Real lower, const Real upper,
                               const Real log_lower, const Real log_upper,
                               const Real fraction,
                               const bool allow_geometric) const {
    if (fraction <= 0.0) return lower;
    if (fraction >= 1.0) return upper;
    if (allow_geometric && lower > 0.0 && upper > 0.0) {
      return exp(Kokkos::fma(
          fraction, log_upper, (1.0-fraction)*log_lower));
    }
    return Kokkos::fma(fraction, upper, (1.0-fraction)*lower);
  }

  KOKKOS_INLINE_FUNCTION
  Real InterpolateTablePair(const int field, const int lower_density,
                            const int upper_density, const int temperature,
                            const Real fraction, const bool allow_geometric) const {
    const Real lower = values(field, lower_density, temperature);
    const Real upper = values(field, upper_density, temperature);
    if (fraction <= 0.0) return lower;
    if (fraction >= 1.0) return upper;
    if (allow_geometric && lower > 0.0 && upper > 0.0) {
      return exp(Kokkos::fma(
          fraction, log_values(field, upper_density, temperature),
          (1.0-fraction)*log_values(field, lower_density, temperature)));
    }
    return Kokkos::fma(fraction, upper, (1.0-fraction)*lower);
  }

  KOKKOS_INLINE_FUNCTION
  bool FieldAllowsGeometricInterpolation(const int field) const {
    if (!geometric_interpolation) return false;
    if (field == ion_specific_internal_energy) {
      return ion_energy_is_strictly_positive;
    }
    if (field == electron_specific_internal_energy) {
      return electron_energy_is_strictly_positive;
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  Real EvaluateWithLocations(const int field, const AxisLocation &density,
                             const AxisLocation &temperature) const {
    const int id0 = density.lower;
    const int id1 = id0+1;
    const int it0 = temperature.lower;
    const int it1 = it0+1;
    const bool geometric = FieldAllowsGeometricInterpolation(field);

    // Interpolating density first makes the exact values on every temperature plane
    // independent of which adjacent temperature cell owns a boundary query.
    const Real at_lower_temperature = InterpolateTablePair(
        field, id0, id1, it0, density.fraction, geometric);
    const Real at_upper_temperature = InterpolateTablePair(
        field, id0, id1, it1, density.fraction, geometric);
    return InterpolatePair(at_lower_temperature, at_upper_temperature,
                           temperature.fraction, geometric);
  }

  KOKKOS_INLINE_FUNCTION
  Real ValueAtTemperatureIndex(const int field, const AxisLocation &density,
                               const int temperature_index) const {
    return InterpolateTablePair(
        field, density.lower, density.lower+1, temperature_index,
        density.fraction, FieldAllowsGeometricInterpolation(field));
  }

  KOKKOS_INLINE_FUNCTION
  int PressureField(const IonmixComponent component) const {
    return (component == IonmixComponent::ion) ? ion_pressure : electron_pressure;
  }

  KOKKOS_INLINE_FUNCTION
  int EnergyField(const IonmixComponent component) const {
    return (component == IonmixComponent::ion)
               ? ion_specific_internal_energy
               : electron_specific_internal_energy;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState StateAtLocations(const IonmixComponent component,
                                        const AxisLocation &density,
                                        const AxisLocation &temperature) const {
    IonmixComponentState result;
    result.temperature = exp(temperature.bounded_log_coordinate)/temperature_to_kelvin;
    result.pressure = pressure_from_cgs*
        EvaluateWithLocations(PressureField(component), density, temperature);
    result.specific_internal_energy = specific_energy_from_cgs*
        EvaluateWithLocations(EnergyField(component), density, temperature);
    result.query_flags = density.query_flags | temperature.query_flags;
    return result;
  }

 public:
  KOKKOS_INLINE_FUNCTION
  Real MinimumDensityCode() const {
    return minimum_density_code;
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumDensityCode() const {
    return maximum_density_code;
  }

  KOKKOS_INLINE_FUNCTION
  Real MinimumTemperatureCode() const {
    return minimum_temperature_code;
  }

  KOKKOS_INLINE_FUNCTION
  Real MaximumTemperatureCode() const {
    return maximum_temperature_code;
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureCodeAtIndex(const int index) const {
    return exp(log_temperature_kelvin(index))/temperature_to_kelvin;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixDensityLocation PrepareDensityLocation(const Real code_density) const {
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    IonmixDensityLocation result;
    result.lower = density.lower;
    result.fraction = density.fraction;
    result.query_flags = density.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState ComponentFromPreparedDensityTemperature(
      const IonmixComponent component, const IonmixDensityLocation &prepared_density,
      const Real code_temperature) const {
    AxisLocation density;
    density.lower = prepared_density.lower;
    density.fraction = prepared_density.fraction;
    density.query_flags = prepared_density.query_flags;
    const AxisLocation temperature = Locate(
        log_temperature_kelvin, ntemperature, code_temperature,
        log_temperature_to_kelvin, ionmix_temperature_below_table,
        ionmix_temperature_above_table);
    return StateAtLocations(component, density, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  Real ComponentEnergyFromPreparedDensityTemperature(
      const IonmixComponent component, const IonmixDensityLocation &prepared_density,
      const Real code_temperature, const Real log_code_temperature,
      IonmixEnergyIntervalCache &cache) const {
    AxisLocation density;
    density.lower = prepared_density.lower;
    density.fraction = prepared_density.fraction;
    density.query_flags = prepared_density.query_flags;
    const AxisLocation temperature = Locate<true>(
        log_temperature_kelvin, ntemperature, code_temperature,
        log_temperature_to_kelvin, ionmix_temperature_below_table,
        ionmix_temperature_above_table, cache.temperature_lower,
        log_code_temperature);
    const int energy_field = EnergyField(component);
    if (cache.temperature_lower != temperature.lower) {
      cache.lower_energy =
          ValueAtTemperatureIndex(energy_field, density, temperature.lower);
      cache.upper_energy =
          ValueAtTemperatureIndex(energy_field, density, temperature.lower+1);
      if (FieldAllowsGeometricInterpolation(energy_field) &&
          cache.lower_energy > 0.0 && cache.upper_energy > 0.0) {
        cache.log_lower_energy = log(cache.lower_energy);
        cache.log_upper_energy = log(cache.upper_energy);
      }
      cache.temperature_lower = temperature.lower;
    }
    return specific_energy_from_cgs*InterpolatePairWithLogs(
        cache.lower_energy, cache.upper_energy,
        cache.log_lower_energy, cache.log_upper_energy,
        temperature.fraction, FieldAllowsGeometricInterpolation(energy_field));
  }

  KOKKOS_INLINE_FUNCTION
  IonmixPressureEnergyState PressureEnergyFromRhoMinimumTemperature(
      const Real code_density) const {
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    if (minimum_temperature_round_trips_exactly == 0) {
      return PressureEnergyFromRhoTemperature(
          code_density, MinimumTemperatureCode());
    }
    AxisLocation temperature;
    temperature.lower = 0;
    temperature.fraction = 0.0;
    temperature.bounded_log_coordinate = log_temperature_kelvin(0);
    temperature.query_flags = ionmix_query_in_bounds;
    IonmixPressureEnergyState result;
    result.ion_pressure = pressure_from_cgs*
        EvaluateWithLocations(ion_pressure, density, temperature);
    result.ion_specific_internal_energy = specific_energy_from_cgs*
        EvaluateWithLocations(ion_specific_internal_energy, density, temperature);
    result.electron_pressure = pressure_from_cgs*
        EvaluateWithLocations(electron_pressure, density, temperature);
    result.electron_specific_internal_energy = specific_energy_from_cgs*
        EvaluateWithLocations(electron_specific_internal_energy, density, temperature);
    result.query_flags = density.query_flags | temperature.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixPressureEnergyState PressureEnergyFromRhoTemperature(
      const Real code_density, const Real code_temperature) const {
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    const AxisLocation temperature = Locate(
        log_temperature_kelvin, ntemperature, code_temperature,
        log_temperature_to_kelvin, ionmix_temperature_below_table,
        ionmix_temperature_above_table);
    IonmixPressureEnergyState result;
    result.ion_pressure = pressure_from_cgs*
        EvaluateWithLocations(ion_pressure, density, temperature);
    result.ion_specific_internal_energy = specific_energy_from_cgs*
        EvaluateWithLocations(ion_specific_internal_energy, density, temperature);
    result.electron_pressure = pressure_from_cgs*
        EvaluateWithLocations(electron_pressure, density, temperature);
    result.electron_specific_internal_energy = specific_energy_from_cgs*
        EvaluateWithLocations(electron_specific_internal_energy, density, temperature);
    result.query_flags = density.query_flags | temperature.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixTemperatureState TemperatureFromRhoTemperature(
      const Real code_density, const Real code_temperature) const {
    // Preserve the same density-then-temperature location order and canonical
    // exp/log round trip as ComponentFromRhoTemperature, while omitting only the
    // pressure and energy field evaluations.
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    const AxisLocation temperature = Locate(
        log_temperature_kelvin, ntemperature, code_temperature,
        log_temperature_to_kelvin, ionmix_temperature_below_table,
        ionmix_temperature_above_table);
    IonmixTemperatureState result;
    result.temperature =
        exp(temperature.bounded_log_coordinate)/temperature_to_kelvin;
    result.query_flags = density.query_flags | temperature.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState ComponentFromRhoTemperature(
      const IonmixComponent component, const Real code_density,
      const Real code_temperature) const {
    const IonmixDensityLocation density = PrepareDensityLocation(code_density);
    return ComponentFromPreparedDensityTemperature(
        component, density, code_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState IonFromRhoTemperature(const Real code_density,
                                              const Real code_temperature) const {
    return ComponentFromRhoTemperature(
        IonmixComponent::ion, code_density, code_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState ElectronFromRhoTemperature(
      const Real code_density, const Real code_temperature) const {
    return ComponentFromRhoTemperature(
        IonmixComponent::electron, code_density, code_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState ComponentFromRhoSpecificEnergy(
      const IonmixComponent component, const Real code_density,
      const Real code_specific_energy) const {
    if (!Kokkos::isfinite(code_specific_energy)) {
      Kokkos::abort("IONMIX inverse energy must be finite.");
    }
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    const int energy_field = EnergyField(component);
    const Real target = code_specific_energy/specific_energy_from_cgs;
    if (!Kokkos::isfinite(target)) {
      Kokkos::abort("IONMIX inverse energy conversion is not finite.");
    }

    const Real minimum = ValueAtTemperatureIndex(energy_field, density, 0);
    const Real maximum = ValueAtTemperatureIndex(
        energy_field, density, ntemperature-1);
    if (target < minimum) {
      if (bounds_error != 0) {
        Kokkos::abort("IONMIX inverse energy is below the table range.");
      }
      AxisLocation temperature;
      temperature.lower = 0;
      temperature.fraction = 0.0;
      temperature.bounded_log_coordinate = log_temperature_kelvin(0);
      temperature.query_flags = ionmix_energy_below_table;
      return StateAtLocations(component, density, temperature);
    }
    if (target > maximum) {
      if (bounds_error != 0) {
        Kokkos::abort("IONMIX inverse energy is above the table range.");
      }
      AxisLocation temperature;
      temperature.lower = ntemperature-2;
      temperature.fraction = 1.0;
      temperature.bounded_log_coordinate = log_temperature_kelvin(ntemperature-1);
      temperature.query_flags = ionmix_energy_above_table;
      return StateAtLocations(component, density, temperature);
    }

    // lower_bound returns the first temperature carrying the target plateau value.
    // This makes a non-unique plateau inversion deterministic and avoids zero slopes.
    int first = 0;
    int last = ntemperature;
    while (first < last) {
      const int middle = first+(last-first)/2;
      if (ValueAtTemperatureIndex(energy_field, density, middle) < target) {
        first = middle+1;
      } else {
        last = middle;
      }
    }

    AxisLocation temperature;
    if (first == 0) {
      temperature.lower = 0;
      temperature.fraction = 0.0;
      temperature.bounded_log_coordinate = log_temperature_kelvin(0);
    } else {
      const Real upper_value = ValueAtTemperatureIndex(
          energy_field, density, first);
      if (upper_value == target) {
        const int exact = first;
        if (exact == ntemperature-1) {
          temperature.lower = ntemperature-2;
          temperature.fraction = 1.0;
        } else {
          temperature.lower = exact;
          temperature.fraction = 0.0;
        }
        temperature.bounded_log_coordinate = log_temperature_kelvin(exact);
      } else {
        const int lower = first-1;
        const Real lower_value = ValueAtTemperatureIndex(
            energy_field, density, lower);
        if (!(upper_value > lower_value)) {
          Kokkos::abort("IONMIX inverse encountered a non-increasing energy bracket.");
        }
        Real fraction;
        if (FieldAllowsGeometricInterpolation(energy_field) &&
            target > 0.0 && lower_value > 0.0 && upper_value > 0.0) {
          fraction = (log(target)-log(lower_value))/
                     (log(upper_value)-log(lower_value));
        } else {
          fraction = (target-lower_value)/(upper_value-lower_value);
        }
        fraction = fmin(fmax(fraction, 0.0), 1.0);
        temperature.lower = lower;
        temperature.fraction = fraction;
        temperature.bounded_log_coordinate =
            (1.0-fraction)*log_temperature_kelvin(lower)+
            fraction*log_temperature_kelvin(first);
      }
    }
    return StateAtLocations(component, density, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState IonFromRhoSpecificEnergy(
      const Real code_density, const Real code_specific_energy) const {
    return ComponentFromRhoSpecificEnergy(
        IonmixComponent::ion, code_density, code_specific_energy);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixComponentState ElectronFromRhoSpecificEnergy(
      const Real code_density, const Real code_specific_energy) const {
    return ComponentFromRhoSpecificEnergy(
        IonmixComponent::electron, code_density, code_specific_energy);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanIonizationFromRhoTemperature(const Real code_density,
                                        const Real code_temperature) const {
    const AxisLocation density = Locate(
        log_density_cgs, ndensity, code_density, log_density_to_cgs,
        ionmix_density_below_table, ionmix_density_above_table);
    const AxisLocation temperature = Locate(
        log_temperature_kelvin, ntemperature, code_temperature,
        log_temperature_to_kelvin, ionmix_temperature_below_table,
        ionmix_temperature_above_table);
    return EvaluateWithLocations(mean_ionization, density, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  IonmixTwoTemperatureState StateFromRhoTemperatures(
      const Real code_density, const Real code_ion_temperature,
      const Real code_electron_temperature) const {
    IonmixTwoTemperatureState result;
    result.ion = IonFromRhoTemperature(code_density, code_ion_temperature);
    result.electron = ElectronFromRhoTemperature(
        code_density, code_electron_temperature);
    result.mean_ionization = MeanIonizationFromRhoTemperature(
        code_density, result.electron.temperature);
    result.query_flags = result.ion.query_flags | result.electron.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  IonmixTwoTemperatureState StateFromRhoSpecificEnergies(
      const Real code_density, const Real code_ion_specific_energy,
      const Real code_electron_specific_energy) const {
    IonmixTwoTemperatureState result;
    result.ion = IonFromRhoSpecificEnergy(
        code_density, code_ion_specific_energy);
    result.electron = ElectronFromRhoSpecificEnergy(
        code_density, code_electron_specific_energy);
    result.mean_ionization = MeanIonizationFromRhoTemperature(
        code_density, result.electron.temperature);
    result.query_flags = result.ion.query_flags | result.electron.query_flags;
    return result;
  }
};

//----------------------------------------------------------------------------------------
//! \class IonmixTwoTemperatureTable
//! \brief Rank-safe owner/reader for native separate ion/electron EOS surfaces.

class IonmixTwoTemperatureTable {
 public:
  explicit IonmixTwoTemperatureTable(
      const std::string &filename,
      const IonmixTwoTemperatureTableOptions &options =
          IonmixTwoTemperatureTableOptions());
  ~IonmixTwoTemperatureTable() = default;

  IonmixTwoTemperatureTableDevice DeviceData() const;
  const IonmixTwoTemperatureTableMetadata &Metadata() const { return metadata_; }
  bool SharesTemperatureGrid(const IonmixTwoTemperatureTable &other) const;

 private:
  IonmixTwoTemperatureTableOptions options_;
  IonmixTwoTemperatureTableMetadata metadata_;
  DualArray1D<Real> log_density_cgs_;
  DualArray1D<Real> log_temperature_kelvin_;
  DualArray3D<Real> values_;
  DualArray3D<Real> log_values_;
  int minimum_temperature_round_trips_exactly_ = 0;
  Real log_density_to_cgs_ = 0.0;
  Real log_temperature_to_kelvin_ = 0.0;
  Real minimum_density_code_ = 0.0;
  Real maximum_density_code_ = 0.0;
  Real minimum_temperature_code_ = 0.0;
  Real maximum_temperature_code_ = 0.0;
};

} // namespace materials

#endif // MATERIALS_IONMIX_TWO_TEMPERATURE_TABLE_HPP_
