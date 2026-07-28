#ifndef EOS_TABLE_EOS_HPP_
#define EOS_TABLE_EOS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file table_eos.hpp
//! \brief Device-copyable density-temperature EOS table and inverse lookups.

#include <math.h>

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "athena.hpp"

struct TableEOSData {
  enum Field : int {
    log_pressure = 0,
    log_specific_eint = 1,
    log_sound_speed2 = 2,
    gamma1 = 3,
    gamma3_minus_one = 4,
    mean_ionization = 5,
    effective_charge = 6,
    mean_atomic_mass = 7,
    mean_molecular_weight = 8
  };
  static constexpr int nfields = 9;
  static constexpr int nmaterial_fields = 6;
#if SINGLE_PRECISION_ENABLED
  static constexpr Real minimum_positive = FLT_MIN;
#else
  static constexpr Real minimum_positive = DBL_MIN;
#endif

  struct ThermoState {
    Real temperature = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real pressure = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real specific_internal_energy = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real sound_speed_squared = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real gamma1 = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real gamma3_minus_one = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real mean_ionization = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real effective_charge = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real mean_atomic_mass = Kokkos::Experimental::quiet_NaN<Real>::value;
    Real mean_molecular_weight = Kokkos::Experimental::quiet_NaN<Real>::value;
    bool has_gamma1 = false;
    bool has_gamma3_minus_one = false;
    bool has_mean_ionization = false;
    bool has_effective_charge = false;
    bool has_mean_atomic_mass = false;
    bool has_mean_molecular_weight = false;
  };

  int ndensity = 0;
  int ntemperature = 0;
  int bounds_error = 0;
  bool has_gamma1 = false;
  bool has_gamma3_minus_one = false;
  bool has_mean_ionization = false;
  bool has_effective_charge = false;
  bool has_mean_atomic_mass = false;
  bool has_mean_molecular_weight = false;
  DvceArray1D<Real> log_density;
  DvceArray1D<Real> log_temperature;
  DvceArray3D<Real> values;
  HostArray1D<Real> log_density_h;
  HostArray1D<Real> log_temperature_h;
  HostArray3D<Real> values_h;

  KOKKOS_INLINE_FUNCTION
  Real SafeLog(Real value) const {
    return log(fmax(value, minimum_positive));
  }

  KOKKOS_INLINE_FUNCTION
  void AxisWeights(const DvceArray1D<Real> &axis, int size, Real coordinate,
                   int &lower, Real &fraction) const {
    if (bounds_error != 0 &&
        (coordinate < axis(0) || coordinate > axis(size-1))) {
      Kokkos::abort("Tabulated EOS query is outside a configured table axis.");
    }
    Real bounded = fmin(fmax(coordinate, axis(0)), axis(size-1));
    if (bounded <= axis(0)) {
      lower = 0;
      fraction = 0.0;
      return;
    }
    if (bounded >= axis(size-1)) {
      lower = size-2;
      fraction = 1.0;
      return;
    }
    int lo = 0;
    int hi = size-1;
    while (hi-lo > 1) {
      int mid = lo+(hi-lo)/2;
      if (axis(mid) > bounded) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    lower = lo;
    fraction = (bounded-axis(lo))/(axis(hi)-axis(lo));
  }

  KOKKOS_INLINE_FUNCTION
  Real FieldAtTemperatureIndex(int field, Real logrho, int temperature_index) const {
    int density_index;
    Real density_fraction;
    AxisWeights(log_density, ndensity, logrho, density_index, density_fraction);
    return (1.0-density_fraction)*values(field, density_index, temperature_index)
           + density_fraction*values(field, density_index+1, temperature_index);
  }

  KOKKOS_INLINE_FUNCTION
  Real EvaluateWithWeights(int field, int density_index, int temperature_index,
                           Real density_fraction, Real temperature_fraction) const {
    Real low = (1.0-temperature_fraction)*
                   values(field, density_index, temperature_index)
               + temperature_fraction*
                   values(field, density_index, temperature_index+1);
    Real high = (1.0-temperature_fraction)*
                    values(field, density_index+1, temperature_index)
                + temperature_fraction*
                    values(field, density_index+1, temperature_index+1);
    return (1.0-density_fraction)*low+density_fraction*high;
  }

  KOKKOS_INLINE_FUNCTION
  Real Evaluate(int field, Real logrho, Real logtemperature) const {
    int density_index, temperature_index;
    Real density_fraction, temperature_fraction;
    AxisWeights(log_density, ndensity, logrho, density_index, density_fraction);
    AxisWeights(log_temperature, ntemperature, logtemperature,
                temperature_index, temperature_fraction);
    return EvaluateWithWeights(field, density_index, temperature_index,
                               density_fraction, temperature_fraction);
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromLogField(int field, Real logrho, Real target) const {
    Real minimum = FieldAtTemperatureIndex(field, logrho, 0);
    Real maximum = FieldAtTemperatureIndex(field, logrho, ntemperature-1);
    if (bounds_error != 0 && (target < minimum || target > maximum)) {
      Kokkos::abort("Tabulated EOS inverse query is outside the table range.");
    }
    if (target <= minimum) return exp(log_temperature(0));
    if (target >= maximum) return exp(log_temperature(ntemperature-1));

    int lo = 0;
    int hi = ntemperature-1;
    Real value_lo = minimum;
    Real value_hi = maximum;
    while (hi-lo > 1) {
      int mid = lo+(hi-lo)/2;
      Real value_mid = FieldAtTemperatureIndex(field, logrho, mid);
      if (value_mid > target) {
        hi = mid;
        value_hi = value_mid;
      } else {
        lo = mid;
        value_lo = value_mid;
      }
    }
    Real fraction = (target-value_lo)/(value_hi-value_lo);
    return exp(log_temperature(lo)+
               fraction*(log_temperature(hi)-log_temperature(lo)));
  }

  KOKKOS_INLINE_FUNCTION
  Real PressureFromRhoTemperature(Real density, Real temperature) const {
    return exp(Evaluate(log_pressure, SafeLog(density), SafeLog(temperature)));
  }

  KOKKOS_INLINE_FUNCTION
  Real SpecificEintFromRhoTemperature(Real density, Real temperature) const {
    return exp(Evaluate(log_specific_eint, SafeLog(density), SafeLog(temperature)));
  }

  KOKKOS_INLINE_FUNCTION
  ThermoState ThermoStateFromRhoTemperature(Real density, Real temperature) const {
    int density_index, temperature_index;
    Real density_fraction, temperature_fraction;
    AxisWeights(log_density, ndensity, SafeLog(density),
                density_index, density_fraction);
    AxisWeights(log_temperature, ntemperature, SafeLog(temperature),
                temperature_index, temperature_fraction);

    ThermoState state;
    Real bounded_log_temperature =
        (1.0-temperature_fraction)*log_temperature(temperature_index) +
        temperature_fraction*log_temperature(temperature_index+1);
    state.temperature = exp(bounded_log_temperature);
    state.pressure = exp(EvaluateWithWeights(
        log_pressure, density_index, temperature_index,
        density_fraction, temperature_fraction));
    state.specific_internal_energy = exp(EvaluateWithWeights(
        log_specific_eint, density_index, temperature_index,
        density_fraction, temperature_fraction));
    state.sound_speed_squared = exp(EvaluateWithWeights(
        log_sound_speed2, density_index, temperature_index,
        density_fraction, temperature_fraction));

    state.has_gamma1 = has_gamma1;
    state.has_gamma3_minus_one = has_gamma3_minus_one;
    state.has_mean_ionization = has_mean_ionization;
    state.has_effective_charge = has_effective_charge;
    state.has_mean_atomic_mass = has_mean_atomic_mass;
    state.has_mean_molecular_weight = has_mean_molecular_weight;
    if (has_gamma1) {
      state.gamma1 = EvaluateWithWeights(
          gamma1, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    if (has_gamma3_minus_one) {
      state.gamma3_minus_one = EvaluateWithWeights(
          gamma3_minus_one, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    if (has_mean_ionization) {
      state.mean_ionization = EvaluateWithWeights(
          mean_ionization, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    if (has_effective_charge) {
      state.effective_charge = EvaluateWithWeights(
          effective_charge, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    if (has_mean_atomic_mass) {
      state.mean_atomic_mass = EvaluateWithWeights(
          mean_atomic_mass, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    if (has_mean_molecular_weight) {
      state.mean_molecular_weight = EvaluateWithWeights(
          mean_molecular_weight, density_index, temperature_index,
          density_fraction, temperature_fraction);
    }
    return state;
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromRhoEint(Real density, Real eint_density) const {
    Real safe_density = fmax(density, minimum_positive);
    return TemperatureFromLogField(
        log_specific_eint, SafeLog(safe_density), SafeLog(eint_density/safe_density));
  }

  KOKKOS_INLINE_FUNCTION
  Real TemperatureFromRhoPressure(Real density, Real pressure) const {
    return TemperatureFromLogField(
        log_pressure, SafeLog(density), SafeLog(pressure));
  }

  KOKKOS_INLINE_FUNCTION
  Real PressureFromRhoEint(Real density, Real eint_density) const {
    Real temperature = TemperatureFromRhoEint(density, eint_density);
    return PressureFromRhoTemperature(density, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed2FromRhoEint(Real density, Real eint_density) const {
    Real temperature = TemperatureFromRhoEint(density, eint_density);
    return exp(Evaluate(
        log_sound_speed2, SafeLog(density), SafeLog(temperature)));
  }

  KOKKOS_INLINE_FUNCTION
  ThermoState ThermoStateFromRhoEint(Real density, Real eint_density) const {
    return ThermoStateFromRhoTemperature(
        density, TemperatureFromRhoEint(density, eint_density));
  }

  KOKKOS_INLINE_FUNCTION
  Real EintDensityFromRhoPressure(Real density, Real pressure) const {
    Real temperature = TemperatureFromRhoPressure(density, pressure);
    return density*SpecificEintFromRhoTemperature(density, temperature);
  }

  KOKKOS_INLINE_FUNCTION
  ThermoState ThermoStateFromRhoPressure(Real density, Real pressure) const {
    return ThermoStateFromRhoTemperature(
        density, TemperatureFromRhoPressure(density, pressure));
  }

  KOKKOS_INLINE_FUNCTION
  Real MinimumEintDensity(Real density, Real pressure_floor,
                          Real temperature_floor) const {
    Real minimum_temperature = exp(log_temperature(0));
    Real temperature = fmax(minimum_temperature, temperature_floor);
    Real eint = density*SpecificEintFromRhoTemperature(density, temperature);
    Real minimum_pressure = PressureFromRhoTemperature(density, minimum_temperature);
    if (pressure_floor > minimum_pressure) {
      Real pressure_eint = EintDensityFromRhoPressure(density, pressure_floor);
      eint = fmax(eint, pressure_eint);
    }
    return eint;
  }

  void HostAxisWeights(const HostArray1D<Real> &axis, int size, Real coordinate,
                       int &lower, Real &fraction) const {
    if (bounds_error != 0 &&
        (coordinate < axis(0) || coordinate > axis(size-1))) {
      Kokkos::abort("Tabulated EOS host query is outside a configured table axis.");
    }
    Real bounded = std::min(std::max(coordinate, axis(0)), axis(size-1));
    if (bounded <= axis(0)) {
      lower = 0;
      fraction = 0.0;
      return;
    }
    if (bounded >= axis(size-1)) {
      lower = size-2;
      fraction = 1.0;
      return;
    }
    int lo = 0;
    int hi = size-1;
    while (hi-lo > 1) {
      int mid = lo+(hi-lo)/2;
      if (axis(mid) > bounded) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    lower = lo;
    fraction = (bounded-axis(lo))/(axis(hi)-axis(lo));
  }

  Real HostFieldAtTemperatureIndex(int field, Real logrho,
                                   int temperature_index) const {
    int density_index;
    Real density_fraction;
    HostAxisWeights(log_density_h, ndensity, logrho, density_index, density_fraction);
    return (1.0-density_fraction)*values_h(field, density_index, temperature_index)
           + density_fraction*values_h(field, density_index+1, temperature_index);
  }

  Real HostTemperatureFromLogField(int field, Real logrho, Real target) const {
    Real minimum = HostFieldAtTemperatureIndex(field, logrho, 0);
    Real maximum = HostFieldAtTemperatureIndex(field, logrho, ntemperature-1);
    if (bounds_error != 0 && (target < minimum || target > maximum)) {
      Kokkos::abort("Tabulated EOS host inverse query is outside the table range.");
    }
    if (target <= minimum) return std::exp(log_temperature_h(0));
    if (target >= maximum) return std::exp(log_temperature_h(ntemperature-1));

    int lo = 0;
    int hi = ntemperature-1;
    Real value_lo = minimum;
    Real value_hi = maximum;
    while (hi-lo > 1) {
      int mid = lo+(hi-lo)/2;
      Real value_mid = HostFieldAtTemperatureIndex(field, logrho, mid);
      if (value_mid > target) {
        hi = mid;
        value_hi = value_mid;
      } else {
        lo = mid;
        value_lo = value_mid;
      }
    }
    Real fraction = (target-value_lo)/(value_hi-value_lo);
    return std::exp(log_temperature_h(lo)+
                    fraction*(log_temperature_h(hi)-log_temperature_h(lo)));
  }

  Real HostEintDensityFromRhoPressure(Real density, Real pressure) const {
    Real safe_density = std::max(density, minimum_positive);
    Real temperature = HostTemperatureFromLogField(
        log_pressure, std::log(safe_density),
        std::log(std::max(pressure, minimum_positive)));
    int density_index, temperature_index;
    Real density_fraction, temperature_fraction;
    HostAxisWeights(log_density_h, ndensity, std::log(safe_density),
                    density_index, density_fraction);
    HostAxisWeights(log_temperature_h, ntemperature, std::log(temperature),
                    temperature_index, temperature_fraction);
    Real low = (1.0-temperature_fraction)*
                   values_h(log_specific_eint, density_index, temperature_index)
               + temperature_fraction*
                   values_h(log_specific_eint, density_index, temperature_index+1);
    Real high = (1.0-temperature_fraction)*
                    values_h(log_specific_eint, density_index+1, temperature_index)
                + temperature_fraction*
                    values_h(log_specific_eint, density_index+1, temperature_index+1);
    return safe_density*std::exp(
        (1.0-density_fraction)*low+density_fraction*high);
  }
};

#endif // EOS_TABLE_EOS_HPP_
