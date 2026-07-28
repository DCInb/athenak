#ifndef MATERIALS_MATERIAL_MIXTURE_HPP_
#define MATERIALS_MATERIAL_MIXTURE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.hpp
//! \brief Two-material ideal or separate-ion/electron tabular plasma closure.

#include <memory>

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

//----------------------------------------------------------------------------------------
//! \struct MaterialMixtureDevice
//! \brief Device-copyable two-material closure represented by rho*Y0.
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
  SpeciesProperties material0;
  SpeciesProperties material1;
  IonmixTwoTemperatureTableDevice material0_table;
  IonmixTwoTemperatureTableDevice material1_table;
  int scalar_index = -1;  //!< Absolute primitive/conserved variable index.
  bool use_tabular_eos = false;
  Real gamma_minus_one = 2.0/3.0;
  Real density_to_cgs = 1.0;
  Real temperature_to_kelvin = 1.0;
  Real wave_speed_safety = 1.05;

  static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;

 private:
  struct ComponentAtTemperature {
    Real temperature = 0.0;
    Real pressure = 0.0;
    Real specific_internal_energy = 0.0;
    int query_flags = ionmix_query_in_bounds;
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

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoTemperature(
      const IonmixComponent component, const Real density,
      const Real temperature, const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real y1 = 1.0-y0;
    const ComponentAtTemperature state0 = SpeciesComponent(
        material0_table, component, density*y0, temperature);
    const ComponentAtTemperature state1 = SpeciesComponent(
        material1_table, component, density*y1, temperature);
    ComponentAtTemperature result;
    result.temperature = (y0 > 0.0) ? state0.temperature : state1.temperature;
    result.pressure = state0.pressure+state1.pressure;
    result.specific_internal_energy =
        y0*state0.specific_internal_energy+y1*state1.specific_internal_energy;
    result.query_flags = state0.query_flags | state1.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real CommonMinimumTemperature() const {
    return fmax(material0_table.MinimumTemperatureCode(),
                material1_table.MinimumTemperatureCode());
  }

  KOKKOS_INLINE_FUNCTION
  Real CommonMaximumTemperature() const {
    return fmin(material0_table.MaximumTemperatureCode(),
                material1_table.MaximumTemperatureCode());
  }

  KOKKOS_INLINE_FUNCTION
  ComponentAtTemperature MixtureComponentFromRhoSpecificEnergy(
      const IonmixComponent component, const Real density,
      const Real target_energy, const Real y0) const {
    if (!Kokkos::isfinite(target_energy)) {
      Kokkos::abort("Mixed IONMIX inverse energy must be finite.");
    }
    const int energy_low_flag = ionmix_energy_below_table;
    const int energy_high_flag = ionmix_energy_above_table;
    const Real fraction0 = ClampMassFraction(y0);
    if (fraction0 >= 1.0) {
      const IonmixComponentState state =
          material0_table.ComponentFromRhoSpecificEnergy(
              component, density, target_energy);
      ComponentAtTemperature result;
      result.temperature = state.temperature;
      result.pressure = state.pressure;
      result.specific_internal_energy = state.specific_internal_energy;
      result.query_flags = state.query_flags;
      return result;
    }
    if (fraction0 <= 0.0) {
      const IonmixComponentState state =
          material1_table.ComponentFromRhoSpecificEnergy(
              component, density, target_energy);
      ComponentAtTemperature result;
      result.temperature = state.temperature;
      result.pressure = state.pressure;
      result.specific_internal_energy = state.specific_internal_energy;
      result.query_flags = state.query_flags;
      return result;
    }
    const bool error_bounds = material0_table.bounds_error != 0 ||
                              material1_table.bounds_error != 0;
    const Real minimum_temperature = CommonMinimumTemperature();
    const Real maximum_temperature = CommonMaximumTemperature();
    ComponentAtTemperature minimum = MixtureComponentFromRhoTemperature(
        component, density, minimum_temperature, y0);
    ComponentAtTemperature maximum = MixtureComponentFromRhoTemperature(
        component, density, maximum_temperature, y0);
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
      maximum.query_flags |= energy_high_flag;
      return maximum;
    }

    // A mass-weighted sum of two geometric surfaces is not itself geometric.  A short
    // safeguarded log-temperature bisection preserves the exact forward rule and also
    // supports CH/He tables with different native temperature grids.
    if (target_energy == minimum.specific_internal_energy) return minimum;
    if (target_energy == maximum.specific_internal_energy) return maximum;
    Real log_low = log(minimum_temperature);
    Real log_high = log(maximum_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real log_trial = 0.5*(log_low+log_high);
      const ComponentAtTemperature trial = MixtureComponentFromRhoTemperature(
          component, density, exp(log_trial), y0);
      if (trial.specific_internal_energy < target_energy) {
        log_low = log_trial;
      } else {
        log_high = log_trial;
      }
    }
    return MixtureComponentFromRhoTemperature(
        component, density, exp(0.5*(log_low+log_high)), y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularMeanIonization(const Real density, const Real electron_temperature,
                             const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real y1 = 1.0-y0;
    Real electron_weight = 0.0;
    Real ion_weight = 0.0;
    if (y0 > 0.0) {
      const Real rho0 = fmax(density*y0, material0_table.MinimumDensityCode());
      const Real z0 = material0_table.MeanIonizationFromRhoTemperature(
          rho0, electron_temperature);
      electron_weight += y0*z0/material0.abar;
      ion_weight += y0/material0.abar;
    }
    if (y1 > 0.0) {
      const Real rho1 = fmax(density*y1, material1_table.MinimumDensityCode());
      const Real z1 = material1_table.MeanIonizationFromRhoTemperature(
          rho1, electron_temperature);
      electron_weight += y1*z1/material1.abar;
      ion_weight += y1/material1.abar;
    }
    return (ion_weight > 0.0) ? electron_weight/ion_weight : 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularElectronNumberPerAtomicMass(
      const Real density, const Real electron_temperature,
      const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real y1 = 1.0-y0;
    Real result = 0.0;
    if (y0 > 0.0) {
      const Real rho0 = fmax(density*y0, material0_table.MinimumDensityCode());
      result += y0*material0_table.MeanIonizationFromRhoTemperature(
                       rho0, electron_temperature)/material0.abar;
    }
    if (y1 > 0.0) {
      const Real rho1 = fmax(density*y1, material1_table.MinimumDensityCode());
      result += y1*material1_table.MeanIonizationFromRhoTemperature(
                       rho1, electron_temperature)/material1.abar;
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularEffectiveCharge(const Real density, const Real electron_temperature,
                              const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real y1 = 1.0-y0;
    Real ne0 = 0.0;
    Real ne1 = 0.0;
    if (y0 > 0.0) {
      const Real rho0 = fmax(density*y0, material0_table.MinimumDensityCode());
      ne0 = y0*material0_table.MeanIonizationFromRhoTemperature(
                    rho0, electron_temperature)/material0.abar;
    }
    if (y1 > 0.0) {
      const Real rho1 = fmax(density*y1, material1_table.MinimumDensityCode());
      ne1 = y1*material1_table.MeanIonizationFromRhoTemperature(
                    rho1, electron_temperature)/material1.abar;
    }
    return (ne0+ne1 > 0.0)
               ? (ne0*material0.zeff+ne1*material1.zeff)/(ne0+ne1)
               : fmax(material0.zeff, material1.zeff);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState TabularStateNoSound(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    const ComponentAtTemperature ion = MixtureComponentFromRhoTemperature(
        IonmixComponent::ion, density, ion_temperature, y0);
    const ComponentAtTemperature electron = MixtureComponentFromRhoTemperature(
        IonmixComponent::electron, density, electron_temperature, y0);
    MaterialThermodynamicState result;
    result.ion_temperature = ion.temperature;
    result.electron_temperature = electron.temperature;
    result.ion_pressure = ion.pressure;
    result.electron_pressure = electron.pressure;
    result.ion_specific_internal_energy = ion.specific_internal_energy;
    result.electron_specific_internal_energy = electron.specific_internal_energy;
    result.mean_ionization = TabularMeanIonization(
        density, electron.temperature, y0);
    result.electron_number_density_cgs = density*density_to_cgs*
        TabularElectronNumberPerAtomicMass(density, electron.temperature, y0)/
        atomic_mass_unit_cgs;
    result.effective_charge = TabularEffectiveCharge(
        density, electron.temperature, y0);
    result.query_flags = ion.query_flags | electron.query_flags;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real TabularSoundSpeedSquared(const Real density, const Real ion_temperature,
                                const Real electron_temperature,
                                const Real y0) const {
    constexpr Real log_step = 1.0e-3;
    const Real density_low = density*exp(-log_step);
    Real density_high = density*exp(log_step);
    const Real fraction0 = ClampMassFraction(y0);
    const Real fraction1 = 1.0-fraction0;
    Real maximum_density = Kokkos::Experimental::infinity<Real>::value;
    if (fraction0 > 0.0) {
      maximum_density = fmin(
          maximum_density, material0_table.MaximumDensityCode()/fraction0);
    }
    if (fraction1 > 0.0) {
      maximum_density = fmin(
          maximum_density, material1_table.MaximumDensityCode()/fraction1);
    }
    density_high = fmin(density_high, maximum_density);

    Real sound_speed_squared = 0.0;
    for (int component_index = 0; component_index < 2; ++component_index) {
      const IonmixComponent component = (component_index == 0)
          ? IonmixComponent::ion : IonmixComponent::electron;
      const Real temperature = (component_index == 0)
          ? ion_temperature : electron_temperature;
      const Real temperature_low = fmax(
          temperature*exp(-log_step), CommonMinimumTemperature());
      const Real temperature_high = fmin(
          temperature*exp(log_step), CommonMaximumTemperature());
      const ComponentAtTemperature center = MixtureComponentFromRhoTemperature(
          component, density, temperature, y0);
      const ComponentAtTemperature rho_low = MixtureComponentFromRhoTemperature(
          component, density_low, temperature, y0);
      const ComponentAtTemperature rho_high = MixtureComponentFromRhoTemperature(
          component, density_high, temperature, y0);
      const ComponentAtTemperature temp_low = MixtureComponentFromRhoTemperature(
          component, density, temperature_low, y0);
      const ComponentAtTemperature temp_high = MixtureComponentFromRhoTemperature(
          component, density, temperature_high, y0);
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
          (center.pressure/(density*density)-de_drho)/de_dtemperature;
      sound_speed_squared += dp_drho+dp_dtemperature*dtemperature_drho;
    }
    const MaterialThermodynamicState center = TabularStateNoSound(
        density, ion_temperature, electron_temperature, y0);
    const Real pressure_scale =
        fmax((center.ion_pressure+center.electron_pressure)/density, 0.0);
    sound_speed_squared = fmax(sound_speed_squared, pressure_scale);
    return SQR(wave_speed_safety)*sound_speed_squared;
  }

 public:
  KOKKOS_INLINE_FUNCTION
  bool UsesTabularEOS() const { return use_tabular_eos; }

  KOKKOS_INLINE_FUNCTION
  Real ClampMassFraction(const Real y0) const {
    return fmin(fmax(y0, 0.0), 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromPrimitive(const DvceArray5D<Real> &prim,
                                           const int m, const int k,
                                           const int j, const int i) const {
    return ClampMassFraction(prim(m, scalar_index, k, j, i));
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromConserved(const DvceArray5D<Real> &cons,
                                          const int m, const int k,
                                          const int j, const int i,
                                          const Real density_floor = 0.0) const {
    const Real density = fmax(cons(m, IDN, k, j, i), density_floor);
    if (!(density > 0.0)) return 0.0;
    return ClampMassFraction(cons(m, scalar_index, k, j, i)/density);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0/material0.abar+(1.0-y0)/material1.abar;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0*material0.zbar/material0.abar+
           (1.0-y0)*material1.zbar/material1.abar;
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
        ion_temperature*exp(-log_step), CommonMinimumTemperature());
    const Real ti_high = fmin(
        ion_temperature*exp(log_step), CommonMaximumTemperature());
    const Real te_low = fmax(
        electron_temperature*exp(-log_step), CommonMinimumTemperature());
    const Real te_high = fmin(
        electron_temperature*exp(log_step), CommonMaximumTemperature());
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
  Real EffectiveCharge(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*material0.zbar/material0.abar;
    const Real ne1 = (1.0-y0)*material1.zbar/material1.abar;
    return (ne0+ne1 > 0.0)
               ? (ne0*material0.zeff+ne1*material1.zeff)/(ne0+ne1)
               : fmax(material0.zeff, material1.zeff);
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

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoTemperatures(
      const Real density, const Real ion_temperature,
      const Real electron_temperature, const Real y0) const {
    if (use_tabular_eos) {
      MaterialThermodynamicState result = TabularStateNoSound(
          density, ion_temperature, electron_temperature, y0);
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature, y0);
      return result;
    }
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
    result.sound_speed_squared = (1.0+gamma_minus_one)*
        (result.ion_pressure+result.electron_pressure)/density;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState StateFromRhoSpecificEnergies(
      const Real density, const Real ion_specific_energy,
      const Real electron_specific_energy, const Real y0) const {
    if (use_tabular_eos) {
      const ComponentAtTemperature ion = MixtureComponentFromRhoSpecificEnergy(
          IonmixComponent::ion, density, ion_specific_energy, y0);
      const ComponentAtTemperature electron = MixtureComponentFromRhoSpecificEnergy(
          IonmixComponent::electron, density, electron_specific_energy, y0);
      MaterialThermodynamicState result = TabularStateNoSound(
          density, ion.temperature, electron.temperature, y0);
      result.sound_speed_squared = TabularSoundSpeedSquared(
          density, result.ion_temperature, result.electron_temperature, y0);
      result.query_flags |= ion.query_flags | electron.query_flags;
      return result;
    }
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    const Real ti = gamma_minus_one*ion_specific_energy/fi;
    const Real te = gamma_minus_one*electron_specific_energy/fe;
    return StateFromRhoTemperatures(density, ti, te, y0);
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
    const bool error_bounds = material0_table.bounds_error != 0 ||
                              material1_table.bounds_error != 0;
    const Real minimum_temperature = CommonMinimumTemperature();
    const Real maximum_temperature = CommonMaximumTemperature();
    const Real low_electron_temperature = fmin(fmax(
        electron_to_ion_temperature*minimum_temperature, minimum_temperature),
        maximum_temperature);
    const Real high_electron_temperature = fmin(fmax(
        electron_to_ion_temperature*maximum_temperature, minimum_temperature),
        maximum_temperature);
    MaterialThermodynamicState low = TabularStateNoSound(
        density, minimum_temperature, low_electron_temperature, y0);
    MaterialThermodynamicState high = TabularStateNoSound(
        density, maximum_temperature, high_electron_temperature, y0);
    const Real low_energy = low.ion_specific_internal_energy+
                            low.electron_specific_internal_energy;
    const Real high_energy = high.ion_specific_internal_energy+
                             high.electron_specific_internal_energy;
    if (total_specific_energy < low_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is below range.");
      low.query_flags |= ionmix_energy_below_table;
      low.sound_speed_squared = TabularSoundSpeedSquared(
          density, low.ion_temperature, low.electron_temperature, y0);
      return low;
    }
    if (total_specific_energy > high_energy) {
      if (error_bounds) Kokkos::abort("Initial mixed IONMIX energy is above range.");
      high.query_flags |= ionmix_energy_above_table;
      high.sound_speed_squared = TabularSoundSpeedSquared(
          density, high.ion_temperature, high.electron_temperature, y0);
      return high;
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(maximum_temperature);
    for (int iteration = 0; iteration < 48; ++iteration) {
      const Real ti = exp(0.5*(log_low+log_high));
      const Real te = fmin(fmax(
          electron_to_ion_temperature*ti, minimum_temperature),
          maximum_temperature);
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
    const Real te = fmin(fmax(
        electron_to_ion_temperature*ti, minimum_temperature),
        maximum_temperature);
    return StateFromRhoTemperatures(density, ti, te, y0);
  }

  KOKKOS_INLINE_FUNCTION
  MaterialThermodynamicState MinimumState(
      const Real density, const Real y0, const Real pressure_floor = 0.0,
      const Real temperature_floor = 0.0) const {
    if (!use_tabular_eos) {
      Real temperature = fmax(temperature_floor, 0.0);
      MaterialThermodynamicState state = StateFromRhoTemperatures(
          density, temperature, temperature, y0);
      if (state.ion_pressure+state.electron_pressure < pressure_floor) {
        temperature = pressure_floor/density;
        state = StateFromRhoTemperatures(density, temperature, temperature, y0);
      }
      return state;
    }
    const Real minimum_temperature =
        fmax(CommonMinimumTemperature(), temperature_floor);
    const Real maximum_temperature = CommonMaximumTemperature();
    MaterialThermodynamicState state = StateFromRhoTemperatures(
        density, minimum_temperature, minimum_temperature, y0);
    if (state.ion_pressure+state.electron_pressure >= pressure_floor) return state;
    MaterialThermodynamicState maximum = StateFromRhoTemperatures(
        density, maximum_temperature, maximum_temperature, y0);
    if (maximum.ion_pressure+maximum.electron_pressure < pressure_floor) {
      if (material0_table.bounds_error != 0 || material1_table.bounds_error != 0) {
        Kokkos::abort("Mixed IONMIX pressure floor is above the table range.");
      }
      maximum.query_flags |= ionmix_temperature_above_table;
      return maximum;
    }
    Real log_low = log(minimum_temperature);
    Real log_high = log(maximum_temperature);
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
    return StateFromRhoTemperatures(
        density, temperature, temperature, y0);
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
  Real ExchangeTime(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*material0.zbar/material0.abar;
    const Real ne1 = (1.0-y0)*material1.zbar/material1.abar;
    if ((ne0 > 0.0 && material0.t_ei == 0.0) ||
        (ne1 > 0.0 && material1.t_ei == 0.0)) return 0.0;
    Real rate = 0.0;
    if (material0.t_ei > 0.0) rate += ne0/material0.t_ei;
    if (material1.t_ei > 0.0) rate += ne1/material1.t_ei;
    return (rate > 0.0) ? (ne0+ne1)/rate : -1.0;
  }
};

class MaterialMixture {
 public:
  MaterialMixture(ParameterInput *pin, int first_user_scalar, int nuser_scalars,
                  Real gamma, units::Units *unit_system = nullptr);
  ~MaterialMixture() = default;

  MaterialMixtureDevice DeviceData() const { return data_; }
  int ScalarIndex() const { return data_.scalar_index; }
  bool UsesTabularEOS() const { return data_.use_tabular_eos; }

 private:
  MaterialMixtureDevice data_;
  std::unique_ptr<IonmixTwoTemperatureTable> material0_table_;
  std::unique_ptr<IonmixTwoTemperatureTable> material1_table_;
};

} // namespace materials

#endif // MATERIALS_MATERIAL_MIXTURE_HPP_
