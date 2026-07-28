//========================================================================================
//! \file tabular_material_mixture_driver.cpp
//! \brief Unequal-grid mixed IONMIX closure and material-LLF checks.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/rsolvers/material_llf_mhd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(const Real actual, const Real expected,
                 const Real tolerance = 3.0e-10) {
  return Kokkos::isfinite(actual) &&
         fabs(actual-expected) <= tolerance*fmax(1.0, fabs(expected));
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

    materials::MaterialMixtureDevice mixture;
    mixture.material0.abar = 6.5;
    mixture.material0.zbar = 2.0;
    mixture.material0.zeff = 6.0;
    mixture.material1.abar = 4.0;
    mixture.material1.zbar = 4.0;
    mixture.material1.zeff = 8.0;
    mixture.material0_table = ch.DeviceData();
    mixture.material1_table = he.DeviceData();
    mixture.use_tabular_eos = true;
    mixture.density_to_cgs = 1.0;
    mixture.temperature_to_kelvin = 1.0;
    mixture.wave_speed_safety = 1.05;

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

          const auto inverse = mixture.StateFromRhoSpecificEnergies(
              density, 250.0, 450.0, ych);
          if (!NearlyEqual(inverse.ion_temperature, 100.0) ||
              !NearlyEqual(inverse.electron_temperature, 100.0) ||
              !NearlyEqual(inverse.ion_pressure, 150.0) ||
              !NearlyEqual(inverse.electron_pressure, 350.0)) {
            ++local_failures;
          }

          const auto pure_ch_floor = mixture.MinimumState(density, 1.0);
          const auto pure_he_floor = mixture.MinimumState(density, 0.0);
          const auto mixed_floor = mixture.MinimumState(density, ych);
          if (!NearlyEqual(pure_ch_floor.ion_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.electron_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.ion_specific_internal_energy, 30.0) ||
              !NearlyEqual(pure_he_floor.ion_temperature, 20.0) ||
              !NearlyEqual(pure_he_floor.ion_specific_internal_energy, 40.0) ||
              !NearlyEqual(mixed_floor.ion_temperature, 20.0) ||
              !NearlyEqual(mixed_floor.ion_specific_internal_energy, 50.0) ||
              !NearlyEqual(mixed_floor.electron_specific_internal_energy, 90.0)) {
            ++local_failures;
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
                  density, exchange_total, old_exchange_state.ion_temperature,
                  old_exchange_state.electron_temperature,
                  target_difference, ych);
          const Real expected_tion =
              (exchange_total-4.5*target_difference)/7.0;
          if (!NearlyEqual(exchange.thermodynamics.ion_temperature,
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
