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
    options.geometric_interpolation = false;
    materials::IonmixTwoTemperatureTable nonlinear_ch(argv[1], options);
    materials::IonmixTwoTemperatureTable nonlinear_he(argv[2], options);

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
    materials::MaterialMixtureDevice nonlinear_mixture = mixture;
    nonlinear_mixture.material0_table = nonlinear_ch.DeviceData();
    nonlinear_mixture.material1_table = nonlinear_he.DeviceData();
    materials::MaterialMixtureDevice clamped_mixture = mixture;
    clamped_mixture.material0_table.bounds_error = 0;
    clamped_mixture.material1_table.bounds_error = 0;

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
          const auto mixed_no_sound = mixture.StateFromRhoTemperaturesNoSound(
              density, 100.0, 100.0, ych);
          const auto inverse_no_sound = mixture.StateFromRhoSpecificEnergiesNoSound(
              density, 250.0, 450.0, ych);
          if (!NearlyEqual(inverse.ion_temperature, 100.0) ||
              !NearlyEqual(inverse.electron_temperature, 100.0) ||
              !NearlyEqual(inverse.ion_pressure, 150.0) ||
              !NearlyEqual(inverse.electron_pressure, 350.0) ||
              mixed_no_sound.sound_speed_squared != 0.0 ||
              inverse_no_sound.sound_speed_squared != 0.0 ||
              !(inverse.sound_speed_squared > 0.0)) {
            ++local_failures;
          }

          const auto pure_ch_floor = mixture.MinimumState(density, 1.0);
          const auto pure_he_floor = mixture.MinimumState(density, 0.0);
          const auto mixed_floor = mixture.MinimumState(density, ych);
          const auto mixed_floor_no_sound = mixture.MinimumStateNoSound(density, ych);
          if (!NearlyEqual(pure_ch_floor.ion_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.electron_temperature, 10.0) ||
              !NearlyEqual(pure_ch_floor.ion_specific_internal_energy, 30.0) ||
              !NearlyEqual(pure_he_floor.ion_temperature, 20.0) ||
              !NearlyEqual(pure_he_floor.ion_specific_internal_energy, 40.0) ||
              !NearlyEqual(mixed_floor.ion_temperature, 20.0) ||
              !NearlyEqual(mixed_floor.ion_specific_internal_energy, 50.0) ||
              !NearlyEqual(mixed_floor.electron_specific_internal_energy, 90.0) ||
              mixed_floor_no_sound.sound_speed_squared != 0.0 ||
              !(mixed_floor.sound_speed_squared > 0.0)) {
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
                  density,
                  old_exchange_state.ion_specific_internal_energy,
                  old_exchange_state.electron_specific_internal_energy,
                  old_exchange_state.ion_temperature,
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

          // A zero-width bracket with a nonzero residual is not convergence. Recovery
          // must decline the exchange and preserve the exact conservative split passed
          // by the caller, independent of the stale cached temperatures.
          const Real recovery_ion_energy = 125.0;
          const Real recovery_electron_energy = 900.0;
          const auto recovery =
              clamped_mixture.StateFromRhoTotalEnergyTemperatureDifference(
                  density, recovery_ion_energy, recovery_electron_energy,
                  2000.0, 2000.0, 0.0, ych);
          if (recovery.used_fallback != 2 || recovery.iterations != 0 ||
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
                  const Real nonlinear_total =
                      nonlinear_old.ion_specific_internal_energy+
                      nonlinear_old.electron_specific_internal_energy;
                  if (nonlinear_exchange.iterations <= 6 ||
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
