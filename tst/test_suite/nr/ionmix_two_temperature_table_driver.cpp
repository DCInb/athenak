//========================================================================================
//! \file ionmix_two_temperature_table_driver.cpp
//! \brief Standalone CPU checks for the native two-temperature IONMIX table API.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(const Real actual, const Real expected, const Real tolerance) {
  return Kokkos::isfinite(actual) &&
         fabs(actual-expected) <= tolerance*fmax(1.0, fabs(expected));
}

KOKKOS_INLINE_FUNCTION
bool ExactTemperatureMatch(
    const materials::IonmixTemperatureState &temperature,
    const materials::IonmixComponentState &ion,
    const materials::IonmixComponentState &electron) {
  return temperature.temperature == ion.temperature &&
         temperature.temperature == electron.temperature &&
         temperature.query_flags == ion.query_flags &&
         temperature.query_flags == electron.query_flags;
}

KOKKOS_INLINE_FUNCTION
bool ExactComponentMatch(const materials::IonmixComponentState &actual,
                         const materials::IonmixComponentState &expected) {
  return actual.temperature == expected.temperature &&
         actual.pressure == expected.pressure &&
         actual.specific_internal_energy == expected.specific_internal_energy &&
         actual.query_flags == expected.query_flags;
}

KOKKOS_INLINE_FUNCTION
bool ExactPressureEnergyMatch(
    const materials::IonmixPressureEnergyState &actual,
    const materials::IonmixComponentState &ion,
    const materials::IonmixComponentState &electron) {
  return actual.ion_pressure == ion.pressure &&
         actual.electron_pressure == electron.pressure &&
         actual.ion_specific_internal_energy == ion.specific_internal_energy &&
         actual.electron_specific_internal_energy ==
             electron.specific_internal_energy &&
         actual.query_flags == (ion.query_flags | electron.query_flags);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: ionmix_two_temperature_table_driver MODE TABLE\n";
    return EXIT_FAILURE;
  }
  const std::string mode(argv[1]);
  const std::string filename(argv[2]);

#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);
  int return_code = EXIT_SUCCESS;
  {
    materials::IonmixTwoTemperatureTableOptions options;
    options.density_to_cgs = 2.0;
    options.temperature_to_kelvin = 100.0;
    options.pressure_from_cgs = 0.1;
    options.specific_energy_from_cgs = 0.01;
    if (mode == "error_bounds" || mode == "error_bounds_component" ||
        mode == "error_bounds_prepared_density") {
      options.bounds_policy = materials::IonmixBoundsPolicy::error;
    }

    materials::IonmixTwoTemperatureTable table(filename, options);
    const auto &metadata = table.Metadata();
    if (metadata.format_version != 1 || metadata.ndensity != 2 ||
        metadata.ntemperature != 3 || metadata.abar != 6.5 ||
        metadata.file_size == 0 || metadata.file_fingerprint_value == 0 ||
        metadata.file_fingerprint.rfind("fnv1a64:", 0) != 0 ||
        metadata.minimum_density_cgs != 1.0 ||
        metadata.maximum_density_cgs != 4.0 ||
        metadata.minimum_temperature_kelvin != 100.0 ||
        metadata.maximum_temperature_kelvin != 1600.0 ||
        !metadata.ion_energy_is_strictly_positive ||
        !metadata.electron_energy_is_strictly_positive) {
      std::cerr << "metadata check failed\n";
      return_code = EXIT_FAILURE;
    }

    const materials::IonmixTwoTemperatureTableDevice device = table.DeviceData();
    int failures = 0;
    if (mode == "error_bounds" || mode == "error_bounds_prepared_density") {
      Kokkos::parallel_reduce(
          "ionmix_prepared_density_error_bounds", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int, int &local_failures) {
            const auto density = device.PrepareDensityLocation(0.25);
            if (density.fraction == density.fraction) ++local_failures;
          }, Kokkos::Sum<int>(failures));
      Kokkos::fence();
      std::cerr << "prepared-density error bounds unexpectedly returned\n";
      // This mode is an expected-abort subprocess. Returning success makes the
      // caller's `assert not run_command(...)` fail if the device query did not abort.
    } else if (mode == "error_bounds_component") {
      // Retain a dedicated mode for the original component-query abort check. The
      // single-process regression invokes the prepared-density error mode above.
      Kokkos::parallel_reduce(
          "ionmix_component_error_bounds", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int, int &local_failures) {
            const auto state = device.IonFromRhoTemperature(0.25, 1.0);
            if (state.pressure == state.pressure) ++local_failures;
          }, Kokkos::Sum<int>(failures));
      Kokkos::fence();
      std::cerr << "component error bounds unexpectedly returned\n";
    } else if (mode == "check" || mode == "mpi_check") {
      if (device.minimum_temperature_round_trips_exactly != 1) {
        std::cerr << "fixture native minimum does not round-trip exactly\n";
        return_code = EXIT_FAILURE;
      }
      Kokkos::parallel_reduce(
          "ionmix_table_device_api", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int, int &local_failures) {
            constexpr Real tolerance = 2.0e-11;
            const Real sqrt_two = sqrt(2.0);

            // One prepared density token must reproduce every ordinary forward query
            // exactly when reused across temperatures and both component fields.
            const Real density_queries[] = {1.0, 0.5, 2.0, 0.25, 4.0};
            const int density_flags[] = {
                materials::ionmix_query_in_bounds,
                materials::ionmix_query_in_bounds,
                materials::ionmix_query_in_bounds,
                materials::ionmix_density_below_table,
                materials::ionmix_density_above_table};
            const Real temperature_queries[] = {2.0, 1.0, 8.0, 4.0, 0.5, 32.0};
            const int temperature_flags[] = {
                materials::ionmix_query_in_bounds,
                materials::ionmix_query_in_bounds,
                materials::ionmix_query_in_bounds,
                materials::ionmix_query_in_bounds,
                materials::ionmix_temperature_below_table,
                materials::ionmix_temperature_above_table};
            for (int idensity = 0; idensity < 5; ++idensity) {
              const auto prepared =
                  device.PrepareDensityLocation(density_queries[idensity]);
              if (prepared.query_flags != density_flags[idensity]) {
                ++local_failures;
              }
              const Real minimum_temperature = device.MinimumTemperatureCode();
              const auto native_minimum =
                  device.PressureEnergyFromRhoMinimumTemperature(
                      density_queries[idensity]);
              const auto generic_minimum =
                  device.PressureEnergyFromRhoTemperature(
                      density_queries[idensity], minimum_temperature);
              if (native_minimum.ion_pressure != generic_minimum.ion_pressure ||
                  native_minimum.electron_pressure !=
                      generic_minimum.electron_pressure ||
                  native_minimum.ion_specific_internal_energy !=
                      generic_minimum.ion_specific_internal_energy ||
                  native_minimum.electron_specific_internal_energy !=
                      generic_minimum.electron_specific_internal_energy ||
                  native_minimum.query_flags != generic_minimum.query_flags ||
                  native_minimum.query_flags != density_flags[idensity]) {
                ++local_failures;
              }
              for (int itemperature = 0; itemperature < 6; ++itemperature) {
                const auto paired = device.PressureEnergyFromRhoTemperature(
                    density_queries[idensity],
                    temperature_queries[itemperature]);
                const auto ordinary_ion = device.IonFromRhoTemperature(
                    density_queries[idensity],
                    temperature_queries[itemperature]);
                const auto ordinary_electron = device.ElectronFromRhoTemperature(
                    density_queries[idensity],
                    temperature_queries[itemperature]);
                if (!ExactPressureEnergyMatch(
                        paired, ordinary_ion, ordinary_electron)) {
                  ++local_failures;
                }
                for (int icomponent = 0; icomponent < 2; ++icomponent) {
                  const auto component = (icomponent == 0)
                      ? materials::IonmixComponent::ion
                      : materials::IonmixComponent::electron;
                  const auto reused =
                      device.ComponentFromPreparedDensityTemperature(
                          component, prepared,
                          temperature_queries[itemperature]);
                  const auto ordinary = device.ComponentFromRhoTemperature(
                      component, density_queries[idensity],
                      temperature_queries[itemperature]);
                  const int expected_flags = density_flags[idensity] |
                                             temperature_flags[itemperature];
                  if (!ExactComponentMatch(reused, ordinary) ||
                      reused.query_flags != expected_flags) {
                    ++local_failures;
                  }
                }
              }
            }

            const auto interior_temperature =
                device.TemperatureFromRhoTemperature(1.0, 2.0);
            const auto interior_ion = device.IonFromRhoTemperature(1.0, 2.0);
            const auto interior_electron =
                device.ElectronFromRhoTemperature(1.0, 2.0);
            if (!ExactTemperatureMatch(
                    interior_temperature, interior_ion, interior_electron) ||
                interior_temperature.query_flags !=
                    materials::ionmix_query_in_bounds) {
              ++local_failures;
            }

            for (int itemperature = 0;
                 itemperature < device.ntemperature; ++itemperature) {
              const Real node = device.TemperatureCodeAtIndex(itemperature);
              const auto node_temperature =
                  device.TemperatureFromRhoTemperature(1.0, node);
              const auto node_ion = device.IonFromRhoTemperature(1.0, node);
              const auto node_electron =
                  device.ElectronFromRhoTemperature(1.0, node);
              if (!ExactTemperatureMatch(
                      node_temperature, node_ion, node_electron)) {
                ++local_failures;
              }
            }

            const int expected_low_coordinate_flags =
                materials::ionmix_density_below_table |
                materials::ionmix_temperature_below_table;
            const auto low_temperature =
                device.TemperatureFromRhoTemperature(0.25, 0.5);
            const auto low_ion = device.IonFromRhoTemperature(0.25, 0.5);
            const auto low_electron =
                device.ElectronFromRhoTemperature(0.25, 0.5);
            if (!ExactTemperatureMatch(low_temperature, low_ion, low_electron) ||
                low_temperature.query_flags != expected_low_coordinate_flags) {
              ++local_failures;
            }

            const int expected_high_coordinate_flags =
                materials::ionmix_density_above_table |
                materials::ionmix_temperature_above_table;
            const auto high_temperature =
                device.TemperatureFromRhoTemperature(4.0, 32.0);
            const auto high_ion = device.IonFromRhoTemperature(4.0, 32.0);
            const auto high_electron =
                device.ElectronFromRhoTemperature(4.0, 32.0);
            if (!ExactTemperatureMatch(
                    high_temperature, high_ion, high_electron) ||
                high_temperature.query_flags != expected_high_coordinate_flags) {
              ++local_failures;
            }

            const auto first_forward_ion =
                device.IonFromRhoTemperature(1.0, 8.0);
            const auto second_forward_ion_temperature =
                device.TemperatureFromRhoTemperature(
                    1.0, first_forward_ion.temperature);
            const auto second_forward_ion = device.IonFromRhoTemperature(
                1.0, first_forward_ion.temperature);
            const auto second_forward_electron_at_ion =
                device.ElectronFromRhoTemperature(
                    1.0, first_forward_ion.temperature);
            if (!ExactTemperatureMatch(
                    second_forward_ion_temperature, second_forward_ion,
                    second_forward_electron_at_ion)) {
              ++local_failures;
            }

            const auto first_forward_electron =
                device.ElectronFromRhoTemperature(1.0, 2.0);
            const auto second_forward_electron_temperature =
                device.TemperatureFromRhoTemperature(
                    1.0, first_forward_electron.temperature);
            const auto second_forward_ion_at_electron =
                device.IonFromRhoTemperature(
                    1.0, first_forward_electron.temperature);
            const auto second_forward_electron =
                device.ElectronFromRhoTemperature(
                    1.0, first_forward_electron.temperature);
            if (!ExactTemperatureMatch(
                    second_forward_electron_temperature,
                    second_forward_ion_at_electron,
                    second_forward_electron)) {
              ++local_failures;
            }

            const auto state = device.StateFromRhoTemperatures(1.0, 8.0, 2.0);
            if (!NearlyEqual(state.ion.temperature, 8.0, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(state.ion.pressure, 160.0, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(
                    state.ion.specific_internal_energy,
                    0.4*sqrt_two, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(state.electron.temperature, 2.0, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(state.electron.pressure, 80.0, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(
                    state.electron.specific_internal_energy,
                    0.2*sqrt_two, tolerance)) {
              ++local_failures;
            }
            if (!NearlyEqual(state.mean_ionization,
                             0.5*sqrt_two, tolerance)) {
              ++local_failures;
            }
            if (state.query_flags != materials::ionmix_query_in_bounds) {
              ++local_failures;
            }

            // The ion floor is a two-point plateau.  Its inverse is defined as the
            // lowest-temperature endpoint, not a division by the zero energy slope.
            const auto plateau = device.IonFromRhoSpecificEnergy(1.0, 0.2);
            if (!NearlyEqual(plateau.temperature, 1.0, tolerance) ||
                !NearlyEqual(plateau.pressure, 20.0, tolerance) ||
                !NearlyEqual(plateau.specific_internal_energy, 0.2, tolerance)) {
              ++local_failures;
            }

            const auto ion_round_trip =
                device.IonFromRhoSpecificEnergy(1.0, 0.4*sqrt_two);
            const auto electron_round_trip =
                device.ElectronFromRhoSpecificEnergy(1.0, 0.2*sqrt_two);
            if (!NearlyEqual(ion_round_trip.temperature, 8.0, tolerance) ||
                !NearlyEqual(ion_round_trip.pressure, 160.0, tolerance) ||
                !NearlyEqual(electron_round_trip.temperature, 2.0, tolerance) ||
                !NearlyEqual(electron_round_trip.pressure, 80.0, tolerance)) {
              ++local_failures;
            }

            const auto inverse_ion_temperature =
                device.TemperatureFromRhoTemperature(
                    1.0, ion_round_trip.temperature);
            const auto inverse_ion_forward = device.IonFromRhoTemperature(
                1.0, ion_round_trip.temperature);
            const auto inverse_electron_forward_at_ion =
                device.ElectronFromRhoTemperature(
                    1.0, ion_round_trip.temperature);
            if (!ExactTemperatureMatch(
                    inverse_ion_temperature, inverse_ion_forward,
                    inverse_electron_forward_at_ion)) {
              ++local_failures;
            }

            const auto inverse_electron_temperature =
                device.TemperatureFromRhoTemperature(
                    1.0, electron_round_trip.temperature);
            const auto inverse_ion_forward_at_electron =
                device.IonFromRhoTemperature(
                    1.0, electron_round_trip.temperature);
            const auto inverse_electron_forward =
                device.ElectronFromRhoTemperature(
                    1.0, electron_round_trip.temperature);
            if (!ExactTemperatureMatch(
                    inverse_electron_temperature,
                    inverse_ion_forward_at_electron,
                    inverse_electron_forward)) {
              ++local_failures;
            }

            const auto inverse = device.StateFromRhoSpecificEnergies(
                1.0, 0.2, 0.2*sqrt_two);
            if (!NearlyEqual(inverse.ion.temperature, 1.0, tolerance) ||
                !NearlyEqual(inverse.electron.temperature, 2.0, tolerance) ||
                !NearlyEqual(inverse.mean_ionization,
                             0.5*sqrt_two, tolerance)) {
              ++local_failures;
            }

            const auto coordinate_clamp =
                device.IonFromRhoTemperature(0.25, 0.5);
            const int expected_coordinate_flags =
                materials::ionmix_density_below_table |
                materials::ionmix_temperature_below_table;
            if (coordinate_clamp.query_flags != expected_coordinate_flags ||
                !NearlyEqual(coordinate_clamp.temperature, 1.0, tolerance) ||
                !NearlyEqual(coordinate_clamp.pressure, 10.0, tolerance) ||
                !NearlyEqual(coordinate_clamp.specific_internal_energy,
                             0.1, tolerance)) {
              ++local_failures;
            }

            const auto low_energy = device.IonFromRhoSpecificEnergy(1.0, 0.01);
            if (low_energy.query_flags != materials::ionmix_energy_below_table ||
                !NearlyEqual(low_energy.temperature, 1.0, tolerance) ||
                !NearlyEqual(low_energy.specific_internal_energy, 0.2, tolerance)) {
              ++local_failures;
            }
            const auto high_energy =
                device.ElectronFromRhoSpecificEnergy(1.0, 100.0);
            if (high_energy.query_flags != materials::ionmix_energy_above_table ||
                !NearlyEqual(high_energy.temperature, 16.0, tolerance) ||
                !NearlyEqual(high_energy.specific_internal_energy,
                             1.6*sqrt_two, tolerance)) {
              ++local_failures;
            }
          }, Kokkos::Sum<int>(failures));
      if (failures != 0) {
        std::cerr << failures << " device API checks failed\n";
        return_code = EXIT_FAILURE;
      }

      if (mode == "check") {
        materials::IonmixTwoTemperatureTableOptions extrapolate_options = options;
        extrapolate_options.bounds_policy =
            materials::IonmixBoundsPolicy::flash_extrapolate;
        materials::IonmixTwoTemperatureTable extrapolate_table(
            filename, extrapolate_options);
        const auto extrapolate_device = extrapolate_table.DeviceData();
        int extrapolate_failures = 0;
        Kokkos::parallel_reduce(
            "ionmix_flash_high_temperature_continuation",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int, int &local_failures) {
              constexpr Real tolerance = 3.0e-11;
              constexpr Real density = 1.0;
              constexpr Real native_temperature = 16.0;
              constexpr Real continued_temperature = 32.0;
              const auto endpoint = extrapolate_device.StateFromRhoTemperatures(
                  density, native_temperature, native_temperature);
              const auto continued = extrapolate_device.StateFromRhoTemperatures(
                  density, continued_temperature, continued_temperature);
              const auto temperature =
                  extrapolate_device.TemperatureFromRhoTemperature(
                      density, continued_temperature);
              const auto paired =
                  extrapolate_device.PressureEnergyFromRhoTemperature(
                      density, continued_temperature);
              const auto ion_inverse =
                  extrapolate_device.IonFromRhoSpecificEnergy(
                      density, continued.ion.specific_internal_energy);
              const auto electron_inverse =
                  extrapolate_device.ElectronFromRhoSpecificEnergy(
                      density, continued.electron.specific_internal_energy);
              const auto prepared =
                  extrapolate_device.PrepareDensityLocation(density);
              materials::IonmixEnergyIntervalCache cache;
              const Real cached_ion_energy =
                  extrapolate_device.ComponentEnergyFromPreparedDensityTemperature(
                      materials::IonmixComponent::ion, prepared,
                      continued_temperature, log(continued_temperature), cache);

              if (!NearlyEqual(temperature.temperature,
                               continued_temperature, tolerance) ||
                  temperature.query_flags != materials::ionmix_query_in_bounds ||
                  continued.query_flags != materials::ionmix_query_in_bounds ||
                  !ExactPressureEnergyMatch(
                      paired, continued.ion, continued.electron) ||
                  !(continued.ion.pressure > endpoint.ion.pressure) ||
                  !(continued.electron.pressure > endpoint.electron.pressure) ||
                  !(continued.ion.specific_internal_energy >
                    endpoint.ion.specific_internal_energy) ||
                  !(continued.electron.specific_internal_energy >
                    endpoint.electron.specific_internal_energy) ||
                  continued.mean_ionization != endpoint.mean_ionization ||
                  !NearlyEqual(cached_ion_energy,
                               continued.ion.specific_internal_energy,
                               tolerance) ||
                  !NearlyEqual(ion_inverse.temperature,
                               continued_temperature, tolerance) ||
                  !NearlyEqual(ion_inverse.pressure,
                               continued.ion.pressure, tolerance) ||
                  !NearlyEqual(ion_inverse.specific_internal_energy,
                               continued.ion.specific_internal_energy,
                               tolerance) ||
                  ion_inverse.query_flags != materials::ionmix_query_in_bounds ||
                  !NearlyEqual(electron_inverse.temperature,
                               continued_temperature, tolerance) ||
                  !NearlyEqual(electron_inverse.pressure,
                               continued.electron.pressure, tolerance) ||
                  !NearlyEqual(electron_inverse.specific_internal_energy,
                               continued.electron.specific_internal_energy,
                               tolerance) ||
                  electron_inverse.query_flags !=
                      materials::ionmix_query_in_bounds) {
                ++local_failures;
              }
            }, Kokkos::Sum<int>(extrapolate_failures));
        if (extrapolate_failures != 0) {
          std::cerr << extrapolate_failures
                    << " high-temperature continuation checks failed\n";
          return_code = EXIT_FAILURE;
        }

        materials::IonmixTwoTemperatureTableOptions linear_options = options;
        linear_options.geometric_interpolation = false;
        materials::IonmixTwoTemperatureTable linear_table(filename, linear_options);
        const auto linear_device = linear_table.DeviceData();
        int linear_failures = 0;
        Kokkos::parallel_reduce(
            "ionmix_linear_value_interpolation", Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int, int &local_failures) {
              const auto ion = linear_device.IonFromRhoTemperature(1.0, 2.0);
              const auto electron =
                  linear_device.ElectronFromRhoTemperature(1.0, 2.0);
              const auto paired =
                  linear_device.PressureEnergyFromRhoTemperature(1.0, 2.0);
              const auto native_minimum =
                  linear_device.PressureEnergyFromRhoMinimumTemperature(1.0);
              const auto generic_minimum =
                  linear_device.PressureEnergyFromRhoTemperature(
                      1.0, linear_device.MinimumTemperatureCode());
              if (!NearlyEqual(ion.pressure, 62.5, 2.0e-11) ||
                  !ExactPressureEnergyMatch(paired, ion, electron) ||
                  native_minimum.ion_pressure != generic_minimum.ion_pressure ||
                  native_minimum.electron_pressure !=
                      generic_minimum.electron_pressure ||
                  native_minimum.ion_specific_internal_energy !=
                      generic_minimum.ion_specific_internal_energy ||
                  native_minimum.electron_specific_internal_energy !=
                      generic_minimum.electron_specific_internal_energy ||
                  native_minimum.query_flags != generic_minimum.query_flags) {
                ++local_failures;
              }
            }, Kokkos::Sum<int>(linear_failures));
        if (linear_failures != 0 ||
            linear_table.Metadata().file_fingerprint_value !=
                metadata.file_fingerprint_value) {
          std::cerr << "linear interpolation check failed\n";
          return_code = EXIT_FAILURE;
        }

        // Extreme but valid unit scales can perturb the exp/log minimum-temperature
        // round trip. The native-minimum API must then fall back to the generic query.
        materials::IonmixTwoTemperatureTableOptions inexact_options = options;
        inexact_options.temperature_to_kelvin = 1.0e100;
        materials::IonmixTwoTemperatureTable inexact_table(
            filename, inexact_options);
        const auto inexact_device = inexact_table.DeviceData();
        int inexact_failures = 0;
        if (inexact_device.minimum_temperature_round_trips_exactly != 0) {
          std::cerr << "inexact native minimum unexpectedly round-trips exactly\n";
          return_code = EXIT_FAILURE;
        }
        Kokkos::parallel_reduce(
            "ionmix-inexact-native-minimum", Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int, int &local_failures) {
              const auto native_minimum =
                  inexact_device.PressureEnergyFromRhoMinimumTemperature(1.0);
              const auto generic_minimum =
                  inexact_device.PressureEnergyFromRhoTemperature(
                      1.0, inexact_device.MinimumTemperatureCode());
              if (native_minimum.ion_pressure != generic_minimum.ion_pressure ||
                  native_minimum.electron_pressure !=
                      generic_minimum.electron_pressure ||
                  native_minimum.ion_specific_internal_energy !=
                      generic_minimum.ion_specific_internal_energy ||
                  native_minimum.electron_specific_internal_energy !=
                      generic_minimum.electron_specific_internal_energy ||
                  native_minimum.query_flags != generic_minimum.query_flags) {
                ++local_failures;
              }
            }, Kokkos::Sum<int>(inexact_failures));
        if (inexact_failures != 0) {
          std::cerr << "inexact native-minimum fallback check failed\n";
          return_code = EXIT_FAILURE;
        }
      }
    } else if (mode != "load_only") {
      std::cerr << "unknown mode: " << mode << "\n";
      return_code = EXIT_FAILURE;
    }
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return return_code;
}
