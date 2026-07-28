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
    if (mode == "error_bounds") {
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
    if (mode == "error_bounds") {
      Kokkos::parallel_reduce(
          "ionmix_error_bounds", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int, int &local_failures) {
            const auto state = device.IonFromRhoTemperature(0.25, 1.0);
            if (state.pressure == state.pressure) ++local_failures;
          }, Kokkos::Sum<int>(failures));
      Kokkos::fence();
      std::cerr << "error bounds unexpectedly returned\n";
      return_code = EXIT_FAILURE;
    } else if (mode == "check" || mode == "mpi_check") {
      Kokkos::parallel_reduce(
          "ionmix_table_device_api", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(const int, int &local_failures) {
            constexpr Real tolerance = 2.0e-11;
            const Real sqrt_two = sqrt(2.0);

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
        materials::IonmixTwoTemperatureTableOptions linear_options = options;
        linear_options.geometric_interpolation = false;
        materials::IonmixTwoTemperatureTable linear_table(filename, linear_options);
        const auto linear_device = linear_table.DeviceData();
        int linear_failures = 0;
        Kokkos::parallel_reduce(
            "ionmix_linear_value_interpolation", Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int, int &local_failures) {
              const auto state = linear_device.IonFromRhoTemperature(1.0, 2.0);
              if (!NearlyEqual(state.pressure, 62.5, 2.0e-11)) {
                ++local_failures;
              }
            }, Kokkos::Sum<int>(linear_failures));
        if (linear_failures != 0 ||
            linear_table.Metadata().file_fingerprint_value !=
                metadata.file_fingerprint_value) {
          std::cerr << "linear interpolation check failed\n";
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
