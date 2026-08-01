//========================================================================================
//! \file biermann_endpoint_closure_driver.cpp
//! \brief Device regression for the shared Biermann endpoint thermodynamic closure.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "two_temperature/biermann_closure.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(const Real actual, const Real expected,
                 const Real tolerance = 8.0e-14) {
  return Kokkos::isfinite(actual) &&
         fabs(actual-expected) <= tolerance*fmax(1.0, fabs(expected));
}

KOKKOS_INLINE_FUNCTION
two_temperature::BiermannEndpointClosure BaseClosure() {
  materials::MaterialMixtureDevice mixture;
  return {mixture, Real(2.0/3.0), Real(0.0), Real(0.0), Real(0.0),
          Real(0.0), Real(1.0e30), Real(1.0e-3), true, false, false};
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    Kokkos::parallel_reduce(
        "biermann_endpoint_closure_regression", Kokkos::RangePolicy<>(0, 5),
        KOKKOS_LAMBDA(const int test, int &failed) {
          auto closure = BaseClosure();
          two_temperature::BiermannClosedState state;
          if (test == 0) {
            // Reliable conservative subtraction wins and clamps Ue to Eint.
            state = closure.CloseConserved(
                1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 0.0,
                0.0, 0.0, 0.0);
            if (!NearlyEqual(state.density, 1.0) ||
                !NearlyEqual(state.internal_energy, 2.0) ||
                !NearlyEqual(state.ion_energy, 0.0) ||
                !NearlyEqual(state.electron_energy, 2.0)) ++failed;
          } else if (test == 1) {
            // Ill-conditioned subtraction uses max(Ui,0)+max(Ue,0), not max(Ui+Ue,0).
            const Real momentum = sqrt(2.0*999.5);
            state = closure.CloseConserved(
                1.0, momentum, 0.0, 0.0, 1000.0, -1.0, 2.0, 0.0,
                0.0, 0.0, 0.0);
            if (!NearlyEqual(state.internal_energy, 2.0) ||
                !NearlyEqual(state.ion_energy, 0.0) ||
                !NearlyEqual(state.electron_energy, 2.0)) ++failed;
          } else if (test == 2) {
            // Magnetization raises rho before kinetic energy and all hydro floors.
            closure.use_dual_energy = false;
            closure.sigma_max = 2.0;
            closure.pressure_floor = 1.0;
            closure.temperature_floor = 2.0;
            closure.entropy_floor = 0.5;
            state = closure.CloseConserved(
                0.1, 0.0, 0.0, 0.0, 2.0, 0.0, 10.0, 0.0,
                2.0, 0.0, 0.0);
            if (!NearlyEqual(state.density, 2.0) ||
                !NearlyEqual(state.internal_energy, 6.0) ||
                !NearlyEqual(state.ion_energy, 0.0) ||
                !NearlyEqual(state.electron_energy, 6.0)) ++failed;
          } else if (test == 3) {
            // Without dual energy the conservative internal state remains authoritative.
            closure.use_dual_energy = false;
            const Real momentum = sqrt(3.0);
            state = closure.CloseConserved(
                1.0, momentum, 0.0, 0.0, 2.0, 20.0, 2.0, 0.0,
                0.0, 0.0, 0.0);
            if (!NearlyEqual(state.internal_energy, 0.5) ||
                !NearlyEqual(state.ion_energy, 0.0) ||
                !NearlyEqual(state.electron_energy, 0.5)) ++failed;
          } else {
            // The shared selected-state closure uses mixed-material floors and composition.
            materials::MaterialMixtureDevice mixture;
            mixture.material0.abar = 6.5;
            mixture.material0.zbar = 3.5;
            mixture.material0.zeff = 3.5;
            mixture.material1.abar = 4.0;
            mixture.material1.zbar = 2.0;
            mixture.material1.zeff = 2.0;
            mixture.gamma_minus_one = 2.0/3.0;
            mixture.use_tabular_eos = false;
            closure.mixture = mixture;
            closure.use_materials = true;
            closure.pressure_floor = 1.0;
            closure.temperature_floor = 0.5;
            const Real density = 2.0;
            const Real y0 = 0.25;
            const auto floor = mixture.MinimumPressureEnergyState(
                density, y0, closure.pressure_floor, closure.temperature_floor);
            state = closure.CloseSelected(density, 0.0, 1.0e9, y0);
            const Real ion_floor = density*floor.ion_specific_internal_energy;
            const Real electron_floor =
                density*floor.electron_specific_internal_energy;
            if (!NearlyEqual(state.material0_mass_fraction, y0) ||
                !NearlyEqual(state.internal_energy, ion_floor+electron_floor) ||
                !NearlyEqual(state.ion_energy, ion_floor) ||
                !NearlyEqual(state.electron_energy, electron_floor)) ++failed;
          }
        }, failures);
    Kokkos::fence();
  }
  Kokkos::finalize();
  if (failures != 0) {
    std::cerr << "Biermann endpoint closure failed " << failures << " cases\n";
    return EXIT_FAILURE;
  }
  std::cout << "Biermann endpoint closure passed 5 cases\n";
  return EXIT_SUCCESS;
}
