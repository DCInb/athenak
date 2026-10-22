//========================================================================================
//! \file tabular_hllc_driver.cpp
//! \brief Device tests of EOS-independent HLLC flux identities and LLF fallback.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "hydro/rsolvers/material_hllc_hyd.hpp"

namespace {
KOKKOS_INLINE_FUNCTION
bool Near(const Real a, const Real b) {
  return Kokkos::isfinite(a) && Kokkos::isfinite(b) &&
         fabs(a-b) <= 128.0*Kokkos::Experimental::epsilon<Real>::value*
                      fmax(1.0, fmax(fabs(a), fabs(b)));
}

KOKKOS_INLINE_FUNCTION
bool SameFlux(const HydCons1D &a, const HydCons1D &b) {
  return Near(a.d, b.d) && Near(a.mx, b.mx) && Near(a.my, b.my) &&
         Near(a.mz, b.mz) && Near(a.e, b.e);
}

KOKKOS_INLINE_FUNCTION
HydCons1D PhysicalFlux(const HydPrim1D &w, const Real pressure) {
  const Real mass = w.d*w.vx;
  const Real energy = w.e+0.5*w.d*(SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
  return {mass, mass*w.vx+pressure, mass*w.vy, mass*w.vz,
          (energy+pressure)*w.vx};
}
} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    constexpr int ntests = 12;
    const char *names[ntests] = {
      "uniform non-gamma-law state", "stationary contact and shear",
      "right-moving contact", "left-moving contact", "supersonic right",
      "supersonic left", "energy reference shift", "reflection symmetry",
      "strong converging flow", "rarefaction fallback", "zero-speed fallback",
      "negative star internal-energy fallback"};
    DvceArray1D<int> passed("hllc_checks", ntests);
    Kokkos::parallel_for("hllc_identities", ntests, KOKKOS_LAMBDA(const int n) {
      HydPrim1D left{1.7, 0.2, 0.7, -0.4, 12.3};
      HydPrim1D right{0.13, -0.1, -0.2, 0.3, 0.4};
      HydCons1D flux, expected;
      bool ok = false;
      if (n == 0) {
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, left, 0.7, 0.7, 0.83, 0.83, flux) &&
             SameFlux(flux, PhysicalFlux(left, 0.7));
      } else if (n <= 3) {
        left.vx = right.vx = (n == 1) ? 0.0 : ((n == 2) ? 0.3 : -0.3);
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 0.7, 0.7, 0.83, 2.1, flux) &&
             SameFlux(flux, PhysicalFlux((n <= 2) ? left : right, 0.7));
      } else if (n <= 5) {
        left.vx = right.vx = (n == 4) ? 10.0 : -10.0;
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 0.7, 0.2, 0.83, 2.1, flux) &&
             SameFlux(flux, PhysicalFlux((n == 4) ? left : right,
                                        (n == 4) ? 0.7 : 0.2));
      } else if (n == 6) {
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 0.7, 0.2, 0.83, 2.1, expected);
        // Changing the material's specific energy zero must transport that offset
        // with mass, without changing pressure, momentum, or the wave speeds.
        left.e += 25.0*left.d;
        right.e += 25.0*right.d;
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 0.7, 0.2, 0.83, 2.1, flux) && ok;
        expected.e += 25.0*expected.d;
        ok = ok && SameFlux(flux, expected);
      } else if (n == 7) {
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 0.7, 0.2, 0.83, 2.1, expected);
        left.vx = -left.vx;
        right.vx = -right.vx;
        ok = hydro::SingleStateHLLC_HydMaterial(
            right, left, 0.2, 0.7, 2.1, 0.83, flux) && ok;
        expected.d = -expected.d;
        expected.my = -expected.my;
        expected.mz = -expected.mz;
        expected.e = -expected.e;
        ok = ok && SameFlux(flux, expected);
      } else if (n == 8) {
        left = {1.0, 100.0, 0.0, 0.0, 2.0};
        right = {1.0, -100.0, 0.0, 0.0, 2.0};
        ok = hydro::SingleStateHLLC_HydMaterial(
            left, right, 1.0, 1.0, 1.0, 1.0, flux) &&
             Near(flux.d, 0.0) && Near(flux.e, 0.0) &&
             flux.mx > 10000.0 && Kokkos::isfinite(flux.mx);
      } else {
        const Real speed = (n == 9) ? 20.0 : ((n == 10) ? 0.0 : 0.1);
        const Real energy = (n == 11) ? 1.0e-5 : 2.0;
        const Real cs2 = (n == 10) ? 0.0 : 1.0;
        left = {1.0, -speed, 0.0, 0.0, energy};
        right = {1.0, speed, 0.0, 0.0, energy};
        ok = !hydro::SingleStateHLLC_HydMaterial(
            left, right, 1.0, 1.0, cs2, cs2, flux);
        hydro::SingleStateLLF_HydMaterial(
            left, right, 1.0, 1.0, cs2, cs2, expected);
        ok = ok && SameFlux(flux, expected);
      }
      passed(n) = ok ? 1 : 0;
    });
    const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), passed);
    for (int n = 0; n < ntests; ++n) {
      std::cout << (host(n) ? "PASS " : "FAIL ") << names[n] << '\n';
      failures += (host(n) == 0);
    }
  }
  Kokkos::finalize();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
