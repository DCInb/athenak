//========================================================================================
//! \file fast_electron_physics_driver.cpp
//! \brief Physical normalization, cold-target range, and device angular-diffusion tests.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "fast_electrons/fast_electron_physics.hpp"

namespace {

bool Close(const double actual, const double expected, const double tolerance) {
  return std::isfinite(actual) && std::abs(actual - expected) <= tolerance*expected;
}

int CheckRatesAndRange() {
  namespace fe = fast_electrons;
  // Independent SI references: n=1e29 m^-3, lnLambda=10, hydrogen (Zeff=1).
  // Computed using q_e^4/(4*pi*epsilon_0^2), rather than the CGS r_e prefactor.
  const double energies[] = {1.0e3, 1.0e5, 1.0e6};
  const double betas[] = {0.062469538918594, 0.548220861669677, 0.941079227918459};
  const double stopping[] = {392063.354594801, 44675.456008407, 26025.457009170};
  const double deflection[] = {2.44467493072048e14, 2.53989965897065e11,
                               8.21026687786248e9};
  int failures = 0;
  const double tolerance = sizeof(Real) == sizeof(float) ? 3.0e-6 : 3.0e-8;
  for (int n = 0; n < 3; ++n) {
    const Real energy = energies[n]*fe::kEvToErg;
    if (!Close(fe::Beta(energy), betas[n], tolerance) ||
        !Close(fe::StoppingRate(energy, 1.0e23, 10.0), stopping[n], tolerance) ||
        !Close(fe::DeflectionRate(energy, 1.0e23, 1.0, 10.0),
               deflection[n], tolerance)) ++failures;
    if (!Close(fe::StoppingRate(energy, 3.0e23, 20.0), 6.0*stopping[n], tolerance) ||
        !Close(fe::DeflectionRate(energy, 1.0e23, 5.0, 10.0),
               3.0*deflection[n], tolerance)) ++failures;
  }
  if (fe::Beta(0.0) != 0.0 || fe::StoppingRate(0.0, 1.0e23, 10.0) != 0.0 ||
      fe::StoppingRate(fe::kEvToErg, 0.0, 10.0) != 0.0 ||
      fe::DeflectionRate(fe::kEvToErg, 1.0e23, 1.0, 0.0) != 0.0) ++failures;
  // At nonrelativistic energies beta must retain precision even when 1+E/mc^2 rounds
  // to one. At ultrarelativistic energies it must not overflow a gamma^2 intermediate.
  if (!Close(fe::Beta(fe::kElectronRestEnergyCgs*1.0e-12), std::sqrt(2.0e-12),
             tolerance) ||
      !Close(fe::Beta(fe::kElectronRestEnergyCgs*1.0e20), 1.0, tolerance)) ++failures;

  // Independently known continuous-slowing-down path length: integrating beta^2 dgamma
  // gives R = [q0^2/(1+q0)-qf^2/(1+qf)]/(4*pi*r_e^2*n_e*lnLambda), q=E/(mc^2).
  // Integrate dx = v(E) dE / StoppingRate(E) numerically over 1 MeV -> 10 keV.
  const double initial = 1.0e6*fe::kEvToErg, final = 1.0e4*fe::kEvToErg;
  const double interval = (initial - final)/20000.0;
  double distance = 0.0;
  for (int n = 0; n < 20000; ++n) {
    const Real midpoint = final + (n + 0.5)*interval;
    distance += fe::kLightSpeedCgs*fe::Beta(midpoint)*interval/
                fe::StoppingRate(midpoint, 1.0e23, 10.0);
  }
  const double q0 = initial/fe::kElectronRestEnergyCgs;
  const double qf = final/fe::kElectronRestEnergyCgs;
  const double re = 2.8179403205e-13;
  const double range = (q0*q0/(1.0 + q0) - qf*qf/(1.0 + qf))/
      (4.0*std::acos(-1.0)*re*re*1.0e23*10.0);
  if (!Close(distance, range, tolerance)) ++failures;
  std::cout << "Cold-target path length: " << distance << " cm, analytic " << range
            << " cm; rate/range failures=" << failures << '\n';
  return failures;
}

struct ScatterSample {
  Real cosine, legendre2, transverse1, transverse2;
  int invalid;
};

int CheckScattering(const Real x, const Real y, const Real z,
                    const int steps, const Real nu_dt) {
  constexpr int samples = 65536;
  DvceArray1D<ScatterSample> results("fast-electron-scattering", samples);
  Kokkos::parallel_for("fast-electron-scattering-oracle",
      Kokkos::RangePolicy<>(0, samples),
      KOKKOS_LAMBDA(const int n) {
        std::uint64_t state = static_cast<std::uint64_t>(n)*UINT64_C(0xd1342543de82ef95);
        std::uint64_t duplicate = state;
        Real nx = x, ny = y, nz = z;
        Real dx = x, dy = y, dz = z;
        int invalid = 0;
        fast_electrons::ScatterDirection(0.0, state, nx, ny, nz);
        if (state != duplicate || nx != x || ny != y || nz != z) ++invalid;
        for (int step = 0; step < steps; ++step) {
          fast_electrons::ScatterDirection(nu_dt, state, nx, ny, nz);
          fast_electrons::ScatterDirection(nu_dt, duplicate, dx, dy, dz);
          const Real error = fabs(nx*nx + ny*ny + nz*nz - 1.0);
          if (!(error < 20.0*std::numeric_limits<Real>::epsilon()) ||
              nx != dx || ny != dy || nz != dz || state != duplicate) ++invalid;
        }
        const Real cosine = nx*x + ny*y + nz*z;
        Real tx, ty, tz;
        if (fabs(z) < 0.9) {
          const Real norm = sqrt(x*x + y*y);
          tx = -y/norm; ty = x/norm; tz = 0.0;
        } else {
          const Real norm = sqrt(y*y + z*z);
          tx = 0.0; ty = -z/norm; tz = y/norm;
        }
        results(n) = {cosine, 0.5*(3.0*cosine*cosine - 1.0),
                      nx*tx + ny*ty + nz*tz,
                      nx*(y*tz - z*ty) + ny*(z*tx - x*tz) + nz*(x*ty - y*tx),
                      invalid};
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  double mean = 0.0, p2 = 0.0, perpendicular1 = 0.0, perpendicular2 = 0.0;
  int failures = 0;
  for (int n = 0; n < samples; ++n) {
    mean += host(n).cosine/samples;
    p2 += host(n).legendre2/samples;
    perpendicular1 += host(n).transverse1/samples;
    perpendicular2 += host(n).transverse2/samples;
    failures += host(n).invalid;
  }
  const double elapsed = steps*nu_dt;
  const double expected = std::exp(-elapsed);
  const double tolerance = 6.0*(1.0 - expected)/std::sqrt(samples);
  if (std::abs(mean - expected) > tolerance ||
      std::abs(p2 - std::exp(-3.0*elapsed)) > 0.025 ||
      std::abs(perpendicular1) > 0.015 || std::abs(perpendicular2) > 0.015) ++failures;
  std::cout << "Angular diffusion: nu*t=" << elapsed << " <cos>=" << mean
            << " expected=" << expected << " <P2>=" << p2
            << " failures=" << failures << '\n';
  return failures;
}

int CheckBoris() {
  namespace fe = fast_electrons;
  DvceArray1D<Real> result("boris-oracles", 8);
  Kokkos::parallel_for("boris-oracles", Kokkos::RangePolicy<>(0, 1),
  KOKKOS_LAMBDA(int) {
    Real n[3] = {1.0, 0.0, 0.0};
    const Real zero[3] = {0.0, 0.0, 0.0}, b[3] = {0.0, 0.0, 1.0};
    Real energy = fe::kElectronRestEnergyCgs/(sqrt(2.0)+1.0);
    for (int t = 0; t < 5000; ++t) {
      fe::BorisKick(energy, n, zero, b, -1.0, 1.0, 0.002);
    }
    result(0) = energy/fe::kElectronRestEnergyCgs;
    result(1) = n[0]; result(2) = n[1]; result(3) = n[2];
    // In a constant parallel electric field the normalized momentum changes linearly,
    // independently of gamma: u_x(t)=2-t for q/(mc)=-1 and E=1.
    energy = 4.0*fe::kElectronRestEnergyCgs/(sqrt(5.0)+1.0);
    n[0] = 1.0; n[1] = 0.0; n[2] = 0.0;
    const Real e[3] = {1.0, 0.0, 0.0};
    for (int t = 0; t < 100; ++t) {
      fe::BorisKick(energy, n, e, zero, -1.0, 1.0, 0.01);
    }
    result(4) = energy/fe::kElectronRestEnergyCgs;
    result(5) = n[0]; result(6) = n[1]; result(7) = n[2];
  });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  const double q = 1.0/(std::sqrt(2.0)+1.0);
  const double tol = sizeof(Real) == sizeof(float) ? 0.002 : 2.0e-11;
  int failures = !Close(host(0), q, tol) + !Close(host(4), q, tol);
  const double phase = 10.0/std::sqrt(2.0);
  if (std::abs(host(1)-std::cos(phase)) > 2.0e-5 ||
      std::abs(host(2)-std::sin(phase)) > 2.0e-5 || host(3) != 0.0 ||
      std::abs(host(5)-1.0) > tol || host(6) != 0.0 || host(7) != 0.0) ++failures;
  std::cout << "Relativistic electric acceleration / magnetic gyration failures="
            << failures << '\n';
  return failures;
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = CheckRatesAndRange()+CheckBoris();
  {
    const Real inverse = 1.0/std::sqrt(29.0);
    const Real axes[][3] = {{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0},
                            {0.0, 0.0, 1.0}, {2.0*inverse, -3.0*inverse, 4.0*inverse}};
    for (const auto &axis : axes) {
      failures += CheckScattering(axis[0], axis[1], axis[2], 1, 0.1);
      failures += CheckScattering(axis[0], axis[1], axis[2], 40, 0.025);
    }
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
