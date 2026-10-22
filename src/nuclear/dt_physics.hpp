#ifndef NUCLEAR_DT_PHYSICS_HPP_
#define NUCLEAR_DT_PHYSICS_HPP_
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
// D-T Maxwellian reactivity and classical test-alpha collision coefficients.
// References, approximations, and units: docs/DT_FUSION.md.

#include <cmath>
#include "athena.hpp"

namespace nuclear {
// Physical arithmetic is double even in a single-precision hydro build. Products of
// CGS charges/masses underflow float, and number-density products overflow it.
constexpr double kAmu = 1.660538921e-24;  // consistent with AthenaK material tables
constexpr double kBoltzmann = 1.3806488e-16;
constexpr double kKev = 1.602176634e-9;
constexpr double kAlphaEnergy = 3500.0*kKev;
constexpr double kNeutronEnergy = 14100.0*kKev;
constexpr double kAlphaMass = 4.0*kAmu;
constexpr double kElectronMass = 9.1093837015e-28;
constexpr double kPi = 3.14159265358979323846;

// Bosch & Hale (1992), Eq. (12), Table VII, T(d,n)4He. Output cm^3/s.
// Below the fit's 0.2 keV lower bound burning is explicitly disabled; callers must
// reject T > 100 keV rather than silently extrapolating the fit through its poles.
KOKKOS_INLINE_FUNCTION
double DTReactivity(double t) {
  if (t < 0.2) return 0.0;
  if (!Kokkos::isfinite(t) || t > 100.0) return -1.0;
  const double theta = t/(1.0-t*(1.51361e-2+t*(4.60643e-3-t*1.06750e-4))/
      (1.0+t*(7.51886e-2+t*(1.35000e-2+t*1.36600e-5))));
  const double xi = cbrt(34.3827*34.3827/(4.0*theta));
  return 1.17302e-9*theta*sqrt(xi/(1124656.0*t*t*t))*exp(-3.0*xi);
}

// Exact extent of dD/dt=dT/dt=-k*D*T for any nonnegative frozen k*dt.
// The expm1 form remains accurate for equal fuels, tiny burns, and exhausted fuels.
KOKKOS_INLINE_FUNCTION
double BurnExtent(double nd, double nt, double exposure) {
  const double a = fmin(nd, nt), b = fmax(nd, nt), difference = b-a;
  if (!(a > 0.0 && exposure > 0.0)) return 0.0;
  if (difference == 0.0) {
    const double tau = a*exposure;
    return tau > 1.0 ? a/(1.0+1.0/tau) : a*(tau/(1.0+tau));
  }
  const double z = -expm1(-difference*exposure);
  return a*z/(difference/b+(a/b)*z);
}

struct AlphaCollision {
  double loss = 0.0;        // erg/s per alpha
  double deflection = 0.0;  // 1/s; <cos theta> = exp(-nu_D dt)
};

// Maxwellian test-particle coefficients from the NRL Plasma Formulary, "Collisions".
// nu_E=2*nu_s-nu_perp-nu_parallel; nu_D=nu_perp/2. This retains the finite
// target/projectile mass correction. Energy diffusion is not sampled (CSDA).
KOKKOS_INLINE_FUNCTION
AlphaCollision AlphaTargetCollision(double energy, double density, double mass,
                                     double charge, double temperature,
                                     double coulomb_log) {
  AlphaCollision result;
  if (!(energy > 0.0 && density > 0.0)) return result;
  const double speed = sqrt(2.0*energy/kAlphaMass);
  const double x = temperature > 0.0 ?
      mass*energy/(kAlphaMass*kBoltzmann*temperature) : 1.0e100;
  const double root = sqrt(x);
  double psi, derivative, angular;
  if (x < 1.0e-3) {
    const double pref = 2.0/sqrt(kPi)*root;
    psi = pref*x*(2.0/3.0+x*(-2.0/5.0+x*(1.0/7.0-x/27.0)));
    derivative = pref*exp(-x);
    angular = psi-0.5*pref*(2.0/3.0+x*(-2.0/5.0+
        x*(1.0/7.0-x/27.0)))+derivative;
  } else {
    derivative = x < 700.0 ? 2.0/sqrt(kPi)*root*exp(-x) : 0.0;
    psi = erf(root)-derivative;
    angular = (1.0-0.5/x)*psi+derivative;
  }
  constexpr double e = 4.803204712570263e-10;
  const double factor = 4.0*kPi*4.0*charge*charge*e*e*e*e*density*coulomb_log;
  result.loss = factor/(mass*speed)*(psi-mass/kAlphaMass*derivative);
  result.deflection = factor/(kAlphaMass*kAlphaMass*speed*speed*speed)*angular;
  return result;
}
}  // namespace nuclear
#endif  // NUCLEAR_DT_PHYSICS_HPP_
