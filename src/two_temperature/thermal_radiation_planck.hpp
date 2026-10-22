#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_PLANCK_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_PLANCK_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_planck.hpp
//! \brief Stable integrals of the Planck spectrum, including weak and narrow bands.

#include "athena.hpp"

namespace two_temperature {
namespace detail {
constexpr Real kPlanckIntegralInfinity = 6.4939394022668291491;  // pi^4/15

// Integral from zero to x, used only for 0 <= x < 2. The Bernoulli series
// converges for |x| < 2*pi; keeping sixteen even terms resolves the switch
// without the truncation discontinuity of the former expansion at x=0.5.
KOKKOS_INLINE_FUNCTION
Real PlanckLowIntegral(Real x) {
  const Real x2 = x*x;
  Real series = -1.64136876253491498436e-27;
  series = 6.87258318890206987973e-26 + x2*series;
  series = -2.88823142807662801626e-24 + x2*series;
  series = 1.21886449642395430058e-22 + x2*series;
  series = -5.16832025400463827434e-21 + x2*series;
  series = 2.20360113134409180608e-19 + x2*series;
  series = -9.45595086329592118714e-18 + x2*series;
  series = 4.08860097917992598292e-16 + x2*series;
  series = -1.78404226122241203517e-14 + x2*series;
  series = 7.87208031216745813696e-13 + x2*series;
  series = -3.52279342579166212323e-11 + x2*series;
  series = 1.60590438368216145994e-9 + x2*series;
  series = -7.51563251563251563252e-8 + x2*series;
  series = 3.67430922986478542034e-6 + x2*series;
  series = -1.98412698412698412698e-4 + x2*series;
  series = 1.66666666666666666667e-2 + x2*series;
  return x*x2*(1.0/3.0-x/8.0+x2*series);
}

// Integral from x to infinity. Evaluate the complementary series itself;
// forming pi^4/15 - tail and subtracting two such values destroys Wien bands.
KOKKOS_INLINE_FUNCTION
Real PlanckTailIntegral(Real x) {
  if (x >= 800.0) return 0.0;  // the integral is below double's subnormal range
  if (x >= 500.0) {
    const Real invx = 1.0/x;
    return exp(3.0*log(x)-x)*(1.0+invx*(3.0+invx*(6.0+6.0*invx)));
  }
  const Real q = exp(-x), x2 = x*x, x3 = x2*x;
  Real tail = 0.0, qn = q;
  for (int n = 1; n <= 64; ++n) {
    const Real invn = 1.0/static_cast<Real>(n), invn2 = invn*invn;
    const Real term = qn*(x3*invn+3.0*x2*invn2
                          +6.0*x*invn2*invn+6.0*invn2*invn2);
    tail += term;
    if (term <= 1.0e-17*tail) break;
    qn *= q;
  }
  return tail;
}

// Eight-point Gauss-Legendre quadrature for narrow intervals (width <= 0.25).
// This avoids cancellation even when the two tails or small-x integrals agree
// to many digits. expm1 also resolves the Rayleigh-Jeans limit near zero.
KOKKOS_INLINE_FUNCTION
Real PlanckNarrowIntegral(Real lower, Real width) {
  constexpr Real nodes[] = {0.183434642495649805, 0.525532409916328986,
                             0.796666477413626740, 0.960289856497536232};
  constexpr Real weights[] = {0.362683783378361983, 0.313706645877887287,
                               0.222381034453374471, 0.101228536290376259};
  const Real half_width = 0.5*width;
  Real sum = 0.0;
  for (int n = 0; n < 4; ++n) {
    for (int sign = -1; sign <= 1; sign += 2) {
      const Real x = lower+half_width*(1.0+sign*nodes[n]);
      const Real spectrum = x >= 800.0 ? 0.0 : (x >= 500.0
          ? exp(3.0*log(x)-x) : x*x*x*exp(-x)/(-expm1(-x)));
      sum += weights[n]*spectrum;
    }
  }
  return half_width*sum;
}

// Reuse each upper endpoint as the next band's lower endpoint. The cached
// value is the low-x integral below x=2 and the complementary tail above it.
struct PlanckSpectrum {
  Real temperature, lower_bound, lower, value;

  KOKKOS_INLINE_FUNCTION
  PlanckSpectrum(Real bound, Real temp)
      : temperature(temp), lower_bound(fmax(bound, 0.0)) {
    lower = temp > 0.0 ? fmax(lower_bound/temp, 0.0) : 0.0;
    value = lower < 2.0 ? PlanckLowIntegral(lower) : PlanckTailIntegral(lower);
  }

  KOKKOS_INLINE_FUNCTION
  Real Fraction(Real upper_bound) {
    if (!(temperature > 0.0)) return 0.0;
    const Real upper = fmax(upper_bound/temperature, 0.0);
    // Subtract photon bounds before scaling: subtracting two separately rounded
    // ratios can lose the relative width of a very narrow band.
    const Real width = (fmax(upper_bound, 0.0)-lower_bound)/temperature;
    const Real next = upper < 2.0
        ? PlanckLowIntegral(upper) : PlanckTailIntegral(upper);
    Real integral = 0.0;
    if (width > 0.0) {
      if (width <= 0.01*fmin(fmax(lower, 1.0), 25.0)) {
        integral = PlanckNarrowIntegral(lower, width);
      } else if (lower >= 2.0) {
        integral = value-next;
      } else if (upper < 2.0) {
        integral = next-value;
      } else {
        integral = (kPlanckIntegralInfinity-next)-value;
      }
    }
    lower = upper;
    lower_bound = fmax(upper_bound, 0.0);
    value = next;
    return fmin(fmax(integral/kPlanckIntegralInfinity, 0.0), 1.0);
  }
};

KOKKOS_INLINE_FUNCTION
Real PlanckGroupFraction(Real lower_bound, Real upper_bound, Real temperature) {
  PlanckSpectrum spectrum(lower_bound, temperature);
  return spectrum.Fraction(upper_bound);
}
}  // namespace detail
}  // namespace two_temperature
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_PLANCK_HPP_
