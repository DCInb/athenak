#ifndef FAST_ELECTRONS_FAST_ELECTRON_PHYSICS_HPP_
#define FAST_ELECTRONS_FAST_ELECTRON_PHYSICS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fast_electron_physics.hpp
//! \brief Relativistic cold-target Coulomb drag and angular diffusion, in CGS.
//!
//! These are the fully ionized, suprathermal, leading-log limits of the collision
//! frequencies in Hesslow et al. (2018), Eqs. II.21 and II.29 and the discussion between:
//! https://arxiv.org/abs/1807.05036 . The normalized momentum there is p = gamma*beta.
//! A shared prescribed Coulomb logarithm is used for electron/electron and electron/ion
//! collisions. No bound-electron stopping, energy diffusion, ion recoil, bremsstrahlung,
//! or radiative loss is included in these collision rates.
//! Background temperatures must be well below the transported kinetic energy; the
//! transport operator must terminate packets at its prescribed thermalization cutoff.
//!
//! PIC references inspected for collision conventions and validation (no code copied):
//! EPOCH: https://github.com/epochpic/epoch , epoch3d/src/physics_packages/collisions.F90
//! Smilei: https://github.com/SmileiPIC/Smilei
//!         benchmarks/collisions/Stopping_power123.py

#include <cmath>
#include <cstdint>
#include <limits>

#include "athena.hpp"

namespace fast_electrons {

constexpr Real kElectronChargeCgs = 4.80320471257e-10; // statcoulomb
constexpr Real kPi = 3.14159265358979323846;
constexpr Real kElectronMassCgs = 9.1093837139e-28;       // g
constexpr Real kLightSpeedCgs = 2.99792458e10;           // cm/s
constexpr Real kElectronRestEnergyCgs = 8.1871057880e-7; // erg
constexpr Real kClassicalElectronRadiusCgs = 2.8179403205e-13; // cm
constexpr Real kBoltzmannCgs = 1.380649e-16;             // erg/K
constexpr Real kEvToErg = 1.602176634e-12;               // erg/eV
// Evaluate the prefactor in double precision to avoid intermediate underflow in float.
constexpr Real kCollisionCoefficientCgs =
    4.0*3.14159265358979323846*2.8179403205e-13*2.8179403205e-13*2.99792458e10;

KOKKOS_INLINE_FUNCTION
Real Beta(const Real energy_erg) {
  if (!(energy_erg > 0.0)) return 0.0;
  const Real kinetic = energy_erg/kElectronRestEnergyCgs;
  const Real gamma = 1.0 + kinetic;
  // Avoid both gamma^2 overflow and cancellation in 1-1/gamma^2 at low energy.
  return sqrt(kinetic/gamma)*sqrt((kinetic + 2.0)/gamma);
}

// Relativistic Boris kick in normalized momentum u=gamma*v/c. E uses AthenaK's
// velocity*B units; signed_gyro = q*B_unit*time_unit/(m*c). Magnetic rotation
// preserves |u|; using the measured energy/momentum change gives conservative feedback.
KOKKOS_INLINE_FUNCTION
void BorisKick(Real &energy, Real n[3], const Real electric[3], const Real magnetic[3],
               Real signed_gyro, Real c_code, Real dt) {
  const Real q = energy/kElectronRestEnergyCgs;
  const Real momentum = sqrt(q)*sqrt(q+2.0);
  Real minus[3], t[3], prime[3], plus[3];
  Real norm2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    minus[d] = momentum*n[d]+0.5*dt*signed_gyro*electric[d]/c_code;
    norm2 += minus[d]*minus[d];
  }
  const Real scale = 0.5*dt*signed_gyro/sqrt(1.0+norm2);
  Real t2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    t[d] = scale*magnetic[d];
    t2 += t[d]*t[d];
  }
  for (int d = 0; d < 3; ++d) {
    const int a = (d+1)%3, b = (d+2)%3;
    prime[d] = minus[d]+minus[a]*t[b]-minus[b]*t[a];
  }
  norm2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    const int a = (d+1)%3, b = (d+2)%3;
    plus[d] = minus[d]+2.0*(prime[a]*t[b]-prime[b]*t[a])/(1.0+t2)+
        0.5*dt*signed_gyro*electric[d]/c_code;
    norm2 += plus[d]*plus[d];
  }
  energy = kElectronRestEnergyCgs*norm2/(sqrt(1.0+norm2)+1.0);
  if (norm2 > 0.0) {
    const Real norm = sqrt(norm2);
    for (int d = 0; d < 3; ++d) n[d] = plus[d]/norm;
  }
}

// Positive electron kinetic-energy loss per physical second (erg/s per electron).
// dE/dt = -4*pi*r_e^2*m_e*c^3*n_e*ln(Lambda)/beta.
KOKKOS_INLINE_FUNCTION
Real StoppingRate(const Real energy_erg, const Real ne_cgs, const Real coulomb_log) {
  if (!(energy_erg > 0.0 && ne_cgs > 0.0 && coulomb_log > 0.0)) return 0.0;
  const Real beta = Beta(energy_erg);
  if (!(beta > 0.0)) return 0.0;
  return (kCollisionCoefficientCgs*ne_cgs)*coulomb_log*kElectronRestEnergyCgs/beta;
}

// The angular generator is (nu_D/2)*Laplacian_on_unit_sphere. Therefore
// <cos(theta)> = exp(-nu_D*dt) and <theta^2> = 2*nu_D*dt for small angles.
// Zeff = sum_i(n_i*Z_i^2)/n_e; the additional 1 is electron/electron scattering.
KOKKOS_INLINE_FUNCTION
Real DeflectionRate(const Real energy_erg, const Real ne_cgs, const Real zeff,
                    const Real coulomb_log) {
  if (!(energy_erg > 0.0 && ne_cgs > 0.0 && coulomb_log > 0.0)) return 0.0;
  const Real beta = Beta(energy_erg);
  if (!(beta > 0.0)) return 0.0;
  const Real inverse_momentum = 1.0/((1.0 + energy_erg/kElectronRestEnergyCgs)*beta);
  return (kCollisionCoefficientCgs*ne_cgs)*coulomb_log*(1.0 + fmax(zeff, 0.0))*
         inverse_momentum*inverse_momentum/beta;
}

// SplitMix64; zero is a valid state. Keeping the state on the packet makes its random
// sequence independent of Kokkos thread scheduling and MPI ownership changes.
KOKKOS_INLINE_FUNCTION
Real Uniform01(std::uint64_t &state) {
  state += UINT64_C(0x9e3779b97f4a7c15);
  std::uint64_t bits = state;
  bits = (bits ^ (bits >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
  bits = (bits ^ (bits >> 27))*UINT64_C(0x94d049bb133111eb);
  bits ^= bits >> 31;
  const Real value = (static_cast<double>(bits >> 12) + 0.5)*0x1.0p-52;
  // Conversion to float can round the largest values to one.
  return fmin(value, Real(1.0) - std::numeric_limits<Real>::epsilon());
}

// Rotate a unit direction by an axisymmetric narrow von Mises-Fisher distribution.
// The caller must subcycle so 0 <= nu_dt <= 0.1. With a = 1-exp(-nu_dt), the sampled
// first moment is coth(1/a)-a = exp(-nu_dt) + O(exp(-2/a)); its absolute error is
// below 2e-9 in this interval. Higher moments approach angular diffusion as dt -> 0.
// The zero step consumes no random numbers. The rotation conserves particle energy.
KOKKOS_INLINE_FUNCTION
void ScatterDirection(const Real nu_dt, std::uint64_t &rng,
                      Real &nx, Real &ny, Real &nz) {
  if (!(nu_dt > 0.0)) return;
  const Real width = -expm1(-nu_dt);
  const Real uniform = Uniform01(rng);
  const Real cosine = fmin(1.0, fmax(-1.0,
      1.0 + width*log(uniform + (1.0 - uniform)*exp(-2.0/width))));
  const Real sine = sqrt(fmax(0.0, (1.0 - cosine)*(1.0 + cosine)));
  const Real azimuth = 2.0*kPi*Uniform01(rng);

  // A perpendicular basis with no coordinate-pole singularity.
  Real tx, ty, tz;
  if (fabs(nz) < 0.9) {
    const Real inverse = 1.0/sqrt(nx*nx + ny*ny);
    tx = -ny*inverse;
    ty = nx*inverse;
    tz = 0.0;
  } else {
    const Real inverse = 1.0/sqrt(ny*ny + nz*nz);
    tx = 0.0;
    ty = -nz*inverse;
    tz = ny*inverse;
  }
  const Real sx = ny*tz - nz*ty;
  const Real sy = nz*tx - nx*tz;
  const Real sz = nx*ty - ny*tx;
  const Real along_t = sine*cos(azimuth), along_s = sine*sin(azimuth);
  const Real new_x = cosine*nx + along_t*tx + along_s*sx;
  const Real new_y = cosine*ny + along_t*ty + along_s*sy;
  const Real new_z = cosine*nz + along_t*tz + along_s*sz;
  const Real inverse_norm = 1.0/sqrt(new_x*new_x + new_y*new_y + new_z*new_z);
  nx = new_x*inverse_norm;
  ny = new_y*inverse_norm;
  nz = new_z*inverse_norm;
}

} // namespace fast_electrons

#endif // FAST_ELECTRONS_FAST_ELECTRON_PHYSICS_HPP_
