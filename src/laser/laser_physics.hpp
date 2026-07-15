#ifndef LASER_LASER_PHYSICS_HPP_
#define LASER_LASER_PHYSICS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_physics.hpp
//! \brief Device-callable optical physics for laser ray transport.

#include "athena.hpp"

namespace laser {

// CODATA cgs constants used by the FLASH inverse-bremsstrahlung model.
constexpr Real electron_charge_cgs = 4.803204712570263e-10;
constexpr Real electron_mass_cgs = 9.1093837015e-28;
constexpr Real boltzmann_cgs = 1.380649e-16;
constexpr Real light_speed_cgs = 2.99792458e10;
constexpr Real pi = 3.141592653589793238462643383279502884;

//----------------------------------------------------------------------------------------
//! Critical electron number density in cm^-3 for a vacuum wavelength in cm.

KOKKOS_INLINE_FUNCTION
Real CriticalDensity(Real wavelength) {
  if (!(wavelength > 0.0)) return 0.0;
  return electron_mass_cgs*pi*SQR(light_speed_cgs)/
      (SQR(electron_charge_cgs)*SQR(wavelength));
}

//----------------------------------------------------------------------------------------
//! FLASH cell-average inverse-bremsstrahlung spatial attenuation in cm^-1.
//!
//! Inputs are electron number density [cm^-3], electron temperature [K], effective
//! charge, and vacuum wavelength [cm]. FLASH writes dP/dt=-(ne/nc)*nu_ei*P. Division
//! by the geometric-optics group speed c*sqrt(1-ne/nc) converts that rate to dP/ds.

KOKKOS_INLINE_FUNCTION
Real InverseBremsstrahlungCoefficient(Real ne, Real te, Real zeff,
                                      Real wavelength) {
  if (!(ne > 0.0) || !(te > 0.0) || !(zeff > 0.0) ||
      !(wavelength > 0.0)) {
    return 0.0;
  }
  const Real nc = CriticalDensity(wavelength);
  if (!(nc > 0.0)) return 0.0;

  // Reflection normally prevents entry above critical density. The cap keeps this
  // weak-collision expression finite at roundoff distance from the turning surface.
  const Real density_ratio = fmin(ne/nc, 1.0 - 1.0e-12);
  if (!(density_ratio > 0.0)) return 0.0;
  const Real coulomb_argument =
      3.0/(2.0*zeff*electron_charge_cgs*electron_charge_cgs*
           electron_charge_cgs)*
      sqrt(SQR(boltzmann_cgs)*boltzmann_cgs*te*te*te/(pi*ne));
  const Real coulomb_log = fmax(log(fmax(coulomb_argument, 1.0)), 1.0);
  const Real collision_frequency =
      (4.0/3.0)*sqrt(2.0*pi/electron_mass_cgs)*ne*zeff*
      SQR(SQR(electron_charge_cgs))*coulomb_log/
      pow(boltzmann_cgs*te, 1.5);
  const Real group_speed = light_speed_cgs*sqrt(fmax(1.0-density_ratio, 1.0e-12));
  return density_ratio*collision_frequency/group_speed;
}

//----------------------------------------------------------------------------------------
//! Stable exact attenuation over one cell segment.

KOKKOS_INLINE_FUNCTION
Real DepositedPower(Real power, Real coefficient, Real path_length) {
  if (!(power > 0.0) || !(coefficient > 0.0) || !(path_length > 0.0)) return 0.0;
  const Real tau = coefficient*path_length;
  return -power*Kokkos::expm1(-tau);
}

} // namespace laser

#endif // LASER_LASER_PHYSICS_HPP_
