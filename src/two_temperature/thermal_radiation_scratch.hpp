//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_SCRATCH_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_SCRATCH_HPP_

#include "athena.hpp"

namespace two_temperature {

// Store the aggregate hierarchy in the active interior of an existing scratch field.
// The next halo exchange may overwrite it after all preconditioner kernels finish.
KOKKOS_INLINE_FUNCTION
Real ImplicitPackedScratchValue(
    const DvceArray5D<Real> scratch, const int m, const int packed,
    const int nx1, const int nx2, const int is, const int js, const int ks) {
  const int plane = nx1*nx2;
  const int k = packed/plane;
  const int remainder = packed-k*plane;
  const int j = remainder/nx1;
  const int i = remainder-j*nx1;
  return scratch(m, 0, ks+k, js+j, is+i);
}

KOKKOS_INLINE_FUNCTION
void SetImplicitPackedScratchValue(
    const DvceArray5D<Real> scratch, const int m, const int packed,
    const int nx1, const int nx2, const int is, const int js, const int ks,
    const Real value) {
  const int plane = nx1*nx2;
  const int k = packed/plane;
  const int remainder = packed-k*plane;
  const int j = remainder/nx1;
  const int i = remainder-j*nx1;
  scratch(m, 0, ks+k, js+j, is+i) = value;
}

}  // namespace two_temperature
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_SCRATCH_HPP_
