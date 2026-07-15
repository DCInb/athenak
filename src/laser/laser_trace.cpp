//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_trace.cpp
//! \brief Device ray initialization and straight-ray tracing kernels.

#include <cmath>

#include "athena.hpp"
#include "laser/laser.hpp"

namespace laser {

void Laser::BuildInitialRays() {
  // Phase-1 skeleton initializes deterministic one-point rays. Phase 2 replaces this
  // with aperture sampling and the DDA tracer without changing the public data layout.
  auto hx = Kokkos::create_mirror_view(ray_x0_);
  auto hy = Kokkos::create_mirror_view(ray_y0_);
  auto hz = Kokkos::create_mirror_view(ray_z0_);
  auto hnx = Kokkos::create_mirror_view(ray_nx0_);
  auto hny = Kokkos::create_mirror_view(ray_ny0_);
  auto hnz = Kokkos::create_mirror_view(ray_nz0_);
  auto hp = Kokkos::create_mirror_view(ray_power0_);
  auto hlambda = Kokkos::create_mirror_view(ray_wavelength_);
  auto hzbar = Kokkos::create_mirror_view(ray_zeff_);
  auto hk = Kokkos::create_mirror_view(ray_constant_absorption_);
  auto hb = Kokkos::create_mirror_view(ray_beam_);

  int ray = 0;
  for (std::size_t b = 0; b < beams_.size(); ++b) {
    const BeamConfig &beam = beams_[b];
    for (int n = 0; n < beam.nrays; ++n, ++ray) {
      hx(ray) = beam.origin[0]; hy(ray) = beam.origin[1]; hz(ray) = beam.origin[2];
      hnx(ray) = beam.direction[0]; hny(ray) = beam.direction[1];
      hnz(ray) = beam.direction[2];
      hp(ray) = beam.power/static_cast<Real>(beam.nrays);
      hlambda(ray) = beam.wavelength;
      hzbar(ray) = beam.zeff;
      hk(ray) = beam.constant_absorption;
      hb(ray) = static_cast<int>(b);
    }
  }
  Kokkos::deep_copy(ray_x0_, hx); Kokkos::deep_copy(ray_y0_, hy);
  Kokkos::deep_copy(ray_z0_, hz); Kokkos::deep_copy(ray_nx0_, hnx);
  Kokkos::deep_copy(ray_ny0_, hny); Kokkos::deep_copy(ray_nz0_, hnz);
  Kokkos::deep_copy(ray_power0_, hp); Kokkos::deep_copy(ray_wavelength_, hlambda);
  Kokkos::deep_copy(ray_zeff_, hzbar); Kokkos::deep_copy(ray_constant_absorption_, hk);
  Kokkos::deep_copy(ray_beam_, hb);
}

void Laser::TraceStraightRays() {
}

void Laser::CompactActiveQueue(DvceArray1D<int> current,
                               DvceArray1D<int> next) {
}

void Laser::FinalizeDiagnostics() {
}

} // namespace laser
