#ifndef HYDRO_RSOLVERS_MATERIAL_LLF_HYD_HPP_
#define HYDRO_RSOLVERS_MATERIAL_LLF_HYD_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_llf_hyd.hpp
//! \brief LLF hydro flux supplied with pressure and sound speed by a material EOS.
//!
//! This is the B=0 limit of mhd/rsolvers/material_llf_mhd.hpp; the two must stay in sync.

#include "athena.hpp"

namespace hydro {

enum MaterialFaceField {
  material_total_pressure = 0,
  material_sound_speed_squared = 1,
  nmaterial_face_fields = 2
};

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_HydMaterial(
    const HydPrim1D &wl, const HydPrim1D &wr,
    const Real pressure_left, const Real pressure_right,
    const Real sound_speed_squared_left, const Real sound_speed_squared_right,
    HydCons1D &flux) {
  const Real mass_flux_left = wl.d*wl.vx;
  const Real mass_flux_right = wr.d*wr.vx;

  HydCons1D flux_sum;
  flux_sum.d = mass_flux_left+mass_flux_right;
  flux_sum.mx = mass_flux_left*wl.vx+mass_flux_right*wr.vx+
      pressure_left+pressure_right;
  flux_sum.my = mass_flux_left*wl.vy+mass_flux_right*wr.vy;
  flux_sum.mz = mass_flux_left*wl.vz+mass_flux_right*wr.vz;

  const Real total_energy_left = wl.e+
      0.5*wl.d*(SQR(wl.vx)+SQR(wl.vy)+SQR(wl.vz));
  const Real total_energy_right = wr.e+
      0.5*wr.d*(SQR(wr.vx)+SQR(wr.vy)+SQR(wr.vz));
  flux_sum.e = (total_energy_left+pressure_left)*wl.vx+
               (total_energy_right+pressure_right)*wr.vx;

  const Real speed_left = sqrt(fmax(sound_speed_squared_left, 0.0));
  const Real speed_right = sqrt(fmax(sound_speed_squared_right, 0.0));
  const Real signal_speed = fmax(
      fabs(wl.vx)+speed_left, fabs(wr.vx)+speed_right);

  HydCons1D jump;
  jump.d = signal_speed*(wr.d-wl.d);
  jump.mx = signal_speed*(wr.d*wr.vx-wl.d*wl.vx);
  jump.my = signal_speed*(wr.d*wr.vy-wl.d*wl.vy);
  jump.mz = signal_speed*(wr.d*wr.vz-wl.d*wl.vz);
  jump.e = signal_speed*(total_energy_right-total_energy_left);

  flux.d = 0.5*(flux_sum.d-jump.d);
  flux.mx = 0.5*(flux_sum.mx-jump.mx);
  flux.my = 0.5*(flux_sum.my-jump.my);
  flux.mz = 0.5*(flux_sum.mz-jump.mz);
  flux.e = 0.5*(flux_sum.e-jump.e);
}

} // namespace hydro

#endif // HYDRO_RSOLVERS_MATERIAL_LLF_HYD_HPP_
