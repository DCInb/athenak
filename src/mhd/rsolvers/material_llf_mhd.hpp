#ifndef MHD_RSOLVERS_MATERIAL_LLF_MHD_HPP_
#define MHD_RSOLVERS_MATERIAL_LLF_MHD_HPP_
//========================================================================================
//! \file material_llf_mhd.hpp
//! \brief LLF MHD flux supplied with pressure and sound speed by a material EOS.

#include "athena.hpp"

namespace mhd {

enum MaterialFaceField {
  material_total_pressure = 0,
  material_sound_speed_squared = 1,
  nmaterial_face_fields = 2
};

KOKKOS_INLINE_FUNCTION
Real MaterialFastMagnetosonicSpeed(const Real density, const Real cs2,
                                   const Real bx, const Real by, const Real bz) {
  const Real sound_speed_squared = fmax(cs2, 0.0);
  const Real va2 = (SQR(bx)+SQR(by)+SQR(bz))/density;
  const Real vax2 = SQR(bx)/density;
  const Real sum = sound_speed_squared+va2;
  const Real discriminant = fmax(
      sum*sum-4.0*sound_speed_squared*vax2, 0.0);
  return sqrt(0.5*(sum+sqrt(discriminant)));
}

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_MHDMaterial(
    const MHDPrim1D &wl, const MHDPrim1D &wr, const Real bxi,
    const Real pressure_left, const Real pressure_right,
    const Real sound_speed_squared_left, const Real sound_speed_squared_right,
    MHDCons1D &flux) {
  const Real mass_flux_left = wl.d*wl.vx;
  const Real mass_flux_right = wr.d*wr.vx;
  const Real magnetic_normal_stress_left =
      0.5*(SQR(wl.by)+SQR(wl.bz)-SQR(bxi));
  const Real magnetic_normal_stress_right =
      0.5*(SQR(wr.by)+SQR(wr.bz)-SQR(bxi));

  MHDCons1D flux_sum;
  flux_sum.d = mass_flux_left+mass_flux_right;
  flux_sum.mx = mass_flux_left*wl.vx+mass_flux_right*wr.vx+
      magnetic_normal_stress_left+magnetic_normal_stress_right+
      pressure_left+pressure_right;
  flux_sum.my = mass_flux_left*wl.vy+mass_flux_right*wr.vy-
                bxi*(wl.by+wr.by);
  flux_sum.mz = mass_flux_left*wl.vz+mass_flux_right*wr.vz-
                bxi*(wl.bz+wr.bz);
  flux_sum.by = wl.by*wl.vx+wr.by*wr.vx-bxi*(wl.vy+wr.vy);
  flux_sum.bz = wl.bz*wl.vx+wr.bz*wr.vx-bxi*(wl.vz+wr.vz);

  const Real total_energy_left = wl.e+
      0.5*wl.d*(SQR(wl.vx)+SQR(wl.vy)+SQR(wl.vz))+
      magnetic_normal_stress_left+SQR(bxi);
  const Real total_energy_right = wr.e+
      0.5*wr.d*(SQR(wr.vx)+SQR(wr.vy)+SQR(wr.vz))+
      magnetic_normal_stress_right+SQR(bxi);
  flux_sum.e = (total_energy_left+pressure_left+
                magnetic_normal_stress_left)*wl.vx+
               (total_energy_right+pressure_right+
                magnetic_normal_stress_right)*wr.vx;
  flux_sum.e -= bxi*(wl.by*wl.vy+wl.bz*wl.vz+
                     wr.by*wr.vy+wr.bz*wr.vz);

  const Real speed_left = MaterialFastMagnetosonicSpeed(
      wl.d, sound_speed_squared_left, bxi, wl.by, wl.bz);
  const Real speed_right = MaterialFastMagnetosonicSpeed(
      wr.d, sound_speed_squared_right, bxi, wr.by, wr.bz);
  const Real signal_speed = fmax(
      fabs(wl.vx)+speed_left, fabs(wr.vx)+speed_right);

  MHDCons1D jump;
  jump.d = signal_speed*(wr.d-wl.d);
  jump.mx = signal_speed*(wr.d*wr.vx-wl.d*wl.vx);
  jump.my = signal_speed*(wr.d*wr.vy-wl.d*wl.vy);
  jump.mz = signal_speed*(wr.d*wr.vz-wl.d*wl.vz);
  jump.e = signal_speed*(total_energy_right-total_energy_left);
  jump.by = signal_speed*(wr.by-wl.by);
  jump.bz = signal_speed*(wr.bz-wl.bz);

  flux.d = 0.5*(flux_sum.d-jump.d);
  flux.mx = 0.5*(flux_sum.mx-jump.mx);
  flux.my = 0.5*(flux_sum.my-jump.my);
  flux.mz = 0.5*(flux_sum.mz-jump.mz);
  flux.e = 0.5*(flux_sum.e-jump.e);
  flux.by = -0.5*(flux_sum.by-jump.by);
  flux.bz = 0.5*(flux_sum.bz-jump.bz);
}

} // namespace mhd

#endif // MHD_RSOLVERS_MATERIAL_LLF_MHD_HPP_
