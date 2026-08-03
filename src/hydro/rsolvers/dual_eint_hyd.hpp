#ifndef HYDRO_RSOLVERS_DUAL_EINT_HYD_HPP_
#define HYDRO_RSOLVERS_DUAL_EINT_HYD_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dual_eint_hyd.hpp
//! \brief Face-velocity helper for two-temperature-backed dual-energy hydrodynamics.
//!
//! Mirrors mhd/rsolvers/dual_eint_mhd.hpp; the two must stay in sync.

namespace hydro {

// Recover the contact/upwind transport velocity used by the auxiliary internal-energy
// equation.  Storing it separately lets AMR flux correction use the same coarse/fine
// face velocity in the subsequent p div(v) update.
KOKKOS_INLINE_FUNCTION
void UpwindDualEnergyVelocity(TeamMember_t const &member, const EOS_Data &eos,
                              const int m, const int k, const int j,
                              const int il, const int iu,
                              const ScrArray2D<Real> &wl,
                              const ScrArray2D<Real> &wr,
                              DvceArray5D<Real> flx,
                              DvceArray5D<Real> vf) {
  par_for_inner(member, il, iu, [&](const int i) {
    const Real mass_flux = flx(m, IDN, k, j, i);
    if (mass_flux > 0.0) {
      vf(m, 0, k, j, i) = mass_flux/fmax(wl(IDN, i), eos.dfloor);
    } else if (mass_flux < 0.0) {
      vf(m, 0, k, j, i) = mass_flux/fmax(wr(IDN, i), eos.dfloor);
    } else {
      vf(m, 0, k, j, i) = 0.0;
    }
  });
}

} // namespace hydro

#endif // HYDRO_RSOLVERS_DUAL_EINT_HYD_HPP_
