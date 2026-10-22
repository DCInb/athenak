#ifndef HYDRO_RSOLVERS_MATERIAL_HLLC_HYD_HPP_
#define HYDRO_RSOLVERS_MATERIAL_HLLC_HYD_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_hllc_hyd.hpp
//! \brief HLLC flux using EOS-supplied pressure, sound speed, and internal energy.

#include "hydro/rsolvers/material_llf_hyd.hpp"

namespace hydro {

// Davis bounds use both input acoustic cones, with no ideal-gas shock correction or
// Roe enthalpy average. The HLLC jump conditions then use the actual internal energy
// density in wl.e/wr.e, which need not equal p/(gamma-1).
KOKKOS_INLINE_FUNCTION
bool TrySingleStateHLLC_HydMaterial(
    const HydPrim1D &wl, const HydPrim1D &wr,
    const Real pl, const Real pr, const Real cs2l, const Real cs2r,
    HydCons1D &flux) {
  const Real cl = sqrt(fmax(cs2l, 0.0));
  const Real cr = sqrt(fmax(cs2r, 0.0));
  const Real sl = fmin(wl.vx-cl, wr.vx-cr);
  const Real sr = fmax(wl.vx+cl, wr.vx+cr);
  if (!(sr > sl) || !Kokkos::isfinite(sl) || !Kokkos::isfinite(sr)) return false;

  if (sl >= 0.0 || sr <= 0.0) {
    const auto &w = (sl >= 0.0) ? wl : wr;
    const Real pressure = (sl >= 0.0) ? pl : pr;
    flux.d = w.d*w.vx;
    flux.mx = flux.d*w.vx+pressure;
    flux.my = flux.d*w.vy;
    flux.mz = flux.d*w.vz;
    const Real energy = w.e+0.5*w.d*(SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
    flux.e = (energy+pressure)*w.vx;
    return true;
  }

  const Real al = wl.d*(wl.vx-sl);
  const Real ar = wr.d*(sr-wr.vx);
  const Real denom = al+ar;
  if (!(al > 0.0 && ar > 0.0) || !Kokkos::isfinite(denom)) return false;
  const Real sm = (al/denom)*wl.vx+(ar/denom)*wr.vx+(pl-pr)/denom;
  const Real pstar = (ar/denom)*pl+(al/denom)*pr+
                     al*(ar/denom)*(wl.vx-wr.vx);
  const Real tolerance = 16.0*Kokkos::Experimental::epsilon<Real>::value*
                         fmax(fabs(sl), fabs(sr));
  if (!Kokkos::isfinite(sm) || !Kokkos::isfinite(pstar) || pstar < 0.0 ||
      sm-sl <= tolerance || sr-sm <= tolerance) return false;

  const Real dl = al/(sm-sl);
  const Real dr = ar/(sr-sm);
  const Real dvl = sm-wl.vx;
  const Real dvr = sm-wr.vx;
  // Specific internal energies in the star states, written without subtracting
  // kinetic energy from total energy (important for the dual-energy carrier).
  const Real eil = wl.e/wl.d+0.5*SQR(dvl)-dvl*pl/al;
  const Real eir = wr.e/wr.d+0.5*SQR(dvr)+dvr*pr/ar;
  if (!(dl > 0.0 && dr > 0.0 && eil > 0.0 && eir > 0.0) ||
      !Kokkos::isfinite(dl) || !Kokkos::isfinite(dr) ||
      !Kokkos::isfinite(eil) || !Kokkos::isfinite(eir)) return false;

  const auto &w = (sm >= 0.0) ? wl : wr;
  const Real density = (sm >= 0.0) ? dl : dr;
  const Real eint = (sm >= 0.0) ? eil : eir;
  const Real energy = density*(eint+0.5*(SQR(sm)+SQR(w.vy)+SQR(w.vz)));
  flux.d = density*sm;
  flux.mx = flux.d*sm+pstar;
  flux.my = flux.d*w.vy;
  flux.mz = flux.d*w.vz;
  flux.e = (energy+pstar)*sm;
  return Kokkos::isfinite(flux.d) && Kokkos::isfinite(flux.mx) &&
         Kokkos::isfinite(flux.my) && Kokkos::isfinite(flux.mz) &&
         Kokkos::isfinite(flux.e);
}

// Returns false when a degenerate/nonphysical intermediate state requires LLF.
// This is a local Riemann-flux fallback, not FOFC: the caller still transports all
// material, ion/electron, and radiation scalars with the resulting mass flux.
KOKKOS_INLINE_FUNCTION
bool SingleStateHLLC_HydMaterial(
    const HydPrim1D &wl, const HydPrim1D &wr,
    const Real pl, const Real pr, const Real cs2l, const Real cs2r,
    HydCons1D &flux) {
  if (TrySingleStateHLLC_HydMaterial(wl, wr, pl, pr, cs2l, cs2r, flux)) {
    return true;
  }
  SingleStateLLF_HydMaterial(wl, wr, pl, pr, cs2l, cs2r, flux);
  return false;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_MATERIAL_HLLC_HYD_HPP_
