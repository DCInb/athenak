#ifndef NUCLEAR_DT_CELL_HPP_
#define NUCLEAR_DT_CELL_HPP_
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
#include "nuclear/dt_physics.hpp"
#include "materials/material_mixture.hpp"

namespace nuclear {
constexpr int kMaxBurnMaterials = 16;

// Physical temperatures, including the composition-dependent molecular weight of the
// legacy ideal mixture. Table temperatures already carry their own Kelvin conversion.
KOKKOS_INLINE_FUNCTION
void Temperatures(const materials::MaterialMixtureDevice &mix,
                  const materials::MaterialComposition &composition,
                  double rho, double ei, double ee, double pressure_unit,
                  double density_unit, double &ti, double &te) {
  if (mix.UsesTabularEOS()) {
    auto state = mix.StateFromRhoSpecificEnergiesNoSound(
        rho, ei/rho, ee/rho, composition);
    ti = state.ion_temperature*mix.temperature_to_kelvin;
    te = state.electron_temperature*mix.temperature_to_kelvin;
  } else {
    const double ions = rho*density_unit/kAmu*mix.IonNumberPerAtomicMass(composition);
    const double electrons = rho*density_unit/kAmu*
        mix.ElectronNumberPerAtomicMass(composition);
    ti = mix.gamma_minus_one*ei*pressure_unit/(ions*kBoltzmann);
    te = mix.gamma_minus_one*ee*pressure_unit/(electrons*kBoltzmann);
  }
}

struct BurnCell {
  materials::MaterialMixtureDevice mixture;
  Real partial[kMaxBurnMaterials];
  double rho, ei, ee, density_unit, pressure_unit, multiplier;
  int d, t, a;

  KOKKOS_INLINE_FUNCTION
  double Rate(double extent) const {
    Real trial[kMaxBurnMaterials];
    for (int n = 0; n < mixture.nmaterials; ++n) trial[n] = partial[n];
    const double mass = extent*kAmu/density_unit;
    trial[d] = fmax(0.0, partial[d]-2.0*mass);
    trial[t] = fmax(0.0, partial[t]-3.0*mass);
    trial[a] += 4.0*mass;
    const double density = rho-mass;
    const auto composition = mixture.CompositionFromPartialDensities(trial, density);
    double ti, te;
    Temperatures(mixture, composition, density, ei, ee, pressure_unit, density_unit,
                 ti, te);
    if (!Kokkos::isfinite(ti) || ti < 0.0) return -1.0;
    const double rate = DTReactivity(ti*kBoltzmann/kKev);
    return rate < 0.0 ? rate : multiplier*rate;
  }

  KOKKOS_INLINE_FUNCTION
  double Midpoint(double extent, double nd, double nt, double dt) const {
    const double rate = Rate(extent);
    if (rate < 0.0) return -1.0;
    const double half = extent+BurnExtent(nd-extent, nt-extent, 0.5*dt*rate);
    const double middle = Rate(half);
    if (middle < 0.0) return -1.0;
    return extent+BurnExtent(nd-extent, nt-extent, dt*middle);
  }

  // Exponential midpoint with step doubling: second order, exact at constant T,
  // stoichiometric and positive without clipping a negative ODE solution. Each cell
  // owns its tiny integrator; no GPU Jacobian storage or host ODE calls are needed.
  KOKKOS_INLINE_FUNCTION
  bool Integrate(double dt, double rtol, double &extent) const {
    const double nd = partial[d]*density_unit/(2.0*kAmu);
    const double nt = partial[t]*density_unit/(3.0*kAmu);
    const double limit = fmin(nd, nt);
    extent = 0.0;
    if (!(limit > 0.0 && multiplier > 0.0)) return true;
    double elapsed = 0.0, h = dt;
    for (int step = 0; step < 4096 && elapsed < dt; ++step) {
      h = fmin(h, dt-elapsed);
      if (!(h > 0.0) || elapsed+h == elapsed) return false;
      const double full = Midpoint(extent, nd, nt, h);
      const double half = Midpoint(extent, nd, nt, 0.5*h);
      if (full < 0.0 || half < 0.0) return false;
      const double fine = Midpoint(half, nd, nt, 0.5*h);
      if (fine < 0.0 || !Kokkos::isfinite(fine)) return false;
      const double error = fabs(fine-full)/(3.0*(1.0e-14*limit+
          rtol*fmax(limit-fine, 1.0e-12*limit)));
      if (error <= 1.0) {
        extent = fmin(fine, limit);
        elapsed += h;
      }
      h *= fmin(2.0, fmax(0.1, 0.8/cbrt(fmax(error, 1.0e-12))));
    }
    return elapsed >= dt;
  }
};

struct PlasmaCollision {
  double ion = 0.0, electron = 0.0, deflection = 0.0;
};
KOKKOS_INLINE_FUNCTION
PlasmaCollision Collision(const materials::MaterialMixtureDevice &mix,
                          const materials::MaterialComposition &composition,
                          double density, double ti, double te,
                          double energy, double log_lambda) {
  PlasmaCollision result;
  double ne = 0.0;
  for (int n = 0; n < mix.nmaterials; ++n) {
    const auto species = mix.Species(n);
    const double number = density*composition[n]/(species.abar*kAmu);
    ne += number*species.zbar;
    const auto rates = AlphaTargetCollision(energy, number, species.abar*kAmu,
        sqrt(species.zbar*species.zeff), ti, log_lambda);
    result.ion += rates.loss;
    result.deflection += rates.deflection;
  }
  const auto rates = AlphaTargetCollision(energy, ne, kElectronMass, 1.0, te,
                                        log_lambda);
  result.electron = rates.loss;
  result.deflection += rates.deflection;
  return result;
}
}  // namespace nuclear
#endif  // NUCLEAR_DT_CELL_HPP_
