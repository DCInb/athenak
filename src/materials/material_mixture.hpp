#ifndef MATERIALS_MATERIAL_MIXTURE_HPP_
#define MATERIALS_MATERIAL_MIXTURE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.hpp
//! \brief Two-material, passive-scalar-aware ideal-plasma closure.

#include "athena.hpp"

class ParameterInput;

namespace materials {

struct SpeciesProperties {
  Real abar = 1.0;  //!< Mean ion mass in atomic-mass units.
  Real zbar = 1.0;  //!< Mean number of free electrons per ion.
  Real zeff = 1.0;  //!< Effective charge used by collisional laser absorption.
  Real t_ei = -1.0; //!< Ion-electron exchange time; negative disables exchange.
};

//----------------------------------------------------------------------------------------
//! \struct MaterialMixtureDevice
//! \brief Device-copyable closure for two materials represented by rho*Y0.
//!
//! Both materials are monatomic, fully ionized ideal gases with gamma=5/3. Their common
//! gamma leaves the MHD pressure-energy relation unchanged. Following FLASH,
//! 1/Abar_mix=sum(Y_s/A_s) and Zbar_mix/Abar_mix=sum(Y_s Z_s/A_s).

struct MaterialMixtureDevice {
  SpeciesProperties material0;
  SpeciesProperties material1;
  int scalar_index = -1;  //!< Absolute primitive/conserved variable index.

  // Match the atomic-mass constant used by src/units/units.hpp.
  static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;

  KOKKOS_INLINE_FUNCTION
  Real ClampMassFraction(const Real y0) const {
    return fmin(fmax(y0, 0.0), 1.0);
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromPrimitive(const DvceArray5D<Real> &prim,
                                           const int m, const int k,
                                           const int j, const int i) const {
    return ClampMassFraction(prim(m, scalar_index, k, j, i));
  }

  KOKKOS_INLINE_FUNCTION
  Real Material0MassFractionFromConserved(const DvceArray5D<Real> &cons,
                                          const int m, const int k,
                                          const int j, const int i,
                                          const Real density_floor = 0.0) const {
    const Real density = fmax(cons(m, IDN, k, j, i), density_floor);
    if (!(density > 0.0)) return 0.0;
    return ClampMassFraction(cons(m, scalar_index, k, j, i)/density);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0/material0.abar + (1.0-y0)/material1.abar;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerAtomicMass(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    return y0*material0.zbar/material0.abar
           + (1.0-y0)*material1.zbar/material1.abar;
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanAtomicMass(const Real y0) const {
    return 1.0/IonNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanIonization(const Real y0) const {
    return ElectronNumberPerAtomicMass(y0)/IonNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real MeanParticleMass(const Real y0) const {
    return 1.0/(IonNumberPerAtomicMass(y0)+ElectronNumberPerAtomicMass(y0));
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacityFraction(const Real y0) const {
    const Real ni = IonNumberPerAtomicMass(y0);
    const Real ne = ElectronNumberPerAtomicMass(y0);
    return ne/(ni+ne);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonHeatCapacityFraction(const Real y0) const {
    return 1.0-ElectronHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronHeatCapacity(const Real gamma_minus_one, const Real y0) const {
    return ElectronHeatCapacityFraction(y0)/gamma_minus_one;
  }

  KOKKOS_INLINE_FUNCTION
  Real IonHeatCapacity(const Real gamma_minus_one, const Real y0) const {
    return IonHeatCapacityFraction(y0)/gamma_minus_one;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberPerGram(const Real y0) const {
    return ElectronNumberPerAtomicMass(y0)/atomic_mass_unit_cgs;
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensity(const Real density, const Real y0) const {
    // Electron number per volume in units whose mass unit is one atomic mass unit.
    // This is distinct from the heat-capacity fraction q_e/(q_i+q_e).
    return density*ElectronNumberPerAtomicMass(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronNumberDensityCgs(const Real code_density,
                                const Real density_scale_cgs,
                                const Real y0) const {
    return code_density*density_scale_cgs*ElectronNumberPerGram(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real EffectiveCharge(const Real y0_in) const {
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*material0.zbar/material0.abar;
    const Real ne1 = (1.0-y0)*material1.zbar/material1.abar;
    return (ne0*material0.zeff + ne1*material1.zeff)/(ne0+ne1);
  }

  KOKKOS_INLINE_FUNCTION
  Real InitialElectronEnergyFraction(const Real y0,
                                     const Real electron_to_ion_temperature) const {
    const Real fe = ElectronHeatCapacityFraction(y0);
    const Real fi = 1.0-fe;
    return fe*electron_to_ion_temperature/
           (fi+fe*electron_to_ion_temperature);
  }

  KOKKOS_INLINE_FUNCTION
  Real ElectronTemperature(const Real gamma_minus_one,
                           const Real electron_specific_energy,
                           const Real y0) const {
    return gamma_minus_one*electron_specific_energy/
           ElectronHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real IonTemperature(const Real gamma_minus_one,
                      const Real ion_specific_energy,
                      const Real y0) const {
    return gamma_minus_one*ion_specific_energy/IonHeatCapacityFraction(y0);
  }

  KOKKOS_INLINE_FUNCTION
  Real ExchangeTime(const Real y0_in) const {
    // Add material collision rates after weighting by their electron populations.
    const Real y0 = ClampMassFraction(y0_in);
    const Real ne0 = y0*material0.zbar/material0.abar;
    const Real ne1 = (1.0-y0)*material1.zbar/material1.abar;
    if ((ne0 > 0.0 && material0.t_ei == 0.0) ||
        (ne1 > 0.0 && material1.t_ei == 0.0)) {
      return 0.0;
    }
    Real rate = 0.0;
    if (material0.t_ei > 0.0) rate += ne0/material0.t_ei;
    if (material1.t_ei > 0.0) rate += ne1/material1.t_ei;
    return (rate > 0.0) ? (ne0+ne1)/rate : -1.0;
  }
};

class MaterialMixture {
 public:
  MaterialMixture(ParameterInput *pin, int first_user_scalar, int nuser_scalars,
                  Real gamma);
  ~MaterialMixture() = default;

  MaterialMixtureDevice DeviceData() const { return data_; }
  int ScalarIndex() const { return data_.scalar_index; }

 private:
  MaterialMixtureDevice data_;
};

} // namespace materials

#endif // MATERIALS_MATERIAL_MIXTURE_HPP_
