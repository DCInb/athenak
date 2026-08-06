//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file two_temperature.cpp
//! \brief Implements the Newtonian ideal-gas ion/electron two-temperature model.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "materials/material_mixture.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "parameter_input.hpp"
#include "two_temperature/biermann_closure.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {

// Electron temperature-relaxation time from the classical Spitzer binary-collision
// result.  The mixture identity n_i Z^2 = n_e Z_eff avoids inventing a single ion charge.
KOKKOS_INLINE_FUNCTION
Real SpitzerExchangeTime(const Real electron_density_cgs,
                         const Real ion_temperature_kelvin,
                         const Real electron_temperature_kelvin,
                         const Real mean_atomic_mass,
                         const Real effective_charge,
                         const Real coulomb_log) {
  constexpr Real electron_mass_cgs = 9.1093837015e-28;
  constexpr Real electron_charge_cgs = 4.803204712570263e-10;
  constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;
  constexpr Real boltzmann_cgs = 1.3806488e-16;
  constexpr Real pi = 3.141592653589793238462643383279502884;
  const Real ion_mass_cgs = mean_atomic_mass*atomic_mass_unit_cgs;
  const Real thermal_speed_squared =
      boltzmann_cgs*electron_temperature_kelvin/electron_mass_cgs+
      boltzmann_cgs*ion_temperature_kelvin/ion_mass_cgs;
  const Real numerator = 3.0*electron_mass_cgs*ion_mass_cgs*
                         pow(thermal_speed_squared, 1.5);
  const Real denominator = 8.0*sqrt(2.0*pi)*electron_density_cgs*
      effective_charge*pow(electron_charge_cgs, 4)*coulomb_log;
  return numerator/denominator;
}

KOKKOS_INLINE_FUNCTION
void StoreMaterialThermodynamics(
    DvceArray5D<Real> temperature, DvceArray5D<Real> thermodynamics,
    const materials::MaterialThermodynamicState &state,
    const int additional_query_flags,
    const int m, const int k, const int j, const int i) {
  temperature(m, 0, k, j, i) = state.ion_temperature;
  temperature(m, 1, k, j, i) = state.electron_temperature;
  thermodynamics(m, two_temperature::TwoTemperature::ion_pressure, k, j, i) =
      state.ion_pressure;
  thermodynamics(
      m, two_temperature::TwoTemperature::electron_pressure, k, j, i) =
      state.electron_pressure;
  thermodynamics(
      m, two_temperature::TwoTemperature::electron_number_density_cgs, k, j, i) =
      state.electron_number_density_cgs;
  thermodynamics(m, two_temperature::TwoTemperature::mean_ionization, k, j, i) =
      state.mean_ionization;
  thermodynamics(
      m, two_temperature::TwoTemperature::sound_speed_squared, k, j, i) =
      state.sound_speed_squared;
  thermodynamics(m, two_temperature::TwoTemperature::effective_charge, k, j, i) =
      state.effective_charge;
  const int previous_flags = static_cast<int>(thermodynamics(
      m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
  thermodynamics(m, two_temperature::TwoTemperature::eos_query_flags, k, j, i) =
      static_cast<Real>(previous_flags | state.query_flags | additional_query_flags);
}

KOKKOS_INLINE_FUNCTION
void StoreMaterialTemperaturesAndFlags(
    DvceArray5D<Real> temperature, DvceArray5D<Real> thermodynamics,
    const materials::MaterialTemperatureState &state,
    const int additional_query_flags,
    const int m, const int k, const int j, const int i) {
  temperature(m, 0, k, j, i) = state.ion_temperature;
  temperature(m, 1, k, j, i) = state.electron_temperature;
  const int previous_flags = static_cast<int>(thermodynamics(
      m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
  thermodynamics(m, two_temperature::TwoTemperature::eos_query_flags, k, j, i) =
      static_cast<Real>(previous_flags | state.query_flags | additional_query_flags);
}

KOKKOS_INLINE_FUNCTION
void StoreMaterialElectronState(
    DvceArray5D<Real> temperature, DvceArray5D<Real> thermodynamics,
    const materials::MaterialElectronState &state,
    const int additional_query_flags,
    const int m, const int k, const int j, const int i) {
  temperature(m, 1, k, j, i) = state.electron_temperature;
  thermodynamics(
      m, two_temperature::TwoTemperature::electron_pressure, k, j, i) =
      state.electron_pressure;
  thermodynamics(
      m, two_temperature::TwoTemperature::electron_number_density_cgs, k, j, i) =
      state.electron_number_density_cgs;
  const int previous_flags = static_cast<int>(thermodynamics(
      m, two_temperature::TwoTemperature::eos_query_flags, k, j, i));
  thermodynamics(m, two_temperature::TwoTemperature::eos_query_flags, k, j, i) =
      static_cast<Real>(previous_flags | state.query_flags | additional_query_flags);
}

} // namespace

namespace two_temperature {

//----------------------------------------------------------------------------------------
// Constructor.  Heat capacities are normalized so cv_i + cv_e = 1/(gamma - 1).

TwoTemperature::TwoTemperature(const std::string &block, MeshBlockPack *ppack,
                               ParameterInput *pin, int first_component_index,
                               materials::MaterialMixture *material_mixture) :
    iion(first_component_index),
    iele(first_component_index + 1),
    temperature("two-temperature", 1, 1, 1, 1, 1),
    thermodynamics("two-temperature-thermodynamics", 1, 1, 1, 1, 1),
    pmy_pack_(ppack),
    use_material_mixture_(material_mixture != nullptr) {
  Real gamma = pin->GetReal(block, "gamma");
  gamma_minus_one_ = gamma - 1.0;
  cv_e_fraction_ = pin->GetOrAddReal(block, "electron_heat_capacity_fraction", 0.5);
  cv_i_fraction_ = 1.0 - cv_e_fraction_;
  initial_temperature_ratio_ =
      pin->GetOrAddReal(block, "initial_electron_temperature_ratio", 1.0);
  t_ei_ = pin->GetOrAddReal(block, "t_ei", -1.0);
  density_floor_ = pin->GetOrAddReal(block, "dfloor", 0.0);
  pressure_floor_ = pin->GetOrAddReal(block, "pfloor", 0.0);
  temperature_floor_ = pin->GetOrAddReal(block, "tfloor", 0.0);
  if (use_material_mixture_) material_mixture_ = material_mixture->DeviceData();
  const std::string exchange_model =
      pin->GetOrAddString(block, "t_ei_model", "constant");
  if (exchange_model == "constant") {
    use_spitzer_exchange_ = false;
  } else if (exchange_model == "spitzer") {
    use_spitzer_exchange_ = true;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block
              << "> t_ei_model must be 'constant' or 'spitzer'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (use_spitzer_exchange_) {
    if (!use_material_mixture_ || ppack->punit == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block
                << "> t_ei_model=spitzer requires <materials> and <units>"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    spitzer_coulomb_log_ =
        pin->GetOrAddReal(block, "t_ei_coulomb_log", 10.0);
    spitzer_multiplier_ =
        pin->GetOrAddReal(block, "t_ei_spitzer_multiplier", 1.0);
    spitzer_temperature_floor_ =
        pin->GetOrAddReal(block, "t_ei_temperature_floor_kelvin", 1.0);
    if (!std::isfinite(spitzer_coulomb_log_) || spitzer_coulomb_log_ <= 0.0 ||
        !std::isfinite(spitzer_multiplier_) || spitzer_multiplier_ <= 0.0 ||
        !std::isfinite(spitzer_temperature_floor_) ||
        spitzer_temperature_floor_ < 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block
                << "> Spitzer Coulomb log and multiplier must be positive and the "
                << "temperature floor non-negative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    density_scale_cgs_ = ppack->punit->density_cgs();
    const Real velocity_cgs = ppack->punit->velocity_cgs();
    velocity_squared_cgs_ = velocity_cgs*velocity_cgs;
    time_scale_cgs_ = ppack->punit->time_cgs();
    if (!std::isfinite(density_scale_cgs_) || density_scale_cgs_ <= 0.0 ||
        !std::isfinite(velocity_squared_cgs_) || velocity_squared_cgs_ <= 0.0 ||
        !std::isfinite(time_scale_cgs_) || time_scale_cgs_ <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block
                << "> t_ei_model=spitzer requires finite positive physical units"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  if (cv_e_fraction_ <= 0.0 || cv_e_fraction_ >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block
              << "> electron_heat_capacity_fraction must lie strictly between 0 and 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (initial_temperature_ratio_ < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block
              << "> initial_electron_temperature_ratio must be non-negative"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // At fixed total internal energy, e_e/e_tot follows from cv_e*Te and cv_i*Ti.
  Real denominator = cv_i_fraction_ + cv_e_fraction_*initial_temperature_ratio_;
  initial_e_fraction_ = cv_e_fraction_*initial_temperature_ratio_/denominator;

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(temperature, nmb, 2, ncells3, ncells2, ncells1);
  Kokkos::realloc(thermodynamics, nmb, nthermodynamic_fields,
                  ncells3, ncells2, ncells1);
  Kokkos::deep_copy(temperature, 0.0);
  Kokkos::deep_copy(thermodynamics, 0.0);

  if (pin->DoesBlockExist("thermal_radiation") &&
      pin->GetOrAddBoolean("thermal_radiation", "enabled", true)) {
    pradiation = new ThermalRadiation(ppack, pin, iele + 1, iele,
        gamma_minus_one_, cv_e_fraction_, material_mixture);
  }
}

//----------------------------------------------------------------------------------------

TwoTemperature::~TwoTemperature() {
  if (pradiation != nullptr) delete pradiation;
}

//----------------------------------------------------------------------------------------

int TwoTemperature::NumberOfRadiationGroups() const {
  return (pradiation == nullptr) ? 0 : pradiation->ngroups;
}

//----------------------------------------------------------------------------------------
//! Initialize the redundant component energies from the total internal energy.

void TwoTemperature::Initialize(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int iion_ = iion;
  int iele_ = iele;
  Real gm1 = gamma_minus_one_;
  Real fi = cv_i_fraction_;
  Real fe = cv_e_fraction_;
  Real fe0 = initial_e_fraction_;
  auto temp = temperature;
  auto thermo = thermodynamics;

  if (!use_material_mixture_) {
    // Keep the legacy path separate so decks without <materials> retain their exact
    // arithmetic and results.
    par_for("two_temp_init", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real density = prim(m, IDN, k, j, i);
      Real eint = prim(m, IEN, k, j, i);
      Real eele = fe0*eint;
      Real eion = eint - eele;

      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      prim(m, iion_, k, j, i) = eion/density;
      prim(m, iele_, k, j, i) = eele/density;
      temp(m, 0, k, j, i) = gm1*eion/(density*fi);
      temp(m, 1, k, j, i) = gm1*eele/(density*fe);
    });
  } else {
    auto mixture = material_mixture_;
    Real initial_ratio = initial_temperature_ratio_;
    Real pressure_floor = pressure_floor_;
    Real temperature_floor = temperature_floor_;
    par_for("two_temp_material_init", DevExeSpace(), 0, nmb1, kl, ku, jl, ju,
            il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = prim(m, IDN, k, j, i);
      const Real eint_old = prim(m, IEN, k, j, i);
      const auto y0 = mixture.CompositionFromPrimitive(prim, m, k, j, i);
      materials::MaterialThermodynamicState state =
          mixture.InitialStateFromTotalSpecificEnergy(
              density, eint_old/density, y0, initial_ratio);
      const materials::MaterialPressureEnergyState floor =
          mixture.MinimumPressureEnergyState(
              density, y0, pressure_floor, temperature_floor);
      int query_flags = state.query_flags | floor.query_flags;
      if (mixture.UsesTabularEOS() &&
          (state.ion_specific_internal_energy <
               floor.ion_specific_internal_energy ||
           state.electron_specific_internal_energy <
               floor.electron_specific_internal_energy)) {
        query_flags |= materials::ionmix_energy_below_table;
      }
      state.ion_specific_internal_energy = fmax(
          state.ion_specific_internal_energy,
          floor.ion_specific_internal_energy);
      state.electron_specific_internal_energy = fmax(
          state.electron_specific_internal_energy,
          floor.electron_specific_internal_energy);
      if (mixture.UsesTabularEOS()) {
        state = mixture.StateFromRhoSpecificEnergies(
            density, state.ion_specific_internal_energy,
            state.electron_specific_internal_energy, y0);
      }
      query_flags |= state.query_flags;
      const Real eion = density*state.ion_specific_internal_energy;
      const Real eele = density*state.electron_specific_internal_energy;
      const Real eint = eion+eele;

      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      prim(m, iion_, k, j, i) = eion/density;
      prim(m, iele_, k, j, i) = eele/density;
      cons(m, IEN, k, j, i) += eint-eint_old;
      prim(m, IEN, k, j, i) = eint;
      StoreMaterialThermodynamics(
          temp, thermo, state, query_flags, m, k, j, i);
    });
  }
  if (pradiation != nullptr) {
    pradiation->Initialize(cons, prim, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------
//! Apply the FLASH/RAGE-like pressure partition to hydrodynamic work and shock heating.

void TwoTemperature::Sync(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                          int il, int iu, int jl, int ju, int kl, int ku) {
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int iion_ = iion;
  int iele_ = iele;
  Real gm1 = gamma_minus_one_;
  Real fi = cv_i_fraction_;
  Real fe = cv_e_fraction_;
  Real fe0 = initial_e_fraction_;
  auto temp = temperature;
  auto thermo = thermodynamics;

  if (!use_material_mixture_) {
    par_for("two_temp_sync", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real density = prim(m, IDN, k, j, i);
      Real eint = prim(m, IEN, k, j, i);
      Real eion_adv = fmax(cons(m, iion_, k, j, i), 0.0);
      Real eele_adv = fmax(cons(m, iele_, k, j, i), 0.0);
      Real component_sum = eion_adv + eele_adv;

      Real eele;
      if (component_sum > 0.0) {
        // With a common gamma, partial pressure is proportional to energy density.
        eele = eint*(eele_adv/component_sum);
      } else {
        eele = fe0*eint;
      }
      eele = fmin(fmax(eele, 0.0), eint);
      Real eion = eint - eele;

      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      prim(m, iion_, k, j, i) = eion/density;
      prim(m, iele_, k, j, i) = eele/density;
      temp(m, 0, k, j, i) = gm1*eion/(density*fi);
      temp(m, 1, k, j, i) = gm1*eele/(density*fe);
    });
  } else {
    auto mixture = material_mixture_;
    Real initial_ratio = initial_temperature_ratio_;
    Real pressure_floor = pressure_floor_;
    Real temperature_floor = temperature_floor_;
    par_for("two_temp_material_sync", DevExeSpace(), 0, nmb1, kl, ku, jl, ju,
            il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = prim(m, IDN, k, j, i);
      Real eint = prim(m, IEN, k, j, i);
      const auto y0 = mixture.CompositionFromPrimitive(prim, m, k, j, i);
      const Real eion_raw = cons(m, iion_, k, j, i);
      const Real eele_raw = cons(m, iele_, k, j, i);
      const Real eion_adv = fmax(eion_raw, 0.0);
      const Real eele_adv = fmax(eele_raw, 0.0);
      const Real component_sum = eion_adv+eele_adv;
      const materials::MaterialPressureEnergyState floor =
          mixture.MinimumPressureEnergyState(
              density, y0, pressure_floor, temperature_floor);
      int query_flags = floor.query_flags;
      const Real eion_floor = density*floor.ion_specific_internal_energy;
      const Real eele_floor = density*floor.electron_specific_internal_energy;
      const Real minimum_sum = eion_floor+eele_floor;
      if (mixture.UsesTabularEOS() &&
          (eion_raw < 0.0 || eele_raw < 0.0 ||
           eion_adv < eion_floor || eele_adv < eele_floor)) {
        query_flags |= materials::ionmix_energy_below_table;
      }
      if (eint < minimum_sum) {
        if (mixture.UsesTabularEOS()) {
          query_flags |= materials::ionmix_energy_below_table;
        }
        cons(m, IEN, k, j, i) += minimum_sum-eint;
        eint = minimum_sum;
        prim(m, IEN, k, j, i) = eint;
      }

      materials::MaterialPressureEnergyState adv_state;
      Real ion_fraction;
      if (component_sum > 0.0) {
        adv_state = mixture.PressureEnergyFromRhoSpecificEnergies(
            density, fmax(eion_adv, eion_floor)/density,
            fmax(eele_adv, eele_floor)/density, y0);
        query_flags |= adv_state.query_flags;
        const Real pressure_sum =
            adv_state.ion_pressure+adv_state.electron_pressure;
        ion_fraction = (pressure_sum > 0.0)
            ? adv_state.ion_pressure/pressure_sum
            : eion_adv/component_sum;
      } else {
        ion_fraction = 1.0-mixture.InitialElectronEnergyFraction(
            y0, initial_ratio);
      }
      const Real residual = eint-component_sum;
      const Real candidate_ion = eion_adv+ion_fraction*residual;
      const Real candidate_electron = eele_adv+(1.0-ion_fraction)*residual;
      if (mixture.UsesTabularEOS() &&
          (candidate_ion < eion_floor || candidate_electron < eele_floor)) {
        query_flags |= materials::ionmix_energy_below_table;
      }
      Real ion_extra = fmax(candidate_ion-eion_floor, 0.0);
      Real electron_extra = fmax(candidate_electron-eele_floor, 0.0);
      const Real available = eint-minimum_sum;
      const Real extra_sum = ion_extra+electron_extra;
      if (extra_sum > 0.0) {
        ion_extra *= available/extra_sum;
        electron_extra *= available/extra_sum;
      } else {
        ion_extra = ion_fraction*available;
        electron_extra = (1.0-ion_fraction)*available;
      }
      const Real eion = eion_floor+ion_extra;
      const Real eele = eele_floor+electron_extra;
      const materials::MaterialThermodynamicState state =
          mixture.StateFromRhoSpecificEnergies(
              density, eion/density, eele/density, y0);
      query_flags |= state.query_flags;

      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      prim(m, iion_, k, j, i) = eion/density;
      prim(m, iele_, k, j, i) = eele/density;
      StoreMaterialThermodynamics(
          temp, thermo, state, query_flags, m, k, j, i);
    });
  }
  if (pradiation != nullptr) {
    pradiation->UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------
//! Exact solution of the constant-t_ei FLASH ion/electron heat-exchange equations.

void TwoTemperature::Exchange(Real dt, DvceArray5D<Real> &cons,
                              DvceArray5D<Real> &prim,
                              int il, int iu, int jl, int ju, int kl, int ku) {
  if (!use_material_mixture_ && t_ei_ >= 0.0) {
    int nmb1 = pmy_pack_->nmb_thispack - 1;
    int iion_ = iion;
    int iele_ = iele;
    Real gm1 = gamma_minus_one_;
    Real fi = cv_i_fraction_;
    Real fe = cv_e_fraction_;
    Real decay = 0.0;
    if (t_ei_ > 0.0) {
      // FLASH: Delta T decays as exp[-(1 + cv_e/cv_i) dt/t_ei].
      decay = exp(-(1.0 + fe/fi)*dt/t_ei_);
    }
    auto temp = temperature;

    par_for("two_temp_exchange", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real density = cons(m, IDN, k, j, i);
      Real eion = cons(m, iion_, k, j, i);
      Real eele = cons(m, iele_, k, j, i);
      Real eint = eion + eele;

      Real tion = gm1*eion/(density*fi);
      Real tele = gm1*eele/(density*fe);
      Real teq = fi*tion + fe*tele;
      Real delta_t = (tion - tele)*decay;
      Real tion_new = teq + fe*delta_t;

      Real eion_new = density*fi*tion_new/gm1;
      eion_new = fmin(fmax(eion_new, 0.0), eint);
      Real eele_new = eint - eion_new;

      cons(m, iion_, k, j, i) = eion_new;
      cons(m, iele_, k, j, i) = eele_new;
      prim(m, iion_, k, j, i) = eion_new/density;
      prim(m, iele_, k, j, i) = eele_new/density;
      temp(m, 0, k, j, i) = gm1*eion_new/(density*fi);
      temp(m, 1, k, j, i) = gm1*eele_new/(density*fe);
    });
  } else if (use_material_mixture_) {
    int nmb1 = pmy_pack_->nmb_thispack - 1;
    int iion_ = iion;
    int iele_ = iele;
    Real gm1 = gamma_minus_one_;
    auto mixture = material_mixture_;
    auto temp = temperature;
    auto thermo = thermodynamics;
    bool use_spitzer = use_spitzer_exchange_;
    Real coulomb_log = spitzer_coulomb_log_;
    Real spitzer_multiplier = spitzer_multiplier_;
    Real temperature_floor = spitzer_temperature_floor_;
    Real density_scale = density_scale_cgs_;
    Real velocity_squared = velocity_squared_cgs_;
    Real time_scale = time_scale_cgs_;
    Real pressure_floor = pressure_floor_;
    Real table_temperature_floor = temperature_floor_;
    const bool radiation_refreshes_cache = pradiation != nullptr;

    par_for("two_temp_material_exchange", DevExeSpace(), 0, nmb1, kl, ku, jl,
            ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = cons(m, IDN, k, j, i);
      const auto y0 = mixture.CompositionFromConserved(
          cons, m, k, j, i);
      const Real eion = cons(m, iion_, k, j, i);
      const Real eele = cons(m, iele_, k, j, i);
      const Real eint = eion+eele;
      if (mixture.UsesTabularEOS()) {
        // Sync filled this cache after the final RK stage.  Reusing it avoids another
        // ion/electron inverse and a redundant finite-difference sound-speed evaluation.
        materials::MaterialThermodynamicState old_state;
        old_state.ion_temperature = temp(m, 0, k, j, i);
        old_state.electron_temperature = temp(m, 1, k, j, i);
        old_state.electron_number_density_cgs = thermo(
            m, electron_number_density_cgs, k, j, i);
        old_state.effective_charge = thermo(m, effective_charge, k, j, i);
        old_state.query_flags = static_cast<int>(thermo(
            m, eos_query_flags, k, j, i));
        Real exchange_time = mixture.ExchangeTime(y0);
        if (use_spitzer) {
          const Real tion_kelvin = fmax(
              old_state.ion_temperature*mixture.temperature_to_kelvin,
              temperature_floor);
          const Real tele_kelvin = fmax(
              old_state.electron_temperature*mixture.temperature_to_kelvin,
              temperature_floor);
          const Real exchange_seconds = SpitzerExchangeTime(
              old_state.electron_number_density_cgs,
              tion_kelvin, tele_kelvin, mixture.MeanAtomicMass(y0),
              old_state.effective_charge, coulomb_log);
          exchange_time = spitzer_multiplier*exchange_seconds/time_scale;
        }
        if (exchange_time < 0.0 || !Kokkos::isfinite(exchange_time)) return;

        Real decay = 0.0;
        if (exchange_time > 0.0) {
          const Real local_fe = mixture.ElectronHeatCapacityFraction(
              density, old_state.ion_temperature,
              old_state.electron_temperature, y0);
          if (local_fe < 1.0) {
            const Real local_fi = fmax(1.0-local_fe,
                Kokkos::Experimental::epsilon<Real>::value);
            decay = exp(-(1.0+local_fe/local_fi)*dt/exchange_time);
          }
        }
        const Real target_difference =
            (old_state.electron_temperature-old_state.ion_temperature)*decay;
        const materials::MaterialPressureEnergyState floor =
            mixture.MinimumPressureEnergyState(
                density, y0, pressure_floor, table_temperature_floor);
        int query_flags = old_state.query_flags | floor.query_flags;
        const Real minimum_ion = density*floor.ion_specific_internal_energy;
        const Real minimum_electron =
            density*floor.electron_specific_internal_energy;
        if (eint < minimum_ion+minimum_electron) {
          query_flags |= materials::ionmix_energy_below_table;
          thermo(m, eos_query_flags, k, j, i) = static_cast<Real>(query_flags);
          return;
        }
        const materials::MaterialTransientExchangeState exchange =
            mixture.StateTemperaturesFromRhoTotalEnergyTemperatureDifference(
                density, eion/density, eele/density,
                old_state.ion_temperature,
                old_state.electron_temperature, target_difference, y0);
        query_flags |= exchange.temperatures.query_flags;
        if (exchange.used_fallback == 2) {
          // A failed/clamped bracket means no exchange. Keep the authoritative Sync
          // state and exact conservative split; only retain diagnostics from the failed
          // queries. Radiation coupling, if enabled, still runs after this kernel.
          thermo(m, eos_query_flags, k, j, i) = static_cast<Real>(query_flags);
          return;
        }
        Real eion_new = density*exchange.ion_specific_internal_energy;
        const Real bounded_eion =
            fmin(fmax(eion_new, minimum_ion), eint-minimum_electron);
        if (bounded_eion != eion_new) {
          query_flags |= materials::ionmix_energy_below_table;
        }
        eion_new = bounded_eion;
        const Real eele_new = eint-eion_new;
        materials::MaterialTemperatureState state = exchange.temperatures;
        // The temperature-space solve respects component floors when Sync supplied a
        // valid old state.  Retain a conservative inverse fallback only for a clamped,
        // externally inconsistent cell.
        if (eion_new != density*exchange.ion_specific_internal_energy) {
          const materials::MaterialThermodynamicState bounded_state =
              mixture.StateFromRhoSpecificEnergiesNoSound(
                  density, eion_new/density, eele_new/density, y0);
          state.ion_temperature = bounded_state.ion_temperature;
          state.electron_temperature = bounded_state.electron_temperature;
          state.query_flags = bounded_state.query_flags;
        }
        query_flags |= state.query_flags;
        if (!radiation_refreshes_cache) {
          const materials::MaterialThermodynamicState full_state =
              mixture.StateFromRhoTemperatures(
                  density, state.ion_temperature,
                  state.electron_temperature, y0);
          query_flags |= full_state.query_flags;
          cons(m, iion_, k, j, i) = eion_new;
          cons(m, iele_, k, j, i) = eele_new;
          prim(m, iion_, k, j, i) = eion_new/density;
          prim(m, iele_, k, j, i) = eele_new/density;
          StoreMaterialThermodynamics(
              temp, thermo, full_state, query_flags, m, k, j, i);
          return;
        }
        cons(m, iion_, k, j, i) = eion_new;
        cons(m, iele_, k, j, i) = eele_new;
        prim(m, iion_, k, j, i) = eion_new/density;
        prim(m, iele_, k, j, i) = eele_new/density;
        // Coupling consumes only the canonical electron temperature.  The full
        // post-coupling refresh overwrites every other thermodynamic cache field.
        StoreMaterialTemperaturesAndFlags(
            temp, thermo, state, query_flags, m, k, j, i);
        return;
      }

      const Real local_fe = mixture.ElectronHeatCapacityFraction(y0);
      const Real local_fi = 1.0-local_fe;
      const Real tion = gm1*eion/(density*local_fi);
      const Real tele = gm1*eele/(density*local_fe);
      Real exchange_time = mixture.ExchangeTime(y0);
      if (use_spitzer) {
        constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;
        constexpr Real boltzmann_cgs = 1.3806488e-16;
        const Real kelvin_per_code_temperature = velocity_squared*
            mixture.MeanParticleMass(y0)*atomic_mass_unit_cgs/boltzmann_cgs;
        const Real tion_kelvin = fmax(
            tion*kelvin_per_code_temperature, temperature_floor);
        const Real tele_kelvin = fmax(
            tele*kelvin_per_code_temperature, temperature_floor);
        const Real electron_density = mixture.ElectronNumberDensityCgs(
            density, density_scale, y0);
        const Real exchange_seconds = SpitzerExchangeTime(
            electron_density, tion_kelvin, tele_kelvin,
            mixture.MeanAtomicMass(y0), mixture.EffectiveCharge(y0), coulomb_log);
        exchange_time = spitzer_multiplier*exchange_seconds/time_scale;
      }
      if (exchange_time < 0.0 || !Kokkos::isfinite(exchange_time)) return;

      const Real teq = local_fi*tion+local_fe*tele;
      Real decay = 0.0;
      if (exchange_time > 0.0) {
        decay = exp(-(1.0+local_fe/local_fi)*dt/exchange_time);
      }
      const Real delta_t = (tion-tele)*decay;
      const Real tion_new = teq+local_fe*delta_t;

      Real eion_new = density*local_fi*tion_new/gm1;
      eion_new = fmin(fmax(eion_new, 0.0), eint);
      const Real eele_new = eint-eion_new;
      cons(m, iion_, k, j, i) = eion_new;
      cons(m, iele_, k, j, i) = eele_new;
      prim(m, iion_, k, j, i) = eion_new/density;
      prim(m, iele_, k, j, i) = eele_new/density;
      temp(m, 0, k, j, i) = gm1*eion_new/(density*local_fi);
      temp(m, 1, k, j, i) = gm1*eele_new/(density*local_fe);
    });
  }

  if (pradiation != nullptr) {
    pradiation->Couple(
        dt, cons, prim, temperature, pressure_floor_, temperature_floor_,
        il, iu, jl, ju, kl, ku);
    // Radiation coupling is the final operator-split source in this task.  It changes
    // electron and total material energy after the cache filled above, so refresh the
    // complete tabular state before the next MHD flux, timestep, laser, or Biermann use.
    RefreshMaterialThermodynamics(cons, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------

void TwoTemperature::RefreshMaterialThermodynamics(
    const DvceArray5D<Real> &cons, int il, int iu, int jl, int ju,
    int kl, int ku) {
  if (!use_material_mixture_ || !material_mixture_.UsesTabularEOS()) return;

  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int iion_ = iion;
  const int iele_ = iele;
  auto mixture = material_mixture_;
  auto temp = temperature;
  auto thermo = thermodynamics;
  par_for("two_temp_refresh_material_thermodynamics", DevExeSpace(), 0, nmb1,
          kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real density = cons(m, IDN, k, j, i);
    const auto y0 = mixture.CompositionFromConserved(
        cons, m, k, j, i);
    const materials::MaterialThermodynamicState state =
        mixture.StateFromRhoSpecificEnergies(
            density, cons(m, iion_, k, j, i)/density,
            cons(m, iele_, k, j, i)/density, y0);
    StoreMaterialThermodynamics(
        temp, thermo, state, state.query_flags, m, k, j, i);
  });
}

//----------------------------------------------------------------------------------------
//! Close one Biermann RK stage and refresh the two-temperature state.
//!
//! The independent Biermann variables are conservative total energy, face-centred B,
//! and electron internal energy.  Ion internal energy is redundant, so reconstruct it
//! from the total internal energy selected by ConsToPrim.  This preserves electron
//! transport/work, exact conservative total energy (unless a physical floor is needed),
//! and the component/total closure used by the next stage.  In a cancellation-dominated
//! dual-energy cell ConsToPrim selects the auxiliary component sum; the reconstruction
//! then retains that fallback rather than forcing an ill-conditioned subtraction.

void TwoTemperature::CloseBiermannStage(
    DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
    int il, int iu, int jl, int ju, int kl, int ku,
    bool full_thermodynamics) {
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int iion_ = iion;
  const int iele_ = iele;
  auto w = prim;
  const auto &eos = pmy_pack_->pmhd->peos->eos_data;
  const BiermannEndpointClosure closure{
      material_mixture_, gamma_minus_one_, eos.dfloor, eos.pfloor, eos.tfloor,
      eos.sfloor, eos.sigma_max, pmy_pack_->pmhd->dual_energy_eta1,
      pmy_pack_->pmhd->use_dual_energy, use_material_mixture_,
      use_material_mixture_ && material_mixture_.UsesTabularEOS()};

  if (!use_material_mixture_) {
    const Real gm1 = gamma_minus_one_;
    const Real fi = cv_i_fraction_;
    const Real fe = cv_e_fraction_;
    auto temp = temperature;
    par_for("two_temp_refresh_components", DevExeSpace(), 0, nmb1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = w(m, IDN, k, j, i);
      const BiermannClosedState closed = closure.CloseSelected(
          density, w(m, IEN, k, j, i), cons(m, iele_, k, j, i), 0.0);
      const Real eele = closed.electron_energy;
      const Real eion = closed.ion_energy;
      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      w(m, iion_, k, j, i) = eion/density;
      w(m, iele_, k, j, i) = eele/density;
      temp(m, 0, k, j, i) = gm1*eion/(density*fi);
      temp(m, 1, k, j, i) = gm1*eele/(density*fe);
    });
  } else if (!full_thermodynamics && material_mixture_.UsesTabularEOS()) {
    auto mixture = material_mixture_;
    auto temp = temperature;
    auto thermo = thermodynamics;
    par_for("two_temp_refresh_biermann_electron", DevExeSpace(), 0, nmb1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = w(m, IDN, k, j, i);
      const auto y0 = mixture.CompositionFromPrimitive(w, m, k, j, i);
      const Real eele_raw = cons(m, iele_, k, j, i);
      const Real eint_before = w(m, IEN, k, j, i);
      const BiermannClosedState closed = closure.CloseSelected(
          density, eint_before, eele_raw, y0);
      if (closed.internal_energy > eint_before) {
        cons(m, IEN, k, j, i) += closed.internal_energy-eint_before;
        w(m, IEN, k, j, i) = closed.internal_energy;
      }
      const Real eele = closed.electron_energy;
      const Real eion = closed.ion_energy;
      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      w(m, iion_, k, j, i) = eion/density;
      w(m, iele_, k, j, i) = eele/density;
      const int ion_query_flags = mixture.IonSpecificEnergyQueryFlags(
          density, eion/density, closed.composition);
      const materials::MaterialElectronState state =
          mixture.ElectronStateFromRhoSpecificEnergy(
              density, eele/density, closed.composition);
      StoreMaterialElectronState(
          temp, thermo, state, closed.query_flags | ion_query_flags, m, k, j, i);
    });
  } else {
    auto mixture = material_mixture_;
    auto temp = temperature;
    auto thermo = thermodynamics;
    par_for("two_temp_refresh_material_components", DevExeSpace(), 0, nmb1,
            kl, ku, jl, ju, il, iu,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real density = w(m, IDN, k, j, i);
      const auto y0 = mixture.CompositionFromPrimitive(w, m, k, j, i);
      const Real eele_raw = cons(m, iele_, k, j, i);
      const Real eint_before = w(m, IEN, k, j, i);
      const BiermannClosedState closed = closure.CloseSelected(
          density, eint_before, eele_raw, y0);
      if (closed.internal_energy > eint_before) {
        cons(m, IEN, k, j, i) += closed.internal_energy-eint_before;
        w(m, IEN, k, j, i) = closed.internal_energy;
      }
      const Real eele = closed.electron_energy;
      const Real eion = closed.ion_energy;
      cons(m, iion_, k, j, i) = eion;
      cons(m, iele_, k, j, i) = eele;
      w(m, iion_, k, j, i) = eion/density;
      w(m, iele_, k, j, i) = eele/density;
      const materials::MaterialThermodynamicState state =
          mixture.StateFromRhoSpecificEnergies(
              density, eion/density, eele/density,
              closed.composition);
      StoreMaterialThermodynamics(
          temp, thermo, state, closed.query_flags, m, k, j, i);
    });
  }
  if (full_thermodynamics && pradiation != nullptr) {
    pradiation->UpdateDiagnostics(cons, prim, il, iu, jl, ju, kl, ku);
  }
}

//----------------------------------------------------------------------------------------

void TwoTemperature::AddRadiationFluxes(const DvceArray5D<Real> &prim,
                                        DvceFaceFld5D<Real> &flx) {
  if (pradiation != nullptr) pradiation->AddFluxes(prim, temperature, flx);
}

//----------------------------------------------------------------------------------------

void TwoTemperature::RadiationNewTimeStep(const DvceArray5D<Real> &prim) {
  if (pradiation != nullptr) pradiation->NewTimeStep(prim, temperature);
}

} // namespace two_temperature
