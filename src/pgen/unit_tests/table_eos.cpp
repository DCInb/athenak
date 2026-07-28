//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file table_eos.cpp
//! \brief Device-side forward/inverse and material-metadata tests for table EOS data.

#include <math.h>

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

namespace {

struct ExpectedTableState {
  Real temperature;
  Real pressure;
  Real specific_internal_energy;
  Real sound_speed_squared;
  Real gamma1;
  Real gamma3_minus_one;
  Real mean_ionization;
  Real effective_charge;
  Real mean_atomic_mass;
  Real mean_molecular_weight;
  bool has_gamma1;
  bool has_gamma3_minus_one;
  bool has_mean_ionization;
  bool has_effective_charge;
  bool has_mean_atomic_mass;
  bool has_mean_molecular_weight;
};

KOKKOS_INLINE_FUNCTION
bool NearlyEqual(Real actual, Real expected, Real tolerance) {
  return Kokkos::isfinite(actual) &&
         fabs(actual-expected) <= tolerance*fmax(1.0, fabs(expected));
}

KOKKOS_INLINE_FUNCTION
void CheckOptional(bool actual_has, Real actual, bool expected_has, Real expected,
                   Real tolerance, int &failures) {
  if (actual_has != expected_has) ++failures;
  if (expected_has) {
    if (!NearlyEqual(actual, expected, tolerance)) ++failures;
  } else if (actual == actual) {
    // An unavailable value must remain NaN so callers cannot mistake a default for data.
    ++failures;
  }
}

KOKKOS_INLINE_FUNCTION
void CheckState(const EOS_Data::ThermoState &state,
                const ExpectedTableState &expected, Real tolerance, int &failures) {
  if (!NearlyEqual(state.temperature, expected.temperature, tolerance)) ++failures;
  if (!NearlyEqual(state.pressure, expected.pressure, tolerance)) ++failures;
  if (!NearlyEqual(state.specific_internal_energy,
                   expected.specific_internal_energy, tolerance)) ++failures;
  if (!NearlyEqual(state.sound_speed_squared,
                   expected.sound_speed_squared, tolerance)) ++failures;
  CheckOptional(state.has_gamma1, state.gamma1,
                expected.has_gamma1, expected.gamma1, tolerance, failures);
  CheckOptional(state.has_gamma3_minus_one, state.gamma3_minus_one,
                expected.has_gamma3_minus_one, expected.gamma3_minus_one,
                tolerance, failures);
  CheckOptional(state.has_mean_ionization, state.mean_ionization,
                expected.has_mean_ionization, expected.mean_ionization,
                tolerance, failures);
  CheckOptional(state.has_effective_charge, state.effective_charge,
                expected.has_effective_charge, expected.effective_charge,
                tolerance, failures);
  CheckOptional(state.has_mean_atomic_mass, state.mean_atomic_mass,
                expected.has_mean_atomic_mass, expected.mean_atomic_mass,
                tolerance, failures);
  CheckOptional(state.has_mean_molecular_weight, state.mean_molecular_weight,
                expected.has_mean_molecular_weight, expected.mean_molecular_weight,
                tolerance, failures);
}

} // namespace

void ProblemGenerator::TableEOSUnitTest(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || !pmbp->phydro->peos->eos_data.is_table) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "table_eos unit test requires <hydro>/eos=table"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real density = pin->GetReal("problem", "query_density");
  ExpectedTableState expected;
  expected.temperature = pin->GetReal("problem", "query_temperature");
  expected.pressure = pin->GetReal("problem", "expected_pressure");
  expected.specific_internal_energy =
      pin->GetReal("problem", "expected_specific_eint");
  expected.sound_speed_squared = pin->GetReal("problem", "expected_sound_speed2");
  expected.has_gamma1 = pin->GetOrAddBoolean(
      "problem", "expected_has_gamma1", false);
  expected.has_gamma3_minus_one = pin->GetOrAddBoolean(
      "problem", "expected_has_gamma3m1", false);
  expected.has_mean_ionization = pin->GetOrAddBoolean(
      "problem", "expected_has_zbar", false);
  expected.has_effective_charge = pin->GetOrAddBoolean(
      "problem", "expected_has_zeff", false);
  expected.has_mean_atomic_mass = pin->GetOrAddBoolean(
      "problem", "expected_has_abar", false);
  expected.has_mean_molecular_weight = pin->GetOrAddBoolean(
      "problem", "expected_has_mu", false);
  expected.gamma1 = expected.has_gamma1 ?
      pin->GetReal("problem", "expected_gamma1") : 0.0;
  expected.gamma3_minus_one = expected.has_gamma3_minus_one ?
      pin->GetReal("problem", "expected_gamma3m1") : 0.0;
  expected.mean_ionization = expected.has_mean_ionization ?
      pin->GetReal("problem", "expected_zbar") : 0.0;
  expected.effective_charge = expected.has_effective_charge ?
      pin->GetReal("problem", "expected_zeff") : 0.0;
  expected.mean_atomic_mass = expected.has_mean_atomic_mass ?
      pin->GetReal("problem", "expected_abar") : 0.0;
  expected.mean_molecular_weight = expected.has_mean_molecular_weight ?
      pin->GetReal("problem", "expected_mu") : 0.0;
  Real tolerance = pin->GetOrAddReal("problem", "tolerance", 2.0e-12);

  EOS_Data eos = pmbp->phydro->peos->eos_data;
  Real eint_density = density*expected.specific_internal_energy;
  int failures = 0;
  Kokkos::parallel_reduce(
      "table_eos_api_test", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(int, int &local_failures) {
        CheckState(eos.EvalThermoStateFromRhoTemperature(
                       density, expected.temperature),
                   expected, tolerance, local_failures);
        CheckState(eos.EvalThermoStateFromRhoEint(density, eint_density),
                   expected, tolerance, local_failures);
        CheckState(eos.EvalThermoStateFromRhoPressure(
                       density, expected.pressure),
                   expected, tolerance, local_failures);

        if (!NearlyEqual(eos.PressureFromRhoTemperature(
                             density, expected.temperature),
                         expected.pressure, tolerance)) ++local_failures;
        if (!NearlyEqual(eos.SpecificEintFromRhoTemperature(
                             density, expected.temperature),
                         expected.specific_internal_energy,
                         tolerance)) ++local_failures;
        if (!NearlyEqual(eos.TemperatureFromRhoEint(density, eint_density),
                         expected.temperature, tolerance)) ++local_failures;
        if (!NearlyEqual(eos.TemperatureFromRhoPressure(
                             density, expected.pressure),
                         expected.temperature, tolerance)) ++local_failures;
        if (!NearlyEqual(eos.PressureFromRhoEint(density, eint_density),
                         expected.pressure, tolerance)) ++local_failures;
        if (!NearlyEqual(eos.InternalEnergyDensityFromRhoPressure(
                             density, expected.pressure),
                         eint_density, tolerance)) ++local_failures;
        if (!NearlyEqual(eos.HydroSoundSpeed2FromRhoEint(
                             density, eint_density),
                         expected.sound_speed_squared,
                         tolerance)) ++local_failures;

        if (eos.table.has_gamma1 != expected.has_gamma1) ++local_failures;
        if (eos.table.has_gamma3_minus_one !=
            expected.has_gamma3_minus_one) ++local_failures;
        if (eos.table.has_mean_ionization !=
            expected.has_mean_ionization) ++local_failures;
        if (eos.table.has_effective_charge !=
            expected.has_effective_charge) ++local_failures;
        if (eos.table.has_mean_atomic_mass !=
            expected.has_mean_atomic_mass) ++local_failures;
        if (eos.table.has_mean_molecular_weight !=
            expected.has_mean_molecular_weight) ++local_failures;
      }, Kokkos::Sum<int>(failures));

  if (failures != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "table_eos unit test recorded " << failures
              << " failed API checks" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &w0 = pmbp->phydro->w0;
  par_for("table_eos_test_init", DevExeSpace(), 0, pmbp->nmb_thispack-1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m, IDN, k, j, i) = density;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = 0.0;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = eint_density;
  });
  pmbp->phydro->peos->PrimToCons(
      w0, pmbp->phydro->u0, indcs.is, indcs.ie, indcs.js, indcs.je,
      indcs.ks, indcs.ke);

  std::cout << "Table EOS material API test passed" << std::endl;
}
