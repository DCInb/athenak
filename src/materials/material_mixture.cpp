//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.cpp
//! \brief Input parsing for ideal and tabular two-material plasma closures.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "materials/material_mixture.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"

namespace materials {
namespace {

[[noreturn]] void MaterialInputError(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "<materials> " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

std::string Key(const int material, const std::string &property) {
  return "material"+std::to_string(material)+"_"+property;
}

SpeciesProperties ReadMaterial(ParameterInput *pin, const int material,
                               const Real default_exchange_time) {
  SpeciesProperties result;
  result.abar = pin->GetReal("materials", Key(material, "abar"));
  result.zbar = pin->GetReal("materials", Key(material, "zbar"));
  result.zeff = pin->GetOrAddReal(
      "materials", Key(material, "zeff"), result.zbar);
  result.t_ei = pin->GetOrAddReal(
      "materials", Key(material, "t_ei"), default_exchange_time);
  if (!std::isfinite(result.abar) || !(result.abar > 0.0)) {
    MaterialInputError(Key(material, "abar")+" must be finite and positive");
  }
  if (!std::isfinite(result.zbar) || !(result.zbar > 0.0)) {
    MaterialInputError(Key(material, "zbar")+" must be finite and positive");
  }
  if (!std::isfinite(result.zeff) || !(result.zeff > 0.0)) {
    MaterialInputError(Key(material, "zeff")+" must be finite and positive");
  }
  if (!std::isfinite(result.t_ei)) {
    MaterialInputError(Key(material, "t_ei")+" must be finite");
  }
  return result;
}

void CheckAbar(const int material, const SpeciesProperties &properties,
               const IonmixTwoTemperatureTableMetadata &metadata) {
  const Real scale = std::max(static_cast<Real>(1.0), std::abs(metadata.abar));
  if (std::abs(properties.abar-metadata.abar) >
      64.0*std::numeric_limits<Real>::epsilon()*scale) {
    MaterialInputError(
        Key(material, "abar")+" does not match abar="+
        std::to_string(metadata.abar)+" in its EOS table");
  }
}

} // namespace

MaterialMixture::MaterialMixture(ParameterInput *pin, const int first_user_scalar,
                                 const int nuser_scalars, const Real gamma,
                                 units::Units *unit_system) {
  const int number_of_materials =
      pin->GetOrAddInteger("materials", "nmaterials", 2);
  if (number_of_materials != 2) {
    MaterialInputError("this closure requires nmaterials=2");
  }
  const int relative_scalar =
      pin->GetOrAddInteger("materials", "scalar_index", 0);
  if (relative_scalar < 0 || relative_scalar >= nuser_scalars) {
    MaterialInputError("scalar_index must select a user passive scalar in <mhd>");
  }
  constexpr Real monatomic_gamma = 5.0/3.0;
  if (!std::isfinite(gamma) ||
      std::abs(gamma-monatomic_gamma) > 32.0*std::numeric_limits<Real>::epsilon()) {
    MaterialInputError("two-material MHD currently requires <mhd>/gamma=5/3");
  }

  const Real default_exchange_time = pin->GetOrAddReal("mhd", "t_ei", -1.0);
  if (!std::isfinite(default_exchange_time)) {
    MaterialInputError("the default <mhd>/t_ei must be finite");
  }
  data_.material0 = ReadMaterial(pin, 0, default_exchange_time);
  data_.material1 = ReadMaterial(pin, 1, default_exchange_time);
  data_.scalar_index = first_user_scalar+relative_scalar;
  data_.gamma_minus_one = gamma-1.0;

  pin->GetOrAddString("materials", "material0_name", "material0");
  pin->GetOrAddString("materials", "material1_name", "material1");

  const bool has_table0 = pin->DoesParameterExist(
      "materials", "material0_eos_table_file");
  const bool has_table1 = pin->DoesParameterExist(
      "materials", "material1_eos_table_file");
  if (has_table0 != has_table1) {
    MaterialInputError(
        "material0_eos_table_file and material1_eos_table_file must be supplied together");
  }
  if (!has_table0) {
    if (unit_system != nullptr) {
      data_.density_to_cgs = unit_system->density_cgs();
      data_.temperature_to_kelvin = unit_system->temperature_cgs();
    }
    return;
  }
  if (unit_system == nullptr) {
    MaterialInputError("tabular two-temperature EOS requires a <units> block");
  }

  const std::string mixing_rule = pin->GetOrAddString(
      "materials", "eos_mixing_rule", "partial_density_additive");
  if (mixing_rule != "partial_density_additive") {
    MaterialInputError(
        "eos_mixing_rule must be 'partial_density_additive'");
  }
  const std::string bounds = pin->GetOrAddString(
      "materials", "eos_table_bounds", "error");
  const std::string interpolation = pin->GetOrAddString(
      "materials", "eos_table_interpolation", "geometric");

  IonmixTwoTemperatureTableOptions options;
  if (bounds == "clamp") {
    options.bounds_policy = IonmixBoundsPolicy::clamp;
  } else if (bounds == "error") {
    options.bounds_policy = IonmixBoundsPolicy::error;
  } else {
    MaterialInputError("eos_table_bounds must be 'clamp' or 'error'");
  }
  if (interpolation == "geometric") {
    options.geometric_interpolation = true;
  } else if (interpolation == "linear") {
    options.geometric_interpolation = false;
  } else {
    MaterialInputError("eos_table_interpolation must be 'geometric' or 'linear'");
  }

  const Real velocity_cgs = unit_system->velocity_cgs();
  options.density_to_cgs = pin->GetOrAddReal(
      "materials", "eos_table_density_to_cgs", unit_system->density_cgs());
  options.temperature_to_kelvin = pin->GetOrAddReal(
      "materials", "eos_table_temperature_to_kelvin",
      unit_system->temperature_cgs());
  options.pressure_from_cgs = pin->GetOrAddReal(
      "materials", "eos_table_pressure_from_cgs",
      1.0/unit_system->pressure_cgs());
  options.specific_energy_from_cgs = pin->GetOrAddReal(
      "materials", "eos_table_specific_energy_from_cgs",
      1.0/(velocity_cgs*velocity_cgs));

  data_.wave_speed_safety = pin->GetOrAddReal(
      "materials", "eos_wave_speed_safety", 1.05);
  if (!std::isfinite(data_.wave_speed_safety) || data_.wave_speed_safety < 1.0) {
    MaterialInputError("eos_wave_speed_safety must be finite and at least one");
  }

  const std::string file0 = pin->GetString(
      "materials", "material0_eos_table_file");
  const std::string file1 = pin->GetString(
      "materials", "material1_eos_table_file");
  material0_table_ = std::make_unique<IonmixTwoTemperatureTable>(file0, options);
  material1_table_ = std::make_unique<IonmixTwoTemperatureTable>(file1, options);
  CheckAbar(0, data_.material0, material0_table_->Metadata());
  CheckAbar(1, data_.material1, material1_table_->Metadata());

  const std::string fingerprint0 = pin->GetOrAddString(
      "materials", "material0_eos_table_fingerprint",
      material0_table_->Metadata().file_fingerprint);
  const std::string fingerprint1 = pin->GetOrAddString(
      "materials", "material1_eos_table_fingerprint",
      material1_table_->Metadata().file_fingerprint);
  if (fingerprint0 != material0_table_->Metadata().file_fingerprint) {
    MaterialInputError(
        "material0_eos_table_fingerprint does not match the loaded file");
  }
  if (fingerprint1 != material1_table_->Metadata().file_fingerprint) {
    MaterialInputError(
        "material1_eos_table_fingerprint does not match the loaded file");
  }

  data_.material0_table = material0_table_->DeviceData();
  data_.material1_table = material1_table_->DeviceData();
  data_.density_to_cgs = options.density_to_cgs;
  data_.temperature_to_kelvin = options.temperature_to_kelvin;
  data_.use_tabular_eos = true;
}

} // namespace materials
