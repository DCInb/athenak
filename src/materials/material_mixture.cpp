//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.cpp
//! \brief Input parsing for ideal and tabular multi-material plasma closures.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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

MaterialMixture::MaterialMixture(ParameterInput *pin, const std::string &block,
                                 const int first_user_scalar,
                                 const int nuser_scalars, const Real gamma,
                                 units::Units *unit_system) {
  const int number_of_materials =
      pin->GetOrAddInteger("materials", "nmaterials", 2);
  if (number_of_materials < 1) {
    MaterialInputError("nmaterials must be a positive integer");
  }
  data_.nmaterials = number_of_materials;
  // Every material has an explicit mass fraction. scalar_index names the first of a
  // contiguous run of nmaterials passive scalars.
  const int relative_scalar =
      pin->GetOrAddInteger("materials", "scalar_index", 0);
  if (relative_scalar < 0 || relative_scalar > nuser_scalars ||
      number_of_materials > nuser_scalars-relative_scalar) {
    MaterialInputError("scalar_index must select " +
                       std::to_string(number_of_materials) +
                       " user passive scalars in <" + block + ">");
  }
  constexpr Real monatomic_gamma = 5.0/3.0;
  if (!std::isfinite(gamma) ||
      std::abs(gamma-monatomic_gamma) > 32.0*std::numeric_limits<Real>::epsilon()) {
    MaterialInputError("the material-mixture closure currently requires <" + block +
                       ">/gamma=5/3");
  }

  const Real default_exchange_time = pin->GetOrAddReal(block, "t_ei", -1.0);
  if (!std::isfinite(default_exchange_time)) {
    MaterialInputError("the default <" + block + ">/t_ei must be finite");
  }
  std::vector<SpeciesProperties> host_species(number_of_materials);
  for (int n = 0; n < number_of_materials; ++n) {
    host_species[n] = ReadMaterial(pin, n, default_exchange_time);
  }
  data_.species = DvceArray1D<SpeciesProperties>(
      "material_species", number_of_materials);
  auto species_host = Kokkos::create_mirror_view(data_.species);
  for (int n = 0; n < number_of_materials; ++n) {
    species_host(n) = host_species[n];
  }
  Kokkos::deep_copy(data_.species, species_host);

  data_.scalar_index = first_user_scalar+relative_scalar;
  data_.scalar_indices = DvceArray1D<int>(
      "material_scalar_indices", number_of_materials);
  auto scalar_indices_host = Kokkos::create_mirror_view(data_.scalar_indices);
  for (int n = 0; n < number_of_materials; ++n) {
    scalar_indices_host(n) = first_user_scalar+relative_scalar+n;
  }
  Kokkos::deep_copy(data_.scalar_indices, scalar_indices_host);

  data_.gamma_minus_one = gamma-1.0;

  for (int n = 0; n < number_of_materials; ++n) {
    pin->GetOrAddString("materials", Key(n, "name"), "material"+std::to_string(n));
  }

  const bool has_table0 = pin->DoesParameterExist(
      "materials", "material0_eos_table_file");
  for (int n = 1; n < number_of_materials; ++n) {
    if (pin->DoesParameterExist("materials", Key(n, "eos_table_file")) !=
        has_table0) {
      MaterialInputError("every material*_eos_table_file must be supplied together");
    }
  }
  if (!has_table0) {
    if (unit_system != nullptr) {
      data_.density_to_cgs = unit_system->density_cgs();
      data_.temperature_to_kelvin = unit_system->temperature_cgs();
    }
    return;
  }
  if (unit_system == nullptr) {
    MaterialInputError(
        "tabular multi-material two-temperature EOS requires a <units> block");
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
  } else if (bounds == "flash-extrapolate") {
    options.bounds_policy = IonmixBoundsPolicy::flash_extrapolate;
  } else {
    MaterialInputError(
        "eos_table_bounds must be 'clamp', 'error', or 'flash-extrapolate'");
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

  tables_.resize(number_of_materials);
  std::vector<IonmixTwoTemperatureTableDevice> host_material_tables(
      number_of_materials);
  for (int n = 0; n < number_of_materials; ++n) {
    const std::string file = pin->GetString("materials", Key(n, "eos_table_file"));
    tables_[n] = std::make_unique<IonmixTwoTemperatureTable>(file, options);
    CheckAbar(n, host_species[n], tables_[n]->Metadata());
    const std::string fingerprint = pin->GetOrAddString(
        "materials", Key(n, "eos_table_fingerprint"),
        tables_[n]->Metadata().file_fingerprint);
    if (fingerprint != tables_[n]->Metadata().file_fingerprint) {
      MaterialInputError(
          Key(n, "eos_table_fingerprint")+" does not match the loaded file");
    }
    host_material_tables[n] = tables_[n]->DeviceData();
  }
  const std::size_t table_bytes =
      number_of_materials*sizeof(IonmixTwoTemperatureTableDevice);
  HostArray1D<unsigned char> host_table_storage(
      "material-eos-table-host-storage", table_bytes);
  std::memcpy(
      host_table_storage.data(), host_material_tables.data(), table_bytes);
  data_.material_table_storage = DvceArray1D<unsigned char>(
      "material-eos-table-storage", table_bytes);
  Kokkos::deep_copy(data_.material_table_storage, host_table_storage);
  data_.material_tables =
      reinterpret_cast<const IonmixTwoTemperatureTableDevice *>(
          data_.material_table_storage.data());
  data_.density_to_cgs = options.density_to_cgs;
  data_.temperature_to_kelvin = options.temperature_to_kelvin;
  data_.use_tabular_eos = true;
}

} // namespace materials
