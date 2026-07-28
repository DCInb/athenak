//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file material_mixture.cpp
//! \brief Input parsing for the two-material ideal-plasma closure.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "materials/material_mixture.hpp"
#include "parameter_input.hpp"

namespace materials {
namespace {

[[noreturn]] void MaterialInputError(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "<materials> " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

std::string Key(const int material, const std::string &property) {
  return "material" + std::to_string(material) + "_" + property;
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
    MaterialInputError(Key(material, "abar") + " must be finite and positive");
  }
  if (!std::isfinite(result.zbar) || !(result.zbar > 0.0)) {
    MaterialInputError(Key(material, "zbar") + " must be finite and positive");
  }
  if (!std::isfinite(result.zeff) || !(result.zeff > 0.0)) {
    MaterialInputError(Key(material, "zeff") + " must be finite and positive");
  }
  if (!std::isfinite(result.t_ei)) {
    MaterialInputError(Key(material, "t_ei") + " must be finite");
  }
  return result;
}

} // namespace

MaterialMixture::MaterialMixture(ParameterInput *pin, const int first_user_scalar,
                                 const int nuser_scalars, const Real gamma) {
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
    MaterialInputError("fully ionized monatomic materials require <mhd>/gamma=5/3");
  }

  const Real default_exchange_time = pin->GetOrAddReal("mhd", "t_ei", -1.0);
  if (!std::isfinite(default_exchange_time)) {
    MaterialInputError("the default <mhd>/t_ei must be finite");
  }
  data_.material0 = ReadMaterial(pin, 0, default_exchange_time);
  data_.material1 = ReadMaterial(pin, 1, default_exchange_time);
  data_.scalar_index = first_user_scalar + relative_scalar;

  // Retain names as echoed input metadata without putting strings in device data.
  pin->GetOrAddString("materials", "material0_name", "material0");
  pin->GetOrAddString("materials", "material1_name", "material1");
}

} // namespace materials
