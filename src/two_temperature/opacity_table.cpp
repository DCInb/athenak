//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file opacity_table.cpp
//! \brief Reader and device interpolation for multigroup opacity tables.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "two_temperature/opacity_table.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace two_temperature {
namespace {

//----------------------------------------------------------------------------------------
//! Tokenize a small human-readable table while removing comments introduced by '#'.

std::vector<std::string> ReadTokens(const std::string &filename) {
  std::ifstream stream(filename);
  if (!stream.is_open()) {
    throw std::runtime_error("could not open the file");
  }

  std::vector<std::string> tokens;
  std::string line;
  while (std::getline(stream, line)) {
    std::size_t comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream words(line);
    std::string token;
    while (words >> token) tokens.push_back(token);
  }
  if (stream.bad()) throw std::runtime_error("I/O error while reading the file");
  return tokens;
}

//----------------------------------------------------------------------------------------
//! Checked sequential access to the tokenized native opacity-table format.

class TableTokens {
 public:
  explicit TableTokens(const std::vector<std::string> &tokens) : tokens_(tokens) {}

  void Expect(const std::string &expected) {
    std::string actual = Next(expected);
    if (actual != expected) {
      throw std::runtime_error(
          "expected '" + expected + "' but found '" + actual + "'");
    }
  }

  int Integer(const std::string &description) {
    std::string token = Next(description);
    std::size_t used = 0;
    int result;
    try {
      result = std::stoi(token, &used);
    } catch (const std::exception &) {
      throw std::runtime_error(
          "invalid integer for " + description + ": '" + token + "'");
    }
    if (used != token.size()) {
      throw std::runtime_error(
          "invalid integer for " + description + ": '" + token + "'");
    }
    return result;
  }

  Real Number(const std::string &description) {
    std::string token = Next(description);
    std::replace(token.begin(), token.end(), 'D', 'e');
    std::replace(token.begin(), token.end(), 'd', 'e');
    std::size_t used = 0;
    Real result;
    try {
      result = std::stod(token, &used);
    } catch (const std::exception &) {
      throw std::runtime_error("invalid number for " + description + ": '" + token + "'");
    }
    if (used != token.size() || !std::isfinite(result)) {
      throw std::runtime_error("invalid number for " + description + ": '" + token + "'");
    }
    return result;
  }

  bool Done() const { return position_ == tokens_.size(); }

 private:
  std::string Next(const std::string &description) {
    if (position_ >= tokens_.size()) {
      throw std::runtime_error("unexpected end of file while reading " + description);
    }
    return tokens_[position_++];
  }

  const std::vector<std::string> &tokens_;
  std::size_t position_ = 0;
};

std::size_t TableValueIndex(int kind, int group, int density, int temperature,
                            int ngroups, int ndensity, int ntemperature) {
  return ((static_cast<std::size_t>(kind)*ngroups+group)*ndensity+density)*ntemperature+
         temperature;
}

[[noreturn]] void OpacityTableError(const std::string &filename,
                                    const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Opacity table '" << filename << "': " << message
            << std::endl;
  // Rank 0 can fail while other ranks are blocked in a broadcast. Abort the whole job
  // rather than leaving those ranks waiting in a collective.
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
//! Load a native table with FLASH opacity semantics.
//!
//! The stored order for each opacity kind is group, density, then temperature, with
//! temperature varying fastest.  Coordinates outside either axis are clamped to the
//! nearest boundary.  Conversion scales let physical tables be used with arbitrary code
//! units without changing the table itself.

OpacityTable::OpacityTable(ParameterInput *pin, int expected_groups,
    const DualArray1D<Real> &expected_group_bounds) :
    OpacityTable(pin, expected_groups, expected_group_bounds,
                 "thermal_radiation", "opacity") {}

OpacityTable::OpacityTable(ParameterInput *pin, int expected_groups,
    const DualArray1D<Real> &expected_group_bounds,
    const std::string &input_block, const std::string &parameter_prefix) :
    ndensity_(0),
    ntemperature_(0),
    ngroups_(expected_groups),
    log_interpolation_(false),
    geometric_interpolation_(false),
    log_coordinates_(false),
    density_scale_(1.0),
    temperature_scale_(1.0),
    transport_scale_(1.0),
    absorption_scale_(1.0),
    emission_scale_(1.0),
    density_(parameter_prefix + "-table-density", 1),
    temperature_(parameter_prefix + "-table-temperature", 1),
    values_(parameter_prefix + "-table-values", 1, 1, 1, 1) {
  const auto key = [&parameter_prefix](const std::string &suffix) {
    return parameter_prefix + "_" + suffix;
  };
  std::string filename = pin->GetString(input_block, key("table_file"));
  std::string interpolation = pin->GetOrAddString(
      input_block, key("interpolation"), "linear");
  if (interpolation == "linear") {
    log_interpolation_ = false;
  } else if (interpolation == "log") {
    log_interpolation_ = true;
  } else if (interpolation == "geometric" || interpolation == "hybrid") {
    geometric_interpolation_ = true;
  } else {
    OpacityTableError(filename, key("interpolation")+
                      " must be 'linear', 'log', or 'geometric'");
  }

  const std::string coordinate_interpolation = pin->GetOrAddString(
      input_block, key("coordinate_interpolation"), "linear");
  if (coordinate_interpolation == "linear") {
    log_coordinates_ = false;
  } else if (coordinate_interpolation == "log") {
    log_coordinates_ = true;
  } else {
    OpacityTableError(filename, key("coordinate_interpolation")+
                      " must be 'linear' or 'log'");
  }

  density_scale_ = pin->GetOrAddReal(
      input_block, key("density_scale"), 1.0);
  temperature_scale_ = pin->GetOrAddReal(
      input_block, key("temperature_scale"), 1.0);
  Real group_scale = pin->GetOrAddReal(
      input_block, key("group_bound_scale"), 1.0);
  Real opacity_scale = pin->GetOrAddReal(
      input_block, key("value_scale"), 1.0);
  transport_scale_ = pin->GetOrAddReal(
      input_block, key("transport_scale"), opacity_scale);
  absorption_scale_ = pin->GetOrAddReal(
      input_block, key("absorption_scale"), opacity_scale);
  emission_scale_ = pin->GetOrAddReal(
      input_block, key("emission_scale"), opacity_scale);
  if (!std::isfinite(density_scale_) || !std::isfinite(temperature_scale_) ||
      !std::isfinite(group_scale) || !std::isfinite(opacity_scale) ||
      !std::isfinite(transport_scale_) || !std::isfinite(absorption_scale_) ||
      !std::isfinite(emission_scale_)) {
    OpacityTableError(filename, "all opacity-table scales must be finite");
  }
  if (density_scale_ <= 0.0 || temperature_scale_ <= 0.0 || group_scale <= 0.0 ||
      opacity_scale <= 0.0 || transport_scale_ <= 0.0 || absorption_scale_ < 0.0 ||
      emission_scale_ < 0.0) {
    OpacityTableError(filename,
                      "coordinate, common opacity, and transport scales must be "
                      "positive, and source-opacity scales must be non-negative");
  }

  std::vector<Real> density_values;
  std::vector<Real> temperature_values;
  std::vector<Real> group_bound_values;
  std::vector<Real> opacity_values;
  bool read_file = true;
#if MPI_PARALLEL_ENABLED
  read_file = (global_variable::my_rank == 0);
#endif

  if (read_file) {
    try {
      std::vector<std::string> raw_tokens = ReadTokens(filename);
      TableTokens tokens(raw_tokens);
      tokens.Expect("athenak_opacity_table");
      int version = tokens.Integer("format version");
      if (version != 1) throw std::runtime_error("only format version 1 is supported");

      tokens.Expect("dimensions");
      ndensity_ = tokens.Integer("density dimension");
      ntemperature_ = tokens.Integer("temperature dimension");
      int table_groups = tokens.Integer("group dimension");
      if (ndensity_ < 1 || ntemperature_ < 1) {
        throw std::runtime_error("density and temperature dimensions must be positive");
      }
      if (table_groups != ngroups_) {
        throw std::runtime_error(
            "group dimension does not match <thermal_radiation>/n_groups");
      }
      std::size_t table_cells =
          static_cast<std::size_t>(ndensity_)*ntemperature_*ngroups_;
      if (ndensity_ > 100000 || ntemperature_ > 100000 || table_cells > 100000000U) {
        throw std::runtime_error("table dimensions are unreasonably large");
      }

      density_values.resize(ndensity_);
      temperature_values.resize(ntemperature_);
      group_bound_values.resize(ngroups_+1);
      opacity_values.resize(3*table_cells);

      tokens.Expect("density");
      for (int id = 0; id < ndensity_; ++id) {
        Real value = tokens.Number("density coordinate");
        if (value <= 0.0 || (id > 0 && value <= density_values[id-1])) {
          throw std::runtime_error("density coordinates must be positive and increasing");
        }
        density_values[id] = value;
      }

      tokens.Expect("temperature");
      for (int it = 0; it < ntemperature_; ++it) {
        Real value = tokens.Number("electron-temperature coordinate");
        if (value < 0.0 || (log_coordinates_ && value <= 0.0) ||
            (it > 0 && value <= temperature_values[it-1])) {
          throw std::runtime_error(
              log_coordinates_
                  ? "log-interpolated temperature coordinates must be positive and "
                    "increasing"
                  : "electron-temperature coordinates must be non-negative and "
                    "increasing");
        }
        temperature_values[it] = value;
      }

      tokens.Expect("group_bound");
      for (int g = 0; g <= ngroups_; ++g) {
        Real converted = group_scale*tokens.Number("radiation group boundary");
        if (!std::isfinite(converted)) {
          throw std::runtime_error("scaled radiation group boundaries must be finite");
        }
        group_bound_values[g] = converted;
      }

      const char *labels[3] = {"transport", "absorption", "emission"};
      const Real scales[3] = {transport_scale_, absorption_scale_, emission_scale_};
      for (int kind = 0; kind < 3; ++kind) {
        tokens.Expect(labels[kind]);
        for (int g = 0; g < ngroups_; ++g) {
          for (int id = 0; id < ndensity_; ++id) {
            for (int it = 0; it < ntemperature_; ++it) {
              Real value = tokens.Number(std::string(labels[kind]) + " opacity");
              if ((kind == opacity_transport && value <= 0.0) || value < 0.0) {
                throw std::runtime_error(
                    "transport opacity must be positive and source opacities "
                    "non-negative");
              }
              Real scaled_value = value*scales[kind];
              if (!std::isfinite(scaled_value) ||
                  (kind == opacity_transport && !(scaled_value > 0.0)) ||
                  (kind != opacity_transport && scaled_value < 0.0)) {
                throw std::runtime_error(
                    "scaled transport opacity must be finite and positive and "
                    "scaled source opacities finite and non-negative");
              }
              if (log_interpolation_ && value <= 0.0) {
                throw std::runtime_error(
                    "log interpolation requires every tabulated opacity to be positive");
              }
              std::size_t index = TableValueIndex(
                  kind, g, id, it, ngroups_, ndensity_, ntemperature_);
              opacity_values[index] = log_interpolation_ ? std::log(value) : value;
            }
          }
        }
      }
      tokens.Expect("end");
      if (!tokens.Done()) {
        throw std::runtime_error("unexpected data follows the end marker");
      }
    } catch (const std::exception &error) {
      OpacityTableError(filename, error.what());
    }
  }

#if MPI_PARALLEL_ENABLED
  int dimensions[3] = {ndensity_, ntemperature_, ngroups_};
  int ierr = MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) OpacityTableError(filename, "could not broadcast dimensions");
  ndensity_ = dimensions[0];
  ntemperature_ = dimensions[1];
  if (ndensity_ < 1 || ntemperature_ < 1 || dimensions[2] != ngroups_) {
    OpacityTableError(filename, "broadcast opacity-table dimensions are invalid");
  }
  std::size_t table_cells = static_cast<std::size_t>(ndensity_)*ntemperature_*ngroups_;
  std::size_t nvalues_size = 3*table_cells;
  if (ndensity_ > 100000 || ntemperature_ > 100000 || table_cells > 100000000U ||
      nvalues_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    OpacityTableError(filename, "broadcast opacity table is unreasonably large");
  }
  int nvalues = static_cast<int>(nvalues_size);
  if (!read_file) {
    density_values.resize(ndensity_);
    temperature_values.resize(ntemperature_);
    group_bound_values.resize(ngroups_+1);
    opacity_values.resize(nvalues);
  }
  ierr = MPI_Bcast(density_values.data(), ndensity_, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(temperature_values.data(), ntemperature_,
                     MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(group_bound_values.data(), ngroups_+1,
                     MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(opacity_values.data(), nvalues, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr != MPI_SUCCESS) {
    OpacityTableError(filename, "could not broadcast opacity-table values");
  }
#endif

  for (int g = 0; g <= ngroups_; ++g) {
    Real converted = group_bound_values[g];
    Real expected = expected_group_bounds.h_view(g);
    Real magnitude = std::max(static_cast<Real>(1.0),
                              std::max(std::abs(converted), std::abs(expected)));
    if (std::abs(converted-expected) > 1.0e-10*magnitude) {
      OpacityTableError(filename,
                        "radiation group boundaries do not match the input file");
    }
  }

  Kokkos::realloc(density_, ndensity_);
  Kokkos::realloc(temperature_, ntemperature_);
  Kokkos::realloc(values_, 3, ngroups_, ndensity_, ntemperature_);
  for (int id = 0; id < ndensity_; ++id) density_.h_view(id) = density_values[id];
  for (int it = 0; it < ntemperature_; ++it) {
    temperature_.h_view(it) = temperature_values[it];
  }
  for (int kind = 0; kind < 3; ++kind) {
    for (int g = 0; g < ngroups_; ++g) {
      for (int id = 0; id < ndensity_; ++id) {
        for (int it = 0; it < ntemperature_; ++it) {
          std::size_t index = TableValueIndex(
              kind, g, id, it, ngroups_, ndensity_, ntemperature_);
          values_.h_view(kind, g, id, it) = opacity_values[index];
        }
      }
    }
  }

  density_.modify_host();
  temperature_.modify_host();
  values_.modify_host();
  density_.sync_device();
  temperature_.sync_device();
  values_.sync_device();

  // Pre-log the two axes.  Both are constant for the lifetime of the table while the
  // lookup evaluates them on every call, so this removes four of the six axis logarithms
  // per material lookup.  The kernel runs on the device deliberately: filling these on
  // the host could differ from the device log() in the last ulp and would turn a pure
  // hoist into a roundoff-perturbing change.
  BuildLogAxes();

  if (global_variable::my_rank == 0) {
    const char *value_mode = log_interpolation_ ? "log-interpolated" :
        (geometric_interpolation_ ? "zero-safe geometrically interpolated" :
         "linearly interpolated");
    std::cout << "Loaded " << value_mode << " opacity table " << filename
              << " with " << (log_coordinates_ ? "log" : "linear")
              << " coordinates" << std::endl
              << "  density = [" << density_.h_view(0) << ", "
              << density_.h_view(ndensity_-1) << "]" << std::endl
              << "  electron temperature = [" << temperature_.h_view(0) << ", "
              << temperature_.h_view(ntemperature_-1) << "]" << std::endl
              << "  radiation groups = " << ngroups_ << std::endl;
  }
}

//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//! Fill the pre-logged axis arrays.  Kept out of the constructor because nvcc forbids
//! extended device lambdas there.

void OpacityTable::BuildLogAxes() {
  Kokkos::realloc(log_density_, ndensity_);
  Kokkos::realloc(log_temperature_, ntemperature_);
  Kokkos::realloc(log_values_, 3, ngroups_, ndensity_, ntemperature_);
  {
    auto table_values = values_.d_view;
    auto table_log_values = log_values_.d_view;
    const Real sentinel = kNonPositiveLog;
    par_for("opacity_log_values", DevExeSpace(), 0, 2, 0, ngroups_-1,
            0, ndensity_-1, 0, ntemperature_-1,
    KOKKOS_LAMBDA(const int kind, const int g, const int id, const int it) {
      const Real value = table_values(kind, g, id, it);
      table_log_values(kind, g, id, it) = (value > 0.0) ? log(value) : sentinel;
    });
  }
  auto d_axis = density_.d_view;
  auto t_axis = temperature_.d_view;
  auto log_d_axis = log_density_.d_view;
  auto log_t_axis = log_temperature_.d_view;
  par_for("opacity_log_density_axis", DevExeSpace(), 0, ndensity_-1,
  KOKKOS_LAMBDA(const int id) {
    log_d_axis(id) = log(d_axis(id));
  });
  par_for("opacity_log_temperature_axis", DevExeSpace(), 0, ntemperature_-1,
  KOKKOS_LAMBDA(const int it) {
    log_t_axis(it) = log(t_axis(it));
  });
}

//----------------------------------------------------------------------------------------

OpacityTableDevice OpacityTable::DeviceData() const {
  OpacityTableDevice result;
  result.density = density_.d_view;
  result.temperature = temperature_.d_view;
  result.log_density = log_density_.d_view;
  result.log_temperature = log_temperature_.d_view;
  result.values = values_.d_view;
  result.log_values = log_values_.d_view;
  result.ndensity = ndensity_;
  result.ntemperature = ntemperature_;
  result.log_interpolation = log_interpolation_;
  result.geometric_interpolation = geometric_interpolation_;
  result.log_coordinates = log_coordinates_;
  result.density_scale = density_scale_;
  result.temperature_scale = temperature_scale_;
  result.transport_scale = transport_scale_;
  result.absorption_scale = absorption_scale_;
  result.emission_scale = emission_scale_;
  return result;
}

//----------------------------------------------------------------------------------------

MixedOpacityTable::MixedOpacityTable(
    ParameterInput *pin, int expected_groups,
    const DualArray1D<Real> &expected_group_bounds) :
    material0_(pin, expected_groups, expected_group_bounds,
               "materials", "material0_opacity"),
    material1_(pin, expected_groups, expected_group_bounds,
               "materials", "material1_opacity") {}

MixedOpacityTableDevice MixedOpacityTable::DeviceData() const {
  MixedOpacityTableDevice result;
  result.material0 = material0_.DeviceData();
  result.material1 = material1_.DeviceData();
  return result;
}

} // namespace two_temperature
