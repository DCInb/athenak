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

[[noreturn]] void OpacityTableError(const std::string &filename,
                                    const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Opacity table '" << filename << "': " << message
            << std::endl;
  // A failure here can be rank-local (e.g. file unreadable on one node); abort the
  // whole job rather than leaving other ranks blocked in collectives.
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
    ndensity_(0),
    ntemperature_(0),
    ngroups_(expected_groups),
    log_interpolation_(false),
    density_scale_(1.0),
    temperature_scale_(1.0),
    transport_scale_(1.0),
    absorption_scale_(1.0),
    emission_scale_(1.0),
    density_("opacity-table-density", 1),
    temperature_("opacity-table-temperature", 1),
    values_("opacity-table-values", 1, 1, 1, 1) {
  std::string filename =
      pin->GetString("thermal_radiation", "opacity_table_file");
  std::string interpolation = pin->GetOrAddString(
      "thermal_radiation", "opacity_interpolation", "linear");
  if (interpolation == "linear") {
    log_interpolation_ = false;
  } else if (interpolation == "log") {
    log_interpolation_ = true;
  } else {
    OpacityTableError(filename, "opacity_interpolation must be 'linear' or 'log'");
  }

  density_scale_ = pin->GetOrAddReal(
      "thermal_radiation", "opacity_density_scale", 1.0);
  temperature_scale_ = pin->GetOrAddReal(
      "thermal_radiation", "opacity_temperature_scale", 1.0);
  Real group_scale = pin->GetOrAddReal(
      "thermal_radiation", "opacity_group_bound_scale", 1.0);
  Real opacity_scale = pin->GetOrAddReal(
      "thermal_radiation", "opacity_value_scale", 1.0);
  transport_scale_ = pin->GetOrAddReal(
      "thermal_radiation", "opacity_transport_scale", opacity_scale);
  absorption_scale_ = pin->GetOrAddReal(
      "thermal_radiation", "opacity_absorption_scale", opacity_scale);
  emission_scale_ = pin->GetOrAddReal(
      "thermal_radiation", "opacity_emission_scale", opacity_scale);
  if (density_scale_ <= 0.0 || temperature_scale_ <= 0.0 || group_scale <= 0.0 ||
      transport_scale_ <= 0.0 || absorption_scale_ < 0.0 || emission_scale_ < 0.0) {
    OpacityTableError(filename, "coordinate scales must be positive, the transport scale "
                      "must be positive, and source-opacity scales must be non-negative");
  }

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
    if (ndensity_ > 100000 || ntemperature_ > 100000 ||
        static_cast<std::size_t>(ndensity_)*ntemperature_*table_groups > 100000000U) {
      throw std::runtime_error("table dimensions are unreasonably large");
    }

    Kokkos::realloc(density_, ndensity_);
    Kokkos::realloc(temperature_, ntemperature_);
    Kokkos::realloc(values_, 3, ngroups_, ndensity_, ntemperature_);

    tokens.Expect("density");
    for (int id = 0; id < ndensity_; ++id) {
      Real value = tokens.Number("density coordinate");
      if (value <= 0.0 || (id > 0 && value <= density_.h_view(id-1))) {
        throw std::runtime_error("density coordinates must be positive and increasing");
      }
      density_.h_view(id) = value;
    }

    tokens.Expect("temperature");
    for (int it = 0; it < ntemperature_; ++it) {
      Real value = tokens.Number("electron-temperature coordinate");
      if (value < 0.0 || (it > 0 && value <= temperature_.h_view(it-1))) {
        throw std::runtime_error(
            "electron-temperature coordinates must be non-negative and increasing");
      }
      temperature_.h_view(it) = value;
    }

    tokens.Expect("group_bound");
    for (int g = 0; g <= ngroups_; ++g) {
      Real converted = group_scale*tokens.Number("radiation group boundary");
      Real expected = expected_group_bounds.h_view(g);
      Real magnitude =
          std::max(1.0, std::max(std::abs(converted), std::abs(expected)));
      if (std::abs(converted-expected) > 1.0e-10*magnitude) {
        throw std::runtime_error(
            "radiation group boundaries do not match the input file");
      }
    }

    const char *labels[3] = {"transport", "absorption", "emission"};
    for (int kind = 0; kind < 3; ++kind) {
      tokens.Expect(labels[kind]);
      for (int g = 0; g < ngroups_; ++g) {
        for (int id = 0; id < ndensity_; ++id) {
          for (int it = 0; it < ntemperature_; ++it) {
            Real value = tokens.Number(std::string(labels[kind]) + " opacity");
            if ((kind == opacity_transport && value <= 0.0) || value < 0.0) {
              throw std::runtime_error(
                  "transport opacity must be positive and source opacities non-negative");
            }
            if (log_interpolation_ && value <= 0.0) {
              throw std::runtime_error(
                  "log interpolation requires every tabulated opacity to be positive");
            }
            values_.h_view(kind, g, id, it) =
                log_interpolation_ ? std::log(value) : value;
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

  density_.modify_host();
  temperature_.modify_host();
  values_.modify_host();
  density_.sync_device();
  temperature_.sync_device();
  values_.sync_device();

  if (global_variable::my_rank == 0) {
    std::cout << "Loaded " << (log_interpolation_ ? "log-interpolated" : "linearly "
              "interpolated") << " opacity table " << filename << std::endl
              << "  density = [" << density_.h_view(0) << ", "
              << density_.h_view(ndensity_-1) << "]" << std::endl
              << "  electron temperature = [" << temperature_.h_view(0) << ", "
              << temperature_.h_view(ntemperature_-1) << "]" << std::endl
              << "  radiation groups = " << ngroups_ << std::endl;
  }
}

//----------------------------------------------------------------------------------------

OpacityTableDevice OpacityTable::DeviceData() const {
  OpacityTableDevice result;
  result.density = density_.d_view;
  result.temperature = temperature_.d_view;
  result.values = values_.d_view;
  result.ndensity = ndensity_;
  result.ntemperature = ntemperature_;
  result.log_interpolation = log_interpolation_;
  result.density_scale = density_scale_;
  result.temperature_scale = temperature_scale_;
  result.transport_scale = transport_scale_;
  result.absorption_scale = absorption_scale_;
  result.emission_scale = emission_scale_;
  return result;
}

} // namespace two_temperature
