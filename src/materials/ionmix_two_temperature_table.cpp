//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ionmix_two_temperature_table.cpp
//! \brief Rank-safe native separate ion/electron IONMIX EOS table reader.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace materials {
namespace {

struct TableFile {
  std::string bytes;
  std::vector<std::string> tokens;
};

TableFile ReadTableFile(const std::string &filename) {
  std::ifstream stream(filename, std::ios::binary);
  if (!stream.is_open()) throw std::runtime_error("could not open the file");

  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad()) throw std::runtime_error("I/O error while reading the file");

  TableFile result;
  result.bytes = contents.str();
  std::istringstream lines(result.bytes);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream words(line);
    std::string token;
    while (words >> token) result.tokens.push_back(token);
  }
  return result;
}

class TableTokens {
 public:
  explicit TableTokens(const std::vector<std::string> &tokens) : tokens_(tokens) {}

  void Expect(const std::string &expected) {
    const std::string actual = Next(expected);
    if (actual != expected) {
      throw std::runtime_error(
          "expected '" + expected + "' but found '" + actual + "'");
    }
  }

  int Integer(const std::string &description) {
    const std::string token = Next(description);
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
      result = static_cast<Real>(std::stod(token, &used));
    } catch (const std::exception &) {
      throw std::runtime_error(
          "invalid number for " + description + ": '" + token + "'");
    }
    if (used != token.size() || !std::isfinite(result)) {
      throw std::runtime_error(
          "invalid number for " + description + ": '" + token + "'");
    }
    return result;
  }

  bool Done() const { return position_ == tokens_.size(); }

 private:
  std::string Next(const std::string &description) {
    if (position_ >= tokens_.size()) {
      throw std::runtime_error(
          "unexpected end of file while reading " + description);
    }
    return tokens_[position_++];
  }

  const std::vector<std::string> &tokens_;
  std::size_t position_ = 0;
};

std::size_t TableValueIndex(const int field, const int density,
                            const int temperature, const int ndensity,
                            const int ntemperature) {
  return (static_cast<std::size_t>(field)*ndensity+density)*ntemperature+
         temperature;
}

std::uint64_t Fingerprint(const std::string &bytes) {
  // FNV-1a is deterministic across ranks and records exact file identity, including
  // comments.  It is an integrity fingerprint, not a cryptographic authenticity check.
  std::uint64_t result = UINT64_C(14695981039346656037);
  for (const unsigned char byte : bytes) {
    result ^= static_cast<std::uint64_t>(byte);
    result *= UINT64_C(1099511628211);
  }
  return result;
}

std::string FingerprintString(const std::uint64_t fingerprint) {
  std::ostringstream result;
  result << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
         << fingerprint;
  return result.str();
}

#if MPI_PARALLEL_ENABLED
bool MpiInitialized() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

int CurrentRank() {
  if (!MpiInitialized()) return 0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}
#else
int CurrentRank() { return 0; }
#endif

[[noreturn]] void IonmixTableError(const std::string &filename,
                                   const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << "Two-temperature IONMIX table '" << filename << "': "
            << message << std::endl;
#if MPI_PARALLEL_ENABLED
  if (MpiInitialized()) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) MPI_Abort(MPI_COMM_WORLD, 1);
  }
#endif
  std::exit(EXIT_FAILURE);
}

void ValidateOptions(const std::string &filename,
                     const IonmixTwoTemperatureTableOptions &options) {
  if (options.bounds_policy != IonmixBoundsPolicy::clamp &&
      options.bounds_policy != IonmixBoundsPolicy::error) {
    IonmixTableError(filename, "bounds policy must be clamp or error");
  }
  const Real scales[] = {
      options.density_to_cgs, options.temperature_to_kelvin,
      options.pressure_from_cgs, options.specific_energy_from_cgs};
  for (const Real scale : scales) {
    if (!std::isfinite(scale) || !(scale > 0.0)) {
      IonmixTableError(filename, "all code-unit conversion scales must be finite and "
                                 "positive");
    }
  }
}

int MinimumTemperatureRoundTripsExactlyOnDevice(
    const DvceArray1D<Real> &log_temperature_kelvin,
    const Real temperature_to_kelvin) {
  const DvceArray1D<Real> temperature_axis = log_temperature_kelvin;
  int exact_count = 0;
  Kokkos::parallel_reduce(
      "ionmix-minimum-temperature-round-trip", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int, int &local_exact_count) {
        const Real code_temperature =
            exp(temperature_axis(0))/temperature_to_kelvin;
        if (!Kokkos::isfinite(code_temperature) || !(code_temperature > 0.0)) {
          return;
        }
        const Real reconstructed =
            log(code_temperature)+log(temperature_to_kelvin);
        if (Kokkos::isfinite(reconstructed) &&
            reconstructed == temperature_axis(0)) {
          ++local_exact_count;
        }
      }, Kokkos::Sum<int>(exact_count));
  return exact_count;
}

struct CachedDeviceConstants {
  Real log_density_to_cgs = 0.0;
  Real log_temperature_to_kelvin = 0.0;
  Real minimum_density_code = 0.0;
  Real maximum_density_code = 0.0;
  Real minimum_temperature_code = 0.0;
  Real maximum_temperature_code = 0.0;
};

CachedDeviceConstants CacheDeviceConstants(
    const DvceArray1D<Real> &log_density, const int ndensity,
    const DvceArray1D<Real> &log_temperature, const int ntemperature,
    const Real density_to_cgs, const Real temperature_to_kelvin) {
  DualArray1D<Real> device_constants("ionmix-2t-device-constants", 6);
  const DvceArray1D<Real> constants = device_constants.d_view;
  Kokkos::parallel_for(
      "ionmix-cache-device-constants", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        constants(0) = log(density_to_cgs);
        constants(1) = log(temperature_to_kelvin);
        constants(2) = exp(log_density(0))/density_to_cgs;
        constants(3) = exp(log_density(ndensity-1))/density_to_cgs;
        constants(4) = exp(log_temperature(0))/temperature_to_kelvin;
        constants(5) = exp(log_temperature(ntemperature-1))/temperature_to_kelvin;
      });
  device_constants.modify_device();
  device_constants.sync_host();

  CachedDeviceConstants result;
  result.log_density_to_cgs = device_constants.h_view(0);
  result.log_temperature_to_kelvin = device_constants.h_view(1);
  result.minimum_density_code = device_constants.h_view(2);
  result.maximum_density_code = device_constants.h_view(3);
  result.minimum_temperature_code = device_constants.h_view(4);
  result.maximum_temperature_code = device_constants.h_view(5);
  return result;
}

void CacheValueLogs(const DvceArray3D<Real> &values,
                    const DvceArray3D<Real> &log_values,
                    const int nfields, const int ndensity,
                    const int ntemperature) {
  const int count = nfields*ndensity*ntemperature;
  Kokkos::parallel_for(
      "ionmix-cache-value-logs", Kokkos::RangePolicy<>(0, count),
      KOKKOS_LAMBDA(const int index) {
        const int temperature = index%ntemperature;
        const int field_density = index/ntemperature;
        const int density = field_density%ndensity;
        const int field = field_density/ndensity;
        const Real value = values(field, density, temperature);
        log_values(field, density, temperature) =
            (value > 0.0) ? log(value) : 0.0;
      });
}

} // namespace

//----------------------------------------------------------------------------------------

IonmixTwoTemperatureTable::IonmixTwoTemperatureTable(
    const std::string &filename,
    const IonmixTwoTemperatureTableOptions &options) :
    options_(options),
    log_density_cgs_("ionmix-2t-log-density", 1),
    log_temperature_kelvin_("ionmix-2t-log-temperature", 1),
    values_("ionmix-2t-values", 1, 1, 1),
    log_values_("ionmix-2t-log-values", 1, 1, 1) {
  ValidateOptions(filename, options_);

  int format_version = 0;
  int ndensity = 0;
  int ntemperature = 0;
  Real abar = 0.0;
  bool ion_energy_positive = false;
  bool electron_energy_positive = false;
  bool pressure_interpolation_safely_finite = false;
  std::uint64_t fingerprint = 0;
  std::uint64_t file_size = 0;
  std::vector<Real> density;
  std::vector<Real> temperature;
  std::vector<Real> values;

  const int rank = CurrentRank();
  const bool read_file = (rank == 0);
  if (read_file) {
    try {
      const TableFile table_file = ReadTableFile(filename);
      fingerprint = Fingerprint(table_file.bytes);
      file_size = static_cast<std::uint64_t>(table_file.bytes.size());
      TableTokens tokens(table_file.tokens);
      tokens.Expect("athenak_two_temperature_eos");
      format_version = tokens.Integer("format version");
      if (format_version != 1) {
        throw std::runtime_error("only format version 1 is supported");
      }

      tokens.Expect("dimensions");
      ndensity = tokens.Integer("density dimension");
      ntemperature = tokens.Integer("temperature dimension");
      if (ndensity < 2 || ntemperature < 2) {
        throw std::runtime_error(
            "density and temperature dimensions must each contain at least two points");
      }
      const std::size_t cells =
          static_cast<std::size_t>(ndensity)*ntemperature;
      if (ndensity > 100000 || ntemperature > 100000 ||
          cells > 100000000U ||
          cells > std::numeric_limits<std::size_t>::max()/
                      IonmixTwoTemperatureTableDevice::nfields) {
        throw std::runtime_error("table dimensions are unreasonably large");
      }

      tokens.Expect("abar");
      abar = tokens.Number("mean atomic mass");
      if (!(abar > 0.0)) {
        throw std::runtime_error("abar must be finite and positive");
      }

      density.resize(ndensity);
      temperature.resize(ntemperature);
      values.resize(IonmixTwoTemperatureTableDevice::nfields*cells);

      tokens.Expect("density");
      for (int id = 0; id < ndensity; ++id) {
        density[id] = tokens.Number("density coordinate");
        if (!(density[id] > 0.0) ||
            (id > 0 && !(density[id] > density[id-1]))) {
          throw std::runtime_error(
              "density coordinates must be positive and strictly increasing");
        }
      }

      tokens.Expect("temperature");
      for (int it = 0; it < ntemperature; ++it) {
        temperature[it] = tokens.Number("temperature coordinate");
        if (!(temperature[it] > 0.0) ||
            (it > 0 && !(temperature[it] > temperature[it-1]))) {
          throw std::runtime_error(
              "temperature coordinates must be positive and strictly increasing");
        }
      }

      const char *labels[IonmixTwoTemperatureTableDevice::nfields] = {
          "ion_pressure", "electron_pressure",
          "ion_specific_internal_energy", "electron_specific_internal_energy",
          "mean_ionization"};
      pressure_interpolation_safely_finite = true;
      // Keep headroom in both the native and code-unit domains for two nested
      // non-negative interpolation stages followed by a four-term mixed pressure sum.
      const Real safe_pressure_limit =
          std::numeric_limits<Real>::max()/64.0;
      for (int field = 0; field < IonmixTwoTemperatureTableDevice::nfields; ++field) {
        tokens.Expect(labels[field]);
        for (int id = 0; id < ndensity; ++id) {
          for (int it = 0; it < ntemperature; ++it) {
            const Real value = tokens.Number(labels[field]);
            if ((field == IonmixTwoTemperatureTableDevice::ion_pressure ||
                 field == IonmixTwoTemperatureTableDevice::electron_pressure ||
                 field == IonmixTwoTemperatureTableDevice::mean_ionization) &&
                value < 0.0) {
              throw std::runtime_error(
                  std::string(labels[field]) + " values must be non-negative");
            }
            const Real output_scale =
                (field == IonmixTwoTemperatureTableDevice::ion_pressure ||
                 field == IonmixTwoTemperatureTableDevice::electron_pressure)
                    ? options_.pressure_from_cgs
                    : ((field == IonmixTwoTemperatureTableDevice::
                                      ion_specific_internal_energy ||
                        field == IonmixTwoTemperatureTableDevice::
                                      electron_specific_internal_energy)
                           ? options_.specific_energy_from_cgs
                           : 1.0);
            const Real scaled_value = value*output_scale;
            if (!std::isfinite(scaled_value)) {
              throw std::runtime_error(
                  std::string(labels[field]) + " overflows its code-unit scale");
            }
            if ((field == IonmixTwoTemperatureTableDevice::ion_pressure ||
                 field == IonmixTwoTemperatureTableDevice::electron_pressure) &&
                (!(value < safe_pressure_limit) ||
                 !(scaled_value < safe_pressure_limit))) {
              pressure_interpolation_safely_finite = false;
            }
            values[TableValueIndex(
                field, id, it, ndensity, ntemperature)] = value;
          }
        }
      }
      tokens.Expect("end");
      if (!tokens.Done()) {
        throw std::runtime_error("unexpected data follows the end marker");
      }

      ion_energy_positive = true;
      electron_energy_positive = true;
      const int energy_fields[2] = {
          IonmixTwoTemperatureTableDevice::ion_specific_internal_energy,
          IonmixTwoTemperatureTableDevice::electron_specific_internal_energy};
      bool *positive_flags[2] = {
          &ion_energy_positive, &electron_energy_positive};
      for (int component = 0; component < 2; ++component) {
        const int field = energy_fields[component];
        for (int id = 0; id < ndensity; ++id) {
          for (int it = 0; it < ntemperature; ++it) {
            const Real value = values[TableValueIndex(
                field, id, it, ndensity, ntemperature)];
            if (!(value > 0.0)) *positive_flags[component] = false;
            if (it > 0) {
              const Real previous = values[TableValueIndex(
                  field, id, it-1, ndensity, ntemperature)];
              if (value < previous) {
                throw std::runtime_error(
                    std::string(labels[field])+
                    " must be nondecreasing with temperature at every density");
              }
            }
          }
        }
      }
    } catch (const std::exception &error) {
      IonmixTableError(filename, error.what());
    }
  }

#if MPI_PARALLEL_ENABLED
  int header[6] = {
      format_version, ndensity, ntemperature,
      ion_energy_positive ? 1 : 0,
      electron_energy_positive ? 1 : 0,
      pressure_interpolation_safely_finite ? 1 : 0};
  int ierr = MPI_Bcast(header, 6, MPI_INT, 0, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) {
    IonmixTableError(filename, "could not broadcast table metadata");
  }
  format_version = header[0];
  ndensity = header[1];
  ntemperature = header[2];
  ion_energy_positive = header[3] != 0;
  electron_energy_positive = header[4] != 0;
  pressure_interpolation_safely_finite = header[5] != 0;
  if (format_version != 1 || ndensity < 2 || ntemperature < 2) {
    IonmixTableError(filename, "broadcast table metadata is invalid");
  }
  const std::size_t cells = static_cast<std::size_t>(ndensity)*ntemperature;
  const std::size_t nvalues_size =
      IonmixTwoTemperatureTableDevice::nfields*cells;
  if (ndensity > 100000 || ntemperature > 100000 || cells > 100000000U ||
      nvalues_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    IonmixTableError(filename, "broadcast table dimensions are unreasonably large");
  }
  if (!read_file) {
    density.resize(ndensity);
    temperature.resize(ntemperature);
    values.resize(nvalues_size);
  }
  unsigned long long fingerprint_wire =
      static_cast<unsigned long long>(fingerprint);
  unsigned long long file_size_wire = static_cast<unsigned long long>(file_size);
  ierr = MPI_Bcast(&abar, 1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(
        &fingerprint_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(
        &file_size_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(
        density.data(), ndensity, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(
        temperature.data(), ntemperature, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(
        values.data(), static_cast<int>(nvalues_size),
        MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr != MPI_SUCCESS) {
    IonmixTableError(filename, "could not broadcast table values");
  }
  fingerprint = static_cast<std::uint64_t>(fingerprint_wire);
  file_size = static_cast<std::uint64_t>(file_size_wire);
#endif

  if (!std::isfinite(abar) || !(abar > 0.0)) {
    IonmixTableError(filename, "broadcast abar is invalid");
  }

  Kokkos::realloc(log_density_cgs_, ndensity);
  Kokkos::realloc(log_temperature_kelvin_, ntemperature);
  Kokkos::realloc(
      values_, IonmixTwoTemperatureTableDevice::nfields, ndensity, ntemperature);
  Kokkos::realloc(
      log_values_, IonmixTwoTemperatureTableDevice::nfields, ndensity, ntemperature);
  for (int id = 0; id < ndensity; ++id) {
    log_density_cgs_.h_view(id) = std::log(density[id]);
  }
  for (int it = 0; it < ntemperature; ++it) {
    log_temperature_kelvin_.h_view(it) = std::log(temperature[it]);
  }
  for (int field = 0; field < IonmixTwoTemperatureTableDevice::nfields; ++field) {
    for (int id = 0; id < ndensity; ++id) {
      for (int it = 0; it < ntemperature; ++it) {
        values_.h_view(field, id, it) = values[TableValueIndex(
            field, id, it, ndensity, ntemperature)];
      }
    }
  }
  log_density_cgs_.modify_host();
  log_temperature_kelvin_.modify_host();
  values_.modify_host();
  log_density_cgs_.sync_device();
  log_temperature_kelvin_.sync_device();
  values_.sync_device();
  CacheValueLogs(
      values_.d_view, log_values_.d_view,
      IonmixTwoTemperatureTableDevice::nfields, ndensity, ntemperature);
  log_values_.modify_device();

  // These expressions used to run in every table query.  Evaluate them once on the
  // active device backend so cached values retain the exact CUDA log/exp/div rounding
  // of the original hot path rather than introducing host-libm values.
  const CachedDeviceConstants device_constants = CacheDeviceConstants(
      log_density_cgs_.d_view, ndensity, log_temperature_kelvin_.d_view,
      ntemperature, options_.density_to_cgs, options_.temperature_to_kelvin);
  log_density_to_cgs_ = device_constants.log_density_to_cgs;
  log_temperature_to_kelvin_ = device_constants.log_temperature_to_kelvin;
  minimum_density_code_ = device_constants.minimum_density_code;
  maximum_density_code_ = device_constants.maximum_density_code;
  minimum_temperature_code_ = device_constants.minimum_temperature_code;
  maximum_temperature_code_ = device_constants.maximum_temperature_code;

  minimum_temperature_round_trips_exactly_ =
      MinimumTemperatureRoundTripsExactlyOnDevice(
          log_temperature_kelvin_.d_view, options_.temperature_to_kelvin);

  metadata_.source_file = filename;
  metadata_.file_fingerprint_value = fingerprint;
  metadata_.file_fingerprint = FingerprintString(fingerprint);
  metadata_.file_size = file_size;
  metadata_.format_version = format_version;
  metadata_.ndensity = ndensity;
  metadata_.ntemperature = ntemperature;
  metadata_.abar = abar;
  metadata_.minimum_density_cgs = density.front();
  metadata_.maximum_density_cgs = density.back();
  metadata_.minimum_temperature_kelvin = temperature.front();
  metadata_.maximum_temperature_kelvin = temperature.back();
  metadata_.ion_energy_is_strictly_positive = ion_energy_positive;
  metadata_.electron_energy_is_strictly_positive = electron_energy_positive;
  metadata_.pressure_interpolation_is_safely_finite =
      pressure_interpolation_safely_finite;

  if (rank == 0) {
    std::cout << "Loaded separate ion/electron IONMIX table " << filename
              << " (" << metadata_.file_fingerprint << ")" << std::endl
              << "  dimensions = " << ndensity << " x " << ntemperature
              << ", abar = " << abar << std::endl
              << "  density [g/cm^3] = [" << density.front() << ", "
              << density.back() << "]" << std::endl
              << "  temperature [K] = [" << temperature.front() << ", "
              << temperature.back() << "]" << std::endl;
  }
}

//----------------------------------------------------------------------------------------

IonmixTwoTemperatureTableDevice IonmixTwoTemperatureTable::DeviceData() const {
  IonmixTwoTemperatureTableDevice result;
  result.log_density_cgs = log_density_cgs_.d_view;
  result.log_temperature_kelvin = log_temperature_kelvin_.d_view;
  result.values = values_.d_view;
  result.log_values = log_values_.d_view;
  result.ndensity = metadata_.ndensity;
  result.ntemperature = metadata_.ntemperature;
  result.bounds_error =
      (options_.bounds_policy == IonmixBoundsPolicy::error) ? 1 : 0;
  result.geometric_interpolation = options_.geometric_interpolation;
  result.ion_energy_is_strictly_positive =
      metadata_.ion_energy_is_strictly_positive;
  result.electron_energy_is_strictly_positive =
      metadata_.electron_energy_is_strictly_positive;
  result.pressure_interpolation_is_safely_finite =
      metadata_.pressure_interpolation_is_safely_finite;
  result.minimum_temperature_round_trips_exactly =
      minimum_temperature_round_trips_exactly_;
  result.abar = metadata_.abar;
  result.density_to_cgs = options_.density_to_cgs;
  result.temperature_to_kelvin = options_.temperature_to_kelvin;
  result.pressure_from_cgs = options_.pressure_from_cgs;
  result.specific_energy_from_cgs = options_.specific_energy_from_cgs;
  result.log_density_to_cgs = log_density_to_cgs_;
  result.log_temperature_to_kelvin = log_temperature_to_kelvin_;
  result.minimum_density_code = minimum_density_code_;
  result.maximum_density_code = maximum_density_code_;
  result.minimum_temperature_code = minimum_temperature_code_;
  result.maximum_temperature_code = maximum_temperature_code_;
  return result;
}

//----------------------------------------------------------------------------------------

bool IonmixTwoTemperatureTable::SharesTemperatureGrid(
    const IonmixTwoTemperatureTable &other) const {
  if (metadata_.ntemperature != other.metadata_.ntemperature) return false;
  for (int it = 0; it < metadata_.ntemperature; ++it) {
    const Real lhs = log_temperature_kelvin_.h_view(it);
    const Real rhs = other.log_temperature_kelvin_.h_view(it);
    const Real scale = std::max(static_cast<Real>(1.0),
                                std::max(std::abs(lhs), std::abs(rhs)));
    if (std::abs(lhs-rhs) > 32.0*std::numeric_limits<Real>::epsilon()*scale) {
      return false;
    }
  }
  return true;
}

} // namespace materials
