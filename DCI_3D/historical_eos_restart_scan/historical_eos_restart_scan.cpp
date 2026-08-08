//========================================================================================
//! \file historical_eos_restart_scan.cpp
//! \brief Read-only EOS validation of the legacy t=1 ns DCI restart.
//!
//! The binary layout is derived from commit 843f2a86.  The old Hydro constructor stored
//! 5 hydro variables + 1 user scalar + 2 material energies + 20 FLD groups in u0, and
//! RestartOutput wrote each LayoutRight (variable,k,j,i) MeshBlock including ghosts.
//! This program deliberately does not use AthenaK's restart reader: the current deck has
//! three explicit material fractions and cannot safely reinterpret the old 28-array file.
//========================================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openssl/evp.h>

#include "athena.hpp"
#include "materials/ionmix_two_temperature_table.hpp"
#include "materials/material_mixture.hpp"

#ifndef ATHENAK_SOURCE_REVISION
#define ATHENAK_SOURCE_REVISION "unknown"
#endif
#ifndef MATERIAL_MIXTURE_SHA256
#define MATERIAL_MIXTURE_SHA256 "unknown"
#endif
#ifndef IONMIX_HEADER_SHA256
#define IONMIX_HEADER_SHA256 "unknown"
#endif
#ifndef IONMIX_SOURCE_SHA256
#define IONMIX_SOURCE_SHA256 "unknown"
#endif
#ifndef SCANNER_SOURCE_SHA256
#define SCANNER_SOURCE_SHA256 "unknown"
#endif

namespace {

constexpr const char *kLayoutRevision = "843f2a86";
constexpr const char *kExpectedRestartSha256 =
    "2a72f97fdd1c3608c57f8cd0b642755052135cff68b4d24c9db6fde2805fdb40";
constexpr const char *kExpectedChTableSha256 =
    "b29624877c7c90ed1d8c385bef6a7882b106dd8202bf0398301e2dee09faa0d8";
constexpr const char *kExpectedHeTableSha256 =
    "aae12f2dde296992ad630094e5755f7f52baa0816c678771e075f4848a9d63d0";
constexpr Real kExpectedRestartTimeNs = 1.0;
constexpr int kExpectedRestartCycle = 48420;
constexpr std::uint64_t kLegacyHeaderBytes =
    3*sizeof(std::int32_t) + 2*sizeof(double) + 9*sizeof(double) +
    2*19*sizeof(std::int32_t);
constexpr int kHydroVariables = 5;
constexpr int kUserScalars = 1;
constexpr int kComponentEnergies = 2;
constexpr int kRadiationGroups = 20;
constexpr int kLegacyVariables =
    kHydroVariables+kUserScalars+kComponentEnergies+kRadiationGroups;
constexpr int kDensityVariable = 0;
constexpr int kMaterial0Variable = 5;
constexpr int kIonEnergyVariable = 6;
constexpr int kElectronEnergyVariable = 7;
static_assert(kIonEnergyVariable == kMaterial0Variable+1 &&
              kElectronEnergyVariable == kMaterial0Variable+2,
              "legacy material fields must be contiguous");
constexpr Real kCellRelativeEnergyTolerance = 1.0e-10;
constexpr Real kGlobalRelativeEnergyTolerance = 1.0e-12;

struct LogicalLocation {
  std::int32_t lx1 = 0;
  std::int32_t lx2 = 0;
  std::int32_t lx3 = 0;
  std::int32_t level = 0;
};
static_assert(sizeof(LogicalLocation) == 16, "legacy logical-location ABI changed");

struct LegacyHeader {
  std::int32_t nmb = 0;
  std::int32_t root_level = 0;
  std::int32_t mesh_indices[19]{};
  std::int32_t block_indices[19]{};
  double time = 0.0;
  double dt = 0.0;
  std::int32_t cycle = 0;
};

struct CellResult {
  // bit 0: invalid scanned EOS input; bit 1: invalid FLASH state; bit 2: raw rho*Y
  // outside [0,rho]; bit 3: invalid historical endpoint state.
  int status = 0;
  int old_ion_flags = 0;
  int old_electron_flags = 0;
  int flash_flags = 0;
  Real ion_temperature = 0.0;
  Real electron_temperature = 0.0;
  Real ion_pressure = 0.0;
  Real electron_pressure = 0.0;
  Real recovered_ion_energy = 0.0;
  Real recovered_electron_energy = 0.0;
  Real endpoint_ratio_ion = 0.0;
  Real endpoint_ratio_electron = 0.0;
};

struct ExtremumLocation {
  double value = -std::numeric_limits<double>::infinity();
  std::uint64_t gid = 0;
  int local_i = 0;
  int local_j = 0;
  int local_k = 0;
  std::int64_t global_i = 0;
  std::int64_t global_j = 0;
  std::int64_t global_k = 0;
};

struct Metrics {
  std::uint64_t total_cells = 0;
  std::uint64_t valid_scanned_eos_input_cells = 0;
  std::uint64_t invalid_scanned_eos_input_cells = 0;
  std::uint64_t raw_fraction_outside_cells = 0;
  std::uint64_t old_high_cells = 0;
  std::uint64_t old_ion_high_cells = 0;
  std::uint64_t old_electron_high_cells = 0;
  std::uint64_t old_both_high_cells = 0;
  std::uint64_t old_below_cells = 0;
  std::uint64_t flash_high_cells = 0;
  std::uint64_t flash_below_cells = 0;
  std::uint64_t flash_density_below_cells = 0;
  std::uint64_t flash_density_above_cells = 0;
  std::uint64_t flash_temperature_below_cells = 0;
  std::uint64_t flash_temperature_above_cells = 0;
  std::uint64_t flash_invalid_state_cells = 0;
  std::uint64_t endpoint_invalid_cells = 0;
  long double stored_ion_energy_density_sum = 0.0L;
  long double stored_electron_energy_density_sum = 0.0L;
  long double recovered_ion_energy_density_sum = 0.0L;
  long double recovered_electron_energy_density_sum = 0.0L;
  double maximum_ion_relative_residual = 0.0;
  double maximum_electron_relative_residual = 0.0;
  double maximum_ion_absolute_specific_residual = 0.0;
  double maximum_electron_absolute_specific_residual = 0.0;
  double minimum_ion_temperature = std::numeric_limits<double>::infinity();
  double maximum_ion_temperature = 0.0;
  double minimum_electron_temperature = std::numeric_limits<double>::infinity();
  double maximum_electron_temperature = 0.0;
  double minimum_ion_pressure = std::numeric_limits<double>::infinity();
  double maximum_ion_pressure = 0.0;
  double minimum_electron_pressure = std::numeric_limits<double>::infinity();
  double maximum_electron_pressure = 0.0;
  ExtremumLocation endpoint_ratio;
  ExtremumLocation endpoint_ratio_ion;
  ExtremumLocation endpoint_ratio_electron;
  ExtremumLocation ion_relative_residual;
  ExtremumLocation electron_relative_residual;
};

std::string Trim(const std::string &input) {
  const std::size_t first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last-first+1);
}

std::string JsonEscape(const std::string &input) {
  std::ostringstream out;
  for (const unsigned char value : input) {
    switch (value) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (value < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(value) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(value);
        }
    }
  }
  return out.str();
}

std::string Sha256File(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open for SHA-256: "+path.string());
  }
  using DigestContext = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("cannot initialize SHA-256 for "+path.string());
  }
  std::array<unsigned char, 1024*1024> buffer{};
  while (input) {
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("SHA-256 update failed for "+path.string());
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("SHA-256 read failed for "+path.string());
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32) {
    throw std::runtime_error("SHA-256 finalization failed for "+path.string());
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    result << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return result.str();
}

void RequireSha256(const std::filesystem::path &path,
                   const std::string &actual, const char *expected) {
  if (actual != expected) {
    throw std::runtime_error(
        path.string()+" SHA-256 mismatch: expected "+expected+", found "+actual);
  }
}

void ReadAt(std::ifstream &stream, const std::uint64_t offset,
            void *destination, const std::uint64_t bytes) {
  stream.clear();
  stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!stream) throw std::runtime_error("restart seek failed at byte " +
                                        std::to_string(offset));
  stream.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes));
  if (!stream || static_cast<std::uint64_t>(stream.gcount()) != bytes) {
    throw std::runtime_error("short restart read at byte " + std::to_string(offset));
  }
}

std::string ReadParameterDump(std::ifstream &stream, std::uint64_t &bytes) {
  constexpr std::size_t maximum_parameter_bytes = 1024*1024;
  std::string buffer(maximum_parameter_bytes, '\0');
  stream.seekg(0, std::ios::beg);
  stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  const std::size_t got = static_cast<std::size_t>(stream.gcount());
  buffer.resize(got);
  constexpr const char marker[] = "<par_end>\n";
  const std::size_t end = buffer.find(marker);
  if (end == std::string::npos) {
    throw std::runtime_error("<par_end> was not found in the first MiB");
  }
  bytes = end+sizeof(marker)-1;
  buffer.resize(static_cast<std::size_t>(bytes));
  return buffer;
}

using Parameters = std::map<std::string, std::string>;

Parameters ParseParameters(const std::string &dump) {
  Parameters result;
  std::istringstream input(dump);
  std::string block;
  std::string line;
  while (std::getline(input, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    if (line.front() == '<' && line.back() == '>') {
      block = line.substr(1, line.size()-2);
      continue;
    }
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || block.empty()) continue;
    std::string value = line.substr(equals+1);
    const std::size_t comment = value.find('#');
    if (comment != std::string::npos) value.erase(comment);
    result[block+"/"+Trim(line.substr(0, equals))] = Trim(value);
  }
  return result;
}

const std::string &RequireParameter(const Parameters &parameters,
                                    const std::string &name) {
  const auto found = parameters.find(name);
  if (found == parameters.end()) {
    throw std::runtime_error("legacy restart is missing parameter " + name);
  }
  return found->second;
}

int IntegerParameter(const Parameters &parameters, const std::string &name) {
  const std::string &text = RequireParameter(parameters, name);
  std::size_t used = 0;
  const int result = std::stoi(text, &used);
  if (used != text.size()) throw std::runtime_error("invalid integer " + name);
  return result;
}

double RealParameter(const Parameters &parameters, const std::string &name) {
  const std::string &text = RequireParameter(parameters, name);
  std::size_t used = 0;
  const double result = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(result)) {
    throw std::runtime_error("invalid real " + name);
  }
  return result;
}

void RequireEqual(const Parameters &parameters, const std::string &name,
                  const std::string &expected) {
  const std::string actual = RequireParameter(parameters, name);
  if (actual != expected) {
    throw std::runtime_error(name+"='"+actual+"', expected '"+expected+"'");
  }
}

LegacyHeader ReadLegacyHeader(std::ifstream &stream,
                              const std::uint64_t parameter_bytes) {
  std::vector<unsigned char> bytes(kLegacyHeaderBytes);
  ReadAt(stream, parameter_bytes, bytes.data(), bytes.size());
  LegacyHeader result;
  std::size_t offset = 0;
  auto copy = [&](void *destination, const std::size_t count) {
    std::memcpy(destination, bytes.data()+offset, count);
    offset += count;
  };
  copy(&result.nmb, sizeof(result.nmb));
  copy(&result.root_level, sizeof(result.root_level));
  offset += 9*sizeof(double);  // RegionSize
  copy(result.mesh_indices, sizeof(result.mesh_indices));
  copy(result.block_indices, sizeof(result.block_indices));
  copy(&result.time, sizeof(result.time));
  copy(&result.dt, sizeof(result.dt));
  copy(&result.cycle, sizeof(result.cycle));
  if (offset != bytes.size()) throw std::runtime_error("legacy header ABI mismatch");
  return result;
}

void SetMaterialTables(
    materials::MaterialMixtureDevice &mixture,
    const std::vector<materials::IonmixTwoTemperatureTableDevice> &tables,
    const std::string &label) {
  const std::size_t bytes =
      tables.size()*sizeof(materials::IonmixTwoTemperatureTableDevice);
  HostArray1D<unsigned char> host(label+"-host", bytes);
  std::memcpy(host.data(), tables.data(), bytes);
  mixture.material_table_storage =
      DvceArray1D<unsigned char>(label+"-device", bytes);
  Kokkos::deep_copy(mixture.material_table_storage, host);
  mixture.material_tables =
      reinterpret_cast<const materials::IonmixTwoTemperatureTableDevice *>(
          mixture.material_table_storage.data());
}

materials::MaterialMixtureDevice MakeMixture(
    const materials::IonmixTwoTemperatureTableDevice &ch,
    const materials::IonmixTwoTemperatureTableDevice &he,
    const Parameters &parameters, const std::string &label,
    const Real density_to_cgs, const Real temperature_to_kelvin) {
  materials::MaterialMixtureDevice result;
  result.nmaterials = 2;
  result.use_tabular_eos = true;
  result.gamma_minus_one = 2.0/3.0;
  result.density_to_cgs = density_to_cgs;
  result.temperature_to_kelvin = temperature_to_kelvin;
  result.wave_speed_safety = 1.05;
  result.species = DvceArray1D<materials::SpeciesProperties>(label+"-species", 2);
  auto host_species = Kokkos::create_mirror_view(result.species);
  host_species(0).abar = RealParameter(parameters, "materials/material0_abar");
  host_species(0).zbar = RealParameter(parameters, "materials/material0_zbar");
  host_species(0).zeff = RealParameter(parameters, "materials/material0_zeff");
  host_species(1).abar = RealParameter(parameters, "materials/material1_abar");
  host_species(1).zbar = RealParameter(parameters, "materials/material1_zbar");
  host_species(1).zeff = RealParameter(parameters, "materials/material1_zeff");
  Kokkos::deep_copy(result.species, host_species);
  SetMaterialTables(result, {ch, he}, label+"-tables");
  return result;
}

void SetLocation(ExtremumLocation &location, const double value,
                 const std::uint64_t gid, const int local_i, const int local_j,
                 const int local_k, const LogicalLocation &block,
                 const int nx1, const int nx2, const int nx3) {
  if (!(value > location.value)) return;
  location.value = value;
  location.gid = gid;
  location.local_i = local_i;
  location.local_j = local_j;
  location.local_k = local_k;
  location.global_i = static_cast<std::int64_t>(block.lx1)*nx1+local_i;
  location.global_j = static_cast<std::int64_t>(block.lx2)*nx2+local_j;
  location.global_k = static_cast<std::int64_t>(block.lx3)*nx3+local_k;
}

void WriteLocation(std::ostream &out, const ExtremumLocation &location,
                   const std::string &indent) {
  out << "{\n"
      << indent << "  \"value\": " << location.value << ",\n"
      << indent << "  \"meshblock_gid\": " << location.gid << ",\n"
      << indent << "  \"local_active_index\": [" << location.local_i << ", "
      << location.local_j << ", " << location.local_k << "],\n"
      << indent << "  \"global_cell_index\": [" << location.global_i << ", "
      << location.global_j << ", " << location.global_k << "]\n"
      << indent << "}";
}

double RelativeDifference(const long double recovered, const long double stored) {
  return static_cast<double>(std::abs(recovered-stored)/
      std::max(std::abs(stored), static_cast<long double>(1.0e-300)));
}

int RunScan(const std::filesystem::path &restart_path,
            const std::filesystem::path &ch_path,
            const std::filesystem::path &he_path,
            const std::filesystem::path &json_path,
            const int chunk_blocks) {
  if (sizeof(Real) != sizeof(double)) {
    throw std::runtime_error("scanner and legacy restart must both use binary64 Real");
  }
  if (chunk_blocks <= 0) throw std::runtime_error("chunk_blocks must be positive");

  const std::uint64_t restart_bytes_before =
      std::filesystem::file_size(restart_path);
  const std::string restart_sha256_before = Sha256File(restart_path);
  RequireSha256(restart_path, restart_sha256_before, kExpectedRestartSha256);
  const std::string ch_sha256 = Sha256File(ch_path);
  const std::string he_sha256 = Sha256File(he_path);
  RequireSha256(ch_path, ch_sha256, kExpectedChTableSha256);
  RequireSha256(he_path, he_sha256, kExpectedHeTableSha256);

  std::ifstream restart(restart_path, std::ios::binary);
  if (!restart.is_open()) throw std::runtime_error("cannot open legacy restart");

  std::uint64_t parameter_bytes = 0;
  const std::string parameter_dump = ReadParameterDump(restart, parameter_bytes);
  const Parameters parameters = ParseParameters(parameter_dump);
  RequireEqual(parameters, "hydro/eos", "ideal");
  RequireEqual(parameters, "hydro/two_temperature", "true");
  RequireEqual(parameters, "materials/eos_table_bounds", "clamp");
  RequireEqual(parameters, "materials/eos_table_interpolation", "geometric");
  RequireEqual(parameters, "thermal_radiation/enabled", "true");
  RequireEqual(parameters, "mesh_refinement/refinement", "none");
  if (IntegerParameter(parameters, "hydro/nscalars") != kUserScalars ||
      IntegerParameter(parameters, "materials/nmaterials") != 2 ||
      IntegerParameter(parameters, "materials/scalar_index") != 0 ||
      IntegerParameter(parameters, "thermal_radiation/n_groups") !=
          kRadiationGroups) {
    throw std::runtime_error("restart is not the legacy 2-material/1-scalar/20-group deck");
  }

  const LegacyHeader header = ReadLegacyHeader(restart, parameter_bytes);
  if (header.time != kExpectedRestartTimeNs ||
      header.cycle != kExpectedRestartCycle || !std::isfinite(header.dt) ||
      !(header.dt > 0.0)) {
    throw std::runtime_error(
        "restart is not the audited t=1 ns, cycle-48420 checkpoint");
  }
  const int nghost = IntegerParameter(parameters, "mesh/nghost");
  const int nx1 = IntegerParameter(parameters, "meshblock/nx1");
  const int nx2 = IntegerParameter(parameters, "meshblock/nx2");
  const int nx3 = IntegerParameter(parameters, "meshblock/nx3");
  const int nout1 = nx1+2*nghost;
  const int nout2 = nx2+2*nghost;
  const int nout3 = nx3+2*nghost;
  const std::uint64_t full_cells =
      static_cast<std::uint64_t>(nout1)*nout2*nout3;
  const std::uint64_t active_cells =
      static_cast<std::uint64_t>(nx1)*nx2*nx3;
  const std::uint64_t field_bytes = full_cells*sizeof(double);
  const std::uint64_t expected_stride =
      kLegacyVariables*field_bytes;
  const Real dx1 = (RealParameter(parameters, "mesh/x1max")-
                    RealParameter(parameters, "mesh/x1min"))/
                   IntegerParameter(parameters, "mesh/nx1");
  const Real dx2 = (RealParameter(parameters, "mesh/x2max")-
                    RealParameter(parameters, "mesh/x2min"))/
                   IntegerParameter(parameters, "mesh/nx2");
  const Real dx3 = (RealParameter(parameters, "mesh/x3max")-
                    RealParameter(parameters, "mesh/x3min"))/
                   IntegerParameter(parameters, "mesh/nx3");
  const Real cell_volume = dx1*dx2*dx3;

  if (header.nmb != 1024 || header.root_level != 4 ||
      nx1 != 32 || nx2 != 32 || nx3 != 32 || nghost != 2) {
    throw std::runtime_error("legacy mesh dimensions differ from the audited t=1 file");
  }
  if (header.mesh_indices[0] != nghost ||
      header.mesh_indices[1] != IntegerParameter(parameters, "mesh/nx1") ||
      header.mesh_indices[2] != IntegerParameter(parameters, "mesh/nx2") ||
      header.mesh_indices[3] != IntegerParameter(parameters, "mesh/nx3") ||
      header.block_indices[0] != nghost || header.block_indices[1] != nx1 ||
      header.block_indices[2] != nx2 || header.block_indices[3] != nx3) {
    throw std::runtime_error("legacy binary header disagrees with its parameter dump");
  }

  const std::uint64_t list_offset = parameter_bytes+kLegacyHeaderBytes;
  std::vector<LogicalLocation> locations(static_cast<std::size_t>(header.nmb));
  ReadAt(restart, list_offset, locations.data(),
         locations.size()*sizeof(LogicalLocation));
  const int blocks_x1 = IntegerParameter(parameters, "mesh/nx1")/nx1;
  const int blocks_x2 = IntegerParameter(parameters, "mesh/nx2")/nx2;
  const int blocks_x3 = IntegerParameter(parameters, "mesh/nx3")/nx3;
  if (blocks_x1*blocks_x2*blocks_x3 != header.nmb) {
    throw std::runtime_error("legacy root-grid block count is inconsistent");
  }
  std::vector<unsigned char> seen_locations(static_cast<std::size_t>(header.nmb), 0);
  for (const LogicalLocation &location : locations) {
    if (location.level != header.root_level || location.lx1 < 0 ||
        location.lx1 >= blocks_x1 || location.lx2 < 0 ||
        location.lx2 >= blocks_x2 || location.lx3 < 0 ||
        location.lx3 >= blocks_x3) {
      throw std::runtime_error("legacy checkpoint is not a uniform root-grid tiling");
    }
    const int logical_index =
        (location.lx3*blocks_x2+location.lx2)*blocks_x1+location.lx1;
    if (seen_locations[logical_index] != 0) {
      throw std::runtime_error("legacy checkpoint repeats a root-grid MeshBlock");
    }
    seen_locations[logical_index] = 1;
  }
  const std::uint64_t list_bytes = static_cast<std::uint64_t>(header.nmb)*
      (sizeof(LogicalLocation)+sizeof(float));
  const std::uint64_t stride_offset = list_offset+list_bytes;
  std::uint64_t stride = 0;
  ReadAt(restart, stride_offset, &stride, sizeof(stride));
  const std::uint64_t payload_offset = stride_offset+sizeof(stride);
  if (stride != expected_stride) {
    throw std::runtime_error("legacy MeshBlock stride is not 28*36^3 doubles");
  }
  if (payload_offset+static_cast<std::uint64_t>(header.nmb)*stride !=
      restart_bytes_before) {
    throw std::runtime_error("restart file size does not exactly match its legacy layout");
  }

  const Real length_cgs = RealParameter(parameters, "units/length_cgs");
  const Real mass_cgs = RealParameter(parameters, "units/mass_cgs");
  const Real time_cgs = RealParameter(parameters, "units/time_cgs");
  const Real mu = RealParameter(parameters, "units/mu");
  const Real velocity_cgs = length_cgs/time_cgs;
  const Real density_to_cgs = mass_cgs/(length_cgs*length_cgs*length_cgs);
  const Real pressure_cgs = mass_cgs*velocity_cgs*velocity_cgs/
                            (length_cgs*length_cgs*length_cgs);
  constexpr Real atomic_mass_unit_cgs = 1.660538921e-24;
  constexpr Real boltzmann_cgs = 1.3806488e-16;
  const Real temperature_to_kelvin =
      velocity_cgs*velocity_cgs*mu*atomic_mass_unit_cgs/boltzmann_cgs;

  materials::IonmixTwoTemperatureTableOptions clamp_options;
  clamp_options.bounds_policy = materials::IonmixBoundsPolicy::clamp;
  clamp_options.geometric_interpolation = true;
  clamp_options.density_to_cgs = density_to_cgs;
  clamp_options.temperature_to_kelvin = temperature_to_kelvin;
  clamp_options.pressure_from_cgs = 1.0/pressure_cgs;
  clamp_options.specific_energy_from_cgs = 1.0/(velocity_cgs*velocity_cgs);
  materials::IonmixTwoTemperatureTableOptions flash_options = clamp_options;
  flash_options.bounds_policy = materials::IonmixBoundsPolicy::flash_extrapolate;

  materials::IonmixTwoTemperatureTable clamp_ch(ch_path.string(), clamp_options);
  materials::IonmixTwoTemperatureTable clamp_he(he_path.string(), clamp_options);
  materials::IonmixTwoTemperatureTable flash_ch(ch_path.string(), flash_options);
  materials::IonmixTwoTemperatureTable flash_he(he_path.string(), flash_options);
  const auto &ch_metadata = flash_ch.Metadata();
  const auto &he_metadata = flash_he.Metadata();
  if (RequireParameter(parameters, "materials/material0_eos_table_fingerprint") !=
          ch_metadata.file_fingerprint ||
      RequireParameter(parameters, "materials/material1_eos_table_fingerprint") !=
          he_metadata.file_fingerprint) {
    throw std::runtime_error("current CH/He table fingerprints differ from the restart");
  }

  const materials::MaterialMixtureDevice clamp = MakeMixture(
      clamp_ch.DeviceData(), clamp_he.DeviceData(), parameters, "legacy-clamp",
      density_to_cgs, temperature_to_kelvin);
  const materials::MaterialMixtureDevice flash = MakeMixture(
      flash_ch.DeviceData(), flash_he.DeviceData(), parameters, "flash-extrapolate",
      density_to_cgs, temperature_to_kelvin);

  const int maximum_chunk = std::min(chunk_blocks, header.nmb);
  HostArray3D<Real> host_fields(
      "historical-restart-host-fields", maximum_chunk, 4,
      static_cast<int>(full_cells));
  DvceArray3D<Real> device_fields(
      "historical-restart-device-fields", maximum_chunk, 4,
      static_cast<int>(full_cells));
  HostArray1D<CellResult> host_results(
      "historical-eos-host-results", maximum_chunk*active_cells);
  DvceArray1D<CellResult> device_results(
      "historical-eos-device-results", maximum_chunk*active_cells);

  Metrics metrics;
  metrics.total_cells = static_cast<std::uint64_t>(header.nmb)*active_cells;
  const auto start = std::chrono::steady_clock::now();
  for (int first_gid = 0; first_gid < header.nmb; first_gid += maximum_chunk) {
    const int blocks = std::min(maximum_chunk, header.nmb-first_gid);
    for (int block = 0; block < blocks; ++block) {
      const std::uint64_t base = payload_offset+
          static_cast<std::uint64_t>(first_gid+block)*stride;
      ReadAt(restart, base+kDensityVariable*field_bytes,
             &host_fields(block, 0, 0), field_bytes);
      ReadAt(restart, base+kMaterial0Variable*field_bytes,
             &host_fields(block, 1, 0), 3*field_bytes);
    }
    Kokkos::deep_copy(device_fields, host_fields);
    const std::uint64_t chunk_active = static_cast<std::uint64_t>(blocks)*active_cells;
    const auto fields = device_fields;
    const auto results = device_results;
    Kokkos::parallel_for(
        "historical-dci-eos-inverse",
        Kokkos::RangePolicy<DevExeSpace>(0, chunk_active),
        KOKKOS_LAMBDA(const std::uint64_t linear) {
          const int block = static_cast<int>(linear/active_cells);
          const std::uint64_t local = linear-block*active_cells;
          const int local_i = static_cast<int>(local%nx1);
          const int local_j = static_cast<int>((local/nx1)%nx2);
          const int local_k = static_cast<int>(local/(nx1*nx2));
          const int i = local_i+nghost;
          const int j = local_j+nghost;
          const int k = local_k+nghost;
          const int cell = (k*nout2+j)*nout1+i;
          const Real density = fields(block, 0, cell);
          const Real material0_density = fields(block, 1, cell);
          const Real ion_energy_density = fields(block, 2, cell);
          const Real electron_energy_density = fields(block, 3, cell);
          CellResult result;
          if (!Kokkos::isfinite(density) || !(density > 0.0) ||
              !Kokkos::isfinite(material0_density) ||
              !Kokkos::isfinite(ion_energy_density) ||
              !(ion_energy_density > 0.0) ||
              !Kokkos::isfinite(electron_energy_density) ||
              !(electron_energy_density > 0.0)) {
            result.status |= 1;
            results(linear) = result;
            return;
          }
          const Real raw_y0 = material0_density/density;
          if (raw_y0 < 0.0 || raw_y0 > 1.0) result.status |= 4;
          // Use the compatibility accessor that preserves the producer's exact
          // y0/(1-y0) arithmetic rather than normalizing a new two-entry array.
          const materials::MaterialComposition composition =
              flash.CompositionFromY0(raw_y0);
          const Real ion_specific_energy = ion_energy_density/density;
          const Real electron_specific_energy = electron_energy_density/density;
          result.old_ion_flags = clamp.IonSpecificEnergyQueryFlags(
              density, ion_specific_energy, composition);
          result.old_electron_flags = clamp.ElectronSpecificEnergyQueryFlags(
              density, electron_specific_energy, composition);
          constexpr int high = materials::ionmix_energy_above_table;
          if (((result.old_ion_flags | result.old_electron_flags) & high) != 0) {
            const Real endpoint_temperature =
                clamp.MaximumTransportTemperature(composition);
            const materials::MaterialThermodynamicState endpoint =
                clamp.StateFromRhoTemperaturesNoSound(
                    density, endpoint_temperature, endpoint_temperature,
                    composition);
            if (!Kokkos::isfinite(endpoint.ion_specific_internal_energy) ||
                !(endpoint.ion_specific_internal_energy > 0.0) ||
                !Kokkos::isfinite(endpoint.electron_specific_internal_energy) ||
                !(endpoint.electron_specific_internal_energy > 0.0)) {
              result.status |= 8;
            } else {
              if ((result.old_ion_flags & high) != 0) {
                result.endpoint_ratio_ion = ion_specific_energy/
                    endpoint.ion_specific_internal_energy;
              }
              if ((result.old_electron_flags & high) != 0) {
                result.endpoint_ratio_electron = electron_specific_energy/
                    endpoint.electron_specific_internal_energy;
              }
            }
          }
          const materials::MaterialThermodynamicState recovered =
              flash.StateFromRhoSpecificEnergiesNoSound(
                  density, ion_specific_energy, electron_specific_energy,
                  composition);
          result.flash_flags = recovered.query_flags;
          result.ion_temperature = recovered.ion_temperature;
          result.electron_temperature = recovered.electron_temperature;
          result.ion_pressure = recovered.ion_pressure;
          result.electron_pressure = recovered.electron_pressure;
          result.recovered_ion_energy = recovered.ion_specific_internal_energy;
          result.recovered_electron_energy =
              recovered.electron_specific_internal_energy;
          if (!Kokkos::isfinite(result.ion_temperature) ||
              !(result.ion_temperature > 0.0) ||
              !Kokkos::isfinite(result.electron_temperature) ||
              !(result.electron_temperature > 0.0) ||
              !Kokkos::isfinite(result.ion_pressure) ||
              !(result.ion_pressure > 0.0) ||
              !Kokkos::isfinite(result.electron_pressure) ||
              !(result.electron_pressure > 0.0) ||
              !Kokkos::isfinite(result.recovered_ion_energy) ||
              !(result.recovered_ion_energy > 0.0) ||
              !Kokkos::isfinite(result.recovered_electron_energy) ||
              !(result.recovered_electron_energy > 0.0)) {
            result.status |= 2;
          }
          results(linear) = result;
        });
    Kokkos::deep_copy(host_results, device_results);

    for (std::uint64_t linear = 0; linear < chunk_active; ++linear) {
      const int block = static_cast<int>(linear/active_cells);
      const std::uint64_t local = linear-block*active_cells;
      const int local_i = static_cast<int>(local%nx1);
      const int local_j = static_cast<int>((local/nx1)%nx2);
      const int local_k = static_cast<int>(local/(nx1*nx2));
      const int cell = ((local_k+nghost)*nout2+(local_j+nghost))*nout1+
                       local_i+nghost;
      const std::uint64_t gid = first_gid+block;
      const CellResult &result = host_results(linear);
      if ((result.status & 4) != 0) ++metrics.raw_fraction_outside_cells;
      if ((result.status & (1 | 4)) != 0) {
        ++metrics.invalid_scanned_eos_input_cells;
        continue;
      }
      ++metrics.valid_scanned_eos_input_cells;
      if ((result.status & 2) != 0) ++metrics.flash_invalid_state_cells;
      if ((result.status & 8) != 0) ++metrics.endpoint_invalid_cells;

      constexpr int high = materials::ionmix_energy_above_table;
      constexpr int low = materials::ionmix_energy_below_table;
      const bool ion_high = (result.old_ion_flags & high) != 0;
      const bool electron_high = (result.old_electron_flags & high) != 0;
      if (ion_high) ++metrics.old_ion_high_cells;
      if (electron_high) ++metrics.old_electron_high_cells;
      if (ion_high && electron_high) ++metrics.old_both_high_cells;
      if (ion_high || electron_high) ++metrics.old_high_cells;
      if (((result.old_ion_flags | result.old_electron_flags) & low) != 0) {
        ++metrics.old_below_cells;
      }
      if ((result.flash_flags & high) != 0) ++metrics.flash_high_cells;
      if ((result.flash_flags & low) != 0) ++metrics.flash_below_cells;
      if ((result.flash_flags & materials::ionmix_density_below_table) != 0) {
        ++metrics.flash_density_below_cells;
      }
      if ((result.flash_flags & materials::ionmix_density_above_table) != 0) {
        ++metrics.flash_density_above_cells;
      }
      if ((result.flash_flags & materials::ionmix_temperature_below_table) != 0) {
        ++metrics.flash_temperature_below_cells;
      }
      if ((result.flash_flags & materials::ionmix_temperature_above_table) != 0) {
        ++metrics.flash_temperature_above_cells;
      }

      const Real density = host_fields(block, 0, cell);
      const Real stored_ion_density = host_fields(block, 2, cell);
      const Real stored_electron_density = host_fields(block, 3, cell);
      const Real target_ion = stored_ion_density/density;
      const Real target_electron = stored_electron_density/density;
      const double ion_abs = std::abs(result.recovered_ion_energy-target_ion);
      const double electron_abs =
          std::abs(result.recovered_electron_energy-target_electron);
      const double ion_rel = ion_abs/std::max(std::abs(target_ion), 1.0e-300);
      const double electron_rel =
          electron_abs/std::max(std::abs(target_electron), 1.0e-300);
      metrics.maximum_ion_absolute_specific_residual = std::max(
          metrics.maximum_ion_absolute_specific_residual, ion_abs);
      metrics.maximum_electron_absolute_specific_residual = std::max(
          metrics.maximum_electron_absolute_specific_residual, electron_abs);
      SetLocation(metrics.ion_relative_residual, ion_rel, gid,
                  local_i, local_j, local_k, locations[gid], nx1, nx2, nx3);
      SetLocation(metrics.electron_relative_residual, electron_rel, gid,
                  local_i, local_j, local_k, locations[gid], nx1, nx2, nx3);
      metrics.maximum_ion_relative_residual = std::max(
          metrics.maximum_ion_relative_residual, ion_rel);
      metrics.maximum_electron_relative_residual = std::max(
          metrics.maximum_electron_relative_residual, electron_rel);
      metrics.stored_ion_energy_density_sum += stored_ion_density;
      metrics.stored_electron_energy_density_sum += stored_electron_density;
      metrics.recovered_ion_energy_density_sum +=
          static_cast<long double>(density)*result.recovered_ion_energy;
      metrics.recovered_electron_energy_density_sum +=
          static_cast<long double>(density)*result.recovered_electron_energy;

      metrics.minimum_ion_temperature = std::min(
          metrics.minimum_ion_temperature,
          static_cast<double>(result.ion_temperature));
      metrics.maximum_ion_temperature = std::max(
          metrics.maximum_ion_temperature,
          static_cast<double>(result.ion_temperature));
      metrics.minimum_electron_temperature = std::min(
          metrics.minimum_electron_temperature,
          static_cast<double>(result.electron_temperature));
      metrics.maximum_electron_temperature = std::max(
          metrics.maximum_electron_temperature,
          static_cast<double>(result.electron_temperature));
      metrics.minimum_ion_pressure = std::min(
          metrics.minimum_ion_pressure, static_cast<double>(result.ion_pressure));
      metrics.maximum_ion_pressure = std::max(
          metrics.maximum_ion_pressure, static_cast<double>(result.ion_pressure));
      metrics.minimum_electron_pressure = std::min(
          metrics.minimum_electron_pressure,
          static_cast<double>(result.electron_pressure));
      metrics.maximum_electron_pressure = std::max(
          metrics.maximum_electron_pressure,
          static_cast<double>(result.electron_pressure));
      if (result.endpoint_ratio_ion > 0.0) {
        SetLocation(metrics.endpoint_ratio_ion, result.endpoint_ratio_ion, gid,
                    local_i, local_j, local_k, locations[gid], nx1, nx2, nx3);
        SetLocation(metrics.endpoint_ratio, result.endpoint_ratio_ion, gid,
                    local_i, local_j, local_k, locations[gid], nx1, nx2, nx3);
      }
      if (result.endpoint_ratio_electron > 0.0) {
        SetLocation(metrics.endpoint_ratio_electron,
                    result.endpoint_ratio_electron, gid, local_i, local_j,
                    local_k, locations[gid], nx1, nx2, nx3);
        SetLocation(metrics.endpoint_ratio, result.endpoint_ratio_electron, gid,
                    local_i, local_j, local_k, locations[gid], nx1, nx2, nx3);
      }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    std::cerr << "scanned " << (first_gid+blocks) << "/" << header.nmb
              << " MeshBlocks in " << std::fixed << std::setprecision(1)
              << elapsed << " s\n";
  }

  const double elapsed_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now()-start).count();
  restart.close();
  if (restart.fail()) throw std::runtime_error("failed to close legacy restart input");
  const std::uint64_t restart_bytes_after =
      std::filesystem::file_size(restart_path);
  const std::string restart_sha256_after = Sha256File(restart_path);
  const bool restart_identity_unchanged =
      restart_bytes_after == restart_bytes_before &&
      restart_sha256_after == restart_sha256_before;
  if (!restart_identity_unchanged) {
    throw std::runtime_error("legacy restart identity changed during read-only scan");
  }

  const double ion_global_relative = RelativeDifference(
      metrics.recovered_ion_energy_density_sum,
      metrics.stored_ion_energy_density_sum);
  const double electron_global_relative = RelativeDifference(
      metrics.recovered_electron_energy_density_sum,
      metrics.stored_electron_energy_density_sum);
  const bool passed =
      metrics.invalid_scanned_eos_input_cells == 0 &&
      metrics.raw_fraction_outside_cells == 0 && metrics.old_high_cells > 0 &&
      metrics.flash_high_cells == 0 && metrics.flash_invalid_state_cells == 0 &&
      metrics.endpoint_invalid_cells == 0 &&
      metrics.maximum_ion_relative_residual <= kCellRelativeEnergyTolerance &&
      metrics.maximum_electron_relative_residual <= kCellRelativeEnergyTolerance &&
      ion_global_relative <= kGlobalRelativeEnergyTolerance &&
      electron_global_relative <= kGlobalRelativeEnergyTolerance;

  if (!json_path.parent_path().empty()) {
    std::filesystem::create_directories(json_path.parent_path());
  }
  std::ofstream json(json_path, std::ios::binary | std::ios::trunc);
  if (!json.is_open()) throw std::runtime_error("cannot create JSON evidence file");
  json << std::setprecision(17) << std::scientific;
  json << "{\n"
       << "  \"schema_version\": 2,\n"
       << "  \"status\": \"" << (passed ? "pass" : "fail") << "\",\n"
       << "  \"purpose\": \"Read-only validation of current flash-extrapolate EOS "
          "on the SHA-256-identified legacy DCI t=1 ns state\",\n"
       << "  \"source\": {\n"
       << "    \"git_head_at_build\": \"" << ATHENAK_SOURCE_REVISION << "\",\n"
       << "    \"legacy_layout_revision\": \"" << kLayoutRevision << "\",\n"
       << "    \"scanner_cpp_sha256\": \"" << SCANNER_SOURCE_SHA256 << "\",\n"
       << "    \"material_mixture_hpp_sha256\": \""
       << MATERIAL_MIXTURE_SHA256 << "\",\n"
       << "    \"ionmix_hpp_sha256\": \"" << IONMIX_HEADER_SHA256 << "\",\n"
       << "    \"ionmix_cpp_sha256\": \"" << IONMIX_SOURCE_SHA256 << "\"\n"
       << "  },\n"
       << "  \"restart\": {\n"
       << "    \"path\": \"" << JsonEscape(std::filesystem::absolute(restart_path).string())
       << "\",\n"
       << "    \"bytes\": " << restart_bytes_before << ",\n"
       << "    \"expected_sha256\": \"" << kExpectedRestartSha256 << "\",\n"
       << "    \"sha256_before_scan\": \"" << restart_sha256_before << "\",\n"
       << "    \"sha256_after_scan\": \"" << restart_sha256_after << "\",\n"
       << "    \"access_mode\": \"read-only\",\n"
       << "    \"identity_unchanged_during_scan\": "
       << (restart_identity_unchanged ? "true" : "false") << ",\n"
       << "    \"expected_time_ns\": " << kExpectedRestartTimeNs << ",\n"
       << "    \"time_ns\": " << header.time << ",\n"
       << "    \"dt_ns\": " << header.dt << ",\n"
       << "    \"expected_cycle\": " << kExpectedRestartCycle << ",\n"
       << "    \"cycle\": " << header.cycle << ",\n"
       << "    \"meshblocks\": " << header.nmb << "\n"
       << "  },\n"
       << "  \"legacy_layout\": {\n"
       << "    \"parameter_bytes\": " << parameter_bytes << ",\n"
       << "    \"abi_header_bytes\": " << kLegacyHeaderBytes << ",\n"
       << "    \"location_and_cost_bytes\": " << list_bytes << ",\n"
       << "    \"stride_field_offset\": " << stride_offset << ",\n"
       << "    \"payload_offset\": " << payload_offset << ",\n"
       << "    \"meshblock_stride_bytes\": " << stride << ",\n"
       << "    \"real_bytes\": " << sizeof(double) << ",\n"
       << "    \"variables\": " << kLegacyVariables << ",\n"
       << "    \"array_order\": [\"rho\", \"mom1\", \"mom2\", \"mom3\", "
          "\"etot\", \"rho_X_CH\", \"rho_eion\", \"rho_eelectron\", "
          "\"radiation_group_0..19\"],\n"
       << "    \"meshblock_active_shape\": [" << nx1 << ", " << nx2 << ", "
       << nx3 << "],\n"
       << "    \"meshblock_restart_shape\": [" << nout1 << ", " << nout2 << ", "
       << nout3 << "],\n"
       << "    \"ghost_zones\": " << nghost << ",\n"
       << "    \"selected_payload_bytes_read\": "
       << static_cast<std::uint64_t>(header.nmb)*4*field_bytes << ",\n"
       << "    \"uniform_root_grid_verified\": true,\n"
       << "    \"exact_file_size_match\": true\n"
       << "  },\n"
       << "  \"units\": {\n"
       << "    \"density_to_cgs\": " << density_to_cgs << ",\n"
       << "    \"temperature_to_kelvin\": " << temperature_to_kelvin << ",\n"
       << "    \"pressure_from_cgs\": " << 1.0/pressure_cgs << ",\n"
       << "    \"specific_energy_from_cgs\": "
       << 1.0/(velocity_cgs*velocity_cgs) << "\n"
       << "  },\n"
       << "  \"tables\": {\n"
       << "    \"CH\": {\"path\": \"" << JsonEscape(std::filesystem::absolute(ch_path).string())
       << "\", \"sha256\": \"" << ch_sha256
       << "\", \"expected_sha256\": \"" << kExpectedChTableSha256
       << "\", \"fingerprint\": \"" << ch_metadata.file_fingerprint
       << "\", \"bytes\": " << ch_metadata.file_size
       << ", \"density_nodes\": " << ch_metadata.ndensity
       << ", \"temperature_nodes\": " << ch_metadata.ntemperature
       << ", \"maximum_temperature_kelvin\": "
       << ch_metadata.maximum_temperature_kelvin << "},\n"
       << "    \"He\": {\"path\": \"" << JsonEscape(std::filesystem::absolute(he_path).string())
       << "\", \"sha256\": \"" << he_sha256
       << "\", \"expected_sha256\": \"" << kExpectedHeTableSha256
       << "\", \"fingerprint\": \"" << he_metadata.file_fingerprint
       << "\", \"bytes\": " << he_metadata.file_size
       << ", \"density_nodes\": " << he_metadata.ndensity
       << ", \"temperature_nodes\": " << he_metadata.ntemperature
       << ", \"maximum_temperature_kelvin\": "
       << he_metadata.maximum_temperature_kelvin << "}\n"
       << "  },\n"
       << "  \"scan\": {\n"
       << "    \"active_cells\": " << metrics.total_cells << ",\n"
       << "    \"valid_scanned_eos_input_cells\": "
       << metrics.valid_scanned_eos_input_cells << ",\n"
       << "    \"invalid_scanned_eos_input_cells\": "
       << metrics.invalid_scanned_eos_input_cells << ",\n"
       << "    \"raw_CH_fraction_outside_0_1_cells\": "
       << metrics.raw_fraction_outside_cells << ",\n"
       << "    \"elapsed_seconds\": " << elapsed_seconds << ",\n"
       << "    \"old_clamp\": {\n"
       << "      \"energy_above_table_cells\": " << metrics.old_high_cells << ",\n"
       << "      \"ion_energy_above_table_cells\": "
       << metrics.old_ion_high_cells << ",\n"
       << "      \"electron_energy_above_table_cells\": "
       << metrics.old_electron_high_cells << ",\n"
       << "      \"both_components_above_table_cells\": "
       << metrics.old_both_high_cells << ",\n"
       << "      \"energy_below_table_cells\": " << metrics.old_below_cells << ",\n"
       << "      \"maximum_target_to_endpoint_energy_ratio\": ";
  WriteLocation(json, metrics.endpoint_ratio, "      ");
  json << ",\n      \"maximum_ion_target_to_endpoint_ratio\": ";
  WriteLocation(json, metrics.endpoint_ratio_ion, "      ");
  json << ",\n      \"maximum_electron_target_to_endpoint_ratio\": ";
  WriteLocation(json, metrics.endpoint_ratio_electron, "      ");
  json << "\n    },\n"
       << "    \"flash_extrapolate\": {\n"
       << "      \"energy_above_table_cells\": " << metrics.flash_high_cells << ",\n"
       << "      \"energy_below_table_cells\": " << metrics.flash_below_cells << ",\n"
       << "      \"density_below_table_cells\": "
       << metrics.flash_density_below_cells << ",\n"
       << "      \"density_above_table_cells\": "
       << metrics.flash_density_above_cells << ",\n"
       << "      \"temperature_below_table_cells\": "
       << metrics.flash_temperature_below_cells << ",\n"
       << "      \"temperature_above_table_cells\": "
       << metrics.flash_temperature_above_cells << ",\n"
       << "      \"nonfinite_or_nonpositive_state_cells\": "
       << metrics.flash_invalid_state_cells << ",\n"
       << "      \"invalid_endpoint_state_cells\": "
       << metrics.endpoint_invalid_cells << ",\n"
       << "      \"ion_temperature_code_range\": ["
       << metrics.minimum_ion_temperature << ", " << metrics.maximum_ion_temperature
       << "],\n"
       << "      \"electron_temperature_code_range\": ["
       << metrics.minimum_electron_temperature << ", "
       << metrics.maximum_electron_temperature << "],\n"
       << "      \"ion_temperature_kelvin_range\": ["
       << metrics.minimum_ion_temperature*temperature_to_kelvin << ", "
       << metrics.maximum_ion_temperature*temperature_to_kelvin << "],\n"
       << "      \"electron_temperature_kelvin_range\": ["
       << metrics.minimum_electron_temperature*temperature_to_kelvin << ", "
       << metrics.maximum_electron_temperature*temperature_to_kelvin << "],\n"
       << "      \"ion_pressure_code_range\": [" << metrics.minimum_ion_pressure
       << ", " << metrics.maximum_ion_pressure << "],\n"
       << "      \"electron_pressure_code_range\": ["
       << metrics.minimum_electron_pressure << ", "
       << metrics.maximum_electron_pressure << "]\n"
       << "    },\n"
       << "    \"energy_conservation\": {\n"
       << "      \"stored_ion_energy_density_sum\": "
       << static_cast<double>(metrics.stored_ion_energy_density_sum) << ",\n"
       << "      \"recovered_ion_energy_density_sum\": "
       << static_cast<double>(metrics.recovered_ion_energy_density_sum) << ",\n"
       << "      \"stored_electron_energy_density_sum\": "
       << static_cast<double>(metrics.stored_electron_energy_density_sum) << ",\n"
       << "      \"recovered_electron_energy_density_sum\": "
       << static_cast<double>(metrics.recovered_electron_energy_density_sum) << ",\n"
       << "      \"uniform_cell_volume_code\": " << cell_volume << ",\n"
       << "      \"stored_volume_integrated_ion_energy\": "
       << static_cast<double>(metrics.stored_ion_energy_density_sum*cell_volume)
       << ",\n"
       << "      \"recovered_volume_integrated_ion_energy\": "
       << static_cast<double>(metrics.recovered_ion_energy_density_sum*cell_volume)
       << ",\n"
       << "      \"stored_volume_integrated_electron_energy\": "
       << static_cast<double>(metrics.stored_electron_energy_density_sum*cell_volume)
       << ",\n"
       << "      \"recovered_volume_integrated_electron_energy\": "
       << static_cast<double>(
              metrics.recovered_electron_energy_density_sum*cell_volume) << ",\n"
       << "      \"ion_global_relative_residual\": " << ion_global_relative << ",\n"
       << "      \"electron_global_relative_residual\": "
       << electron_global_relative << ",\n"
       << "      \"maximum_ion_absolute_specific_residual\": "
       << metrics.maximum_ion_absolute_specific_residual << ",\n"
       << "      \"maximum_electron_absolute_specific_residual\": "
       << metrics.maximum_electron_absolute_specific_residual << ",\n"
       << "      \"maximum_ion_relative_residual\": ";
  WriteLocation(json, metrics.ion_relative_residual, "      ");
  json << ",\n      \"maximum_electron_relative_residual\": ";
  WriteLocation(json, metrics.electron_relative_residual, "      ");
  json << "\n    }\n"
       << "  },\n"
       << "  \"acceptance_criteria\": {\n"
       << "    \"restart_sha256_matches_expected\": true,\n"
       << "    \"restart_time_and_cycle_match_expected\": true,\n"
       << "    \"restart_identity_unchanged_during_read_only_scan\": "
       << (restart_identity_unchanged ? "true" : "false") << ",\n"
       << "    \"CH_and_He_table_sha256_match_expected\": true,\n"
       << "    \"all_required_scanned_eos_inputs_are_valid\": "
       << ((metrics.invalid_scanned_eos_input_cells == 0 &&
            metrics.raw_fraction_outside_cells == 0) ? "true" : "false") << ",\n"
       << "    \"old_clamp_finds_high_energy_cells\": "
       << (metrics.old_high_cells > 0 ? "true" : "false") << ",\n"
       << "    \"flash_has_no_high_energy_flags\": "
       << (metrics.flash_high_cells == 0 ? "true" : "false") << ",\n"
       << "    \"flash_states_are_finite_and_positive\": "
       << (metrics.flash_invalid_state_cells == 0 ? "true" : "false") << ",\n"
       << "    \"low_endpoint_flags_are_within_energy_tolerance\": "
       << ((metrics.maximum_ion_relative_residual <= kCellRelativeEnergyTolerance &&
            metrics.maximum_electron_relative_residual <=
                kCellRelativeEnergyTolerance) ? "true" : "false") << ",\n"
       << "    \"all_stored_component_energies_are_recovered\": "
       << ((metrics.maximum_ion_relative_residual <= kCellRelativeEnergyTolerance &&
            metrics.maximum_electron_relative_residual <= kCellRelativeEnergyTolerance &&
            ion_global_relative <= kGlobalRelativeEnergyTolerance &&
            electron_global_relative <= kGlobalRelativeEnergyTolerance)
               ? "true" : "false") << ",\n"
       << "    \"cell_relative_energy_tolerance\": "
       << kCellRelativeEnergyTolerance << ",\n"
       << "    \"global_relative_energy_tolerance\": "
       << kGlobalRelativeEnergyTolerance << "\n"
       << "  }\n"
       << "}\n";
  json.close();
  if (!json) throw std::runtime_error("failed while writing JSON evidence");

  std::cout << (passed ? "PASS" : "FAIL") << ": old clamp high cells="
            << metrics.old_high_cells << ", flash high cells="
            << metrics.flash_high_cells << ", max endpoint ratio="
            << metrics.endpoint_ratio.value << ", max ion/electron relative residual="
            << metrics.maximum_ion_relative_residual << "/"
            << metrics.maximum_electron_relative_residual << "\n"
            << "evidence: " << json_path << "\n";
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: " << argv[0]
              << " RESTART CH_TABLE HE_TABLE OUTPUT_JSON [CHUNK_BLOCKS]\n";
    return EXIT_FAILURE;
  }
  const int chunk_blocks = (argc == 6) ? std::stoi(argv[5]) : 32;
  Kokkos::initialize(argc, argv);
  int result = EXIT_FAILURE;
  try {
    result = RunScan(argv[1], argv[2], argv[3], argv[4], chunk_blocks);
  } catch (const std::exception &error) {
    std::cerr << "historical EOS restart scan failed: " << error.what() << "\n";
  }
  Kokkos::finalize();
  return result;
}
