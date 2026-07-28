//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file table_eos.cpp
//! \brief Portable EOS table readers and nonrelativistic Hydro/MHD closures.

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "eos/table_eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "utils/tr_table.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

[[noreturn]] void TableEOSError(const std::string &filename, const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << "EOS table '" << filename << "': " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

class Tokens {
 public:
  explicit Tokens(const std::string &filename) : filename_(filename) {
    std::ifstream input(filename);
    if (!input.is_open()) TableEOSError(filename, "could not open file");
    std::string line;
    while (std::getline(input, line)) {
      std::size_t comment = line.find('#');
      if (comment != std::string::npos) line.erase(comment);
      std::istringstream row(line);
      std::string token;
      while (row >> token) values_.push_back(token);
    }
  }

  void Expect(const std::string &expected) {
    std::string actual = Next(expected);
    if (actual != expected) {
      TableEOSError(filename_, "expected '" + expected + "', found '" + actual + "'");
    }
  }

  int Integer(const std::string &name) {
    std::string token = Next(name);
    std::size_t used = 0;
    int result = 0;
    try {
      result = std::stoi(token, &used);
    } catch (const std::exception &) {
      TableEOSError(filename_, name + " is not an integer");
    }
    if (used != token.size()) TableEOSError(filename_, name + " is not an integer");
    return result;
  }

  std::string String(const std::string &name) {
    return Next(name);
  }

  Real Number(const std::string &name) {
    std::string token = Next(name);
    std::size_t used = 0;
    Real result = 0.0;
    try {
      result = static_cast<Real>(std::stod(token, &used));
    } catch (const std::exception &) {
      TableEOSError(filename_, name + " is not numeric");
    }
    if (used != token.size() || !std::isfinite(result)) {
      TableEOSError(filename_, name + " must be finite and numeric");
    }
    return result;
  }

  void Finish() {
    if (position_ != values_.size()) {
      TableEOSError(filename_, "unexpected token '" + values_[position_] + "' after end");
    }
  }

 private:
  std::string Next(const std::string &name) {
    if (position_ >= values_.size()) {
      TableEOSError(filename_, "unexpected end while reading " + name);
    }
    return values_[position_++];
  }

  std::string filename_;
  std::vector<std::string> values_;
  std::size_t position_ = 0;
};

Real ReadScale(ParameterInput *pin, const std::string &block, const std::string &name,
               Real default_value, const std::string &filename) {
  Real value = pin->GetOrAddReal(block, name, default_value);
  if (!(value > 0.0) || !std::isfinite(value)) {
    TableEOSError(filename, "<" + block + ">/" + name + " must be positive");
  }
  return value;
}

bool IsNativeTable(const std::string &filename) {
  std::ifstream input(filename);
  if (!input.is_open()) TableEOSError(filename, "could not open file");
  std::string line;
  while (std::getline(input, line)) {
    std::size_t comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream row(line);
    std::string token;
    if (row >> token) return token == "athenak_eos_table";
  }
  TableEOSError(filename, "file contains no table data");
}

std::size_t TableValueIndex(int field, int density, int temperature,
                            int ndensity, int ntemperature) {
  return (static_cast<std::size_t>(field)*ndensity+density)*ntemperature+temperature;
}

struct MaterialFieldDescriptor {
  const char *name;
  TableEOSData::Field field;
  bool strictly_positive;
};

constexpr std::array<MaterialFieldDescriptor, TableEOSData::nmaterial_fields>
    material_fields = {{
      {"gamma1", TableEOSData::gamma1, true},
      {"gamma3m1", TableEOSData::gamma3_minus_one, true},
      {"zbar", TableEOSData::mean_ionization, false},
      {"zeff", TableEOSData::effective_charge, false},
      {"abar", TableEOSData::mean_atomic_mass, true},
      {"mu", TableEOSData::mean_molecular_weight, true},
    }};

int MaterialFieldIndex(TableEOSData::Field field) {
  return static_cast<int>(field)-static_cast<int>(TableEOSData::gamma1);
}

const MaterialFieldDescriptor *FindMaterialField(const std::string &name) {
  for (const auto &field : material_fields) {
    if (name == field.name) return &field;
  }
  return nullptr;
}

void ValidateMaterialValue(const std::string &filename,
                           const MaterialFieldDescriptor &field, Real value) {
  if (!std::isfinite(value) || (field.strictly_positive ? value <= 0.0 : value < 0.0)) {
    TableEOSError(filename, std::string(field.name) +
                  (field.strictly_positive ? " must be finite and positive" :
                                             " must be finite and non-negative"));
  }
}

std::string MaterialFieldSummary(
    const std::array<int, TableEOSData::nmaterial_fields> &available) {
  std::string result;
  for (int i = 0; i < TableEOSData::nmaterial_fields; ++i) {
    if (available[i] == 0) continue;
    if (!result.empty()) result += ",";
    result += material_fields[i].name;
  }
  return result.empty() ? "none" : result;
}

void LoadTableEOS(const std::string &block, MeshBlockPack *pp, ParameterInput *pin,
                  EOS_Data &eos) {
  if (!pin->DoesParameterExist(block, "table_file") &&
      !pin->DoesParameterExist(block, "table")) {
    TableEOSError("<unset>", "<" + block + "> requires table_file or table");
  }
  std::string filename = pin->DoesParameterExist(block, "table_file")
      ? pin->GetString(block, "table_file") : pin->GetString(block, "table");
  if (pp->pcoord->is_special_relativistic || pp->pcoord->is_general_relativistic ||
      pp->pcoord->is_dynamical_relativistic) {
    TableEOSError(filename, "tabulated EOS is supported only for Newtonian dynamics");
  }
  if (pp->pmesh->multilevel) {
    TableEOSError(filename, "tabulated EOS does not yet support SMR/AMR prolongation");
  }
  if (eos.sfloor > static_cast<Real>(FLT_MIN)) {
    TableEOSError(filename,
                  "sfloor is unavailable because the table has no entropy field");
  }
  if (pin->DoesBlockExist("ion-neutral") || pin->DoesBlockExist("radiation")) {
    TableEOSError(filename,
                  "tabulated EOS does not support ion-neutral or radiation fluids");
  }

  std::string units = pin->GetOrAddString(block, "table_unit_system", "code");
  Real density_default = 1.0;
  Real temperature_default = 1.0;
  Real pressure_default = 1.0;
  Real specific_eint_default = 1.0;
  Real sound_speed2_default = 1.0;
  if (units == "cgs") {
    if (pp->punit == nullptr) {
      TableEOSError(filename, "table_unit_system=cgs requires a <units> block");
    }
    density_default = 1.0/pp->punit->density_cgs();
    temperature_default = 1.0/pp->punit->temperature_cgs();
    pressure_default = 1.0/pp->punit->pressure_cgs();
    specific_eint_default = 1.0/SQR(pp->punit->velocity_cgs());
    sound_speed2_default = specific_eint_default;
  } else if (units != "code") {
    TableEOSError(filename, "table_unit_system must be 'code' or 'cgs'");
  }

  Real density_scale = ReadScale(
      pin, block, "table_density_scale", density_default, filename);
  Real temperature_scale = ReadScale(
      pin, block, "table_temperature_scale", temperature_default, filename);
  Real pressure_scale = ReadScale(
      pin, block, "table_pressure_scale", pressure_default, filename);
  Real specific_eint_scale = ReadScale(
      pin, block, "table_specific_eint_scale", specific_eint_default, filename);
  Real sound_speed2_scale = ReadScale(
      pin, block, "table_sound_speed2_scale", sound_speed2_default, filename);

  std::string bounds = pin->GetOrAddString(block, "table_bounds", "clamp");
  if (bounds == "clamp") {
    eos.table.bounds_error = 0;
  } else if (bounds == "error") {
    eos.table.bounds_error = 1;
  } else {
    TableEOSError(filename, "table_bounds must be 'clamp' or 'error'");
  }

  int ndensity = 0;
  int ntemperature = 0;
  std::vector<Real> density;
  std::vector<Real> temperature;
  std::vector<Real> values;
  std::array<int, TableEOSData::nmaterial_fields> material_available{};
  std::string file_format;
  bool read_file = true;
#if MPI_PARALLEL_ENABLED
  read_file = (global_variable::my_rank == 0);
#endif

  if (read_file) {
    if (IsNativeTable(filename)) {
      Tokens tokens(filename);
      tokens.Expect("athenak_eos_table");
      int format_version = tokens.Integer("format version");
      if (format_version != 1 && format_version != 2) {
        TableEOSError(filename, "unsupported format version");
      }
      file_format = "native ASCII v" + std::to_string(format_version);
      tokens.Expect("dimensions");
      ndensity = tokens.Integer("density dimension");
      ntemperature = tokens.Integer("temperature dimension");
      if (ndensity < 2 || ntemperature < 2) {
        TableEOSError(filename, "both dimensions must contain at least two points");
      }
      std::size_t nvalues = static_cast<std::size_t>(TableEOSData::nfields)*
                            ndensity*ntemperature;
      if (nvalues > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TableEOSError(filename, "table contains too many values");
      }
      density.resize(ndensity);
      temperature.resize(ntemperature);
      values.assign(nvalues, std::numeric_limits<Real>::quiet_NaN());

      tokens.Expect("density");
      for (int i = 0; i < ndensity; ++i) {
        Real value = tokens.Number("density")*density_scale;
        if (!(value > 0.0) || !std::isfinite(value)) {
          TableEOSError(filename, "scaled densities must be finite and positive");
        }
        density[i] = std::log(value);
      }

      tokens.Expect("temperature");
      for (int i = 0; i < ntemperature; ++i) {
        Real value = tokens.Number("temperature")*temperature_scale;
        if (!(value > 0.0) || !std::isfinite(value)) {
          TableEOSError(filename, "scaled temperatures must be finite and positive");
        }
        temperature[i] = std::log(value);
      }

      constexpr int ncore_fields = 3;
      const char *labels[ncore_fields] = {
          "pressure", "specific_internal_energy", "sound_speed_squared"};
      Real scales[ncore_fields] = {
          pressure_scale, specific_eint_scale, sound_speed2_scale};
      for (int field = 0; field < ncore_fields; ++field) {
        tokens.Expect(labels[field]);
        for (int ir = 0; ir < ndensity; ++ir) {
          for (int it = 0; it < ntemperature; ++it) {
            Real value = tokens.Number(labels[field])*scales[field];
            if (!(value > 0.0) || !std::isfinite(value)) {
              TableEOSError(filename, std::string("scaled ")+labels[field]+
                            " values must be finite and positive");
            }
            values[TableValueIndex(field, ir, it, ndensity, ntemperature)] =
                std::log(value);
          }
        }
      }
      if (format_version == 2) {
        tokens.Expect("material_fields");
        int number_material_fields = tokens.Integer("material field count");
        if (number_material_fields < 0 ||
            number_material_fields > TableEOSData::nmaterial_fields) {
          TableEOSError(filename, "material field count must be between 0 and " +
                        std::to_string(TableEOSData::nmaterial_fields));
        }
        for (int n = 0; n < number_material_fields; ++n) {
          std::string name = tokens.String("material field name");
          const MaterialFieldDescriptor *descriptor = FindMaterialField(name);
          if (descriptor == nullptr) {
            TableEOSError(filename, "unknown material field '" + name + "'");
          }
          int availability_index = MaterialFieldIndex(descriptor->field);
          if (material_available[availability_index] != 0) {
            TableEOSError(filename, "duplicate material field '" + name + "'");
          }
          material_available[availability_index] = 1;
          for (int ir = 0; ir < ndensity; ++ir) {
            for (int it = 0; it < ntemperature; ++it) {
              Real value = tokens.Number(name);
              ValidateMaterialValue(filename, *descriptor, value);
              values[TableValueIndex(descriptor->field, ir, it,
                                     ndensity, ntemperature)] = value;
            }
          }
        }
      }
      tokens.Expect("end");
      tokens.Finish();
    } else {
      file_format = "TableReader binary";
      TableReader::Table table;
      TableReader::ReadResult result = table.ReadTable(filename);
      if (result.error != TableReader::ReadResult::SUCCESS) {
        TableEOSError(filename, result.message);
      }
      const auto metadata = table.GetMetadata();
      auto log_base = metadata.find("log_axis_base");
      if (log_base != metadata.end() && log_base->second != "e") {
        TableEOSError(filename, "TableReader log_axis_base must be 'e'");
      }
      const auto &points = table.GetPointInfo();
      if (points.size() != 2 || points[0].first != "logrho" ||
          points[1].first != "logtemp") {
        TableEOSError(filename,
                      "TableReader axes must be ordered as logrho, logtemp");
      }
      if (points[0].second > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          points[1].second > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TableEOSError(filename, "table dimensions are too large");
      }
      ndensity = static_cast<int>(points[0].second);
      ntemperature = static_cast<int>(points[1].second);
      if (ndensity < 2 || ntemperature < 2 || !table.HasField("logpress") ||
          !table.HasField("logeps") || !table.HasField("logcs2")) {
        TableEOSError(filename,
                      "TableReader table needs two-point axes and "
                      "logpress/logeps/logcs2");
      }
      std::size_t nvalues = static_cast<std::size_t>(TableEOSData::nfields)*
                            ndensity*ntemperature;
      if (nvalues > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TableEOSError(filename, "table contains too many values");
      }
      density.resize(ndensity);
      temperature.resize(ntemperature);
      values.assign(nvalues, std::numeric_limits<Real>::quiet_NaN());
      const Real axis_scales[2] = {density_scale, temperature_scale};
      const char *axis_names[2] = {"logrho", "logtemp"};
      std::vector<Real> *axes[2] = {&density, &temperature};
      const int axis_sizes[2] = {ndensity, ntemperature};
      for (int axis = 0; axis < 2; ++axis) {
        const double *source = table[axis_names[axis]];
        Real shift = std::log(axis_scales[axis]);
        for (int i = 0; i < axis_sizes[axis]; ++i) {
          (*axes[axis])[i] = static_cast<Real>(source[i])+shift;
          if (!std::isfinite((*axes[axis])[i])) {
            TableEOSError(filename, std::string(axis_names[axis])+
                          " contains a non-finite scaled value");
          }
        }
      }
      constexpr int ncore_fields = 3;
      const char *fields[ncore_fields] = {"logpress", "logeps", "logcs2"};
      const Real scales[ncore_fields] = {
          pressure_scale, specific_eint_scale, sound_speed2_scale};
      for (int field = 0; field < ncore_fields; ++field) {
        const double *source = table[fields[field]];
        Real shift = std::log(scales[field]);
        for (int ir = 0; ir < ndensity; ++ir) {
          for (int it = 0; it < ntemperature; ++it) {
            Real value = static_cast<Real>(source[ir*ntemperature+it])+shift;
            if (!std::isfinite(value)) {
              TableEOSError(filename, std::string(fields[field])+
                            " contains a non-finite scaled value");
            }
            values[TableValueIndex(field, ir, it, ndensity, ntemperature)] = value;
          }
        }
      }
      for (const auto &descriptor : material_fields) {
        if (!table.HasField(descriptor.name)) continue;
        int availability_index = MaterialFieldIndex(descriptor.field);
        material_available[availability_index] = 1;
        const double *source = table[descriptor.name];
        for (int ir = 0; ir < ndensity; ++ir) {
          for (int it = 0; it < ntemperature; ++it) {
            Real value = static_cast<Real>(source[ir*ntemperature+it]);
            ValidateMaterialValue(filename, descriptor, value);
            values[TableValueIndex(descriptor.field, ir, it,
                                   ndensity, ntemperature)] = value;
          }
        }
      }
    }

    for (int i = 1; i < ndensity; ++i) {
      if (!(density[i] > density[i-1])) {
        TableEOSError(filename, "density axis must be strictly increasing");
      }
    }
    for (int i = 1; i < ntemperature; ++i) {
      if (!(temperature[i] > temperature[i-1])) {
        TableEOSError(filename, "temperature axis must be strictly increasing");
      }
    }
    for (int ir = 0; ir < ndensity; ++ir) {
      for (int it = 1; it < ntemperature; ++it) {
        if (!(values[TableValueIndex(TableEOSData::log_specific_eint, ir, it,
                                     ndensity, ntemperature)] >
              values[TableValueIndex(TableEOSData::log_specific_eint, ir, it-1,
                                     ndensity, ntemperature)])) {
          TableEOSError(filename,
                        "specific internal energy must increase with temperature");
        }
        if (!(values[TableValueIndex(TableEOSData::log_pressure, ir, it,
                                     ndensity, ntemperature)] >
              values[TableValueIndex(TableEOSData::log_pressure, ir, it-1,
                                     ndensity, ntemperature)])) {
          TableEOSError(filename, "pressure must increase with temperature");
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  int dimensions[2] = {ndensity, ntemperature};
  int ierr = MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) TableEOSError(filename, "could not broadcast dimensions");
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(material_available.data(), TableEOSData::nmaterial_fields,
                     MPI_INT, 0, MPI_COMM_WORLD);
  }
  if (ierr != MPI_SUCCESS) {
    TableEOSError(filename, "could not broadcast material-field metadata");
  }
  ndensity = dimensions[0];
  ntemperature = dimensions[1];
  if (ndensity < 2 || ntemperature < 2) {
    TableEOSError(filename,
                  "broadcast table dimensions must contain at least two points");
  }
  std::size_t nvalues_size = static_cast<std::size_t>(TableEOSData::nfields)*
                             static_cast<std::size_t>(ndensity)*
                             static_cast<std::size_t>(ntemperature);
  if (nvalues_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    TableEOSError(filename, "broadcast table contains too many values");
  }
  int nvalues = static_cast<int>(nvalues_size);
  if (!read_file) {
    density.resize(ndensity);
    temperature.resize(ntemperature);
    values.resize(nvalues);
  }
  ierr = MPI_Bcast(density.data(), ndensity, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(temperature.data(), ntemperature,
                     MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(values.data(), nvalues, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr != MPI_SUCCESS) TableEOSError(filename, "could not broadcast table values");
#endif

  eos.table.ndensity = ndensity;
  eos.table.ntemperature = ntemperature;
  eos.table.has_gamma1 = material_available[0] != 0;
  eos.table.has_gamma3_minus_one = material_available[1] != 0;
  eos.table.has_mean_ionization = material_available[2] != 0;
  eos.table.has_effective_charge = material_available[3] != 0;
  eos.table.has_mean_atomic_mass = material_available[4] != 0;
  eos.table.has_mean_molecular_weight = material_available[5] != 0;
  eos.table.log_density = DvceArray1D<Real>("eos-table-density", ndensity);
  eos.table.log_temperature = DvceArray1D<Real>("eos-table-temperature", ntemperature);
  eos.table.values = DvceArray3D<Real>(
      "eos-table-values", TableEOSData::nfields, ndensity, ntemperature);
  eos.table.log_density_h = HostArray1D<Real>("eos-table-density-host", ndensity);
  eos.table.log_temperature_h =
      HostArray1D<Real>("eos-table-temperature-host", ntemperature);
  eos.table.values_h = HostArray3D<Real>(
      "eos-table-values-host", TableEOSData::nfields, ndensity, ntemperature);
  for (int ir = 0; ir < ndensity; ++ir) eos.table.log_density_h(ir) = density[ir];
  for (int it = 0; it < ntemperature; ++it) {
    eos.table.log_temperature_h(it) = temperature[it];
  }
  for (int field = 0; field < TableEOSData::nfields; ++field) {
    for (int ir = 0; ir < ndensity; ++ir) {
      for (int it = 0; it < ntemperature; ++it) {
        eos.table.values_h(field, ir, it) =
            values[TableValueIndex(field, ir, it, ndensity, ntemperature)];
      }
    }
  }
  Kokkos::deep_copy(eos.table.log_density, eos.table.log_density_h);
  Kokkos::deep_copy(eos.table.log_temperature, eos.table.log_temperature_h);
  Kokkos::deep_copy(eos.table.values, eos.table.values_h);
  eos.is_ideal = true;
  eos.is_gamma_law = false;
  eos.is_table = true;
  eos.gamma = 5.0/3.0;
  eos.iso_cs = 0.0;
  if (pin->DoesParameterExist(block, "tfloor_kelvin")) {
    if (pp->punit == nullptr) {
      TableEOSError(filename, "tfloor_kelvin requires a <units> block");
    }
    eos.tfloor = pin->GetReal(block, "tfloor_kelvin")/pp->punit->temperature_cgs();
  }

  if (global_variable::my_rank == 0) {
    std::cout << "Loaded " << ndensity << " x " << ntemperature
              << " tabulated EOS from " << filename << " (" << file_format
              << ", " << bounds << " bounds, material fields: "
              << MaterialFieldSummary(material_available) << ")" << std::endl;
  }
}

KOKKOS_INLINE_FUNCTION
void HydroP2C(const HydPrim1D &w, HydCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e+0.5*w.d*(SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
}

KOKKOS_INLINE_FUNCTION
void MHDP2C(const MHDPrim1D &w, MHDCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e+0.5*w.d*(SQR(w.vx)+SQR(w.vy)+SQR(w.vz))
        +0.5*(SQR(w.bx)+SQR(w.by)+SQR(w.bz));
  u.by = w.by;
  u.bz = w.bz;
}

} // namespace

TabulatedHydro::TabulatedHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  LoadTableEOS("hydro", pp, pin, eos_data);
}

void TabulatedHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                const bool only_testfloors,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku) {
  int nhydro = pmy_pack->phydro->nhydro;
  int nscalars = pmy_pack->phydro->nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  auto fofc = pmy_pack->phydro->fofc;
  const int ni = iu-il+1;
  const int nji = (ju-jl+1)*ni;
  const int nkji = (ku-kl+1)*nji;
  int nfloord = 0, nfloore = 0;
  Kokkos::parallel_reduce(
      "table_hyd_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(int index, int &sumd, int &sume) {
        int m = index/nkji;
        int k = (index-m*nkji)/nji+kl;
        int j = (index-m*nkji-(k-kl)*nji)/ni+jl;
        int i = index-m*nkji-(k-kl)*nji-(j-jl)*ni+il;
        HydCons1D u;
        u.d = cons(m, IDN, k, j, i);
        u.mx = cons(m, IM1, k, j, i);
        u.my = cons(m, IM2, k, j, i);
        u.mz = cons(m, IM3, k, j, i);
        u.e = cons(m, IEN, k, j, i);
        bool density_floor = u.d < eos.dfloor;
        if (density_floor) u.d = eos.dfloor;
        HydPrim1D w;
        w.d = u.d;
        w.vx = u.mx/u.d;
        w.vy = u.my/u.d;
        w.vz = u.mz/u.d;
        w.e = u.e-0.5/u.d*(SQR(u.mx)+SQR(u.my)+SQR(u.mz));
        Real energy_floor = eos.HydroInternalEnergyDensityFloor(w.d);
        bool energy_floor_used = w.e < energy_floor;
        if (energy_floor_used) w.e = energy_floor;
        if (only_testfloors) {
          if (density_floor || energy_floor_used) {
            fofc(m, k, j, i) = true;
            ++sumd;
          }
          return;
        }
        HydroP2C(w, u);
        cons(m, IDN, k, j, i) = u.d;
        cons(m, IM1, k, j, i) = u.mx;
        cons(m, IM2, k, j, i) = u.my;
        cons(m, IM3, k, j, i) = u.mz;
        cons(m, IEN, k, j, i) = u.e;
        prim(m, IDN, k, j, i) = w.d;
        prim(m, IVX, k, j, i) = w.vx;
        prim(m, IVY, k, j, i) = w.vy;
        prim(m, IVZ, k, j, i) = w.vz;
        prim(m, IEN, k, j, i) = w.e;
        for (int n = nhydro; n < nhydro+nscalars; ++n) {
          cons(m, n, k, j, i) = fmax(cons(m, n, k, j, i), 0.0);
          prim(m, n, k, j, i) = cons(m, n, k, j, i)/u.d;
        }
        if (density_floor) ++sumd;
        if (energy_floor_used) ++sume;
      }, Kokkos::Sum<int>(nfloord), Kokkos::Sum<int>(nfloore));
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore;
  }
}

void TabulatedHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku) {
  int nhydro = pmy_pack->phydro->nhydro;
  int nscalars = pmy_pack->phydro->nscalars;
  int nmb = pmy_pack->nmb_thispack;
  par_for("table_hyd_p2c", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    HydPrim1D w;
    w.d = prim(m, IDN, k, j, i);
    w.vx = prim(m, IVX, k, j, i);
    w.vy = prim(m, IVY, k, j, i);
    w.vz = prim(m, IVZ, k, j, i);
    w.e = prim(m, IEN, k, j, i);
    HydCons1D u;
    HydroP2C(w, u);
    cons(m, IDN, k, j, i) = u.d;
    cons(m, IM1, k, j, i) = u.mx;
    cons(m, IM2, k, j, i) = u.my;
    cons(m, IM3, k, j, i) = u.mz;
    cons(m, IEN, k, j, i) = u.e;
    for (int n = nhydro; n < nhydro+nscalars; ++n) {
      cons(m, n, k, j, i) = w.d*prim(m, n, k, j, i);
    }
  });
}

TabulatedMHD::TabulatedMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  LoadTableEOS("mhd", pp, pin, eos_data);
  eos_data.sigma_max = pin->GetOrAddReal("mhd", "sigma_max", FLT_MAX);
}

void TabulatedMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                              DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int nmhd = pmy_pack->pmhd->nmhd;
  int nscalars = pmy_pack->pmhd->nscalars;
  int nmb = pmy_pack->nmb_thispack;
  auto eos = eos_data;
  auto fofc = pmy_pack->pmhd->fofc;
  const int ni = iu-il+1;
  const int nji = (ju-jl+1)*ni;
  const int nkji = (ku-kl+1)*nji;
  int nfloord = 0, nfloore = 0;
  Kokkos::parallel_reduce(
      "table_mhd_c2p", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(int index, int &sumd, int &sume) {
        int m = index/nkji;
        int k = (index-m*nkji)/nji+kl;
        int j = (index-m*nkji-(k-kl)*nji)/ni+jl;
        int i = index-m*nkji-(k-kl)*nji-(j-jl)*ni+il;
        MHDCons1D u;
        u.d = cons(m, IDN, k, j, i);
        u.mx = cons(m, IM1, k, j, i);
        u.my = cons(m, IM2, k, j, i);
        u.mz = cons(m, IM3, k, j, i);
        u.e = cons(m, IEN, k, j, i);
        if (only_testfloors) {
          u.bx = bcc(m, IBX, k, j, i);
          u.by = bcc(m, IBY, k, j, i);
          u.bz = bcc(m, IBZ, k, j, i);
        } else {
          u.bx = 0.5*(b.x1f(m, k, j, i)+b.x1f(m, k, j, i+1));
          u.by = 0.5*(b.x2f(m, k, j, i)+b.x2f(m, k, j+1, i));
          u.bz = 0.5*(b.x3f(m, k, j, i)+b.x3f(m, k+1, j, i));
        }
        Real magnetic2 = SQR(u.bx)+SQR(u.by)+SQR(u.bz);
        Real density_floor_value = fmax(eos.dfloor, magnetic2/eos.sigma_max);
        bool density_floor = u.d < density_floor_value;
        if (density_floor) u.d = density_floor_value;
        MHDPrim1D w;
        w.d = u.d;
        w.vx = u.mx/u.d;
        w.vy = u.my/u.d;
        w.vz = u.mz/u.d;
        w.bx = u.bx;
        w.by = u.by;
        w.bz = u.bz;
        w.e = u.e-0.5/u.d*(SQR(u.mx)+SQR(u.my)+SQR(u.mz))-0.5*magnetic2;
        Real energy_floor = eos.HydroInternalEnergyDensityFloor(w.d);
        bool energy_floor_used = w.e < energy_floor;
        if (energy_floor_used) w.e = energy_floor;
        if (only_testfloors) {
          if (density_floor || energy_floor_used) {
            fofc(m, k, j, i) = true;
            ++sumd;
          }
          return;
        }
        MHDP2C(w, u);
        cons(m, IDN, k, j, i) = u.d;
        cons(m, IM1, k, j, i) = u.mx;
        cons(m, IM2, k, j, i) = u.my;
        cons(m, IM3, k, j, i) = u.mz;
        cons(m, IEN, k, j, i) = u.e;
        prim(m, IDN, k, j, i) = w.d;
        prim(m, IVX, k, j, i) = w.vx;
        prim(m, IVY, k, j, i) = w.vy;
        prim(m, IVZ, k, j, i) = w.vz;
        prim(m, IEN, k, j, i) = w.e;
        bcc(m, IBX, k, j, i) = w.bx;
        bcc(m, IBY, k, j, i) = w.by;
        bcc(m, IBZ, k, j, i) = w.bz;
        for (int n = nmhd; n < nmhd+nscalars; ++n) {
          cons(m, n, k, j, i) = fmax(cons(m, n, k, j, i), 0.0);
          prim(m, n, k, j, i) = cons(m, n, k, j, i)/u.d;
        }
        if (density_floor) ++sumd;
        if (energy_floor_used) ++sume;
      }, Kokkos::Sum<int>(nfloord), Kokkos::Sum<int>(nfloore));
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore;
  }
}

void TabulatedMHD::PrimToCons(const DvceArray5D<Real> &prim,
                              const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int nmhd = pmy_pack->pmhd->nmhd;
  int nscalars = pmy_pack->pmhd->nscalars;
  int nmb = pmy_pack->nmb_thispack;
  par_for("table_mhd_p2c", DevExeSpace(), 0, nmb-1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    MHDPrim1D w;
    w.d = prim(m, IDN, k, j, i);
    w.vx = prim(m, IVX, k, j, i);
    w.vy = prim(m, IVY, k, j, i);
    w.vz = prim(m, IVZ, k, j, i);
    w.e = prim(m, IEN, k, j, i);
    w.bx = bcc(m, IBX, k, j, i);
    w.by = bcc(m, IBY, k, j, i);
    w.bz = bcc(m, IBZ, k, j, i);
    MHDCons1D u;
    MHDP2C(w, u);
    cons(m, IDN, k, j, i) = u.d;
    cons(m, IM1, k, j, i) = u.mx;
    cons(m, IM2, k, j, i) = u.my;
    cons(m, IM3, k, j, i) = u.mz;
    cons(m, IEN, k, j, i) = u.e;
    for (int n = nmhd; n < nmhd+nscalars; ++n) {
      cons(m, n, k, j, i) = w.d*prim(m, n, k, j, i);
    }
  });
}
