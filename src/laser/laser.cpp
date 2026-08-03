//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser.cpp
//! \brief Construction, input validation, and ray allocation for laser transport.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "laser/laser.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

namespace {

std::string BeamKey(int beam, const std::string &name) {
  return "beam" + std::to_string(beam) + "_" + name;
}

[[noreturn]] void LaserInputError(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "<laser> " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::exit(EXIT_FAILURE);
}

bool Finite(Real value) {
  return std::isfinite(value);
}

void LoadPulseFile(const std::string &filename, Real time_scale, Real value_scale,
                   std::vector<Real> &time, std::vector<Real> &value) {
  bool read_file = true;
#if MPI_PARALLEL_ENABLED
  read_file = (global_variable::my_rank == 0);
#endif

  if (read_file) {
    std::ifstream input(filename);
    if (!input.is_open()) {
      LaserInputError("could not open pulse file '" + filename + "'");
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      std::size_t comment = line.find('#');
      if (comment != std::string::npos) line.erase(comment);
      std::replace(line.begin(), line.end(), ',', ' ');
      if (line.find_first_not_of(" \r\n") == std::string::npos) continue;

      std::istringstream row(line);
      Real t, v;
      if (!(row >> t >> v)) {
        LaserInputError("pulse file '" + filename + "' line " +
                        std::to_string(line_number) +
                        " must contain numeric time and power");
      }
      std::string extra;
      if (row >> extra) {
        LaserInputError("pulse file '" + filename + "' line " +
                        std::to_string(line_number) + " has more than two columns");
      }
      t *= time_scale;
      v *= value_scale;
      if (!Finite(t) || !Finite(v) || v < 0.0) {
        LaserInputError("pulse file '" + filename +
                        "' requires finite times and finite non-negative powers");
      }
      if (!time.empty() && t <= time.back()) {
        LaserInputError("pulse file '" + filename +
                        "' times must be strictly increasing");
      }
      time.push_back(t);
      value.push_back(v);
    }
    if (input.bad()) {
      LaserInputError("failed while reading pulse file '" + filename + "'");
    }
    if (time.size() < 2) {
      LaserInputError("pulse file '" + filename + "' must contain at least two rows");
    }
  }

#if MPI_PARALLEL_ENABLED
  int count = static_cast<int>(time.size());
  if (read_file && static_cast<std::size_t>(count) != time.size()) {
    LaserInputError("pulse file '" + filename + "' contains too many rows");
  }
  int ierr = MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) LaserInputError("could not broadcast pulse table size");
  if (!read_file) {
    time.resize(count);
    value.resize(count);
  }
  ierr = MPI_Bcast(time.data(), count, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  if (ierr == MPI_SUCCESS) {
    ierr = MPI_Bcast(value.data(), count, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  }
  if (ierr != MPI_SUCCESS) LaserInputError("could not broadcast pulse table values");
#endif
}

} // namespace

namespace laser {

Laser::Laser(MeshBlockPack *ppack, ParameterInput *pin) :
    cell_data("laser-cell-data", 1, 1, 1, 1, 1),
    coarse_cell_data("laser-coarse-cell-data", 1, 1, 1, 1, 1),
    ray_x("laser-ray-x", 1), ray_y("laser-ray-y", 1),
    ray_z("laser-ray-z", 1), ray_nx("laser-ray-nx", 1),
    ray_ny("laser-ray-ny", 1), ray_nz("laser-ray-nz", 1),
    ray_power("laser-ray-power", 1), ray_gid("laser-ray-gid", 1),
    ray_i("laser-ray-i", 1), ray_j("laser-ray-j", 1),
    ray_k("laser-ray-k", 1), ray_status("laser-ray-status", 1),
    ray_x0_("laser-ray-x0", 1), ray_y0_("laser-ray-y0", 1),
    ray_z0_("laser-ray-z0", 1), ray_nx0_("laser-ray-nx0", 1),
    ray_ny0_("laser-ray-ny0", 1), ray_nz0_("laser-ray-nz0", 1),
    ray_power0_("laser-ray-power0", 1),
    ray_power_fraction_("laser-ray-power-fraction", 1),
    ray_wavelength_("laser-ray-lambda", 1),
    ray_zeff_("laser-ray-zeff", 1),
    ray_constant_absorption_("laser-ray-constant-k", 1),
    ray_start_time_("laser-ray-start", 1), ray_end_time_("laser-ray-end", 1),
    ray_beam_("laser-ray-beam", 1), beam_power_("laser-beam-power", 1),
    ray_segments_("laser-ray-segments", 1),
    ray_reflections_("laser-ray-reflections", 1),
    ray_reflection_armed_("laser-ray-reflection-armed", 1),
    ray_last_turning_density_("laser-ray-last-turning-density", 1),
    ray_path_length_("laser-ray-path", 1),
    ray_kx_("laser-ray-kx", 1), ray_ky_("laser-ray-ky", 1),
    ray_kz_("laser-ray-kz", 1),
    ray_dispersion_error_("laser-ray-dispersion-error", 1),
    active_queue_a_("laser-active-a", 1), active_queue_b_("laser-active-b", 1),
    ray_destination_rank_("laser-ray-destination-rank", 1),
    global_block_info_("laser-global-block-info", 1),
    mpi_send_counts_("laser-mpi-send-counts", 1),
    mpi_send_offsets_("laser-mpi-send-offsets", 1),
    mpi_pack_cursors_("laser-mpi-pack-cursors", 1),
    mpi_send_packets_("laser-mpi-send-packets", 1),
    mpi_recv_packets_("laser-mpi-recv-packets", 1),
    mpi_host_send_packets_("laser-mpi-host-send-packets", 1),
    mpi_host_recv_packets_("laser-mpi-host-recv-packets", 1),
    device_diagnostics_("laser-diagnostics", 6),
    device_counters_("laser-counters", 8),
    cumulative_energy_start_("laser-energy-start", 1, 1, 1, 1, 1),
    pmy_pack_(ppack) {
  // Couple to whichever Newtonian 2T fluid the deck configured.  <hydro> and <mhd> are
  // mutually exclusive here: a two-fluid <ion-neutral> deck has no single 2T carrier.
  const bool have_hydro = ppack->phydro != nullptr &&
                          ppack->phydro->ptwo_temp != nullptr;
  const bool have_mhd = ppack->pmhd != nullptr &&
                        ppack->pmhd->ptwo_temp != nullptr;
  if (have_hydro && have_mhd) {
    LaserInputError("cannot be used with both <hydro> and <mhd> two-temperature fluids");
  }
  if (!have_hydro && !have_mhd) {
    LaserInputError("requires <hydro>/two_temperature=true or "
                    "<mhd>/two_temperature=true");
  }
  use_mhd_fluid_ = have_mhd;

  EquationOfState *fluid_eos =
      use_mhd_fluid_ ? ppack->pmhd->peos : ppack->phydro->peos;
  two_temperature::TwoTemperature *fluid_two_temp =
      use_mhd_fluid_ ? ppack->pmhd->ptwo_temp : ppack->phydro->ptwo_temp;
  materials::MaterialMixture *fluid_materials =
      use_mhd_fluid_ ? ppack->pmhd->pmaterials : ppack->phydro->pmaterials;

  if (!fluid_eos->eos_data.is_gamma_law ||
      ppack->pcoord->is_special_relativistic ||
      ppack->pcoord->is_general_relativistic) {
    LaserInputError("currently supports only Newtonian gamma-law hydro/MHD");
  }
  electron_index_ = fluid_two_temp->iele;
  gamma_minus_one_ = fluid_eos->eos_data.gamma - 1.0;
  electron_heat_capacity_fraction_ =
      fluid_two_temp->ElectronHeatCapacityFraction();
  use_material_mixture_ = fluid_materials != nullptr;
  if (use_material_mixture_) {
    material_mixture_ = fluid_materials->DeviceData();
  }

  std::string target = pin->GetOrAddString("laser", "deposition_target", "electron");
  if (target == "electron") {
    deposition_target_ = DepositionTarget::electron;
  } else if (target == "total") {
    deposition_target_ = DepositionTarget::total;
  } else {
    LaserInputError("deposition_target must be 'electron' or 'total'");
  }
  std::string temperature_model =
      pin->GetOrAddString("laser", "electron_temperature_model", "two_temperature");
  if (temperature_model != "two_temperature") {
    LaserInputError("electron_temperature_model must be 'two_temperature'");
  }
  std::string absorption =
      pin->GetOrAddString("laser", "absorption_model", "constant");
  if (absorption == "constant") {
    absorption_model_ = AbsorptionModel::constant;
  } else if (absorption == "inverse_bremsstrahlung") {
    absorption_model_ = AbsorptionModel::inverse_bremsstrahlung;
  } else {
    LaserInputError("absorption_model must be 'constant' or "
                    "'inverse_bremsstrahlung'");
  }
  std::string propagation =
      pin->GetOrAddString("laser", "model", "straight");
  if (propagation == "straight") {
    propagation_model_ = PropagationModel::straight;
  } else if (propagation == "refractive") {
    propagation_model_ = PropagationModel::refractive;
  } else {
    LaserInputError("model must be 'straight' or 'refractive'");
  }

  std::string unit_system = pin->GetOrAddString("laser", "unit_system", "code");
  if (unit_system == "code") {
    use_cgs_ = false;
  } else if (unit_system == "cgs") {
    use_cgs_ = true;
  } else {
    LaserInputError("unit_system must be 'code' or 'cgs'");
  }
  if (ppack->punit != nullptr) {
    length_scale_cgs_ = ppack->punit->length_cgs();
    density_scale_cgs_ = ppack->punit->density_cgs();
    temperature_scale_cgs_ = ppack->punit->temperature_cgs();
    unit_mean_molecular_weight_ = ppack->punit->mu();
    power_scale_cgs_ = ppack->punit->energy_cgs()/ppack->punit->time_cgs();
  }
  length_scale_cgs_ = pin->GetOrAddReal(
      "laser", "length_scale_cgs", length_scale_cgs_);
  density_scale_cgs_ = pin->GetOrAddReal(
      "laser", "density_scale_cgs", density_scale_cgs_);
  temperature_scale_cgs_ = pin->GetOrAddReal(
      "laser", "temperature_scale_cgs", temperature_scale_cgs_);
  unit_mean_molecular_weight_ = pin->GetOrAddReal(
      "laser", "temperature_mean_molecular_weight", unit_mean_molecular_weight_);
  power_scale_cgs_ = pin->GetOrAddReal(
      "laser", "power_scale_cgs", power_scale_cgs_);
  if (use_cgs_ &&
      (!Finite(length_scale_cgs_) || !(length_scale_cgs_ > 0.0) ||
       !Finite(power_scale_cgs_) || !(power_scale_cgs_ > 0.0))) {
    LaserInputError("unit_system=cgs requires finite positive length and power scales");
  }
  inverse_bremsstrahlung_coulomb_log_ = pin->GetOrAddReal(
      "laser", "inverse_bremsstrahlung_coulomb_log", -1.0);
  inverse_bremsstrahlung_temperature_floor_ = pin->GetOrAddReal(
      "laser", "inverse_bremsstrahlung_temperature_floor", 0.0);
  electron_number_per_gram_ = pin->GetOrAddReal(
      "laser", "electron_number_per_gram", 0.0);
  if (absorption_model_ == AbsorptionModel::inverse_bremsstrahlung &&
      (!Finite(length_scale_cgs_) || !(length_scale_cgs_ > 0.0) ||
       !Finite(density_scale_cgs_) || !(density_scale_cgs_ > 0.0) ||
       !Finite(temperature_scale_cgs_) || !(temperature_scale_cgs_ > 0.0) ||
       (use_material_mixture_ &&
        (!Finite(unit_mean_molecular_weight_) ||
         !(unit_mean_molecular_weight_ > 0.0))) ||
       (!use_material_mixture_ &&
        (!Finite(electron_number_per_gram_) || !(electron_number_per_gram_ > 0.0))))) {
    LaserInputError("inverse_bremsstrahlung requires finite positive cgs scales and "
                    "electron_number_per_gram when <materials> is absent");
  }
  if (!Finite(inverse_bremsstrahlung_coulomb_log_)) {
    LaserInputError("inverse_bremsstrahlung_coulomb_log must be finite");
  }
  if (!Finite(inverse_bremsstrahlung_temperature_floor_) ||
      inverse_bremsstrahlung_temperature_floor_ < 0.0) {
    LaserInputError("inverse_bremsstrahlung_temperature_floor must be finite and "
                    "non-negative");
  }

  max_segments_per_launch_ = pin->GetOrAddInteger(
      "laser", "max_segments_per_launch", 16);
  max_transport_iterations_ = pin->GetOrAddInteger(
      "laser", "max_transport_iterations", 64);
  max_mpi_waves_ = pin->GetOrAddInteger(
      "laser", "max_mpi_waves", 1024);
  minimum_power_fraction_ = pin->GetOrAddReal(
      "laser", "minimum_power_fraction", 1.0e-14);
  conservation_tolerance_ = pin->GetOrAddReal(
      "laser", "conservation_tolerance", 1.0e-10);
  periodic_transport_ = pin->GetOrAddBoolean(
      "laser", "periodic_transport", false);
  report_diagnostics_ = pin->GetOrAddBoolean(
      "laser", "report_diagnostics", true);
  gpu_aware_mpi_ = pin->GetOrAddBoolean(
      "laser", "gpu_aware_mpi", false);
  critical_reflection_ = pin->GetOrAddBoolean(
      "laser", "critical_reflection",
      propagation_model_ == PropagationModel::straight &&
      absorption_model_ == AbsorptionModel::inverse_bremsstrahlung);
  oblique_turning_ = pin->GetOrAddBoolean(
      "laser", "oblique_turning", true);
  max_reflections_per_ray_ = pin->GetOrAddInteger(
      "laser", "max_reflections_per_ray", 8);
  reflection_offset_fraction_ = pin->GetOrAddReal(
      "laser", "reflection_offset_fraction", 1.0e-10);
  reflection_hysteresis_fraction_ = pin->GetOrAddReal(
      "laser", "reflection_hysteresis_fraction", 0.0);
  refractive_cell_fraction_ = pin->GetOrAddReal(
      "laser", "refractive_cell_fraction", 0.25);
  refractive_curvature_fraction_ = pin->GetOrAddReal(
      "laser", "refractive_curvature_fraction", 0.25);
  refractive_tau_max_ = pin->GetOrAddReal(
      "laser", "refractive_tau_max", 0.25);
  dispersion_tolerance_ = pin->GetOrAddReal(
      "laser", "dispersion_tolerance", 1.0e-3);
  if (max_segments_per_launch_ <= 0 || max_transport_iterations_ <= 0 ||
      max_mpi_waves_ <= 0) {
    LaserInputError("segment, transport iteration, and MPI wave limits must be positive");
  }
  if (!Finite(minimum_power_fraction_) || minimum_power_fraction_ < 0.0 ||
      minimum_power_fraction_ >= 1.0) {
    LaserInputError("minimum_power_fraction must lie in [0,1)");
  }
  if (!Finite(conservation_tolerance_) || conservation_tolerance_ <= 0.0) {
    LaserInputError("conservation_tolerance must be positive");
  }
  conservation_tolerance_ = std::max(
      conservation_tolerance_,
      static_cast<Real>(128.0)*std::numeric_limits<Real>::epsilon());
  if (max_reflections_per_ray_ < 0 ||
      !Finite(reflection_offset_fraction_) || reflection_offset_fraction_ <= 0.0 ||
      !Finite(reflection_hysteresis_fraction_) ||
      reflection_hysteresis_fraction_ < 0.0 ||
      reflection_hysteresis_fraction_ >= 1.0) {
    LaserInputError("reflection limit must be non-negative, offset must be positive, "
                    "and hysteresis fraction must lie in [0,1)");
  }
  if (critical_reflection_ &&
      (!Finite(length_scale_cgs_) || !(length_scale_cgs_ > 0.0) ||
       !Finite(density_scale_cgs_) || !(density_scale_cgs_ > 0.0) ||
       (!use_material_mixture_ &&
        (!Finite(electron_number_per_gram_) || !(electron_number_per_gram_ > 0.0))))) {
    LaserInputError("critical reflection requires finite positive cgs density/length "
                    "scales and electron_number_per_gram when <materials> is absent");
  }
  if (propagation_model_ == PropagationModel::refractive) {
    if (critical_reflection_) {
      LaserInputError("refractive model must not also enable critical_reflection");
    }
    if (!Finite(length_scale_cgs_) || !(length_scale_cgs_ > 0.0) ||
        !Finite(density_scale_cgs_) || !(density_scale_cgs_ > 0.0) ||
        (!use_material_mixture_ &&
         (!Finite(electron_number_per_gram_) || !(electron_number_per_gram_ > 0.0)))) {
      LaserInputError("refractive model requires finite positive cgs density/length "
                      "scales and electron_number_per_gram when <materials> is absent");
    }
    if (!Finite(refractive_cell_fraction_) || refractive_cell_fraction_ <= 0.0 ||
        refractive_cell_fraction_ > 1.0 ||
        !Finite(refractive_curvature_fraction_) ||
        refractive_curvature_fraction_ <= 0.0 ||
        !Finite(refractive_tau_max_) || refractive_tau_max_ <= 0.0 ||
        !Finite(dispersion_tolerance_) || dispersion_tolerance_ <= 0.0) {
      LaserInputError("refractive step fractions, tau limit, and dispersion tolerance "
                      "must be positive (cell fraction must not exceed one)");
    }
  }
  if (periodic_transport_ && !ppack->pmesh->strictly_periodic) {
    LaserInputError("periodic_transport requires all mesh boundaries to be periodic");
  }

  int nbeams = pin->GetOrAddInteger("laser", "nbeams", 1);
  if (nbeams <= 0) LaserInputError("nbeams must be positive");
  beams_.reserve(nbeams);
  for (int b = 0; b < nbeams; ++b) {
    BeamConfig beam;
    beam.power = pin->GetOrAddReal("laser", BeamKey(b, "power"), 0.0);
    beam.wavelength = pin->GetOrAddReal("laser", BeamKey(b, "wavelength"), 1.0);
    beam.nrays = pin->GetOrAddInteger("laser", BeamKey(b, "nrays"), 1);
    beam.origin[0] = pin->GetOrAddReal("laser", BeamKey(b, "origin_x1"),
                                       ppack->pmesh->mesh_size.x1min);
    beam.origin[1] = pin->GetOrAddReal("laser", BeamKey(b, "origin_x2"), 0.0);
    beam.origin[2] = pin->GetOrAddReal("laser", BeamKey(b, "origin_x3"), 0.0);
    beam.direction[0] = pin->GetOrAddReal("laser", BeamKey(b, "direction_x1"), 1.0);
    beam.direction[1] = pin->GetOrAddReal("laser", BeamKey(b, "direction_x2"), 0.0);
    beam.direction[2] = pin->GetOrAddReal("laser", BeamKey(b, "direction_x3"), 0.0);
    beam.radius = pin->GetOrAddReal("laser", BeamKey(b, "radius"), 0.0);
    if (pin->DoesParameterExist("laser", BeamKey(b, "aperture_radius"))) {
      beam.radius = pin->GetReal("laser", BeamKey(b, "aperture_radius"));
    }
    beam.profile_radius = pin->GetOrAddReal(
        "laser", BeamKey(b, "profile_radius"), beam.radius);
    beam.target_radius = pin->GetOrAddReal(
        "laser", BeamKey(b, "target_radius"), 0.0);
    beam.start_time = pin->GetOrAddReal("laser", BeamKey(b, "start_time"),
                                        -std::numeric_limits<Real>::max());
    beam.end_time = pin->GetOrAddReal("laser", BeamKey(b, "end_time"),
                                      std::numeric_limits<Real>::max());
    beam.zeff = pin->GetOrAddReal("laser", BeamKey(b, "zeff"), 1.0);
    beam.constant_absorption = pin->GetOrAddReal(
        "laser", BeamKey(b, "absorption_coefficient"),
        pin->GetOrAddReal("laser", "absorption_coefficient", 0.0));
    beam.profile = pin->GetOrAddString("laser", BeamKey(b, "profile"), "uniform");
    std::string geometry = pin->GetOrAddString(
        "laser", BeamKey(b, "geometry"), "direction");
    if (geometry == "direction" || geometry == "parallel") {
      beam.geometry = BeamGeometry::direction;
      for (int n = 0; n < 3; ++n) beam.target[n] = beam.origin[n] + beam.direction[n];
    } else if (geometry == "lens" || geometry == "focused") {
      beam.geometry = BeamGeometry::lens;
      for (int n = 0; n < 3; ++n) {
        const std::string axis = "_x" + std::to_string(n+1);
        beam.origin[n] = pin->GetOrAddReal(
            "laser", BeamKey(b, "lens" + axis), beam.origin[n]);
        beam.target[n] = pin->GetOrAddReal(
            "laser", BeamKey(b, "target" + axis),
            beam.origin[n] + beam.direction[n]);
        beam.direction[n] = beam.target[n] - beam.origin[n];
      }
    } else {
      LaserInputError(BeamKey(b, "geometry") +
                      " must be 'direction' or 'lens'");
    }

    beam.pulse_is_absolute = false;
    std::string pulse_mode = pin->GetOrAddString(
        "laser", BeamKey(b, "pulse_mode"), "relative");
    if (pulse_mode == "relative" || pulse_mode == "multiplier") {
      beam.pulse_is_absolute = false;
    } else if (pulse_mode == "absolute" || pulse_mode == "power") {
      beam.pulse_is_absolute = true;
    } else {
      LaserInputError(BeamKey(b, "pulse_mode") +
                      " must be 'relative' or 'absolute'");
    }
    std::string pulse_interpolation = pin->GetOrAddString(
        "laser", BeamKey(b, "pulse_interpolation"), "linear");
    if (pulse_interpolation == "linear") {
      beam.pulse_interpolation = PulseInterpolation::linear;
    } else if (pulse_interpolation == "step") {
      beam.pulse_interpolation = PulseInterpolation::step;
    } else {
      LaserInputError(BeamKey(b, "pulse_interpolation") +
                      " must be 'linear' or 'step'");
    }
    Real pulse_time_scale = pin->GetOrAddReal(
        "laser", BeamKey(b, "pulse_time_scale"), 1.0);
    Real pulse_power_scale = pin->GetOrAddReal(
        "laser", BeamKey(b, "pulse_power_scale"), 1.0);
    if (!Finite(pulse_time_scale) || pulse_time_scale <= 0.0 ||
        !Finite(pulse_power_scale) || pulse_power_scale < 0.0) {
      LaserInputError("pulse time scale must be positive and power scale non-negative");
    }
    std::string pulse_file = pin->GetOrAddString(
        "laser", BeamKey(b, "pulse_file"), "");
    if (!pulse_file.empty()) {
      LoadPulseFile(pulse_file, pulse_time_scale, pulse_power_scale,
                    beam.pulse_time, beam.pulse_value);
    }

    if (!Finite(beam.power) || beam.power < 0.0) {
      LaserInputError(BeamKey(b, "power") + " must be finite and non-negative");
    }
    if (!Finite(beam.wavelength) || beam.wavelength <= 0.0) {
      LaserInputError(BeamKey(b, "wavelength") + " must be finite and positive");
    }
    if (beam.nrays <= 0) {
      LaserInputError(BeamKey(b, "nrays") + " must be positive");
    }
    Real norm = sqrt(SQR(beam.direction[0]) + SQR(beam.direction[1]) +
                     SQR(beam.direction[2]));
    if (!Finite(norm) || norm <= 0.0) {
      LaserInputError(
          "beam direction, or lens-to-target vector, must be finite and nonzero");
    }
    for (int n = 0; n < 3; ++n) beam.direction[n] /= norm;
    if ((!ppack->pmesh->multi_d &&
         (beam.direction[1] != 0.0 || beam.direction[2] != 0.0)) ||
        (ppack->pmesh->two_d && beam.direction[2] != 0.0)) {
      LaserInputError("beam direction cannot point through a collapsed mesh dimension");
    }
    for (int n = 0; n < 3; ++n) {
      if (!Finite(beam.origin[n]) || !Finite(beam.target[n])) {
        LaserInputError("beam lens/origin and target coordinates must be finite");
      }
    }
    if (!Finite(beam.radius) || beam.radius < 0.0 ||
        !Finite(beam.profile_radius) || beam.profile_radius < 0.0 ||
        !Finite(beam.target_radius) || beam.target_radius < 0.0) {
      LaserInputError("beam aperture, profile, and target radii must be finite and "
                      "non-negative");
    }
    if (beam.profile != "uniform" && beam.profile != "gaussian") {
      LaserInputError(BeamKey(b, "profile") + " must be 'uniform' or 'gaussian'");
    }
    if (beam.profile == "gaussian" && beam.radius > 0.0 &&
        !(beam.profile_radius > 0.0)) {
      LaserInputError(BeamKey(b, "profile_radius") +
                      " must be positive for a finite Gaussian aperture");
    }
    if (beam.target_radius > 0.0 && !(beam.radius > 0.0)) {
      LaserInputError(BeamKey(b, "target_radius") +
                      " requires a positive aperture radius");
    }
    if (!Finite(beam.zeff) || beam.zeff <= 0.0) {
      LaserInputError(BeamKey(b, "zeff") + " must be finite and positive");
    }
    if (!Finite(beam.constant_absorption) || beam.constant_absorption < 0.0) {
      LaserInputError(BeamKey(b, "absorption_coefficient") +
                      " must be finite and non-negative");
    }
    if (!(beam.end_time >= beam.start_time)) {
      LaserInputError("beam end_time must not precede start_time");
    }
    if (use_cgs_) {
      beam.power /= power_scale_cgs_;
      beam.wavelength /= length_scale_cgs_;
      if (beam.pulse_is_absolute) {
        for (Real &value : beam.pulse_value) value /= power_scale_cgs_;
      }
    }
    nrays_ += beam.nrays;
    beams_.push_back(beam);
  }

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(cell_data, nmb, ncell_data, ncells3, ncells2, ncells1);
  Kokkos::realloc(cumulative_energy_start_, nmb, 1, ncells3, ncells2, ncells1);
  if (ppack->pmesh->multilevel) {
    int n_ccells1 = indcs.cnx1 + 2*indcs.ng;
    int n_ccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(
        coarse_cell_data, nmb, ncell_data, n_ccells3, n_ccells2, n_ccells1);
  }

  Kokkos::realloc(ray_x, nrays_); Kokkos::realloc(ray_y, nrays_);
  Kokkos::realloc(ray_z, nrays_); Kokkos::realloc(ray_nx, nrays_);
  Kokkos::realloc(ray_ny, nrays_); Kokkos::realloc(ray_nz, nrays_);
  Kokkos::realloc(ray_power, nrays_); Kokkos::realloc(ray_gid, nrays_);
  Kokkos::realloc(ray_i, nrays_); Kokkos::realloc(ray_j, nrays_);
  Kokkos::realloc(ray_k, nrays_); Kokkos::realloc(ray_status, nrays_);
  Kokkos::realloc(ray_x0_, nrays_); Kokkos::realloc(ray_y0_, nrays_);
  Kokkos::realloc(ray_z0_, nrays_); Kokkos::realloc(ray_nx0_, nrays_);
  Kokkos::realloc(ray_ny0_, nrays_); Kokkos::realloc(ray_nz0_, nrays_);
  Kokkos::realloc(ray_power0_, nrays_);
  Kokkos::realloc(ray_power_fraction_, nrays_);
  Kokkos::realloc(ray_wavelength_, nrays_);
  Kokkos::realloc(ray_zeff_, nrays_); Kokkos::realloc(ray_constant_absorption_, nrays_);
  Kokkos::realloc(ray_start_time_, nrays_); Kokkos::realloc(ray_end_time_, nrays_);
  Kokkos::realloc(ray_beam_, nrays_); Kokkos::realloc(beam_power_, nbeams);
  Kokkos::realloc(ray_segments_, nrays_);
  Kokkos::realloc(ray_reflections_, nrays_);
  Kokkos::realloc(ray_reflection_armed_, nrays_);
  Kokkos::realloc(ray_last_turning_density_, nrays_);
  Kokkos::realloc(ray_path_length_, nrays_);
  Kokkos::realloc(ray_kx_, nrays_); Kokkos::realloc(ray_ky_, nrays_);
  Kokkos::realloc(ray_kz_, nrays_);
  Kokkos::realloc(ray_dispersion_error_, nrays_);
  Kokkos::realloc(active_queue_a_, nrays_); Kokkos::realloc(active_queue_b_, nrays_);
  Kokkos::realloc(ray_destination_rank_, nrays_);

  int nranks = global_variable::nranks;
  Kokkos::realloc(mpi_send_counts_, nranks);
  Kokkos::realloc(mpi_send_offsets_, nranks);
  Kokkos::realloc(mpi_pack_cursors_, nranks);
  Kokkos::realloc(mpi_send_packets_, nrays_);
  Kokkos::realloc(mpi_recv_packets_, nrays_);
  Kokkos::realloc(mpi_host_send_packets_, nrays_);
  Kokkos::realloc(mpi_host_recv_packets_, nrays_);
  mpi_send_counts_host_.resize(nranks, 0);
  mpi_recv_counts_host_.resize(nranks, 0);
  mpi_send_offsets_host_.resize(nranks, 0);
  mpi_recv_offsets_host_.resize(nranks, 0);
  if (static_cast<std::size_t>(nrays_) >
      static_cast<std::size_t>(std::numeric_limits<int>::max())/
      sizeof(LaserRayPacket)) {
    LaserInputError("ray packet buffers exceed the MPI byte-count limit");
  }

#if MPI_PARALLEL_ENABLED
  mpi_send_requests_.reset(new MPI_Request[nranks]);
  mpi_recv_requests_.reset(new MPI_Request[nranks]);
  for (int rank = 0; rank < nranks; ++rank) {
    mpi_send_requests_[rank] = MPI_REQUEST_NULL;
    mpi_recv_requests_[rank] = MPI_REQUEST_NULL;
  }
  if (MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_) != MPI_SUCCESS) {
    LaserInputError("could not create the laser MPI communicator");
  }
#endif

  BuildInitialRays();
  Kokkos::deep_copy(cell_data, 0.0);
  if (ppack->pmesh->multilevel) Kokkos::deep_copy(coarse_cell_data, 0.0);
  Kokkos::deep_copy(cumulative_energy_start_, 0.0);
}

//----------------------------------------------------------------------------------------
//! Evaluate one beam's total power. Pulse files are zero outside their tabulated span;
//! start_time/end_time remain an independent gate for backward compatibility.

Real Laser::BeamPowerAtTime(const BeamConfig &beam, Real time) const {
  if (time < beam.start_time || time > beam.end_time) return 0.0;
  if (beam.pulse_time.empty()) return beam.power;
  if (time < beam.pulse_time.front() || time > beam.pulse_time.back()) return 0.0;

  Real pulse = beam.pulse_value.back();
  if (time < beam.pulse_time.back()) {
    auto upper = std::upper_bound(beam.pulse_time.begin(), beam.pulse_time.end(), time);
    std::size_t hi = static_cast<std::size_t>(upper - beam.pulse_time.begin());
    std::size_t lo = hi - 1;
    pulse = beam.pulse_value[lo];
    if (beam.pulse_interpolation == PulseInterpolation::linear) {
      Real fraction = (time-beam.pulse_time[lo])/
                      (beam.pulse_time[hi]-beam.pulse_time[lo]);
      pulse += fraction*(beam.pulse_value[hi]-beam.pulse_value[lo]);
    }
  }
  return beam.pulse_is_absolute ? pulse : beam.power*pulse;
}

//----------------------------------------------------------------------------------------
//! Return the prescribed power averaged over a complete hydro step. All RK stages use
//! this same value, so integrating the source preserves the pulse-file energy exactly.

Real Laser::BeamPowerForStep(const BeamConfig &beam, Real time, Real dt) const {
  if (!(dt > 0.0) || !Finite(dt)) return BeamPowerAtTime(beam, time);
  Real lower = std::max(time, beam.start_time);
  Real upper = std::min(time+dt, beam.end_time);
  if (!(upper > lower)) return 0.0;

  if (beam.pulse_time.empty()) {
    return beam.power*(upper-lower)/dt;
  }
  lower = std::max(lower, beam.pulse_time.front());
  upper = std::min(upper, beam.pulse_time.back());
  if (!(upper > lower)) return 0.0;

  auto next = std::upper_bound(beam.pulse_time.begin(), beam.pulse_time.end(), lower);
  std::size_t segment = static_cast<std::size_t>(next-beam.pulse_time.begin())-1;
  Real integral = 0.0;
  Real left = lower;
  while (left < upper && segment+1 < beam.pulse_time.size()) {
    Real right = std::min(upper, beam.pulse_time[segment+1]);
    Real value_left = beam.pulse_value[segment];
    Real value_right = value_left;
    if (beam.pulse_interpolation == PulseInterpolation::linear) {
      Real inv_width = 1.0/(beam.pulse_time[segment+1]-beam.pulse_time[segment]);
      Real slope = (beam.pulse_value[segment+1]-beam.pulse_value[segment])*inv_width;
      value_left += slope*(left-beam.pulse_time[segment]);
      value_right += slope*(right-beam.pulse_time[segment]);
    }
    integral += 0.5*(value_left+value_right)*(right-left);
    left = right;
    ++segment;
  }
  Real average = integral/dt;
  return beam.pulse_is_absolute ? average : beam.power*average;
}

//----------------------------------------------------------------------------------------
//! Update total beam powers before starting stage-wide laser work. When every beam is
//! dark, leave the device view untouched because no ray kernel will consume it.

bool Laser::UpdateBeamPowers(Real time, Real dt) {
  auto host_beam_power = Kokkos::create_mirror_view(beam_power_);
  bool any_power = false;
  for (std::size_t b = 0; b < beams_.size(); ++b) {
    const Real power = BeamPowerForStep(beams_[b], time, dt);
    host_beam_power(static_cast<int>(b)) = power;
    any_power = any_power || power > 0.0;
  }
  if (any_power) Kokkos::deep_copy(beam_power_, host_beam_power);
  return any_power;
}

void Laser::RefreshGlobalBlockInfo() {
  Mesh *mesh = pmy_pack_->pmesh;
  int nmb_total = mesh->nmb_total;
  // The descriptors are a pure function of the block tree and the rank map, so on a
  // static mesh they only need building once.  AMR is the only thing in this code that
  // remaps gids or ranks after setup, so rebuild unconditionally there and cache
  // otherwise; this was O(nmb_total) of host work plus one H2D on every RK stage.
  if (global_block_info_built_ && !mesh->adaptive &&
      global_block_info_.extent_int(0) == nmb_total) {
    return;
  }
  if (global_block_info_.extent_int(0) != nmb_total) {
    Kokkos::realloc(global_block_info_, nmb_total);
  }
  auto host_blocks = Kokkos::create_mirror_view(global_block_info_);
  const RegionSize &domain = mesh->mesh_size;
  const RegionIndcs &indcs = mesh->mb_indcs;
  for (int gid = 0; gid < nmb_total; ++gid) {
    const LogicalLocation &loc = mesh->lloc_eachmb[gid];
    int level_offset = loc.level - mesh->root_level;
    int blocks_x1 = mesh->nmb_rootx1 << level_offset;
    int blocks_x2 = mesh->nmb_rootx2 << level_offset;
    int blocks_x3 = mesh->nmb_rootx3 << level_offset;
    LaserBlockInfo info;
    // Keep these byte-identical to MeshBlock's canonical bounds. Algebraically
    // equivalent interpolation can differ by a few ulps at a shared rank face.
    info.x1min = (loc.lx1 == 0) ? domain.x1min :
        LeftEdgeX(loc.lx1, blocks_x1, domain.x1min, domain.x1max);
    info.x1max = (loc.lx1 == blocks_x1-1) ? domain.x1max :
        LeftEdgeX(loc.lx1+1, blocks_x1, domain.x1min, domain.x1max);
    info.x2min = domain.x2min;
    info.x2max = domain.x2max;
    info.x3min = domain.x3min;
    info.x3max = domain.x3max;
    if (mesh->multi_d) {
      info.x2min = (loc.lx2 == 0) ? domain.x2min :
          LeftEdgeX(loc.lx2, blocks_x2, domain.x2min, domain.x2max);
      info.x2max = (loc.lx2 == blocks_x2-1) ? domain.x2max :
          LeftEdgeX(loc.lx2+1, blocks_x2, domain.x2min, domain.x2max);
    }
    if (mesh->three_d) {
      info.x3min = (loc.lx3 == 0) ? domain.x3min :
          LeftEdgeX(loc.lx3, blocks_x3, domain.x3min, domain.x3max);
      info.x3max = (loc.lx3 == blocks_x3-1) ? domain.x3max :
          LeftEdgeX(loc.lx3+1, blocks_x3, domain.x3min, domain.x3max);
    }
    info.dx1 = (info.x1max-info.x1min)/static_cast<Real>(indcs.nx1);
    info.dx2 = (info.x2max-info.x2min)/static_cast<Real>(indcs.nx2);
    info.dx3 = (info.x3max-info.x3min)/static_cast<Real>(indcs.nx3);
    info.gid = gid;
    info.rank = mesh->rank_eachmb[gid];
    host_blocks(gid) = info;
  }
  Kokkos::deep_copy(global_block_info_, host_blocks);
  global_block_info_built_ = true;
}

Laser::~Laser() {
#if MPI_PARALLEL_ENABLED
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && mpi_comm_ != MPI_COMM_NULL) MPI_Comm_free(&mpi_comm_);
#endif
}

//----------------------------------------------------------------------------------------
//! Fluid state accessors.  Resolve the configured carrier at call time: the Hydro/MHD
//! constructors reallocate these Views after the Laser object is built.

DvceArray5D<Real> Laser::FluidCons() const {
  return use_mhd_fluid_ ? pmy_pack_->pmhd->u0 : pmy_pack_->phydro->u0;
}

DvceArray5D<Real> Laser::FluidPrim() const {
  return use_mhd_fluid_ ? pmy_pack_->pmhd->w0 : pmy_pack_->phydro->w0;
}

DvceArray5D<Real> Laser::FluidTemperature() const {
  return use_mhd_fluid_ ? pmy_pack_->pmhd->ptwo_temp->temperature
                        : pmy_pack_->phydro->ptwo_temp->temperature;
}

DvceArray5D<Real> Laser::FluidThermodynamics() const {
  return use_mhd_fluid_ ? pmy_pack_->pmhd->ptwo_temp->thermodynamics
                        : pmy_pack_->phydro->ptwo_temp->thermodynamics;
}

} // namespace laser
