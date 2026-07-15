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
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "laser/laser.hpp"
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
  std::exit(EXIT_FAILURE);
}

bool Finite(Real value) {
  return std::isfinite(value);
}

} // namespace

namespace laser {

Laser::Laser(MeshBlockPack *ppack, ParameterInput *pin) :
    cell_data("laser-cell-data", 1, 1, 1, 1, 1),
    ray_x("laser-ray-x", 1), ray_y("laser-ray-y", 1),
    ray_z("laser-ray-z", 1), ray_nx("laser-ray-nx", 1),
    ray_ny("laser-ray-ny", 1), ray_nz("laser-ray-nz", 1),
    ray_power("laser-ray-power", 1), ray_gid("laser-ray-gid", 1),
    ray_i("laser-ray-i", 1), ray_j("laser-ray-j", 1),
    ray_k("laser-ray-k", 1), ray_status("laser-ray-status", 1),
    ray_x0_("laser-ray-x0", 1), ray_y0_("laser-ray-y0", 1),
    ray_z0_("laser-ray-z0", 1), ray_nx0_("laser-ray-nx0", 1),
    ray_ny0_("laser-ray-ny0", 1), ray_nz0_("laser-ray-nz0", 1),
    ray_power0_("laser-ray-power0", 1), ray_wavelength_("laser-ray-lambda", 1),
    ray_zeff_("laser-ray-zeff", 1),
    ray_constant_absorption_("laser-ray-constant-k", 1),
    ray_start_time_("laser-ray-start", 1), ray_end_time_("laser-ray-end", 1),
    ray_beam_("laser-ray-beam", 1), ray_segments_("laser-ray-segments", 1),
    ray_reflections_("laser-ray-reflections", 1),
    ray_path_length_("laser-ray-path", 1),
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
    device_diagnostics_("laser-diagnostics", 4),
    device_counters_("laser-counters", 4),
    cumulative_energy_start_("laser-energy-start", 1, 1, 1, 1, 1),
    pmy_pack_(ppack) {
  if (ppack->pmhd == nullptr || ppack->pmhd->ptwo_temp == nullptr) {
    LaserInputError("requires <mhd>/two_temperature=true");
  }
  if (!ppack->pmhd->peos->eos_data.is_ideal ||
      ppack->pcoord->is_special_relativistic ||
      ppack->pcoord->is_general_relativistic) {
    LaserInputError("currently supports only Newtonian ideal-gas MHD");
  }
  if (ppack->pmesh->multilevel) {
    LaserInputError("initial straight-ray implementation requires a uniform grid");
  }

  electron_index_ = ppack->pmhd->ptwo_temp->iele;
  gamma_minus_one_ = ppack->pmhd->peos->eos_data.gamma - 1.0;
  electron_heat_capacity_fraction_ =
      ppack->pmhd->ptwo_temp->ElectronHeatCapacityFraction();

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
    power_scale_cgs_ = ppack->punit->energy_cgs()/ppack->punit->time_cgs();
  }
  length_scale_cgs_ = pin->GetOrAddReal(
      "laser", "length_scale_cgs", length_scale_cgs_);
  density_scale_cgs_ = pin->GetOrAddReal(
      "laser", "density_scale_cgs", density_scale_cgs_);
  temperature_scale_cgs_ = pin->GetOrAddReal(
      "laser", "temperature_scale_cgs", temperature_scale_cgs_);
  power_scale_cgs_ = pin->GetOrAddReal(
      "laser", "power_scale_cgs", power_scale_cgs_);
  electron_number_per_density_ = pin->GetOrAddReal(
      "laser", "electron_number_per_density", 1.0);
  electron_number_per_gram_ = pin->GetOrAddReal(
      "laser", "electron_number_per_gram", 0.0);
  if (absorption_model_ == AbsorptionModel::inverse_bremsstrahlung &&
      (!(length_scale_cgs_ > 0.0) || !(density_scale_cgs_ > 0.0) ||
       !(temperature_scale_cgs_ > 0.0) || !(electron_number_per_gram_ > 0.0))) {
    LaserInputError("inverse_bremsstrahlung requires positive cgs scales and "
                    "electron_number_per_gram");
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
      absorption_model_ == AbsorptionModel::inverse_bremsstrahlung);
  oblique_turning_ = pin->GetOrAddBoolean(
      "laser", "oblique_turning", true);
  max_reflections_per_ray_ = pin->GetOrAddInteger(
      "laser", "max_reflections_per_ray", 8);
  reflection_offset_fraction_ = pin->GetOrAddReal(
      "laser", "reflection_offset_fraction", 1.0e-10);
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
  if (max_reflections_per_ray_ < 0 ||
      !Finite(reflection_offset_fraction_) || reflection_offset_fraction_ <= 0.0) {
    LaserInputError("reflection limit must be non-negative and offset must be positive");
  }
  if (critical_reflection_ &&
      (!(length_scale_cgs_ > 0.0) || !(density_scale_cgs_ > 0.0) ||
       !(electron_number_per_gram_ > 0.0))) {
    LaserInputError("critical reflection requires positive cgs density/length scales "
                    "and electron_number_per_gram");
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
    beam.start_time = pin->GetOrAddReal("laser", BeamKey(b, "start_time"),
                                        -std::numeric_limits<Real>::max());
    beam.end_time = pin->GetOrAddReal("laser", BeamKey(b, "end_time"),
                                      std::numeric_limits<Real>::max());
    beam.zeff = pin->GetOrAddReal("laser", BeamKey(b, "zeff"), 1.0);
    beam.constant_absorption = pin->GetOrAddReal(
        "laser", BeamKey(b, "absorption_coefficient"),
        pin->GetOrAddReal("laser", "absorption_coefficient", 0.0));
    beam.profile = pin->GetOrAddString("laser", BeamKey(b, "profile"), "uniform");

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
      LaserInputError("beam direction vector must be finite and nonzero");
    }
    for (int n = 0; n < 3; ++n) beam.direction[n] /= norm;
    if ((!ppack->pmesh->multi_d &&
         (beam.direction[1] != 0.0 || beam.direction[2] != 0.0)) ||
        (ppack->pmesh->two_d && beam.direction[2] != 0.0)) {
      LaserInputError("beam direction cannot point through a collapsed mesh dimension");
    }
    if (!Finite(beam.radius) || beam.radius < 0.0) {
      LaserInputError(BeamKey(b, "radius") + " must be finite and non-negative");
    }
    if (beam.profile != "uniform" && beam.profile != "gaussian") {
      LaserInputError(BeamKey(b, "profile") + " must be 'uniform' or 'gaussian'");
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
    }
    nrays_ += beam.nrays;
    beams_.push_back(beam);
  }

  int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(cell_data, nmb, 5, ncells3, ncells2, ncells1);
  Kokkos::realloc(cumulative_energy_start_, nmb, 1, ncells3, ncells2, ncells1);

  Kokkos::realloc(ray_x, nrays_); Kokkos::realloc(ray_y, nrays_);
  Kokkos::realloc(ray_z, nrays_); Kokkos::realloc(ray_nx, nrays_);
  Kokkos::realloc(ray_ny, nrays_); Kokkos::realloc(ray_nz, nrays_);
  Kokkos::realloc(ray_power, nrays_); Kokkos::realloc(ray_gid, nrays_);
  Kokkos::realloc(ray_i, nrays_); Kokkos::realloc(ray_j, nrays_);
  Kokkos::realloc(ray_k, nrays_); Kokkos::realloc(ray_status, nrays_);
  Kokkos::realloc(ray_x0_, nrays_); Kokkos::realloc(ray_y0_, nrays_);
  Kokkos::realloc(ray_z0_, nrays_); Kokkos::realloc(ray_nx0_, nrays_);
  Kokkos::realloc(ray_ny0_, nrays_); Kokkos::realloc(ray_nz0_, nrays_);
  Kokkos::realloc(ray_power0_, nrays_); Kokkos::realloc(ray_wavelength_, nrays_);
  Kokkos::realloc(ray_zeff_, nrays_); Kokkos::realloc(ray_constant_absorption_, nrays_);
  Kokkos::realloc(ray_start_time_, nrays_); Kokkos::realloc(ray_end_time_, nrays_);
  Kokkos::realloc(ray_beam_, nrays_); Kokkos::realloc(ray_segments_, nrays_);
  Kokkos::realloc(ray_reflections_, nrays_);
  Kokkos::realloc(ray_path_length_, nrays_);
  Kokkos::realloc(active_queue_a_, nrays_); Kokkos::realloc(active_queue_b_, nrays_);
  Kokkos::realloc(ray_destination_rank_, nrays_);

  int nranks = global_variable::nranks;
  int nmb_total = ppack->pmesh->nmb_total;
  Kokkos::realloc(global_block_info_, nmb_total);
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

  auto host_blocks = Kokkos::create_mirror_view(global_block_info_);
  const RegionSize &domain = ppack->pmesh->mesh_size;
  for (int gid = 0; gid < nmb_total; ++gid) {
    const LogicalLocation &loc = ppack->pmesh->lloc_eachmb[gid];
    int level_offset = loc.level - ppack->pmesh->root_level;
    int blocks_x1 = ppack->pmesh->nmb_rootx1 << level_offset;
    int blocks_x2 = ppack->pmesh->nmb_rootx2 << level_offset;
    int blocks_x3 = ppack->pmesh->nmb_rootx3 << level_offset;
    LaserBlockInfo info;
    info.x1min = domain.x1min + (domain.x1max-domain.x1min)*loc.lx1/blocks_x1;
    info.x1max = domain.x1min + (domain.x1max-domain.x1min)*(loc.lx1+1)/blocks_x1;
    info.x2min = domain.x2min;
    info.x2max = domain.x2max;
    info.x3min = domain.x3min;
    info.x3max = domain.x3max;
    if (ppack->pmesh->multi_d) {
      info.x2min = domain.x2min + (domain.x2max-domain.x2min)*loc.lx2/blocks_x2;
      info.x2max = domain.x2min +
                   (domain.x2max-domain.x2min)*(loc.lx2+1)/blocks_x2;
    }
    if (ppack->pmesh->three_d) {
      info.x3min = domain.x3min + (domain.x3max-domain.x3min)*loc.lx3/blocks_x3;
      info.x3max = domain.x3min +
                   (domain.x3max-domain.x3min)*(loc.lx3+1)/blocks_x3;
    }
    info.dx1 = (info.x1max-info.x1min)/indcs.nx1;
    info.dx2 = (info.x2max-info.x2min)/indcs.nx2;
    info.dx3 = (info.x3max-info.x3min)/indcs.nx3;
    info.gid = gid;
    info.rank = ppack->pmesh->rank_eachmb[gid];
    host_blocks(gid) = info;
  }
  Kokkos::deep_copy(global_block_info_, host_blocks);

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
  Kokkos::deep_copy(cumulative_energy_start_, 0.0);
}

Laser::~Laser() {
#if MPI_PARALLEL_ENABLED
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && mpi_comm_ != MPI_COMM_NULL) MPI_Comm_free(&mpi_comm_);
#endif
}

} // namespace laser
