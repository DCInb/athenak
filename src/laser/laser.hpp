#ifndef LASER_LASER_HPP_
#define LASER_LASER_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser.hpp
//! \brief GPU-resident laser ray transport and two-temperature energy deposition.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "tasklist/task_list.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace laser {

enum class DepositionTarget {total, electron};
enum class AbsorptionModel {constant, inverse_bremsstrahlung};
enum class PropagationModel {straight, refractive};
enum class RayStatus : int {inactive = -1, active = 0, escaped = 1, absorbed = 2,
                            off_rank = 3, remaining = 4, failed = 5};

enum class LaserTransportState {
  initialize,
  trace_local,
  count_outgoing,
  exchange_counts,
  post_receives,
  pack_sends,
  send,
  poll_receives,
  unpack,
  check_global_completion,
  finished
};

struct LaserBlockInfo {
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  Real dx1, dx2, dx3;
  int gid, rank;
};

struct LaserRayPacket {
  Real x, y, z;
  Real nx, ny, nz;
  Real kx, ky, kz;
  Real power, path_length;
  Real dispersion_error;
  int ray, gid, i, j, k;
  int segments, reflections;
};

struct BeamConfig {
  Real power;
  Real wavelength;
  Real origin[3];
  Real direction[3];
  Real radius;
  Real start_time;
  Real end_time;
  Real zeff;
  Real constant_absorption;
  int nrays;
  std::string profile;
};

struct LaserDiagnostics {
  Real launched_power = 0.0;
  Real deposited_power = 0.0;
  Real escaped_power = 0.0;
  Real remaining_power = 0.0;
  Real conservation_residual = 0.0;
  int active_rays = 0;
  int reflected_rays = 0;
  int off_rank_transfers = 0;
  int transport_iterations = 0;
  int traced_segments = 0;
  Real total_path_length = 0.0;
  Real max_dispersion_error = 0.0;
};

struct LaserTaskIDs {
  TaskID initialize;
  TaskID trace;
  TaskID apply;
  TaskID clear;
};

//----------------------------------------------------------------------------------------
//! \class Laser
//! \brief Traces ray power through the frozen MHD stage state and heats 2T electrons.

class Laser {
 public:
  static constexpr int ncell_data = 12;

  Laser(MeshBlockPack *ppack, ParameterInput *pin);
  ~Laser();

  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);

  TaskStatus InitializeStep(Driver *pdrive, int stage);
  TaskStatus TraceAndDeposit(Driver *pdrive, int stage);
  TaskStatus ApplySource(Driver *pdrive, int stage);
  TaskStatus ClearBuffers(Driver *pdrive, int stage);

  // 0: power density; 1: cumulative energy density; 2: segment count; 3: tau;
  // 4: path; 5-7: direction*path; 8: dispersion error*path;
  // 9-11: segment-midpoint position*path.
  DvceArray5D<Real> cell_data;
  DvceArray5D<Real> coarse_cell_data;

  // Structure-of-arrays ray state. All tracing kernels operate on these device Views
  // with one thread per active queue entry.
  DvceArray1D<Real> ray_x, ray_y, ray_z;
  DvceArray1D<Real> ray_nx, ray_ny, ray_nz;
  DvceArray1D<Real> ray_power;
  DvceArray1D<int> ray_gid, ray_i, ray_j, ray_k, ray_status;

  int NumberOfRays() const { return nrays_; }
  const LaserDiagnostics &Diagnostics() const { return diagnostics_; }

  // Public because CUDA extended device lambdas cannot be instantiated from private
  // member functions on all supported toolchains.
  void BuildInitialRays();
  void RefreshGlobalBlockInfo();
  void InitializeRays(Real time);
  void TraceStraightRays(bool preserve_off_rank = false);
  void TraceRefractiveRays(bool preserve_off_rank = false);
  void CompactActiveQueue(DvceArray1D<int> current, DvceArray1D<int> next);
  void SeedActiveQueue();
  int CountActiveRays();
  void BookRemainingRays();
  void PrepareOutgoingRays();
  void UnpackReceivedRays(int count);
  TaskStatus AdvanceDistributedTransport();
  void FinalizeDiagnostics();

  LaserTaskIDs id;

 private:
  MeshBlockPack *pmy_pack_;
  std::vector<BeamConfig> beams_;
  DepositionTarget deposition_target_;
  AbsorptionModel absorption_model_;
  PropagationModel propagation_model_ = PropagationModel::straight;

  int nrays_ = 0;
  int electron_index_ = -1;
  int max_segments_per_launch_ = 16;
  int max_transport_iterations_ = 64;
  int max_mpi_waves_ = 1024;
  Real gamma_minus_one_ = 0.0;
  Real electron_heat_capacity_fraction_ = 0.0;
  Real electron_number_per_gram_ = 0.0;
  Real density_scale_cgs_ = 1.0;
  Real temperature_scale_cgs_ = 1.0;
  Real length_scale_cgs_ = 1.0;
  Real power_scale_cgs_ = 1.0;
  Real inverse_bremsstrahlung_coulomb_log_ = -1.0;
  Real minimum_power_fraction_ = 1.0e-14;
  Real conservation_tolerance_ = 1.0e-10;
  bool use_cgs_ = false;
  bool periodic_transport_ = false;
  bool report_diagnostics_ = true;
  bool critical_reflection_ = false;
  bool oblique_turning_ = true;
  bool gpu_aware_mpi_ = false;
  int max_reflections_per_ray_ = 8;
  Real reflection_offset_fraction_ = 1.0e-10;
  Real refractive_cell_fraction_ = 0.25;
  Real refractive_curvature_fraction_ = 0.25;
  Real refractive_tau_max_ = 0.25;
  Real dispersion_tolerance_ = 1.0e-3;

  DvceArray1D<Real> ray_x0_, ray_y0_, ray_z0_;
  DvceArray1D<Real> ray_nx0_, ray_ny0_, ray_nz0_;
  DvceArray1D<Real> ray_power0_, ray_wavelength_;
  DvceArray1D<Real> ray_zeff_, ray_constant_absorption_;
  DvceArray1D<Real> ray_start_time_, ray_end_time_;
  DvceArray1D<int> ray_beam_, ray_segments_, ray_reflections_;
  DvceArray1D<Real> ray_path_length_;
  DvceArray1D<Real> ray_kx_, ray_ky_, ray_kz_, ray_dispersion_error_;
  DvceArray1D<int> active_queue_a_, active_queue_b_;
  DvceArray1D<int> ray_destination_rank_;
  DvceArray1D<LaserBlockInfo> global_block_info_;
  DvceArray1D<int> mpi_send_counts_, mpi_send_offsets_, mpi_pack_cursors_;
  DvceArray1D<LaserRayPacket> mpi_send_packets_, mpi_recv_packets_;
  Kokkos::View<LaserRayPacket *, LayoutWrapper, Kokkos::SharedHostPinnedSpace>
      mpi_host_send_packets_, mpi_host_recv_packets_;
  DvceArray1D<Real> device_diagnostics_;
  DvceArray1D<int> device_counters_;
  DvceArray5D<Real> cumulative_energy_start_;

  LaserDiagnostics diagnostics_;
  LaserTransportState transport_state_ = LaserTransportState::initialize;
  std::vector<int> mpi_send_counts_host_, mpi_recv_counts_host_;
  std::vector<int> mpi_send_offsets_host_, mpi_recv_offsets_host_;
  int mpi_send_total_ = 0;
  int mpi_recv_total_ = 0;
  int mpi_local_active_ = 0;
  int mpi_global_active_ = 0;
  int mpi_wave_ = 0;

#if MPI_PARALLEL_ENABLED
  MPI_Comm mpi_comm_ = MPI_COMM_NULL;
  MPI_Request mpi_count_request_ = MPI_REQUEST_NULL;
  MPI_Request mpi_completion_request_ = MPI_REQUEST_NULL;
  std::unique_ptr<MPI_Request[]> mpi_send_requests_, mpi_recv_requests_;
#endif
};

} // namespace laser

#endif // LASER_LASER_HPP_
