//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_mpi.cpp
//! \brief Nonblocking device-buffer ray migration between MPI ranks.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "laser/laser.hpp"
#include "mesh/mesh.hpp"

namespace laser {

//----------------------------------------------------------------------------------------
//! Count off-rank rays by destination and compact them into rank-contiguous packets.

void Laser::PrepareOutgoingRays() {
  int nranks = global_variable::nranks;
  auto status = ray_status;
  auto destination = ray_destination_rank_;
  auto counts = mpi_send_counts_;
  Kokkos::deep_copy(counts, 0);
  Kokkos::parallel_for(
      "laser_count_outgoing", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        if (status(r) == static_cast<int>(RayStatus::off_rank)) {
          int rank = destination(r);
          if (rank >= 0 && rank < nranks) Kokkos::atomic_inc(&counts(rank));
        }
      });

  auto host_counts = Kokkos::create_mirror_view(counts);
  Kokkos::deep_copy(host_counts, counts);
  mpi_send_total_ = 0;
  for (int rank = 0; rank < nranks; ++rank) {
    mpi_send_counts_host_[rank] = host_counts(rank);
    mpi_send_offsets_host_[rank] = mpi_send_total_;
    mpi_send_total_ += host_counts(rank);
  }

  auto host_offsets = Kokkos::create_mirror_view(mpi_send_offsets_);
  for (int rank = 0; rank < nranks; ++rank) {
    host_offsets(rank) = mpi_send_offsets_host_[rank];
  }
  Kokkos::deep_copy(mpi_send_offsets_, host_offsets);
  Kokkos::deep_copy(mpi_pack_cursors_, 0);

  auto offsets = mpi_send_offsets_;
  auto cursors = mpi_pack_cursors_;
  auto packets = mpi_send_packets_;
  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto wave_x = ray_kx_; auto wave_y = ray_ky_; auto wave_z = ray_kz_;
  auto dispersion_error = ray_dispersion_error_;
  auto power = ray_power; auto path = ray_path_length_;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto segments = ray_segments_; auto reflections = ray_reflections_;
  auto reflection_armed = ray_reflection_armed_;
  auto last_turning_density = ray_last_turning_density_;
  auto queue_a = active_queue_a_; auto queue_b = active_queue_b_;
  auto counters = device_counters_;
  Kokkos::parallel_for(
      "laser_pack_outgoing", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        if (status(r) != static_cast<int>(RayStatus::off_rank)) return;
        int rank = destination(r);
        if (rank < 0 || rank >= nranks) {
          status(r) = static_cast<int>(RayStatus::failed);
          Kokkos::atomic_inc(&counters(3));
          return;
        }
        int slot = offsets(rank) + Kokkos::atomic_fetch_add(&cursors(rank), 1);
        LaserRayPacket packet;
        packet.x = x(r); packet.y = y(r); packet.z = z(r);
        packet.nx = nx(r); packet.ny = ny(r); packet.nz = nz(r);
        packet.kx = wave_x(r); packet.ky = wave_y(r); packet.kz = wave_z(r);
        packet.power = power(r); packet.path_length = path(r);
        packet.dispersion_error = dispersion_error(r);
        packet.last_turning_density = last_turning_density(r);
        packet.ray = r; packet.gid = gid(r);
        packet.i = ci(r); packet.j = cj(r); packet.k = ck(r);
        packet.segments = segments(r); packet.reflections = reflections(r);
        packet.reflection_armed = reflection_armed(r);
        packets(slot) = packet;

        // Only the current owner contributes cumulative per-ray diagnostics.
        path(r) = 0.0;
        segments(r) = 0;
        reflections(r) = 0;
        reflection_armed(r) = 0;
        last_turning_density(r) = 0.0;
        destination(r) = -1;
        queue_a(r) = -1;
        queue_b(r) = -1;
        status(r) = static_cast<int>(RayStatus::inactive);
      });
}

//----------------------------------------------------------------------------------------
//! Restore received packets into their globally unique ray slots and activate queue A.

void Laser::UnpackReceivedRays(int count) {
  int first_gid = pmy_pack_->gids;
  int nmb = pmy_pack_->nmb_thispack;
  auto packets = mpi_recv_packets_;
  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto wave_x = ray_kx_; auto wave_y = ray_ky_; auto wave_z = ray_kz_;
  auto dispersion_error = ray_dispersion_error_;
  auto power = ray_power; auto path = ray_path_length_;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status; auto destination = ray_destination_rank_;
  auto segments = ray_segments_; auto reflections = ray_reflections_;
  auto reflection_armed = ray_reflection_armed_;
  auto last_turning_density = ray_last_turning_density_;
  auto queue_a = active_queue_a_; auto queue_b = active_queue_b_;
  auto counters = device_counters_;
  Kokkos::parallel_for(
      "laser_unpack_received", Kokkos::RangePolicy<>(DevExeSpace(), 0, count),
      KOKKOS_LAMBDA(int index) {
        LaserRayPacket packet = packets(index);
        int r = packet.ray;
        int local_block = packet.gid-first_gid;
        if (r < 0 || r >= static_cast<int>(status.extent(0)) ||
            local_block < 0 || local_block >= nmb) {
          Kokkos::atomic_inc(&counters(3));
          return;
        }
        x(r) = packet.x; y(r) = packet.y; z(r) = packet.z;
        nx(r) = packet.nx; ny(r) = packet.ny; nz(r) = packet.nz;
        wave_x(r) = packet.kx; wave_y(r) = packet.ky; wave_z(r) = packet.kz;
        power(r) = packet.power; path(r) = packet.path_length;
        dispersion_error(r) = packet.dispersion_error;
        gid(r) = packet.gid;
        ci(r) = packet.i; cj(r) = packet.j; ck(r) = packet.k;
        segments(r) = packet.segments;
        reflections(r) = packet.reflections;
        reflection_armed(r) = packet.reflection_armed;
        last_turning_density(r) = packet.last_turning_density;
        destination(r) = -1;
        status(r) = static_cast<int>(RayStatus::active);
        queue_a(r) = r;
        queue_b(r) = -1;
      });
}

//----------------------------------------------------------------------------------------
//! Advance one state of distributed ray transport without blocking the task scheduler.

TaskStatus Laser::AdvanceDistributedTransport() {
#if !MPI_PARALLEL_ENABLED
  // Single-process transport uses the same wave semantics as the distributed path:
  // rays that hit the per-wave work cap stay active and are re-traced, so results
  // do not depend on how the work caps interact with the rank decomposition.
  for (int wave = 0; ; ++wave) {
    if (propagation_model_ == PropagationModel::refractive) {
      TraceRefractiveRays(true);
    } else {
      TraceStraightRays(true);
    }
    if (CountActiveRays() == 0) break;
    if (wave+1 >= max_mpi_waves_) {
      BookRemainingRays();
      break;
    }
  }
  FinalizeDiagnostics();
  transport_state_ = LaserTransportState::finished;
  return TaskStatus::complete;
#else
  const int nranks = global_variable::nranks;
  if (nranks == 1) {
    for (int wave = 0; ; ++wave) {
      if (propagation_model_ == PropagationModel::refractive) {
        TraceRefractiveRays(true);
      } else {
        TraceStraightRays(true);
      }
      if (CountActiveRays() == 0) break;
      if (wave+1 >= max_mpi_waves_) {
        BookRemainingRays();
        break;
      }
    }
    FinalizeDiagnostics();
    transport_state_ = LaserTransportState::finished;
    return TaskStatus::complete;
  }

  if (transport_state_ == LaserTransportState::initialize) {
    transport_state_ = LaserTransportState::trace_local;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::trace_local) {
    if (propagation_model_ == PropagationModel::refractive) {
      TraceRefractiveRays(true);
    } else {
      TraceStraightRays(true);
    }
    transport_state_ = LaserTransportState::count_outgoing;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::count_outgoing) {
    PrepareOutgoingRays();
    transport_state_ = LaserTransportState::exchange_counts;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::exchange_counts) {
    int ierr = MPI_Ialltoall(
        mpi_send_counts_host_.data(), 1, MPI_INT,
        mpi_recv_counts_host_.data(), 1, MPI_INT,
        mpi_comm_, &mpi_count_request_);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI count exchange failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    transport_state_ = LaserTransportState::post_receives;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::post_receives) {
    int complete = 0;
    int ierr = MPI_Test(&mpi_count_request_, &complete, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI count poll failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    if (!complete) return TaskStatus::incomplete;

    mpi_recv_total_ = 0;
    for (int rank = 0; rank < nranks; ++rank) {
      mpi_recv_offsets_host_[rank] = mpi_recv_total_;
      mpi_recv_total_ += mpi_recv_counts_host_[rank];
      mpi_send_requests_[rank] = MPI_REQUEST_NULL;
      mpi_recv_requests_[rank] = MPI_REQUEST_NULL;
    }
    if (mpi_recv_total_ > nrays_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser receive queue capacity exceeded" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }

    if (!gpu_aware_mpi_ && mpi_send_total_ > 0) {
      Kokkos::deep_copy(mpi_host_send_packets_, mpi_send_packets_);
    } else {
      Kokkos::fence();
    }

    for (int rank = 0; rank < nranks; ++rank) {
      int recv_bytes = mpi_recv_counts_host_[rank]*
                       static_cast<int>(sizeof(LaserRayPacket));
      if (recv_bytes > 0) {
        LaserRayPacket *recv_base = gpu_aware_mpi_
            ? mpi_recv_packets_.data() : mpi_host_recv_packets_.data();
        ierr = MPI_Irecv(recv_base+mpi_recv_offsets_host_[rank], recv_bytes, MPI_BYTE,
                         rank, 0, mpi_comm_, &mpi_recv_requests_[rank]);
        if (ierr != MPI_SUCCESS) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Laser MPI receive post failed" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
          std::exit(EXIT_FAILURE);
        }
      }

      int send_bytes = mpi_send_counts_host_[rank]*
                       static_cast<int>(sizeof(LaserRayPacket));
      if (send_bytes > 0) {
        LaserRayPacket *send_base = gpu_aware_mpi_
            ? mpi_send_packets_.data() : mpi_host_send_packets_.data();
        ierr = MPI_Isend(send_base+mpi_send_offsets_host_[rank], send_bytes, MPI_BYTE,
                         rank, 0, mpi_comm_, &mpi_send_requests_[rank]);
        if (ierr != MPI_SUCCESS) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Laser MPI send post failed" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    transport_state_ = LaserTransportState::poll_receives;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::poll_receives) {
    int receives_complete = 0;
    int sends_complete = 0;
    int ierr = MPI_Testall(nranks, mpi_recv_requests_.get(),
                           &receives_complete, MPI_STATUSES_IGNORE);
    if (ierr == MPI_SUCCESS) {
      ierr = MPI_Testall(nranks, mpi_send_requests_.get(),
                         &sends_complete, MPI_STATUSES_IGNORE);
    }
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI packet poll failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    if (!receives_complete || !sends_complete) return TaskStatus::incomplete;

    if (!gpu_aware_mpi_ && mpi_recv_total_ > 0) {
      Kokkos::deep_copy(mpi_recv_packets_, mpi_host_recv_packets_);
    } else {
      Kokkos::fence();
    }
    UnpackReceivedRays(mpi_recv_total_);
    // Count all still-active rays (received migrants and rays that hit the local
    // per-wave work cap) so the wave loop keeps running until every ray finishes.
    mpi_local_active_ = CountActiveRays();
    ierr = MPI_Iallreduce(&mpi_local_active_, &mpi_global_active_, 1, MPI_INT,
                          MPI_SUM, mpi_comm_, &mpi_completion_request_);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI completion reduction failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    transport_state_ = LaserTransportState::check_global_completion;
    return TaskStatus::incomplete;
  }

  if (transport_state_ == LaserTransportState::check_global_completion) {
    int complete = 0;
    int ierr = MPI_Test(&mpi_completion_request_, &complete, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI completion poll failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    if (!complete) return TaskStatus::incomplete;

    if (mpi_global_active_ > 0 && ++mpi_wave_ < max_mpi_waves_) {
      transport_state_ = LaserTransportState::trace_local;
      return TaskStatus::incomplete;
    }
    if (mpi_global_active_ > 0) {
      BookRemainingRays();
    }
    transport_state_ = LaserTransportState::finished;
    FinalizeDiagnostics();
    return TaskStatus::complete;
  }

  return (transport_state_ == LaserTransportState::finished)
      ? TaskStatus::complete : TaskStatus::incomplete;
#endif
}

} // namespace laser
