//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_aggregate.cpp
//! \brief Optional global, volume-symmetric AMR aggregate corrections.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"
#include "two_temperature/thermal_radiation_hypre.hpp"
#include "two_temperature/thermal_radiation_cusparse.hpp"
#include "two_temperature/thermal_radiation_dense.hpp"
#include "two_temperature/thermal_radiation_scratch.hpp"

#ifndef ATHENA_CHOLMOD_ENABLED
#define ATHENA_CHOLMOD_ENABLED 0
#endif
#ifndef ATHENA_CUSPARSE_ENABLED
#define ATHENA_CUSPARSE_ENABLED 0
#endif
#ifndef ATHENA_HYPRE_ENABLED
#define ATHENA_HYPRE_ENABLED 0
#endif

#if ATHENA_CHOLMOD_ENABLED
#include <cholmod.h>
#endif

namespace two_temperature {
namespace {
[[noreturn]] void AggregateError(const std::string &message) {
  std::cerr << "### FATAL ERROR in " << __FILE__ << " on rank "
            << global_variable::my_rank << std::endl << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
}  // namespace

#if ATHENA_CHOLMOD_ENABLED
namespace {
// One aggregate-face patch. Its fine faces all connect the same two aggregate
// nodes, including on a coarse/fine interface (the refinement ratio is two).
struct AggregateFacePatch {
  int block, direction;
  int index[3];
  int extent1, extent2;
  double weight;
};

// A fine-side face supplies a coarse row's coupling to this individual fine
// cell. Only these interface values cross ranks; the cell CSR stays distributed.
struct CellAMGFace {
  int block, cell, face;
};

template <typename T>
void UploadAggregateArray(const std::vector<T> &source, DualArray1D<T> &target) {
  Kokkos::realloc(target, source.size());
  for (std::size_t q = 0; q < source.size(); ++q) target.h_view(q) = source[q];
  target.template modify<HostMemSpace>();
  target.template sync<DevExeSpace>();
}
}  // namespace

struct ImplicitAMRAggregateState {
  cholmod_common common;
  cholmod_sparse *matrix = nullptr;
  cholmod_factor *factor = nullptr;
  cholmod_dense *rhs = nullptr;
  cholmod_dense *solution = nullptr;
  cholmod_dense *solve_workspace_y = nullptr;
  cholmod_dense *solve_workspace_e = nullptr;
  std::vector<int> columns, rows;
  // Exact topology data, not a hash: a same-count regrid or rank migration must
  // invalidate both the device scatter and the MPI ordering.
  std::vector<int> topology, counts, offsets, scatter, diagonal_slots;
  std::vector<double> gathered, numerical;
  std::vector<double> scaling;
  DualArray1D<Real> values;
  DualArray1D<AggregateFacePatch> patches;
  DualArray1D<int> slot_offsets, slot_terms, slot_mass_block;
  DvceArray1D<double> patch_values;
  DualArray1D<double> compact_values;
  bool topology_valid = false;
  bool use_device = false;
  bool use_dense = false;
#if ATHENA_HYPRE_ENABLED
  std::unique_ptr<ImplicitAMGHypre> amg;
  std::vector<int> amg_row_offsets, amg_columns;
  DualArray1D<int> amg_compact_slots;
  DvceArray1D<double> amg_numerical, amg_rhs, amg_correction;
  HostArray1D<double> amg_host_numerical;
  DualArray1D<int> cell_row_offsets;
  DualArray1D<unsigned char> cell_entry_faces;
  DualArray1D<CellAMGFace> cell_fine_faces;
  DualArray1D<int> cell_local_slots, cell_remote_slots;
  DualArray1D<double> cell_face_values, cell_remote_values;
  std::vector<int> cell_send_counts, cell_send_offsets;
  std::vector<int> cell_recv_counts, cell_recv_offsets;
  bool cell_volume_weighted = false;
  std::size_t cell_krylov_reserve = 0;
#endif
#if ATHENA_CUSPARSE_ENABLED
  std::unique_ptr<ImplicitAggregateDeviceFactor> device_factor;
  std::unique_ptr<ImplicitAggregateDenseFactor> dense_factor;
#endif

  ImplicitAMRAggregateState() {
    if (!cholmod_start(&common)) AggregateError("Could not initialize CHOLMOD");
    common.print = 0;
    common.final_ll = 1;
    common.useGPU = 0;  // The coarse factor is host-only, including CUDA builds.
  }
  ~ImplicitAMRAggregateState() {
    if (solution != nullptr) cholmod_free_dense(&solution, &common);
    if (solve_workspace_y != nullptr) cholmod_free_dense(&solve_workspace_y, &common);
    if (solve_workspace_e != nullptr) cholmod_free_dense(&solve_workspace_e, &common);
    if (rhs != nullptr) cholmod_free_dense(&rhs, &common);
    if (factor != nullptr) cholmod_free_factor(&factor, &common);
    if (matrix != nullptr) cholmod_free_sparse(&matrix, &common);
    cholmod_finish(&common);
  }
};
#else
struct ImplicitAMRAggregateState {};
#endif

bool ThermalRadiation::HasImplicitAMRAggregateSupport() {
  return ATHENA_CHOLMOD_ENABLED != 0;
}

bool ThermalRadiation::HasImplicitAMRAggregateDeviceSupport() {
  return ATHENA_CHOLMOD_ENABLED != 0 && ATHENA_CUSPARSE_ENABLED != 0;
}

bool ThermalRadiation::HasImplicitAMRAMGSupport() {
  return ATHENA_CHOLMOD_ENABLED != 0 && ATHENA_HYPRE_ENABLED != 0;
}

void ThermalRadiation::DestroyImplicitAMRAggregateState() {
  delete implicit_amr_aggregate_state_;
  implicit_amr_aggregate_state_ = nullptr;
}

// Internal group-worker hook: keep the reusable aggregate topology while
// dropping references to a shared hierarchy that has finished its solves.
void ReleaseImplicitAMRBatch(ImplicitAMRAggregateState *state) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr && state->amg) state->amg->ReleaseBatch();
#endif
}

void TrimImplicitAMRReuseCache(ImplicitAMRAggregateState *state,
                              std::size_t required, std::size_t credit) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr && state->amg) state->amg->TrimReuseCache(required, credit);
#endif
}
void PrepareImplicitAMRReuseGroup(ImplicitAMRAggregateState *state,
                                 int group, int cycle, bool enabled) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr && state->amg) state->amg->SetReuseContext(group, cycle, enabled);
#endif
}
void ObserveImplicitAMRIterations(ImplicitAMRAggregateState *state,
                                 int iterations, bool converged) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr && state->amg) state->amg->ObserveIterations(iterations, converged);
#endif
}
bool ImplicitAMRHierarchyReused(ImplicitAMRAggregateState *state) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  return state != nullptr && state->amg && state->amg->hierarchy_reused();
#else
  return false;
#endif
}

// Sum across the live group workers to count each shared hierarchy at most once.
// A retired lane contributes nothing, so partial cohorts remain conservative.
std::size_t ImplicitAMRReusableBytes(const ImplicitAMRAggregateState *state) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr && state->amg) return state->amg->device_bytes();
#endif
  return 0;
}

std::size_t ImplicitAMRCellWorkspaceBytes(const ImplicitAMRAggregateState *state) {
#if ATHENA_CHOLMOD_ENABLED && ATHENA_HYPRE_ENABLED
  if (state != nullptr) {
    return state->cell_row_offsets.d_view.size()*sizeof(int)+
        state->cell_entry_faces.d_view.size()*sizeof(unsigned char)+
        state->cell_fine_faces.d_view.size()*sizeof(CellAMGFace)+
        (state->cell_local_slots.d_view.size()+state->cell_remote_slots.d_view.size())*
            sizeof(int)+
        (state->cell_face_values.d_view.size()+state->cell_remote_values.d_view.size())*
            sizeof(double)+
        (state->amg_numerical.size()+state->amg_rhs.size()+state->amg_correction.size())*
            sizeof(double)+(state->amg ? state->amg->batch_workspace_bytes() : 0);
  }
#endif
  return 0;
}

//----------------------------------------------------------------------------------------
//! Build E=P^T V A_positive P. Each edge is dt*face_area*frozen_conductance.
//! Only fine faces own unequal-level edges. Interior aggregate faces are retained;
//! interior fine faces cancel exactly. The scalar mass makes E positive definite.
void ThermalRadiation::BuildImplicitAMRAggregatePreconditioner(const Real dt) {
#if ATHENA_CHOLMOD_ENABLED
  RadiationTimerScope build_timer(solver_timers_, RadiationTimer::AggregateBuild);
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
#if ATHENA_HYPRE_ENABLED
  if (implicit_preconditioner_mode_ == 4) {
    // Retain every cell: fixed cubes can straddle insulating interfaces and
    // omit the low-energy modes of a high-contrast diffusion operator.
    const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int blocks = pmy_pack_->nmb_thispack, gid0 = pmy_pack_->gids;
    auto *mesh = pmy_pack_->pmesh;
    const int total_blocks = mesh->nmb_total;
    const int faces = nz > 1 ? 6 : (ny > 1 ? 4 : 2);
    const std::size_t cells = static_cast<std::size_t>(nx)*ny*nz;
    if (cells*total_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cells*blocks*(1+faces*(1 << (faces/2-1))) >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      AggregateError("cell-amg graph exceeds the integer index limit");
    }
    const int per_block = static_cast<int>(cells), rows = blocks*per_block;
    bool mixed_levels = false;
    for (int gid = 1; gid < total_blocks; ++gid) {
      mixed_levels |= mesh->lloc_eachmb[gid].level != mesh->lloc_eachmb[0].level;
    }
    if (gid0 != mesh->gids_eachrank[global_variable::my_rank] ||
        blocks != mesh->nmb_eachrank[global_variable::my_rank]) {
      AggregateError("cell-amg requires one complete MeshBlockPack per rank");
    }
    auto *mb = pmy_pack_->pmb;
    mb->nghbr.sync_host();
    const auto neighbors = mb->nghbr.h_view;
    std::vector<std::array<int, 4>> peers(blocks*faces, {-1, -1, -1, -1});
    std::vector<int> topology{nx, ny, nz, gid0, blocks, total_blocks};
    for (int gid = 0; gid < total_blocks; ++gid) {
      const auto &loc = mesh->lloc_eachmb[gid];
      topology.insert(topology.end(), {loc.lx1, loc.lx2, loc.lx3, loc.level,
                                       mesh->rank_eachmb[gid]});
    }
    for (int m = 0; m < blocks; ++m) {
      for (int face = 0; face < faces; ++face) {
        const int begin = (face/2 == 2 ? 24 : 8*(face/2))+4*(face%2);
        for (int q = begin; q < begin+4; ++q) {
          peers[m*faces+face][q-begin] = neighbors(m, q).gid;
        }
      }
    }
    for (const auto &face : peers) topology.insert(topology.end(), face.begin(), face.end());
    for (int face = 0; face < faces; ++face) topology.push_back(implicit_boundary_type_[face]);
    if (implicit_amr_aggregate_state_ == nullptr) {
      implicit_amr_aggregate_state_ = new ImplicitAMRAggregateState;
    }
    auto &state = *implicit_amr_aggregate_state_;
    const bool rebuild = !state.topology_valid || state.topology != topology;
    if (rebuild) {
      RadiationTimerScope graph_timer(solver_timers_, RadiationTimer::AggregateGraph, false);
      // Release the old hierarchy before allocating replacement graph storage.
      state.amg.reset();
      state.amg_row_offsets.clear();
      state.amg_columns.clear();
      state.amg_row_offsets.reserve(rows+1);
      state.amg_columns.reserve(static_cast<std::size_t>(rows)*(faces+1));
      std::vector<unsigned char> tags;
      tags.reserve(static_cast<std::size_t>(rows)*(faces+1));
      const int ranks = global_variable::nranks, my_rank = global_variable::my_rank;
      std::vector<std::vector<int>> requests(ranks), request_slots(ranks);
      std::vector<CellAMGFace> fine_faces;
      std::vector<int> local_slots;
      const int dims[3] = {nx, ny, nz};
      const int dimensions = faces/2;
      struct Entry { int column; unsigned char tag; int fine_face; };
      for (int m = 0; m < blocks; ++m) {
        const auto &location = mesh->lloc_eachmb[gid0+m];
        const int logical[3] = {location.lx1, location.lx2, location.lx3};
        for (int k = 0; k < nz; ++k) {
          for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
              state.amg_row_offsets.push_back(state.amg_columns.size());
              // A coarse face touches up to four fine cells in 3D.
              Entry entries[25];
              int count = 1;
              entries[0] = {(gid0+m)*per_block+(k*ny+j)*nx+i, 64, -1};
              for (int face = 0; face < faces; ++face) {
                int index[3] = {i, j, k};
                const int dir = face/2;
                index[dir] += face%2 == 0 ? -1 : 1;
                int peer = gid0+m;
                if (index[dir] < 0 || index[dir] == dims[dir]) {
                  peer = -1;
                  for (int candidate : peers[m*faces+face]) {
                    if (candidate < 0) continue;
                    const auto &other = mesh->lloc_eachmb[candidate];
                    if (other.level <= location.level) { peer = candidate; break; }
                    const int child[3] = {other.lx1&1, other.lx2&1, other.lx3&1};
                    bool covers = true;
                    for (int axis = 0; axis < dimensions; ++axis) {
                      if (axis != dir && child[axis] != 2*index[axis]/dims[axis]) covers = false;
                    }
                    if (covers) { peer = candidate; break; }
                  }
                  index[dir] = face%2 == 0 ? dims[dir]-1 : 0;
                  if (peer >= 0) {
                    const auto &other = mesh->lloc_eachmb[peer];
                    if (other.level > location.level) {
                      const int child[3] = {other.lx1&1, other.lx2&1, other.lx3&1};
                      for (int subface = 0; subface < (1 << (dimensions-1)); ++subface) {
                        int fine[3] = {index[0], index[1], index[2]}, bit = 0;
                        for (int axis = 0; axis < dimensions; ++axis) {
                          if (axis != dir) {
                            fine[axis] = 2*index[axis]-child[axis]*dims[axis]+
                                ((subface >> bit++) & 1);
                          }
                        }
                        entries[count++] = {peer*per_block+(fine[2]*ny+fine[1])*nx+fine[0],
                                             0, face^1};
                      }
                      continue;
                    }
                    if (other.level < location.level) {
                      for (int axis = 0; axis < dimensions; ++axis) {
                        if (axis != dir) {
                          index[axis] = ((logical[axis]&1)*dims[axis]+index[axis])/2;
                        }
                      }
                    }
                  }
                }
                if (peer < 0) continue;
                entries[count++] = {peer*per_block+(index[2]*ny+index[1])*nx+index[0],
                                     static_cast<unsigned char>(1 << face), -1};
              }
              std::sort(entries, entries+count,
                        [](const Entry &a, const Entry &b) { return a.column < b.column; });
              for (int q = 0; q < count; ++q) {
                if (q > 0 && entries[q].column == entries[q-1].column) {
                  tags.back() |= entries[q].tag;
                } else {
                  state.amg_columns.push_back(entries[q].column);
                  tags.push_back(entries[q].tag);
                }
                if (entries[q].fine_face >= 0) {
                  const int cell = entries[q].column, gid = cell/per_block;
                  const int owner = mesh->rank_eachmb[gid];
                  const int slot = static_cast<int>(tags.size())-1;
                  if (owner == my_rank) {
                    fine_faces.push_back({gid-gid0, cell%per_block, entries[q].fine_face});
                    local_slots.push_back(slot);
                  } else {
                    requests[owner].insert(requests[owner].end(), {cell, entries[q].fine_face});
                    request_slots[owner].push_back(slot);
                  }
                }
              }
            }
          }
        }
      }
      state.amg_row_offsets.push_back(state.amg_columns.size());
      if (state.amg_columns.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        AggregateError("Cell AMG CSR exceeds integer indexing");
      }
      // Coarse owners request fine-face values once per topology. Subsequent
      // numerical setups exchange only those doubles, in the cached ordering.
      std::vector<int> outgoing, incoming, remote_slots;
      auto &recv_counts = state.cell_recv_counts;
      auto &recv_offsets = state.cell_recv_offsets;
      auto &send_counts = state.cell_send_counts;
      auto &send_offsets = state.cell_send_offsets;
      recv_counts.assign(ranks, 0); recv_offsets.assign(ranks, 0);
      send_counts.assign(ranks, 0); send_offsets.assign(ranks, 0);
      for (int rank = 0; rank < ranks; ++rank) {
        recv_counts[rank] = request_slots[rank].size();
        recv_offsets[rank] = remote_slots.size();
        outgoing.insert(outgoing.end(), requests[rank].begin(), requests[rank].end());
        remote_slots.insert(remote_slots.end(), request_slots[rank].begin(), request_slots[rank].end());
      }
#if MPI_PARALLEL_ENABLED
      if (mixed_levels && ranks > 1) {
        MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT,
                     MPI_COMM_WORLD);
        int received = 0;
        std::vector<int> sc(ranks), sd(ranks), rc(ranks), rd(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
          send_offsets[rank] = received;
          received += send_counts[rank];
          sc[rank] = 2*recv_counts[rank]; sd[rank] = 2*recv_offsets[rank];
          rc[rank] = 2*send_counts[rank]; rd[rank] = 2*send_offsets[rank];
        }
        incoming.resize(2*received);
        MPI_Alltoallv(outgoing.data(), sc.data(), sd.data(), MPI_INT,
                      incoming.data(), rc.data(), rd.data(), MPI_INT, MPI_COMM_WORLD);
      }
#endif
      for (std::size_t q = 0; q < incoming.size(); q += 2) {
        const int block = incoming[q]/per_block-gid0;
        if (block < 0 || block >= blocks) AggregateError("Cell AMG face request has wrong owner");
        fine_faces.push_back({block, incoming[q]%per_block, incoming[q+1]});
      }
      UploadAggregateArray(fine_faces, state.cell_fine_faces);
      UploadAggregateArray(local_slots, state.cell_local_slots);
      UploadAggregateArray(remote_slots, state.cell_remote_slots);
      Kokkos::realloc(state.cell_face_values, fine_faces.size());
      Kokkos::realloc(state.cell_remote_values, remote_slots.size());
      UploadAggregateArray(state.amg_row_offsets, state.cell_row_offsets);
      UploadAggregateArray(tags, state.cell_entry_faces);
      Kokkos::realloc(state.amg_numerical, tags.size());
      Kokkos::realloc(state.amg_rhs, rows);
      Kokkos::realloc(state.amg_correction, rows);
      state.amg.reset(new ImplicitAMGHypre);
      state.amg->ConfigureReuse(implicit_amg_reuse_ ? implicit_amg_reuse_steps_ : 0,
                               implicit_amg_reuse_slots_, implicit_amg_reuse_drift_);
      state.topology = std::move(topology);
      state.topology_valid = true;
      state.cell_volume_weighted = mixed_levels;
    }
    auto offsets = state.cell_row_offsets.d_view;
    auto tags = state.cell_entry_faces.d_view;
    auto numerical = state.amg_numerical;
    auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
    auto f1 = ImplicitAMRFaceBand(implicit_amr_conductance_.x1f);
    auto f2 = ImplicitAMRFaceBand(implicit_amr_conductance_.x2f);
    auto f3 = ImplicitAMRFaceBand(implicit_amr_conductance_.x3f);
    auto size = mb->mb_size.d_view;
    par_for("thermal_rad_cell_amg_matrix", DevExeSpace(), 0, rows-1,
    KOKKOS_LAMBDA(const int row) {
      const int m = row/per_block, a = row%per_block;
      const int i = is+a%nx, j = js+(a/nx)%ny, k = ks+a/(nx*ny);
      for (int q = offsets(row); q < offsets(row+1); ++q) {
        const auto tag = tags(q);
        double value = (tag & 64) ? diagonal(m, 0, k, j, i) : 0.0;
        if (tag & 1) value -= dt*f1(m, 0, k, j, i)/size(m).dx1;
        if (tag & 2) value -= dt*f1(m, 0, k, j, i+1)/size(m).dx1;
        if (tag & 4) value -= dt*f2(m, 0, k, j, i)/size(m).dx2;
        if (tag & 8) value -= dt*f2(m, 0, k, j+1, i)/size(m).dx2;
        if (tag & 16) value -= dt*f3(m, 0, k, j, i)/size(m).dx3;
        if (tag & 32) value -= dt*f3(m, 0, k+1, j, i)/size(m).dx3;
        const double volume = mixed_levels ?
            size(m).dx1*size(m).dx2*size(m).dx3 : 1.0;
        numerical(q) = value*volume;
      }
    });
    if (mixed_levels) {
      auto fine_faces = state.cell_fine_faces.d_view;
      auto values = state.cell_face_values.d_view;
      par_for("thermal_rad_cell_amg_fine_faces", DevExeSpace(), 0, fine_faces.extent_int(0)-1,
      KOKKOS_LAMBDA(const int q) {
        const auto face = fine_faces(q);
        const int m = face.block, a = face.cell, dir = face.face/2;
        const int i = is+a%nx+(dir == 0 ? face.face%2 : 0);
        const int j = js+(a/nx)%ny+(dir == 1 ? face.face%2 : 0);
        const int k = ks+a/(nx*ny)+(dir == 2 ? face.face%2 : 0);
        const double c = dir == 0 ? f1(m, 0, k, j, i) :
            (dir == 1 ? f2(m, 0, k, j, i) : f3(m, 0, k, j, i));
        const double dx = dir == 0 ? size(m).dx1 : (dir == 1 ? size(m).dx2 : size(m).dx3);
        values(q) = dt*c/dx*(size(m).dx1*size(m).dx2*size(m).dx3);
      });
      auto slots = state.cell_local_slots.d_view;
      const int local_count = slots.extent_int(0);
      par_for("thermal_rad_cell_amg_local_interfaces", DevExeSpace(), 0, local_count-1,
      KOKKOS_LAMBDA(const int q) { Kokkos::atomic_add(&numerical(slots(q)), -values(q)); });
#if MPI_PARALLEL_ENABLED
      if (global_variable::nranks > 1) {
        state.cell_face_values.template modify<DevExeSpace>();
        state.cell_face_values.template sync<HostMemSpace>();
        auto send = state.cell_face_values.h_view;
        auto recv = state.cell_remote_values.h_view;
        {
          RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI, false);
          MPI_Alltoallv(send.data() ? send.data()+local_count : nullptr,
                        state.cell_send_counts.data(), state.cell_send_offsets.data(), MPI_DOUBLE,
                        recv.data(), state.cell_recv_counts.data(), state.cell_recv_offsets.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);
        }
        state.cell_remote_values.template modify<HostMemSpace>();
        state.cell_remote_values.template sync<DevExeSpace>();
        auto remote = state.cell_remote_values.d_view;
        auto remote_slots = state.cell_remote_slots.d_view;
        par_for("thermal_rad_cell_amg_remote_interfaces", DevExeSpace(),
                0, remote_slots.extent_int(0)-1,
        KOKKOS_LAMBDA(const int q) {
          Kokkos::atomic_add(&numerical(remote_slots(q)), -remote(q));
        });
      }
#endif
    }
    const std::size_t restart = std::min(implicit_gmres_restart_, implicit_max_iterations_);
    const std::size_t basis_bytes = implicit_amr_high_order_ ?
        (restart+1)*static_cast<std::size_t>(rows)*sizeof(Real) : 0;
    const std::size_t basis_owned = implicit_gmres_basis_.size()*sizeof(Real);
    std::size_t krylov_growth = basis_bytes > basis_owned ? basis_bytes-basis_owned : 0;
    if (implicit_amr_high_order_ && implicit_amr_inner_iterations_ > 0) {
      const std::size_t bytes = restart*static_cast<std::size_t>(rows)*sizeof(Real);
      const std::size_t owned = implicit_gmres_corrections_.size()*sizeof(Real);
      if (bytes > owned) krylov_growth += bytes-owned;
    }
    int configure = rebuild || krylov_growth != state.cell_krylov_reserve;
#if MPI_PARALLEL_ENABLED
    if (mixed_levels && global_variable::nranks > 1) {
      RadiationBatchAllreduce(MPI_IN_PLACE, &configure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif
    {
      RadiationTimerScope setup_timer(solver_timers_, RadiationTimer::AggregateDeviceFactor);
      // Every rank's topology signature contains the complete ownership map.
      // Configure the graph collectively only when that signature changes.
      // Revalidating millions of unchanged CSR rows for every spectral group
      // is host work unrelated to the changing numerical coefficients. Setup
      // still checks the current free-memory budget on every call.
      state.amg->SetReuseContext(implicit_group_begin_, mesh->ncycle,
          implicit_amg_reuse_ && RadiationGroupBatch::Active() != nullptr &&
          implicit_group_end_ == implicit_group_begin_+1);
      if ((configure && !state.amg->ConfigureGraph(
              state.amg_row_offsets, state.amg_columns,
              gid0*per_block, total_blocks*per_block, implicit_amr_amg_max_bytes_, krylov_growth)) ||
          !state.amg->Setup(state.amg_numerical.data())) {
        AggregateError("Cell AMG setup failed: "+state.amg->error());
      }
      state.cell_krylov_reserve = krylov_growth;
    }
    if (solver_timers_.enabled) {
      std::cout << "# radiation cell AMG: rank=" << global_variable::my_rank
                << " rows=" << rows << " entries=" << state.amg_columns.size()
                << " topology_rebuilt=" << rebuild
                << " volume_weighted=" << mixed_levels
                << " local_interface_faces=" << state.cell_local_slots.extent(0)
                << " remote_interface_faces=" << state.cell_remote_slots.extent(0)
                << " krylov_growth_bytes=" << krylov_growth
                << " device_bytes=" << state.amg->device_bytes() << std::endl;
    }
    return;
  }
#endif
  const int cells[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  const int start[3] = {indcs.is, indcs.js, indcs.ks};
  const bool use_amg = implicit_preconditioner_mode_ == 3;
  const int aggregate_width = use_amg ? 3 : 9;
  const int width[3] = {aggregate_width, indcs.nx2 == 1 ? 1 : aggregate_width,
                       indcs.nx3 == 1 ? 1 : aggregate_width};
  const int count[3] = {cells[0]/width[0], cells[1]/width[1], cells[2]/width[2]};
  const int per_block = count[0]*count[1]*count[2];
  const int gid0 = pmy_pack_->gids;
  const int blocks = pmy_pack_->nmb_thispack;
  const int total_blocks = pmy_pack_->pmesh->nmb_total;
  if (total_blocks > std::numeric_limits<int>::max()/per_block) {
    AggregateError("Global AMR aggregate graph exceeds the integer index limit");
  }
  const int n = total_blocks*per_block;
  if (use_amg && (blocks <= 0 ||
      gid0 != pmy_pack_->pmesh->gids_eachrank[global_variable::my_rank] ||
      blocks != pmy_pack_->pmesh->nmb_eachrank[global_variable::my_rank])) {
    AggregateError("amr-amg requires one nonempty MeshBlockPack per MPI rank");
  }
  const int first_row = gid0*per_block;
  const int local_rows = blocks*per_block;
  const bool distributed_amg = use_amg && global_variable::nranks > 1;
  const int dimensions = pmy_pack_->pmesh->three_d ? 3 :
      (pmy_pack_->pmesh->multi_d ? 2 : 1);
  auto *mb = pmy_pack_->pmb;
  mb->mb_size.sync_host();
  mb->mb_lev.sync_host();
  mb->mb_bcs.sync_host();
  mb->nghbr.sync_host();
  const auto levels = mb->mb_lev.h_view;
  const auto boundaries = mb->mb_bcs.h_view;
  const auto neighbors = mb->nghbr.h_view;
  if (implicit_amr_aggregate_state_ == nullptr) {
    implicit_amr_aggregate_state_ = new ImplicitAMRAggregateState;
  }
  auto &state = *implicit_amr_aggregate_state_;
  const char *dense_fallback = "none";
  std::size_t dense_free_bytes = 0, dense_additional_bytes = 0, krylov_growth_bytes = 0;
  bool dense_memory_checked = false;
  state.use_device = !use_amg && HasImplicitAMRAggregateDeviceSupport() &&
      implicit_amr_coarse_solver_ != 1;
  state.use_dense = false;
#if ATHENA_CUSPARSE_ENABLED
  // The smallest possible factor has n diagonal entries. Avoid even exporting
  // a dense-ineligible auto factor, and release a previous dense allocation
  // before building the new, potentially larger graph.
  if (implicit_amr_coarse_solver_ == 0 &&
      !ImplicitAggregateDenseFactor::SupportsDense(n, n)) {
    state.use_device = false;
    dense_fallback = "storage_budget";
  }
  if (!state.use_device) {
    state.dense_factor.reset();
    state.device_factor.reset();
  }
#endif

  std::vector<int> topology = {aggregate_width, gid0, blocks, total_blocks, dimensions,
      global_variable::nranks, cells[0], cells[1], cells[2],
      start[0], start[1], start[2], neighbors.extent_int(1)};
  for (int face = 0; face < 6; ++face) topology.push_back(implicit_boundary_type_[face]);
  for (int gid = 0; gid < total_blocks; ++gid) {
    const auto &location = pmy_pack_->pmesh->lloc_eachmb[gid];
    topology.insert(topology.end(), {location.lx1, location.lx2, location.lx3,
        location.level, pmy_pack_->pmesh->rank_eachmb[gid]});
  }
  for (int m = 0; m < blocks; ++m) {
    topology.push_back(levels(m));
    for (int face = 0; face < 6; ++face) {
      topology.push_back(static_cast<int>(boundaries(m, face)));
    }
    for (int q = 0; q < neighbors.extent_int(1); ++q) {
      const auto &neighbor = neighbors(m, q);
      topology.push_back(neighbor.gid);
      // The remaining fields of an absent neighbor need not be initialized.
      if (neighbor.gid >= 0) {
        topology.insert(topology.end(), {neighbor.lev, neighbor.rank, neighbor.dest});
      }
    }
  }
  int rebuild = !state.topology_valid || topology != state.topology;
#if MPI_PARALLEL_ENABLED
  {
    RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI, false);
    RadiationBatchAllreduce(MPI_IN_PLACE, &rebuild, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  bool new_pattern = false;
  if (rebuild) {
    RadiationTimerScope topology_timer(solver_timers_, RadiationTimer::AggregateTopology);
#if ATHENA_HYPRE_ENABLED
    // Release the old hierarchy before a regrid allocates replacement graph
    // storage; retaining both hierarchies can exceed the GPU memory allowance.
    if (use_amg) state.amg.reset();
#endif
    struct LocalEntry {
      std::vector<int> terms;
      int mass_block = -1;
    };
    using Key = std::pair<int, int>;
    // Sorted lower-triangle CSC keys preserve canonical rank-independent ordering.
    std::map<Key, LocalEntry> local;
    std::vector<AggregateFacePatch> patches;
    std::vector<int> validation_terms;
    const auto node = [&](int gid, const int *index) {
      return gid*per_block+((index[2]/width[2])*count[1]+index[1]/width[1])*count[0]+
          index[0]/width[0];
    };
    const auto append_patch = [&](const AggregateFacePatch &patch) {
      if (patches.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()-1)) {
        AggregateError("Too many aggregate face patches");
      }
      patches.push_back(patch);
      return static_cast<int>(patches.size());  // Signed, one-based term identifier.
    };
    const auto edge = [&](int left, int right, int term) {
      if (left == right) {
        validation_terms.push_back(term);
        return;
      }
      local[{left, left}].terms.push_back(term);
      local[{right, right}].terms.push_back(term);
      local[{std::min(left, right), std::max(left, right)}].terms.push_back(-term);
    };
    for (int m = 0; m < blocks; ++m) {
      const int gid = gid0+m;
      const auto &location = pmy_pack_->pmesh->lloc_eachmb[gid];
      const int logical[3] = {location.lx1, location.lx2, location.lx3};
      for (int q = 0; q < per_block; ++q) {
        local[{gid*per_block+q, gid*per_block+q}].mass_block = m;
      }
      for (int dir = 0; dir < dimensions; ++dir) {
        const int tangent1 = (dir+1)%3, tangent2 = (dir+2)%3;
        for (int normal = 0; normal <= cells[dir]; normal += width[dir]) {
          const bool lower = normal == 0, upper = normal == cells[dir];
          const int face = 2*dir+(upper ? 1 : 0);
          int peer = -1, peer_level = levels(m);
          bool physical = false;
          if (lower || upper) {
            const int begin = (dir == 2 ? 24 : 8*dir)+(upper ? 4 : 0);
            for (int q = begin; q < begin+4; ++q) {
              if (neighbors(m, q).gid >= 0 && neighbors(m, q).lev <= levels(m)) {
                peer = neighbors(m, q).gid;
                peer_level = neighbors(m, q).lev;
              }
            }
            const auto boundary = boundaries(m, face);
            physical = boundary != BoundaryFlag::block &&
                boundary != BoundaryFlag::periodic &&
                boundary != BoundaryFlag::shear_periodic && boundary != BoundaryFlag::undef;
            if (peer < 0 && !physical) continue;  // Fine neighbors own these edges.
            if (peer >= 0 && peer_level == levels(m) &&
                (gid > peer || (gid == peer && upper))) continue;
          }
          // Aggregate widths divide the block. Under factor-two refinement the
          // coarse-node transitions also fall on these patch boundaries, even
          // when a block has an odd number of aggregates in a tangent direction.
          for (int b = 0; b < cells[tangent2]; b += width[tangent2]) {
            for (int a = 0; a < cells[tangent1]; a += width[tangent1]) {
              AggregateFacePatch patch;
              patch.block = m;
              patch.direction = dir;
              patch.index[0] = patch.index[1] = patch.index[2] = 0;
              patch.index[dir] = normal;
              patch.index[tangent1] = a;
              patch.index[tangent2] = b;
              patch.extent1 = width[tangent1];
              patch.extent2 = width[tangent2];
              patch.weight = 1.0;
              int index[3] = {patch.index[0], patch.index[1], patch.index[2]};
              if (!(lower || upper)) {
                const int right = node(gid, index);
                --index[dir];
                edge(node(gid, index), right, append_patch(patch));
                continue;
              }
              index[dir] = lower ? 0 : cells[dir]-1;
              const int here = node(gid, index);
              if (peer >= 0) {
                index[dir] = lower ? cells[dir]-1 : 0;
                if (peer_level < levels(m)) {
                  for (int axis = 0; axis < dimensions; ++axis) {
                    if (axis != dir) {
                      index[axis] = ((logical[axis]&1)*cells[axis]+index[axis])/2;
                    }
                  }
                }
                edge(here, node(peer, index), append_patch(patch));
              } else {
                const int type = implicit_boundary_type_[face];
                patch.weight = type == 0 ? 0.0 : (type == 1 ? 2.0 : 1.0);
                local[{here, here}].terms.push_back(append_patch(patch));
              }
            }
          }
        }
      }
    }
    if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()/2)) {
      AggregateError("Local sparse coarse matrix exceeds the MPI count limit");
    }
    const int entries = static_cast<int>(local.size());
    std::vector<int> indices(2*entries), offsets(1, 0), terms, mass_blocks;
    int entry = 0;
    for (const auto &item : local) {
      indices[2*entry] = item.first.first;
      indices[2*entry+1] = item.first.second;
      ++entry;
      terms.insert(terms.end(), item.second.terms.begin(), item.second.terms.end());
      if (terms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        AggregateError("Aggregate slot map exceeds integer indexing");
      }
      offsets.push_back(static_cast<int>(terms.size()));
      mass_blocks.push_back(item.second.mass_block);
    }
    // This extra slot validates discarded periodic self-edges as well. It is
    // checked after download but never enters the matrix or MPI assembly.
    terms.insert(terms.end(), validation_terms.begin(), validation_terms.end());
    if (terms.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      AggregateError("Aggregate validation map exceeds integer indexing");
    }
    offsets.push_back(static_cast<int>(terms.size()));
    mass_blocks.push_back(-1);
    UploadAggregateArray(patches, state.patches);
    UploadAggregateArray(offsets, state.slot_offsets);
    UploadAggregateArray(terms, state.slot_terms);
    UploadAggregateArray(mass_blocks, state.slot_mass_block);
    Kokkos::realloc(state.patch_values, patches.size());
    Kokkos::realloc(state.compact_values, entries+1);

    std::vector<int> all_indices;
#if MPI_PARALLEL_ENABLED
    const int ranks = global_variable::nranks;
    state.counts.resize(ranks);
    state.offsets.resize(ranks);
    std::vector<int> index_counts(ranks), index_offsets(ranks);
    {
      RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI, false);
      MPI_Allgather(&entries, 1, MPI_INT, state.counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    }
    int total = 0;
    for (int rank = 0; rank < ranks; ++rank) {
      if (state.counts[rank] > std::numeric_limits<int>::max()/2-total) {
        AggregateError("Global sparse coarse matrix exceeds the MPI count limit");
      }
      state.offsets[rank] = total;
      index_offsets[rank] = 2*total;
      index_counts[rank] = 2*state.counts[rank];
      total += state.counts[rank];
    }
    all_indices.resize(2*total);
    {
      RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI, false);
      MPI_Allgatherv(indices.data(), 2*entries, MPI_INT, all_indices.data(),
                    index_counts.data(), index_offsets.data(), MPI_INT, MPI_COMM_WORLD);
    }
#else
    all_indices = std::move(indices);
#endif
    std::map<Key, int> global_slots;
    for (std::size_t q = 0; q < all_indices.size()/2; ++q) {
      global_slots[{all_indices[2*q], all_indices[2*q+1]}] = 0;
    }
    if (global_slots.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      AggregateError("Sparse coarse matrix exceeds CHOLMOD integer indexing");
    }
    std::vector<int> columns(n+1, 0), rows;
    rows.reserve(global_slots.size());
    state.diagonal_slots.assign(n, -1);
    int slot = 0;
    for (auto &item : global_slots) {
      item.second = slot;
      const int column = item.first.first, row = item.first.second;
      ++columns[column+1];
      rows.push_back(row);
      if (column == row) state.diagonal_slots[row] = slot;
      ++slot;
    }
    for (int col = 0; col < n; ++col) columns[col+1] += columns[col];
    state.scatter.resize(all_indices.size()/2);
    state.gathered.resize(state.scatter.size());
    for (std::size_t q = 0; q < state.scatter.size(); ++q) {
      state.scatter[q] = global_slots.at({all_indices[2*q], all_indices[2*q+1]});
    }
    new_pattern = columns != state.columns || rows != state.rows;
    if (!use_amg && (new_pattern || state.matrix == nullptr)) {
      if (state.matrix != nullptr) cholmod_free_sparse(&state.matrix, &state.common);
      state.matrix = cholmod_allocate_sparse(n, n, rows.size(), 1, 1, -1,
                                            CHOLMOD_REAL, &state.common);
      if (state.matrix == nullptr) AggregateError("CHOLMOD coarse matrix allocation failed");
      std::copy(columns.begin(), columns.end(), static_cast<int *>(state.matrix->p));
      std::copy(rows.begin(), rows.end(), static_cast<int *>(state.matrix->i));
    }
    state.columns = std::move(columns);
    state.rows = std::move(rows);
    if (!use_amg || distributed_amg) {
      state.numerical.resize(state.rows.size());
    }
    if (!use_amg) {
      state.scaling.resize(n);
    }
#if ATHENA_HYPRE_ENABLED
    if (use_amg) {
      // Expand only owned rows of the global lower triangle. Off-rank columns
      // retain global indices for hypre's distributed matrix and halo exchange.
      // On one rank, global slots also index the compact device contributions.
      if (!distributed_amg) {
        for (std::size_t q = 0; q < state.scatter.size(); ++q) {
          if (state.scatter[q] != static_cast<int>(q)) {
            AggregateError("Invalid single-rank AMG compact matrix ordering");
          }
        }
      }
      const int last_row = first_row+local_rows;
      std::vector<int> row_offsets(local_rows+1, 0);
      for (int col = 0; col < n; ++col) {
        for (int q = state.columns[col]; q < state.columns[col+1]; ++q) {
          const int row = state.rows[q];
          if (row >= first_row && row < last_row) ++row_offsets[row-first_row+1];
          if (col != row && col >= first_row && col < last_row) {
            ++row_offsets[col-first_row+1];
          }
        }
      }
      for (int row = 0; row < local_rows; ++row) row_offsets[row+1] += row_offsets[row];
      std::vector<int> next(row_offsets), compact_slots(row_offsets[local_rows]);
      std::vector<int> csr_columns(row_offsets[local_rows]);
      for (int col = 0; col < n; ++col) {
        for (int q = state.columns[col]; q < state.columns[col+1]; ++q) {
          const int row = state.rows[q];
          if (row >= first_row && row < last_row) {
            const int slot = next[row-first_row]++;
            csr_columns[slot] = col;
            compact_slots[slot] = q;
          }
          if (col != row && col >= first_row && col < last_row) {
            const int slot = next[col-first_row]++;
            csr_columns[slot] = row;
            compact_slots[slot] = q;
          }
        }
      }
      UploadAggregateArray(compact_slots, state.amg_compact_slots);
      Kokkos::realloc(state.amg_numerical, compact_slots.size());
      if (distributed_amg) Kokkos::realloc(state.amg_host_numerical, compact_slots.size());
      Kokkos::realloc(state.amg_rhs, local_rows);
      Kokkos::realloc(state.amg_correction, local_rows);
      state.amg_row_offsets = std::move(row_offsets);
      state.amg_columns = std::move(csr_columns);
      if (state.amg == nullptr) state.amg = std::make_unique<ImplicitAMGHypre>();
    }
#endif
    state.topology = std::move(topology);
    state.topology_valid = true;
  }

  // Reduce every owned face patch once, then gather its value into the two
  // diagonals and one off-diagonal. No atomic assembly or full-face host mirrors.
  {
    RadiationTimerScope numeric_timer(solver_timers_, RadiationTimer::AggregateNumeric);
    auto patches = state.patches.d_view;
    auto patch_values = state.patch_values;
    auto device_size = mb->mb_size.d_view;
    auto face1 = ImplicitAMRFaceBand(implicit_amr_conductance_.x1f);
    auto face2 = ImplicitAMRFaceBand(implicit_amr_conductance_.x2f);
    auto face3 = ImplicitAMRFaceBand(implicit_amr_conductance_.x3f);
    const int is = start[0], js = start[1], ks = start[2];
    const double step = dt;
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (patches.extent_int(0) > 0) {
      par_for_outer("thermal_rad_aggregate_patch_sum", DevExeSpace(), 0, 0,
                    0, patches.extent_int(0)-1,
      KOKKOS_LAMBDA(TeamMember_t member, const int q) {
        const auto patch = patches(q);
        const int dir = patch.direction;
        const int tangent1 = (dir+1)%3, tangent2 = (dir+2)%3;
        const auto geometry = device_size(patch.block);
        const double volume = geometry.dx1*geometry.dx2*geometry.dx3;
        const double spacing = dir == 0 ? geometry.dx1 : (dir == 1 ? geometry.dx2 :
                                                                        geometry.dx3);
        const double area_step = step*volume/spacing;
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(member, patch.extent1*patch.extent2),
        [=](const int child, double &value) {
          int index[3] = {patch.index[0], patch.index[1], patch.index[2]};
          index[tangent1] += child%patch.extent1;
          index[tangent2] += child/patch.extent1;
          const int i = index[0]+is, j = index[1]+js, k = index[2]+ks;
          const double conductance = dir == 0 ? face1(patch.block, 0, k, j, i) :
              (dir == 1 ? face2(patch.block, 0, k, j, i) :
                          face3(patch.block, 0, k, j, i));
          const double integrated = area_step*conductance;
          value += Kokkos::isfinite(integrated) && integrated >= 0.0 ?
              integrated : invalid;
        }, sum);
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
          patch_values(q) = patch.weight*sum;
        });
      });
    }
    auto slot_offsets = state.slot_offsets.d_view;
    auto slot_terms = state.slot_terms.d_view;
    auto mass_blocks = state.slot_mass_block.d_view;
    auto compact = state.compact_values.d_view;
    const int aggregate_cells = width[0]*width[1]*width[2];
    par_for("thermal_rad_aggregate_slot_sum", DevExeSpace(),
            0, compact.extent_int(0)-1,
    KOKKOS_LAMBDA(int slot) {
      const int m = mass_blocks(slot);
      double value = 0.0;
      if (m >= 0) {
        const auto geometry = device_size(m);
        value = aggregate_cells*(geometry.dx1*geometry.dx2*geometry.dx3);
      }
      for (int q = slot_offsets(slot); q < slot_offsets(slot+1); ++q) {
        const int term = slot_terms(q);
        value += term > 0 ? patch_values(term-1) : -patch_values(-term-1);
      }
      compact(slot) = value;
    });
  }
  state.compact_values.template modify<DevExeSpace>();
#if ATHENA_HYPRE_ENABLED
  if (use_amg) {
    if (distributed_amg) {
      // A face owned by this rank also contributes to the remote endpoint's
      // diagonal. Combine all compact contributions before extracting owned
      // rows; dropping those remote terms would break conservation and symmetry.
      // Topology/scatter maps and staging buffers are reused for every group.
      {
        RadiationTimerScope copy_timer(solver_timers_, RadiationTimer::AggregateNumericD2H);
        state.compact_values.template sync<HostMemSpace>();
      }
      {
        RadiationTimerScope graph_timer(solver_timers_, RadiationTimer::AggregateGraph,
                                         false);
        const auto compact = state.compact_values.h_view;
        const int entries = compact.extent_int(0)-1;
        for (int q = 0; q <= entries; ++q) {
          if (!std::isfinite(compact(q))) {
            AggregateError("Invalid distributed AMG aggregate contribution");
          }
        }
#if MPI_PARALLEL_ENABLED
        {
          RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI,
                                         false);
          MPI_Allgatherv(compact.data(), entries, MPI_DOUBLE, state.gathered.data(),
                        state.counts.data(), state.offsets.data(), MPI_DOUBLE,
                        MPI_COMM_WORLD);
        }
#endif
        std::fill(state.numerical.begin(), state.numerical.end(), 0.0);
        for (std::size_t q = 0; q < state.gathered.size(); ++q) {
          state.numerical[state.scatter[q]] += state.gathered[q];
        }
        for (int row = 0; row < n; ++row) {
          const int slot = state.diagonal_slots[row];
          if (slot < 0 || !(state.numerical[slot] > 0.0) ||
              !std::isfinite(state.numerical[slot])) {
            AggregateError("Invalid distributed AMG aggregate diagonal");
          }
        }
        for (std::size_t q = 0; q < state.amg_columns.size(); ++q) {
          const double value = state.numerical[state.amg_compact_slots.h_view(q)];
          if (!std::isfinite(value)) AggregateError("Invalid distributed AMG matrix entry");
          state.amg_host_numerical(q) = value;
        }
      }
      {
        RadiationTimerScope copy_timer(solver_timers_, RadiationTimer::AggregateNumeric);
        Kokkos::deep_copy(state.amg_numerical, state.amg_host_numerical);
      }
    } else {
      // Preserve the single-rank path: no numerical coefficient download.
      RadiationTimerScope numeric_timer(solver_timers_, RadiationTimer::AggregateNumeric);
      auto compact = state.compact_values.d_view;
      auto mass_blocks = state.slot_mass_block.d_view;
      int invalid = 0;
      Kokkos::parallel_reduce("thermal_rad_amg_validate_compact",
          Kokkos::RangePolicy<DevExeSpace>(0, compact.extent_int(0)),
      KOKKOS_LAMBDA(const int q, int &bad) {
        if (!Kokkos::isfinite(compact(q)) ||
            (mass_blocks(q) >= 0 && !(compact(q) > 0.0))) ++bad;
      }, invalid);
      if (invalid != 0) AggregateError("Invalid AMG aggregate matrix contribution");
      auto slots = state.amg_compact_slots.d_view;
      auto numerical = state.amg_numerical;
      par_for("thermal_rad_amg_expand_symmetric_csr", DevExeSpace(),
              0, numerical.extent_int(0)-1,
      KOKKOS_LAMBDA(const int q) { numerical(q) = compact(slots(q)); });
    }
    // Setup precedes lazy FGMRES allocation only for a high-order mixed-level
    // solve. CG (including equal-level meshes) never grows an Arnoldi basis.
    const std::size_t local_cells = static_cast<std::size_t>(blocks)*cells[0]*
        cells[1]*cells[2];
    const std::size_t restart =
        std::min(implicit_gmres_restart_, implicit_max_iterations_);
    const std::size_t basis_bytes = implicit_amr_high_order_ ?
        (restart+1)*local_cells*sizeof(Real) : 0;
    const std::size_t basis_owned = implicit_gmres_basis_.size()*sizeof(Real);
    krylov_growth_bytes = basis_bytes > basis_owned ? basis_bytes-basis_owned : 0;
    if (implicit_amr_high_order_ && implicit_amr_inner_iterations_ > 0) {
      const std::size_t correction_bytes = restart*local_cells*sizeof(Real);
      const std::size_t correction_owned =
          implicit_gmres_corrections_.size()*sizeof(Real);
      if (correction_bytes > correction_owned) {
        krylov_growth_bytes += correction_bytes-correction_owned;
      }
    }
    {
      RadiationTimerScope setup_timer(solver_timers_,
                                      RadiationTimer::AggregateDeviceFactor);
      if (!state.amg->ConfigureGraph(state.amg_row_offsets, state.amg_columns, first_row, n,
                                    implicit_amr_amg_max_bytes_, krylov_growth_bytes) ||
          !state.amg->Setup(state.amg_numerical.data())) {
        AggregateError("AMG aggregate setup failed: "+state.amg->error());
      }
    }
    if (solver_timers_.enabled) {
      std::cout << "# radiation aggregate cache: rank=" << global_variable::my_rank
                << " backend=amg nodes=" << n
                << " first_row=" << first_row << " local_rows=" << local_rows
                << " matrix_entries=" << state.amg_columns.size()
                << " local_patches=" << state.patches.extent(0)
                << " compact_download_bytes=" << (distributed_amg ?
                     state.compact_values.extent(0)*sizeof(double) : 0)
                << " topology_rebuilt=" << rebuild
                << " device_factor_bytes=" << state.amg->device_bytes()
                << " amg_max_bytes=" << implicit_amr_amg_max_bytes_
                << " krylov_growth_bytes=" << krylov_growth_bytes << '\n';
    }
    return;
  }
#else
  if (use_amg) AggregateError("amr-amg requires Athena_ENABLE_HYPRE=ON");
#endif
  {
    RadiationTimerScope copy_timer(solver_timers_, RadiationTimer::AggregateNumericD2H);
    state.compact_values.template sync<HostMemSpace>();
  }
  {
    RadiationTimerScope graph_timer(solver_timers_, RadiationTimer::AggregateGraph, false);
    const auto compact = state.compact_values.h_view;
    const int entries = compact.extent_int(0)-1;
    for (int q = 0; q <= entries; ++q) {
      if (!std::isfinite(compact(q))) AggregateError("Invalid aggregate face contribution");
    }
#if MPI_PARALLEL_ENABLED
    {
      RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateBuildMPI, false);
      MPI_Allgatherv(compact.data(), entries, MPI_DOUBLE, state.gathered.data(),
                    state.counts.data(), state.offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
#else
    std::copy(compact.data(), compact.data()+entries, state.gathered.begin());
#endif
    std::fill(state.numerical.begin(), state.numerical.end(), 0.0);
    // Same rank/local-key ordering as the original MPI map assembly.
    for (std::size_t q = 0; q < state.gathered.size(); ++q) {
      state.numerical[state.scatter[q]] += state.gathered[q];
    }
    for (int row = 0; row < n; ++row) {
      if (state.diagonal_slots[row] < 0) AggregateError("Missing aggregate diagonal");
      const double diagonal = state.numerical[state.diagonal_slots[row]];
      if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
        AggregateError("Invalid global aggregate diagonal");
      }
      state.scaling[row] = 1.0/std::sqrt(diagonal);
    }
    auto *numerical = static_cast<double *>(state.matrix->x);
    for (int col = 0; col < n; ++col) {
      for (int q = state.columns[col]; q < state.columns[col+1]; ++q) {
        numerical[q] = state.numerical[q]*state.scaling[col]*state.scaling[state.rows[q]];
      }
    }
  }
  if (new_pattern || state.factor == nullptr) {
    if (state.factor != nullptr) cholmod_free_factor(&state.factor, &state.common);
    RadiationTimerScope analyze_timer(solver_timers_, RadiationTimer::CholmodAnalyze, false);
    state.factor = cholmod_analyze(state.matrix, &state.common);
  }
  bool factorized = false;
  if (state.factor != nullptr) {
    RadiationTimerScope factor_timer(solver_timers_, RadiationTimer::CholmodFactor, false);
    factorized = cholmod_factorize(state.matrix, state.factor, &state.common);
  }
  if (!factorized ||
      state.common.status != CHOLMOD_OK ||
      state.factor->minor != static_cast<std::size_t>(n)) {
    AggregateError("CHOLMOD global AMR aggregate factorization failed");
  }
#if ATHENA_CUSPARSE_ENABLED
  if (state.use_device) {
    cholmod_factor *copy = nullptr;
    cholmod_sparse *lower = nullptr;
    {
      RadiationTimerScope export_timer(solver_timers_, RadiationTimer::AggregateFactorExport,
                                       false);
      // final_asis can leave the original factor in LDL' form. Convert a copy
      // explicitly, retaining CHOLMOD's original host solve representation.
      copy = cholmod_copy_factor(state.factor, &state.common);
      if (copy == nullptr ||
          !cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, copy, &state.common) ||
          !copy->is_ll || copy->is_super || copy->itype != CHOLMOD_INT ||
          copy->dtype != CHOLMOD_DOUBLE || copy->xtype != CHOLMOD_REAL) {
        AggregateError("Could not export a packed real LL' aggregate factor");
      }
      lower = cholmod_factor_to_sparse(copy, &state.common);
      if (lower == nullptr || state.common.status != CHOLMOD_OK ||
          !lower->packed || lower->itype != CHOLMOD_INT ||
          lower->dtype != CHOLMOD_DOUBLE || lower->xtype != CHOLMOD_REAL) {
        AggregateError("Invalid sparse aggregate LL' factor");
      }
      if ((!lower->sorted && !cholmod_sort(lower, &state.common)) ||
          state.common.status != CHOLMOD_OK) {
        AggregateError("Could not sort the aggregate LL' factor");
      }
    }
    const int factor_nnz = static_cast<const int *>(lower->p)[n];
    state.use_dense = ImplicitAggregateDenseFactor::SupportsDense(n, factor_nnz);
    if (!state.use_dense) dense_fallback = "storage_budget";
    if (state.use_dense) {
      // Reserve lazy GMRES growth only when this solve actually uses GMRES.
      // Equal-level and low-order CG solves need no additional basis storage.
      const std::size_t local_cells = static_cast<std::size_t>(blocks)*cells[0]*
          cells[1]*cells[2];
      const std::size_t restart = std::min(implicit_gmres_restart_, implicit_max_iterations_);
      const std::size_t basis_bytes = implicit_amr_high_order_ ?
          (restart+1)*local_cells*sizeof(Real) : 0;
      const std::size_t basis_owned = implicit_gmres_basis_.size()*sizeof(Real);
      krylov_growth_bytes = basis_bytes > basis_owned ? basis_bytes-basis_owned : 0;
      if (implicit_amr_high_order_ && implicit_amr_inner_iterations_ > 0) {
        const std::size_t correction_bytes = restart*local_cells*sizeof(Real);
        const std::size_t correction_owned = implicit_gmres_corrections_.size()*sizeof(Real);
        if (correction_bytes > correction_owned) {
          krylov_growth_bytes += correction_bytes-correction_owned;
        }
      }
      // Matches the helper's owned-view budget: L, inverse(L), sparse export
      // values/indices, permutation and four vectors. Inactive helpers are freed
      // before allocating the selected backend, so their storage is reusable.
      const std::size_t nodes = static_cast<std::size_t>(n);
      const std::size_t requested = 16*nodes*nodes+16*static_cast<std::size_t>(factor_nnz)+
          36*nodes;
      const std::size_t owned = (state.dense_factor ? state.dense_factor->DeviceBytes() : 0)+
          (state.device_factor ? state.device_factor->DeviceBytes() : 0);
      dense_additional_bytes = requested > owned ? requested-owned : 0;
      constexpr std::size_t margin = std::size_t{128} << 20;
      std::size_t total_bytes = 0;
      dense_memory_checked = true;
      const auto memory_status = cudaMemGetInfo(&dense_free_bytes, &total_bytes);
      if (memory_status != cudaSuccess || dense_free_bytes < dense_additional_bytes+
          krylov_growth_bytes+margin) {
        // Memory pressure selects host even when device was requested explicitly;
        // a sparse fallback could still consume the remaining Krylov headroom.
        state.use_device = false;
        state.use_dense = false;
        dense_fallback = memory_status == cudaSuccess ? "free_memory" : "memory_query";
      }
    }
    if (!state.use_device || (!state.use_dense && implicit_amr_coarse_solver_ == 0)) {
      // Auto prefers the host solve when the actual factor exceeds the dense
      // workspace budget. Explicit device retains the sparse compatibility path.
      state.use_device = false;
      state.dense_factor.reset();
      state.device_factor.reset();
    } else {
      RadiationTimerScope device_timer(solver_timers_, RadiationTimer::AggregateDeviceFactor);
      if (state.use_dense) {
        state.device_factor.reset();
        if (state.dense_factor == nullptr) {
          state.dense_factor = std::make_unique<ImplicitAggregateDenseFactor>();
        }
        const bool usable = state.dense_factor->Update(n, static_cast<const int *>(lower->p),
            static_cast<const int *>(lower->i), static_cast<const double *>(lower->x),
            static_cast<const int *>(state.factor->Perm), state.scaling.data());
        if (!usable) {
          state.dense_factor.reset();
          state.use_dense = false;
          dense_fallback = "invalid_inverse_factor";
          if (implicit_amr_coarse_solver_ == 0) state.use_device = false;
        }
      }
      if (state.use_device && !state.use_dense) {
        state.dense_factor.reset();
        if (state.device_factor == nullptr) {
          state.device_factor = std::make_unique<ImplicitAggregateDeviceFactor>();
        }
        state.device_factor->Update(n, static_cast<const int *>(lower->p),
            static_cast<const int *>(lower->i), static_cast<const double *>(lower->x),
            static_cast<const int *>(state.factor->Perm), state.scaling.data());
      }
    }
    cholmod_free_sparse(&lower, &state.common);
    cholmod_free_factor(&copy, &state.common);
  }
#endif
  if (state.values.extent_int(0) != n) {
    Kokkos::realloc(state.values, n);
    if (state.rhs != nullptr) cholmod_free_dense(&state.rhs, &state.common);
    if (state.solution != nullptr) cholmod_free_dense(&state.solution, &state.common);
    if (state.solve_workspace_y != nullptr) {
      cholmod_free_dense(&state.solve_workspace_y, &state.common);
    }
    if (state.solve_workspace_e != nullptr) {
      cholmod_free_dense(&state.solve_workspace_e, &state.common);
    }
  }
  if (!state.use_device && state.rhs == nullptr) {
    state.rhs = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &state.common);
    if (state.rhs == nullptr) AggregateError("CHOLMOD coarse RHS allocation failed");
  }
  if (solver_timers_.enabled) {
    std::cout << "# radiation aggregate cache: rank=" << global_variable::my_rank
              << " backend=" << (state.use_device ?
                  (state.use_dense ? "device_dense" : "device_sparse") : "host")
              << " nodes=" << n << " matrix_entries=" << state.rows.size()
              << " local_patches=" << state.patches.extent(0)
              << " compact_download_bytes=" << state.compact_values.extent(0)*sizeof(double)
              << " topology_rebuilt=" << rebuild
              << " dense_fallback=" << dense_fallback;
    if (dense_memory_checked) {
      std::cout << " dense_free_bytes=" << dense_free_bytes
                << " dense_additional_bytes=" << dense_additional_bytes
                << " krylov_growth_bytes=" << krylov_growth_bytes
                << " dense_library_margin_bytes=" << (std::size_t{128} << 20);
    }
#if ATHENA_CUSPARSE_ENABLED
    if (state.use_device) {
      const int factor_nnz = state.use_dense ? state.dense_factor->FactorNonzeros() :
                                              state.device_factor->FactorNonzeros();
      const std::size_t bytes = state.use_dense ? state.dense_factor->DeviceBytes() :
                                                 state.device_factor->DeviceBytes();
      const std::size_t analyses = state.use_dense ? state.dense_factor->AnalysisBuilds() :
                                                    state.device_factor->AnalysisBuilds();
      std::cout << " factor_nnz=" << factor_nnz
                << " device_factor_bytes=" << bytes
                << " device_analysis_builds=" << analyses;
    }
#endif
    std::cout << '\n';
  }
#else
  (void)dt;
  AggregateError("amr-aggregate requires Athena_ENABLE_CHOLMOD=ON");
#endif
}

//----------------------------------------------------------------------------------------
//! Complete Jacobi plus local and global aggregate corrections. The caller has
//! stored local corrections and unscaled global aggregate sums in packed scratch.
void ThermalRadiation::ApplyImplicitAMRAggregatePreconditioner(
    const DvceArray5D<Real> &residual, DvceArray5D<Real> &preconditioned) {
#if ATHENA_CHOLMOD_ENABLED
  RadiationTimerScope apply_timer(solver_timers_, RadiationTimer::AggregateApply);
  auto &state = *implicit_amr_aggregate_state_;
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
#if ATHENA_HYPRE_ENABLED
  if (implicit_preconditioner_mode_ == 4) {
    const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
    const int per_block = nx*ny*nz;
    const int rows = pmy_pack_->nmb_thispack*per_block;
    auto rhs = state.amg_rhs;
    auto correction = state.amg_correction;
    auto size = pmy_pack_->pmb->mb_size.d_view;
    const bool volume_weighted = state.cell_volume_weighted;
    par_for("thermal_rad_cell_amg_pack", DevExeSpace(), 0, rows-1,
    KOKKOS_LAMBDA(const int row) {
      const int a = row%per_block;
      const int m = row/per_block;
      const double volume = volume_weighted ? size(m).dx1*size(m).dx2*size(m).dx3 : 1.0;
      rhs(row) = volume*residual(m, 0, ks+a/(nx*ny), js+(a/nx)%ny, is+a%nx);
    });
    {
      RadiationTimerScope solve_timer(solver_timers_, RadiationTimer::AggregateDeviceSolve);
      if (!state.amg->Apply(rhs.data(), correction.data())) {
        AggregateError("Cell AMG application failed: "+state.amg->error());
      }
    }
    par_for("thermal_rad_cell_amg_unpack", DevExeSpace(), 0, rows-1,
    KOKKOS_LAMBDA(const int row) {
      const int a = row%per_block;
      preconditioned(row/per_block, 0, ks+a/(nx*ny), js+(a/nx)%ny, is+a%nx) =
          correction(row);
    });
    return;
  }
  if (implicit_preconditioner_mode_ == 3) {
    const int ay = indcs.nx2 == 1 ? 1 : 3, az = indcs.nx3 == 1 ? 1 : 3;
    const int nx = indcs.nx1/3, ny = indcs.nx2/ay, nz = indcs.nx3/az;
    const int per_block = nx*ny*nz;
    const int level1_ay = indcs.nx2 == 1 ? 1 : 9;
    const int level1_az = indcs.nx3 == 1 ? 1 : 9;
    const int level1_nx = implicit_multilevel_nx1_[1];
    const int level1_ny = implicit_multilevel_nx2_[1];
    const int level1_nz = implicit_multilevel_nx3_[1];
    const int level1_offset = implicit_multilevel_offset_[1];
    const int sum_offset = level1_offset+level1_nx*level1_ny*level1_nz;
    const int fine_nx1 = indcs.nx1, fine_nx2 = indcs.nx2;
    const int nmb1 = pmy_pack_->nmb_thispack-1;
    auto rhs = state.amg_rhs;
    auto correction = state.amg_correction;
    auto local = implicit_coarse_scratch_;
    auto size = pmy_pack_->pmb->mb_size.d_view;
    auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
    auto r = residual;
    auto z = preconditioned;
    {
      RadiationTimerScope restrict_timer(solver_timers_,
                                         RadiationTimer::AggregateRestrict);
      par_for("thermal_rad_amg_restrict", DevExeSpace(), 0, nmb1, 0, per_block-1,
      KOKKOS_LAMBDA(const int m, const int a) {
        const double sum = ImplicitPackedScratchValue(
            local, m, sum_offset+a, fine_nx1, fine_nx2, is, js, ks);
        rhs(m*per_block+a) = sum*size(m).dx1*size(m).dx2*size(m).dx3;
      });
    }
    {
      RadiationTimerScope solve_timer(solver_timers_,
                                      RadiationTimer::AggregateDeviceSolve);
      if (!state.amg->Apply(rhs.data(), correction.data())) {
        AggregateError("AMG aggregate correction failed: "+state.amg->error());
      }
    }
    {
      RadiationTimerScope prolong_timer(solver_timers_, RadiationTimer::AggregateProlong);
      par_for("thermal_rad_amg_prolong", DevExeSpace(), 0, nmb1,
              ks, indcs.ke, js, indcs.je, is, indcs.ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const int a0 = (((k-ks)/az)*ny+(j-js)/ay)*nx+(i-is)/3;
        const int a1 = (((k-ks)/level1_az)*level1_ny+(j-js)/level1_ay)*level1_nx+
            (i-is)/9;
        Real value = r(m, 0, k, j, i)/diagonal(m, 0, k, j, i);
        value += ImplicitPackedScratchValue(local, m, a0, fine_nx1, fine_nx2, is, js, ks);
        value += ImplicitPackedScratchValue(
            local, m, level1_offset+a1, fine_nx1, fine_nx2, is, js, ks);
        z(m, 0, k, j, i) = value+static_cast<Real>(correction(m*per_block+a0));
      });
    }
    return;
  }
#endif
  const int ax = 9, ay = indcs.nx2 == 1 ? 1 : 9, az = indcs.nx3 == 1 ? 1 : 9;
  const int nx = indcs.nx1/ax, ny = indcs.nx2/ay, nz = indcs.nx3/az;
  const int per_block = nx*ny*nz;
  const int fine_nx1 = indcs.nx1, fine_nx2 = indcs.nx2;
  const int local_nx = implicit_multilevel_nx1_[0];
  const int local_ny = implicit_multilevel_nx2_[0];
  const int local_ay = indcs.nx2 == 1 ? 1 : 3;
  const int local_az = indcs.nx3 == 1 ? 1 : 3;
  const int level1_offset = implicit_multilevel_offset_[1];
  const int sum_offset = level1_offset+per_block;
  const int gid0 = pmy_pack_->gids;
  const int nmb1 = pmy_pack_->nmb_thispack-1;
  const int n = state.values.extent_int(0);
  auto coarse = state.values.d_view;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  auto r = residual;
  auto z = preconditioned;
  auto local = implicit_coarse_scratch_;
  auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
  RadiationTimerScope restrict_timer(solver_timers_, RadiationTimer::AggregateRestrict);
  // On one rank every entry is overwritten. Other ranks must contribute zeros
  // outside their owned block range to the replicated coarse RHS collective.
  if (global_variable::nranks > 1) Kokkos::deep_copy(DevExeSpace(), coarse, 0.0);
  par_for("thermal_rad_global_aggregate_restrict", DevExeSpace(), 0, nmb1,
          0, per_block-1,
  KOKKOS_LAMBDA(const int m, const int a) {
    const Real sum = ImplicitPackedScratchValue(
        local, m, sum_offset+a, fine_nx1, fine_nx2, is, js, ks);
    coarse((gid0+m)*per_block+a) = sum*size(m).dx1*size(m).dx2*size(m).dx3;
  });
  restrict_timer.Stop();
  state.values.template modify<DevExeSpace>();
  auto host = state.values.h_view;
  // A single-rank device solve stays entirely on device. Multi-rank runs retain
  // the portable host MPI collective, then solve the replicated factor locally.
  const bool needs_host = !state.use_device || global_variable::nranks > 1;
  if (needs_host) {
    RadiationTimerScope rhs_copy_timer(solver_timers_, RadiationTimer::AggregateRhsD2H);
    state.values.template sync<HostMemSpace>();
  }
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    RadiationTimerScope mpi_timer(solver_timers_, RadiationTimer::AggregateApplyMPI, false);
    RadiationBatchAllreduce(MPI_IN_PLACE, host.data(), n, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
#if ATHENA_CUSPARSE_ENABLED
  if (state.use_device) {
    if (needs_host) {
      state.values.template modify<HostMemSpace>();
      RadiationTimerScope copy_timer(solver_timers_, RadiationTimer::AggregateCorrectionH2D);
      state.values.template sync<DevExeSpace>();
    }
    {
      RadiationTimerScope solve_timer(solver_timers_, RadiationTimer::AggregateDeviceSolve);
      if (state.use_dense) {
        state.dense_factor->SolveInPlace(coarse);
      } else {
        state.device_factor->SolveInPlace(coarse);
      }
    }
    state.values.template modify<DevExeSpace>();
  } else
#endif
  {
    {
      RadiationTimerScope rhs_scale_timer(solver_timers_, RadiationTimer::AggregateRhsScale,
                                          false);
      auto *rhs = static_cast<double *>(state.rhs->x);
      for (int a = 0; a < n; ++a) rhs[a] = state.scaling[a]*host(a);
    }
    int solved = 0;
    {
      RadiationTimerScope solve_timer(solver_timers_, RadiationTimer::CholmodSolve, false);
      solved = cholmod_solve2(CHOLMOD_A, state.factor, state.rhs, nullptr,
          &state.solution, nullptr, &state.solve_workspace_y, &state.solve_workspace_e,
          &state.common);
    }
    if (!solved || state.solution == nullptr || state.common.status != CHOLMOD_OK) {
      AggregateError("CHOLMOD global AMR aggregate solve failed");
    }
    {
      RadiationTimerScope solution_scale_timer(solver_timers_,
                                               RadiationTimer::AggregateSolutionScale, false);
      const auto *solution = static_cast<double *>(state.solution->x);
      for (int a = 0; a < n; ++a) {
        host(a) = static_cast<Real>(state.scaling[a]*solution[a]);
        if (!std::isfinite(host(a))) {
          AggregateError("Nonfinite global AMR aggregate correction");
        }
      }
    }
    state.values.template modify<HostMemSpace>();
    {
      RadiationTimerScope correction_copy_timer(solver_timers_,
                                                RadiationTimer::AggregateCorrectionH2D);
      state.values.template sync<DevExeSpace>();
    }
  }
  RadiationTimerScope prolong_timer(solver_timers_, RadiationTimer::AggregateProlong);
  par_for("thermal_rad_global_aggregate_prolong", DevExeSpace(), 0, nmb1,
          ks, indcs.ke, js, indcs.je, is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const int a = (((k-ks)/az)*ny+(j-js)/ay)*nx+(i-is)/ax;
    const int a0 = (((k-ks)/local_az)*local_ny+(j-js)/local_ay)*local_nx+(i-is)/3;
    Real value = r(m, 0, k, j, i)/diagonal(m, 0, k, j, i);
    value += ImplicitPackedScratchValue(local, m, a0, fine_nx1, fine_nx2, is, js, ks);
    value += ImplicitPackedScratchValue(
        local, m, level1_offset+a, fine_nx1, fine_nx2, is, js, ks);
    z(m, 0, k, j, i) = value+coarse((gid0+m)*per_block+a);
  });
#else
  (void)residual;
  (void)preconditioned;
  AggregateError("amr-aggregate requires Athena_ENABLE_CHOLMOD=ON");
#endif
}

}  // namespace two_temperature
