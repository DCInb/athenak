//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! Shared conservative leaf graph for the native diffusion preconditioner.

#include "diffusion/native_diffusion_solver.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "bvals/bvals.hpp"
#include "diffusion/native_multigrid.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"

namespace diffusion {
namespace {
#if MPI_PARALLEL_ENABLED
struct NativeHaloWait {
  int count;
  MPI_Request *requests;
};
void CompleteNativeHalos(const std::vector<void *> &pending) {
  // Every lane has already posted its nonblocking sends/receives. Waiting in
  // sequence here still allows all lanes' transfers to progress concurrently.
  for (void *entry : pending) {
    const auto &wait = *static_cast<NativeHaloWait *>(entry);
    MPI_Waitall(wait.count, wait.requests, MPI_STATUSES_IGNORE);
  }
}
void WaitNativeHalo(int count, MPI_Request *requests) {
  auto *batch = two_temperature::RadiationGroupBatch::Active();
  if (batch == nullptr) {
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    return;
  }
  NativeHaloWait wait{count, requests};
  batch->Submit(two_temperature::RadiationGroupBatch::Kind::Halo,
                &wait, CompleteNativeHalos);
}
#endif
[[noreturn]] void NativeError(const std::string &message) {
  std::cerr << "### FATAL ERROR in native diffusion: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
struct NativeFace { int block, cell, face; };
template <typename T>
void UploadArray(const std::vector<T> &source, DualArray1D<T> &target) {
  Kokkos::realloc(target, source.size());
  for (std::size_t i = 0; i < source.size(); ++i) target.h_view(i) = source[i];
  target.template modify<HostMemSpace>();
  target.template sync<DevExeSpace>();
}
KOKKOS_INLINE_FUNCTION
bool Physical(const BoundaryFlag flag) {
  return flag != BoundaryFlag::block && flag != BoundaryFlag::periodic &&
         flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef;
}
}  // namespace

struct NativeDiffusionSolver::State {
  NativeMultigrid solver;
  std::vector<int> topology, row_offsets, columns;
  bool topology_valid = false, cell_volume_weighted = false;
  DualArray1D<int> cell_row_offsets, cell_local_slots, cell_remote_slots;
  DualArray1D<unsigned char> cell_entry_faces;
  DualArray1D<NativeFace> cell_fine_faces;
  DualArray1D<double> cell_face_values, cell_remote_values;
  std::vector<int> cell_send_counts, cell_send_offsets;
  std::vector<int> cell_recv_counts, cell_recv_offsets;
  DvceArray1D<double> conductance, mass, rhs, correction;
};

NativeDiffusionSolver::NativeDiffusionSolver() : state_(new State) {}
NativeDiffusionSolver::~NativeDiffusionSolver() = default;

void NativeDiffusionSolver::Build(MeshBlockPack *pack,
    const DvceArray5D<Real> &coefficient, const DvceFaceFld5D<Real> &face_fields,
    FaceFldOrigin origin, const DvceArray5D<Real> &capacity, Real dt,
    const int boundary[6], Real vacuum_cap, bool device_mpi, std::size_t max_bytes) {
  const auto &indcs = pack->pmesh->mb_indcs;
  const int nx = indcs.nx1, ny = indcs.nx2, nz = indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int blocks = pack->nmb_thispack, gid0 = pack->gids;
  auto *mesh = pack->pmesh;
  const int total_blocks = mesh->nmb_total;
  const int faces = nz > 1 ? 6 : (ny > 1 ? 4 : 2);
  const std::size_t cells = static_cast<std::size_t>(nx)*ny*nz;
  if (cells*total_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      cells*blocks*(1+faces*(1 << (faces/2-1))) >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    NativeError("native-mg graph exceeds the integer index limit");
  }
  const int per_block = static_cast<int>(cells), rows = blocks*per_block;
  bool mixed_levels = false;
  for (int gid = 1; gid < total_blocks; ++gid) {
    mixed_levels |= mesh->lloc_eachmb[gid].level != mesh->lloc_eachmb[0].level;
  }
  if (gid0 != mesh->gids_eachrank[global_variable::my_rank] ||
      blocks != mesh->nmb_eachrank[global_variable::my_rank]) {
    NativeError("native-mg requires one complete MeshBlockPack per rank");
  }
  auto *mb = pack->pmb;
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
  for (const auto &face : peers) {
    topology.insert(topology.end(), face.begin(), face.end());
  }
  for (int face = 0; face < faces; ++face) topology.push_back(boundary[face]);
  auto &state = *state_;
  const bool rebuild = !state.topology_valid || state.topology != topology;
  if (rebuild) {
    state.row_offsets.clear();
    state.columns.clear();
    state.row_offsets.reserve(rows+1);
    state.columns.reserve(static_cast<std::size_t>(rows)*(faces+1));
    std::vector<unsigned char> tags;
    tags.reserve(static_cast<std::size_t>(rows)*(faces+1));
    const int ranks = global_variable::nranks, my_rank = global_variable::my_rank;
    std::vector<std::vector<int>> requests(ranks), request_slots(ranks);
    std::vector<NativeFace> fine_faces;
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
            state.row_offsets.push_back(state.columns.size());
            // A coarse face touches up to four fine cells in 3D.
            Entry entries[25];
            int count = 0;
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
                    if (axis != dir && child[axis] != 2*index[axis]/dims[axis]) {
                      covers = false;
                    }
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
                state.columns.push_back(entries[q].column);
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
                  requests[owner].insert(requests[owner].end(),
                                         {cell, entries[q].fine_face});
                  request_slots[owner].push_back(slot);
                }
              }
            }
          }
        }
      }
    }
    state.row_offsets.push_back(state.columns.size());
    if (state.columns.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      NativeError("Native multigrid CSR exceeds integer indexing");
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
      remote_slots.insert(remote_slots.end(),
                           request_slots[rank].begin(), request_slots[rank].end());
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
      if (block < 0 || block >= blocks) {
        NativeError("Native multigrid face request has wrong owner");
      }
      fine_faces.push_back({block, incoming[q]%per_block, incoming[q+1]});
    }
    UploadArray(fine_faces, state.cell_fine_faces);
    UploadArray(local_slots, state.cell_local_slots);
    UploadArray(remote_slots, state.cell_remote_slots);
    Kokkos::realloc(state.cell_face_values, fine_faces.size());
    Kokkos::realloc(state.cell_remote_values, remote_slots.size());
    UploadArray(state.row_offsets, state.cell_row_offsets);
    UploadArray(tags, state.cell_entry_faces);
    Kokkos::realloc(state.conductance, tags.size());
    Kokkos::realloc(state.rhs, rows);
    Kokkos::realloc(state.correction, rows);

    state.topology = std::move(topology);
    state.topology_valid = true;
    state.cell_volume_weighted = mesh->multilevel;
    Kokkos::realloc(state.mass, rows);
    state.solver.Configure(state.row_offsets, state.columns,
                           gid0*per_block, total_blocks*per_block, device_mpi, max_bytes);
  }

  auto offsets = state.cell_row_offsets.d_view;
  auto tags = state.cell_entry_faces.d_view;
  auto numerical = state.conductance;
  auto mass = state.mass;
  auto f1 = MakeBandView(face_fields.x1f, origin);
  auto f2 = MakeBandView(face_fields.x2f, origin);
  auto f3 = MakeBandView(face_fields.x3f, origin);
  auto size = mb->mb_size.d_view;
  auto bcs = mb->mb_bcs.d_view;
  const bool composite = mesh->multilevel;
  const bool has_capacity = capacity.size() != 0;
  const int b0 = boundary[0], b1 = boundary[1], b2 = boundary[2];
  const int b3 = boundary[3], b4 = boundary[4], b5 = boundary[5];
  par_for("native_diffusion_matrix", DevExeSpace(), 0, rows-1,
  KOKKOS_LAMBDA(const int row) {
    const int m = row/per_block, a = row%per_block;
    const int i = is+a%nx, j = js+(a/nx)%ny, k = ks+a/(nx*ny);
    const Real spacing[3] = {size(m).dx1, size(m).dx2, size(m).dx3};
    const int bc[6] = {b0, b1, b2, b3, b4, b5};
    const bool edge[6] = {i == is, i == is+nx-1, j == js,
                         j == js+ny-1, k == ks, k == ks+nz-1};
    const double volume = composite ? size(m).dx1*size(m).dx2*size(m).dx3 : 1.0;
    double rates[6] = {};
    double cell_mass = has_capacity ? capacity(m, 0, k, j, i) : 1.0;
    for (int face = 0; face < faces; ++face) {
      const int dir = face/2, sign = face%2 == 0 ? -1 : 1;
      const bool physical = edge[face] && Physical(bcs(m, face));
      double conductance;
      if (composite) {
        conductance = dir == 0 ? f1(m, 0, k, j, i+face%2) :
            (dir == 1 ? f2(m, 0, k, j+face%2, i) : f3(m, 0, k+face%2, j, i));
      } else {
        const double left = coefficient(m, 0, k, j, i);
        const double right = coefficient(m, 0, k+(dir == 2 ? sign : 0),
            j+(dir == 1 ? sign : 0), i+(dir == 0 ? sign : 0));
        conductance = 0.5*(left+right)/spacing[dir];
        if (physical && bc[face] == 2) conductance = fmin(conductance, vacuum_cap);
      }
      rates[face] = dt*conductance/spacing[dir];
      if (physical) {
        cell_mass += rates[face]*(bc[face] == 0 ? 0 : (bc[face] == 1 ? 2 : 1));
      }
    }
    mass(row) = cell_mass*volume;
    for (int q = offsets(row); q < offsets(row+1); ++q) {
      double value = 0.0;
      for (int face = 0; face < faces; ++face) {
        if (tags(q) & (1 << face)) value += rates[face];
      }
      numerical(q) = value*volume;
    }
  });
  if (mixed_levels) {
    auto fine_faces = state.cell_fine_faces.d_view;
    auto values = state.cell_face_values.d_view;
    par_for("native_diffusion_fine_faces", DevExeSpace(), 0, fine_faces.extent_int(0)-1,
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
    par_for("native_diffusion_local_interfaces", DevExeSpace(), 0, local_count-1,
    KOKKOS_LAMBDA(const int q) { Kokkos::atomic_add(&numerical(slots(q)), values(q)); });
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      state.cell_face_values.template modify<DevExeSpace>();
      state.cell_face_values.template sync<HostMemSpace>();
      auto send = state.cell_face_values.h_view;
      auto recv = state.cell_remote_values.h_view;
      MPI_Alltoallv(send.data() ? send.data()+local_count : nullptr,
                    state.cell_send_counts.data(), state.cell_send_offsets.data(),
                    MPI_DOUBLE, recv.data(), state.cell_recv_counts.data(),
                    state.cell_recv_offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
      state.cell_remote_values.template modify<HostMemSpace>();
      state.cell_remote_values.template sync<DevExeSpace>();
      auto remote = state.cell_remote_values.d_view;
      auto remote_slots = state.cell_remote_slots.d_view;
      par_for("native_diffusion_remote_interfaces", DevExeSpace(),
              0, remote_slots.extent_int(0)-1,
      KOKKOS_LAMBDA(const int q) {
        Kokkos::atomic_add(&numerical(remote_slots(q)), remote(q));
      });
    }
#endif
  }
  state.solver.Setup(state.conductance, state.mass);
}

void NativeDiffusionSolver::Apply(MeshBlockPack *pack,
    const DvceArray5D<Real> &residual, DvceArray5D<Real> &correction) {
  auto &state = *state_;
#if MPI_PARALLEL_ENABLED
  state.solver.SetHaloWait(two_temperature::RadiationGroupBatch::Active() ?
                           WaitNativeHalo : nullptr);
#endif
  const auto &indcs = pack->pmesh->mb_indcs;
  const int nx = indcs.nx1, ny = indcs.nx2, per_block = nx*ny*indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int rows = pack->nmb_thispack*per_block;
  auto rhs = state.rhs;
  auto z = state.correction;
  auto size = pack->pmb->mb_size.d_view;
  const bool weighted = state.cell_volume_weighted;
  par_for("native_diffusion_pack", DevExeSpace(), 0, rows-1,
  KOKKOS_LAMBDA(const int row) {
    const int a = row%per_block, m = row/per_block;
    const double volume = weighted ? size(m).dx1*size(m).dx2*size(m).dx3 : 1.0;
    rhs(row) = volume*residual(m, 0, ks+a/(nx*ny), js+(a/nx)%ny, is+a%nx);
  });
  state.solver.Apply(rhs, z);
  par_for("native_diffusion_unpack", DevExeSpace(), 0, rows-1,
  KOKKOS_LAMBDA(const int row) {
    const int a = row%per_block;
    correction(row/per_block, 0, ks+a/(nx*ny), js+(a/nx)%ny, is+a%nx) = z(row);
  });
}

int NativeDiffusionSolver::levels() const { return state_->solver.levels(); }
std::size_t NativeDiffusionSolver::bytes() const {
  const auto &s = *state_;
  // The group preflight reserves a complete future adapter workspace as well
  // as its hierarchy. Credit all existing device arrays: counting only the
  // hierarchy charges persistent adapter storage twice on subsequent steps.
  // DualView host mirrors are not GPU storage and must not be counted here.
  return s.solver.bytes()+
      sizeof(int)*(s.cell_row_offsets.d_view.size()+s.cell_local_slots.d_view.size()+
                   s.cell_remote_slots.d_view.size())+
      sizeof(unsigned char)*s.cell_entry_faces.d_view.size()+
      sizeof(NativeFace)*s.cell_fine_faces.d_view.size()+
      sizeof(double)*(s.cell_face_values.d_view.size()+s.cell_remote_values.d_view.size()+
                      s.conductance.size()+s.mass.size()+s.rhs.size()+s.correction.size());
}
}  // namespace diffusion
