//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "bvals/bvals.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation_halo.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"

namespace two_temperature {
namespace {

[[noreturn]] void HaloError(const char *message) {
  std::cerr << "### FATAL implicit scalar halo error: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

#if MPI_PARALLEL_ENABLED
void CheckMPI(int status) {
  if (status != MPI_SUCCESS) HaloError("MPI operation failed");
}
#endif

// The rank and destination keys are used only when constructing the host plan.
// Each device region describes a subset of an existing block boundary buffer.
struct RegionBounds {
  int bis, bie, bjs, bje, bks, bke;
};

struct Region {
  int block, slot, peer, destination, destination_slot;
  int offset = 0, count = 0, coarse = 0, local = 0;
  RegionBounds bounds;
};

struct Peer {
  int rank, send_offset = 0, send_count = 0, recv_offset = 0, recv_count = 0;
};

#ifdef KOKKOS_ENABLE_CUDA
using StagingSpace = Kokkos::CudaHostPinnedSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using StagingSpace = Kokkos::HIPHostPinnedSpace;
#else
using StagingSpace = Kokkos::HostSpace;
#endif

struct Plan {
  DvceArray1D<Region> send_regions, recv_regions;
  DvceArray1D<Real> send, recv;
  Kokkos::View<Real *, StagingSpace> host_send, host_recv;
  std::vector<Peer> peers;
  int remote_recv_count = 0;
  bool built = false;
#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> requests;
#endif
};

int AppendCount(int offset, int count) {
  if (count <= 0 || offset > std::numeric_limits<int>::max()-count) {
    HaloError("invalid or oversized packed message");
  }
  return offset+count;
}

void BuildPlan(Plan &plan, MeshBlockPack *pack, MeshBoundaryValuesCC *bvals,
               bool faces_only, bool host_staging, bool flux = false, int width = 0) {
  const int rank = global_variable::my_rank, ranks = global_variable::nranks;
  const int nmb = pack->nmb_thispack, slots = pack->pmb->nnghbr;
  const int first_gid = pack->pmb->mb_gid.h_view(0);
  std::vector<Region> sends, receives;
  for (int m = 0; m < nmb; ++m) {
    for (int n = 0; n < slots; ++n) {
      const auto neighbor = pack->pmb->nghbr.h_view(m, n);
      if (neighbor.gid < 0 || (faces_only && !ImplicitRadiationHalo::IsFace(n))) continue;
      const int level = pack->pmb->mb_lev.h_view(m);
      for (int sending = 0; sending < 2; ++sending) {
        if (flux && (sending ? neighbor.lev >= level : neighbor.lev <= level)) continue;
        const auto &buffer = sending ? bvals->sendbuf[n] : bvals->recvbuf[n];
        Region region;
        region.block = m;
        region.slot = n;
        region.peer = neighbor.rank;
        region.destination = sending ? neighbor.gid : first_gid+m;
        region.destination_slot = sending ? neighbor.dest : n;
        region.coarse = neighbor.lev < level;
        region.local = neighbor.rank == rank;
        const auto &b = flux ? buffer.iflux_coar[0] :
            (neighbor.lev < level ? buffer.icoar[0] :
             neighbor.lev == level ? buffer.isame[0] : buffer.ifine[0]);
        // MeshBufferIndcs has a host-only constructor. Device plan elements
        // use plain integer bounds, with no host construction on the GPU.
        region.bounds = {b.bis, b.bie, b.bjs, b.bje, b.bks, b.bke};
        if (!flux && width != 0) {
          // Trim in the receiver's index space, then translate the same offsets
          // to the sender. All blocks have identical logical dimensions, and
          // dest identifies the matching receiver slot, including child parity.
          // This keeps both ends identical for every refinement relationship.
          const auto &remote = bvals->recvbuf[neighbor.dest];
          const auto &received = !sending ? b :
              (neighbor.lev > level ? remote.icoar[0] :
               neighbor.lev == level ? remote.isame[0] : remote.ifine[0]);
          const bool receiver_coarse = sending ? neighbor.lev > level : region.coarse;
          const int depth = neighbor.lev == level ? width : 1;
          const auto clipped = ImplicitRadiationHalo::CompactBounds(
              received, pack->pmesh->mb_indcs, receiver_coarse, faces_only, depth);
          if (b.bie-b.bis != received.bie-received.bis ||
              b.bje-b.bjs != received.bje-received.bjs ||
              b.bke-b.bks != received.bke-received.bks) {
            HaloError("sender and receiver boundary shapes disagree");
          }
          region.bounds = {
              b.bis+clipped.bis-received.bis, b.bie+clipped.bie-received.bie,
              b.bjs+clipped.bjs-received.bjs, b.bje+clipped.bje-received.bje,
              b.bks+clipped.bks-received.bks, b.bke+clipped.bke-received.bke};
        }
        const auto &bounds = region.bounds;
        const std::size_t count = static_cast<std::size_t>(bounds.bie-bounds.bis+1)*
                                  (bounds.bje-bounds.bjs+1)*(bounds.bke-bounds.bks+1);
        if (count == 0 || count > std::numeric_limits<int>::max()) {
          HaloError("invalid packed boundary extent");
        }
        region.count = static_cast<int>(count);
        (sending ? sends : receives).push_back(region);
      }
    }
  }
  // Both ends order messages by the RECEIVING block and slot. Local regions go
  // last, leaving one contiguous prefix for device/host MPI transfers.
  const auto less = [rank, ranks](const Region &a, const Region &b) {
    const int ap = a.peer == rank ? ranks : a.peer;
    const int bp = b.peer == rank ? ranks : b.peer;
    return std::tie(ap, a.destination, a.destination_slot) <
           std::tie(bp, b.destination, b.destination_slot);
  };
  std::sort(sends.begin(), sends.end(), less);
  std::sort(receives.begin(), receives.end(), less);
  std::vector<int> receive_index(nmb*slots, -1);
  std::vector<Peer> peers(ranks);
  for (int p = 0; p < ranks; ++p) peers[p].rank = p;
  int receive_count = 0, send_count = 0;
  for (int i = 0; i < static_cast<int>(receives.size()); ++i) {
    auto &region = receives[i];
    region.offset = receive_count;
    receive_count = AppendCount(receive_count, region.count);
    receive_index[region.block*slots+region.slot] = i;
    if (!region.local) {
      auto &peer = peers[region.peer];
      if (peer.recv_count == 0) peer.recv_offset = region.offset;
      peer.recv_count = AppendCount(peer.recv_count, region.count);
      plan.remote_recv_count = receive_count;
    }
  }
  for (auto &region : sends) {
    if (region.local) {
      const int m = region.destination-first_gid;
      if (m < 0 || m >= nmb) HaloError("local neighbor is outside the MeshBlockPack");
      const int index = receive_index[m*slots+region.destination_slot];
      if (index < 0 || receives[index].count != region.count) {
        HaloError("local send/receive boundary mismatch");
      }
      region.offset = receives[index].offset;
    } else {
      region.offset = send_count;
      send_count = AppendCount(send_count, region.count);
      auto &peer = peers[region.peer];
      if (peer.send_count == 0) peer.send_offset = region.offset;
      peer.send_count = AppendCount(peer.send_count, region.count);
    }
  }
  for (const auto &peer : peers) {
    if (peer.send_count || peer.recv_count) plan.peers.push_back(peer);
  }
  Kokkos::realloc(plan.send_regions, sends.size());
  Kokkos::realloc(plan.recv_regions, receives.size());
  auto hs = Kokkos::create_mirror_view(plan.send_regions);
  auto hr = Kokkos::create_mirror_view(plan.recv_regions);
  for (std::size_t i = 0; i < sends.size(); ++i) hs(i) = sends[i];
  for (std::size_t i = 0; i < receives.size(); ++i) hr(i) = receives[i];
  Kokkos::deep_copy(plan.send_regions, hs);
  Kokkos::deep_copy(plan.recv_regions, hr);
  Kokkos::realloc(plan.send, send_count);
  Kokkos::realloc(plan.recv, receive_count);
  if (host_staging) {
    Kokkos::realloc(plan.host_send, send_count);
    Kokkos::realloc(plan.host_recv, plan.remote_recv_count);
  }
#if MPI_PARALLEL_ENABLED
  plan.requests.resize(2*plan.peers.size(), MPI_REQUEST_NULL);
#endif
  plan.built = true;
}

}  // namespace

struct ImplicitRadiationHalo::State {
  std::vector<int> topology;
  // Allocate only the stencil layouts actually used by this solver.
  std::array<std::array<Plan, 3>, 2> halos;
  Plan flux;
  MeshBlockPack *pack = nullptr;
  MeshBoundaryValuesCC *bvals = nullptr;
  int cis = 0, cjs = 0, cks = 0, dimensions = 1;
  bool host_staging = false;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  DvceArray1D<Real> batch_send, batch_recv;
  Kokkos::View<Real *, StagingSpace> batch_host_send, batch_host_recv;
  struct BatchRequest {
    State *state;
    Plan *plan;
    static void Execute(const std::vector<void *> &requests) {
      auto &first = *static_cast<BatchRequest *>(requests.front());
      if (requests.size() == 1) { first.state->Communicate(*first.plan); return; }
      auto &s = *first.state;
      int ns = 0, nr = 0;
      bool host = s.host_staging;
      // Order each peer's payload by lane, independently of the local neighbor
      // list. Flux plans can be send-only, receive-only, or empty on a rank.
      std::map<int, Peer> peers;
      for (void *p : requests) {
        auto &r = *static_cast<BatchRequest *>(p);
        host = host || r.state->host_staging;
        for (const auto &peer : r.plan->peers) {
          auto &combined = peers.try_emplace(peer.rank, Peer{peer.rank}).first->second;
          if (peer.send_count) combined.send_count = AppendCount(
              combined.send_count, peer.send_count);
          if (peer.recv_count) combined.recv_count = AppendCount(
              combined.recv_count, peer.recv_count);
        }
      }
      if (peers.empty()) return;
      for (auto &entry : peers) {
        auto &peer = entry.second;
        peer.send_offset = ns;
        peer.recv_offset = nr;
        if (peer.send_count) ns = AppendCount(ns, peer.send_count);
        if (peer.recv_count) nr = AppendCount(nr, peer.recv_count);
      }
      const auto reserve = [](auto &v, int count) {
        if (v.extent(0) < static_cast<std::size_t>(count)) Kokkos::realloc(v, count);
      };
      reserve(s.batch_send, ns); reserve(s.batch_recv, nr);
      // Advance per-peer cursors while packing, then restore the start offsets
      // before posting MPI. Only remote regions enter the combined buffers.
      for (void *p : requests) {
        auto &plan = *static_cast<BatchRequest *>(p)->plan;
        for (const auto &peer : plan.peers) {
          if (!peer.send_count) continue;
          auto &offset = peers.at(peer.rank).send_offset;
          Kokkos::deep_copy(DevExeSpace(),
              Kokkos::subview(s.batch_send, std::make_pair(offset, offset+peer.send_count)),
              Kokkos::subview(plan.send,
                  std::make_pair(peer.send_offset, peer.send_offset+peer.send_count)));
          offset += peer.send_count;
        }
      }
      for (auto &entry : peers) entry.second.send_offset -= entry.second.send_count;
      if (host) {
        reserve(s.batch_host_send, ns); reserve(s.batch_host_recv, nr);
        if (ns) Kokkos::deep_copy(DevExeSpace(),
            Kokkos::subview(s.batch_host_send, std::make_pair(0, ns)),
            Kokkos::subview(s.batch_send, std::make_pair(0, ns)));
      }
      Kokkos::fence("radiation batched boundary producer");
      std::vector<MPI_Request> mpi(2*peers.size(), MPI_REQUEST_NULL);
      const auto *send = host ? s.batch_host_send.data() : s.batch_send.data();
      auto *recv = host ? s.batch_host_recv.data() : s.batch_recv.data();
      int posted = 0;
      for (const auto &entry : peers) {
        const auto &peer = entry.second;
        if (peer.recv_count) CheckMPI(MPI_Irecv(recv+peer.recv_offset, peer.recv_count,
            MPI_ATHENA_REAL, peer.rank, 0, s.comm, &mpi[posted++]));
      }
      for (const auto &entry : peers) {
        const auto &peer = entry.second;
        if (peer.send_count) CheckMPI(MPI_Isend(send+peer.send_offset, peer.send_count,
            MPI_ATHENA_REAL, peer.rank, 0, s.comm, &mpi[posted++]));
      }
      CheckMPI(MPI_Waitall(posted, mpi.data(), MPI_STATUSES_IGNORE));
      if (host && nr) {
        Kokkos::deep_copy(DevExeSpace(),
            Kokkos::subview(s.batch_recv, std::make_pair(0, nr)),
            Kokkos::subview(s.batch_host_recv, std::make_pair(0, nr)));
      }
      for (void *p : requests) {
        auto &plan = *static_cast<BatchRequest *>(p)->plan;
        for (const auto &peer : plan.peers) {
          if (!peer.recv_count) continue;
          auto &offset = peers.at(peer.rank).recv_offset;
          Kokkos::deep_copy(DevExeSpace(),
              Kokkos::subview(plan.recv,
                  std::make_pair(peer.recv_offset, peer.recv_offset+peer.recv_count)),
              Kokkos::subview(s.batch_recv, std::make_pair(offset, offset+peer.recv_count)));
          offset += peer.recv_count;
        }
      }
    }
  };
#endif

  void Communicate(Plan &plan) {
#if MPI_PARALLEL_ENABLED
    // Even ranks without remote faces must yield at the same point. Skipping
    // an empty flux plan lets their lanes reach reductions/AMG before peers,
    // changing the selected batch and potentially deadlocking collectives.
    if (auto *batch = RadiationGroupBatch::Active()) {
      BatchRequest request{this, &plan};
      batch->Submit(RadiationGroupBatch::Kind::Halo, &request, &BatchRequest::Execute);
      return;
    }
    if (plan.peers.empty()) return;
    if (host_staging) Kokkos::deep_copy(DevExeSpace(), plan.host_send, plan.send);
    Kokkos::fence("radiation packed boundary producer");
    // Also finish the previous unpack/H2D copy before MPI reuses its storage.
    Real *recv_pointer = host_staging ? plan.host_recv.data() : plan.recv.data();
    int requests = 0;
    for (const auto &peer : plan.peers) {
      if (peer.recv_count) {
        CheckMPI(MPI_Irecv(recv_pointer+peer.recv_offset, peer.recv_count,
                          MPI_ATHENA_REAL, peer.rank, 0, comm, &plan.requests[requests++]));
      }
    }
    const Real *send_pointer = host_staging ? plan.host_send.data() : plan.send.data();
    for (const auto &peer : plan.peers) {
      if (peer.send_count) {
        CheckMPI(MPI_Isend(send_pointer+peer.send_offset, peer.send_count,
                          MPI_ATHENA_REAL, peer.rank, 0, comm, &plan.requests[requests++]));
      }
    }
    CheckMPI(MPI_Waitall(requests, plan.requests.data(), MPI_STATUSES_IGNORE));
    if (host_staging) {
      auto remote = Kokkos::subview(plan.recv, std::make_pair(0, plan.remote_recv_count));
      Kokkos::deep_copy(DevExeSpace(), remote, plan.host_recv);
    }
#endif
  }
};

ImplicitRadiationHalo::ImplicitRadiationHalo() : state_(new State) {}

ImplicitRadiationHalo::~ImplicitRadiationHalo() {
#if MPI_PARALLEL_ENABLED
  if (state_->comm != MPI_COMM_NULL) CheckMPI(MPI_Comm_free(&state_->comm));
#endif
  delete state_;
}

void ImplicitRadiationHalo::Prepare(MeshBlockPack *pack, MeshBoundaryValuesCC *bvals,
                                    bool host_staging) {
  auto &s = *state_;
#if MPI_PARALLEL_ENABLED
  if (s.comm == MPI_COMM_NULL) CheckMPI(MPI_Comm_dup(bvals->comm_vars, &s.comm));
#endif
  const auto &ind = pack->pmesh->mb_indcs;
  std::vector<int> key{pack->nmb_thispack, pack->pmb->nnghbr, ind.ng,
                       ind.nx1, ind.nx2, ind.nx3, static_cast<int>(host_staging)};
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    key.push_back(pack->pmb->mb_gid.h_view(m));
    key.push_back(pack->pmb->mb_lev.h_view(m));
    for (int n = 0; n < pack->pmb->nnghbr; ++n) {
      const auto neighbor = pack->pmb->nghbr.h_view(m, n);
      key.insert(key.end(), {neighbor.gid, neighbor.lev, neighbor.rank, neighbor.dest});
    }
  }
  if (key == s.topology) return;
  Kokkos::fence("radiation halo topology change");
  s.halos = {};
  s.flux = Plan{};
  s.pack = pack;
  s.bvals = bvals;
  s.cis = ind.cis;
  s.cjs = ind.cjs;
  s.cks = ind.cks;
  s.dimensions = pack->pmesh->three_d ? 3 : (pack->pmesh->multi_d ? 2 : 1);
  s.topology = std::move(key);
  s.host_staging = host_staging;
}

void ImplicitRadiationHalo::Exchange(DvceArray5D<Real> &field,
                                     DvceArray5D<Real> &coarse, bool faces_only,
                                     int width) {
  auto &s = *state_;
  if (s.pack == nullptr || width < 0 || width > 2) HaloError("invalid scalar halo plan");
  auto &plan = s.halos[faces_only ? 1 : 0][width];
  if (!plan.built) {
    BuildPlan(plan, s.pack, s.bvals, faces_only, s.host_staging, false, width);
  }
  auto send_regions = plan.send_regions, recv_regions = plan.recv_regions;
  auto send = plan.send, recv = plan.recv;
  auto f = field, c = coarse;
  Kokkos::parallel_for("radiation_halo_pack",
      Kokkos::TeamPolicy<>(DevExeSpace(), send_regions.extent_int(0), Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const auto region = send_regions(team.league_rank());
    const auto b = region.bounds;
    const int ni = b.bie-b.bis+1, nj = b.bje-b.bjs+1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, region.count), [&](int p) {
      const int i = b.bis+p%ni, j = b.bjs+(p/ni)%nj, k = b.bks+p/(ni*nj);
      const Real value = region.coarse ? c(region.block, 0, k, j, i) :
                                        f(region.block, 0, k, j, i);
      if (region.local) recv(region.offset+p) = value;
      else send(region.offset+p) = value;
    });
  });
  s.Communicate(plan);
  Kokkos::parallel_for("radiation_halo_unpack",
      Kokkos::TeamPolicy<>(DevExeSpace(), recv_regions.extent_int(0), Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const auto region = recv_regions(team.league_rank());
    const auto b = region.bounds;
    const int ni = b.bie-b.bis+1, nj = b.bje-b.bjs+1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, region.count), [&](int p) {
      const int i = b.bis+p%ni, j = b.bjs+(p/ni)%nj, k = b.bks+p/(ni*nj);
      if (region.coarse) c(region.block, 0, k, j, i) = recv(region.offset+p);
      else f(region.block, 0, k, j, i) = recv(region.offset+p);
    });
  });
}

void ImplicitRadiationHalo::Reflux(DvceFaceFld5D<Real> &flux,
                                   const FaceFldOrigin origin) {
  auto &s = *state_;
  auto &plan = s.flux;
  if (!plan.built) {
    if (s.pack == nullptr) HaloError("missing flux halo topology");
    BuildPlan(plan, s.pack, s.bvals, true, s.host_staging, true);
  }
  const int cis = s.cis, cjs = s.cjs, cks = s.cks, dimensions = s.dimensions;
  auto sends = plan.send_regions, receives = plan.recv_regions;
  auto send = plan.send, recv = plan.recv;
  auto f1 = MakeBandView(flux.x1f, origin);
  auto f2 = MakeBandView(flux.x2f, origin);
  auto f3 = MakeBandView(flux.x3f, origin);
  // Only coarse/fine face regions launch teams. Match the legacy restriction's
  // summation order exactly, including its 1D and 2D special cases.
  Kokkos::parallel_for("radiation_flux_pack",
      Kokkos::TeamPolicy<>(DevExeSpace(), sends.extent_int(0), Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const auto r = sends(team.league_rank());
    const auto b = r.bounds;
    const int ni = b.bie-b.bis+1, nj = b.bje-b.bjs+1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, r.count), [&](int p) {
      const int i = 2*(b.bis+p%ni)-cis;
      const int j = dimensions > 1 ? 2*(b.bjs+(p/ni)%nj)-cjs : 0;
      const int k = dimensions > 2 ? 2*(b.bks+p/(ni*nj))-cks : 0;
      Real value;
      if (r.slot < 8) {
        if (dimensions == 1) value = f1(r.block, 0, 0, 0, i);
        else if (dimensions == 2) {
          value = 0.5*(f1(r.block, 0, 0, j, i)+f1(r.block, 0, 0, j+1, i));
        } else {
          value = 0.25*(f1(r.block, 0, k, j, i)+f1(r.block, 0, k, j+1, i)+
                        f1(r.block, 0, k+1, j, i)+f1(r.block, 0, k+1, j+1, i));
        }
      } else if (r.slot < 16) {
        if (dimensions == 2) {
          value = 0.5*(f2(r.block, 0, 0, j, i)+f2(r.block, 0, 0, j, i+1));
        } else {
          value = 0.25*(f2(r.block, 0, k, j, i)+f2(r.block, 0, k, j, i+1)+
                        f2(r.block, 0, k+1, j, i)+f2(r.block, 0, k+1, j, i+1));
        }
      } else {
        value = 0.25*(f3(r.block, 0, k, j, i)+f3(r.block, 0, k, j, i+1)+
                      f3(r.block, 0, k, j+1, i)+f3(r.block, 0, k, j+1, i+1));
      }
      if (r.local) recv(r.offset+p) = value;
      else send(r.offset+p) = value;
    });
  });
  s.Communicate(plan);
  Kokkos::parallel_for("radiation_flux_unpack",
      Kokkos::TeamPolicy<>(DevExeSpace(), receives.extent_int(0), Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t team) {
    const auto r = receives(team.league_rank());
    const auto b = r.bounds;
    const int ni = b.bie-b.bis+1, nj = b.bje-b.bjs+1;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, r.count), [&](int p) {
      const int i = b.bis+p%ni, j = b.bjs+(p/ni)%nj, k = b.bks+p/(ni*nj);
      if (r.slot < 8) f1(r.block, 0, k, j, i) = recv(r.offset+p);
      else if (r.slot < 16) f2(r.block, 0, k, j, i) = recv(r.offset+p);
      else f3(r.block, 0, k, j, i) = recv(r.offset+p);
    });
  });
}

}  // namespace two_temperature
