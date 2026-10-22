// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
#include <algorithm>
#include <limits>
#include <utility>

#include "bvals/bvals.hpp"
#include "globals.hpp"

#if MPI_PARALLEL_ENABLED
namespace {
void CheckMPI(int error) {
  if (error != MPI_SUCCESS) Kokkos::abort("MPI error in batched boundary exchange");
}
}  // namespace

void BoundaryMPIBatch::SetPlan(Plan &plan, std::vector<Segment> segments) {
  // Tags identify the receiving block/face. Sorting by these tags gives identical
  // wire order on both ranks even when local block and neighbor orders differ.
  std::sort(segments.begin(), segments.end(), [](const Segment &a, const Segment &b) {
    return a.peer < b.peer || (a.peer == b.peer && a.tag < b.tag);
  });
  if (segments == plan.segments) return;
  plan.segments = std::move(segments);
  plan.peers.clear();
  std::size_t total = 0;
  for (const auto &s : plan.segments) {
    if (s.count < 0 || total+s.count > std::numeric_limits<int>::max()) {
      Kokkos::abort("Batched boundary buffer exceeds MPI count limit");
    }
    if (plan.peers.empty() || plan.peers.back().rank != s.peer) {
      plan.peers.push_back({s.peer, 0, static_cast<int>(total)});
    }
    plan.peers.back().count += s.count;
    total += s.count;
  }
  plan.requests.assign(plan.peers.size(), MPI_REQUEST_NULL);
  Kokkos::realloc(Kokkos::WithoutInitializing, plan.buffer, total);
  Kokkos::realloc(Kokkos::WithoutInitializing, plan.spans, plan.segments.size());
  auto host = Kokkos::create_mirror_view(plan.spans);
  int offset = 0;
  for (std::size_t q = 0; q < plan.segments.size(); ++q) {
    const auto &s = plan.segments[q];
    host(q) = {s.data, s.count, offset};
    offset += s.count;
  }
  Kokkos::deep_copy(plan.spans, host);
}

void BoundaryMPIBatch::PostRecv(std::vector<Segment> segments, MPI_Comm comm) {
  SetPlan(recv_, std::move(segments));
  // The previous scatter must finish before MPI can overwrite its source buffer.
  // This also orders local buffer copies from preceding task-list stages.
  Kokkos::fence();
  for (std::size_t q = 0; q < recv_.peers.size(); ++q) {
    const auto &p = recv_.peers[q];
    CheckMPI(MPI_Irecv(recv_.buffer.data()+p.offset, p.count, MPI_ATHENA_REAL,
                      p.rank, 0, comm, &recv_.requests[q]));
  }
  unpacked_ = false;
}

void BoundaryMPIBatch::PackAndSend(std::vector<Segment> segments, MPI_Comm comm) {
  SetPlan(send_, std::move(segments));
  auto spans = send_.spans;
  auto buffer = send_.buffer;
  if (spans.extent(0) > 0) {
    Kokkos::parallel_for("batch_halo_gather",
        Kokkos::TeamPolicy<DevExeSpace>(spans.extent_int(0), Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t team) {
      const auto s = spans(team.league_rank());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, s.count), [&](const int j) {
        buffer(s.offset+j) = s.data[j];
      });
    });
  }
  Kokkos::fence();
  for (std::size_t q = 0; q < send_.peers.size(); ++q) {
    const auto &p = send_.peers[q];
    CheckMPI(MPI_Isend(send_.buffer.data()+p.offset, p.count, MPI_ATHENA_REAL,
                      p.rank, 0, comm, &send_.requests[q]));
  }
}

bool BoundaryMPIBatch::Receive(bool wait) {
  if (unpacked_) return true;
  const int count = recv_.requests.size();
  if (wait) {
    CheckMPI(MPI_Waitall(count, recv_.requests.data(), MPI_STATUSES_IGNORE));
  } else {
    int ready = 0;
    CheckMPI(MPI_Testall(count, recv_.requests.data(), &ready, MPI_STATUSES_IGNORE));
    if (!ready) return false;
  }
  auto spans = recv_.spans;
  auto buffer = recv_.buffer;
  if (spans.extent(0) > 0) {
    Kokkos::parallel_for("batch_halo_scatter",
        Kokkos::TeamPolicy<DevExeSpace>(spans.extent_int(0), Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t team) {
      const auto s = spans(team.league_rank());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, s.count), [&](const int j) {
        s.data[j] = buffer(s.offset+j);
      });
    });
  }
  unpacked_ = true;
  return true;
}

void BoundaryMPIBatch::WaitSend() {
  CheckMPI(MPI_Waitall(send_.requests.size(), send_.requests.data(), MPI_STATUSES_IGNORE));
}

std::vector<BoundaryMPIBatch::Segment> MeshBoundaryValues::BatchSegments(
    bool sending, bool flux, int nvars) {
  std::vector<BoundaryMPIBatch::Segment> segments;
  const auto &nghbr = pmy_pack->pmb->nghbr.h_view;
  auto &buffers = sending ? sendbuf : recvbuf;
  for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
    for (int n = 0; n < pmy_pack->pmb->nnghbr; ++n) {
      const auto &nb = nghbr(m,n);
      if (nb.gid < 0 || nb.rank == global_variable::my_rank || (flux && n >= 48)) continue;
      const int count = nvars*(flux ? buffers[n].iflxs_ndat : buffers[n].isame_ndat);
      if (count == 0) continue;
      const int lid = sending ? nb.gid-pmy_pack->pmesh->gids_eachrank[nb.rank] : m;
      const int tag = CreateBvals_MPI_Tag(lid, sending ? nb.dest : n);
      auto ptr = Kokkos::subview(flux ? buffers[n].flux : buffers[n].vars,
                                 m, Kokkos::ALL);
      segments.push_back({ptr.data(), count, nb.rank, tag});
    }
  }
  return segments;
}
#endif

void MeshBoundaryValues::EnableUniformBatch() {
#if MPI_PARALLEL_ENABLED
  // Refinement uses the original per-neighbor protocol, including its coarse/fine
  // and Z4c layouts. This optimization is explicitly enabled by the hybrid solver.
  batch_enabled_ = global_variable::nranks > 1 && !is_z4c_ &&
      !pmy_pack->pmesh->multilevel && !pmy_pack->pmesh->adaptive;
#endif
}
