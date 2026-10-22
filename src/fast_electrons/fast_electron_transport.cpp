//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! MPI transport waves visit only unfinished or newly received particles.
//========================================================================================

#include <algorithm>
#include <limits>
#include <utility>

#include "athena.hpp"
#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_memory.hpp"
#include "globals.hpp"

namespace fast_electrons {
namespace {
struct WaveOffsets {
  int finished, work, outgoing;
  KOKKOS_INLINE_FUNCTION WaveOffsets() : finished(0), work(0), outgoing(0) {}
  KOKKOS_INLINE_FUNCTION void operator+=(const WaveOffsets &rhs) {
    finished += rhs.finished;
    work += rhs.work;
    outgoing += rhs.outgoing;
  }
};

void GrowPackets(DvceArray1D<ElectronPacket> &storage, std::size_t needed,
                 std::size_t used, int limit, const char *label) {
  if (needed > static_cast<std::size_t>(limit)) {
    Kokkos::abort("Fast-electron transport exceeds max_packets on a rank");
  }
  if (needed <= storage.extent(0)) return;
  const auto capacity = std::min(static_cast<std::size_t>(limit),
      std::max(needed, std::max(std::size_t(256), 2*storage.extent(0))));
  memory::Preflight(capacity*sizeof(ElectronPacket), label);
  DvceArray1D<ElectronPacket> larger(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), capacity);
  if (used > 0) {
    auto old_live = Kokkos::subview(storage, std::make_pair(std::size_t(0), used));
    auto new_live = Kokkos::subview(larger, std::make_pair(std::size_t(0), used));
    Kokkos::deep_copy(new_live, old_live);
  }
  storage = larger;
}
}  // namespace

void FastElectrons::TransportDistributed() {
  // The initial population is read once. Completed packets accumulate in the
  // other full-size buffer; only unfinished local packets and MPI arrivals enter
  // subsequent waves. Two small reusable queues alternate to avoid aliasing.
  const auto initial_storage = packet_storage_;
  auto completed_storage = packet_scratch_;
  std::size_t completed = 0;
  auto blocks = blocks_;
  const int rank = global_variable::my_rank;
  int global_work = 1;
  for (int wave = 0; global_work != 0; ++wave) {
    if (wave >= max_waves_) Kokkos::abort("Fast-electron max_transport_waves exceeded");
    memory::Context("trace_prepare", completed+device_packets_.extent(0));
    TraceLocal();
    auto packets = device_packets_;
    int finished = 0, work = 0, outgoing = 0, failed = 0;
    Real escaped = 0.0;
    Kokkos::parallel_reduce("fe_wave_status",
        Kokkos::RangePolicy<DevExeSpace>(0, packets.extent_int(0)),
    KOKKOS_LAMBDA(int q, int &done, int &todo, int &send, int &bad, Real &loss) {
      const auto p = packets(q);
      if (p.status == 0) {
        if (blocks(p.gid).rank != rank) ++send;
        else if (p.remaining > 0.0) ++todo;
        else ++done;
      } else if (p.status == 2) {
        loss += p.weight*p.energy;
      } else if (p.status == 3) {
        bad = 1;
      }
    }, Kokkos::Sum<int>(finished), Kokkos::Sum<int>(work),
       Kokkos::Sum<int>(outgoing), Kokkos::Max<int>(failed), Kokkos::Sum<Real>(escaped));
    if (failed > 0) Kokkos::abort("Invalid fast-electron plasma state or particle timestep");
    escaped_ += escaped;
    memory::Context("partition", completed+finished+work+outgoing);
    int flags[2] = {work > 0 || outgoing > 0 ? 1 : 0, outgoing > 0 ? 1 : 0};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    global_work = flags[0];
    // Collisionless local populations often finish without moving or losing a
    // packet. Preserve the original storage/order with no compaction or copy.
    if (wave == 0 && global_work == 0 && finished == packets.extent_int(0)) return;

    GrowPackets(completed_storage, completed+finished, completed, max_packets_,
                "fe-completed-packets");
    auto &next_storage = transport_work_[wave%2];
    GrowPackets(next_storage, work, 0, max_packets_, "fe-pending-packets");
    GrowPackets(transport_exports_, outgoing, 0, max_packets_, "fe-export-packets");
    auto next = next_storage;
    auto done = completed_storage;
    auto exports = transport_exports_;
    const auto base = completed;
    // One stable partition replaces separate survivor compaction and migrant
    // partition scans. Each class preserves packet RNG state and relative order.
    Kokkos::parallel_scan("fe_partition_wave",
        Kokkos::RangePolicy<DevExeSpace>(0, packets.extent_int(0)),
    KOKKOS_LAMBDA(int q, WaveOffsets &offset, bool final) {
      const auto p = packets(q);
      if (p.status != 0) return;
      if (blocks(p.gid).rank != rank) {
        if (final) exports(offset.outgoing) = p;
        ++offset.outgoing;
      } else if (p.remaining > 0.0) {
        if (final) next(offset.work) = p;
        ++offset.work;
      } else {
        if (final) done(base+offset.finished) = p;
        ++offset.finished;
      }
    });
    completed += finished;
    std::size_t next_count = work;
    if (flags[1]) {
      packets_.resize(outgoing);
      if (outgoing > 0) {
        auto live_exports = Kokkos::subview(transport_exports_,
            std::make_pair(std::size_t(0), static_cast<std::size_t>(outgoing)));
        Kokkos::View<ElectronPacket*, HostMemSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(packets_.data(), outgoing);
        Kokkos::deep_copy(host, live_exports);
      }
      // Every rank participates, including empty receivers and ranks whose
      // original population has already finished the interval.
      ExchangePackets();
      next_count += packets_.size();
      memory::Context("received", completed+next_count);
      if (completed+next_count > static_cast<std::size_t>(max_packets_)) {
        Kokkos::abort("Fast-electron transport exceeds max_packets on a rank");
      }
      GrowPackets(next_storage, next_count, work, max_packets_, "fe-pending-packets");
      if (!packets_.empty()) {
        auto tail = Kokkos::subview(next_storage,
            std::make_pair(static_cast<std::size_t>(work), next_count));
        Kokkos::View<const ElectronPacket*, HostMemSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(
            packets_.data(), packets_.size());
        Kokkos::deep_copy(tail, host);
      }
    }
    device_packets_ = Kokkos::subview(next_storage,
        std::make_pair(std::size_t(0), next_count));
  }
  packet_storage_ = completed_storage;
  packet_scratch_ = initial_storage;
  device_packets_ = Kokkos::subview(packet_storage_,
      std::make_pair(std::size_t(0), completed));
}
}  // namespace fast_electrons
