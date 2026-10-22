//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Leaf-block ownership and moments on uniform, static, and adaptive meshes.
//========================================================================================

#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_memory.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "fast_electrons/fast_electron_physics.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"

namespace fast_electrons {

void FastElectrons::RefreshMesh() {
  memory::Time(pmy_pack_->pmesh->time);
  memory::Context("regrid", device_packets_.extent(0));
  DownloadPackets();
  auto mesh = pmy_pack_->pmesh;
  const auto domain = mesh->mesh_size;
  const auto ind = mesh->mb_indcs;
  std::vector<laser::LaserBlockInfo> blocks(mesh->nmb_total);
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  const Real lower[3] = {domain.x1min, domain.x2min, domain.x3min};
  const Real upper[3] = {domain.x1max, domain.x2max, domain.x3max};
  const int nc[3] = {ind.nx1, ind.nx2, ind.nx3};
  for (int g = 0; g < mesh->nmb_total; ++g) {
    const auto loc = mesh->lloc_eachmb[g];
    const int level = loc.level-mesh->root_level;
    const int nb[3] = {mesh->nmb_rootx1 << level, mesh->nmb_rootx2 << level,
                       mesh->nmb_rootx3 << level};
    const std::int64_t lx[3] = {loc.lx1, loc.lx2, loc.lx3};
    Real lo[3], hi[3], dx[3];
    for (int d = 0; d < 3; ++d) {
      lo[d] = lower[d]; hi[d] = upper[d];
      if (d < ndim) {
        if (lx[d] != 0) lo[d] = LeftEdgeX(lx[d], nb[d], lower[d], upper[d]);
        if (lx[d] != nb[d]-1) hi[d] = LeftEdgeX(lx[d]+1, nb[d], lower[d], upper[d]);
      }
      dx[d] = (hi[d]-lo[d])/nc[d];
    }
    blocks[g] = {lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                 dx[0], dx[1], dx[2], g, mesh->rank_eachmb[g]};
  }
  const auto tree = laser::BuildBlockTree(blocks, mesh->multi_d, mesh->three_d);
  Kokkos::realloc(blocks_, blocks.size());
  Kokkos::realloc(tree_, tree.size());
  host_blocks_ = Kokkos::create_mirror_view(blocks_);
  host_tree_ = Kokkos::create_mirror_view(tree_);
  for (std::size_t g = 0; g < blocks.size(); ++g) host_blocks_(g) = blocks[g];
  for (std::size_t n = 0; n < tree.size(); ++n) host_tree_(n) = tree[n];
  Kokkos::deep_copy(blocks_, host_blocks_);
  Kokkos::deep_copy(tree_, host_tree_);
  source_ready_ = false;
  for (auto &p : packets_) {
    LocatePacket(p, host_tree_, host_blocks_, domain, ind, ndim, periodic_);
    if (p.status != 0) Kokkos::abort("Invalid fast-electron ownership after regrid");
  }
  ExchangePackets();
  UploadPackets();
  DepositMoments();
  // During construction primitives and the refinement operator are not ready yet.
  // Regrids reach here after the driver has restored the fluid state on the new mesh.
  if (electromagnetic_ && mesh->pmr != nullptr) BuildFields();
}

int FastElectrons::Owner(const ElectronPacket &p) const {
  return host_blocks_(p.gid).rank;
}

void FastElectrons::UploadPackets() {
  memory::Time(pmy_pack_->pmesh->time);
  memory::Context("upload", packets_.size());
  ReservePackets(packets_.size());
  device_packets_ = Kokkos::subview(packet_storage_,
      std::make_pair(std::size_t(0), packets_.size()));
  if (!packets_.empty()) {
    Kokkos::View<const ElectronPacket*, HostMemSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(
        packets_.data(), packets_.size());
    Kokkos::deep_copy(device_packets_, host);
  }
}

void FastElectrons::ReservePackets(std::size_t count) {
  if (count > static_cast<std::size_t>(max_packets_)) {
    Kokkos::abort("### FATAL ERROR: <fast_electrons> max_packets exceeded on a rank");
  }
  if (count <= packet_storage_.extent(0)) return;
  const auto old_count = device_packets_.extent(0);
  const auto capacity = std::min(static_cast<std::size_t>(max_packets_),
      std::max(count, std::max(std::size_t(256),
                              packet_storage_.extent(0)*2)));
  memory::Preflight(capacity*sizeof(ElectronPacket), "particle storage growth");
  DvceArray1D<ElectronPacket> storage(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fast-electron-packet-storage"),
      capacity);
  auto live = Kokkos::subview(storage, std::make_pair(std::size_t(0), old_count));
  Kokkos::deep_copy(live, device_packets_);
  packet_storage_ = storage;
  device_packets_ = live;
}

void FastElectrons::ReclaimPacketMemory() {
  // Transport has already discarded thermalized/escaped packets. Only the live
  // prefix is meaningful here; the other buffers contain no pending work once
  // Advance has finished all field substeps and MPI transport waves.
  const std::size_t live = device_packets_.extent(0);
  const auto target = live == 0 ? std::size_t(0) :
      std::min(static_cast<std::size_t>(max_packets_),
               std::max(std::size_t(256), 2*live));
  const auto oversized = [live, target](const DvceArray1D<ElectronPacket> &buffer) {
    // A factor-of-four trigger and factor-of-two headroom avoid shrinking on
    // ordinary fluctuations. An empty population releases all packet buffers.
    return buffer.extent(0) > target && live <= buffer.extent(0)/4;
  };
  const bool shrink_live = oversized(packet_storage_);
  const bool release_scratch = oversized(packet_scratch_);
  const bool release_work0 = oversized(transport_work_[0]);
  const bool release_work1 = oversized(transport_work_[1]);
  const bool release_exports = oversized(transport_exports_);
  if (!(shrink_live || release_scratch || release_work0 || release_work1 ||
        release_exports)) return;
  const auto capacity = [&]() {
    return packet_storage_.extent(0)+packet_scratch_.extent(0)+
           transport_work_[0].extent(0)+transport_work_[1].extent(0)+
           transport_exports_.extent(0);
  };
  const auto before = capacity();
  Kokkos::fence("fast-electron packet memory reclamation");
  // Release disposable scratch before allocating a smaller live buffer, keeping
  // the temporary old/new overlap as small as possible. Never copy dead slots.
  if (release_scratch) packet_scratch_ = DvceArray1D<ElectronPacket>();
  if (release_work0) transport_work_[0] = DvceArray1D<ElectronPacket>();
  if (release_work1) transport_work_[1] = DvceArray1D<ElectronPacket>();
  if (release_exports) transport_exports_ = DvceArray1D<ElectronPacket>();
  if (shrink_live) {
    if (live == 0) {
      device_packets_ = DvceArray1D<ElectronPacket>();
      packet_storage_ = DvceArray1D<ElectronPacket>();
    } else {
      memory::Preflight(target*sizeof(ElectronPacket), "particle storage shrink");
      DvceArray1D<ElectronPacket> storage(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "fast-electron-packet-storage"),
          target);
      auto survivors = Kokkos::subview(storage, std::make_pair(std::size_t(0), live));
      Kokkos::deep_copy(survivors, device_packets_);
      device_packets_ = survivors;
      packet_storage_ = storage;
    }
  }
  if (report_) {
    const auto after = capacity();
    std::ostringstream out;
    out << "# fast_electrons_memory rank=" << global_variable::my_rank
        << " live_packets=" << live << " reserved_before=" << before
        << " reserved_after=" << after
        << " released_bytes=" << (before-after)*sizeof(ElectronPacket);
    std::cout << out.str() << std::endl;
  }
}

void FastElectrons::DownloadPackets() {
  const auto count = device_packets_.extent(0);
  // The vector is the host population. Copy into it directly, rather than keeping
  // an additional device-capacity-sized host mirror and copying every packet twice.
  // No old host values survive this download, so release before growing to avoid
  // a transient old-plus-new allocation for large checkpoint/AMR populations.
  if (count == 0 || count > packets_.capacity()) {
    std::vector<ElectronPacket>().swap(packets_);
  }
  packets_.resize(count);
  if (count > 0) {
    Kokkos::View<ElectronPacket*, HostMemSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(packets_.data(), count);
    Kokkos::deep_copy(host, device_packets_);
  }
}

int FastElectrons::PrepareDeposits() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  const int nmb = pmy_pack_->nmb_thispack;
  const std::size_t cells = static_cast<std::size_t>(nmb)*ind.nx1*ind.nx2*ind.nx3;
  // Replication pays for itself only for a dense population. Bound the scratch
  // footprint (five moments, 32 replicas) to 16 MiB even on large AMR packs.
  constexpr int replicas = 32;
  if (device_packets_.extent(0) >= 4096 && device_packets_.extent(0) > 8*cells &&
      cells*5*replicas*sizeof(Real) <= 16*1024*1024) {
    if (deposit_scratch_.extent_int(0) != nmb) {
      memory::Preflight(cells*5*replicas*sizeof(Real), "deposition scratch");
      Kokkos::realloc(Kokkos::WithoutInitializing, deposit_scratch_,
                      nmb, 5, ind.nx3, ind.nx2, ind.nx1, replicas);
    }
    Kokkos::deep_copy(deposit_scratch_, 0.0);
    return replicas;
  }
#endif
  return 1;
}

void FastElectrons::DepositMoments(Real duration) {
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  auto data = cell_data;
  auto current = current_;
  auto packet = device_packets_;
  auto blocks = blocks_;
  const int first_gid = pmy_pack_->gids;
  const Real volume_scale = length_cgs_*length_cgs_*length_cgs_;
  const Real energy_scale = energy_cgs_;
  const Real jscale = -kElectronChargeCgs*kLightSpeedCgs/(current_unit_*volume_scale);
  const Real c_code = kLightSpeedCgs*time_cgs_/length_cgs_;
  const bool em = electromagnetic_;
  const int replicas = PrepareDeposits();
  auto scratch = deposit_scratch_;
  DvceArray5D<Real> w;
  if (em) w = pmy_pack_->pmhd->w0;
  if (em) Kokkos::deep_copy(current, 0.0);
  par_for("fe_clear_moments", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    data(m, 2, k, j, i) = 0.0;
    data(m, 3, k, j, i) = 0.0;
    for (int d = 0; d < 3; ++d) data(m, 5+d, k, j, i) = 0.0;
  });
  par_for("fe_moments", DevExeSpace(), 0, static_cast<int>(packet.extent(0))-1,
  KOKKOS_LAMBDA(int q) {
    const auto p = packet(q);
    const auto block = blocks(p.gid);
    const Real volume = block.dx1*block.dx2*block.dx3;
    const int m = p.gid-first_gid;
    const int i = p.cell[0]+ind.is, j = p.cell[1]+ind.js, k = p.cell[2]+ind.ks;
    const int r = q & (replicas-1);
    Real *number = replicas > 1 ? &scratch(m, 0, p.cell[2], p.cell[1], p.cell[0], r)
                                : &data(m, 2, k, j, i);
    Real *energy = replicas > 1 ? &scratch(m, 1, p.cell[2], p.cell[1], p.cell[0], r)
                                : &data(m, 3, k, j, i);
    Kokkos::atomic_add(number, p.weight/(volume*volume_scale));
    Kokkos::atomic_add(energy, p.weight*p.energy/(energy_scale*volume));
    if (em) {
      // Newly emitted packets contribute only their fraction of this field interval.
      const Real fraction = duration > 0.0 ? fmin(p.remaining/duration, 1.0) : 1.0;
      const Real scale = jscale*p.weight*fraction/volume;
      for (int d = 0; d < 3; ++d) {
        const Real value = scale*(Beta(p.energy)*p.n[d]-w(m, IVX+d, k, j, i)/c_code);
        if (replicas > 1) {
          Kokkos::atomic_add(&scratch(m, 2+d, p.cell[2], p.cell[1], p.cell[0], r), value);
        } else {
          Kokkos::atomic_add(&current(m, d, k, j, i), value);
          Kokkos::atomic_add(&data(m, 5+d, k, j, i), value);
        }
      }
    }
  });
  if (replicas > 1) {
    par_for("fe_merge_moments", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
            ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int v = 0; v < (em ? 5 : 2); ++v) {
        Real sum = 0.0;
        for (int r = 0; r < replicas; ++r) {
          sum += scratch(m, v, k-ind.ks, j-ind.js, i-ind.is, r);
        }
        if (v < 2) data(m, 2+v, k, j, i) = sum;
        else {
          current(m, v-2, k, j, i) = sum;
          data(m, 3+v, k, j, i) = sum;
        }
      }
    });
  }
}

}  // namespace fast_electrons
