// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
// The leaf traversal is shared with fast-electron transport.
#include <algorithm>
#include <utility>
#include <vector>
#include "nuclear/dt_fusion.hpp"
#include "nuclear/dt_cell.hpp"
#include "coordinates/cell_locations.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
namespace nuclear {
using fast_electrons::LocatePacket;
void DTFusion::RefreshMesh() {
  auto mesh = pack_->pmesh;
  const auto domain = mesh->mesh_size;
  const auto ind = mesh->mb_indcs;
  const int nmb = std::max(pack_->nmb_thispack, mesh->nmb_maxperrank);
  const int ni = ind.nx1+2*ind.ng;
  const int nj = ind.nx2 > 1 ? ind.nx2+2*ind.ng : 1;
  const int nk = ind.nx3 > 1 ? ind.nx3+2*ind.ng : 1;
  Kokkos::realloc(cell_data, nmb, ncell_data, nk, nj, ni);
  Kokkos::realloc(feedback_, nmb, 5, nk, nj, ni);
  Kokkos::realloc(births_, nmb, nk, nj, ni);
  Kokkos::realloc(stopping_, nmb, nk, nj, ni);
  Kokkos::realloc(birth_offsets_, nmb*ind.nx1*ind.nx2*ind.nx3);
  Kokkos::realloc(send_counts_, global_variable::nranks);
  Kokkos::realloc(send_offsets_, global_variable::nranks);
  Kokkos::deep_copy(cell_data, 0.0);
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
  auto packets = device_packets_;
  auto device_tree = tree_; auto device_blocks = blocks_;
  const bool periodic = periodic_;
  par_for("dt_relocate", DevExeSpace(), 0, packet_count_-1,
  KOKKOS_LAMBDA(int q) {
    auto p = packets(q);
    LocatePacket(p, device_tree, device_blocks, domain, ind, ndim, periodic);
    if (p.status != 0) Kokkos::abort("Invalid alpha ownership after regrid");
    packets(q) = p;
  });
  (void)RoutePackets();
}

void DTFusion::ReservePackets(int count) {
  if (count < 0 || count > max_packets_) Kokkos::abort("D-T max_packets exceeded");
  const int old_capacity = device_packets_.extent_int(0);
  if (count <= old_capacity) return;
  const int capacity = std::min<std::int64_t>(max_packets_,
      std::max<std::int64_t>(count, std::max<std::int64_t>(1024,
          old_capacity+static_cast<std::int64_t>(old_capacity)/2)));
  DvceArray1D<AlphaPacket> next(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "dt_alpha_packets"), capacity);
  if (packet_count_ > 0) {
    const auto active = std::make_pair(0, packet_count_);
    Kokkos::deep_copy(Kokkos::subview(next, active),
                     Kokkos::subview(device_packets_, active));
  }
  device_packets_ = next;
}

std::vector<AlphaPacket> DTFusion::DownloadPackets() const {
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
      Kokkos::subview(device_packets_, std::make_pair(0, packet_count_)));
  std::vector<AlphaPacket> result(packet_count_);
  for (int q = 0; q < packet_count_; ++q) result[q] = host(q);
  return result;
}

void DTFusion::RestorePackets(const std::vector<AlphaPacket> &packets) {
  if (packets.size() > static_cast<std::size_t>(max_packets_)) {
    Kokkos::abort("D-T restart exceeds max_packets");
  }
  packet_count_ = 0;
  ReservePackets(static_cast<int>(packets.size()));
  HostArray1D<AlphaPacket> host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "dt_restart_packets"), packets.size());
  for (std::size_t q = 0; q < packets.size(); ++q) host(q) = packets[q];
  packet_count_ = packets.size();
  Kokkos::deep_copy(Kokkos::subview(device_packets_, std::make_pair(0, packet_count_)), host);
}


void DTFusion::DepositMoments() {
  const auto ind = pack_->pmesh->mb_indcs;
  auto data = cell_data; auto stopping = stopping_;
  auto u = mhd_ ? pack_->pmhd->u0 : pack_->phydro->u0;
  const auto mix = mixture_;
  auto packet = device_packets_; auto blocks = blocks_;
  const int first = pack_->gids, ii = iion_, ie = iele_;
  const double volume_unit = length_*length_*length_, energy_unit = energy_;
  const double density_unit = density_, pressure_unit = energy_/volume_unit;
  const double logl = log_lambda_, cutoff = cutoff_, thermal = thermal_factor_;
  const bool collisions = collisions_;
  Kokkos::deep_copy(stopping, 0.0);
  par_for("dt_clear_moments", DevExeSpace(), 0, pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    data(m,1,k,j,i) = 0.0; data(m,2,k,j,i) = 0.0;
  });
  par_for("dt_moments", DevExeSpace(), 0, packet_count_-1,
  KOKKOS_LAMBDA(int q) {
    const auto p = packet(q); const auto b = blocks(p.gid);
    const double volume = b.dx1*b.dx2*b.dx3;
    const int m = p.gid-first, i = p.cell[0]+ind.is;
    const int j = p.cell[1]+ind.js, k = p.cell[2]+ind.ks;
    Kokkos::atomic_add(&data(m,1,k,j,i), Real(p.weight/(volume*volume_unit)));
    Kokkos::atomic_add(&data(m,2,k,j,i), Real(p.weight*p.energy/(volume*energy_unit)));
    if (collisions) {
      const auto composition = mix.CompositionFromConserved(u,m,k,j,i);
      const double rho = u(m,IDN,k,j,i);
      double ti, te;
      Temperatures(mix, composition, rho, u(m,ii,k,j,i), u(m,ie,k,j,i),
                   pressure_unit, density_unit, ti, te);
      // Use a positive suprathermal probe even for a packet about to thermalize.
      const double e = fmax(p.energy, fmax(cutoff, thermal*kBoltzmann*fmax(ti,te)));
      const auto rate = Collision(mix, composition, rho*density_unit, ti,te,e,logl);
      Kokkos::atomic_add(&stopping(m,k,j,i),
          p.weight*(rate.ion+rate.electron)/(volume*volume_unit));
    }
  });
}
}  // namespace nuclear
