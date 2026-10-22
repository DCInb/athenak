// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
// Finite-speed alpha transport with face-limited Coulomb collision substeps.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "nuclear/dt_fusion.hpp"
#include "nuclear/dt_cell.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "fast_electrons/fast_electron_physics.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace nuclear {
using fast_electrons::LocatePacket;
namespace {
struct PacketStatistics {
  int kept, work, failed;
  double escaped;
  KOKKOS_INLINE_FUNCTION PacketStatistics() : kept(0), work(0), failed(0), escaped(0.0) {}
  KOKKOS_INLINE_FUNCTION void operator+=(const PacketStatistics &other) {
    kept += other.kept; work += other.work; failed += other.failed;
    escaped += other.escaped;
  }
};

// Scratch and MPI staging buffers need capacity, not initialized packet objects.
// Their contents are overwritten before use and retained across transport waves.
template <typename View>
void GrowBuffer(View &view, int needed, int limit, const char *label) {
  if (needed <= view.extent_int(0)) return;
  const auto capacity = std::min<std::int64_t>(limit, std::max<std::int64_t>(needed,
      view.extent_int(0)+static_cast<std::int64_t>(view.extent_int(0))/2));
  view = View(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(label)), capacity);
}

void TransportError(const std::string &message) {
  Kokkos::abort(("D-T alpha transport: "+message).c_str());
}
}
void DTFusion::Transport(Real dt) {
  Kokkos::deep_copy(feedback_, 0.0);
  auto packets = device_packets_;
  par_for("dt_reset_flight", DevExeSpace(), 0, packet_count_-1,
  KOKKOS_LAMBDA(int q) { packets(q).remaining = dt; });
  int global_work = 1;
  for (int wave = 0; global_work != 0; ++wave) {
    if (wave >= max_waves_) {
      TransportError("max_transport_waves exceeded; no energy was discarded");
    }
    TraceLocal();
    const int local_work = RoutePackets();
    global_work = local_work;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local_work, &global_work, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
}

int DTFusion::RoutePackets() {
  const int nranks = global_variable::nranks;
  const int rank = global_variable::my_rank;
  auto packets = device_packets_; auto blocks = blocks_; auto counts = send_counts_;
  Kokkos::deep_copy(counts, 0);
  PacketStatistics stats;
  Kokkos::parallel_reduce("dt_classify_packets",
      Kokkos::RangePolicy<DevExeSpace>(0,packet_count_),
  KOKKOS_LAMBDA(int q, PacketStatistics &sum) {
    const auto p = packets(q);
    if (p.status == 0) {
      if (p.gid < 0 || p.gid >= blocks.extent_int(0)) { ++sum.failed; return; }
      const int owner = blocks(p.gid).rank;
      if (owner < 0 || owner >= nranks) { ++sum.failed; return; }
      if (owner == rank) ++sum.kept;
      else Kokkos::atomic_inc(&counts(owner));
      if (p.remaining > 0.0) ++sum.work;
    } else if (p.status == 2) {
      sum.escaped += p.weight*p.energy;
    } else if (p.status != 1) {
      ++sum.failed;
    }
  }, stats);
  if (stats.failed) TransportError("invalid plasma state or unresolvable particle timestep");
  ledger_[3] += stats.escaped;

  int received = 0;
#if MPI_PARALLEL_ENABLED
  std::vector<int> send_count(nranks,0), recv_count(nranks,0),
                   send_offset(nranks,0), recv_offset(nranks,0);
  if (nranks > 1) {
    auto host_counts = Kokkos::create_mirror_view_and_copy(HostMemSpace(),counts);
    auto host_offsets = Kokkos::create_mirror_view(send_offsets_);
    std::size_t send_size = 0;
    for (int r = 0; r < nranks; ++r) {
      const auto bytes = static_cast<std::size_t>(host_counts(r))*sizeof(AlphaPacket);
      if (send_size+bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TransportError("outgoing packets exceed MPI byte-count limit");
      }
      host_offsets(r) = send_size/sizeof(AlphaPacket);
      send_offset[r] = static_cast<int>(send_size);
      send_count[r] = static_cast<int>(bytes);
      send_size += bytes;
    }
    MPI_Alltoall(send_count.data(), 1, MPI_INT,
                 recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::size_t recv_size = 0;
    for (int r = 0; r < nranks; ++r) {
      if (recv_count[r] < 0 || recv_count[r]%sizeof(AlphaPacket) != 0 ||
          recv_size+recv_count[r] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        TransportError("incoming packets exceed MPI byte-count limit");
      }
      recv_offset[r] = static_cast<int>(recv_size);
      recv_size += recv_count[r];
    }
    received = recv_size/sizeof(AlphaPacket);
    if (received > max_packets_-stats.kept) TransportError("incoming population exceeds max_packets");
    const int sent = send_size/sizeof(AlphaPacket);
    if (sent > 0) {
      GrowBuffer(outgoing_, sent, max_packets_, "dt_outgoing_packets");
      GrowBuffer(send_host_, sent, max_packets_, "dt_send_host");
      Kokkos::deep_copy(send_offsets_, host_offsets);
      auto offsets = send_offsets_; auto outgoing = outgoing_;
      par_for("dt_pack_migrants", DevExeSpace(), 0, packet_count_-1,
      KOKKOS_LAMBDA(int q) {
        const auto p = packets(q);
        if (p.status == 0 && blocks(p.gid).rank != rank) {
          const int slot = Kokkos::atomic_fetch_add(&offsets(blocks(p.gid).rank), 1);
          outgoing(slot) = p;
        }
      });
      const auto active = std::make_pair(0,sent);
      Kokkos::deep_copy(Kokkos::subview(send_host_,active), Kokkos::subview(outgoing_,active));
    }
    GrowBuffer(recv_host_, received, max_packets_, "dt_recv_host");
  }
#endif
  // The common single-rank/no-loss path retains its existing device allocation.
  // Compact only when a packet leaves this rank or is removed, using a stable scan.
  if (stats.kept != packet_count_) {
    GrowBuffer(packet_scratch_, stats.kept+received, max_packets_, "dt_packet_scratch");
    auto retained = packet_scratch_;
    int kept = 0;
    Kokkos::parallel_scan("dt_compact_packets",
        Kokkos::RangePolicy<DevExeSpace>(0,packet_count_),
    KOKKOS_LAMBDA(int q, int &offset, bool final) {
      const auto p = packets(q);
      const bool keep = p.status == 0 && blocks(p.gid).rank == rank;
      if (final && keep) retained(offset) = p;
      if (keep) ++offset;
    }, kept);
    std::swap(device_packets_, packet_scratch_);
  }
  packet_count_ = stats.kept;
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    MPI_Alltoallv(send_host_.data(), send_count.data(), send_offset.data(), MPI_BYTE,
                 recv_host_.data(), recv_count.data(), recv_offset.data(), MPI_BYTE,
                 MPI_COMM_WORLD);
    if (received > 0) {
      ReservePackets(packet_count_+received);
      Kokkos::deep_copy(Kokkos::subview(device_packets_,
          std::make_pair(packet_count_,packet_count_+received)),
          Kokkos::subview(recv_host_,std::make_pair(0,received)));
      packet_count_ += received;
    }
  }
#endif
  // Work includes outgoing packets. Transport globally reduces it after exchange,
  // so a finished sender cannot cause a receiver's unfinished flight to be skipped.
  return stats.work > 0 ? 1 : 0;
}


void DTFusion::TraceLocal() {
  auto mesh = pack_->pmesh;
  const auto ind = mesh->mb_indcs;
  const auto domain = mesh->mesh_size;
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  const int rank = global_variable::my_rank, first = pack_->gids;
  auto packet = device_packets_; auto blocks = blocks_; auto tree = tree_;
  auto feedback = feedback_;
  auto u = mhd_ ? pack_->pmhd->u0 : pack_->phydro->u0;
  const auto mix = mixture_;
  const bool collisions = collisions_, scattering = scattering_, periodic = periodic_;
  const double density_unit = density_, pressure_unit = energy_/(length_*length_*length_);
  const double time_unit = time_, speed_unit = time_/length_, energy_unit = energy_;
  const double momentum_unit = energy_*time_/length_;
  const double logl = log_lambda_, cutoff = cutoff_, thermal = thermal_factor_;
  const double fraction = loss_fraction_, scatter_limit = scatter_limit_;
  const double epsilon = std::numeric_limits<Real>::epsilon();
  const int segments = max_segments_, ii = iion_, ie = iele_;
  par_for("dt_alpha_transport", DevExeSpace(), 0, packet_count_-1,
  KOKKOS_LAMBDA(int q) {
    auto p = packet(q);
    if (!(p.remaining > 0.0)) return;
    for (int segment = 0; segment < segments && p.remaining > 0.0; ++segment) {
      LocatePacket(p,tree,blocks,domain,ind,ndim,periodic);
      if (p.status != 0) break;
      const auto b = blocks(p.gid);
      if (b.rank != rank) break;
      const int m = p.gid-first, i = p.cell[0]+ind.is;
      const int j = p.cell[1]+ind.js, k = p.cell[2]+ind.ks;
      const double lo[3] = {b.x1min,b.x2min,b.x3min};
      const double hi[3] = {b.x1max,b.x2max,b.x3max};
      const double dx[3] = {b.dx1,b.dx2,b.dx3};
      const int nc[3] = {ind.nx1,ind.nx2,ind.nx3};
      const double volume = dx[0]*dx[1]*dx[2];
      const double old_energy = p.energy, old_p = sqrt(2.0*kAlphaMass*p.energy);
      const double old_n[3] = {p.n[0],p.n[1],p.n[2]};
      PlasmaCollision rates;
      bool thermalize = false;
      if (collisions) {
        const auto composition = mix.CompositionFromConserved(u,m,k,j,i);
        const double rho = u(m,IDN,k,j,i);
        double ti, te;
        Temperatures(mix,composition,rho,u(m,ii,k,j,i),u(m,ie,k,j,i),
                     pressure_unit,density_unit,ti,te);
        if (!Kokkos::isfinite(ti) || !Kokkos::isfinite(te) || ti < 0.0 || te < 0.0) {
          p.status = 3; break;
        }
        thermalize = p.energy <= fmax(cutoff,thermal*kBoltzmann*fmax(ti,te));
        if (!thermalize) {
          rates = Collision(mix,composition,rho*density_unit,ti,te,p.energy,logl);
          if (!Kokkos::isfinite(rates.ion) || !Kokkos::isfinite(rates.electron) ||
              rates.ion < 0.0 || rates.electron < 0.0) {p.status = 3; break;}
        }
      }
      double loss_i = 0.0, loss_e = 0.0;
      if (thermalize) {
        // Merge the remaining suprathermal energy into the ion population.
        loss_i = p.energy; p.energy = 0.0; p.remaining = 0.0; p.status = 1;
      } else {
        const double speed = sqrt(2.0*p.energy/kAlphaMass)*speed_unit;
        const double rate = (rates.ion+rates.electron)*time_unit;
        const double nu = scattering ? rates.deflection*time_unit : 0.0;
        double h = p.remaining;
        if (rate > 0.0) h = fmin(h,fraction*p.energy/rate);
        if (nu > 0.0) h = fmin(h,scatter_limit/nu);
        double face_dt[3] = {1.0e300,1.0e300,1.0e300};
        Real face[3] = {p.x[0],p.x[1],p.x[2]};
        for (int d = 0; d < ndim; ++d) {
          if (p.n[d] == 0.0) continue;
          const int f = p.cell[d]+(p.n[d] > 0.0 ? 1 : 0);
          face[d] = f == 0 ? lo[d] : (f == nc[d] ? hi[d] : lo[d]+f*dx[d]);
          const double distance = (face[d]-p.x[d])/p.n[d];
          if (distance < -128.0*epsilon*
              fmax(hi[d]-lo[d],fabs(p.x[d]))) {p.status = 3; break;}
          face_dt[d] = fmax(distance,0.0)/speed;
          h = fmin(h,face_dt[d]);
        }
        if (p.status != 0) break;
        if (!Kokkos::isfinite(h) || (h > 0.0 && p.remaining-Real(h) == p.remaining)) {
          p.status = 3; break;
        }
        loss_i = rates.ion*time_unit*h; loss_e = rates.electron*time_unit*h;
        p.energy -= loss_i+loss_e;
        p.remaining = fmax(0.0, p.remaining-Real(h));
        for (int d = 0; d < ndim; ++d) {
          p.x[d] += speed*p.n[d]*h;
          if (face_dt[d] <= h) p.x[d] = face[d];
        }
        if (nu*h > 0.0) {
          fast_electrons::ScatterDirection(nu*h,p.rng,p.n[0],p.n[1],p.n[2]);
        }
      }
      const double new_p = sqrt(2.0*kAlphaMass*p.energy);
      Kokkos::atomic_add(&feedback(m,0,k,j,i),Real(p.weight*loss_i/(energy_unit*volume)));
      Kokkos::atomic_add(&feedback(m,1,k,j,i),Real(p.weight*loss_e/(energy_unit*volume)));
      for (int d = 0; d < 3; ++d) {
        Kokkos::atomic_add(&feedback(m,2+d,k,j,i),
            Real(p.weight*(old_p*old_n[d]-new_p*p.n[d])/(momentum_unit*volume)));
      }
      if (p.status != 0) break;
      if (!(p.energy > 0.0) || !Kokkos::isfinite(p.energy) || p.energy > old_energy) {
        p.status = 3; break;
      }
    }
    if (p.status == 0) LocatePacket(p,tree,blocks,domain,ind,ndim,periodic);
    packet(q) = p;
  });
}
}  // namespace nuclear
