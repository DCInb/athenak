// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
// Explicit, versioned records indexed by leaf block; shared restarts may repartition.
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "nuclear/dt_fusion.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace nuclear {
namespace {
constexpr char magic[8] = {'D','T','A','L','P','H','0','1'};
constexpr std::uint64_t width = 6*sizeof(Real)+2*sizeof(double)+sizeof(std::uint64_t);
void Write(IOWrapper &file, const void *p, std::uint64_t bytes,
           IOWrapperSizeT &offset, bool per_rank) {
  if (bytes && file.Write_any_type_at(p,bytes,offset,"byte",per_rank) != bytes) {
    Kokkos::abort("Could not write D-T particle restart");
  }
  offset += bytes;
}
void Read(IOWrapper &file, void *p, std::uint64_t bytes,
          IOWrapperSizeT &offset, bool per_rank) {
  if (bytes && file.Read_bytes_at(p,1,bytes,offset,per_rank) != bytes) {
    Kokkos::abort("Incomplete D-T particle restart");
  }
  offset += bytes;
}
}  // namespace

IOWrapperSizeT DTFusion::WriteRestart(IOWrapper &file, IOWrapperSizeT offset,
                                     bool per_rank) {
  auto mesh = pack_->pmesh;
  const int first = pack_->gids, nmb = pack_->nmb_thispack;
  const int begin = per_rank ? first : 0, end = per_rank ? first+nmb : mesh->nmb_total;
  const std::uint64_t header[10] = {1,sizeof(Real),
      static_cast<std::uint64_t>(mesh->nmb_total),static_cast<std::uint64_t>(begin),
      static_cast<std::uint64_t>(end),static_cast<std::uint64_t>(per_rank ?
          global_variable::nranks : 0),static_cast<std::uint64_t>(per_rank ?
          global_variable::my_rank : 0),seed_,birth_epoch_,width};
  std::vector<std::vector<AlphaPacket>> blocks(nmb);
  std::vector<std::uint64_t> counts(mesh->nmb_total,0);
  for (const auto &p : DownloadPackets()) {
    if (p.gid < first || p.gid >= first+nmb || p.status != 0 || p.remaining != 0.0) {
      Kokkos::abort("D-T checkpoint requested during transport");
    }
    blocks[p.gid-first].push_back(p); ++counts[p.gid];
  }
  double ledger[7];
  for (int n = 0; n < 7; ++n) ledger[n] = ledger_[n];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,counts.data(),mesh->nmb_total,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,ledger,7,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
  const auto body = offset+sizeof(magic)+sizeof(header)+sizeof(ledger)+
                    counts.size()*sizeof(std::uint64_t);
  if (per_rank || global_variable::my_rank == 0) {
    Write(file,magic,sizeof(magic),offset,per_rank);
    Write(file,header,sizeof(header),offset,per_rank);
    Write(file,ledger,sizeof(ledger),offset,per_rank);
    Write(file,counts.data(),counts.size()*sizeof(std::uint64_t),offset,per_rank);
  }
  offset = body;
  IOWrapperSizeT footer_end = body;
  for (int g = begin; g < end; ++g) footer_end += counts[g]*width;
  for (int g = begin; g < first; ++g) offset += counts[g]*width;
  for (const auto &block : blocks) {
    std::vector<unsigned char> bytes(block.size()*width);
    for (std::size_t n = 0; n < block.size(); ++n) {
      const auto &p = block[n];
      const Real geometry[6] = {p.x[0],p.x[1],p.x[2],p.n[0],p.n[1],p.n[2]};
      const double physical[2] = {p.energy,p.weight};
      auto record = bytes.data()+n*width;
      std::memcpy(record,geometry,sizeof(geometry));
      std::memcpy(record+sizeof(geometry),physical,sizeof(physical));
      std::memcpy(record+sizeof(geometry)+sizeof(physical),&p.rng,sizeof(p.rng));
    }
    Write(file,bytes.data(),bytes.size(),offset,per_rank);
  }
  return footer_end;
}

IOWrapperSizeT DTFusion::ReadRestart(IOWrapper &file, IOWrapperSizeT offset,
                                    bool per_rank) {
  auto mesh = pack_->pmesh;
  const int first = pack_->gids, nmb = pack_->nmb_thispack;
  const int begin = per_rank ? first : 0, end = per_rank ? first+nmb : mesh->nmb_total;
  char marker[8]; std::uint64_t header[10]; double ledger[7];
  Read(file,marker,sizeof(marker),offset,per_rank);
  Read(file,header,sizeof(header),offset,per_rank);
  Read(file,ledger,sizeof(ledger),offset,per_rank);
  if (std::memcmp(marker,magic,8) != 0 || header[0] != 1 || header[1] != sizeof(Real) ||
      header[2] != static_cast<std::uint64_t>(mesh->nmb_total) ||
      header[3] != static_cast<std::uint64_t>(begin) ||
      header[4] != static_cast<std::uint64_t>(end) || header[9] != width ||
      header[5] != static_cast<std::uint64_t>(per_rank ? global_variable::nranks : 0) ||
      header[6] != static_cast<std::uint64_t>(per_rank ? global_variable::my_rank : 0)) {
    Kokkos::abort("Invalid/incompatible D-T particle restart");
  }
  seed_ = header[7]; birth_epoch_ = header[8];
  for (int n = 0; n < 7; ++n) {
    if (!std::isfinite(ledger[n]) || ledger[n] < 0.0) Kokkos::abort("Invalid D-T ledger");
    ledger_[n] = global_variable::my_rank == 0 ? ledger[n] : 0.0;
  }
  std::vector<std::uint64_t> counts(mesh->nmb_total);
  Read(file,counts.data(),counts.size()*sizeof(std::uint64_t),offset,per_rank);
  std::uint64_t count = 0;
  for (int g = 0; g < mesh->nmb_total; ++g) {
    if (counts[g] > static_cast<std::uint64_t>(max_packets_)) {
      Kokkos::abort("D-T checkpoint exceeds max_packets in one block");
    }
    if (g >= first && g < first+nmb) count += counts[g];
  }
  if (count > static_cast<std::uint64_t>(max_packets_)) Kokkos::abort("Too many restart alphas");
  IOWrapperSizeT footer_end = offset;
  for (int g = begin; g < end; ++g) footer_end += counts[g]*width;
  for (int g = begin; g < first; ++g) offset += counts[g]*width;
  std::vector<AlphaPacket> packets;
  packets.reserve(count);
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  for (int g = first; g < first+nmb; ++g) {
    std::vector<unsigned char> bytes(counts[g]*width);
    Read(file,bytes.data(),bytes.size(),offset,per_rank);
    for (std::uint64_t n = 0; n < counts[g]; ++n) {
      AlphaPacket p{}; Real geometry[6]; double physical[2];
      const auto record = bytes.data()+n*width;
      std::memcpy(geometry,record,sizeof(geometry));
      std::memcpy(physical,record+sizeof(geometry),sizeof(physical));
      std::memcpy(&p.rng,record+sizeof(geometry)+sizeof(physical),sizeof(p.rng));
      double norm = 0.0;
      for (int d = 0; d < 3; ++d) {
        p.x[d] = geometry[d]; p.n[d] = geometry[3+d]; norm += p.n[d]*p.n[d];
        if (!std::isfinite(p.x[d])) Kokkos::abort("Invalid alpha checkpoint position");
      }
      p.energy = physical[0]; p.weight = physical[1];
      if (!std::isfinite(norm) || fabs(norm-1.0) > 1.0e-5 ||
          !std::isfinite(p.energy) || !(p.energy > 0.0) ||
          !std::isfinite(p.weight) || !(p.weight > 0.0)) Kokkos::abort("Invalid saved alpha");
      fast_electrons::LocatePacket(p,host_tree_,host_blocks_,mesh->mesh_size,
                                  mesh->mb_indcs,ndim,periodic_);
      if (p.status != 0 || p.gid != g) Kokkos::abort("Alpha restart leaf mismatch");
      packets.push_back(p);
    }
  }
  RestorePackets(packets);
  return footer_end;
}
}  // namespace nuclear
