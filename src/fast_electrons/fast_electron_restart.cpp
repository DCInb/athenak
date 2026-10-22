//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Versioned particle footer, indexed by leaf block so shared files can be repartitioned.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace fast_electrons {
namespace {
constexpr char magic[8] = {'F', 'E', 'P', 'I', 'C', '0', '0', '1'};
constexpr std::uint64_t version = 1;
constexpr std::uint64_t record_bytes = 8*sizeof(Real)+sizeof(std::uint64_t);

void WriteBytes(IOWrapper &file, const void *p, std::uint64_t bytes,
                IOWrapperSizeT &offset, bool per_rank) {
  if (bytes != 0 && file.Write_any_type_at(p, bytes, offset, "byte", per_rank) != bytes) {
    Kokkos::abort("Could not write fast-electron restart footer");
  }
  offset += bytes;
}
void ReadBytes(IOWrapper &file, void *p, std::uint64_t bytes,
               IOWrapperSizeT &offset, bool per_rank) {
  if (bytes != 0 && file.Read_bytes_at(p, 1, bytes, offset, per_rank) != bytes) {
    Kokkos::abort("Incomplete fast-electron restart footer");
  }
  offset += bytes;
}
}  // namespace

void FastElectrons::WriteRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank) {
  DownloadPackets();
  auto mesh = pmy_pack_->pmesh;
  const int first = pmy_pack_->gids, nmb = pmy_pack_->nmb_thispack;
  const std::uint64_t cells = cell_data.extent(2)*cell_data.extent(3)*cell_data.extent(4);
  const int start = per_rank ? first : 0, blocks = per_rank ? nmb : mesh->nmb_total;
  const bool scheduled = emission_interval_ > 0.0;
  const Real schedule[3] = {start_, end_, emission_interval_};
  const std::uint64_t rule[2] = {static_cast<std::uint64_t>(packets_per_bin_),
                                source_points_.size()};
  std::uint64_t header[13] = {scheduled ? 2 : version, sizeof(Real), ncell_data, cells,
      static_cast<std::uint64_t>(mesh->nmb_total), spectrum_.size(), next_packet_, seed_,
      static_cast<std::uint64_t>(start), static_cast<std::uint64_t>(blocks),
      static_cast<std::uint64_t>(per_rank ? global_variable::nranks : 0),
      static_cast<std::uint64_t>(per_rank ? global_variable::my_rank : 0), record_bytes};
  std::vector<std::vector<ElectronPacket>> by_block(nmb);
  std::vector<std::uint64_t> counts(mesh->nmb_total, 0);
  for (const auto &p : packets_) {
    if (p.gid < first || p.gid >= first+nmb || p.remaining != 0.0 || p.status != 0) {
      Kokkos::abort("Fast-electron checkpoint requested during a particle substep");
    }
    by_block[p.gid-first].push_back(p);
    ++counts[p.gid];
  }
  Real ledger[2] = {injected_, escaped_};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), mesh->nmb_total, MPI_UINT64_T,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ledger, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  std::vector<Real> spectrum(5*spectrum_.size());
  for (std::size_t n = 0; n < spectrum_.size(); ++n) {
    spectrum[5*n] = spectrum_[n].energy;
    for (int d = 0; d < 3; ++d) spectrum[5*n+1+d] = spectrum_[n].n[d];
    spectrum[5*n+4] = spectrum_[n].weight;
  }
  const auto body = offset+sizeof(magic)+sizeof(header)+sizeof(ledger)+
                    spectrum.size()*sizeof(Real)+counts.size()*sizeof(std::uint64_t)+
                    (scheduled ? sizeof(schedule)+sizeof(rule) : 0);
  if (global_variable::my_rank == 0 || per_rank) {
    WriteBytes(file, magic, sizeof(magic), offset, per_rank);
    WriteBytes(file, header, sizeof(header), offset, per_rank);
    WriteBytes(file, ledger, sizeof(ledger), offset, per_rank);
    if (scheduled) {
      WriteBytes(file, schedule, sizeof(schedule), offset, per_rank);
      WriteBytes(file, rule, sizeof(rule), offset, per_rank);
    }
    WriteBytes(file, spectrum.data(), spectrum.size()*sizeof(Real), offset, per_rank);
    WriteBytes(file, counts.data(), counts.size()*sizeof(std::uint64_t),
               offset, per_rank);
  }
  offset = body;
  for (int g = start; g < first; ++g) {
    offset += ncell_data*cells*sizeof(Real)+counts[g]*record_bytes;
  }
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cell_data);
  for (int m = 0; m < nmb; ++m) {
    auto cell = Kokkos::subview(host, m, Kokkos::ALL, Kokkos::ALL,
                               Kokkos::ALL, Kokkos::ALL);
    WriteBytes(file, cell.data(), cell.size()*sizeof(Real), offset, per_rank);
    // Explicit records avoid persisting C++ struct padding, cached cell indices, or
    // compiler-dependent layouts. Native Real size and record width are checked on read.
    std::vector<unsigned char> bytes(by_block[m].size()*record_bytes);
    for (std::size_t n = 0; n < by_block[m].size(); ++n) {
      const auto p = by_block[m][n];
      const Real values[8] = {p.x[0], p.x[1], p.x[2], p.n[0], p.n[1], p.n[2],
                              p.energy, p.weight};
      std::memcpy(bytes.data()+n*record_bytes, values, sizeof(values));
      std::memcpy(bytes.data()+n*record_bytes+sizeof(values), &p.rng, sizeof(p.rng));
    }
    WriteBytes(file, bytes.data(), bytes.size(), offset, per_rank);
  }
}

void FastElectrons::ReadRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank) {
  auto mesh = pmy_pack_->pmesh;
  char marker[8];
  std::uint64_t header[13];
  Real ledger[2];
  ReadBytes(file, marker, sizeof(marker), offset, per_rank);
  ReadBytes(file, header, sizeof(header), offset, per_rank);
  ReadBytes(file, ledger, sizeof(ledger), offset, per_rank);
  const std::uint64_t cells = cell_data.extent(2)*cell_data.extent(3)*cell_data.extent(4);
  const int first = pmy_pack_->gids, nmb = pmy_pack_->nmb_thispack;
  if (std::memcmp(marker, magic, sizeof(magic)) != 0 ||
      (header[0] != version && header[0] != 2) ||
      header[1] != sizeof(Real) || header[2] != ncell_data || header[3] != cells ||
      header[4] != static_cast<std::uint64_t>(mesh->nmb_total) ||
      header[5] == 0 || header[5] > static_cast<std::uint64_t>(max_packets_) ||
      header[12] != record_bytes || !std::isfinite(ledger[0]) || ledger[0] < 0.0 ||
      !std::isfinite(ledger[1]) || ledger[1] < 0.0) {
    Kokkos::abort("Invalid or incompatible fast-electron restart footer");
  }
  if (header[0] == 2) {
    Real schedule[3];
    std::uint64_t rule[2];
    ReadBytes(file, schedule, sizeof(schedule), offset, per_rank);
    ReadBytes(file, rule, sizeof(rule), offset, per_rank);
    if (schedule[0] != start_ || schedule[1] != end_ || schedule[2] != emission_interval_ ||
        emission_interval_ <= 0.0 || rule[0] != static_cast<std::uint64_t>(packets_per_bin_) ||
        rule[1] != source_points_.size()) {
      Kokkos::abort("### FATAL ERROR: fixed emission schedule cannot change on restart");
    }
  } else if (emission_interval_ > 0.0) {
    Kokkos::abort("### FATAL ERROR: cannot enable fixed emission on a legacy checkpoint");
  }
  if (per_rank && (header[8] != static_cast<std::uint64_t>(first) ||
      header[9] != static_cast<std::uint64_t>(nmb) ||
      header[10] != static_cast<std::uint64_t>(global_variable::nranks) ||
      header[11] != static_cast<std::uint64_t>(global_variable::my_rank))) {
    Kokkos::abort("Per-rank particle restarts require the saved MPI decomposition");
  }
  if (!per_rank && (header[8] != 0 || header[9] != header[4] || header[10] != 0)) {
    Kokkos::abort("Expected a shared fast-electron checkpoint");
  }
  next_packet_ = header[6]; seed_ = header[7];
  if (header[0] == 2) {
    const auto rows = header[5]*source_points_.size();
    const auto events = emission_bins_*static_cast<std::uint64_t>(packets_per_bin_);
    if (next_packet_%rows != 0 || next_packet_/rows > events) {
      Kokkos::abort("Invalid fixed emission event ordinal in checkpoint");
    }
  }
  injected_ = global_variable::my_rank == 0 ? ledger[0] : 0.0;
  escaped_ = global_variable::my_rank == 0 ? ledger[1] : 0.0;
  std::vector<Real> spectrum(5*header[5]);
  ReadBytes(file, spectrum.data(), spectrum.size()*sizeof(Real), offset, per_rank);
  spectrum_.resize(header[5]);
  Real normalization = 0.0;
  for (std::size_t n = 0; n < spectrum_.size(); ++n) {
    auto &bin = spectrum_[n];
    bin.energy = spectrum[5*n]; bin.weight = spectrum[5*n+4];
    Real norm = 0.0;
    for (int d = 0; d < 3; ++d) {
      bin.n[d] = spectrum[5*n+1+d];
      norm += bin.n[d]*bin.n[d];
    }
    if (!(bin.energy > 0.0) || !(bin.weight > 0.0) || !std::isfinite(norm) ||
        std::abs(norm-1.0) > 128*std::numeric_limits<Real>::epsilon()) {
      Kokkos::abort("Invalid fast-electron spectrum in checkpoint");
    }
    normalization += bin.energy*bin.weight;
  }
  if (!std::isfinite(normalization) || std::abs(normalization-1.0) > 1.0e-5) {
    Kokkos::abort("Invalid fast-electron source normalization in checkpoint");
  }
  std::vector<std::uint64_t> counts(mesh->nmb_total);
  ReadBytes(file, counts.data(), counts.size()*sizeof(std::uint64_t),
            offset, per_rank);
  std::uint64_t local_count = 0;
  for (int g = 0; g < mesh->nmb_total; ++g) {
    if (counts[g] > static_cast<std::uint64_t>(max_packets_)) {
      Kokkos::abort("Fast-electron restart exceeds max_packets in one block");
    }
    if (g >= first && g < first+nmb) local_count += counts[g];
  }
  if (local_count > static_cast<std::uint64_t>(max_packets_)) {
    Kokkos::abort("Fast-electron restart exceeds max_packets on this rank");
  }
  for (int g = per_rank ? first : 0; g < first; ++g) {
    offset += ncell_data*cells*sizeof(Real)+counts[g]*record_bytes;
  }
  auto host = Kokkos::create_mirror_view(cell_data);
  Kokkos::deep_copy(host, 0.0);
  packets_.clear(); packets_.reserve(local_count);
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  for (int m = 0; m < nmb; ++m) {
    auto cell = Kokkos::subview(host, m, Kokkos::ALL, Kokkos::ALL,
                               Kokkos::ALL, Kokkos::ALL);
    ReadBytes(file, cell.data(), cell.size()*sizeof(Real), offset, per_rank);
    for (std::size_t n = 0; n < cell.size(); ++n) {
      if (!std::isfinite(cell.data()[n])) {
        Kokkos::abort("Nonfinite particle checkpoint data");
      }
    }
    std::vector<unsigned char> bytes(counts[first+m]*record_bytes);
    ReadBytes(file, bytes.data(), bytes.size(), offset, per_rank);
    for (std::size_t n = 0; n < counts[first+m]; ++n) {
      ElectronPacket p{};
      Real values[8];
      std::memcpy(values, bytes.data()+n*record_bytes, sizeof(values));
      std::memcpy(&p.rng, bytes.data()+n*record_bytes+sizeof(values), sizeof(p.rng));
      Real norm = 0.0;
      for (int d = 0; d < 3; ++d) {
        p.x[d] = values[d]; p.n[d] = values[d+3]; norm += p.n[d]*p.n[d];
      }
      p.energy = values[6]; p.weight = values[7];
      if (!std::isfinite(p.energy) || p.energy < 0.0 || !std::isfinite(p.weight) ||
          !(p.weight > 0.0) || !std::isfinite(norm) || std::abs(norm-1.0) > 1.0e-5) {
        Kokkos::abort("Invalid electron packet in restart");
      }
      LocatePacket(p, host_tree_, host_blocks_, mesh->mesh_size, mesh->mb_indcs,
                   ndim, periodic_);
      if (p.status != 0 || p.gid != first+m) {
        Kokkos::abort("Particle restart position does not match its leaf block");
      }
      packets_.push_back(p);
    }
  }
  Kokkos::deep_copy(cell_data, host);
  UploadPackets();
  // Recompute number/energy moments when primitives have been initialized by Driver.
  // The saved current and E are retained here because MHD primitive arrays are not ready.
}
}  // namespace fast_electrons
