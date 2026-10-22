// Optional simultaneous particle/capacity and local weighted-sampling diagnostics.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_memory.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"

namespace fast_electrons {
void FastElectrons::ReportMemory(const char *phase, Real time) {
  memory::Time(time);
  memory::Context(phase, device_packets_.extent(0));
  if (!memory::Reporting()) return;
  const auto s = memory::Get();
  const auto live = device_packets_.extent(0);
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<Real>::max_digits10)
      << "# fast_electrons_capacity rank=" << global_variable::my_rank
      << " time=" << time << " phase=" << phase
      << " live_packets=" << live << " live_bytes=" << live*sizeof(ElectronPacket)
      << " storage_slots=" << packet_storage_.extent(0)
      << " scratch_slots=" << packet_scratch_.extent(0)
      << " work0_slots=" << transport_work_[0].extent(0)
      << " work1_slots=" << transport_work_[1].extent(0)
      << " export_slots=" << transport_exports_.extent(0)
      << " source_slots=" << source_packets_.extent(0)
      << " ratios_have_live=" << (live > 0)
      << " capacity_to_live=" << (live ? double(s.packets)/(live*sizeof(ElectronPacket)) : 0.0)
      << " particle_to_mesh=" << (s.mesh ? double(s.packets)/s.mesh : 0.0)
      << " live_to_mesh=" << (s.mesh ? double(live*sizeof(ElectronPacket))/s.mesh : 0.0)
      << " packet_bytes=" << s.packets << " source_bytes=" << s.sources
      << " mesh_bytes=" << s.mesh << " runtime_bytes=" << s.runtime
      << " diagnostic_bytes=" << s.diagnostics << " total_bytes=" << s.total
      << " peak_bytes=" << s.peak << " allocation_count=" << s.allocations
      << " cumulative_allocated_bytes=" << s.allocated_bytes
      << " untracked_free_bytes=" << s.untracked_frees;
  memory::Report(out.str());
}

void FastElectrons::DiagnoseParticles(const char *phase, Real time) {
  if (particle_diagnostics_interval_ == 0 ||
      diagnostic_substeps_%particle_diagnostics_interval_ != 0) return;
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  const int first = pmy_pack_->gids;
  const int cells = pmy_pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  if (sampling_counts_.extent(0) != static_cast<std::size_t>(cells)) {
    sampling_counts_ = DvceArray1D<std::uint64_t>();
    sampling_weights_ = DvceArray2D<Real>();
    memory::Preflight(static_cast<std::uint64_t>(cells)*(sizeof(std::uint64_t)+3*sizeof(Real)),
                      "local sampling diagnostics");
    sampling_counts_ = DvceArray1D<std::uint64_t>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-sampling-counts"), cells);
    sampling_weights_ = DvceArray2D<Real>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-sampling-weights"), 3, cells);
  }
  Kokkos::deep_copy(sampling_counts_, std::uint64_t(0));
  Kokkos::deep_copy(sampling_weights_, 0.0);
  auto counts = sampling_counts_;
  auto weights = sampling_weights_;
  auto packets = device_packets_;
  int invalid = 0;
  Kokkos::parallel_reduce("fe_sampling_count", Kokkos::RangePolicy<DevExeSpace>(
      0, packets.extent_int(0)), KOKKOS_LAMBDA(int n, int &bad) {
    const auto p = packets(n);
    const int i = p.cell[0], j = p.cell[1], k = p.cell[2];
    const int c = (((p.gid-first)*ind.nx3+k)*ind.nx2+j)*ind.nx1+i;
    if (p.status != 0 || c < 0 || c >= cells || i < 0 || i >= ind.nx1 ||
        j < 0 || j >= ind.nx2 || k < 0 || k >= ind.nx3 ||
        !(p.weight > 0.0) || !Kokkos::isfinite(p.weight)) { ++bad; return; }
    Kokkos::atomic_inc(&counts(c));
    Kokkos::atomic_max(&weights(0, c), p.weight);
  }, invalid);
  if (invalid) Kokkos::abort("Invalid live particle in sampling diagnostics");
  // Normalize by the cell maximum before squaring. This keeps Neff finite even
  // when raw w*w or (sum w)^2 would overflow or underflow.
  par_for("fe_sampling_weights", DevExeSpace(), 0, packets.extent_int(0)-1,
  KOKKOS_LAMBDA(int n) {
    const auto p = packets(n);
    const int c = (((p.gid-first)*ind.nx3+p.cell[2])*ind.nx2+
                    p.cell[1])*ind.nx1+p.cell[0];
    const Real w = p.weight/weights(0, c);
    Kokkos::atomic_add(&weights(1, c), w);
    Kokkos::atomic_add(&weights(2, c), w*w);
  });
  const auto hc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), counts);
  const auto hw = Kokkos::create_mirror_view_and_copy(HostMemSpace(), weights);
  std::vector<std::uint64_t> sorted_counts(cells);
  std::vector<Real> effective;
  std::uint64_t total = 0;
  Real effective_sum = 0.0;
  std::ofstream csv;
  if (particle_diagnostics_cells_) {
    const auto name = "fast_electron_sampling.rank"+
                     std::to_string(global_variable::my_rank)+".csv";
    csv.open(name, sampling_file_started_ ? std::ios::app : std::ios::out);
    if (!csv) Kokkos::abort("Cannot write fast-electron cell sampling diagnostics");
    if (!sampling_file_started_) {
      csv << "cycle,substep,time,phase,rank,gid,i,j,k,ppc,neff,weight_scale,"
             "sum_scaled_w,sum_scaled_w2\n";
      sampling_file_started_ = true;
    }
    csv << std::setprecision(std::numeric_limits<Real>::max_digits10);
  }
  for (int c = 0; c < cells; ++c) {
    const auto count = hc(c);
    sorted_counts[c] = count;
    total += count;
    const Real neff = count ? hw(1, c)*hw(1, c)/hw(2, c) : 0.0;
    if (!std::isfinite(neff) || neff < 0.0 ||
        neff > count*(1.0+128*count*std::numeric_limits<Real>::epsilon())) {
      Kokkos::abort("Invalid fast-electron effective particle count");
    }
    if (count) { effective.push_back(neff); effective_sum += neff; }
    if (particle_diagnostics_cells_) {
      const int i = c%ind.nx1, j = (c/ind.nx1)%ind.nx2;
      const int k = (c/(ind.nx1*ind.nx2))%ind.nx3;
      const int gid = first+c/(ind.nx1*ind.nx2*ind.nx3);
      csv << pmy_pack_->pmesh->ncycle << ',' << diagnostic_substeps_ << ',' << time
          << ',' << phase << ',' << global_variable::my_rank << ',' << gid
          << ',' << i << ',' << j << ',' << k << ',' << count << ',' << neff
          << ',' << hw(0, c) << ',' << hw(1, c) << ',' << hw(2, c) << '\n';
    }
  }
  if (total != device_packets_.extent(0)) Kokkos::abort("Local PPC count mismatch");
  if (csv.is_open()) {
    csv.flush();
    if (!csv) Kokkos::abort("Failed writing cell sampling diagnostics");
  }
  std::sort(sorted_counts.begin(), sorted_counts.end());
  std::sort(effective.begin(), effective.end());
  const auto quantile = [&sorted_counts](double q) {
    return sorted_counts.empty() ? std::uint64_t(0) :
        sorted_counts[static_cast<std::size_t>(std::ceil(q*sorted_counts.size()))-1];
  };
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<Real>::max_digits10)
      << "# fast_electrons_sampling rank=" << global_variable::my_rank
      << " time=" << time << " substep=" << diagnostic_substeps_ << " phase=" << phase
      << " cells=" << cells << " occupied_cells=" << effective.size()
      << " packets=" << total << " ppc_mean=" << (cells ? Real(total)/cells : 0.0)
      << " ppc_p50=" << quantile(.50) << " ppc_p90=" << quantile(.90)
      << " ppc_p99=" << quantile(.99) << " ppc_max=" << quantile(1.0)
      << " neff_sum=" << effective_sum
      << " neff_mean_occupied=" << (effective.empty() ? 0.0 : effective_sum/effective.size())
      << " neff_min_occupied=" << (effective.empty() ? 0.0 : effective.front())
      << " neff_p50_occupied=" << (effective.empty() ? 0.0 :
          effective[(effective.size()-1)/2]);
  // A separate rank-local stream avoids MPI launcher stdout interleaving even
  // when cell CSV output and memory diagnostics are disabled.
  static std::ofstream summary("fast_electron_sampling.rank"+
      std::to_string(global_variable::my_rank)+".log");
  if (!summary) Kokkos::abort("### FATAL ERROR: opening fast-electron sampling log");
  summary << out.str() << std::endl;
  if (!summary) Kokkos::abort("### FATAL ERROR: writing fast-electron sampling log");
  std::cout << out.str() << std::endl;
}
}  // namespace fast_electrons
