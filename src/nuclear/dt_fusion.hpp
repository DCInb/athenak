#ifndef NUCLEAR_DT_FUSION_HPP_
#define NUCLEAR_DT_FUSION_HPP_
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
#include <cstdint>
#include <vector>
#include "athena.hpp"
#include "laser/laser_block_tree.hpp"
#include "materials/material_mixture.hpp"
#include "outputs/io_wrapper.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;
namespace nuclear {
struct AlphaPacket {
  Real x[3], n[3];
  double energy, weight;
  Real remaining;
  std::uint64_t rng;
  int cell[3], gid, status;  // live=0, thermalized=1, escaped=2, failed=3
};

class DTFusion {
 public:
  DTFusion(MeshBlockPack *pack, ParameterInput *pin);
  void Advance(Real dt, Driver *driver);
  void RefreshMesh();
  void DepositMoments();
  void Burn(Real dt);
  void TraceLocal();
  void ApplyFeedback();
  Real SourceTimeStep();
  void Diagnose();
  int RoutePackets();
  void Transport(Real dt);
  IOWrapperSizeT WriteRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank);
  IOWrapperSizeT ReadRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank);

  // rate [reactions/cm^3/s], n_alpha [cm^-3], E_alpha [code energy density],
  // last half-step deposited ion/electron energy and momentum densities (5 fields).
  static constexpr int ncell_data = 8;
  DvceArray5D<Real> cell_data;

 private:
  MeshBlockPack *pack_;
  materials::MaterialMixtureDevice mixture_;
  // Only [0, packet_count_) is live. Capacity is reused between source intervals.
  int packet_count_ = 0;
  DvceArray1D<AlphaPacket> device_packets_, packet_scratch_, outgoing_;
  HostArray1D<AlphaPacket> send_host_, recv_host_;
  DvceArray1D<int> birth_offsets_, send_counts_, send_offsets_;
  DvceArray1D<laser::LaserBlockInfo> blocks_;
  DvceArray1D<laser::LaserBlockTreeNode> tree_;
  HostArray1D<laser::LaserBlockInfo> host_blocks_;
  HostArray1D<laser::LaserBlockTreeNode> host_tree_;
  DvceArray5D<Real> feedback_;
  DvceArray4D<double> births_, stopping_;
  int deuterium_, tritium_, helium_, iion_, iele_;
  int pairs_, max_packets_, max_segments_, max_waves_, max_substeps_;
  bool mhd_, collisions_, scattering_, periodic_, report_;
  double length_, time_, energy_, density_, temperature_, mu_;
  double log_lambda_, cutoff_, thermal_factor_, loss_fraction_, scatter_limit_;
  double rate_multiplier_, burn_rtol_, source_fraction_;
  Real max_source_dt_;
  std::uint64_t seed_, birth_epoch_ = 0;
  // Cumulative rank-local accounting, independent of grid/regridding.
  // reactions, alpha born/deposited/escaped, neutron Q, neutron mass, neutron bulk E.
  double ledger_[7] = {};
  void ReservePackets(int count);
  std::vector<AlphaPacket> DownloadPackets() const;
  void RestorePackets(const std::vector<AlphaPacket> &packets);
  void SyncThermodynamics(Driver *driver);
};
}  // namespace nuclear
#endif  // NUCLEAR_DT_FUSION_HPP_
