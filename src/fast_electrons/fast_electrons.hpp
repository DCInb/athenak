#ifndef FAST_ELECTRONS_FAST_ELECTRONS_HPP_
#define FAST_ELECTRONS_FAST_ELECTRONS_HPP_
//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Weighted relativistic electron transport and quasineutral resistive hybrid coupling.
//========================================================================================

#include <cstdint>
#include <vector>

#include "athena.hpp"
#include "laser/laser_block_tree.hpp"
#include "outputs/io_wrapper.hpp"

class Driver;
class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace fast_electrons {

// x and remaining use code units; energy is erg/electron, weight is physical electrons.
// gid and cell cache the leaf-block identifier and indices within its interior.
struct ElectronPacket {
  Real x[3], n[3];
  Real energy, weight, remaining;
  std::uint64_t rng;
  int cell[3], gid;
  int status;  // 0: live, 1: thermalized, 2: escaped, 3: failed
};

struct SpectrumBin {
  Real energy, n[3], weight;
};

struct SourcePoint {
  Real x2, x3, weight;
};

class FastElectrons {
 public:
  FastElectrons(MeshBlockPack *ppack, ParameterInput *pin);
  ~FastElectrons();
  void Advance(Real time, Real dt, Driver *driver);
  void RefreshMesh();
  bool Electromagnetic() const { return electromagnetic_; }
  void WriteRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank);
  void ReadRestart(IOWrapper &file, IOWrapperSizeT offset, bool per_rank);

  // q_e, cumulative beam energy transferred, n_fast (cm^-3), kinetic density,
  // Ohmic heating, J_fast[3], E[3]. All except n_fast use AthenaK code units.
  static constexpr int ncell_data = 11;
  DvceArray5D<Real> cell_data, coarse_cell_data;

  // Public because CUDA extended device lambdas require public enclosing functions.
  void TraceLocal();
  void DepositMoments(Real duration = 0.0);
  void ApplyFeedback(Real dt);
  void Diagnose();
  void BuildFields();
  Real FieldTimeStep(bool include_electric);
  void AdvanceFields(Real dt);
  void FillCurrentBoundaries();
  void Inject(Real time, Real dt);
  void Transport();
  void TransportDistributed();
  void InjectScheduled(Real time, Real dt);
  void ReportMemory(const char *phase, Real time);
  void DiagnoseParticles(const char *phase, Real time);

 private:
  MeshBlockPack *pmy_pack_;
  std::vector<SpectrumBin> spectrum_;
  std::vector<SourcePoint> source_points_;
  std::vector<ElectronPacket> packets_;
  // device_packets_ is the live prefix of reusable storage. The full host
  // population is refreshed only for checkpoints and regrids.
  DvceArray1D<ElectronPacket> device_packets_;
  DvceArray1D<ElectronPacket> packet_storage_, packet_scratch_, source_packets_;
  DvceArray1D<ElectronPacket> transport_work_[2], transport_exports_;
  DvceArray1D<std::uint64_t> sampling_counts_;
  DvceArray2D<Real> sampling_weights_;
  int particle_diagnostics_interval_ = 0;
  bool particle_diagnostics_cells_ = false, sampling_file_started_ = false;
  std::uint64_t diagnostic_substeps_ = 0;
  Real emission_interval_ = 0.0;
  std::uint64_t emission_bins_ = 0;
  bool source_ready_ = false;
  Real source_injected_weight_ = 0.0, source_escaped_weight_ = 0.0;
  DvceArray1D<laser::LaserBlockInfo> blocks_;
  DvceArray1D<laser::LaserBlockTreeNode> tree_;
  HostArray1D<laser::LaserBlockInfo> host_blocks_;
  HostArray1D<laser::LaserBlockTreeNode> host_tree_;
  // Per-substep energy/momentum feedback and current with its AMR halo.
  DvceArray5D<Real> feedback_, current_, coarse_current_;
  // Small GPU replicas spread contended deposits from dense point-source beams.
  DvceArray6D<Real> deposit_scratch_;
  DvceEdgeFld4D<Real> emf_;
  DvceFaceFld5D<Real> energy_flux_;
  MeshBoundaryValuesCC *current_bvals_ = nullptr;
  bool use_mhd_, use_materials_, collisions_, scattering_, periodic_, report_;
  bool reclaim_packet_memory_ = false;
  bool electromagnetic_ = false;
  int electron_index_, packets_per_bin_, max_segments_, max_waves_, max_packets_;
  int max_field_steps_ = 100000;
  Real diffusivity_ = 0.0, field_cfl_ = 0.1, bunit_ = 1.0, current_unit_ = 1.0;
  Real length_cgs_, time_cgs_, energy_cgs_, density_cgs_, temperature_cgs_;
  Real number_per_gram_, zeff_, coulomb_log_, cutoff_, thermal_factor_;
  Real loss_fraction_, scatter_limit_, power_, start_, end_, origin_[3];
  Real injected_ = 0.0, escaped_ = 0.0;
  std::uint64_t seed_, next_packet_ = 0;

  void LoadSpectrum(ParameterInput *pin);
  void ExchangePackets();
  void PrepareSource();
  void ReservePackets(std::size_t count);
  void ReclaimPacketMemory();
  int PrepareDeposits();
  void UploadPackets();
  void DownloadPackets();
  int Owner(const ElectronPacket &packet) const;
};

}  // namespace fast_electrons
#endif  // FAST_ELECTRONS_FAST_ELECTRONS_HPP_
