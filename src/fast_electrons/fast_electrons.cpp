//========================================================================================
// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
//! Finite-speed cold-target electron transport with conservative collisional heating.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "fast_electrons/fast_electron_mesh.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "fast_electrons/fast_electrons.hpp"
#include "fast_electrons/fast_electron_memory.hpp"
#include "fast_electrons/fast_electron_physics.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "materials/material_mixture.hpp"
#include "mhd/mhd.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace fast_electrons {
namespace {
[[noreturn]] void Error(const std::string &message) {
  std::cerr << "### FATAL ERROR: <fast_electrons> " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

void Positive(Real value, const std::string &name, bool allow_zero = false) {
  if (!std::isfinite(value) || (allow_zero ? value < 0.0 : value <= 0.0)) {
    Error(name + " must be finite and " + (allow_zero ? "nonnegative" : "positive"));
  }
}
}  // namespace

FastElectrons::FastElectrons(MeshBlockPack *ppack, ParameterInput *pin)
    : cell_data("fast-electron-cell-data", 1, ncell_data, 1, 1, 1),
      coarse_cell_data("fast-electron-coarse-data", 1, ncell_data, 1, 1, 1),
      pmy_pack_(ppack), device_packets_("fast-electron-packets", 0),
      blocks_("fast-electron-blocks", 0), tree_("fast-electron-tree", 0),
      feedback_("fast-electron-feedback", 1, 4, 1, 1, 1),
      current_("fast-electron-current", 1, 3, 1, 1, 1),
      coarse_current_("fast-electron-coarse-current", 1, 3, 1, 1, 1),
      emf_("fast-electron-emf", 1, 1, 1, 1),
      energy_flux_("fast-electron-poynting", 1, 1, 1, 1, 1) {
  auto mesh = ppack->pmesh;
  if (mesh->nmb_packs_thisrank != 1) Error("requires one MeshBlockPack per rank");
  use_mhd_ = ppack->pmhd != nullptr;
  if ((ppack->phydro == nullptr) == (ppack->pmhd == nullptr) ||
      ppack->prad != nullptr || ppack->pcoord->is_special_relativistic ||
      ppack->pcoord->is_general_relativistic ||
      ppack->pcoord->is_dynamical_relativistic) {
    Error("requires a single Newtonian hydro or MHD fluid");
  }
  auto tt = use_mhd_ ? ppack->pmhd->ptwo_temp : ppack->phydro->ptwo_temp;
  if (tt == nullptr) Error("requires two_temperature=true in the fluid block");
  auto eos = use_mhd_ ? ppack->pmhd->peos : ppack->phydro->peos;
  if (!eos->eos_data.is_gamma_law) Error("requires a gamma-law carrier fluid");
  electron_index_ = tt->iele;
  auto material = use_mhd_ ? ppack->pmhd->pmaterials : ppack->phydro->pmaterials;
  use_materials_ = material != nullptr;
  if (ppack->punit == nullptr) Error("requires explicit <units> MLT scales");
  length_cgs_ = ppack->punit->length_cgs();
  time_cgs_ = ppack->punit->time_cgs();
  energy_cgs_ = ppack->punit->energy_cgs();
  density_cgs_ = ppack->punit->density_cgs();
  temperature_cgs_ = ppack->punit->temperature_cgs();
  Positive(length_cgs_, "units/length_cgs");
  Positive(time_cgs_, "units/time_cgs");
  Positive(energy_cgs_, "derived energy unit");
  Positive(density_cgs_, "derived density unit");
  Positive(temperature_cgs_, "derived temperature unit");

  const std::string b = "fast_electrons";
  collisions_ = pin->GetOrAddBoolean(b, "collisions", true);
  scattering_ = pin->GetOrAddBoolean(b, "scattering", true);
  periodic_ = pin->GetOrAddBoolean(b, "periodic_transport", false);
  report_ = pin->GetOrAddBoolean(b, "report_diagnostics", true);
  reclaim_packet_memory_ = pin->GetOrAddBoolean(b, "reclaim_packet_memory", false);
  particle_diagnostics_interval_ = pin->GetOrAddInteger(b, "particle_diagnostics_interval", 0);
  particle_diagnostics_cells_ = pin->GetOrAddBoolean(b, "particle_diagnostics_cells", false);
  if (particle_diagnostics_interval_ < 0) Error("particle_diagnostics_interval must be >=0");
  number_per_gram_ = pin->GetOrAddReal(b, "electron_number_per_gram", 0.0);
  if (!use_materials_ && collisions_) {
    Positive(number_per_gram_, "electron_number_per_gram");
  }
  zeff_ = pin->GetOrAddReal(b, "zeff", 1.0);
  coulomb_log_ = pin->GetOrAddReal(b, "coulomb_log", 10.0);
  cutoff_ = pin->GetOrAddReal(b, "thermalization_kev", 1.0)*1.0e3*kEvToErg;
  thermal_factor_ = pin->GetOrAddReal(b, "thermalization_factor", 3.0);
  loss_fraction_ = pin->GetOrAddReal(b, "max_fractional_loss", 0.05);
  scatter_limit_ = pin->GetOrAddReal(b, "max_scatter", 0.05);
  Positive(zeff_, "zeff", true);
  Positive(coulomb_log_, "coulomb_log");
  Positive(cutoff_, "thermalization_kev");
  Positive(thermal_factor_, "thermalization_factor", true);
  Positive(loss_fraction_, "max_fractional_loss");
  Positive(scatter_limit_, "max_scatter");
  if (loss_fraction_ > 0.25 || scatter_limit_ > 0.1) {
    Error("max_fractional_loss must be <=0.25 and max_scatter <=0.1");
  }
  packets_per_bin_ = pin->GetOrAddInteger(b, "packets_per_bin", 1);
  max_segments_ = pin->GetOrAddInteger(b, "max_segments", 256);
  max_waves_ = pin->GetOrAddInteger(b, "max_transport_waves", 10000);
  max_packets_ = pin->GetOrAddInteger(b, "max_packets", 1000000);
  const int seed = pin->GetOrAddInteger(b, "seed", 12345);
  if (packets_per_bin_ < 1 || max_segments_ < 1 || max_waves_ < 1 ||
      max_packets_ < 1 || seed < 0) {
    Error("packet/work limits must be positive; seed >=0");
  }
  seed_ = static_cast<std::uint64_t>(seed);
  power_ = pin->GetReal(b, "power_erg_s");
  start_ = pin->GetOrAddReal(b, "start_time", 0.0);
  end_ = pin->GetOrAddReal(b, "end_time", pin->GetReal("time", "tlim"));
  Positive(power_, "power_erg_s", true);
  if (!std::isfinite(start_) || !std::isfinite(end_) || end_ < start_) {
    Error("start_time/end_time must be finite and ordered");
  }
  emission_interval_ = pin->GetOrAddReal(b, "emission_interval", 0.0);
  Positive(emission_interval_, "emission_interval", true);
  if (emission_interval_ > 0.0 && end_ > start_) {
    const Real bins = std::ceil((end_-start_)/emission_interval_);
    if (!std::isfinite(bins) || bins > 1.0e12 ||
        emission_interval_/packets_per_bin_ <= 32*std::numeric_limits<Real>::epsilon()*
            std::max(std::abs(start_), std::abs(end_))) {
      Error("emission_interval is too small to resolve its event times");
    }
    emission_bins_ = static_cast<std::uint64_t>(bins);
    while (emission_bins_ && start_+(emission_bins_-1)*emission_interval_ >= end_) {
      --emission_bins_;
    }
    if (emission_bins_ > std::numeric_limits<std::uint64_t>::max()/packets_per_bin_) {
      Error("fixed emission sequence exceeds the event identifier range");
    }
  }
  const Real lower[3] = {mesh->mesh_size.x1min, mesh->mesh_size.x2min,
                         mesh->mesh_size.x3min};
  const Real upper[3] = {mesh->mesh_size.x1max, mesh->mesh_size.x2max,
                         mesh->mesh_size.x3max};
  for (int d = 0; d < 3; ++d) {
    origin_[d] = pin->GetOrAddReal(b, "origin_x" + std::to_string(d+1),
                                  d == 0 ? lower[d] : 0.5*(lower[d]+upper[d]));
    if (!std::isfinite(origin_[d]) || origin_[d] < lower[d] || origin_[d] > upper[d]) {
      Error("origin must lie inside the domain (a boundary face is allowed)");
    }
    Positive(upper[d]-lower[d], "domain width");
  }
  // A transverse disk uses a tensor quadrature independent of energy and direction.
  // Its parameters live in the ordinary input/restart header; the spectrum footer
  // stays unchanged. Radius zero retains the original point source and RNG order.
  const Real radius = pin->GetOrAddReal(b, "source_radius", 0.0);
  Positive(radius, "source_radius", true);
  if (radius == 0.0) {
    source_points_.push_back({origin_[1], origin_[2], 1.0});
  } else {
    const Real cutoff = pin->GetOrAddReal(b, "source_cutoff", 1.5);
    const Real exponent = pin->GetOrAddReal(b, "source_radial_power", 8.0);
    const int nr = pin->GetOrAddInteger(b, "source_radial_order", 8);
    const int na = pin->GetOrAddInteger(b, "source_azimuthal_points", 16);
    Positive(cutoff, "source_cutoff");
    Positive(exponent, "source_radial_power");
    if (!mesh->three_d) Error("finite-radius source requires a 3D mesh");
    if (nr < 1 || nr > 64 || na < 4 || na > 4096 || na%4 != 0) {
      Error("source_radial_order must be 1..64; source_azimuthal_points "
            "must be a multiple of four in 4..4096");
    }
    const Real rmax = radius*cutoff;
    Positive(rmax, "source cutoff radius");
    for (int d = 1; d < 3; ++d) {
      if (origin_[d]-rmax < lower[d] || origin_[d]+rmax > upper[d]) {
        Error("finite-radius source disk must fit inside the transverse domain");
      }
    }
    Real normalization = 0.0;
    for (int ir = 0; ir < nr; ++ir) {
      // Gauss-Legendre in squared radius: dA = pi*rmax^2 du, 0 < u < 1.
      Real z = std::cos(kPi*(ir+0.75)/(nr+0.5)), derivative = 0.0;
      for (int iteration = 0; iteration < 64; ++iteration) {
        Real p = 1.0, previous = 0.0;
        for (int j = 1; j <= nr; ++j) {
          const Real older = previous;
          previous = p;
          p = ((2*j-1)*z*previous-(j-1)*older)/j;
        }
        derivative = nr*(z*p-previous)/(z*z-1.0);
        const Real dz = p/derivative;
        z -= dz;
        if (std::abs(dz) <= 8*std::numeric_limits<Real>::epsilon()) break;
        if (iteration == 63) Error("source radial quadrature did not converge");
      }
      const Real r = rmax*std::sqrt((1.0+z)*0.5);
      const Real weight = std::exp(-std::pow(r/radius, exponent))/
                          ((1.0-z*z)*derivative*derivative*na);
      // Extremely small tails can underflow, and carry no representable weight.
      if (weight == 0.0) continue;
      for (int ia = 0; ia < na; ++ia) {
        const Real phi = 2*kPi*(ia+0.5)/na;
        source_points_.push_back({origin_[1]+r*std::cos(phi),
                                  origin_[2]+r*std::sin(phi), weight});
        normalization += weight;
      }
    }
    Positive(normalization, "source spatial normalization");
    for (auto &point : source_points_) point.weight /= normalization;
  }
  electromagnetic_ = pin->GetOrAddBoolean(b, "electromagnetic", false);
  field_cfl_ = pin->GetOrAddReal(b, "field_cfl", 0.1);
  max_field_steps_ = pin->GetOrAddInteger(b, "max_field_substeps", 100000);
  const Real eta_cgs = pin->GetOrAddReal(b, "magnetic_diffusivity_cm2_s", 0.0);
  Positive(eta_cgs, "magnetic_diffusivity_cm2_s", true);
  Positive(field_cfl_, "field_cfl");
  if (field_cfl_ > 0.25 || max_field_steps_ < 1) {
    Error("field_cfl must be <=0.25; max_field_substeps must be positive");
  }
  diffusivity_ = eta_cgs*time_cgs_/(length_cgs_*length_cgs_);
  if (electromagnetic_) {
    if (!use_mhd_) Error("electromagnetic=true requires a two-temperature MHD fluid");
    if (ppack->pmhd->presist != nullptr || ppack->pmhd->pbiermann != nullptr ||
        pin->DoesBlockExist("shearing_box")) {
      Error("hybrid Ohm's law requires mhd resistivity, Biermann, "
            "and shearing_box disabled");
    }
    const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
    for (int f = 0; f < 2*ndim; ++f) {
      const auto bc = mesh->mesh_bcs[f];
      if (bc != BoundaryFlag::outflow && bc != BoundaryFlag::periodic) {
        Error("hybrid fields currently support outflow or periodic physical boundaries");
      }
      if (periodic_ && bc != BoundaryFlag::periodic) {
        Error("periodic_transport with hybrid fields requires periodic fluid boundaries");
      }
    }
    const Real pressure_unit = energy_cgs_/(length_cgs_*length_cgs_*length_cgs_);
    bunit_ = std::sqrt(4.0*kPi*pressure_unit);
    current_unit_ = kLightSpeedCgs*bunit_/(4.0*kPi*length_cgs_);
  }
  // The binary footer owns the normalized spectrum on restart. The external PIC
  // distribution is not needed again, and cannot silently change the saved source.
  if (!pin->DoesParameterExist(b, "restart_version")) LoadSpectrum(pin);

  auto &ind = mesh->mb_indcs;
  const int nmb = std::max(ppack->nmb_thispack, mesh->nmb_maxperrank);
  const int n1 = ind.nx1+2*ind.ng;
  const int n2 = mesh->multi_d ? ind.nx2+2*ind.ng : 1;
  const int n3 = mesh->three_d ? ind.nx3+2*ind.ng : 1;
  Kokkos::realloc(cell_data, nmb, ncell_data, n3, n2, n1);
  Kokkos::realloc(feedback_, nmb, 4, n3, n2, n1);
  Kokkos::deep_copy(cell_data, 0.0);
  if (mesh->multilevel) {
    Kokkos::realloc(coarse_cell_data, nmb, ncell_data,
        mesh->three_d ? ind.cnx3+2*ind.ng : 1,
        mesh->multi_d ? ind.cnx2+2*ind.ng : 1, ind.cnx1+2*ind.ng);
    Kokkos::deep_copy(coarse_cell_data, 0.0);
  }
  if (electromagnetic_) {
    Kokkos::realloc(current_, nmb, 3, n3, n2, n1);
    if (mesh->multilevel) {
      Kokkos::realloc(coarse_current_, nmb, 3,
          mesh->three_d ? ind.cnx3+2*ind.ng : 1,
          mesh->multi_d ? ind.cnx2+2*ind.ng : 1, ind.cnx1+2*ind.ng);
    }
    Kokkos::realloc(emf_.x1e, nmb, n3+1, n2+1, n1);
    Kokkos::realloc(emf_.x2e, nmb, n3+1, n2, n1+1);
    Kokkos::realloc(emf_.x3e, nmb, n3, n2+1, n1+1);
    Kokkos::realloc(energy_flux_.x1f, nmb, 1, n3, n2, n1+1);
    Kokkos::realloc(energy_flux_.x2f, nmb, 1, n3, n2+1, n1);
    Kokkos::realloc(energy_flux_.x3f, nmb, 1, n3+1, n2, n1);
    current_bvals_ = new MeshBoundaryValuesCC(ppack, pin, false);
    current_bvals_->InitializeBuffers(3, 1);
    current_bvals_->EnableUniformBatch();
    ppack->pmhd->pbval_u->EnableUniformBatch();
    ppack->pmhd->pbval_b->EnableUniformBatch();
  }
  RefreshMesh();
}

FastElectrons::~FastElectrons() {
  delete current_bvals_;
}

void FastElectrons::LoadSpectrum(ParameterInput *pin) {
  const std::string b = "fast_electrons";
  const std::string file = pin->GetOrAddString(b, "distribution_file", "");
  std::vector<Real> rows;
  if (global_variable::my_rank == 0) {
    if (file.empty()) {
      rows = {pin->GetOrAddReal(b, "energy_kev", 100.0),
              pin->GetOrAddReal(b, "theta_deg", 0.0),
              pin->GetOrAddReal(b, "phi_deg", 0.0), 1.0};
    } else {
      std::ifstream stream(file);
      if (!stream) Error("cannot open distribution_file '" + file + "'");
      std::string line;
      int lineno = 0;
      while (std::getline(stream, line)) {
        ++lineno;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::replace(line.begin(), line.end(), ',', ' ');
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::istringstream row(line);
        Real e, theta, phi, weight;
        std::string extra;
        if (!(row >> e >> theta >> phi >> weight) || (row >> extra)) {
          Error("distribution line " + std::to_string(lineno) +
                " needs four columns: energy_kev theta_deg phi_deg number_weight");
        }
        rows.insert(rows.end(), {e, theta, phi, weight});
        if (rows.size()/4 > static_cast<std::size_t>(max_packets_/packets_per_bin_)) {
          Error("distribution exceeds max_packets/packets_per_bin");
        }
      }
      if (stream.bad()) Error("failed reading distribution_file");
    }
  }
#if MPI_PARALLEL_ENABLED
  if (rows.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    Error("distribution exceeds MPI count limit");
  }
  int count = static_cast<int>(rows.size());
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  rows.resize(count);
  MPI_Bcast(rows.data(), count, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
  if (rows.empty()) Error("distribution must contain a positive number weight");
  Real max_weight = 0.0;
  for (std::size_t i = 0; i < rows.size(); i += 4) {
    Positive(rows[i], "distribution energy_kev");
    Positive(rows[i+3], "distribution number_weight", true);
    if (!std::isfinite(rows[i+1]) || rows[i+1] < 0.0 || rows[i+1] > 180.0 ||
        !std::isfinite(rows[i+2])) Error("theta must lie in [0,180]; phi must be finite");
    max_weight = std::max(max_weight, rows[i+3]);
  }
  Positive(max_weight, "distribution total weight");
  Real weighted_energy = 0.0;
  constexpr Real radians = 3.14159265358979323846/180.0;
  for (std::size_t i = 0; i < rows.size(); i += 4) {
    if (rows[i+3] == 0.0) continue;
    const Real theta = rows[i+1]*radians;
    const Real phi = std::remainder(rows[i+2], 360.0)*radians;
    SpectrumBin bin{rows[i]*1.0e3*kEvToErg,
                    {std::cos(theta), std::sin(theta)*std::cos(phi),
                     std::sin(theta)*std::sin(phi)}, rows[i+3]/max_weight};
    // Remove trig roundoff at exactly axial angles; tangential source rows must
    // remain tangential rather than leaking through an injection boundary.
    for (auto &n : bin.n) {
      if (std::abs(n) < 8*std::numeric_limits<Real>::epsilon()) n = 0.0;
    }
    weighted_energy += bin.weight*bin.energy;
    spectrum_.push_back(bin);
  }
  Positive(weighted_energy, "distribution weighted energy");
  for (auto &bin : spectrum_) bin.weight /= weighted_energy;
  if (spectrum_.size() > static_cast<std::size_t>(max_packets_/packets_per_bin_)) {
    Error("source batch exceeds max_packets");
  }
}

void FastElectrons::PrepareSource() {
  // Ownership depends on the direction and mesh, not on emission time. Locate each
  // spectrum row once per mesh and retain its global row for the packet RNG sequence.
  auto mesh = pmy_pack_->pmesh;
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  std::vector<ElectronPacket> local;
  if (source_points_.size() > static_cast<std::size_t>(max_packets_/packets_per_bin_)/
                               spectrum_.size()) {
    Error("source spatial/spectrum quadrature exceeds max_packets");
  }
  source_injected_weight_ = source_escaped_weight_ = 0.0;
  for (std::size_t n = 0; n < spectrum_.size(); ++n) {
    const auto &bin = spectrum_[n];
    for (std::size_t s = 0; s < source_points_.size(); ++s) {
      const auto &point = source_points_[s];
      ElectronPacket p{};
      p.energy = bin.energy;
      p.weight = bin.weight*point.weight;
      p.rng = n*source_points_.size()+s;
      for (int d = 0; d < 3; ++d) {
        p.x[d] = origin_[d];
        p.n[d] = bin.n[d];
      }
      p.x[1] = point.x2;
      p.x[2] = point.x3;
      LocatePacket(p, host_tree_, host_blocks_, mesh->mesh_size, mesh->mb_indcs,
                   ndim, periodic_);
      if (p.status == 3) Error("cannot locate injected packet");
      const int owner = p.status == 2 ? 0 : Owner(p);
      if (owner != global_variable::my_rank) continue;
      source_injected_weight_ += p.weight*p.energy;
      if (p.status == 2) source_escaped_weight_ += p.weight*p.energy;
      else local.push_back(p);
    }
  }
  source_packets_ = DvceArray1D<ElectronPacket>();
  memory::Preflight(local.size()*sizeof(ElectronPacket), "source templates");
  source_packets_ = DvceArray1D<ElectronPacket>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-source-templates"), local.size());
  auto host = Kokkos::create_mirror_view(source_packets_);
  for (std::size_t n = 0; n < local.size(); ++n) host(n) = local[n];
  Kokkos::deep_copy(source_packets_, host);
  source_ready_ = true;
}

void FastElectrons::Inject(Real time, Real dt) {
  memory::Time(time);
  memory::Context("injection", device_packets_.extent(0));
  if (emission_interval_ > 0.0) { InjectScheduled(time, dt); return; }
  const Real begin = std::max(time, start_);
  const Real end = std::min(time+dt, end_);
  if (end <= begin || power_ == 0.0) return;
  if (!source_ready_) PrepareSource();
  const Real source_energy = power_*(end-begin)*time_cgs_;
  Positive(source_energy, "injected energy");
  const int samples = packets_per_bin_;
  const auto old_count = device_packets_.extent(0);
  const auto added = source_packets_.extent(0)*samples;
  ReservePackets(old_count+added);
  device_packets_ = Kokkos::subview(packet_storage_,
      std::make_pair(std::size_t(0), old_count+added));
  auto packets = device_packets_;
  auto source = source_packets_;
  const auto seed = seed_, next = next_packet_;
  par_for("fe_inject", DevExeSpace(), 0, static_cast<int>(added)-1,
  KOKKOS_LAMBDA(int idx) {
    auto p = source(idx/samples);
    const int q = idx%samples;
    p.weight = source_energy*p.weight/samples;
    if (!(p.weight > 0.0) || !Kokkos::isfinite(p.weight)) {
      Kokkos::abort("Invalid fast-electron packet number weight");
    }
    const Real emitted_at = begin+(q+0.5)*(end-begin)/samples;
    p.remaining = time+dt-emitted_at;
    p.rng = seed+(next+p.rng*samples+q+1)*0x9e3779b97f4a7c15ULL;
    packets(old_count+idx) = p;
  });
  next_packet_ += spectrum_.size()*source_points_.size()*samples;
  injected_ += source_energy*source_injected_weight_;
  escaped_ += source_energy*source_escaped_weight_;
  memory::Context("injected_trial", device_packets_.extent(0));
}

void FastElectrons::InjectScheduled(Real time, Real dt) {
  if (power_ == 0.0 || emission_bins_ == 0) return;
  const auto rows = spectrum_.size()*source_points_.size();
  if (next_packet_%rows != 0) Error("incompatible fixed emission event ordinal");
  const auto first_event = next_packet_/rows;
  const auto samples = packets_per_bin_;
  const auto total_events = emission_bins_*static_cast<std::uint64_t>(samples);
  if (total_events > std::numeric_limits<std::uint64_t>::max()/rows) {
    Error("fixed emission sequence exceeds the particle identifier range");
  }
  const auto event_time = [&](std::uint64_t event) {
    const Real begin = start_+(event/samples)*emission_interval_;
    const Real width = std::min(emission_interval_, end_-begin);
    return begin+(event%samples+0.5)*width/samples;
  };
  if (first_event > total_events) Error("fixed emission ordinal exceeds the schedule");
  if (first_event == total_events || event_time(first_event) >= time+dt) return;
  const Real tolerance = 64*std::numeric_limits<Real>::epsilon()*
      std::max({std::abs(time), std::abs(start_), std::abs(end_)});
  if (event_time(first_event) < time-tolerance) Error("fixed emission schedule is behind time");
  if (!source_ready_) PrepareSource();
  const auto old_count = device_packets_.extent(0);
  const auto local_rows = source_packets_.extent(0);
  auto lo = first_event, hi = total_events;
  while (lo < hi) {
    const auto middle = lo+(hi-lo)/2;
    if (event_time(middle) < time+dt) lo = middle+1;
    else hi = middle;
  }
  const auto events = lo-first_event;
  if (local_rows && events > (static_cast<std::size_t>(max_packets_)-old_count)/local_rows) {
    Error("fixed emission batch exceeds max_packets on a rank");
  }
  const auto last_bin_first = (emission_bins_-1)*static_cast<std::uint64_t>(samples);
  const auto last_events = lo > last_bin_first ? lo-std::max(first_event, last_bin_first) : 0;
  const Real last_width = end_-(start_+(emission_bins_-1)*emission_interval_);
  const Real emitted_duration = (events-last_events)*(emission_interval_/samples)+
                                 last_events*(last_width/samples);
  const auto added = local_rows*events;
  ReservePackets(old_count+added);
  device_packets_ = Kokkos::subview(packet_storage_,
      std::make_pair(std::size_t(0), old_count+added));
  const auto packets = device_packets_, source = source_packets_;
  const Real start = start_, end = end_, interval = emission_interval_;
  const Real power_time = power_*time_cgs_;
  const auto seed = seed_;
  par_for("fe_inject_scheduled", DevExeSpace(), 0, static_cast<int>(added)-1,
  KOKKOS_LAMBDA(int idx) {
    const auto event = first_event+idx/local_rows;
    auto p = source(idx%local_rows);
    const Real begin = start+(event/samples)*interval;
    const Real width = fmin(interval, end-begin);
    const Real emitted_at = begin+(event%samples+0.5)*width/samples;
    p.weight *= power_time*width/samples;
    if (!(p.weight > 0.0) || !Kokkos::isfinite(p.weight)) {
      Kokkos::abort("Invalid fixed emission packet weight");
    }
    p.remaining = fmax(0.0, time+dt-emitted_at);
    p.rng = seed+(event*rows+p.rng+1)*0x9e3779b97f4a7c15ULL;
    packets(old_count+idx) = p;
  });
  next_packet_ += events*rows;
  injected_ += power_time*emitted_duration*source_injected_weight_;
  escaped_ += power_time*emitted_duration*source_escaped_weight_;
  memory::Context("injected_trial", device_packets_.extent(0));
}

void FastElectrons::Advance(Real time, Real dt, Driver *driver) {
  ReportMemory("advance_begin", time);
  memory::Preflight(0, "advance");
  Positive(dt, "transport timestep");
  if (spectrum_.empty()) Error("source/restart spectrum was not initialized");
  auto data = cell_data;
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  par_for("fe_clear_step", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    data(m, 0, k, j, i) = 0.0;
    data(m, 4, k, j, i) = 0.0;
  });
  Real advanced = 0.0;
  int field_steps = 0;
  const Real saved_time = pmy_pack_->pmesh->time;
  while (advanced < dt) {
    if (++field_steps > max_field_steps_) Error("max_field_substeps exceeded");
    Real h = electromagnetic_ ? std::min(dt-advanced, FieldTimeStep(false)) : dt;
    // An injected current can make the electric acceleration limit stricter. Retry
    // emission with the shorter interval before changing any fluid/particle state.
    const auto saved_count = device_packets_.extent(0);
    const Real saved_injected = injected_, saved_escaped = escaped_;
    const auto saved_next = next_packet_;
    for (int attempt = 0;; ++attempt) {
      if (attempt > 20) Error("hybrid electric timestep failed to converge");
      device_packets_ = Kokkos::subview(packet_storage_,
          std::make_pair(std::size_t(0), saved_count));
      injected_ = saved_injected; escaped_ = saved_escaped; next_packet_ = saved_next;
      memory::Context("injection_retry", saved_count);
      auto packets = device_packets_;
      par_for("fe_reset_flight", DevExeSpace(), 0, static_cast<int>(saved_count)-1,
      KOKKOS_LAMBDA(int q) { packets(q).remaining = h; });
      Inject(time+advanced, h);
      if (!electromagnetic_) break;
      if (diffusivity_ > 0.0) DepositMoments(h);
      BuildFields();
      const Real limit = FieldTimeStep(true);
      if (h <= limit*(1.0+32*std::numeric_limits<Real>::epsilon())) break;
      h = std::min(h*0.9, limit*0.9);
    }
    if (!(h > 0.0) || advanced+h == advanced) Error("unresolvable hybrid timestep");
    ReportMemory("injected", time+advanced);
    DiagnoseParticles("injected", time+advanced);
    Kokkos::deep_copy(feedback_, 0.0);
    Transport();
    ReportMemory("transported", time+advanced+h);
    DiagnoseParticles("transported", time+advanced+h);
    ++diagnostic_substeps_;
    if (electromagnetic_) AdvanceFields(h);
    ApplyFeedback(h);
    advanced += h;
    if (electromagnetic_) {
      pmy_pack_->pmesh->time = time+advanced;
      if (diffusivity_ > 0.0) {
        driver->InitBoundaryValuesAndPrimitives(pmy_pack_->pmesh);
      } else {
        // The ideal particle operator updates active cells only and leaves B
        // fixed. Refresh the local two-temperature closure for the next force;
        // Driver exchanges/restricts/prolongates the full MHD state on return.
        auto mhd = pmy_pack_->pmhd;
        mhd->peos->ConsToPrim(mhd->u0, mhd->b0, mhd->w0, mhd->bcc0, false,
                              ind.is, ind.ie, ind.js, ind.je, ind.ks, ind.ke);
        mhd->ptwo_temp->CloseBiermannStage(mhd->u0, mhd->w0,
                              ind.is, ind.ie, ind.js, ind.je, ind.ks, ind.ke);
      }
    }
  }
  pmy_pack_->pmesh->time = saved_time;
  DepositMoments();
  if (electromagnetic_) BuildFields();
  par_for("fe_average_heat", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    data(m, 0, k, j, i) /= dt;
    data(m, 4, k, j, i) /= dt;
  });
  Diagnose();
  if (reclaim_packet_memory_) ReclaimPacketMemory();
  ReportMemory("advance_end", time+dt);
}

void FastElectrons::Transport() {
  if (global_variable::nranks > 1) {
    TransportDistributed();
    return;
  }
  int global_work = 1;
  for (int wave = 0; global_work != 0; ++wave) {
    if (wave >= max_waves_) {
      Error("max_transport_waves exceeded; no energy was discarded");
    }
    memory::Context("trace_prepare", device_packets_.extent(0));
    TraceLocal();
    auto packets = device_packets_;
    auto blocks = blocks_;
    const int rank = global_variable::my_rank;
    int live = 0, work = 0, outgoing = 0, failed = 0;
    Real escaped = 0.0;
    Kokkos::parallel_reduce("fe_transport_status",
        Kokkos::RangePolicy<DevExeSpace>(0, packets.extent_int(0)),
    KOKKOS_LAMBDA(int q, int &nlive, int &todo, int &send, int &bad, Real &loss) {
      const auto p = packets(q);
      if (p.status == 0) {
        ++nlive;
        if (p.remaining > 0.0) todo = 1;
        if (blocks(p.gid).rank != rank) ++send;
      } else if (p.status == 2) {
        loss += p.weight*p.energy;
      } else if (p.status == 3) {
        bad = 1;
      }
    }, Kokkos::Sum<int>(live), Kokkos::Max<int>(work),
       Kokkos::Sum<int>(outgoing), Kokkos::Max<int>(failed),
       Kokkos::Sum<Real>(escaped));
    if (failed > 0) Error("invalid plasma state or unresolvable particle timestep");
    escaped_ += escaped;
    memory::Context("compaction", live);
    if (live != packets.extent_int(0)) {
      // Stable compaction preserves packet order and each packet's RNG. Allocate
      // scratch only when a packet is actually removed, and retain it for reuse.
      if (packet_scratch_.extent(0) != packet_storage_.extent(0)) {
        packet_scratch_ = DvceArray1D<ElectronPacket>();
        memory::Preflight(packet_storage_.extent(0)*sizeof(ElectronPacket), "compaction");
        packet_scratch_ = DvceArray1D<ElectronPacket>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-packet-scratch"),
            packet_storage_.extent(0));
      }
      auto compact = packet_scratch_;
      Kokkos::parallel_scan("fe_compact_packets",
          Kokkos::RangePolicy<DevExeSpace>(0, packets.extent_int(0)),
      KOKKOS_LAMBDA(int q, int &offset, bool final) {
        if (packets(q).status == 0) {
          if (final) compact(offset) = packets(q);
          ++offset;
        }
      });
      std::swap(packet_storage_, packet_scratch_);
      device_packets_ = Kokkos::subview(packet_storage_,
          std::make_pair(std::size_t(0), static_cast<std::size_t>(live)));
    }
    int flags[2] = {work > 0 ? 1 : 0, outgoing > 0 ? 1 : 0};
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif
    global_work = flags[0];
    if (flags[1]) {
      // Transfer only emigrants to the host. Locally retained packets stay on
      // the device even when another rank has particles crossing an interface.
      packets_.clear();
      if (outgoing > 0) {
        if (packet_scratch_.extent(0) != packet_storage_.extent(0)) {
          packet_scratch_ = DvceArray1D<ElectronPacket>();
          memory::Preflight(packet_storage_.extent(0)*sizeof(ElectronPacket), "migration scratch");
          packet_scratch_ = DvceArray1D<ElectronPacket>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-packet-scratch"),
              packet_storage_.extent(0));
        }
        memory::Preflight(outgoing*sizeof(ElectronPacket), "migration exports");
        DvceArray1D<ElectronPacket> exports(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fe-export-packets"), outgoing);
        auto traveling = device_packets_;
        auto retained = packet_scratch_;
        Kokkos::parallel_scan("fe_partition_migrants",
            Kokkos::RangePolicy<DevExeSpace>(0, traveling.extent_int(0)),
        KOKKOS_LAMBDA(int q, int &offset, bool final) {
          const auto p = traveling(q);
          const bool remote = blocks(p.gid).rank != rank;
          if (final) {
            if (remote) exports(offset) = p;
            else retained(q-offset) = p;
          }
          if (remote) ++offset;
        });
        std::swap(packet_storage_, packet_scratch_);
        device_packets_ = Kokkos::subview(packet_storage_,
            std::make_pair(std::size_t(0), static_cast<std::size_t>(live-outgoing)));
        auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), exports);
        packets_.resize(outgoing);
        for (int q = 0; q < outgoing; ++q) packets_[q] = host(q);
      }
      // All ranks participate, including receivers with no local packets. This
      // host exchange now receives just the new arrivals into packets_.
      ExchangePackets();
      if (!packets_.empty()) {
        const auto retained = device_packets_.extent(0);
        const auto count = retained+packets_.size();
        ReservePackets(count);
        device_packets_ = Kokkos::subview(packet_storage_,
            std::make_pair(std::size_t(0), count));
        auto tail = Kokkos::subview(device_packets_, std::make_pair(retained, count));
        Kokkos::View<const ElectronPacket*, HostMemSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(
            packets_.data(), packets_.size());
        Kokkos::deep_copy(tail, host);
      }
    }
  }
}

void FastElectrons::ExchangePackets() {
#if MPI_PARALLEL_ENABLED
  const int nranks = global_variable::nranks;
  if (nranks == 1) return;
  std::vector<std::vector<ElectronPacket>> outgoing(nranks);
  std::vector<ElectronPacket> retained;
  for (auto &p : packets_) {
    const int owner = Owner(p);
    if (owner == global_variable::my_rank) retained.push_back(p);
    else outgoing[owner].push_back(p);
  }
  std::vector<int> send_count(nranks), recv_count(nranks), send_offset(nranks),
                   recv_offset(nranks);
  std::size_t send_size = 0;
  for (int r = 0; r < nranks; ++r) {
    const auto bytes = outgoing[r].size()*sizeof(ElectronPacket);
    if (send_size+bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      Error("outgoing packets exceed MPI byte-count limit");
    }
    send_offset[r] = static_cast<int>(send_size);
    send_count[r] = static_cast<int>(bytes);
    send_size += bytes;
  }
  MPI_Alltoall(send_count.data(), 1, MPI_INT,
               recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::size_t recv_size = 0;
  for (int r = 0; r < nranks; ++r) {
    if (recv_count[r] < 0 || recv_count[r]%sizeof(ElectronPacket) != 0 ||
        recv_size+recv_count[r] >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      Error("incoming packets exceed MPI byte-count limit");
    }
    recv_offset[r] = static_cast<int>(recv_size);
    recv_size += recv_count[r];
  }
  if (retained.size()+recv_size/sizeof(ElectronPacket) >
      static_cast<std::size_t>(max_packets_)) {
    Error("incoming population exceeds max_packets");
  }
  std::vector<ElectronPacket> send, recv(recv_size/sizeof(ElectronPacket));
  send.reserve(send_size/sizeof(ElectronPacket));
  for (auto &v : outgoing) send.insert(send.end(), v.begin(), v.end());
  MPI_Alltoallv(send.data(), send_count.data(), send_offset.data(), MPI_BYTE,
                recv.data(), recv_count.data(), recv_offset.data(), MPI_BYTE,
                MPI_COMM_WORLD);
  retained.insert(retained.end(), recv.begin(), recv.end());
  packets_.swap(retained);
#endif
}

void FastElectrons::TraceLocal() {
  auto mesh = pmy_pack_->pmesh;
  const auto ind = mesh->mb_indcs;
  const auto domain = mesh->mesh_size;
  const Real width[3] = {domain.x1max-domain.x1min, domain.x2max-domain.x2min,
                         domain.x3max-domain.x3min};
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  const int rank = global_variable::my_rank, first_gid = pmy_pack_->gids;
  auto packet = device_packets_;
  auto blocks = blocks_;
  auto tree = tree_;
  auto data = feedback_;
  const int replicas = PrepareDeposits();
  auto scratch = deposit_scratch_;
  auto diagnostics = cell_data;
  const bool electromagnetic = electromagnetic_;
  DvceArray5D<Real> bcc;
  if (electromagnetic) bcc = pmy_pack_->pmhd->bcc0;
  const Real gyro = -kElectronChargeCgs*bunit_*time_cgs_/
                    (kElectronMassCgs*kLightSpeedCgs);
  const Real momentum_unit = energy_cgs_*time_cgs_/length_cgs_;
  const Real momentum_scale = kElectronRestEnergyCgs/(kLightSpeedCgs*momentum_unit);
  auto u = use_mhd_ ? pmy_pack_->pmhd->u0 : pmy_pack_->phydro->u0;
  auto w = use_mhd_ ? pmy_pack_->pmhd->w0 : pmy_pack_->phydro->w0;
  auto tt = use_mhd_ ? pmy_pack_->pmhd->ptwo_temp : pmy_pack_->phydro->ptwo_temp;
  auto temperature = tt->temperature;
  auto thermo = tt->thermodynamics;
  auto material = use_mhd_ ? pmy_pack_->pmhd->pmaterials : pmy_pack_->phydro->pmaterials;
  materials::MaterialMixtureDevice mixture;
  if (use_materials_) mixture = material->DeviceData();
  const bool materials = use_materials_, collisions = collisions_;
  const bool scattering = scattering_, periodic = periodic_;
  const Real density_scale = density_cgs_, temperature_scale = temperature_cgs_;
  const Real unit_mu = pmy_pack_->punit->mu();
  const Real number_per_gram = number_per_gram_, zeff = zeff_, logl = coulomb_log_;
  const Real energy_unit = energy_cgs_, time_unit = time_cgs_;
  const Real c_code = kLightSpeedCgs*time_cgs_/length_cgs_;
  const Real cutoff = cutoff_, thermal_factor = thermal_factor_;
  const Real fraction = loss_fraction_, max_scatter = scatter_limit_;
  const int max_segments = max_segments_;
  const Real eps = std::numeric_limits<Real>::epsilon();

  memory::Context("trace", 0, false);
  par_for("fe_transport", DevExeSpace(), 0, static_cast<int>(packet.extent(0))-1,
  KOKKOS_LAMBDA(int q) {
    auto p = packet(q);
    for (int segment = 0; segment < max_segments && p.remaining > 0.0; ++segment) {
      LocatePacket(p, tree, blocks, domain, ind, ndim, periodic);
      if (p.status != 0) break;
      const auto block = blocks(p.gid);
      if (block.rank != rank) break;
      const int m = p.gid-first_gid;
      const int i = p.cell[0]+ind.is;
      const int j = p.cell[1]+ind.js;
      const int k = p.cell[2]+ind.ks;
      const int r = q & (replicas-1);
      Real *deposit[4];
      for (int v = 0; v < 4; ++v) {
        deposit[v] = replicas > 1 ? &scratch(m, v, p.cell[2], p.cell[1], p.cell[0], r)
                                   : &data(m, v, k, j, i);
      }
      const Real lo[3] = {block.x1min, block.x2min, block.x3min};
      const Real hi[3] = {block.x1max, block.x2max, block.x3max};
      const Real dx[3] = {block.dx1, block.dx2, block.dx3};
      const int nc[3] = {ind.nx1, ind.nx2, ind.nx3};
      const Real volume = dx[0]*dx[1]*dx[2];
      const Real old_energy = p.energy;
      const Real q0 = old_energy/kElectronRestEnergyCgs;
      const Real p0 = sqrt(q0)*sqrt(q0+2.0);
      const Real old_momentum[3] = {p0*p.n[0], p0*p.n[1], p0*p.n[2]};

      Real ne = 0.0, local_z = zeff, te = 0.0;
      if (collisions) {
        ne = u(m, IDN, k, j, i)*density_scale*number_per_gram;
        te = temperature(m, 1, k, j, i)*temperature_scale;
        if (materials) {
          ne = thermo(m, two_temperature::TwoTemperature::electron_number_density_cgs,
                      k, j, i);
          local_z = thermo(m, two_temperature::TwoTemperature::effective_charge, k, j, i);
          if (mixture.UsesTabularEOS()) {
            te = temperature(m, 1, k, j, i)*mixture.temperature_to_kelvin;
          } else {
            const auto composition = mixture.CompositionFromPrimitive(w, m, k, j, i);
            te *= mixture.MeanParticleMass(composition)/unit_mu;
          }
        }
        if (!Kokkos::isfinite(ne) || ne < 0.0 || !Kokkos::isfinite(te) || te < 0.0 ||
            !Kokkos::isfinite(local_z) || local_z < 0.0) {
          p.status = 3;
          break;
        }
        if (ne > 0.0 && p.energy <= fmax(cutoff, thermal_factor*kBoltzmannCgs*te)) {
          Kokkos::atomic_add(deposit[0],
                             p.weight*p.energy/(energy_unit*volume));
          if (electromagnetic) {
            for (int d = 0; d < 3; ++d) {
              Kokkos::atomic_add(deposit[1+d],
                  p.weight*momentum_scale*old_momentum[d]/volume);
            }
          }
          p.energy = 0.0;
          p.remaining = 0.0;
          p.status = 1;
          break;
        }
      }
      const Real speed = c_code*Beta(p.energy);
      const Real loss_rate = collisions
          ? StoppingRate(p.energy, ne, logl)*time_unit : 0.0;
      const Real nu = collisions && scattering
          ? DeflectionRate(p.energy, ne, local_z, logl)*time_unit : 0.0;
      if ((!electromagnetic && !(speed > 0.0)) ||
          !Kokkos::isfinite(loss_rate) || !Kokkos::isfinite(nu)) {
        p.status = 3;
        break;
      }
      // Cell-face-limited drift deposits only in the cell actually traversed. Exact
      // integer transitions avoid position nudges, lost path length, and rank-face gaps.
      Real face_dt[3] = {std::numeric_limits<Real>::max(),
                          std::numeric_limits<Real>::max(),
                          std::numeric_limits<Real>::max()};
      Real face[3] = {p.x[0], p.x[1], p.x[2]};
      Real h = p.remaining;
      if (loss_rate > 0.0) h = fmin(h, fraction*p.energy/loss_rate);
      if (nu > 0.0) h = fmin(h, max_scatter/nu);
      for (int d = 0; d < ndim; ++d) {
        if (p.n[d] == 0.0 || speed == 0.0) continue;
        const int f = p.cell[d]+(p.n[d] > 0.0 ? 1 : 0);
        face[d] = f == 0 ? lo[d] : (f == nc[d] ? hi[d] : lo[d]+f*dx[d]);
        const Real distance = (face[d]-p.x[d])/p.n[d];
        if (distance < -64*eps*fmax(width[d], fabs(p.x[d]))) {
          p.status = 3;
          break;
        }
        face_dt[d] = fmax(distance, 0.0)/speed;
        h = fmin(h, face_dt[d]);
      }
      if (p.status != 0) break;
      if (h > 0.0 && p.remaining-h == p.remaining) {
        p.status = 3;
        break;
      }
      const Real lost = fmin(p.energy, loss_rate*h);
      p.energy -= lost;
      p.remaining = fmax(p.remaining-h, 0.0);
      for (int d = 0; d < ndim; ++d) {
        p.x[d] += speed*p.n[d]*h;
        if (face_dt[d] <= h) {
          p.x[d] = face[d];
        }
      }
      if (nu*h > 0.0) ScatterDirection(nu*h, p.rng, p.n[0], p.n[1], p.n[2]);
      if (electromagnetic) {
        Real electric[3], magnetic[3];
        for (int d = 0; d < 3; ++d) {
          electric[d] = diagnostics(m, 8+d, k, j, i);
          magnetic[d] = bcc(m, d, k, j, i);
        }
        BorisKick(p.energy, p.n, electric, magnetic, gyro, c_code, h);
        const Real q1 = p.energy/kElectronRestEnergyCgs;
        const Real p1 = sqrt(q1)*sqrt(q1+2.0);
        for (int d = 0; d < 3; ++d) {
          Kokkos::atomic_add(deposit[1+d],
              p.weight*momentum_scale*(old_momentum[d]-p1*p.n[d])/volume);
        }
      }
      Kokkos::atomic_add(deposit[0],
          p.weight*(old_energy-p.energy)/(energy_unit*volume));
    }
    if (p.status == 0) LocatePacket(p, tree, blocks, domain, ind, ndim, periodic);
    if (!Kokkos::isfinite(p.energy) || p.energy < 0.0) p.status = 3;
    packet(q) = p;
  });
  if (replicas > 1) {
    par_for("fe_merge_feedback", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
            ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int v = 0; v < (electromagnetic ? 4 : 1); ++v) {
        Real sum = 0.0;
        for (int r = 0; r < replicas; ++r) {
          sum += scratch(m, v, k-ind.ks, j-ind.js, i-ind.is, r);
        }
        data(m, v, k, j, i) += sum;
      }
    });
  }
}

void FastElectrons::ApplyFeedback(Real dt) {
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  auto u = use_mhd_ ? pmy_pack_->pmhd->u0 : pmy_pack_->phydro->u0;
  auto data = cell_data;
  auto feedback = feedback_;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  const int ie = electron_index_;
  const bool em = electromagnetic_;
  const bool multi_d = pmy_pack_->pmesh->multi_d, three_d = pmy_pack_->pmesh->three_d;
  DvceArray5D<Real> bcc;
  DvceArray4D<Real> bx, by, bz;
  if (em) {
    bcc = pmy_pack_->pmhd->bcc0;
    bx = pmy_pack_->pmhd->b0.x1f;
    by = pmy_pack_->pmhd->b0.x2f;
    bz = pmy_pack_->pmhd->b0.x3f;
  }
  auto flux = energy_flux_;
  auto emf = emf_;
  const Real eta = diffusivity_;
  par_for("fe_apply_feedback", DevExeSpace(), 0, pmy_pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real de = feedback(m, 0, k, j, i);
    Real heat = de;
    data(m, 1, k, j, i) += de;
    if (em) {
      if (eta > 0.0) {
        de -= dt*(flux.x1f(m, 0, k, j, i+1)-flux.x1f(m, 0, k, j, i))/size(m).dx1;
        if (multi_d) {
          de -= dt*(flux.x2f(m, 0, k, j+1, i)-flux.x2f(m, 0, k, j, i))/size(m).dx2;
        }
        if (three_d) {
          de -= dt*(flux.x3f(m, 0, k+1, j, i)-flux.x3f(m, 0, k, j, i))/size(m).dx3;
        }
      }
      Real bn[3] = {0.0, 0.0, 0.0};
      if (eta > 0.0) {
        bn[0] = 0.5*(bx(m, k, j, i)+bx(m, k, j, i+1));
        bn[1] = 0.5*(by(m, k, j, i)+by(m, k, j+1, i));
        bn[2] = 0.5*(bz(m, k, j, i)+bz(m, k+1, j, i));
      }
      heat = de;
      for (int d = 0; d < 3; ++d) {
        const Real old_p = u(m, IM1+d, k, j, i);
        const Real new_p = old_p+feedback(m, 1+d, k, j, i);
        heat -= 0.5*(new_p-old_p)*(new_p+old_p)/u(m, IDN, k, j, i);
        if (eta > 0.0) {
          heat -= 0.5*(bn[d]-bcc(m, d, k, j, i))*(bn[d]+bcc(m, d, k, j, i));
        }
        u(m, IM1+d, k, j, i) = new_p;
      }
      if (eta > 0.0) {
        // Diagnostic eta*J_background^2 uses the same edge Ohm field as CT.
        // Electron heating includes its conservative discretization correction.
        Real e2sum = 0.0;
        for (int a = 0; a < 2; ++a) {
          for (int c = 0; c < 2; ++c) {
            const Real ex = emf.x1e(m, k+a, j+c, i);
            const Real ey = emf.x2e(m, k+a, j, i+c);
            const Real ez = emf.x3e(m, k, j+a, i+c);
            e2sum += ex*ex+ey*ey+ez*ez;
          }
        }
        data(m, 4, k, j, i) += dt*0.25*e2sum/eta;
      }
    }
    u(m, IEN, k, j, i) += de;
    u(m, ie, k, j, i) += heat;
    data(m, 0, k, j, i) += heat;
    if (!Kokkos::isfinite(u(m, ie, k, j, i)) || u(m, ie, k, j, i) < 0.0) {
      Kokkos::abort("Hybrid electron energy became negative; reduce field_cfl");
    }
  });
}

void FastElectrons::Diagnose() {
  const auto ind = pmy_pack_->pmesh->mb_indcs;
  auto data = cell_data;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  // Diagnostics use the actual deposited grid energy, so a lost particle or duplicated
  // MPI transfer cannot be hidden by merely balancing particle-local loss counters.
  Real deposited_density = 0.0;
  Kokkos::parallel_reduce("fe_grid_energy", Kokkos::RangePolicy<DevExeSpace>(
      0, pmy_pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3),
  KOKKOS_LAMBDA(int idx, Real &sum) {
    const int i = idx%ind.nx1+ind.is;
    idx /= ind.nx1;
    const int j = idx%ind.nx2+ind.js;
    idx /= ind.nx2;
    const int k = idx%ind.nx3+ind.ks;
    const int m = idx/ind.nx3;
    sum += data(m, 1, k, j, i)*size(m).dx1*size(m).dx2*size(m).dx3;
  }, deposited_density);
  auto packets = device_packets_;
  Real active = 0.0;
  Kokkos::parallel_reduce("fe_active_energy",
      Kokkos::RangePolicy<DevExeSpace>(0, packets.extent_int(0)),
  KOKKOS_LAMBDA(int q, Real &sum) {
    sum += packets(q).weight*packets(q).energy;
  }, active);
  Real ledger[5] = {injected_, deposited_density*energy_cgs_, escaped_, active,
                    static_cast<Real>(packets.extent(0))};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, ledger, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real residual = ledger[0]-ledger[1]-ledger[2]-ledger[3];
  const Real tolerance = std::max(static_cast<Real>(1.0e-10),
                                  512*std::numeric_limits<Real>::epsilon());
  if (!std::isfinite(residual) || fabs(residual) > tolerance*ledger[0]) {
    Error("injected/deposited/escaped/in-flight energy ledger failed to close");
  }
  if (report_ && global_variable::my_rank == 0) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Real>::max_digits10)
        << "# fast_electrons injected_erg=" << ledger[0]
        << " deposited_erg=" << ledger[1] << " escaped_erg=" << ledger[2]
        << " active_erg=" << ledger[3] << " residual_erg=" << residual
        << " packets=" << ledger[4];
    std::cout << out.str() << std::endl;
  }
}

}  // namespace fast_electrons
