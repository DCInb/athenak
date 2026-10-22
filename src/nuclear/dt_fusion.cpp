// AthenaK, licensed under the 3-clause BSD License (see LICENSE).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include "nuclear/dt_fusion.hpp"
#include "nuclear/dt_cell.hpp"
#include "fast_electrons/fast_electron_physics.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "two_temperature/two_temperature.hpp"
#include "units/units.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace nuclear {
namespace {
struct BirthTotals {
  double number, bulk;
  KOKKOS_INLINE_FUNCTION BirthTotals() : number(0.0), bulk(0.0) {}
  KOKKOS_INLINE_FUNCTION void operator+=(const BirthTotals &other) {
    number += other.number; bulk += other.bulk;
  }
};

[[noreturn]] void Error(const std::string &message) {
  std::cerr << "### FATAL ERROR <dt_fusion>: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void Positive(double value, const char *name) {
  if (!std::isfinite(value) || !(value > 0.0)) Error(std::string(name)+" must be positive");
}
}  // namespace

DTFusion::DTFusion(MeshBlockPack *pack, ParameterInput *pin) : pack_(pack) {
  if (pack->pmesh->nmb_packs_thisrank != 1) Error("requires one MeshBlockPack per rank");
  mhd_ = pack->pmhd != nullptr;
  if ((pack->phydro == nullptr) == (pack->pmhd == nullptr) || pack->punit == nullptr) {
    Error("requires one Newtonian two-temperature fluid, <materials>, and <units>");
  }
  auto tt = mhd_ ? pack->pmhd->ptwo_temp : pack->phydro->ptwo_temp;
  auto materials = mhd_ ? pack->pmhd->pmaterials : pack->phydro->pmaterials;
  if (tt == nullptr || materials == nullptr) Error("requires two_temperature and materials");
  if (pin->DoesBlockExist("shearing_box")) Error("shearing boundaries are unsupported");
  mixture_ = materials->DeviceData();
  if (mixture_.nmaterials > kMaxBurnMaterials) Error("at most 16 materials are supported");
  iion_ = tt->iion; iele_ = tt->iele;
  const std::string b = "dt_fusion";
  deuterium_ = pin->GetOrAddInteger(b, "deuterium_material", 0);
  tritium_ = pin->GetOrAddInteger(b, "tritium_material", 2);
  helium_ = pin->GetOrAddInteger(b, "helium_material", 1);
  auto species = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mixture_.species);
  const int ids[3] = {deuterium_, tritium_, helium_};
  const double masses[3] = {2.0, 3.0, 4.0}, charges[3] = {1.0, 1.0, 2.0};
  for (int n = 0; n < 3; ++n) {
    if (ids[n] < 0 || ids[n] >= mixture_.nmaterials) Error("invalid material index");
    if (std::abs(species(ids[n]).abar-masses[n]) > 1.0e-6 ||
        std::abs(species(ids[n]).zbar-charges[n]) > 1.0e-6) {
      Error("D/T/He must use baryon masses 2/3/4 and ion charges 1/1/2");
    }
    for (int j = 0; j < n; ++j) if (ids[n] == ids[j]) Error("material indices must differ");
  }
  const int samples = pin->GetOrAddInteger(b, "packets_per_cell", 8);
  if (samples < 2 || samples%2 != 0) Error("packets_per_cell must be positive and even");
  pairs_ = samples/2;
  max_packets_ = pin->GetOrAddInteger(b, "max_packets", 2000000);
  max_segments_ = pin->GetOrAddInteger(b, "max_segments", 256);
  max_waves_ = pin->GetOrAddInteger(b, "max_transport_waves", 10000);
  max_substeps_ = pin->GetOrAddInteger(b, "max_substeps", 10000);
  if (max_packets_ < samples || max_segments_ < 1 || max_waves_ < 1 || max_substeps_ < 1) {
    Error("invalid packet/work limit");
  }
  collisions_ = pin->GetOrAddBoolean(b, "collisions", true);
  scattering_ = pin->GetOrAddBoolean(b, "scattering", true);
  periodic_ = pin->GetOrAddBoolean(b, "periodic_transport", false);
  report_ = pin->GetOrAddBoolean(b, "report_diagnostics", true);
  log_lambda_ = pin->GetOrAddReal(b, "coulomb_log", 7.0);
  cutoff_ = pin->GetOrAddReal(b, "thermalization_kev", 1.0)*kKev;
  thermal_factor_ = pin->GetOrAddReal(b, "thermalization_factor", 3.0);
  loss_fraction_ = pin->GetOrAddReal(b, "max_fractional_loss", 0.03);
  scatter_limit_ = pin->GetOrAddReal(b, "max_scatter", 0.03);
  source_fraction_ = pin->GetOrAddReal(b, "source_fraction", 0.03);
  rate_multiplier_ = pin->GetOrAddReal(b, "rate_multiplier", 1.0);
  burn_rtol_ = pin->GetOrAddReal(b, "burn_rtol", 1.0e-6);
  max_source_dt_ = pin->GetOrAddReal(b, "max_source_dt", std::numeric_limits<Real>::max());
  const int seed = pin->GetOrAddInteger(b, "seed", 1);
  if (seed < 0) Error("seed must be nonnegative");
  seed_ = seed;
  Positive(log_lambda_, "coulomb_log"); Positive(cutoff_, "thermalization_kev");
  Positive(thermal_factor_, "thermalization_factor");
  Positive(loss_fraction_, "max_fractional_loss");
  Positive(scatter_limit_, "max_scatter"); Positive(source_fraction_, "source_fraction");
  Positive(burn_rtol_, "burn_rtol"); Positive(max_source_dt_, "max_source_dt");
  if (loss_fraction_ > 0.1 || scatter_limit_ > 0.1 || source_fraction_ > 0.1 ||
      thermal_factor_ < 3.0 || burn_rtol_ >= 0.1 ||
      !std::isfinite(rate_multiplier_) || rate_multiplier_ < 0.0) Error("invalid solver tolerance");
  length_ = pack->punit->length_cgs(); time_ = pack->punit->time_cgs();
  energy_ = pack->punit->energy_cgs(); density_ = pack->punit->density_cgs();
  temperature_ = pack->punit->temperature_cgs(); mu_ = pack->punit->mu();
  Positive(length_, "length unit"); Positive(time_, "time unit");
  Positive(energy_, "energy unit"); Positive(density_, "density unit");
  const auto mesh = pack->pmesh;
  const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
  if (periodic_) {
    for (int d = 0; d < 2*ndim; ++d) {
      if (mesh->mesh_bcs[d] != BoundaryFlag::periodic) {
        Error("periodic_transport requires periodic fluid boundaries in every active direction");
      }
    }
  }
  RefreshMesh();
}

void DTFusion::SyncThermodynamics(Driver *driver) {
  // Nuclear kernels read active cells on the packet's owning rank, never ghosts.
  // Retain the usual EOS/dual-energy projection between source operators without
  // exchanging unchanged magnetic fields or fluid halos at every substep.
  if (mhd_) (void)pack_->pmhd->ConToPrim(driver, 0);
  else (void)pack_->phydro->ConToPrim(driver, 0);
}

void DTFusion::Advance(Real dt, Driver *driver) {
  if (!(dt > 0.0)) return;
  Kokkos::deep_copy(cell_data, 0.0);
  Real elapsed = 0.0;
  for (int step = 0; elapsed < dt; ++step) {
    if (step >= max_substeps_) Error("max_substeps exhausted; reduce the hydro timestep");
    DepositMoments();
    const Real h = std::min({dt-elapsed, max_source_dt_, SourceTimeStep()});
    if (!(h > 0.0) || elapsed+h == elapsed) Error("source timestep underflow");
    // Strang T(h/2) B(h) T(h/2): old alphas travel h, midpoint births travel h/2.
    Transport(0.5*h); ApplyFeedback(); SyncThermodynamics(driver);
    Burn(h); SyncThermodynamics(driver);
    Transport(0.5*h); ApplyFeedback();
    elapsed += h;
    if (elapsed < dt) SyncThermodynamics(driver);
  }
  // Refresh coarse states, boundary conditions, ghosts, and final thermodynamics
  // once before returning to hydro/AMR or any other stencil-based operator.
  driver->InitBoundaryValuesAndPrimitives(pack_->pmesh);
  DepositMoments();
  (void)SourceTimeStep();  // refresh the instantaneous rate output after feedback
  Diagnose();
}

void DTFusion::Burn(Real dt) {
  const auto ind = pack_->pmesh->mb_indcs;
  auto u = mhd_ ? pack_->pmhd->u0 : pack_->phydro->u0;
  auto births = births_;
  auto mixture = mixture_;
  const int d = deuterium_, t = tritium_, a = helium_, ii = iion_, ie = iele_;
  const double density_unit = density_, pressure_unit = energy_/(length_*length_*length_);
  const double duration = dt*time_, multiplier = rate_multiplier_, rtol = burn_rtol_;
  const double tolerance = fmax(1.0e-10, 256.0*std::numeric_limits<Real>::epsilon());
  auto neutron_kinetic = stopping_;  // reusable scratch, consumed before moments rebuild
  par_for("dt_burn", DevExeSpace(), 0, pack_->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    BurnCell cell;
    cell.mixture = mixture; cell.d = d; cell.t = t; cell.a = a;
    cell.rho = u(m, IDN, k, j, i); cell.ei = u(m, ii, k, j, i);
    cell.ee = u(m, ie, k, j, i); cell.density_unit = density_unit;
    cell.pressure_unit = pressure_unit; cell.multiplier = multiplier;
    if (!(cell.rho > 0.0) || !Kokkos::isfinite(cell.rho) ||
        !(cell.ei >= 0.0) || !(cell.ee >= 0.0)) Kokkos::abort("Invalid D-T fluid state");
    double sum = 0.0;
    for (int n = 0; n < mixture.nmaterials; ++n) {
      const double value = u(m, mixture.scalar_indices(n), k, j, i);
      if (!Kokkos::isfinite(value) || value < -tolerance*cell.rho) {
        Kokkos::abort("Invalid D-T material abundance");
      }
      cell.partial[n] = fmax(0.0, value); sum += cell.partial[n];
    }
    if (fabs(sum-cell.rho) > tolerance*cell.rho) {
      Kokkos::abort("D-T material partial densities must sum to fluid density");
    }
    double extent;
    if (!cell.Integrate(duration, rtol, extent)) {
      Kokkos::abort("D-T burn failed: Ti outside [0,100] keV or tolerance exhausted");
    }
    births(m,k,j,i) = extent;
    const double dm = extent*kAmu/density_unit;
    double kinetic_loss = 0.0;
    // Neutrons free-stream: remove their baryon mass and comoving bulk momentum/E.
    // Alpha ash mass is retained at birth (the local-ash approximation).
    for (int n = 0; n < 3; ++n) {
      const double momentum = u(m, IM1+n, k, j, i);
      const double velocity = momentum/cell.rho;
      kinetic_loss += 0.5*dm*velocity*velocity;
      u(m, IM1+n, k, j, i) = momentum-dm*velocity;
    }
    neutron_kinetic(m,k,j,i) = kinetic_loss;
    u(m, IDN, k, j, i) = cell.rho-dm;
    u(m, IEN, k, j, i) -= kinetic_loss;
    u(m, mixture.scalar_indices(d), k, j, i) = fmax(0.0, cell.partial[d]-2.0*dm);
    u(m, mixture.scalar_indices(t), k, j, i) = fmax(0.0, cell.partial[t]-3.0*dm);
    u(m, mixture.scalar_indices(a), k, j, i) = cell.partial[a]+4.0*dm;
  });
  // Allocate source slots on the device. The prefix counts cells, so multiplying
  // by the sampling count cannot overflow before the host checks max_packets.
  const int ncells = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  auto offsets = birth_offsets_;
  int active_cells = 0;
  Kokkos::parallel_scan("dt_birth_offsets", Kokkos::RangePolicy<DevExeSpace>(0,ncells),
  KOKKOS_LAMBDA(int idx, int &offset, bool final) {
    int cell = idx;
    const int i = cell%ind.nx1+ind.is; cell /= ind.nx1;
    const int j = cell%ind.nx2+ind.js; cell /= ind.nx2;
    const int k = cell%ind.nx3+ind.ks, m = cell/ind.nx3;
    if (final) offsets(idx) = offset;
    if (births(m,k,j,i) > 0.0) ++offset;
  }, active_cells);
  const std::uint64_t epoch = ++birth_epoch_, seed = seed_;
  const std::int64_t added = static_cast<std::int64_t>(active_cells)*2*pairs_;
  if (added > max_packets_-packet_count_) {
    Error("max_packets exceeded; increase capacity or reduce source sampling");
  }
  if (added == 0) return;
  const int base = packet_count_, pairs = pairs_, first = pack_->gids;
  ReservePackets(packet_count_+static_cast<int>(added));
  auto packets = device_packets_; auto blocks = blocks_;
  const double volume_unit = length_*length_*length_, energy_unit = energy_;
  BirthTotals totals;
  Kokkos::parallel_reduce("dt_emit_alphas", Kokkos::RangePolicy<DevExeSpace>(0,ncells),
  KOKKOS_LAMBDA(int idx, BirthTotals &sum) {
    int cell_index = idx;
    const int i = cell_index%ind.nx1+ind.is; cell_index /= ind.nx1;
    const int j = cell_index%ind.nx2+ind.js; cell_index /= ind.nx2;
    const int k = cell_index%ind.nx3+ind.ks, m = cell_index/ind.nx3;
    if (!(births(m,k,j,i) > 0.0)) return;
    const int gid = first+m;
    const auto block = blocks(gid);
    const double volume = block.dx1*block.dx2*block.dx3;
    const double number = births(m,k,j,i)*volume*volume_unit;
    sum.number += number;
    sum.bulk += neutron_kinetic(m,k,j,i)*volume*energy_unit;
    const std::uint64_t cell = (k*ind.nx2+j)*ind.nx1+i;
    std::uint64_t rng = seed ^ (epoch*UINT64_C(0x9e3779b97f4a7c15)) ^
        (static_cast<std::uint64_t>(gid)*UINT64_C(0xbf58476d1ce4e5b9)) ^ cell;
    const int start = base+2*pairs*offsets(idx);
    for (int q = 0; q < pairs; ++q) {
      const double cosine = 2.0*fast_electrons::Uniform01(rng)-1.0;
      const double phi = 2.0*kPi*fast_electrons::Uniform01(rng);
      const double sine = sqrt(fmax(0.0, 1.0-cosine*cosine));
      AlphaPacket p{};
      p.n[0] = cosine; p.n[1] = sine*cos(phi); p.n[2] = sine*sin(phi);
      p.x[0] = block.x1min+(i-ind.is+0.5)*block.dx1;
      p.x[1] = block.x2min+(j-ind.js+0.5)*block.dx2;
      p.x[2] = block.x3min+(k-ind.ks+0.5)*block.dx3;
      p.cell[0] = i-ind.is; p.cell[1] = j-ind.js; p.cell[2] = k-ind.ks;
      p.gid = gid; p.energy = kAlphaEnergy; p.weight = number/(2*pairs);
      p.rng = rng; packets(start+2*q) = p;
      for (int n = 0; n < 3; ++n) p.n[n] = -p.n[n];
      p.rng ^= UINT64_C(0x94d049bb133111eb); packets(start+2*q+1) = p;
    }
  }, totals);
  packet_count_ += static_cast<int>(added);
  ledger_[0] += totals.number; ledger_[1] += totals.number*kAlphaEnergy;
  ledger_[4] += totals.number*kNeutronEnergy; ledger_[5] += totals.number*kAmu;
  ledger_[6] += totals.bulk;
}

Real DTFusion::SourceTimeStep() {
  const auto ind = pack_->pmesh->mb_indcs;
  auto u = mhd_ ? pack_->pmhd->u0 : pack_->phydro->u0;
  auto stopping = stopping_;
  auto data = cell_data;
  auto mix = mixture_;
  const int ii = iion_, ie = iele_, d = deuterium_, t = tritium_;
  const double density_unit = density_, pressure_unit = energy_/(length_*length_*length_);
  const double time_unit = time_, fraction = source_fraction_, multiplier = rate_multiplier_;
  const double maximum = std::numeric_limits<Real>::max();
  Real result;
  Kokkos::parallel_reduce("dt_source_limit", Kokkos::RangePolicy<DevExeSpace>(0,
      pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3),
  KOKKOS_LAMBDA(int idx, Real &minimum) {
    const int i = idx%ind.nx1+ind.is; idx /= ind.nx1;
    const int j = idx%ind.nx2+ind.js; idx /= ind.nx2;
    const int k = idx%ind.nx3+ind.ks, m = idx/ind.nx3;
    const auto composition = mix.CompositionFromConserved(u, m, k, j, i);
    const double rho = u(m, IDN, k, j, i);
    const double nd = fmax(0.0, u(m, mix.scalar_indices(d), k, j, i))*density_unit/(2*kAmu);
    const double nt = fmax(0.0, u(m, mix.scalar_indices(t), k, j, i))*density_unit/(3*kAmu);
    double rate = 0.0;
    if (nd > 0.0 && nt > 0.0 && multiplier > 0.0) {
      double ti, te;
      Temperatures(mix, composition, rho, u(m,ii,k,j,i), u(m,ie,k,j,i),
                   pressure_unit, density_unit, ti, te);
      rate = DTReactivity(ti*kBoltzmann/kKev);
      if (rate < 0.0) Kokkos::abort("D-T reactivity requires Ti <= 100 keV");
      rate *= nd*nt*multiplier;
    }
    data(m,0,k,j,i) = rate;
    double h = maximum;
    if (rate > 0.0) h = fraction*fmin(nd,nt)/rate/time_unit;
    const double power = rate*kAlphaEnergy+stopping(m,k,j,i);
    if (power > 0.0) {
      const double energy = fmin(u(m,ii,k,j,i), u(m,ie,k,j,i))*pressure_unit;
      h = fmin(h, fraction*energy/power/time_unit);
    }
    minimum = fmin(minimum, h);
  }, Kokkos::Min<Real>(result));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return result;
}

void DTFusion::ApplyFeedback() {
  const auto ind = pack_->pmesh->mb_indcs;
  auto u = mhd_ ? pack_->pmhd->u0 : pack_->phydro->u0;
  auto feedback = feedback_; auto data = cell_data;
  auto size = pack_->pmb->mb_size.d_view;
  const int ii = iion_, ie = iele_;
  double deposited;
  Kokkos::parallel_reduce("dt_alpha_feedback", Kokkos::RangePolicy<DevExeSpace>(0,
      pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3),
  KOKKOS_LAMBDA(int idx, double &sum) {
    const int i = idx%ind.nx1+ind.is; idx /= ind.nx1;
    const int j = idx%ind.nx2+ind.js; idx /= ind.nx2;
    const int k = idx%ind.nx3+ind.ks, m = idx/ind.nx3;
    const double de_i = feedback(m,0,k,j,i), de_e = feedback(m,1,k,j,i);
    double work = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double dp = feedback(m,2+d,k,j,i), old = u(m,IM1+d,k,j,i);
      work += dp*(old+0.5*dp)/u(m,IDN,k,j,i);
      u(m,IM1+d,k,j,i) = old+dp;
      data(m,5+d,k,j,i) += dp;
    }
    // Total energy includes momentum work; do not count that work again as heat.
    u(m,IEN,k,j,i) += de_i+de_e;
    u(m,ii,k,j,i) += de_i-work; u(m,ie,k,j,i) += de_e;
    if (!Kokkos::isfinite(u(m,IEN,k,j,i)) || u(m,ii,k,j,i) < 0.0 ||
        u(m,ie,k,j,i) < 0.0) Kokkos::abort("Alpha feedback invalid; reduce source_fraction");
    data(m,3,k,j,i) += de_i-work; data(m,4,k,j,i) += de_e;
    sum += (de_i+de_e)*size(m).dx1*size(m).dx2*size(m).dx3;
  }, deposited);
  ledger_[2] += deposited*energy_;
}

void DTFusion::Diagnose() {
  double values[9];
  for (int n = 0; n < 7; ++n) values[n] = ledger_[n];
  values[8] = packet_count_;
  auto packets = device_packets_;
  Kokkos::parallel_reduce("dt_live_energy", Kokkos::RangePolicy<DevExeSpace>(0,packet_count_),
  KOKKOS_LAMBDA(int q, double &sum) {
    sum += packets(q).weight*packets(q).energy;
  }, values[7]);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  const double residual = values[1]-values[2]-values[3]-values[7];
  const double tol = fmax(1.0e-9, 2048.0*std::numeric_limits<Real>::epsilon());
  if (!std::isfinite(residual) || fabs(residual) > tol*fmax(values[1],1.0e-100)) {
    Error("alpha born/deposited/escaped/in-flight energy does not close");
  }
  if (report_ && global_variable::my_rank == 0) {
    std::ostringstream out;
    out << std::setprecision(17) << "# dt_fusion reactions=" << values[0]
        << " alpha_born_erg=" << values[1] << " deposited_erg=" << values[2]
        << " escaped_erg=" << values[3] << " neutron_erg=" << values[4]
        << " neutron_mass_g=" << values[5] << " neutron_bulk_erg=" << values[6]
        << " active_erg=" << values[7] << " packets=" << values[8]
        << " residual_erg=" << residual;
    std::cout << out.str() << std::endl;
  }
}
}  // namespace nuclear
