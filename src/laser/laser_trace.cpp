//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_trace.cpp
//! \brief GPU-resident aperture initialization, DDA marching, and queue compaction.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "laser/laser.hpp"
#include "laser/laser_physics.hpp"
#include "mhd/mhd.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

#if SINGLE_PRECISION_ENABLED
constexpr Real kRealMaximum = FLT_MAX;
#else
constexpr Real kRealMaximum = DBL_MAX;
#endif

KOKKOS_INLINE_FUNCTION
bool ContainsPoint(const RegionSize &size, Real x, Real y, Real z,
                   bool multi_d, bool three_d) {
  bool inside = (x >= size.x1min && x < size.x1max);
  if (multi_d) inside = inside && (y >= size.x2min && y < size.x2max);
  if (three_d) inside = inside && (z >= size.x3min && z < size.x3max);
  return inside;
}

KOKKOS_INLINE_FUNCTION
int FindLocalBlock(const DvceArray1D<RegionSize> &sizes, int nmb,
                   Real x, Real y, Real z, bool multi_d, bool three_d) {
  for (int m = 0; m < nmb; ++m) {
    if (ContainsPoint(sizes(m), x, y, z, multi_d, three_d)) return m;
  }
  return -1;
}

KOKKOS_INLINE_FUNCTION
bool ContainsPoint(const laser::LaserBlockInfo &block, Real x, Real y, Real z,
                   bool multi_d, bool three_d) {
  bool inside = (x >= block.x1min && x < block.x1max);
  if (multi_d) inside = inside && (y >= block.x2min && y < block.x2max);
  if (three_d) inside = inside && (z >= block.x3min && z < block.x3max);
  return inside;
}

KOKKOS_INLINE_FUNCTION
int FindGlobalBlock(const DvceArray1D<laser::LaserBlockInfo> &blocks, int nmb,
                    Real x, Real y, Real z, bool multi_d, bool three_d) {
  for (int gid = 0; gid < nmb; ++gid) {
    if (ContainsPoint(blocks(gid), x, y, z, multi_d, three_d)) return gid;
  }
  return -1;
}

KOKKOS_INLINE_FUNCTION
int ActiveCellIndex(Real x, Real xmin, Real dx, int is, int ie) {
  int index = is + static_cast<int>(floor((x-xmin)/dx));
  return (index < is) ? is : ((index > ie) ? ie : index);
}

KOKKOS_INLINE_FUNCTION
Real ProbeForward(Real coordinate, Real direction, Real distance) {
  if (direction == 0.0) return coordinate;
  Real result = coordinate+distance*direction;
  if (result == coordinate) {
    Real limit = (direction > 0.0) ? kRealMaximum : -kRealMaximum;
    result = nextafter(coordinate, limit);
  }
  return result;
}

bool FirstDomainIntersection(const RegionSize &domain, bool multi_d, bool three_d,
                             const Real origin[3], const Real direction[3],
                             Real &distance) {
  Real lower[3] = {domain.x1min, domain.x2min, domain.x3min};
  Real upper[3] = {domain.x1max, domain.x2max, domain.x3max};
  int dimensions = three_d ? 3 : (multi_d ? 2 : 1);
  Real enter = 0.0;
  Real leave = std::numeric_limits<Real>::max();
  for (int n = 0; n < dimensions; ++n) {
    if (direction[n] == 0.0) {
      if (origin[n] < lower[n] || origin[n] > upper[n]) return false;
      continue;
    }
    Real first = (lower[n]-origin[n])/direction[n];
    Real second = (upper[n]-origin[n])/direction[n];
    enter = std::max(enter, std::min(first, second));
    leave = std::min(leave, std::max(first, second));
  }
  if (leave < enter || leave < 0.0) return false;
  distance = enter;
  return true;
}

} // namespace

namespace laser {

//----------------------------------------------------------------------------------------
//! Build a deterministic equal-area Fibonacci aperture. Direction beams remain parallel;
//! lens beams connect each aperture sample to the corresponding finite target spot.

void Laser::BuildInitialRays() {
  auto hx = Kokkos::create_mirror_view(ray_x0_);
  auto hy = Kokkos::create_mirror_view(ray_y0_);
  auto hz = Kokkos::create_mirror_view(ray_z0_);
  auto hnx = Kokkos::create_mirror_view(ray_nx0_);
  auto hny = Kokkos::create_mirror_view(ray_ny0_);
  auto hnz = Kokkos::create_mirror_view(ray_nz0_);
  auto hp = Kokkos::create_mirror_view(ray_power0_);
  auto hfraction = Kokkos::create_mirror_view(ray_power_fraction_);
  auto hlambda = Kokkos::create_mirror_view(ray_wavelength_);
  auto hzbar = Kokkos::create_mirror_view(ray_zeff_);
  auto hk = Kokkos::create_mirror_view(ray_constant_absorption_);
  auto ht0 = Kokkos::create_mirror_view(ray_start_time_);
  auto ht1 = Kokkos::create_mirror_view(ray_end_time_);
  auto hb = Kokkos::create_mirror_view(ray_beam_);

  const Real golden_angle = pi*(3.0-std::sqrt(5.0));
  int ray = 0;
  for (std::size_t b = 0; b < beams_.size(); ++b) {
    const BeamConfig &beam = beams_[b];

    Real reference[3] = {0.0, 0.0, 1.0};
    if (std::abs(beam.direction[2]) > 0.9) {
      reference[1] = 1.0;
      reference[2] = 0.0;
    }
    Real basis_u[3] = {
      beam.direction[1]*reference[2] - beam.direction[2]*reference[1],
      beam.direction[2]*reference[0] - beam.direction[0]*reference[2],
      beam.direction[0]*reference[1] - beam.direction[1]*reference[0]
    };
    Real unorm = std::sqrt(SQR(basis_u[0])+SQR(basis_u[1])+SQR(basis_u[2]));
    for (int d = 0; d < 3; ++d) basis_u[d] /= unorm;
    Real basis_v[3] = {
      beam.direction[1]*basis_u[2] - beam.direction[2]*basis_u[1],
      beam.direction[2]*basis_u[0] - beam.direction[0]*basis_u[2],
      beam.direction[0]*basis_u[1] - beam.direction[1]*basis_u[0]
    };

    std::vector<Real> offset_u(beam.nrays, 0.0);
    std::vector<Real> offset_v(beam.nrays, 0.0);
    std::vector<Real> sample_radius(beam.nrays, 0.0);
    std::vector<long double> log_weight(beam.nrays, 0.0L);
    long double maximum_log_weight = -std::numeric_limits<long double>::infinity();
    for (int n = 0; n < beam.nrays; ++n) {
      if (beam.radius > 0.0 && pmy_pack_->pmesh->three_d) {
        sample_radius[n] =
            beam.radius*std::sqrt((n+0.5)/static_cast<Real>(beam.nrays));
        Real angle = golden_angle*n;
        offset_u[n] = sample_radius[n]*std::cos(angle);
        offset_v[n] = sample_radius[n]*std::sin(angle);
      } else if (beam.radius > 0.0 && pmy_pack_->pmesh->multi_d) {
        offset_u[n] = beam.radius*
            (2.0*(n+0.5)/static_cast<Real>(beam.nrays)-1.0);
        sample_radius[n] = std::abs(offset_u[n]);
      }
      if (beam.profile == "gaussian" && beam.radius > 0.0) {
        long double scaled_radius =
            static_cast<long double>(sample_radius[n])/
            static_cast<long double>(beam.profile_radius);
        log_weight[n] = -2.0L*scaled_radius*scaled_radius;
        maximum_log_weight = std::max(maximum_log_weight, log_weight[n]);
      }
    }
    std::vector<Real> weight(beam.nrays, 1.0);
    long double weight_sum = static_cast<long double>(beam.nrays);
    if (beam.profile == "gaussian" && beam.radius > 0.0) {
      weight_sum = 0.0L;
      for (int n = 0; n < beam.nrays; ++n) {
        weight[n] = static_cast<Real>(
            std::exp(log_weight[n]-maximum_log_weight));
        weight_sum += static_cast<long double>(weight[n]);
      }
    }
    if (!std::isfinite(weight_sum) || !(weight_sum > 0.0L)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<laser> could not normalize beam " << b
                << " ray weights" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    for (int n = 0; n < beam.nrays; ++n, ++ray) {
      Real du = offset_u[n];
      Real dv = offset_v[n];
      hx(ray) = beam.origin[0] + du*basis_u[0] + dv*basis_v[0];
      hy(ray) = beam.origin[1] + du*basis_u[1] + dv*basis_v[1];
      hz(ray) = beam.origin[2] + du*basis_u[2] + dv*basis_v[2];
      if (beam.geometry == BeamGeometry::lens) {
        Real spot_scale = (beam.radius > 0.0) ? beam.target_radius/beam.radius : 0.0;
        Real tx = beam.target[0] + spot_scale*(du*basis_u[0] + dv*basis_v[0]);
        Real ty = beam.target[1] + spot_scale*(du*basis_u[1] + dv*basis_v[1]);
        Real tz = beam.target[2] + spot_scale*(du*basis_u[2] + dv*basis_v[2]);
        Real dx = tx-hx(ray);
        Real dy = ty-hy(ray);
        Real dz = tz-hz(ray);
        Real dnorm = std::sqrt(SQR(dx)+SQR(dy)+SQR(dz));
        hnx(ray) = dx/dnorm; hny(ray) = dy/dnorm; hnz(ray) = dz/dnorm;
      } else {
        hnx(ray) = beam.direction[0]; hny(ray) = beam.direction[1];
        hnz(ray) = beam.direction[2];
      }
      if (beam.geometry == BeamGeometry::lens) {
        Real origin[3] = {hx(ray), hy(ray), hz(ray)};
        Real direction[3] = {hnx(ray), hny(ray), hnz(ray)};
        Real distance = 0.0;
        if (FirstDomainIntersection(pmy_pack_->pmesh->mesh_size,
                                    pmy_pack_->pmesh->multi_d,
                                    pmy_pack_->pmesh->three_d,
                                    origin, direction, distance)) {
          hx(ray) += distance*hnx(ray);
          hy(ray) += distance*hny(ray);
          hz(ray) += distance*hnz(ray);
        }
      }
      hfraction(ray) = static_cast<Real>(
          static_cast<long double>(weight[n])/weight_sum);
      hp(ray) = beam.power*hfraction(ray);
      hlambda(ray) = beam.wavelength;
      hzbar(ray) = beam.zeff;
      hk(ray) = beam.constant_absorption;
      ht0(ray) = beam.start_time;
      ht1(ray) = beam.end_time;
      hb(ray) = static_cast<int>(b);
    }
  }
  Kokkos::deep_copy(ray_x0_, hx); Kokkos::deep_copy(ray_y0_, hy);
  Kokkos::deep_copy(ray_z0_, hz); Kokkos::deep_copy(ray_nx0_, hnx);
  Kokkos::deep_copy(ray_ny0_, hny); Kokkos::deep_copy(ray_nz0_, hnz);
  Kokkos::deep_copy(ray_power0_, hp);
  Kokkos::deep_copy(ray_power_fraction_, hfraction);
  Kokkos::deep_copy(ray_wavelength_, hlambda);
  Kokkos::deep_copy(ray_zeff_, hzbar); Kokkos::deep_copy(ray_constant_absorption_, hk);
  Kokkos::deep_copy(ray_start_time_, ht0); Kokkos::deep_copy(ray_end_time_, ht1);
  Kokkos::deep_copy(ray_beam_, hb);
}

//----------------------------------------------------------------------------------------
//! Restore immutable launch data and map every live ray into a local MeshBlock/cell.

void Laser::InitializeRays(Real time) {
  auto host_beam_power = Kokkos::create_mirror_view(beam_power_);
  Real dt = pmy_pack_->pmesh->dt;
  for (std::size_t b = 0; b < beams_.size(); ++b) {
    host_beam_power(static_cast<int>(b)) = BeamPowerForStep(beams_[b], time, dt);
  }
  Kokkos::deep_copy(beam_power_, host_beam_power);

  auto sizes = pmy_pack_->pmb->mb_size.d_view;
  auto gids = pmy_pack_->pmb->mb_gid.d_view;
  auto global_blocks = global_block_info_;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int nmb = pmy_pack_->nmb_thispack;
  int nmb_total = pmy_pack_->pmesh->nmb_total;
  int my_rank = global_variable::my_rank;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  RegionSize domain = pmy_pack_->pmesh->mesh_size;
  Real domain_scale = fmax(domain.x1max-domain.x1min, 1.0);
  if (multi_d) domain_scale = fmax(domain_scale, domain.x2max-domain.x2min);
  if (three_d) domain_scale = fmax(domain_scale, domain.x3max-domain.x3min);
  Real probe_eps = 128.0*std::numeric_limits<Real>::epsilon()*domain_scale;

  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto wave_x = ray_kx_; auto wave_y = ray_ky_; auto wave_z = ray_kz_;
  auto dispersion_error = ray_dispersion_error_;
  auto power = ray_power;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status;
  auto x0 = ray_x0_; auto y0 = ray_y0_; auto z0 = ray_z0_;
  auto nx0 = ray_nx0_; auto ny0 = ray_ny0_; auto nz0 = ray_nz0_;
  auto power0 = ray_power0_;
  auto power_fraction = ray_power_fraction_;
  auto beam_index = ray_beam_;
  auto beam_power = beam_power_;
  auto segments = ray_segments_; auto reflections = ray_reflections_;
  auto path = ray_path_length_;
  auto destination_rank = ray_destination_rank_;
  auto queue_a = active_queue_a_; auto queue_b = active_queue_b_;
  auto diag = device_diagnostics_;
  auto counters = device_counters_;
  auto primitive = pmy_pack_->pmhd->w0;
  auto wavelength = ray_wavelength_;
  bool refractive = propagation_model_ == PropagationModel::refractive;
  Real number_scale = density_scale_cgs_*electron_number_per_gram_;
  Real length_scale = length_scale_cgs_;

  Kokkos::parallel_for(
      "laser_initialize_rays", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        x(r) = x0(r); y(r) = y0(r); z(r) = z0(r);
        nx(r) = nx0(r); ny(r) = ny0(r); nz(r) = nz0(r);
        wave_x(r) = 0.0; wave_y(r) = 0.0; wave_z(r) = 0.0;
        dispersion_error(r) = 0.0;
        power0(r) = power_fraction(r)*beam_power(beam_index(r));
        power(r) = power0(r);
        segments(r) = 0;
        reflections(r) = 0;
        path(r) = 0.0;
        destination_rank(r) = -1;
        gid(r) = -1; ci(r) = is; cj(r) = js; ck(r) = ks;
        queue_a(r) = -1; queue_b(r) = -1;
        status(r) = static_cast<int>(RayStatus::inactive);
        if (!(power(r) > 0.0)) return;

        Real px = ProbeForward(x(r), nx(r), probe_eps);
        Real py = ProbeForward(y(r), ny(r), probe_eps);
        Real pz = ProbeForward(z(r), nz(r), probe_eps);
        int m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);
        if (m < 0) {
          int global_m = FindGlobalBlock(
              global_blocks, nmb_total, px, py, pz, multi_d, three_d);
          if (global_m < 0 && my_rank == 0) {
            status(r) = static_cast<int>(RayStatus::escaped);
            Kokkos::atomic_add(&diag(0), power(r));
            Kokkos::atomic_add(&diag(2), power(r));
          }
          return;
        }
        Kokkos::atomic_add(&diag(0), power(r));
        gid(r) = gids(m);
        ci(r) = ActiveCellIndex(px, sizes(m).x1min, sizes(m).dx1, is, ie);
        if (multi_d) {
          cj(r) = ActiveCellIndex(py, sizes(m).x2min, sizes(m).dx2, js, je);
        }
        if (three_d) {
          ck(r) = ActiveCellIndex(pz, sizes(m).x3min, sizes(m).dx3, ks, ke);
        }
        Real wave_magnitude = 1.0;
        if (refractive) {
          int ii = ci(r), jj = cj(r), kk = ck(r);
          Real center_x = sizes(m).x1min + (ii-is+0.5)*sizes(m).dx1;
          Real center_y = sizes(m).x2min + (jj-js+0.5)*sizes(m).dx2;
          Real center_z = sizes(m).x3min + (kk-ks+0.5)*sizes(m).dx3;
          Real offset_x = px-center_x;
          Real offset_y = py-center_y;
          Real offset_z = pz-center_z;
          Real density = primitive(m, IDN, kk, jj, ii);
          Real grad_x = (primitive(m, IDN, kk, jj, ii+1) -
                         primitive(m, IDN, kk, jj, ii-1))/(2.0*sizes(m).dx1);
          Real hess_x = (primitive(m, IDN, kk, jj, ii+1) - 2.0*density +
                         primitive(m, IDN, kk, jj, ii-1))/SQR(sizes(m).dx1);
          Real reconstructed_density =
              density + grad_x*offset_x + 0.5*hess_x*SQR(offset_x);
          if (multi_d) {
            Real grad_y = (primitive(m, IDN, kk, jj+1, ii) -
                           primitive(m, IDN, kk, jj-1, ii))/(2.0*sizes(m).dx2);
            Real hess_y = (primitive(m, IDN, kk, jj+1, ii) - 2.0*density +
                           primitive(m, IDN, kk, jj-1, ii))/SQR(sizes(m).dx2);
            reconstructed_density +=
                grad_y*offset_y + 0.5*hess_y*SQR(offset_y);
          }
          if (three_d) {
            Real grad_z = (primitive(m, IDN, kk+1, jj, ii) -
                           primitive(m, IDN, kk-1, jj, ii))/(2.0*sizes(m).dx3);
            Real hess_z = (primitive(m, IDN, kk+1, jj, ii) - 2.0*density +
                           primitive(m, IDN, kk-1, jj, ii))/SQR(sizes(m).dx3);
            reconstructed_density +=
                grad_z*offset_z + 0.5*hess_z*SQR(offset_z);
          }
          Real critical_density = CriticalDensity(wavelength(r)*length_scale);
          Real electron_density = number_scale*fmax(reconstructed_density, 0.0);
          Real normalized_density = electron_density/critical_density;
          if (!(normalized_density >= 0.0 && normalized_density < 1.0)) {
            status(r) = static_cast<int>(RayStatus::failed);
            Kokkos::atomic_inc(&counters(3));
            return;
          }
          wave_magnitude = sqrt(1.0-normalized_density);
        }
        wave_x(r) = wave_magnitude*nx(r);
        wave_y(r) = wave_magnitude*ny(r);
        wave_z(r) = wave_magnitude*nz(r);
        status(r) = static_cast<int>(RayStatus::active);
        queue_a(r) = r;
      });
}

//----------------------------------------------------------------------------------------
//! Compact live ray IDs into the next queue using a device prefix scan.

void Laser::CompactActiveQueue(DvceArray1D<int> current,
                               DvceArray1D<int> next) {
  Kokkos::deep_copy(next, -1);
  auto status = ray_status;
  Kokkos::parallel_scan(
      "laser_compact_queue", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int index, int &offset, bool final) {
        int ray = current(index);
        int keep = (ray >= 0 &&
                    status(ray) == static_cast<int>(RayStatus::active)) ? 1 : 0;
        if (final && keep) next(offset) = ray;
        offset += keep;
      });
}

//----------------------------------------------------------------------------------------
//! Rebuild queue A from ray status. Called at every trace entry so rays that survived
//! a previous work-capped wave and rays activated by MPI unpack both re-enter the
//! queue without relying on incremental queue state.

void Laser::SeedActiveQueue() {
  Kokkos::deep_copy(active_queue_a_, -1);
  auto status = ray_status;
  auto queue = active_queue_a_;
  Kokkos::parallel_scan(
      "laser_seed_queue", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r, int &offset, bool final) {
        int keep = (status(r) == static_cast<int>(RayStatus::active)) ? 1 : 0;
        if (final && keep) queue(offset) = r;
        offset += keep;
      });
}

//----------------------------------------------------------------------------------------
//! Count rays still marked active (host-synchronous device reduction).

int Laser::CountActiveRays() {
  auto status = ray_status;
  int count = 0;
  Kokkos::parallel_reduce(
      "laser_count_active", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r, int &sum) {
        if (status(r) == static_cast<int>(RayStatus::active)) sum += 1;
      }, Kokkos::Sum<int>(count));
  return count;
}

//----------------------------------------------------------------------------------------
//! Book rays that exhausted the global wave budget as remaining power.

void Laser::BookRemainingRays() {
  auto status = ray_status;
  auto power = ray_power;
  auto diag = device_diagnostics_;
  auto counters = device_counters_;
  Kokkos::parallel_for(
      "laser_book_remaining", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        if (status(r) == static_cast<int>(RayStatus::active) ||
            status(r) == static_cast<int>(RayStatus::off_rank)) {
          status(r) = static_cast<int>(RayStatus::remaining);
          Kokkos::atomic_add(&diag(3), power(r));
          Kokkos::atomic_inc(&counters(0));
        }
      });
}

//----------------------------------------------------------------------------------------
//! Straight Cartesian DDA. One device thread advances one ray through several cells.

void Laser::TraceStraightRays(bool preserve_off_rank) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  int nmb_total = pmy_pack_->pmesh->nmb_total;
  int first_gid = pmy_pack_->gids;
  int my_rank = global_variable::my_rank;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto sizes = pmy_pack_->pmb->mb_size.d_view;
  auto gids = pmy_pack_->pmb->mb_gid.d_view;
  auto global_blocks = global_block_info_;
  RegionSize domain = pmy_pack_->pmesh->mesh_size;
  Real machine_eps = std::numeric_limits<Real>::epsilon();
  Real infinite = std::numeric_limits<Real>::max();

  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto power = ray_power;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status; auto segments = ray_segments_;
  auto destination_rank = ray_destination_rank_;
  auto reflections = ray_reflections_;
  auto path = ray_path_length_; auto data = cell_data;
  auto diag = device_diagnostics_; auto counters = device_counters_;
  auto primitive = pmy_pack_->pmhd->w0;
  auto constant_absorption = ray_constant_absorption_;
  auto wavelength = ray_wavelength_;
  auto zeff = ray_zeff_;
  auto initial_power = ray_power0_;
  bool periodic = periodic_transport_;
  int absorption_mode = static_cast<int>(absorption_model_);
  int inverse_bremsstrahlung =
      static_cast<int>(AbsorptionModel::inverse_bremsstrahlung);
  int electron_index = electron_index_;
  Real gamma_minus_one = gamma_minus_one_;
  Real electron_heat_capacity_fraction = electron_heat_capacity_fraction_;
  Real electron_number_per_gram = electron_number_per_gram_;
  Real density_scale_cgs = density_scale_cgs_;
  Real temperature_scale_cgs = temperature_scale_cgs_;
  Real length_scale_cgs = length_scale_cgs_;
  Real fixed_coulomb_log = inverse_bremsstrahlung_coulomb_log_;
  Real ib_temperature_floor = inverse_bremsstrahlung_temperature_floor_;
  Real minimum_power_fraction = minimum_power_fraction_;
  bool reflect_at_critical = critical_reflection_;
  bool oblique_turning = oblique_turning_;
  int max_reflections = max_reflections_per_ray_;
  Real reflection_offset_fraction = reflection_offset_fraction_;
  int seg_per_launch = max_segments_per_launch_;

  SeedActiveQueue();
  for (int iteration = 0; iteration < max_transport_iterations_; ++iteration) {
    DvceArray1D<int> current = (iteration % 2 == 0) ? active_queue_a_ : active_queue_b_;
    DvceArray1D<int> next = (iteration % 2 == 0) ? active_queue_b_ : active_queue_a_;
    Kokkos::parallel_for(
        "laser_trace_dda", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
        KOKKOS_LAMBDA(int queue_index) {
          int r = current(queue_index);
          if (r < 0 || status(r) != static_cast<int>(RayStatus::active)) return;

          for (int segment = 0; segment < seg_per_launch; ++segment) {
            if (status(r) != static_cast<int>(RayStatus::active)) break;
            int m = gid(r)-first_gid;
            if (m < 0 || m >= nmb) {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
              break;
            }
            RegionSize size = sizes(m);
            int i = ci(r), j = cj(r), k = ck(r);

            Real sx = infinite, sy = infinite, sz = infinite;
            if (nx(r) > 0.0) {
              Real face = size.x1min + (i-is+1)*size.dx1;
              sx = (face-x(r))/nx(r);
            } else if (nx(r) < 0.0) {
              Real face = size.x1min + (i-is)*size.dx1;
              sx = (face-x(r))/nx(r);
            }
            if (multi_d) {
              if (ny(r) > 0.0) {
                Real face = size.x2min + (j-js+1)*size.dx2;
                sy = (face-y(r))/ny(r);
              } else if (ny(r) < 0.0) {
                Real face = size.x2min + (j-js)*size.dx2;
                sy = (face-y(r))/ny(r);
              }
            }
            if (three_d) {
              if (nz(r) > 0.0) {
                Real face = size.x3min + (k-ks+1)*size.dx3;
                sz = (face-z(r))/nz(r);
              } else if (nz(r) < 0.0) {
                Real face = size.x3min + (k-ks)*size.dx3;
                sz = (face-z(r))/nz(r);
              }
            }
            Real ds = fmin(sx, fmin(sy, sz));
            Real scale = fmin(size.dx1, multi_d ? size.dx2 : size.dx1);
            if (three_d) scale = fmin(scale, size.dx3);
            Real tolerance = 128.0*machine_eps*fmax(scale, 1.0);
            if (!(ds >= -tolerance) || ds == infinite || !Kokkos::isfinite(ds)) {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
              break;
            }
            ds = fmax(ds, 0.0);
            bool turning_point = false;
            Real normal_x = 0.0, normal_y = 0.0, normal_z = 0.0;
            if (reflect_at_critical) {
              Real number_scale = density_scale_cgs*electron_number_per_gram;
              Real grad_x = number_scale*
                  (primitive(m, IDN, k, j, i+1) -
                   primitive(m, IDN, k, j, i-1))/(2.0*size.dx1);
              Real grad_y = 0.0;
              Real grad_z = 0.0;
              if (multi_d) {
                grad_y = number_scale*
                    (primitive(m, IDN, k, j+1, i) -
                     primitive(m, IDN, k, j-1, i))/(2.0*size.dx2);
              }
              if (three_d) {
                grad_z = number_scale*
                    (primitive(m, IDN, k+1, j, i) -
                     primitive(m, IDN, k-1, j, i))/(2.0*size.dx3);
              }
              Real grad_norm = sqrt(SQR(grad_x)+SQR(grad_y)+SQR(grad_z));
              if (grad_norm > 0.0) {
                normal_x = grad_x/grad_norm;
                normal_y = grad_y/grad_norm;
                normal_z = grad_z/grad_norm;
                Real center_x = size.x1min + (i-is+0.5)*size.dx1;
                Real center_y = size.x2min + (j-js+0.5)*size.dx2;
                Real center_z = size.x3min + (k-ks+0.5)*size.dx3;
                Real electron_density =
                    number_scale*primitive(m, IDN, k, j, i) +
                    grad_x*(x(r)-center_x);
                if (multi_d) electron_density += grad_y*(y(r)-center_y);
                if (three_d) electron_density += grad_z*(z(r)-center_z);
                Real approach = grad_x*nx(r) + grad_y*ny(r) + grad_z*nz(r);
                Real cosine = fabs(nx(r)*normal_x + ny(r)*normal_y +
                                   nz(r)*normal_z);
                Real critical_density =
                    CriticalDensity(wavelength(r)*length_scale_cgs);
                Real turning_density = critical_density*
                    (oblique_turning ? SQR(cosine) : 1.0);
                Real density_tolerance = 128.0*machine_eps*
                    fmax(turning_density, 1.0);
                if (approach > 0.0 && turning_density > 0.0) {
                  Real density_at_face = electron_density + approach*ds;
                  if (electron_density >= turning_density-density_tolerance) {
                    ds = 0.0;
                    turning_point = true;
                  } else if (density_at_face >=
                             turning_density-density_tolerance) {
                    ds = fmin(fmax((turning_density-electron_density)/approach,
                                   0.0), ds);
                    turning_point = true;
                  }
                }
              }
            }

            bool extinguished = false;
            if (ds > 0.0) {
              Real coefficient = constant_absorption(r);
              if (absorption_mode == inverse_bremsstrahlung) {
                // w0(iele) here is the stage state already updated by the duale
                // task, so the IB coefficient sees an O(dt) advanced electron
                // energy in cells with hydro flux — consistent with the overall
                // first-order operator splitting of the deposition source.
                Real density = fmax(primitive(m, IDN, k, j, i), 0.0);
                Real electron_energy =
                    fmax(primitive(m, electron_index, k, j, i), 0.0);
                Real electron_temperature = fmax(
                    gamma_minus_one*electron_energy/
                    electron_heat_capacity_fraction*temperature_scale_cgs,
                    ib_temperature_floor);
                Real electron_density = density*density_scale_cgs*
                    electron_number_per_gram;
                Real wavelength_cgs = wavelength(r)*length_scale_cgs;
                coefficient = InverseBremsstrahlungCoefficient(
                    electron_density, electron_temperature, zeff(r), wavelength_cgs,
                    fixed_coulomb_log)*length_scale_cgs;
              }
              coefficient = fmax(coefficient, 0.0);
              Real optical_depth = coefficient*ds;
              Real deposited = fmin(
                  DepositedPower(power(r), coefficient, ds), power(r));
              Real outgoing = fmax(power(r)-deposited, 0.0);
              if (coefficient > 0.0 &&
                  outgoing <= minimum_power_fraction*initial_power(r)) {
                deposited += outgoing;
                outgoing = 0.0;
                extinguished = true;
              }
              Real volume = size.dx1;
              if (multi_d) volume *= size.dx2;
              if (three_d) volume *= size.dx3;
              if (deposited > 0.0) {
                Kokkos::atomic_add(&data(m, 0, k, j, i), deposited/volume);
                Kokkos::atomic_add(&diag(1), deposited);
              }
              if (optical_depth > 0.0) {
                Kokkos::atomic_add(&data(m, 3, k, j, i), optical_depth);
              }
              power(r) = outgoing;
              Kokkos::atomic_add(&data(m, 2, k, j, i), 1.0);
              Kokkos::atomic_add(&data(m, 4, k, j, i), ds);
              Kokkos::atomic_add(&data(m, 5, k, j, i), nx(r)*ds);
              Kokkos::atomic_add(&data(m, 6, k, j, i), ny(r)*ds);
              Kokkos::atomic_add(&data(m, 7, k, j, i), nz(r)*ds);
              Kokkos::atomic_add(
                  &data(m, 9, k, j, i), (x(r)+0.5*nx(r)*ds)*ds);
              Kokkos::atomic_add(
                  &data(m, 10, k, j, i), (y(r)+0.5*ny(r)*ds)*ds);
              Kokkos::atomic_add(
                  &data(m, 11, k, j, i), (z(r)+0.5*nz(r)*ds)*ds);
              path(r) += ds;
              segments(r) += 1;
              x(r) += nx(r)*ds;
              y(r) += ny(r)*ds;
              z(r) += nz(r)*ds;
            }
            if (extinguished) {
              status(r) = static_cast<int>(RayStatus::absorbed);
              break;
            }
            if (turning_point) {
              if (reflections(r) >= max_reflections) {
                status(r) = static_cast<int>(RayStatus::remaining);
                Kokkos::atomic_add(&diag(3), power(r));
                Kokkos::atomic_inc(&counters(0));
                break;
              }
              Real normal_projection = nx(r)*normal_x + ny(r)*normal_y +
                                       nz(r)*normal_z;
              nx(r) -= 2.0*normal_projection*normal_x;
              ny(r) -= 2.0*normal_projection*normal_y;
              nz(r) -= 2.0*normal_projection*normal_z;
              Real direction_norm = sqrt(SQR(nx(r))+SQR(ny(r))+SQR(nz(r)));
              nx(r) /= direction_norm;
              ny(r) /= direction_norm;
              nz(r) /= direction_norm;
              reflections(r) += 1;
              Kokkos::atomic_inc(&counters(1));

              Real offset = fmax(reflection_offset_fraction*scale,
                                  128.0*machine_eps*fmax(scale, 1.0));
              x(r) += offset*nx(r);
              y(r) += offset*ny(r);
              z(r) += offset*nz(r);
              int reflected_m =
                  FindLocalBlock(sizes, nmb, x(r), y(r), z(r), multi_d, three_d);
              if (reflected_m < 0) {
                int reflected_gid = FindGlobalBlock(
                    global_blocks, nmb_total, x(r), y(r), z(r), multi_d, three_d);
                if (reflected_gid >= 0 &&
                    global_blocks(reflected_gid).rank != my_rank) {
                  LaserBlockInfo block = global_blocks(reflected_gid);
                  gid(r) = block.gid;
                  destination_rank(r) = block.rank;
                  ci(r) = ActiveCellIndex(x(r), block.x1min, block.dx1, is, ie);
                  cj(r) = multi_d ? ActiveCellIndex(
                      y(r), block.x2min, block.dx2, js, je) : js;
                  ck(r) = three_d ? ActiveCellIndex(
                      z(r), block.x3min, block.dx3, ks, ke) : ks;
                  status(r) = static_cast<int>(RayStatus::off_rank);
                  Kokkos::atomic_inc(&counters(2));
                } else if (reflected_gid < 0) {
                  status(r) = static_cast<int>(RayStatus::escaped);
                  Kokkos::atomic_add(&diag(2), power(r));
                } else {
                  status(r) = static_cast<int>(RayStatus::failed);
                  Kokkos::atomic_inc(&counters(3));
                }
                break;
              }
              gid(r) = gids(reflected_m);
              ci(r) = ActiveCellIndex(x(r), sizes(reflected_m).x1min,
                                      sizes(reflected_m).dx1, is, ie);
              cj(r) = multi_d ? ActiveCellIndex(
                  y(r), sizes(reflected_m).x2min,
                  sizes(reflected_m).dx2, js, je) : js;
              ck(r) = three_d ? ActiveCellIndex(
                  z(r), sizes(reflected_m).x3min,
                  sizes(reflected_m).dx3, ks, ke) : ks;
              continue;
            }

            Real tie = 128.0*machine_eps*fmax(ds, 1.0);
            if (fabs(sx-ds) <= tie) ci(r) += (nx(r) > 0.0) ? 1 : -1;
            if (multi_d && fabs(sy-ds) <= tie) cj(r) += (ny(r) > 0.0) ? 1 : -1;
            if (three_d && fabs(sz-ds) <= tie) ck(r) += (nz(r) > 0.0) ? 1 : -1;
            if (ci(r) >= is && ci(r) <= ie && cj(r) >= js && cj(r) <= je &&
                ck(r) >= ks && ck(r) <= ke) {
              continue;
            }

            // Probe infinitesimally into the destination so an exact block face is
            // assigned to the block on the forward side of the ray.
            Real probe = 128.0*machine_eps*fmax(scale, 1.0);
            Real px = x(r)+probe*nx(r);
            Real py = y(r)+probe*ny(r);
            Real pz = z(r)+probe*nz(r);
            int new_m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);

            if (new_m < 0 && periodic) {
              if (x(r) <= domain.x1min+tolerance && nx(r) < 0.0) x(r) = domain.x1max;
              if (x(r) >= domain.x1max-tolerance && nx(r) > 0.0) x(r) = domain.x1min;
              if (multi_d) {
                if (y(r) <= domain.x2min+tolerance && ny(r) < 0.0) y(r) = domain.x2max;
                if (y(r) >= domain.x2max-tolerance && ny(r) > 0.0) y(r) = domain.x2min;
              }
              if (three_d) {
                if (z(r) <= domain.x3min+tolerance && nz(r) < 0.0) z(r) = domain.x3max;
                if (z(r) >= domain.x3max-tolerance && nz(r) > 0.0) z(r) = domain.x3min;
              }
              px = x(r)+probe*nx(r);
              py = y(r)+probe*ny(r);
              pz = z(r)+probe*nz(r);
              new_m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);
            }

            if (new_m < 0) {
              bool inside_global = px >= domain.x1min && px < domain.x1max;
              if (multi_d) inside_global = inside_global &&
                  py >= domain.x2min && py < domain.x2max;
              if (three_d) inside_global = inside_global &&
                  pz >= domain.x3min && pz < domain.x3max;
              if (inside_global) {
                int destination_gid = FindGlobalBlock(
                    global_blocks, nmb_total, px, py, pz, multi_d, three_d);
                if (destination_gid >= 0 &&
                    global_blocks(destination_gid).rank != my_rank) {
                  LaserBlockInfo block = global_blocks(destination_gid);
                  gid(r) = block.gid;
                  destination_rank(r) = block.rank;
                  ci(r) = ActiveCellIndex(px, block.x1min, block.dx1, is, ie);
                  cj(r) = multi_d ? ActiveCellIndex(
                      py, block.x2min, block.dx2, js, je) : js;
                  ck(r) = three_d ? ActiveCellIndex(
                      pz, block.x3min, block.dx3, ks, ke) : ks;
                  status(r) = static_cast<int>(RayStatus::off_rank);
                  Kokkos::atomic_inc(&counters(2));
                } else {
                  status(r) = static_cast<int>(RayStatus::failed);
                  Kokkos::atomic_inc(&counters(3));
                }
              } else {
                status(r) = static_cast<int>(RayStatus::escaped);
                Kokkos::atomic_add(&diag(2), power(r));
              }
              break;
            }

            gid(r) = gids(new_m);
            ci(r) = ActiveCellIndex(px, sizes(new_m).x1min,
                                    sizes(new_m).dx1, is, ie);
            cj(r) = multi_d ? ActiveCellIndex(py, sizes(new_m).x2min,
                                              sizes(new_m).dx2, js, je) : js;
            ck(r) = three_d ? ActiveCellIndex(pz, sizes(new_m).x3min,
                                              sizes(new_m).dx3, ks, ke) : ks;
          }
        });
    CompactActiveQueue(current, next);
  }

  // Rays that hit the per-wave work cap stay active: the caller re-traces them in
  // subsequent waves (serial and MPI alike) so results are independent of the rank
  // decomposition. Only the final wave books leftovers as remaining power.
  if (!preserve_off_rank) {
    BookRemainingRays();
  }
}

//----------------------------------------------------------------------------------------

void Laser::TraceRefractiveRays(bool preserve_off_rank) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  int nmb_total = pmy_pack_->pmesh->nmb_total;
  int first_gid = pmy_pack_->gids;
  int my_rank = global_variable::my_rank;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto sizes = pmy_pack_->pmb->mb_size.d_view;
  auto gids = pmy_pack_->pmb->mb_gid.d_view;
  auto global_blocks = global_block_info_;
  RegionSize domain = pmy_pack_->pmesh->mesh_size;
  Real machine_eps = std::numeric_limits<Real>::epsilon();
  Real infinite = std::numeric_limits<Real>::max();

  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto wave_x = ray_kx_; auto wave_y = ray_ky_; auto wave_z = ray_kz_;
  auto dispersion_error = ray_dispersion_error_;
  auto power = ray_power;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status; auto destination_rank = ray_destination_rank_;
  auto segments = ray_segments_; auto path = ray_path_length_;
  auto data = cell_data;
  auto diag = device_diagnostics_; auto counters = device_counters_;
  auto primitive = pmy_pack_->pmhd->w0;
  auto constant_absorption = ray_constant_absorption_;
  auto wavelength = ray_wavelength_;
  auto zeff = ray_zeff_;
  auto initial_power = ray_power0_;
  bool periodic = periodic_transport_;
  int absorption_mode = static_cast<int>(absorption_model_);
  int inverse_bremsstrahlung =
      static_cast<int>(AbsorptionModel::inverse_bremsstrahlung);
  int electron_index = electron_index_;
  Real gamma_minus_one = gamma_minus_one_;
  Real electron_heat_capacity_fraction = electron_heat_capacity_fraction_;
  Real electron_number_per_gram = electron_number_per_gram_;
  Real density_scale_cgs = density_scale_cgs_;
  Real temperature_scale_cgs = temperature_scale_cgs_;
  Real length_scale_cgs = length_scale_cgs_;
  Real fixed_coulomb_log = inverse_bremsstrahlung_coulomb_log_;
  Real ib_temperature_floor = inverse_bremsstrahlung_temperature_floor_;
  Real minimum_power_fraction = minimum_power_fraction_;
  Real cell_fraction = refractive_cell_fraction_;
  Real curvature_fraction = refractive_curvature_fraction_;
  Real tau_max = refractive_tau_max_;
  int seg_per_launch = max_segments_per_launch_;

  SeedActiveQueue();
  for (int iteration = 0; iteration < max_transport_iterations_; ++iteration) {
    DvceArray1D<int> current = (iteration % 2 == 0) ? active_queue_a_ : active_queue_b_;
    DvceArray1D<int> next = (iteration % 2 == 0) ? active_queue_b_ : active_queue_a_;
    Kokkos::parallel_for(
        "laser_trace_refractive", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
        KOKKOS_LAMBDA(int queue_index) {
          int r = current(queue_index);
          if (r < 0 || status(r) != static_cast<int>(RayStatus::active)) return;

          for (int segment = 0; segment < seg_per_launch; ++segment) {
            if (status(r) != static_cast<int>(RayStatus::active)) break;
            int m = gid(r)-first_gid;
            if (m < 0 || m >= nmb) {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
              break;
            }
            RegionSize size = sizes(m);
            int i = ci(r), j = cj(r), k = ck(r);
            Real scale = fmin(size.dx1, multi_d ? size.dx2 : size.dx1);
            if (three_d) scale = fmin(scale, size.dx3);
            Real tolerance = 128.0*machine_eps*fmax(scale, 1.0);

            Real number_scale = density_scale_cgs*electron_number_per_gram;
            Real critical_density =
                CriticalDensity(wavelength(r)*length_scale_cgs);
            Real density = primitive(m, IDN, k, j, i);
            Real grad_x = number_scale*
                (primitive(m, IDN, k, j, i+1) -
                 primitive(m, IDN, k, j, i-1))/(2.0*size.dx1);
            Real hess_x = number_scale*
                (primitive(m, IDN, k, j, i+1) - 2.0*density +
                 primitive(m, IDN, k, j, i-1))/SQR(size.dx1);
            Real grad_y = 0.0;
            Real grad_z = 0.0;
            Real hess_y = 0.0;
            Real hess_z = 0.0;
            if (multi_d) {
              grad_y = number_scale*
                  (primitive(m, IDN, k, j+1, i) -
                   primitive(m, IDN, k, j-1, i))/(2.0*size.dx2);
              hess_y = number_scale*
                  (primitive(m, IDN, k, j+1, i) - 2.0*density +
                   primitive(m, IDN, k, j-1, i))/SQR(size.dx2);
            }
            if (three_d) {
              grad_z = number_scale*
                  (primitive(m, IDN, k+1, j, i) -
                   primitive(m, IDN, k-1, j, i))/(2.0*size.dx3);
              hess_z = number_scale*
                  (primitive(m, IDN, k+1, j, i) - 2.0*density +
                   primitive(m, IDN, k-1, j, i))/SQR(size.dx3);
            }
            Real center_x = size.x1min + (i-is+0.5)*size.dx1;
            Real center_y = size.x2min + (j-js+0.5)*size.dx2;
            Real center_z = size.x3min + (k-ks+0.5)*size.dx3;
            Real offset_x = x(r)-center_x;
            Real offset_y = y(r)-center_y;
            Real offset_z = z(r)-center_z;
            Real electron_density = number_scale*density +
                                    grad_x*offset_x +
                                    0.5*hess_x*SQR(offset_x);
            if (multi_d) {
              electron_density += grad_y*offset_y +
                                  0.5*hess_y*SQR(offset_y);
            }
            if (three_d) {
              electron_density += grad_z*offset_z +
                                  0.5*hess_z*SQR(offset_z);
            }
            Real normalized_density = electron_density/critical_density;
            Real normalized_grad_x = (grad_x+hess_x*offset_x)/critical_density;
            Real normalized_grad_y = (grad_y+hess_y*offset_y)/critical_density;
            Real normalized_grad_z = (grad_z+hess_z*offset_z)/critical_density;

            Real qx = wave_x(r), qy = wave_y(r), qz = wave_z(r);
            Real qnorm = sqrt(SQR(qx)+SQR(qy)+SQR(qz));
            if (!Kokkos::isfinite(qnorm) || !Kokkos::isfinite(normalized_density)) {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
              break;
            }
            Real local_dispersion = fabs(
                SQR(qx)+SQR(qy)+SQR(qz)+normalized_density-1.0);
            dispersion_error(r) = fmax(dispersion_error(r), local_dispersion);

            Real coefficient = constant_absorption(r);
            if (absorption_mode == inverse_bremsstrahlung) {
              // Same O(dt) state-timing note as in the straight tracer: w0(iele)
              // is the post-duale stage state.
              Real density = fmax(primitive(m, IDN, k, j, i), 0.0);
              Real electron_energy =
                  fmax(primitive(m, electron_index, k, j, i), 0.0);
              Real electron_temperature = fmax(
                  gamma_minus_one*electron_energy/
                  electron_heat_capacity_fraction*temperature_scale_cgs,
                  ib_temperature_floor);
              Real ne = density*density_scale_cgs*electron_number_per_gram;
              Real wavelength_cgs = wavelength(r)*length_scale_cgs;
              coefficient = InverseBremsstrahlungCoefficient(
                  ne, electron_temperature, zeff(r), wavelength_cgs,
                  fixed_coulomb_log)*length_scale_cgs;
            }
            coefficient = fmax(coefficient, 0.0);

            Real q_floor = sqrt(machine_eps);
            Real speed = fmax(qnorm, q_floor);
            Real step = cell_fraction*scale/speed;
            Real force = 0.5*sqrt(SQR(normalized_grad_x)+
                                  SQR(normalized_grad_y)+
                                  SQR(normalized_grad_z));
            if (force > 0.0) {
              step = fmin(step, curvature_fraction*speed/force);
            }
            if (coefficient > 0.0) {
              step = fmin(step, tau_max/(coefficient*speed));
            }

            // Re-evaluate the half-step wave vector after limiting the drift to the
            // nearest cell face. Two passes account for the step-dependent half kick.
            Real qhx = qx, qhy = qy, qhz = qz;
            for (int pass = 0; pass < 2; ++pass) {
              qhx = qx-0.25*step*normalized_grad_x;
              qhy = qy-0.25*step*normalized_grad_y;
              qhz = qz-0.25*step*normalized_grad_z;
              Real face_step = infinite;
              if (qhx > 0.0) {
                Real face = size.x1min + (i-is+1)*size.dx1;
                face_step = fmin(face_step, (face-x(r))/qhx);
              } else if (qhx < 0.0) {
                Real face = size.x1min + (i-is)*size.dx1;
                face_step = fmin(face_step, (face-x(r))/qhx);
              }
              if (multi_d) {
                if (qhy > 0.0) {
                  Real face = size.x2min + (j-js+1)*size.dx2;
                  face_step = fmin(face_step, (face-y(r))/qhy);
                } else if (qhy < 0.0) {
                  Real face = size.x2min + (j-js)*size.dx2;
                  face_step = fmin(face_step, (face-y(r))/qhy);
                }
              }
              if (three_d) {
                if (qhz > 0.0) {
                  Real face = size.x3min + (k-ks+1)*size.dx3;
                  face_step = fmin(face_step, (face-z(r))/qhz);
                } else if (qhz < 0.0) {
                  Real face = size.x3min + (k-ks)*size.dx3;
                  face_step = fmin(face_step, (face-z(r))/qhz);
                }
              }
              if (face_step >= 0.0 && face_step < step) step = face_step;
            }
            if (!(step >= 0.0) || !Kokkos::isfinite(step)) {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
              break;
            }

            // Kick-drift-kick: recompute the half kick with the final face-limited
            // step, drift, then finish the kick with the cell's quadratic force
            // model evaluated at the drift endpoint. The endpoint evaluation keeps
            // the trajectory second order in the step now that the force varies
            // within a cell (grad + hess*offset); with a constant force it reduces
            // exactly to the previous update.
            qhx = qx-0.25*step*normalized_grad_x;
            qhy = qy-0.25*step*normalized_grad_y;
            qhz = qz-0.25*step*normalized_grad_z;
            Real dx = step*qhx;
            Real dy = step*qhy;
            Real dz = step*qhz;
            Real end_grad_x = (grad_x+hess_x*(offset_x+dx))/critical_density;
            Real end_grad_y = (grad_y+hess_y*(offset_y+dy))/critical_density;
            Real end_grad_z = (grad_z+hess_z*(offset_z+dz))/critical_density;
            Real new_qx = qhx-0.25*step*end_grad_x;
            Real new_qy = qhy-0.25*step*end_grad_y;
            Real new_qz = qhz-0.25*step*end_grad_z;
            Real ds = sqrt(SQR(dx)+SQR(dy)+SQR(dz));
            wave_x(r) = new_qx;
            wave_y(r) = new_qy;
            wave_z(r) = new_qz;
            Real new_qnorm = sqrt(SQR(new_qx)+SQR(new_qy)+SQR(new_qz));
            if (new_qnorm > q_floor) {
              nx(r) = new_qx/new_qnorm;
              ny(r) = new_qy/new_qnorm;
              nz(r) = new_qz/new_qnorm;
            }
            if (ds <= tolerance*machine_eps) continue;

            Real deposited = fmin(
                DepositedPower(power(r), coefficient, ds), power(r));
            Real outgoing = fmax(power(r)-deposited, 0.0);
            bool extinguished = false;
            if (coefficient > 0.0 &&
                outgoing <= minimum_power_fraction*initial_power(r)) {
              deposited += outgoing;
              outgoing = 0.0;
              extinguished = true;
            }
            Real volume = size.dx1;
            if (multi_d) volume *= size.dx2;
            if (three_d) volume *= size.dx3;
            if (deposited > 0.0) {
              Kokkos::atomic_add(&data(m, 0, k, j, i), deposited/volume);
              Kokkos::atomic_add(&diag(1), deposited);
            }
            Real optical_depth = coefficient*ds;
            if (optical_depth > 0.0) {
              Kokkos::atomic_add(&data(m, 3, k, j, i), optical_depth);
            }
            Real tangent_x = dx/ds;
            Real tangent_y = dy/ds;
            Real tangent_z = dz/ds;
            Kokkos::atomic_add(&data(m, 2, k, j, i), 1.0);
            Kokkos::atomic_add(&data(m, 4, k, j, i), ds);
            Kokkos::atomic_add(&data(m, 5, k, j, i), tangent_x*ds);
            Kokkos::atomic_add(&data(m, 6, k, j, i), tangent_y*ds);
            Kokkos::atomic_add(&data(m, 7, k, j, i), tangent_z*ds);
            Kokkos::atomic_add(&data(m, 8, k, j, i), local_dispersion*ds);
            Kokkos::atomic_add(&data(m, 9, k, j, i), (x(r)+0.5*dx)*ds);
            Kokkos::atomic_add(&data(m, 10, k, j, i), (y(r)+0.5*dy)*ds);
            Kokkos::atomic_add(&data(m, 11, k, j, i), (z(r)+0.5*dz)*ds);
            power(r) = outgoing;
            path(r) += ds;
            segments(r) += 1;
            x(r) += dx;
            y(r) += dy;
            z(r) += dz;
            if (extinguished) {
              status(r) = static_cast<int>(RayStatus::absorbed);
              break;
            }

            Real probe = 128.0*machine_eps*fmax(scale, 1.0);
            Real px = x(r)+probe*nx(r);
            Real py = y(r)+probe*ny(r);
            Real pz = z(r)+probe*nz(r);
            int new_m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);
            if (new_m < 0 && periodic) {
              if (x(r) <= domain.x1min+tolerance && nx(r) < 0.0) x(r) = domain.x1max;
              if (x(r) >= domain.x1max-tolerance && nx(r) > 0.0) x(r) = domain.x1min;
              if (multi_d) {
                if (y(r) <= domain.x2min+tolerance && ny(r) < 0.0) y(r) = domain.x2max;
                if (y(r) >= domain.x2max-tolerance && ny(r) > 0.0) y(r) = domain.x2min;
              }
              if (three_d) {
                if (z(r) <= domain.x3min+tolerance && nz(r) < 0.0) z(r) = domain.x3max;
                if (z(r) >= domain.x3max-tolerance && nz(r) > 0.0) z(r) = domain.x3min;
              }
              px = x(r)+probe*nx(r);
              py = y(r)+probe*ny(r);
              pz = z(r)+probe*nz(r);
              new_m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);
            }

            if (new_m >= 0) {
              gid(r) = gids(new_m);
              ci(r) = ActiveCellIndex(px, sizes(new_m).x1min,
                                      sizes(new_m).dx1, is, ie);
              cj(r) = multi_d ? ActiveCellIndex(
                  py, sizes(new_m).x2min, sizes(new_m).dx2, js, je) : js;
              ck(r) = three_d ? ActiveCellIndex(
                  pz, sizes(new_m).x3min, sizes(new_m).dx3, ks, ke) : ks;
              continue;
            }

            int destination_gid = FindGlobalBlock(
                global_blocks, nmb_total, px, py, pz, multi_d, three_d);
            if (destination_gid >= 0 &&
                global_blocks(destination_gid).rank != my_rank) {
              LaserBlockInfo block = global_blocks(destination_gid);
              gid(r) = block.gid;
              destination_rank(r) = block.rank;
              ci(r) = ActiveCellIndex(px, block.x1min, block.dx1, is, ie);
              cj(r) = multi_d ? ActiveCellIndex(
                  py, block.x2min, block.dx2, js, je) : js;
              ck(r) = three_d ? ActiveCellIndex(
                  pz, block.x3min, block.dx3, ks, ke) : ks;
              status(r) = static_cast<int>(RayStatus::off_rank);
              Kokkos::atomic_inc(&counters(2));
            } else if (destination_gid < 0) {
              status(r) = static_cast<int>(RayStatus::escaped);
              Kokkos::atomic_add(&diag(2), power(r));
            } else {
              status(r) = static_cast<int>(RayStatus::failed);
              Kokkos::atomic_inc(&counters(3));
            }
            break;
          }
        });
    CompactActiveQueue(current, next);
  }

  // Rays that hit the per-wave work cap stay active: the caller re-traces them in
  // subsequent waves (serial and MPI alike) so results are independent of the rank
  // decomposition. Only the final wave books leftovers as remaining power.
  if (!preserve_off_rank) {
    BookRemainingRays();
  }
}

//----------------------------------------------------------------------------------------

void Laser::FinalizeDiagnostics() {
  auto host_diag = Kokkos::create_mirror_view(device_diagnostics_);
  auto host_count = Kokkos::create_mirror_view(device_counters_);
  auto host_segments = Kokkos::create_mirror_view(ray_segments_);
  auto host_path = Kokkos::create_mirror_view(ray_path_length_);
  auto host_dispersion = Kokkos::create_mirror_view(ray_dispersion_error_);
  Kokkos::deep_copy(host_diag, device_diagnostics_);
  Kokkos::deep_copy(host_count, device_counters_);
  Kokkos::deep_copy(host_segments, ray_segments_);
  Kokkos::deep_copy(host_path, ray_path_length_);
  Kokkos::deep_copy(host_dispersion, ray_dispersion_error_);
  Real global_diag[4] = {host_diag(0), host_diag(1), host_diag(2), host_diag(3)};
  int global_count[4] = {host_count(0), host_count(1), host_count(2), host_count(3)};
  int segment_count = 0;
  Real path_length = 0.0;
  Real max_dispersion_error = 0.0;
  for (int r = 0; r < nrays_; ++r) {
    segment_count += host_segments(r);
    path_length += host_path(r);
    max_dispersion_error = fmax(max_dispersion_error, host_dispersion(r));
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    Real reduced_diag[4] = {0.0, 0.0, 0.0, 0.0};
    int reduced_count[4] = {0, 0, 0, 0};
    int reduced_segments = 0;
    Real reduced_path = 0.0;
    Real reduced_dispersion_error = 0.0;
    int ierr = MPI_Allreduce(global_diag, reduced_diag, 4, MPI_ATHENA_REAL,
                             MPI_SUM, mpi_comm_);
    if (ierr == MPI_SUCCESS) {
      ierr = MPI_Allreduce(global_count, reduced_count, 4, MPI_INT,
                           MPI_SUM, mpi_comm_);
    }
    if (ierr == MPI_SUCCESS) {
      ierr = MPI_Allreduce(&segment_count, &reduced_segments, 1, MPI_INT,
                           MPI_SUM, mpi_comm_);
    }
    if (ierr == MPI_SUCCESS) {
      ierr = MPI_Allreduce(&path_length, &reduced_path, 1, MPI_ATHENA_REAL,
                           MPI_SUM, mpi_comm_);
    }
    if (ierr == MPI_SUCCESS) {
      ierr = MPI_Allreduce(&max_dispersion_error, &reduced_dispersion_error, 1,
                           MPI_ATHENA_REAL, MPI_MAX, mpi_comm_);
    }
    if (ierr != MPI_SUCCESS) {
      // print on every rank: this failure can be rank-local
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser MPI diagnostics reduction failed" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
      std::exit(EXIT_FAILURE);
    }
    for (int n = 0; n < 4; ++n) {
      global_diag[n] = reduced_diag[n];
      global_count[n] = reduced_count[n];
    }
    segment_count = reduced_segments;
    path_length = reduced_path;
    max_dispersion_error = reduced_dispersion_error;
  }
#endif

  diagnostics_.launched_power = global_diag[0];
  diagnostics_.deposited_power = global_diag[1];
  diagnostics_.escaped_power = global_diag[2];
  diagnostics_.remaining_power = global_diag[3];
  diagnostics_.active_rays = global_count[0];
  diagnostics_.reflected_rays = global_count[1];
  diagnostics_.off_rank_transfers = global_count[2];
  diagnostics_.transport_iterations = max_transport_iterations_*(mpi_wave_+1);
  diagnostics_.traced_segments = segment_count;
  diagnostics_.total_path_length = path_length;
  diagnostics_.max_dispersion_error = max_dispersion_error;
  Real mismatch = fabs(diagnostics_.launched_power-diagnostics_.deposited_power-
                       diagnostics_.escaped_power-diagnostics_.remaining_power);
  diagnostics_.conservation_residual = (diagnostics_.launched_power > 0.0)
      ? mismatch/diagnostics_.launched_power : 0.0;

  if (report_diagnostics_ && global_variable::my_rank == 0) {
    std::ios::fmtflags old_flags = std::cout.flags();
    std::streamsize old_precision = std::cout.precision();
    std::cout << std::scientific << std::setprecision(17)
              << "laser: launched=" << diagnostics_.launched_power
              << " deposited=" << diagnostics_.deposited_power
              << " escaped=" << diagnostics_.escaped_power
              << " remaining=" << diagnostics_.remaining_power
              << " residual=" << diagnostics_.conservation_residual
              << " active=" << diagnostics_.active_rays
              << " reflected=" << diagnostics_.reflected_rays
              << " transfers=" << diagnostics_.off_rank_transfers
              << " segments=" << diagnostics_.traced_segments
              << " path=" << diagnostics_.total_path_length
              << " dispersion=" << diagnostics_.max_dispersion_error << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
  }
  bool dispersion_failed = propagation_model_ == PropagationModel::refractive &&
      diagnostics_.max_dispersion_error > dispersion_tolerance_;
  if (global_count[3] > 0 ||
      diagnostics_.conservation_residual > conservation_tolerance_ ||
      dispersion_failed) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Laser transport failed for " << global_count[3]
                << " rays; conservation residual="
                << diagnostics_.conservation_residual << " exceeds tolerance="
                << conservation_tolerance_ << "; dispersion error="
                << diagnostics_.max_dispersion_error << " exceeds tolerance="
                << dispersion_tolerance_ << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
}

} // namespace laser
