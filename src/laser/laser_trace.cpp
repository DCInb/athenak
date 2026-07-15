//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file laser_trace.cpp
//! \brief GPU-resident aperture initialization, DDA marching, and queue compaction.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "laser/laser.hpp"
#include "laser/laser_physics.hpp"

namespace {

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
int ActiveCellIndex(Real x, Real xmin, Real dx, int is, int ie) {
  int index = is + static_cast<int>(floor((x-xmin)/dx));
  return (index < is) ? is : ((index > ie) ? ie : index);
}

} // namespace

namespace laser {

//----------------------------------------------------------------------------------------
//! Build a deterministic equal-area Fibonacci aperture. Gaussian weights are evaluated
//! at those equal-area samples and normalized exactly to the requested beam power.

void Laser::BuildInitialRays() {
  auto hx = Kokkos::create_mirror_view(ray_x0_);
  auto hy = Kokkos::create_mirror_view(ray_y0_);
  auto hz = Kokkos::create_mirror_view(ray_z0_);
  auto hnx = Kokkos::create_mirror_view(ray_nx0_);
  auto hny = Kokkos::create_mirror_view(ray_ny0_);
  auto hnz = Kokkos::create_mirror_view(ray_nz0_);
  auto hp = Kokkos::create_mirror_view(ray_power0_);
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
    std::vector<Real> weight(beam.nrays, 1.0);
    Real weight_sum = 0.0;
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
        weight[n] = std::exp(-2.0*SQR(sample_radius[n]/beam.radius));
      }
      weight_sum += weight[n];
    }

    for (int n = 0; n < beam.nrays; ++n, ++ray) {
      Real du = offset_u[n];
      Real dv = offset_v[n];
      hx(ray) = beam.origin[0] + du*basis_u[0] + dv*basis_v[0];
      hy(ray) = beam.origin[1] + du*basis_u[1] + dv*basis_v[1];
      hz(ray) = beam.origin[2] + du*basis_u[2] + dv*basis_v[2];
      hnx(ray) = beam.direction[0]; hny(ray) = beam.direction[1];
      hnz(ray) = beam.direction[2];
      hp(ray) = beam.power*weight[n]/weight_sum;
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
  Kokkos::deep_copy(ray_power0_, hp); Kokkos::deep_copy(ray_wavelength_, hlambda);
  Kokkos::deep_copy(ray_zeff_, hzbar); Kokkos::deep_copy(ray_constant_absorption_, hk);
  Kokkos::deep_copy(ray_start_time_, ht0); Kokkos::deep_copy(ray_end_time_, ht1);
  Kokkos::deep_copy(ray_beam_, hb);
}

//----------------------------------------------------------------------------------------
//! Restore immutable launch data and map every live ray into a local MeshBlock/cell.

void Laser::InitializeRays(Real time) {
  auto sizes = pmy_pack_->pmb->mb_size.d_view;
  auto gids = pmy_pack_->pmb->mb_gid.d_view;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int nmb = pmy_pack_->nmb_thispack;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  Real domain_scale = fmax(pmy_pack_->pmesh->mesh_size.x1max-
                           pmy_pack_->pmesh->mesh_size.x1min, 1.0);
  Real probe_eps = 128.0*std::numeric_limits<Real>::epsilon()*domain_scale;

  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto power = ray_power;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status;
  auto x0 = ray_x0_; auto y0 = ray_y0_; auto z0 = ray_z0_;
  auto nx0 = ray_nx0_; auto ny0 = ray_ny0_; auto nz0 = ray_nz0_;
  auto power0 = ray_power0_;
  auto start = ray_start_time_; auto end = ray_end_time_;
  auto segments = ray_segments_; auto path = ray_path_length_;
  auto queue_a = active_queue_a_; auto queue_b = active_queue_b_;
  auto diag = device_diagnostics_;

  Kokkos::parallel_for(
      "laser_initialize_rays", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        x(r) = x0(r); y(r) = y0(r); z(r) = z0(r);
        nx(r) = nx0(r); ny(r) = ny0(r); nz(r) = nz0(r);
        power(r) = (time >= start(r) && time <= end(r)) ? power0(r) : 0.0;
        segments(r) = 0;
        path(r) = 0.0;
        gid(r) = -1; ci(r) = is; cj(r) = js; ck(r) = ks;
        queue_a(r) = -1; queue_b(r) = -1;
        status(r) = static_cast<int>(RayStatus::inactive);
        if (!(power(r) > 0.0)) return;

        Kokkos::atomic_add(&diag(0), power(r));
        Real px = x(r) + probe_eps*nx(r);
        Real py = y(r) + probe_eps*ny(r);
        Real pz = z(r) + probe_eps*nz(r);
        int m = FindLocalBlock(sizes, nmb, px, py, pz, multi_d, three_d);
        if (m < 0) {
          status(r) = static_cast<int>(RayStatus::escaped);
          Kokkos::atomic_add(&diag(2), power(r));
          return;
        }
        gid(r) = gids(m);
        ci(r) = ActiveCellIndex(px, sizes(m).x1min, sizes(m).dx1, is, ie);
        if (multi_d) {
          cj(r) = ActiveCellIndex(py, sizes(m).x2min, sizes(m).dx2, js, je);
        }
        if (three_d) {
          ck(r) = ActiveCellIndex(pz, sizes(m).x3min, sizes(m).dx3, ks, ke);
        }
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
//! Straight Cartesian DDA. One device thread advances one ray through several cells.

void Laser::TraceStraightRays() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  int first_gid = pmy_pack_->gids;
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;
  auto sizes = pmy_pack_->pmb->mb_size.d_view;
  auto gids = pmy_pack_->pmb->mb_gid.d_view;
  RegionSize domain = pmy_pack_->pmesh->mesh_size;
  Real machine_eps = std::numeric_limits<Real>::epsilon();
  Real infinite = std::numeric_limits<Real>::max();

  auto x = ray_x; auto y = ray_y; auto z = ray_z;
  auto nx = ray_nx; auto ny = ray_ny; auto nz = ray_nz;
  auto power = ray_power;
  auto gid = ray_gid; auto ci = ray_i; auto cj = ray_j; auto ck = ray_k;
  auto status = ray_status; auto segments = ray_segments_;
  auto path = ray_path_length_; auto data = cell_data;
  auto diag = device_diagnostics_; auto counters = device_counters_;
  bool periodic = periodic_transport_;
  int seg_per_launch = max_segments_per_launch_;

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
            if (ds > 0.0) {
              Kokkos::atomic_add(&data(m, 2, k, j, i), 1.0);
              Kokkos::atomic_add(&data(m, 4, k, j, i), ds);
              path(r) += ds;
              segments(r) += 1;
              x(r) += nx(r)*ds;
              y(r) += ny(r)*ds;
              z(r) += nz(r)*ds;
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
                status(r) = static_cast<int>(RayStatus::off_rank);
                Kokkos::atomic_inc(&counters(2));
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

  // A fixed number of launch waves avoids a host synchronization in the propagation
  // loop. Rays that hit the configured work cap remain explicitly accounted for.
  auto remaining_status = ray_status;
  auto remaining_power = ray_power;
  Kokkos::parallel_for(
      "laser_mark_remaining", Kokkos::RangePolicy<>(DevExeSpace(), 0, nrays_),
      KOKKOS_LAMBDA(int r) {
        if (remaining_status(r) == static_cast<int>(RayStatus::active) ||
            remaining_status(r) == static_cast<int>(RayStatus::off_rank)) {
          remaining_status(r) = static_cast<int>(RayStatus::remaining);
          Kokkos::atomic_add(&diag(3), remaining_power(r));
          Kokkos::atomic_inc(&counters(0));
        }
      });
}

//----------------------------------------------------------------------------------------

void Laser::FinalizeDiagnostics() {
  auto host_diag = Kokkos::create_mirror_view(device_diagnostics_);
  auto host_count = Kokkos::create_mirror_view(device_counters_);
  auto host_segments = Kokkos::create_mirror_view(ray_segments_);
  auto host_path = Kokkos::create_mirror_view(ray_path_length_);
  Kokkos::deep_copy(host_diag, device_diagnostics_);
  Kokkos::deep_copy(host_count, device_counters_);
  Kokkos::deep_copy(host_segments, ray_segments_);
  Kokkos::deep_copy(host_path, ray_path_length_);
  diagnostics_.launched_power = host_diag(0);
  diagnostics_.deposited_power = host_diag(1);
  diagnostics_.escaped_power = host_diag(2);
  diagnostics_.remaining_power = host_diag(3);
  diagnostics_.active_rays = host_count(0);
  diagnostics_.reflected_rays = host_count(1);
  diagnostics_.off_rank_transfers = host_count(2);
  diagnostics_.transport_iterations = max_transport_iterations_;
  for (int r = 0; r < nrays_; ++r) {
    diagnostics_.traced_segments += host_segments(r);
    diagnostics_.total_path_length += host_path(r);
  }
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
              << " path=" << diagnostics_.total_path_length << std::endl;
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
  }
}

} // namespace laser
