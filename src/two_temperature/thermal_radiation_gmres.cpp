//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_gmres.cpp
//! \brief Restarted right-preconditioned (flexible) GMRES for the AMR operator.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"

namespace two_temperature {

bool ThermalRadiation::SolveImplicitAMRGMRES(
    const Real dt, MeshBoundaryValuesCC *pbval, const Real scale,
    int &iterations, int &restarts, Real &relative_residual, Real &backward_error) {
  RadiationTimerScope phase_timer(solver_timers_, RadiationTimer::GMRES);
  const auto &indcs = pmy_pack_->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int nji = nx1*nx2, nkji = nji*indcs.nx3;
  const int ncell = pmy_pack_->nmb_thispack*nkji;
  const int restart = std::min(implicit_gmres_restart_max_, implicit_max_iterations_);
  int window = std::min(implicit_gmres_restart_, restart);
  Real previous_beta = 0.0;
  int previous_used = 0;
  if (implicit_gmres_basis_.extent_int(0) != restart+1 ||
      implicit_gmres_basis_.extent_int(1) != ncell) {
    Kokkos::realloc(implicit_gmres_basis_, restart+1, ncell);
  }
  auto basis = implicit_gmres_basis_;
  // The low-order caller uses a component-relative metric only after PCG
  // reaches its arithmetic floor. It retains the original physical equation.
  const bool component_metric = implicit_preconditioner_mode_ == 5 &&
      !implicit_amr_high_order_;
  const bool flexible = implicit_amr_inner_iterations_ > 0 && !component_metric;
  const bool stored_corrections = flexible || implicit_preconditioner_mode_ == 5;
  if (stored_corrections && (implicit_gmres_corrections_.extent_int(0) != restart ||
                            implicit_gmres_corrections_.extent_int(1) != ncell)) {
    Kokkos::realloc(implicit_gmres_corrections_, restart, ncell);
  }
  auto correction_basis = implicit_gmres_corrections_;
  auto old = implicit_old_;
  auto solution = implicit_solution_;
  auto residual = implicit_residual_;
  auto direction = implicit_direction_;
  auto preconditioned = implicit_preconditioned_;
  auto applied = implicit_operator_;
  auto size = pmy_pack_->pmb->mb_size.d_view;
  const bool multilevel = pmy_pack_->pmesh->multilevel;
  bool preconditioner_metric = implicit_gmres_preconditioner_metric_ && !component_metric;
  DvceArray1D<Real> component_denominator, component_direction;
  if (component_metric) {
    component_denominator = DvceArray1D<Real>(Kokkos::view_alloc(
        Kokkos::WithoutInitializing, "native GMRES component denominator"), ncell);
    component_direction = DvceArray1D<Real>(Kokkos::view_alloc(
        Kokkos::WithoutInitializing, "native GMRES component projection"), ncell);
  }
  // Divide before forming products: spectral bands can span hundreds of
  // orders of magnitude, so explicitly squaring inverse weights is unsafe.
  // The max-scaled norm also avoids squaring a large intermediate ratio.
  auto component_max = [&](const DvceArray5D<Real> &values) {
    auto vector = values;
    Real maximum = 0.0;
    Kokkos::parallel_reduce("thermal_rad_gmres_component_max",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(int idx, Real &local_maximum) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      const Real value = fabs(vector(m, 0, k, j, i)/component_denominator(idx));
      local_maximum = fmax(local_maximum, Kokkos::isfinite(value) ? value :
          std::numeric_limits<Real>::infinity());
    }, Kokkos::Max<Real>(maximum));
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &maximum, 1, MPI_ATHENA_REAL,
                            MPI_MAX, MPI_COMM_WORLD);
#endif
    return maximum;
  };
  auto component_norm = [&](const DvceArray5D<Real> &values) {
    auto vector = values;
    const Real maximum = component_max(values);
    if (!(maximum > 0.0) || !std::isfinite(maximum)) return maximum;
    Real squared = 0.0;
    Kokkos::parallel_reduce("thermal_rad_gmres_component_norm",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(int idx, Real &sum) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      const Real value = (vector(m, 0, k, j, i)/component_denominator(idx))/maximum;
      sum += value*value;
    }, squared);
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &squared, 1, MPI_ATHENA_REAL,
                            MPI_SUM, MPI_COMM_WORLD);
#endif
    return maximum*std::sqrt(squared);
  };
  auto refresh_component_weights = [&]() {
    auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
    const Real noise_weight = std::min(1.0,
        32.0*std::numeric_limits<Real>::epsilon()/implicit_tolerance_);
    par_for("thermal_rad_gmres_component_weights", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      // Resolve mass/RHS-relative error except where evaluation roundoff in
      // the stiff stencil is larger. Each denominator candidate is <= the
      // original |b|+|A||x| bound for this monotone mass-plus-diffusion matrix.
      // The residual term handles initially zero cells receiving flux.
      const Real x = fabs(solution(m, 0, k, j, i));
      const Real value = fmax(fabs(old(m, 0, k, j, i))+x,
          fmax(noise_weight*diagonal(m, 0, k, j, i)*x,
               fabs(residual(m, 0, k, j, i))));
      component_denominator(idx) = value > 0.0 ? value : 1.0;
    });
  };
  bool cached_metric = false;
  // Append a few previous physical corrections to a restarted native volume
  // space (LGMRES augmentation). Z stores the actual corrections, so A*Z=V*H
  // remains the least-squares relation even for these non-Krylov columns.
  // Keep the total window bounded and retain at least 3/4 fresh directions.
  // The cached inverse metric needs B*v in every column and is not augmented.
  const int recycle_slots = implicit_preconditioner_mode_ == 5 && !flexible &&
      !preconditioner_metric && !component_metric ? std::min(3, restart/4) : 0;
  DvceArray2D<Real> recycled;
  if (recycle_slots > 0) {
    recycled = DvceArray2D<Real>(Kokkos::view_alloc(
        Kokkos::WithoutInitializing, "native GMRES restart corrections"), recycle_slots, ncell);
  }
  int recycle_count = 0, recycle_next = 0, recycled_columns = 0;
  int projected_misses = 0;
  std::vector<Real> h((restart+1)*restart), cosine(restart), sine(restart);
  std::vector<Real> projected(restart+1), update(restart);
  // Small tiled dot-product scratch; each row still uses the physical cell
  // volume. Two complete CGS passes retain reorthogonalization while replacing
  // one host/GPU synchronization and collective per vector by one per pass.
  constexpr int projection_tile = 4096;
  const int projection_tiles = (ncell+projection_tile-1)/projection_tile;
  DvceArray2D<Real> projection_partials;
  DvceArray1D<Real> projection_device;
  if (implicit_gmres_cgs2_) {
    projection_partials = DvceArray2D<Real>(Kokkos::view_alloc(
        Kokkos::WithoutInitializing, "GMRES projection partials"), restart, projection_tiles);
    projection_device = DvceArray1D<Real>("GMRES projections", restart);
  }
  auto projection_host = Kokkos::create_mirror_view(projection_device);
  // The last Arnoldi row is not a projection input: columns stop at restart-1.
  // Reuse it for packed weighted values until the final column normalizes v_R.
  auto projection_input = Kokkos::subview(basis, restart, Kokkos::ALL());
  const Real physical_target = implicit_tolerance_*std::sqrt(scale);
  Real target = component_metric ? implicit_tolerance_ : physical_target;
  iterations = 0;
  restarts = 0;
  int inner_cap_hits = 0;
  int inner_max_iterations = 0;
  bool projected_converged = false;
  relative_residual = std::numeric_limits<Real>::infinity();
  backward_error = std::numeric_limits<Real>::infinity();
  auto report = [&](const char *event, const char *status) {
    if (implicit_report_ && global_variable::my_rank == 0) {
      std::cout << "# implicit radiation AMR GMRES " << event
                << ": iterations=" << iterations
                << " restarts=" << restarts
                << " true_relative_residual=" << relative_residual
                << " backward_error=" << backward_error
                << " projected_converged=" << projected_converged
                << " inner_limit=" << implicit_amr_inner_iterations_
                << " inner_cap_hits=" << inner_cap_hits
                << " inner_max_iterations=" << inner_max_iterations
                << " inner_norm=" << (implicit_amr_inner_preconditioner_norm_ ?
                                      "preconditioner" : "volume")
                << " metric=" << (component_metric ? "componentwise" :
                    (preconditioner_metric ? "preconditioner" : "volume"))
                << " cached_metric=" << cached_metric
                << " stored_corrections=" << stored_corrections
                << " recycled_columns=" << recycled_columns
                << " projected_norm=" << (component_metric ? "component-infinity" :
                    (cached_metric ? "volume" : "arnoldi"))
                << " orthogonalization=" <<
                   (implicit_gmres_cgs2_ || preconditioner_metric ? "cgs2" : "mgs2")
                << " fused_projections=" << implicit_gmres_cgs2_;
      if (status != nullptr) std::cout << " status=" << status;
      std::cout << std::endl;
    }
  };
  auto finish = [&](const bool converged) {
    if (!converged && implicit_report_ && iterations > 0) {
      // Diagnostics only: distinguish a Krylov stall from the arithmetic floor
      // of a stiff matrix application. These estimates do not accept a solve.
      auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
      Real roundoff_norm = 0.0, diagonal_error = 0.0;
      Real maximum_diagonal = 0.0, maximum_solution = 0.0;
      Kokkos::parallel_reduce("thermal_rad_gmres_roundoff_norm",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(int idx, Real &sum) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        const Real amplitude = std::numeric_limits<Real>::epsilon()*diagonal(m, 0, k, j, i)*
            fabs(solution(m, 0, k, j, i));
        const Real volume = size(m).dx1*size(m).dx2*size(m).dx3;
        sum += volume*amplitude*amplitude;
      }, roundoff_norm);
      Kokkos::parallel_reduce("thermal_rad_gmres_diagonal_error",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(int idx, Real &maximum) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        const Real denominator = fabs(old(m, 0, k, j, i))+
            0.5*(1.0+diagonal(m, 0, k, j, i))*fabs(solution(m, 0, k, j, i));
        const Real numerator = fabs(residual(m, 0, k, j, i));
        const Real error = denominator > 0.0 ? numerator/denominator :
            (numerator == 0.0 ? 0.0 : std::numeric_limits<Real>::max());
        maximum = fmax(maximum, error);
      }, Kokkos::Max<Real>(diagonal_error));
      Kokkos::parallel_reduce("thermal_rad_gmres_max_diagonal",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(int idx, Real &maximum) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        maximum = fmax(maximum, diagonal(m, 0, k, j, i));
      }, Kokkos::Max<Real>(maximum_diagonal));
      Kokkos::parallel_reduce("thermal_rad_gmres_max_solution",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
      KOKKOS_LAMBDA(int idx, Real &maximum) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        maximum = fmax(maximum, fabs(solution(m, 0, k, j, i)));
      }, Kokkos::Max<Real>(maximum_solution));
#if MPI_PARALLEL_ENABLED
      RadiationBatchAllreduce(MPI_IN_PLACE, &roundoff_norm, 1, MPI_ATHENA_REAL,
                             MPI_SUM, MPI_COMM_WORLD);
      for (Real *value : {&diagonal_error, &maximum_diagonal, &maximum_solution}) {
        RadiationBatchAllreduce(MPI_IN_PLACE, value, 1, MPI_ATHENA_REAL,
                               MPI_MAX, MPI_COMM_WORLD);
      }
#endif
      if (global_variable::my_rank == 0) {
        std::cout << std::scientific << std::setprecision(17)
                  << "# implicit radiation AMR roundoff diagnostic: cycle="
                  << pmy_pack_->pmesh->ncycle << " scale=" << scale
                  << " estimated_relative_roundoff=" << std::sqrt(roundoff_norm/scale)
                  << " diagonal_scaled_error=" << diagonal_error
                  << " maximum_diagonal=" << maximum_diagonal
                  << " maximum_solution=" << maximum_solution << std::endl;
      }
    }
    report("finished", converged ? "converged" : "failed");
    return converged;
  };
  auto metric_target = [&]() {
    // The fixed AMR preconditioner B is Jacobi plus positive coarse terms:
    // B >= D^-1 in the volume inner product. Native multigrid's symmetric
    // degree-two Chebyshev smoother instead guarantees B >= (1/4) D^-1:
    // (1-q(lambda)^2)/lambda > 1/4 for 0 < lambda <= 2. Its positive coarse
    // correction can only increase this bound. One exact independent-set
    // elimination has a diagonally scaled triangular factor of norm <= 2,
    // so the combined native preconditioner has B >= (1/16) D^-1.
    // This conservative projected target never replaces the physical true check.
    auto diagonal = ImplicitAMRCellBand(implicit_amr_diagonal_);
    const Real invalid = std::numeric_limits<Real>::infinity();
    Real maximum_diagonal = 0.0;
    Kokkos::parallel_reduce("thermal_rad_gmres_metric_bound",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      const Real value = diagonal(m, 0, k, j, i);
      maximum = fmax(maximum, value > 0.0 && Kokkos::isfinite(value) ? value : invalid);
    }, Kokkos::Max<Real>(maximum_diagonal));
#if MPI_PARALLEL_ENABLED
    RadiationBatchAllreduce(MPI_IN_PLACE, &maximum_diagonal, 1, MPI_ATHENA_REAL,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    if (!(maximum_diagonal > 0.0) || !std::isfinite(maximum_diagonal)) {
      return false;
    }
    target = implicit_tolerance_*std::sqrt(scale)*
        (implicit_preconditioner_mode_ == 5 ? 0.25 : 1.0)/std::sqrt(maximum_diagonal);
    return true;
  };
  if (preconditioner_metric && !metric_target()) return finish(false);

  // Approximate the inverse of a positive, volume-symmetric two-point graph.
  // Retain the frozen positive face conductances, but disable reconstruction in
  // the inner operator. Reflux then gives matching integrated fluxes on both
  // sides of every coarse/fine face. Truncated PCG depends on its RHS, so the
  // outer iteration must retain each preconditioned vector separately (FGMRES).
  auto inner_cg = [&](const int column) {
    RadiationTimerScope inner_timer(solver_timers_, RadiationTimer::InnerCG);
    const bool outer_high_order = implicit_amr_high_order_;
    implicit_amr_high_order_ = false;
    auto z = implicit_amr_inner_z_;
    Kokkos::deep_copy(DevExeSpace(), implicit_preconditioned_, 0.0);
    par_for("thermal_rad_fgmres_inner_rhs", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      residual(m, 0, k, j, i) = basis(column, idx);
    });
    const Real rhs_norm = ImplicitGlobalDot(implicit_residual_, implicit_residual_);
    ApplyImplicitPreconditioner(implicit_residual_, implicit_amr_inner_z_, dt);
    par_for("thermal_rad_fgmres_inner_direction", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      direction(m, 0, k, j, i) = z(m, 0, k, j, i);
    });
    Real rz = ImplicitGlobalDot(implicit_residual_, implicit_amr_inner_z_);
    const Real initial_rz = rz;
    bool valid = rhs_norm > 0.0 && std::isfinite(rhs_norm) &&
        rz > 0.0 && std::isfinite(rz);
    const int max_inner_iterations = implicit_amr_inner_iterations_;
    constexpr Real inner_tolerance_squared = 1.0e-2;
    for (int inner = 0; valid && inner < max_inner_iterations; ++inner) {
      solver_timers_.Add(RadiationCounter::InnerIterations);
      solver_timers_.ObserveInnerIterations(inner+1);
      // All inner search directions satisfy homogeneous physical boundaries.
      // The original RHS, outer solution and Arnoldi basis remain untouched.
      ExchangeImplicitField(implicit_direction_, pbval, true, false, true);
      ApplyImplicitOperator(implicit_direction_, implicit_operator_, dt);
      const Real pap = ImplicitGlobalDot(implicit_direction_, implicit_operator_);
      if (!(pap > 0.0) || !std::isfinite(pap)) {
        valid = false;
        break;
      }
      const Real step = rz/pap;
      if (!std::isfinite(step)) {
        valid = false;
        break;
      }
      Real norm = 0.0;
      {
        // Accumulate the same volume norm as ImplicitGlobalDot while updating
        // the residual, avoiding a second full-array read and kernel launch.
        // The dot phase includes the fused vector update and keeps one call.
        RadiationTimerScope dot_timer(solver_timers_, RadiationTimer::Dot);
        Kokkos::parallel_reduce("thermal_rad_fgmres_inner_update_norm",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
        KOKKOS_LAMBDA(int idx, Real &sum) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          preconditioned(m, 0, k, j, i) += step*direction(m, 0, k, j, i);
          const Real updated_residual = residual(m, 0, k, j, i)-
                                        step*applied(m, 0, k, j, i);
          residual(m, 0, k, j, i) = updated_residual;
          const Real volume = multilevel ? size(m).dx1*size(m).dx2*size(m).dx3 : 1.0;
          sum += volume*updated_residual*updated_residual;
        }, norm);
#if MPI_PARALLEL_ENABLED
        if (global_variable::nranks > 1) {
          RadiationBatchAllreduce(MPI_IN_PLACE, &norm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
      }
      inner_max_iterations = std::max(inner_max_iterations, inner+1);
      if (!std::isfinite(norm)) {
        valid = false;
        break;
      }
      Real rz_new = 0.0;
      if (implicit_amr_inner_preconditioner_norm_) {
        if (norm == 0.0) break;
        // Use the fixed positive preconditioner as an optional residual norm.
        // This is independent of the outer Arnoldi metric and changes only the
        // accuracy of its variable correction, never the physical true check.
        ApplyImplicitPreconditioner(implicit_residual_, implicit_amr_inner_z_, dt);
        rz_new = ImplicitGlobalDot(implicit_residual_, implicit_amr_inner_z_);
        if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {
          valid = false;
          break;
        }
      }
      const Real stopping_norm = implicit_amr_inner_preconditioner_norm_ ? rz_new : norm;
      const Real initial_norm =
          implicit_amr_inner_preconditioner_norm_ ? initial_rz : rhs_norm;
      if (stopping_norm <= inner_tolerance_squared*initial_norm ||
          inner+1 == max_inner_iterations) {
        if (stopping_norm > inner_tolerance_squared*initial_norm) {
          ++inner_cap_hits;
          solver_timers_.Add(RadiationCounter::InnerCapHits);
        }
        break;
      }
      if (!implicit_amr_inner_preconditioner_norm_) {
        ApplyImplicitPreconditioner(implicit_residual_, implicit_amr_inner_z_, dt);
        rz_new = ImplicitGlobalDot(implicit_residual_, implicit_amr_inner_z_);
      }
      const Real ratio = rz_new/rz;
      if (!(rz_new > 0.0) || !std::isfinite(rz_new) || !std::isfinite(ratio)) {
        valid = false;
        break;
      }
      par_for("thermal_rad_fgmres_inner_recur", DevExeSpace(), 0, ncell-1,
      KOKKOS_LAMBDA(int idx) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        direction(m, 0, k, j, i) = z(m, 0, k, j, i)+
            ratio*direction(m, 0, k, j, i);
      });
      rz = rz_new;
    }
    implicit_amr_high_order_ = outer_high_order;
    return valid;
  };

  while (true) {
    // Always test the original equation, including inhomogeneous physical data.
    ExchangeImplicitField(implicit_solution_, pbval, false);
    ApplyImplicitOperator(implicit_solution_, implicit_operator_, dt);
    par_for("thermal_rad_gmres_true_residual", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      residual(m, 0, k, j, i) = old(m, 0, k, j, i)-applied(m, 0, k, j, i);
    });
    const Real physical_beta = std::sqrt(ImplicitGlobalDot(implicit_residual_,
                                                          implicit_residual_));
    relative_residual = physical_beta/std::sqrt(scale);
    backward_error = ImplicitComponentwiseBackwardError(
        implicit_solution_, implicit_old_, implicit_residual_, dt);
    if (restarts == 1 || (restarts > 0 && restarts%10 == 0)) report("progress", nullptr);
    if (!std::isfinite(relative_residual) || !std::isfinite(backward_error)) {
      return finish(false);
    }
    Real component_error = 0.0;
    if (component_metric) {
      refresh_component_weights();
      component_error = component_max(implicit_residual_);
    }
    if ((relative_residual <= 16.0*implicit_tolerance_ ||
         ((projected_converged || component_metric) &&
          backward_error <= implicit_tolerance_)) &&
        component_error <= implicit_tolerance_) {
      return finish(true);
    }
    if (iterations >= implicit_max_iterations_ || !(physical_beta > 0.0)) {
      return finish(false);
    }
    // With very stiff, heterogeneous diffusion the volume residual can reach
    // the roundoff floor in high-diagonal cells while errors in weakly weighted
    // cells remain unresolved. Repeating short projected-converged cycles then
    // makes no progress. Change the Arnoldi metric to the fixed SPD native
    // inverse after repeated failed true checks. This redistributes accuracy;
    // neither physical acceptance test nor the total iteration budget changes.
    projected_misses = projected_converged ? projected_misses+1 : 0;
    if (implicit_preconditioner_mode_ == 5 && !preconditioner_metric &&
        !component_metric &&
        projected_misses >= 3) {
      preconditioner_metric = true;
      if (!metric_target()) return finish(false);
      if (implicit_report_ && global_variable::my_rank == 0) {
        std::cout << "# implicit radiation AMR GMRES metric switch: iterations="
                  << iterations << " from=volume to=preconditioner"
                  << " true_relative_residual=" << relative_residual
                  << " backward_error=" << backward_error << std::endl;
      }
    }
    // A short restarted basis can repeatedly discard the slow modes of a stiff,
    // heterogeneous diffusion operator. Grow only after a cycle reduces the
    // true residual by less than a factor ten or fills the available basis,
    // up to the explicit memory bound.
    // The equation, iteration budget and all acceptance tests stay unchanged.
    if (previous_beta > 0.0 && window < restart &&
        (previous_used == window || physical_beta > 0.1*previous_beta)) {
      const int next_window = std::min(restart, 2*window);
      if (implicit_report_ && global_variable::my_rank == 0) {
        std::cout << "# implicit radiation AMR GMRES restart growth: from=" << window
                  << " to=" << next_window
                  << " residual_ratio=" << physical_beta/previous_beta << std::endl;
      }
      window = next_window;
      // The same worker solves a sequence of groups and timesteps. Remember
      // the required native window instead of paying for the same failed short
      // restarts on every band. Easy systems still stop at projected convergence.
      if (implicit_preconditioner_mode_ == 5) implicit_gmres_restart_ = window;
    }
    previous_beta = physical_beta;
    // For the fixed, volume-symmetric native inverse B, retain p_i = B*v_i.
    // Then <v_i,B*w>_V = <p_i,w>_V: both projection passes need no additional
    // multigrid applications. Compute B*w afresh after orthogonalization for
    // the norm and the next p_i, rather than recursively subtracting B*v_i
    // from B*w (which can lose positivity through cancellation).
    cached_metric = preconditioner_metric && !flexible && implicit_preconditioner_mode_ == 5;
    if (cached_metric && (implicit_gmres_corrections_.extent_int(0) != restart ||
                          implicit_gmres_corrections_.extent_int(1) != ncell)) {
      Kokkos::realloc(implicit_gmres_corrections_, restart, ncell);
      correction_basis = implicit_gmres_corrections_;
    }
    Real beta = physical_beta;
    if (component_metric) {
      beta = component_norm(implicit_residual_);
      if (!(beta > 0.0) || !std::isfinite(beta)) return finish(false);
    }
    if (preconditioner_metric) {
      ApplyImplicitPreconditioner(implicit_residual_, implicit_preconditioned_, dt);
      beta = std::sqrt(ImplicitGlobalDot(implicit_residual_, implicit_preconditioned_));
      if (!(beta > 0.0) || !std::isfinite(beta)) return finish(false);
    }

    par_for("thermal_rad_gmres_first_vector", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      basis(0, idx) = residual(m, 0, k, j, i)/beta;
      if (component_metric) component_direction(idx) = basis(0, idx);
      if (cached_metric) {
        correction_basis(0, idx) = preconditioned(m, 0, k, j, i)/beta;
        // With cached B*v, the direction scratch is free throughout Arnoldi.
        // Retain the last column of V*Q^T for the physical projected residual.
        direction(m, 0, k, j, i) = basis(0, idx);
      }
    });
    std::fill(h.begin(), h.end(), 0.0);
    std::fill(projected.begin(), projected.end(), 0.0);
    projected[0] = beta;
    projected_converged = false;
    const int augment = !preconditioner_metric ? std::min(recycle_count, window/4) : 0;
    const int fresh_columns = window-augment;
    int used = 0;
    for (int column = 0; column < window && iterations < implicit_max_iterations_;
         ++column) {
      if (column >= fresh_columns) {
        const int slot = (recycle_next-1-(column-fresh_columns)+recycle_slots)%recycle_slots;
        par_for("thermal_rad_gmres_load_recycled_correction", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          preconditioned(m, 0, k, j, i) = recycled(slot, idx);
        });
        ++recycled_columns;
      } else if (flexible) {
        if (!inner_cg(column)) return finish(false);
      } else if (cached_metric) {
        par_for("thermal_rad_gmres_load_correction", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          preconditioned(m, 0, k, j, i) = correction_basis(column, idx);
        });
      } else {
        par_for("thermal_rad_gmres_load_vector", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          direction(m, 0, k, j, i) = basis(column, idx);
        });
        ApplyImplicitPreconditioner(implicit_direction_, implicit_preconditioned_, dt);
      }
      if (stored_corrections && !cached_metric) {
        // Retain the vectors actually multiplied by A. Even a mathematically
        // fixed multigrid map is only linear up to roundoff; applying it to a
        // combined basis at restart can amplify that discrepancy in stiff A.
        par_for("thermal_rad_gmres_store_correction", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          correction_basis(column, idx) = preconditioned(m, 0, k, j, i);
        });
      }
      ExchangeImplicitField(implicit_preconditioned_, pbval, true);
      ApplyImplicitOperator(implicit_preconditioned_, implicit_operator_, dt);
      auto projection_basis = cached_metric ? correction_basis : basis;
      if (implicit_gmres_cgs2_) {
        for (int pass = 0; pass < 2; ++pass) {
          if (preconditioner_metric && !cached_metric) {
            ApplyImplicitPreconditioner(implicit_operator_, implicit_preconditioned_, dt);
          }
          auto weighted = preconditioner_metric && !cached_metric ? preconditioned : applied;
          const int rows = column+1;
          par_for("thermal_rad_gmres_cgs2_pack", DevExeSpace(), 0, ncell-1,
          KOKKOS_LAMBDA(int idx) {
            const int m = idx/nkji, local = idx%nkji;
            const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
            const Real volume = size(m).dx1*size(m).dx2*size(m).dx3;
            projection_input(idx) = component_metric ?
                weighted(m, 0, k, j, i)/component_denominator(idx) :
                volume*weighted(m, 0, k, j, i);
          });
          Kokkos::parallel_for("thermal_rad_gmres_cgs2_partial",
              Kokkos::TeamPolicy<>(rows*projection_tiles, Kokkos::AUTO),
          KOKKOS_LAMBDA(const TeamMember_t &team) {
            const int row = team.league_rank()/projection_tiles;
            const int tile = team.league_rank()%projection_tiles;
            const int begin = tile*projection_tile;
            const int end = begin+projection_tile < ncell ? begin+projection_tile : ncell;
            Real dot = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, begin, end),
            [&](const int idx, Real &sum) {
              sum += (component_metric ?
                  projection_basis(row, idx)/component_denominator(idx) :
                  projection_basis(row, idx))*projection_input(idx);
            }, dot);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              projection_partials(row, tile) = dot;
            });
          });
          Kokkos::parallel_for("thermal_rad_gmres_cgs2_sum",
              Kokkos::TeamPolicy<>(rows, Kokkos::AUTO),
          KOKKOS_LAMBDA(const TeamMember_t &team) {
            const int row = team.league_rank();
            Real dot = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, projection_tiles),
            [&](const int tile, Real &sum) { sum += projection_partials(row, tile); }, dot);
            Kokkos::single(Kokkos::PerTeam(team), [&]() { projection_device(row) = dot; });
          });
          Kokkos::deep_copy(projection_host, projection_device);
#if MPI_PARALLEL_ENABLED
          RadiationBatchAllreduce(MPI_IN_PLACE, projection_host.data(), rows,
              MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
          for (int row = 0; row < rows; ++row) {
            h[row*restart+column] += projection_host(row);
          }
          Kokkos::deep_copy(projection_device, projection_host);
          par_for("thermal_rad_gmres_cgs2_subtract", DevExeSpace(), 0, ncell-1,
          KOKKOS_LAMBDA(int idx) {
            const int m = idx/nkji, local = idx%nkji;
            const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
            Real value = applied(m, 0, k, j, i);
            for (int row = 0; row < rows; ++row) {
              value -= projection_device(row)*basis(row, idx);
            }
            applied(m, 0, k, j, i) = value;
          });
        }
      } else {
        // The default retains MGS2 in the volume metric. In the fixed B metric,
        // use CGS2: all projections precede subtraction. The cached fixed native
        // inverse supplies B*v_i; other preconditioners still apply B per pass.
        for (int pass = 0; pass < 2; ++pass) {
          if (preconditioner_metric && !cached_metric) {
            ApplyImplicitPreconditioner(implicit_operator_, implicit_preconditioned_, dt);
          }
          auto weighted = preconditioner_metric && !cached_metric ? preconditioned : applied;
          for (int row = 0; row <= column; ++row) {
            Real projection = 0.0;
            Kokkos::parallel_reduce("thermal_rad_gmres_projection",
                Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
            KOKKOS_LAMBDA(int idx, Real &sum) {
              const int m = idx/nkji, local = idx%nkji;
              const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
              const Real volume = size(m).dx1*size(m).dx2*size(m).dx3;
              sum += component_metric ?
                  (projection_basis(row, idx)/component_denominator(idx))*
                  (weighted(m, 0, k, j, i)/component_denominator(idx)) :
                  volume*projection_basis(row, idx)*weighted(m, 0, k, j, i);
            }, projection);
#if MPI_PARALLEL_ENABLED
            RadiationBatchAllreduce(MPI_IN_PLACE, &projection, 1, MPI_ATHENA_REAL,
                          MPI_SUM, MPI_COMM_WORLD);
#endif
            h[row*restart+column] += projection;
            if (preconditioner_metric) {
              update[row] = projection;  // Host scratch; triangular solve overwrites it.
              continue;
            }
            par_for("thermal_rad_gmres_orthogonalize", DevExeSpace(), 0, ncell-1,
            KOKKOS_LAMBDA(int idx) {
              const int m = idx/nkji, local = idx%nkji;
              const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
              applied(m, 0, k, j, i) -= projection*basis(row, idx);
            });
          }
          if (preconditioner_metric) {
            for (int row = 0; row <= column; ++row) {
              const Real projection = update[row];
              par_for("thermal_rad_gmres_metric_orthogonalize", DevExeSpace(), 0, ncell-1,
              KOKKOS_LAMBDA(int idx) {
                const int m = idx/nkji, local = idx%nkji;
                const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
                applied(m, 0, k, j, i) -= projection*basis(row, idx);
              });
            }
          }
        }
      }
      if (preconditioner_metric) {
        ApplyImplicitPreconditioner(implicit_operator_, implicit_preconditioned_, dt);
      }
      const Real next_norm = component_metric ? component_norm(implicit_operator_) :
          std::sqrt(ImplicitGlobalDot(implicit_operator_,
              preconditioner_metric ? implicit_preconditioned_ : implicit_operator_));
      if (!std::isfinite(next_norm)) return finish(false);
      h[(column+1)*restart+column] = next_norm;
      if (next_norm > std::numeric_limits<Real>::min()) {
        par_for("thermal_rad_gmres_next_vector", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          basis(column+1, idx) = applied(m, 0, k, j, i)/next_norm;
          if (cached_metric && column+1 < restart) {
            correction_basis(column+1, idx) = preconditioned(m, 0, k, j, i)/next_norm;
          }
        });
      }
      for (int row = 0; row < column; ++row) {
        const Real a = h[row*restart+column];
        const Real b = h[(row+1)*restart+column];
        h[row*restart+column] = cosine[row]*a+sine[row]*b;
        h[(row+1)*restart+column] = -sine[row]*a+cosine[row]*b;
      }
      const Real a = h[column*restart+column];
      const Real b = h[(column+1)*restart+column];
      const Real magnitude = std::hypot(a, b);
      if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return finish(false);
      cosine[column] = a/magnitude;
      sine[column] = b/magnitude;
      h[column*restart+column] = magnitude;
      h[(column+1)*restart+column] = 0.0;
      projected[column+1] = -sine[column]*projected[column];
      projected[column] *= cosine[column];
      ++iterations;
      solver_timers_.Add(RadiationCounter::OuterIterations);
      used = column+1;
      projected_converged = std::abs(projected[column+1]) <= target;
      if (component_metric && next_norm > std::numeric_limits<Real>::min()) {
        // Track V*Q^T's final column, as in the physical cached-metric check.
        // The infinity norm avoids a sqrt(N) penalty from many harmless
        // roundoff-sized residuals in a large bright region.
        const Real c = cosine[column], s = sine[column], g = projected[column+1];
        Real model_error = 0.0;
        Kokkos::parallel_reduce("thermal_rad_gmres_component_projection",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
        KOKKOS_LAMBDA(int idx, Real &maximum) {
          const Real value = -s*component_direction(idx)+c*basis(column+1, idx);
          component_direction(idx) = value;
          const Real error = fabs((value/component_denominator(idx))*g);
          maximum = fmax(maximum, Kokkos::isfinite(error) ? error :
              std::numeric_limits<Real>::infinity());
        }, Kokkos::Max<Real>(model_error));
#if MPI_PARALLEL_ENABLED
        RadiationBatchAllreduce(MPI_IN_PLACE, &model_error, 1, MPI_ATHENA_REAL,
                                MPI_MAX, MPI_COMM_WORLD);
#endif
        if (!std::isfinite(model_error)) return finish(false);
        projected_converged = model_error <= implicit_tolerance_;
      }
      if (cached_metric && next_norm > std::numeric_limits<Real>::min()) {
        // The metric-norm residual bound based on max(D) can oversolve by many
        // orders of magnitude. After the new Givens rotation, the projected
        // residual is g_{k+1}*(-s_k*d_k+c_k*v_{k+1}), where d_k is the previous
        // last column of V*Q^T. Measure this vector in physical volume directly.
        // This is the same physical projected target, with no worst-case bound;
        // the independently recomputed true equation still decides acceptance.
        const Real c = cosine[column], s = sine[column];
        const Real g = projected[column+1];
        Real model_norm = 0.0;
        Kokkos::parallel_reduce("thermal_rad_gmres_physical_projection",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
        KOKKOS_LAMBDA(int idx, Real &sum) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          const Real value = -s*direction(m, 0, k, j, i)+c*basis(column+1, idx);
          direction(m, 0, k, j, i) = value;
          const Real model_residual = g*value;
          const Real volume = size(m).dx1*size(m).dx2*size(m).dx3;
          sum += volume*model_residual*model_residual;
        }, model_norm);
#if MPI_PARALLEL_ENABLED
        RadiationBatchAllreduce(MPI_IN_PLACE, &model_norm, 1, MPI_ATHENA_REAL,
                               MPI_SUM, MPI_COMM_WORLD);
#endif
        if (!std::isfinite(model_norm)) return finish(false);
        projected_converged = std::sqrt(model_norm) <= physical_target;
      }
      if (projected_converged ||
          next_norm <= std::numeric_limits<Real>::min()) break;
    }
    for (int row = used-1; row >= 0; --row) {
      Real value = projected[row];
      for (int column = row+1; column < used; ++column) {
        value -= h[row*restart+column]*update[column];
      }
      update[row] = value/h[row*restart+row];
      if (!std::isfinite(update[row])) return finish(false);
    }
    previous_used = used;
    // Combine stored corrections when available; otherwise apply the fixed
    // inverse once to the combined Arnoldi vectors.
    auto combination = stored_corrections ? preconditioned : direction;
    auto vectors = stored_corrections ? correction_basis : basis;
    if (implicit_gmres_cgs2_) {
      for (int row = 0; row < used; ++row) projection_host(row) = update[row];
      Kokkos::deep_copy(projection_device, projection_host);
      par_for("thermal_rad_gmres_cgs2_combine", DevExeSpace(), 0, ncell-1,
      KOKKOS_LAMBDA(int idx) {
        const int m = idx/nkji, local = idx%nkji;
        const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
        Real value = 0.0;
        for (int row = 0; row < used; ++row) value += projection_device(row)*vectors(row, idx);
        combination(m, 0, k, j, i) = value;
      });
    } else {
      Kokkos::deep_copy(DevExeSpace(), combination, 0.0);
      for (int row = 0; row < used; ++row) {
        const Real weight = update[row];
        par_for("thermal_rad_gmres_combine", DevExeSpace(), 0, ncell-1,
        KOKKOS_LAMBDA(int idx) {
          const int m = idx/nkji, local = idx%nkji;
          const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
          combination(m, 0, k, j, i) += weight*vectors(row, idx);
        });
      }
    }
    if (!stored_corrections) {
      ApplyImplicitPreconditioner(implicit_direction_, implicit_preconditioned_, dt);
    }
    // Normalize the actual update, not a difference of nearby solution fields.
    // Recompute A times each retained correction when it is reused; the true
    // residual check still uses the original physical equation at every restart.
    Real recycle_norm = 0.0;
    if (recycle_slots > 0 && !preconditioner_metric) {
      recycle_norm = std::sqrt(ImplicitGlobalDot(implicit_preconditioned_,
                                               implicit_preconditioned_));
    }
    const bool save_recycle = recycle_norm > 0.0 && std::isfinite(recycle_norm);
    const int save_slot = recycle_next;
    par_for("thermal_rad_gmres_update", DevExeSpace(), 0, ncell-1,
    KOKKOS_LAMBDA(int idx) {
      const int m = idx/nkji, local = idx%nkji;
      const int k = local/nji+ks, j = (local%nji)/nx1+js, i = local%nx1+is;
      solution(m, 0, k, j, i) += preconditioned(m, 0, k, j, i);
      if (save_recycle) recycled(save_slot, idx) = preconditioned(m, 0, k, j, i)/recycle_norm;
    });
    if (save_recycle) {
      recycle_next = (recycle_next+1)%recycle_slots;
      recycle_count = std::min(recycle_slots, recycle_count+1);
    }
    ++restarts;
  }
}

}  // namespace two_temperature
