#include "two_temperature/thermal_radiation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "globals.hpp"
#include "diffusion/native_diffusion_solver.hpp"
#include "mesh/mesh.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace two_temperature {
void ReleaseImplicitAMRBatch(ImplicitAMRAggregateState *state);
void TrimImplicitAMRReuseCache(ImplicitAMRAggregateState *, std::size_t, std::size_t);
std::size_t ImplicitAMRReusableBytes(const ImplicitAMRAggregateState *state);
std::size_t ImplicitAMRCellWorkspaceBytes(const ImplicitAMRAggregateState *state);

struct ImplicitRadiationGroupWorkers {
  std::vector<std::unique_ptr<ThermalRadiation>> extra;
};

void ThermalRadiation::DestroyImplicitGroupWorkers() {
  delete implicit_group_workers_;
  implicit_group_workers_ = nullptr;
}

void ThermalRadiation::SolveImplicitTransportBatched(const Real dt,
    DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
    const DvceArray5D<Real> &temperature, MeshBoundaryValuesCC *pbval) {
  RadiationSummaryScope summary(solver_summary_, pmy_pack_->pmesh->ncycle, dt,
      global_variable::my_rank, "transport");
  const auto start = std::chrono::steady_clock::now();
  int batch_size = std::min(implicit_group_batch_size_, ngroups);
  const int requested_size = implicit_group_batch_size_;
  const auto report = [&]() {
    if (implicit_report_ && global_variable::my_rank == 0) {
      std::cout << "# implicit thermal radiation: groups=" << ngroups
                << " max_iterations=" << implicit_iterations_last_solve
                << " max_residual_replacements=" << implicit_residual_replacements_last_solve
                << " max_relative_residual=" << implicit_residual_last_solve
                << " max_componentwise_backward_error=" << implicit_backward_error_last_solve
                << std::endl;
    }
  };
#ifdef KOKKOS_ENABLE_CUDA
  // Account for every future Arnoldi basis before allocating extra workers.
  // Capacity can change after regridding. Decide collectively and keep the
  // smaller batch or scalar path when a complete workspace will not fit.
  std::size_t free_bytes = 0, total_bytes = 0, needed = 0, reusable_bytes = 0;
  const auto &idx = pmy_pack_->pmesh->mb_indcs;
  const std::size_t cells = static_cast<std::size_t>(pmy_pack_->nmb_thispack)*
      idx.nx1*idx.nx2*idx.nx3;
  bool mixed_levels = false;
  const auto *mesh = pmy_pack_->pmesh;
  for (int gid = 1; gid < mesh->nmb_total; ++gid) {
    mixed_levels |= mesh->lloc_eachmb[gid].level != mesh->lloc_eachmb[0].level;
  }
  const std::size_t vectors = std::min(implicit_gmres_restart_max_, implicit_max_iterations_)+1+
      // Native can switch to the fixed inverse metric between restarts. Reserve
      // its cached B*v basis even when the first cycle uses the volume metric.
      (implicit_amr_inner_iterations_ > 0 || implicit_preconditioner_mode_ == 5 ?
          std::min(implicit_gmres_restart_max_, implicit_max_iterations_) : 0)+
      // Short restart-history vectors are local to each active native solve.
      (implicit_preconditioner_mode_ == 5 ?
          std::max(2, implicit_amr_inner_iterations_ == 0 &&
              !implicit_gmres_preconditioner_metric_ ?
              std::min(3, std::min(implicit_gmres_restart_max_, implicit_max_iterations_)/4) : 0) : 0);
  // Native low-order PCG can enter component-relative GMRES at its roundoff
  // plateau. Reserve both Krylov bases and its two temporary metric vectors
  // even for order-one transport. They do not overlap the restart history.
  const bool may_use_gmres = implicit_amr_order_ == 2 ||
      (implicit_preconditioner_mode_ == 5 && mesh->multilevel);
  const std::size_t basis_bytes = may_use_gmres ? cells*vectors*sizeof(Real) : 0;
  const auto growth = [basis_bytes](const ThermalRadiation &w) {
    const std::size_t present = (w.implicit_gmres_basis_.size()+
                                w.implicit_gmres_corrections_.size())*sizeof(Real);
    return basis_bytes > present ? basis_bytes-present : 0;
  };
  std::size_t scratch_bytes = implicit_multilevel_vector_.size()*sizeof(Real);
  for (const auto *v : {&implicit_old_, &implicit_solution_, &implicit_coefficient_,
         &implicit_residual_, &implicit_direction_, &implicit_preconditioned_,
         &implicit_operator_, &implicit_coarse_scratch_, &implicit_amr_diagonal_,
         &implicit_amr_gradient_, &implicit_amr_inner_z_,
         &implicit_amr_conductance_.x1f, &implicit_amr_conductance_.x2f,
         &implicit_amr_conductance_.x3f, &implicit_amr_flux_.x1f,
         &implicit_amr_flux_.x2f, &implicit_amr_flux_.x3f}) scratch_bytes += v->size()*sizeof(Real);
  while (batch_size > 1) {
    if (implicit_group_workers_ != nullptr &&
        implicit_group_workers_->extra.size() > static_cast<std::size_t>(batch_size-1)) {
      implicit_group_workers_->extra.resize(batch_size-1);
    }
    const bool native = implicit_preconditioner_mode_ == 5;
    const auto reusable = [native](const ThermalRadiation &w) {
      return native ? (w.implicit_native_solver_ ? w.implicit_native_solver_->bytes() : 0) :
          ImplicitAMRReusableBytes(w.implicit_amr_aggregate_state_);
    };
    needed = batch_size*(native ? implicit_native_max_bytes_ : implicit_amr_amg_max_bytes_)+
        growth(*this)+256ULL*1024*1024;
    reusable_bytes = reusable(*this);
    std::size_t cell_workspace = ImplicitAMRCellWorkspaceBytes(implicit_amr_aggregate_state_);
    for (int lane = 1; lane < batch_size; ++lane) {
      if (implicit_group_workers_ != nullptr &&
          static_cast<std::size_t>(lane-1) < implicit_group_workers_->extra.size()) {
        needed += growth(*implicit_group_workers_->extra[lane-1]);
        reusable_bytes += reusable(*implicit_group_workers_->extra[lane-1]);
        cell_workspace += ImplicitAMRCellWorkspaceBytes(
            implicit_group_workers_->extra[lane-1]->implicit_amr_aggregate_state_);
      } else {
        needed += basis_bytes+scratch_bytes;
      }
    }
    if (implicit_preconditioner_mode_ == 4 || native) {
      const int dimensions = 1+(idx.nx2 > 1)+(idx.nx3 > 1);
      const std::size_t subfaces = 1 << (dimensions-1);
      const std::size_t block_faces = 2*static_cast<std::size_t>(pmy_pack_->nmb_thispack)*
          (idx.nx2*idx.nx3+(idx.nx2 > 1 ? idx.nx1*idx.nx3 : 0)+
           (idx.nx3 > 1 ? idx.nx1*idx.nx2 : 0));
      const std::size_t entries = cells*(1+2*dimensions)+
          (mixed_levels ? (subfaces-1)*block_faces : 0);
      // Per lane: cell numerical values/tags + packed numerical values (17 B
      // per entry), four double vectors and three row-offset arrays (44 B
      // per row). These application arrays are outside the HYPRE budget.
      // AMR adds fine-face descriptors/values, local and remote scatter slots,
      // and received values. Bound every block boundary as a coarse/fine face;
      // actual storage only covers interfaces present in the current topology.
      const std::size_t interfaces = mixed_levels ? (32*subfaces+20)*block_faces : 0;
      const std::size_t target = batch_size*(17*entries+44*cells+interfaces+12);
      if (target > cell_workspace) needed += target-cell_workspace;
    }
    // Setup replaces the previous hierarchy. Its accounted bytes are reusable,
    // just as in the adapter's own per-setup memory check; counting only free
    // bytes would needlessly shrink a persistent cell-AMG batch on later steps.
    TrimImplicitAMRReuseCache(implicit_amr_aggregate_state_, needed, reusable_bytes);
    int fits = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        free_bytes+reusable_bytes >= needed;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &fits, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    if (fits) break;
    batch_size = batch_size > 2 ? 2 : 1;
  }
  if (batch_size == 1) {
    DestroyImplicitGroupWorkers();
    if (implicit_report_ || solver_timers_.enabled) {
      std::cout << "# radiation group batch: rank=" << global_variable::my_rank
                << " cycle=" << pmy_pack_->pmesh->ncycle
                << " size=1 reason=workspace_budget required_bytes=" << needed
                << " free_bytes=" << free_bytes
                << " reusable_amg_bytes=" << reusable_bytes << std::endl;
    }
    implicit_batch_worker_ = true;
    SolveImplicitTransport(dt, cons, prim, temperature, pbval);
    implicit_batch_worker_ = false;
    report();
    solver_timers_.Report(pmy_pack_->pmesh->ncycle, dt, global_variable::my_rank, "transport");
    return;
  }
#endif
  if (implicit_group_workers_ == nullptr) {
    implicit_group_workers_ = new ImplicitRadiationGroupWorkers;
  }
  while (implicit_group_workers_->extra.size() < static_cast<std::size_t>(batch_size-1)) {
    auto *w = new ThermalRadiation(*this);
    implicit_group_workers_->extra.emplace_back(w);
    w->implicit_group_workers_ = nullptr;
    w->borrowed_opacity_ = true;
    w->implicit_amr_aggregate_state_ = nullptr;
    w->implicit_native_solver_ = nullptr;
    w->implicit_halo_ = nullptr;
    w->implicit_bvals_ = nullptr;
#if MPI_PARALLEL_ENABLED
    w->implicit_multilevel_comm_ = MPI_COMM_NULL;
#endif
    w->solver_timers_ = RadiationSolverTimers{};
    w->solver_timers_.enabled = solver_timers_.enabled;
    // Fixed-capacity mesh storage has the same shape as the scalar worker.
    // Detach every mutable transport view; immutable opacity tables and material
    // configuration remain shared. Aggregate topology is rebuilt by each worker.
    const auto detach = [](DvceArray5D<Real> &v) {
      if (v.size() == 0) { v = {}; return; }
      v = DvceArray5D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "group scratch"),
          v.extent(0), v.extent(1), v.extent(2), v.extent(3), v.extent(4));
    };
    for (auto *v : {&w->implicit_old_, &w->implicit_solution_, &w->implicit_coefficient_,
         &w->implicit_residual_, &w->implicit_direction_, &w->implicit_preconditioned_,
         &w->implicit_operator_, &w->implicit_coarse_scratch_, &w->implicit_amr_diagonal_,
         &w->implicit_amr_gradient_, &w->implicit_amr_inner_z_,
         &w->implicit_amr_conductance_.x1f, &w->implicit_amr_conductance_.x2f,
         &w->implicit_amr_conductance_.x3f, &w->implicit_amr_flux_.x1f,
         &w->implicit_amr_flux_.x2f, &w->implicit_amr_flux_.x3f}) detach(*v);
    w->implicit_gmres_basis_ = {};
    w->implicit_gmres_corrections_ = {};
    w->implicit_amr_block_matrix_ = {};
    // AMR AMG also uses the two local aggregate-diagonal corrections.
    w->implicit_multilevel_vector_ = DvceArray2D<Real>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "group aggregate diagonals"),
        implicit_multilevel_vector_.extent(0), implicit_multilevel_vector_.extent(1));
    // The remaining uniform-grid block-coarse data is unused by AMR AMG.
    w->implicit_multilevel_conductance_ = {};
    w->implicit_multilevel_send_faces_ = {};
    w->implicit_multilevel_recv_faces_ = {};
    w->implicit_coarse_faces_ = {};
    w->implicit_coarse_vector_ = {};
    w->implicit_coarse_neighbor_gid_device_ = {};
    w->implicit_coarse_neighbor_rank_device_ = {};
    w->implicit_multilevel_block_parity_ = {};
  }
  RadiationGroupBatch batch(solver_summary_);
  RadiationSolverTimers total_timers;
  total_timers.enabled = solver_timers_.enabled;
  int max_iterations = 0, max_replacements = 0, fallbacks = 0, scalar_retries = 0;
  Real max_residual = 0.0, max_backward = 0.0, boundary_power = 0.0;
  auto accumulate = [&](ThermalRadiation &w) {
    max_iterations = std::max(max_iterations, w.implicit_iterations_last_solve);
    max_replacements = std::max(max_replacements, w.implicit_residual_replacements_last_solve);
    max_residual = std::max(max_residual, w.implicit_residual_last_solve);
    max_backward = std::max(max_backward, w.implicit_backward_error_last_solve);
    fallbacks += w.implicit_amr_fallbacks_last_solve;
    boundary_power += w.implicit_boundary_power;
  };
  try {
    for (int first = 0; first < ngroups; first += batch_size) {
      std::vector<ThermalRadiation *> workers{this};
      const int active = std::min(batch_size, ngroups-first);
      for (int lane = 1; lane < batch_size; ++lane) {
        auto *w = implicit_group_workers_->extra[lane-1].get();
        if (lane < active) {
          workers.push_back(w);
        } else {
          // A smaller tail rebuilds its hierarchy. Release retired lanes'
          // references so the previous shared operator can be freed.
          ReleaseImplicitAMRBatch(w->implicit_amr_aggregate_state_);
        }
      }
      std::vector<std::function<void()>> work;
      for (std::size_t lane = 0; lane < workers.size(); ++lane) {
        auto *w = workers[lane];
        w->implicit_batch_worker_ = true;
        w->implicit_group_begin_ = first+static_cast<int>(lane);
        w->implicit_group_end_ = w->implicit_group_begin_+1;
        w->implicit_batch_retry_ = false;
        w->implicit_batch_retry_low_order_ = false;
        work.emplace_back([&, w]() {
          w->SolveImplicitTransport(dt, cons, prim, temperature, pbval);
        });
      }
      batch.Run(work);
      if (std::any_of(workers.begin(), workers.end(),
          [](const ThermalRadiation *w) { return w->implicit_batch_retry_; })) {
        // All lanes have finished. Release every reference to their shared
        // hierarchy before building any scalar retry hierarchy, so failed
        // groups cannot accumulate extra AMG allocations beside the old batch.
        for (auto *w : workers) ReleaseImplicitAMRBatch(w->implicit_amr_aggregate_state_);
      }
      for (auto *w : workers) {
        total_timers.Accumulate(w->solver_timers_);
        if (w->implicit_batch_retry_) {
          // No state was committed by this lane. A converged high-order
          // candidate resumes directly at its monotone retry; a failed reused
          // hierarchy still retries the original equation with fresh setup.
          ++scalar_retries;
          w->implicit_batch_retry_ = false;
          w->SolveImplicitTransport(dt, cons, prim, temperature, pbval);
          total_timers.Accumulate(w->solver_timers_);
        }
        accumulate(*w);
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "### FATAL radiation group batch error: " << error.what() << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::abort();
  }
  implicit_batch_worker_ = false;
  implicit_group_begin_ = 0;
  implicit_group_end_ = -1;
  implicit_iterations_last_solve = max_iterations;
  implicit_residual_replacements_last_solve = max_replacements;
  implicit_residual_last_solve = max_residual;
  implicit_backward_error_last_solve = max_backward;
  implicit_amr_fallbacks_last_solve = fallbacks;
  implicit_boundary_power = boundary_power;
  report();
  const double wall = std::chrono::duration<double>(
      std::chrono::steady_clock::now()-start).count();
  total_timers.SetTransportWall(wall);
  solver_timers_ = total_timers;
  solver_timers_.Report(pmy_pack_->pmesh->ncycle, dt, global_variable::my_rank, "transport");
  if (solver_summary_) {
    const double total = summary.Stop();
    const auto &c = batch.counts();
    const double dispatched = c.seconds[0]+c.seconds[1]+c.seconds[2]+c.seconds[3];
    std::ostringstream line;
    line << std::scientific << std::setprecision(9)
         << "# radiation batch timing: rank=" << global_variable::my_rank
         << " cycle=" << pmy_pack_->pmesh->ncycle << " size=" << batch_size
         << " transport_seconds=" << total << " amg_setup_seconds=" << c.seconds[0]
         << " amg_apply_seconds=" << c.seconds[1] << " halo_seconds=" << c.seconds[2]
         << " reduction_seconds=" << c.seconds[3]
         << " other_seconds=" << total-dispatched
         << " exclusive_dispatch=1 setup_gpu_fences=0 setup_existing_gpu_completion=1\n";
    std::cout << line.str();
  }
  if (implicit_report_ || solver_timers_.enabled) {
    const auto &c = batch.counts();
    std::cout << "# radiation group batch: rank=" << global_variable::my_rank
              << " cycle=" << pmy_pack_->pmesh->ncycle << " size=" << batch_size
              << " requested_size=" << requested_size
              << " setup_requests=" << c.requests[0] << " setup_batches=" << c.batches[0]
              << " setup_paired=" << c.paired[0]
              << " amg_requests=" << c.requests[1] << " amg_batches=" << c.batches[1]
              << " amg_paired=" << c.paired[1]
              << " halo_requests=" << c.requests[2] << " halo_batches=" << c.batches[2]
              << " halo_paired=" << c.paired[2]
              << " reduction_requests=" << c.requests[3]
              << " reduction_batches=" << c.batches[3]
              << " scalar_retries=" << scalar_retries
#ifdef KOKKOS_ENABLE_CUDA
              << " required_bytes=" << needed << " free_bytes=" << free_bytes
              << " reusable_amg_bytes=" << reusable_bytes
#endif
              << " phase_times_include_lane_waits=1" << std::endl;
  }
}
}  // namespace two_temperature
