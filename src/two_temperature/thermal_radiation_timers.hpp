#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_TIMERS_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_TIMERS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_timers.hpp
//! \brief Opt-in, rank-local wall timers for implicit radiation diagnostics.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "athena.hpp"

namespace two_temperature {

enum class RadiationTimer {
  Transport, Group, Coefficient, OperatorSetup, PreconditionerSetup, Exchange,
  Matvec, Dot, ResidualCheck, GMRES, InnerCG, Preconditioner, AggregateBuild,
  AggregateConductanceD2H, AggregateGraph, AggregateBuildMPI, CholmodAnalyze,
  CholmodFactor, AggregateApply, AggregateRestrict, AggregateRhsD2H,
  AggregateApplyMPI, AggregateRhsScale, CholmodSolve, AggregateSolutionScale,
  AggregateCorrectionH2D, AggregateProlong, AggregateTopology, AggregateNumeric,
  AggregateNumericD2H, AggregateFactorExport, AggregateDeviceFactor, AggregateDeviceSolve,
  Source, Count
};

enum class RadiationCounter { InnerIterations, OuterIterations, CGIterations,
                              InnerCapHits, Count };

class RadiationTimerScope;

// A production summary fences only at whole-solve boundaries. Fine-grained
// fenced profiling remains separately opt-in through solver_timers.
class RadiationSummaryScope {
 public:
  RadiationSummaryScope(bool enabled, int cycle, Real dt, int rank, const char *phase)
      : enabled_(enabled), cycle_(cycle), rank_(rank), dt_(dt), phase_(phase) {
    if (enabled_) {
      Kokkos::fence("radiation summary begin");
      start_ = Clock::now();
    }
  }
  ~RadiationSummaryScope() {
    if (!enabled_) return;
    const double wall = Stop();
    std::ostringstream line;
    line << std::scientific << std::setprecision(9)
         << "# radiation performance: rank=" << rank_ << " cycle=" << cycle_
         << " dt=" << dt_ << " phase=" << phase_ << " wall_seconds=" << wall
         << " gpu_boundary_fences=1\n";
    std::cout << line.str();
  }
  double Stop() {
    if (enabled_ && !stopped_) {
      Kokkos::fence("radiation summary end");
      wall_ = std::chrono::duration<double>(Clock::now()-start_).count();
      stopped_ = true;
    }
    return wall_;
  }
  RadiationSummaryScope(const RadiationSummaryScope &) = delete;
  RadiationSummaryScope &operator=(const RadiationSummaryScope &) = delete;

 private:
  using Clock = std::chrono::steady_clock;
  bool enabled_, stopped_ = false;
  int cycle_, rank_;
  Real dt_;
  const char *phase_;
  Clock::time_point start_;
  double wall_ = 0.0;
};

class RadiationSolverTimers {
 public:
  bool enabled = false;

  void Reset() {
    if (!enabled) return;
    for (auto &bucket : buckets_) bucket = {};
    counters_.fill(0);
    maximum_inner_iterations_ = 0;
    groups_.clear();
  }

  void Add(RadiationCounter counter, std::uint64_t amount = 1) {
    if (enabled) counters_[static_cast<int>(counter)] += amount;
  }

  void ObserveInnerIterations(int iterations) {
    if (enabled && iterations > maximum_inner_iterations_) {
      maximum_inner_iterations_ = iterations;
    }
  }

  // Batched groups retain separate scope trees. Their inclusive phase durations
  // include waits for other lanes; only the enclosing transport wall is elapsed time.
  void Accumulate(const RadiationSolverTimers &other) {
    if (!enabled) return;
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
      buckets_[i].wall += other.buckets_[i].wall;
      buckets_[i].self += other.buckets_[i].self;
      buckets_[i].calls += other.buckets_[i].calls;
    }
    for (std::size_t i = 0; i < counters_.size(); ++i) counters_[i] += other.counters_[i];
    maximum_inner_iterations_ = std::max(maximum_inner_iterations_,
                                        other.maximum_inner_iterations_);
    groups_.insert(groups_.end(), other.groups_.begin(), other.groups_.end());
  }
  void SetTransportWall(double seconds) {
    buckets_[static_cast<int>(RadiationTimer::Transport)] = {seconds, 0.0, 1};
  }

  // Each rank reports its own additive accounting; no extra MPI reductions are added.
  void Report(int cycle, Real dt, int rank, const char *solve) const {
    if (!enabled) return;
    static constexpr const char *names[] = {
      "transport", "group", "coefficient", "operator_setup", "preconditioner_setup",
      "exchange", "matvec", "dot", "residual_check", "gmres", "inner_cg",
      "preconditioner", "aggregate_build", "aggregate_conductance_d2h", "aggregate_graph",
      "aggregate_build_mpi", "cholmod_analyze", "cholmod_factor", "aggregate_apply",
      "aggregate_restrict", "aggregate_rhs_d2h", "aggregate_apply_mpi", "aggregate_rhs_scale",
      "cholmod_solve", "aggregate_solution_scale", "aggregate_correction_h2d",
      "aggregate_prolong", "aggregate_topology", "aggregate_numeric",
      "aggregate_numeric_d2h", "aggregate_factor_export", "aggregate_device_factor",
      "aggregate_device_solve", "source"
    };
    static_assert(sizeof(names)/sizeof(names[0]) == static_cast<int>(RadiationTimer::Count),
                  "Every radiation timer needs a report label");
    for (int i = 0; i < static_cast<int>(RadiationTimer::Count); ++i) {
      const auto &bucket = buckets_[i];
      if (bucket.calls == 0) continue;
      std::ostringstream line;
      line << std::scientific << std::setprecision(9)
           << "# radiation solver timer: rank=" << rank << " cycle=" << cycle
           << " dt=" << dt << " solve=" << solve << " phase=" << names[i]
           << " calls=" << bucket.calls << " wall_seconds=" << bucket.wall
           << " self_seconds=" << bucket.self << " gpu_phase_fences=1\n";
      std::cout << line.str();
    }
    for (const auto &group : groups_) {
      std::ostringstream line;
      line << std::scientific << std::setprecision(9)
           << "# radiation solver group timer: rank=" << rank << " cycle=" << cycle
           << " group=" << group.group << " attempt=" << group.attempt
           << " order=" << group.order << " wall_seconds=" << group.wall
           << " inner_iterations=" << group.counters[0]
           << " outer_iterations=" << group.counters[1]
           << " cg_iterations=" << group.counters[2]
           << " inner_cap_hits=" << group.counters[3] << '\n';
      std::cout << line.str();
    }
    std::ostringstream line;
    line << "# radiation solver counters: rank=" << rank << " cycle=" << cycle
         << " solve=" << solve << " inner_iterations=" << counters_[0]
         << " outer_iterations=" << counters_[1] << " cg_iterations=" << counters_[2]
         << " inner_cap_hits=" << counters_[3]
         << " maximum_inner_iterations=" << maximum_inner_iterations_ << '\n';
    std::cout << line.str() << std::flush;
  }

 private:
  friend class RadiationTimerScope;
  struct Bucket {
    double wall = 0.0, self = 0.0;
    std::uint64_t calls = 0;
  };
  using Counters = std::array<std::uint64_t, static_cast<int>(RadiationCounter::Count)>;
  struct Group {
    int group, attempt, order;
    double wall;
    Counters counters;
  };
  std::array<Bucket, static_cast<int>(RadiationTimer::Count)> buckets_{};
  Counters counters_{};
  int maximum_inner_iterations_ = 0;
  std::vector<Group> groups_;
  RadiationTimerScope *active_ = nullptr;
};

// GPU phases fence before starting and before stopping. The entry fence drains any
// preceding asynchronous parent work and is charged to the parent's self time.
// Pure CPU phases pass gpu_phase=false. Disabled scopes read no clocks and add no fences.
// Inclusive wall times overlap along the nesting tree; only self times are additive.
class RadiationTimerScope {
 public:
  RadiationTimerScope(RadiationSolverTimers &timers, RadiationTimer phase,
                      bool gpu_phase = true, int group = -1, int attempt = 0, int order = 0)
      : timers_(timers.enabled ? &timers : nullptr), phase_(phase), gpu_phase_(gpu_phase),
        group_(group), attempt_(attempt), order_(order) {
    if (timers_ == nullptr) return;
    if (gpu_phase_) Kokkos::fence();
    parent_ = timers_->active_;
    timers_->active_ = this;
    if (group_ >= 0) counters_before_ = timers_->counters_;
    start_ = Clock::now();
  }
  ~RadiationTimerScope() { Stop(); }
  RadiationTimerScope(const RadiationTimerScope &) = delete;
  RadiationTimerScope &operator=(const RadiationTimerScope &) = delete;

  void Stop() {
    if (timers_ == nullptr) return;
    if (gpu_phase_) Kokkos::fence();
    const double wall = std::chrono::duration<double>(Clock::now()-start_).count();
    auto &bucket = timers_->buckets_[static_cast<int>(phase_)];
    bucket.wall += wall;
    bucket.self += wall-children_;
    ++bucket.calls;
    timers_->active_ = parent_;
    if (parent_ != nullptr) parent_->children_ += wall;
    if (group_ >= 0) {
      auto counters = timers_->counters_;
      for (unsigned int i = 0; i < counters.size(); ++i) counters[i] -= counters_before_[i];
      timers_->groups_.push_back({group_, attempt_, order_, wall, counters});
    }
    timers_ = nullptr;
  }

 private:
  using Clock = std::chrono::steady_clock;
  RadiationSolverTimers *timers_;
  RadiationTimer phase_;
  bool gpu_phase_;
  int group_, attempt_, order_;
  Clock::time_point start_;
  RadiationTimerScope *parent_ = nullptr;
  double children_ = 0.0;
  RadiationSolverTimers::Counters counters_before_{};
};

}  // namespace two_temperature
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_TIMERS_HPP_
