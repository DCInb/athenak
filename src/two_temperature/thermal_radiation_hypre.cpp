//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Device-resident BoomerAMG with optional HYPRE 2.33 transfer reuse.

#include "two_temperature/thermal_radiation_hypre.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../athena.hpp"

#ifdef ATHENA_HYPRE_ENABLED
#include <HYPRE.h>
#ifdef ATHENA_HYPRE_REUSE_ENABLED
#include <_hypre_parcsr_ls.h>
#if HYPRE_RELEASE_NUMBER != 23300
#error "This isolated transfer-reuse adapter requires HYPRE 2.33.0"
#endif
extern "C" void athena_hypre_reuse_transfers(hypre_ParCSRMatrix **,
                                           hypre_IntArray **, int);
extern "C" HYPRE_Int athena_hypre_BoomerAMGSetup(void *, hypre_ParCSRMatrix *,
                                               hypre_ParVector *, hypre_ParVector *);
#endif

#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_utilities.h>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#if !defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_UNIFIED_MEMORY)
#error "Athena CUDA AMG requires CUDA hypre without unified memory"
#endif
#elif defined(HYPRE_USING_GPU)
#error "Athena host AMG requires host hypre"
#endif

static_assert(std::is_same<HYPRE_Real, double>::value,
              "AMG requires double precision hypre");
static_assert(std::is_same<HYPRE_Complex, double>::value,
              "AMG requires real-valued hypre");
static_assert(std::is_same<HYPRE_Int, int>::value,
              "AMG requires 32-bit local hypre indices");

namespace {
int hypre_users = 0;
bool hypre_owned = false;
// Process-local generation tokens also distinguish a newly allocated State
// that happens to reuse a retired lane's host address.
std::uint64_t next_graph_revision = 0;
constexpr std::size_t kMemoryMargin = 128ULL * 1024 * 1024;

}  // namespace
#endif

struct ImplicitAMGHypre::BatchWorkspace {
  int lanes = 0;
  std::vector<const State *> owners;
  std::vector<std::uint64_t> revisions;
  std::vector<DvceArray1D<int>> source_offsets;
  DvceArray1D<int> destination_offsets;
  DvceArray1D<double> values, rhs, solution;
#ifdef ATHENA_HYPRE_ENABLED
  Kokkos::View<HYPRE_Int *, DevMemSpace> counts;
  Kokkos::View<HYPRE_Int *, HostMemSpace> host_counts;
  Kokkos::View<HYPRE_BigInt *, DevMemSpace> rows, cols;
#endif
};

struct ImplicitAMGHypre::BatchData {
  ImplicitAMGHypre solver;
  int lanes = 0, rows = 0;
  std::vector<int> groups, baseline_iterations;
  DvceArray1D<double> reference_row_norm;

  int built_cycle = -1, last_cycle = -1;
  bool refresh_requested = false;

  std::vector<const State *> owners;
  std::vector<std::uint64_t> revisions;
  std::vector<int> offsets, columns;
  std::vector<DvceArray1D<int>> source_offsets;
  DvceArray1D<int> destination_offsets;
  bool configured = false;
  std::size_t max_bytes = 0, reserve_bytes = 0;
  DvceArray1D<double> values, rhs, solution;
};

struct ImplicitAMGHypre::BatchRequest {
  ImplicitAMGHypre *solver;
  const double *input;
  double *output = nullptr;
  bool success = false;
};

struct ImplicitAMGHypre::State {
  std::string error;
  std::size_t max_bytes = 0, reserve_bytes = 0, device_bytes = 0;
  int first_row = 0, global_rows = 0, nrows = 0;
  std::uint64_t graph_revision = 0;
  std::vector<int> row_offsets, columns;
  std::shared_ptr<BatchData> batch;
  int batch_lane = 0;
  int reuse_group = -1, reuse_cycle = -1;
  int reuse_steps = 0, reuse_slots = 5;
  double reuse_drift = 64.0;
  bool reuse_enabled = false, hierarchy_reused = false;
  // Only the first lane owns the cache. Owners inside BatchData are non-owning.
  std::map<std::vector<int>, std::shared_ptr<BatchData>> reuse_cache;
  std::shared_ptr<BatchWorkspace> batch_workspace;

#ifdef ATHENA_HYPRE_ENABLED
  bool initialized = false, ready = false;
  bool account_first_apply = false;
  int ranks = 1;
  HYPRE_IJMatrix matrix = nullptr;
  HYPRE_IJVector rhs = nullptr, solution = nullptr;
  HYPRE_ParCSRMatrix par_matrix = nullptr;
  HYPRE_ParVector par_rhs = nullptr, par_solution = nullptr;
  HYPRE_Solver solver = nullptr;
  bool use_saved_transfers = false;
#ifdef ATHENA_HYPRE_REUSE_ENABLED
  std::vector<hypre_ParCSRMatrix *> saved_p;
  std::vector<hypre_IntArray *> saved_cf;
  void DestroyTransfers() {
    const bool owned = std::any_of(saved_p.begin(), saved_p.end(),
                                  [](const auto *p) { return p != nullptr; });
    std::size_t before = 0, after = 0;
    if (owned) { Synchronize(); FreeBytes(before); }
    for (auto *p : saved_p) if (p) hypre_ParCSRMatrixDestroy(p);
    for (auto *cf : saved_cf) if (cf) hypre_IntArrayDestroy(cf);
    saved_p.clear(); saved_cf.clear();
    if (owned) {
      Synchronize(); FreeBytes(after);
      const std::size_t released = after > before ? after-before : 0;
      device_bytes = device_bytes > released ? device_bytes-released : 0;
    }
  }
  // Called between completed cohorts only. Keep interpolation (including its
  // communication plan and cached transpose) but release numerical A levels,
  // smoothers, fine IJ storage and solve workspaces to bound GPU memory.
  void KeepTransfersOnly() {
    if (!solver || !ready) return;
    auto *amg = reinterpret_cast<hypre_ParAMGData *>(solver);
    const int levels = hypre_ParAMGDataNumLevels(amg);
    if (levels < 2) return;
    DestroyTransfers();
    saved_p.resize(levels-1, nullptr);
    saved_cf.resize(levels-1, nullptr);
    for (int level = 0; level < levels-1; ++level) {
      saved_p[level] = hypre_ParAMGDataPArray(amg)[level];
      saved_cf[level] = hypre_ParAMGDataCFMarkerArray(amg)[level];
      hypre_ParAMGDataPArray(amg)[level] = nullptr;
      hypre_ParAMGDataCFMarkerArray(amg)[level] = nullptr;
    }
    std::size_t before = 0, after = 0, previous = device_bytes;
    Synchronize();
    FreeBytes(before);
    auto saved_counts = counts;
    auto saved_host_counts = host_counts;
    auto saved_rows = rows, saved_cols = cols;
    DestroyObjects();
    counts = saved_counts; host_counts = saved_host_counts;
    rows = saved_rows; cols = saved_cols;
    Synchronize();
    FreeBytes(after);
    const std::size_t released = after > before ? after-before : 0;
    device_bytes = previous > released ? previous-released : 0;
  }
  bool HasSavedTransfers() const { return !saved_p.empty(); }
#else
  void DestroyTransfers() {}
  void KeepTransfersOnly() {}
  bool HasSavedTransfers() const { return false; }
#endif
  Kokkos::View<HYPRE_Int *, DevMemSpace> counts;
  Kokkos::View<HYPRE_Int *, HostMemSpace> host_counts;
  Kokkos::View<HYPRE_BigInt *, DevMemSpace> rows, cols;

  // Fatal runtime errors may occur while peers are inside hypre collectives.
  // Abort the communicator instead of letting one rank return and strand them.
  // The successful Apply path adds no MPI collectives of its own.
  void AbortDistributed() {
#if MPI_PARALLEL_ENABLED
    if (ranks > 1) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::cerr << "### FATAL AMG ERROR on rank " << rank << ": " << error << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      std::abort();
    }
#endif
  }

  // Configuration and memory checks happen before entering hypre. Reject them
  // collectively so callers on every rank receive the same success/failure state.
  bool Agree(bool local_ok) {
    int accepted = local_ok ? 1 : 0;
#if MPI_PARALLEL_ENABLED
    if (ranks > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &accepted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
#endif
    if (!accepted) {
      if (local_ok) error = "AMG configuration or setup rejected on another MPI rank";
      ready = false;
    }
    return accepted != 0;
  }

  bool Check(HYPRE_Int status, const char *operation) {
    if (status == 0) return true;
    char message[256] = {};
    HYPRE_DescribeError(status, message);
    error = std::string(operation) + ": " + message;
    HYPRE_ClearAllErrors();
    ready = false;
    AbortDistributed();
    return false;
  }

  bool Synchronize() {
#ifdef KOKKOS_ENABLE_CUDA
    // hypre 2.33 has no public stream setter. Fence the producer and consumer
    // explicitly; both packages may own independent CUDA streams.
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      error = std::string("hypre CUDA synchronization: ") + cudaGetErrorString(status);
      ready = false;
      AbortDistributed();
      return false;
    }
#endif
    return true;
  }

  bool FreeBytes(std::size_t &free_bytes) {
#ifdef KOKKOS_ENABLE_CUDA
    std::size_t total = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total);
    if (status != cudaSuccess) {
      error = std::string("hypre CUDA memory query: ") + cudaGetErrorString(status);
      AbortDistributed();
      return false;
    }
#else
    free_bytes = std::numeric_limits<std::size_t>::max();
#endif
    return true;
  }

  bool MemoryCheck() {
    // hypre does not expose a bounded allocator through its public API. Refuse
    // setup unless the entire configured budget plus future solver storage and
    // a safety margin fit. Also reject obviously insufficient configured caps.
    const std::size_t matrix_bytes = columns.size() * (sizeof(double)
        + sizeof(HYPRE_BigInt)) + row_offsets.size() * sizeof(HYPRE_Int);
    const std::size_t estimate = 10 * matrix_bytes
        + static_cast<std::size_t>(nrows) * 128;
    if (max_bytes == 0 || estimate > max_bytes) {
      error = "AMG hierarchy estimate exceeds implicit_amr_amg_max_mib";
      return false;
    }
#ifdef KOKKOS_ENABLE_CUDA
    std::size_t free_bytes = 0;
    if (!FreeBytes(free_bytes)) return false;
    if (reserve_bytes > std::numeric_limits<std::size_t>::max() - kMemoryMargin
        || max_bytes > std::numeric_limits<std::size_t>::max()
            - reserve_bytes - kMemoryMargin
        || free_bytes + device_bytes < max_bytes + reserve_bytes + kMemoryMargin) {
      error = "insufficient free GPU memory for AMG budget, Krylov reserve and margin"
          " (free_bytes=" + std::to_string(free_bytes)
          + " reusable_amg_bytes=" + std::to_string(device_bytes)
          + " amg_budget_bytes=" + std::to_string(max_bytes)
          + " krylov_reserve_bytes=" + std::to_string(reserve_bytes)
          + " margin_bytes=" + std::to_string(kMemoryMargin) + ")";
      return false;
    }
#endif
    return true;
  }

  void DestroyObjects() {
    batch.reset();
    ready = false;
    if (solver != nullptr) HYPRE_BoomerAMGDestroy(solver);
    if (matrix != nullptr) HYPRE_IJMatrixDestroy(matrix);
    if (rhs != nullptr) HYPRE_IJVectorDestroy(rhs);
    if (solution != nullptr) HYPRE_IJVectorDestroy(solution);
    solver = nullptr;
    matrix = nullptr;
    rhs = solution = nullptr;
    par_matrix = nullptr;
    par_rhs = par_solution = nullptr;
    counts = {};
    host_counts = {};
    rows = {};
    cols = {};
    device_bytes = 0;
  }

  ~State() {
    reuse_cache.clear();
    batch_workspace.reset();
    DestroyObjects();
    DestroyTransfers();
    if (initialized && --hypre_users == 0 && hypre_owned) {
      HYPRE_Finalize();
      hypre_owned = false;
    }
  }
#endif
};

ImplicitAMGHypre::ImplicitAMGHypre() : state_(new State) {}
ImplicitAMGHypre::~ImplicitAMGHypre() = default;

bool ImplicitAMGHypre::Available() {
#ifdef ATHENA_HYPRE_ENABLED
  return true;
#else
  return false;
#endif
}

bool ImplicitAMGHypre::ReuseAvailable() {
#if defined(ATHENA_HYPRE_ENABLED) && defined(ATHENA_HYPRE_REUSE_ENABLED)
  return true;
#else
  return false;
#endif
}

const std::string &ImplicitAMGHypre::error() const { return state_->error; }
std::size_t ImplicitAMGHypre::device_bytes() const {
#ifdef ATHENA_HYPRE_ENABLED
  // The shared solver can allocate persistent buffers on its first application.
  // Query its current accounting instead of the lane's setup-time snapshot.
  if (state_->batch) {
    const auto &data = *state_->batch;
    const int lane = state_->batch_lane;
    // A converged group can retain a reference after a later cohort reuses the
    // shared owner with fewer lanes. Only its current owners earn memory credit.
    if (lane >= data.lanes || data.owners[lane] != state_.get()) return 0;
    return data.solver.device_bytes()/data.lanes;
  }
#endif
  return state_->device_bytes;
}

std::size_t ImplicitAMGHypre::batch_workspace_bytes() const {
  const auto &data = state_->batch;
  if (!data) return 0;
  const int lane = state_->batch_lane;
  if (lane >= data->lanes || data->owners[lane] != state_.get()) return 0;
  std::size_t bytes = (data->values.size()+data->rhs.size()+data->solution.size())*
      sizeof(double)+data->destination_offsets.size()*sizeof(int);
  for (const auto &offsets : data->source_offsets) bytes += offsets.size()*sizeof(int);
  return bytes/data->lanes;
}

void ImplicitAMGHypre::ReleaseBatch() {
#ifdef ATHENA_HYPRE_ENABLED
  auto &s = *state_;
  s.reuse_cache.clear();
  s.batch_workspace.reset();
  if (s.batch) {
    s.batch.reset();
    s.ready = false;
    s.device_bytes = 0;
  }
#endif
}

void ImplicitAMGHypre::ConfigureReuse(int steps, int slots, double drift) {
  state_->reuse_steps = ReuseAvailable() ? steps : 0;
  state_->reuse_slots = slots;
  state_->reuse_drift = drift;
}

void ImplicitAMGHypre::SetReuseContext(int group, int cycle, bool enabled) {
  state_->reuse_group = group;
  state_->reuse_cycle = cycle;
#ifdef ATHENA_HYPRE_ENABLED
  state_->reuse_enabled = enabled && state_->reuse_steps > 0;
#else
  state_->reuse_enabled = false;
#endif
}

bool ImplicitAMGHypre::hierarchy_reused() const { return state_->hierarchy_reused; }

void ImplicitAMGHypre::ObserveIterations(int iterations, bool converged) {
  auto &s = *state_;
  if (!s.batch || !s.reuse_enabled) return;
  auto &data = *s.batch;
  if (data.last_cycle != s.reuse_cycle || s.batch_lane >= static_cast<int>(data.groups.size()) ||
      data.groups[s.batch_lane] != s.reuse_group) return;
  if (!converged) data.refresh_requested = true;
  if (s.batch_lane >= static_cast<int>(data.baseline_iterations.size())) return;
  int &baseline = data.baseline_iterations[s.batch_lane];
  if (!s.hierarchy_reused) {
    baseline = iterations;
  } else if (iterations > std::max(baseline+8, (3*baseline+1)/2)) {
    data.refresh_requested = true;
  }
}

void ImplicitAMGHypre::TrimReuseCache(std::size_t required, std::size_t credit) {
#ifdef ATHENA_HYPRE_ENABLED
  auto &s = *state_;
  while (!s.reuse_cache.empty()) {
    std::size_t available = 0;
    if (!s.FreeBytes(available)) return;
    int fits = available >= required || credit >= required-available;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &fits, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    if (fits) break;
    // Preserve the earliest admitted cohorts; LRU would thrash a five-cohort
    // sweep whenever fewer than five hierarchies fit.
    s.reuse_cache.erase(std::prev(s.reuse_cache.end()));
  }
#endif
}

bool ImplicitAMGHypre::ConfigureGraph(const std::vector<int> &row_offsets,
    const std::vector<int> &global_columns, int first_row, int global_rows,
    std::size_t max_bytes, std::size_t reserve_bytes) {
  auto &s = *state_;
  s.error.clear();
#ifndef ATHENA_HYPRE_ENABLED
  s.error = "AMG requires Athena_ENABLE_HYPRE=ON";
  return false;
#else
#if MPI_PARALLEL_ENABLED
  MPI_Comm_size(MPI_COMM_WORLD, &s.ranks);
#endif
  const auto validate = [&]() {
    if (row_offsets.size() < 2 || row_offsets.front() != 0
        || global_columns.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || row_offsets.back() != static_cast<int>(global_columns.size())
        || first_row < 0 || global_rows <= 0 || first_row >= global_rows
        || row_offsets.size() - 1 > static_cast<std::size_t>(global_rows - first_row)) {
      s.error = "invalid AMG CSR dimensions";
      return false;
    }
    for (std::size_t row = 0; row + 1 < row_offsets.size(); ++row) {
      if (row_offsets[row + 1] <= row_offsets[row]
          || row_offsets[row + 1] > row_offsets.back()) {
        s.error = "AMG CSR rows must be nonempty and ordered";
        return false;
      }
      int previous = -1;
      bool diagonal = false;
      for (int p = row_offsets[row]; p < row_offsets[row + 1]; ++p) {
        const int col = global_columns[p];
        if (col <= previous || col >= global_rows) {
          s.error = "AMG CSR columns must be sorted, unique and within the global range";
          return false;
        }
        diagonal = diagonal || col == first_row + static_cast<int>(row);
        previous = col;
      }
      if (!diagonal) {
        s.error = "AMG CSR row has no diagonal";
        return false;
      }
    }
    return true;
  };
  if (!s.Agree(validate())) return false;
  const int local_rows = static_cast<int>(row_offsets.size())-1;
  std::vector<int> ownership(3*s.ranks);
  const int local[3] = {first_row, local_rows, global_rows};
#if MPI_PARALLEL_ENABLED
  if (s.ranks > 1) {
    MPI_Allgather(local, 3, MPI_INT, ownership.data(), 3, MPI_INT, MPI_COMM_WORLD);
  } else {
    std::copy(local, local+3, ownership.begin());
  }
#else
  std::copy(local, local+3, ownership.begin());
#endif
  int next_row = 0;
  for (int rank = 0; rank < s.ranks; ++rank) {
    if (ownership[3*rank] != next_row || ownership[3*rank+2] != global_rows) {
      s.error = "AMG row ownership must be contiguous, disjoint and rank ordered";
      s.ready = false;
      return false;
    }
    next_row += ownership[3*rank+1];
  }
  if (next_row != global_rows) {
    s.error = "AMG row ownership must cover the entire global matrix";
    s.ready = false;
    return false;
  }
  s.max_bytes = max_bytes;
  s.reserve_bytes = reserve_bytes;
  int changed = s.first_row != first_row || s.global_rows != global_rows ||
      s.row_offsets != row_offsets || s.columns != global_columns;
#if MPI_PARALLEL_ENABLED
  if (s.ranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  if (!changed) return s.Agree(s.MemoryCheck());
  // Repartitioning on any rank invalidates hypre's distributed matrix metadata
  // on all ranks, even if this rank's CSR rows happen to be unchanged.
  s.reuse_cache.clear();
  s.batch_workspace.reset();
  s.DestroyObjects();
  s.DestroyTransfers();
  s.graph_revision = ++next_graph_revision;
  s.first_row = first_row;
  s.global_rows = global_rows;
  s.nrows = local_rows;
  s.row_offsets = row_offsets;
  s.columns = global_columns;
  return s.Agree(s.MemoryCheck());
#endif
}

bool ImplicitAMGHypre::Setup(const double *values) {
  auto &s = *state_;
  s.error.clear();
#ifndef ATHENA_HYPRE_ENABLED
  s.error = "AMG requires Athena_ENABLE_HYPRE=ON";
  return false;
#else
  s.ready = false;
  s.hierarchy_reused = false;
  if (auto *batch = two_temperature::RadiationGroupBatch::Active()) {
    BatchRequest request{this, values};
    batch->Submit(two_temperature::RadiationGroupBatch::Kind::AMGSetup,
                  &request, &ImplicitAMGHypre::SetupBatch);
    return request.success;
  }
  s.batch.reset();
  TrimReuseCache(s.max_bytes+s.reserve_bytes+kMemoryMargin, s.device_bytes);
  if (!s.use_saved_transfers) s.DestroyTransfers();
  const bool configured = s.nrows > 0 && values != nullptr;
  if (!configured) {
    s.error = "AMG setup requires a configured graph and numerical values";
  }
  if (!s.Agree(configured && s.MemoryCheck())) return false;
  Kokkos::fence("AMG numerical setup producer");
  if (!s.Synchronize()) return false;
  std::size_t free_before = 0;
  if (!s.FreeBytes(free_before)) return false;
  const std::size_t previous_bytes = s.device_bytes;
#define AMG_CHECK(call) do { if (!s.Check((call), #call)) return false; } while (false)
  if (!s.initialized) {
    if (hypre_users == 0 && !HYPRE_Initialized()) {
      AMG_CHECK(HYPRE_Init());
      hypre_owned = true;
#ifdef KOKKOS_ENABLE_CUDA
      // HYPRE_Init creates device metadata but does not bind it to the CUDA
      // device selected by Kokkos. Initialize once, before any hypre allocation.
      AMG_CHECK(HYPRE_DeviceInitialize());
#endif
    }
    ++hypre_users;
    s.initialized = true;
    AMG_CHECK(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
#ifdef KOKKOS_ENABLE_CUDA
    // RAP and interpolation repeatedly allocate workspaces larger than 64 MiB.
    // Admit bins through 256 MiB while bounding unused storage by both 1 GiB
    // and a quarter of this hierarchy's budget. CUB reuses raw storage only:
    // every matrix entry is rebuilt, including after coefficient/graph changes.
    // Live bin rounding and cached storage remain subject to the free-memory
    // and hierarchy-budget checks below.
    // This public call is a no-op in HYPRE builds without a device pool.
    AMG_CHECK(HYPRE_SetGPUMemoryPoolSize(2, 6, 28,
        std::min<std::size_t>(1024ULL*1024*1024, s.max_bytes/4)));
    AMG_CHECK(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
    AMG_CHECK(HYPRE_SetSpGemmUseCusparse(0));
    // Native SpMV preserves beta*y on scalar coarse levels; the CUDA 12.3
    // vendor path returned zero for A*0+y in the scalar regression.
    AMG_CHECK(HYPRE_SetSpMVUseVendor(0));
#endif
  }
  if (s.matrix == nullptr) {
    if (s.counts.extent_int(0) != s.nrows || s.cols.size() != s.columns.size()) {
      s.counts = decltype(s.counts)("amg row counts", s.nrows);
      s.host_counts = decltype(s.host_counts)("amg host row counts", s.nrows);
      s.rows = decltype(s.rows)("amg row ids", s.nrows);
      s.cols = decltype(s.cols)("amg column ids", s.columns.size());
      auto hrows = Kokkos::create_mirror_view(s.rows);
      auto hcols = Kokkos::create_mirror_view(s.cols);
      for (int row = 0; row < s.nrows; ++row) {
        s.host_counts(row) = s.row_offsets[row+1]-s.row_offsets[row];
        hrows(row) = s.first_row+row;
      }
      for (std::size_t p = 0; p < s.columns.size(); ++p) hcols(p) = s.columns[p];
      Kokkos::deep_copy(s.counts, s.host_counts);
      Kokkos::deep_copy(s.rows, hrows);
      Kokkos::deep_copy(s.cols, hcols);
    }
    const HYPRE_BigInt first = s.first_row, last = first + s.nrows - 1;
#if MPI_PARALLEL_ENABLED
    const MPI_Comm comm = MPI_COMM_WORLD;
#else
    const MPI_Comm comm = 0;
#endif
    AMG_CHECK(HYPRE_IJMatrixCreate(comm, first, last, first, last, &s.matrix));
    AMG_CHECK(HYPRE_IJMatrixSetObjectType(s.matrix, HYPRE_PARCSR));
    AMG_CHECK(HYPRE_IJMatrixSetRowSizes(s.matrix, s.host_counts.data()));
    AMG_CHECK(HYPRE_IJVectorCreate(comm, first, last, &s.rhs));
    AMG_CHECK(HYPRE_IJVectorCreate(comm, first, last, &s.solution));
    AMG_CHECK(HYPRE_IJVectorSetObjectType(s.rhs, HYPRE_PARCSR));
    AMG_CHECK(HYPRE_IJVectorSetObjectType(s.solution, HYPRE_PARCSR));
    AMG_CHECK(HYPRE_IJVectorInitialize_v2(s.rhs, HYPRE_MEMORY_DEVICE));
    AMG_CHECK(HYPRE_IJVectorInitialize_v2(s.solution, HYPRE_MEMORY_DEVICE));
    AMG_CHECK(HYPRE_IJVectorAssemble(s.rhs));
    AMG_CHECK(HYPRE_IJVectorAssemble(s.solution));
    AMG_CHECK(HYPRE_IJVectorGetObject(s.rhs, reinterpret_cast<void **>(&s.par_rhs)));
    AMG_CHECK(HYPRE_IJVectorGetObject(s.solution,
                                    reinterpret_cast<void **>(&s.par_solution)));
  }
#ifdef KOKKOS_ENABLE_CUDA
  if (s.solver != nullptr) {
    // GPU IJ SetValues/Assemble builds a COO stack and merges it with the old
    // ParCSR entries, even when every coefficient is replaced. Assemble this
    // complete matrix afresh to avoid sorting/merging two copies of its graph
    // and the corresponding peak workspace. Retain row/column metadata and
    // vectors; destroy the old hierarchy before its borrowed fine matrix.
    AMG_CHECK(HYPRE_BoomerAMGDestroy(s.solver));
    s.solver = nullptr;
    AMG_CHECK(HYPRE_IJMatrixDestroy(s.matrix));
    s.matrix = nullptr;
    s.par_matrix = nullptr;
#if MPI_PARALLEL_ENABLED
    const MPI_Comm comm = MPI_COMM_WORLD;
#else
    const MPI_Comm comm = 0;
#endif
    const HYPRE_BigInt first = s.first_row, last = first+s.nrows-1;
    AMG_CHECK(HYPRE_IJMatrixCreate(comm, first, last, first, last, &s.matrix));
    AMG_CHECK(HYPRE_IJMatrixSetObjectType(s.matrix, HYPRE_PARCSR));
  }
#endif
  // The host IJ implementation can update existing CSR entries in place.
  AMG_CHECK(HYPRE_IJMatrixInitialize_v2(s.matrix, HYPRE_MEMORY_DEVICE));
  AMG_CHECK(HYPRE_IJMatrixSetValues(s.matrix, s.nrows, s.counts.data(), s.rows.data(),
                                   s.cols.data(), values));
  AMG_CHECK(HYPRE_IJMatrixAssemble(s.matrix));
  AMG_CHECK(HYPRE_IJMatrixGetObject(s.matrix, reinterpret_cast<void **>(&s.par_matrix)));
  if (s.solver != nullptr) {
    AMG_CHECK(HYPRE_BoomerAMGDestroy(s.solver));
    s.solver = nullptr;
  }
  AMG_CHECK(HYPRE_BoomerAMGCreate(&s.solver));
#ifdef ATHENA_HYPRE_DIAGNOSTICS
  AMG_CHECK(HYPRE_BoomerAMGSetPrintLevel(s.solver, 1));
#else
  AMG_CHECK(HYPRE_BoomerAMGSetPrintLevel(s.solver, 0));
#endif
  AMG_CHECK(HYPRE_BoomerAMGSetMaxIter(s.solver, 1));
  AMG_CHECK(HYPRE_BoomerAMGSetTol(s.solver, 0.0));
  AMG_CHECK(HYPRE_BoomerAMGSetCycleType(s.solver, 1));
  AMG_CHECK(HYPRE_BoomerAMGSetCoarsenType(s.solver, 8));
  AMG_CHECK(HYPRE_BoomerAMGSetInterpType(s.solver, 6));
  AMG_CHECK(HYPRE_BoomerAMGSetStrongThreshold(s.solver, 0.25));
  AMG_CHECK(HYPRE_BoomerAMGSetPMaxElmts(s.solver, 4));

  AMG_CHECK(HYPRE_BoomerAMGSetMaxLevels(s.solver, 25));
  // Tiny distributed levels are dominated by latency. Stop at a bounded coarse
  // matrix and use HYPRE's replicated dense inverse (one RHS gather per cycle).
  // Single-rank runs retain the scalar coarse solve, which needs no MPI gather.
  // If the entire input fits the coarse bound, HYPRE can build a one-level
  // hierarchy whose solve uses UserRelaxType instead of the coarse type. With
  // type 199 this leaves the l1 smoother without norms. Small input graphs keep
  // ordinary coarsening and l1 coarse relaxation, including nearly diagonal A.
  const bool distributed_coarse = s.ranks > 1 && s.global_rows > 1024;
  AMG_CHECK(HYPRE_BoomerAMGSetMaxCoarseSize(s.solver, distributed_coarse ? 1024 : 1));
  AMG_CHECK(HYPRE_BoomerAMGSetRelaxType(s.solver, 18));
  AMG_CHECK(HYPRE_BoomerAMGSetRelaxOrder(s.solver, 0));
  AMG_CHECK(HYPRE_BoomerAMGSetRelaxWt(s.solver, 1.0));
  AMG_CHECK(HYPRE_BoomerAMGSetNumSweeps(s.solver, 1));
  AMG_CHECK(HYPRE_BoomerAMGSetCycleRelaxType(s.solver, distributed_coarse ? 199 : 18, 3));
  AMG_CHECK(HYPRE_BoomerAMGSetCycleNumSweeps(s.solver, distributed_coarse ? 1 : 8, 3));
  AMG_CHECK(HYPRE_BoomerAMGSetKeepTranspose(s.solver, 1));
  AMG_CHECK(HYPRE_BoomerAMGSetRAP2(s.solver, 0));
  AMG_CHECK(HYPRE_ParVectorSetConstantValues(s.par_rhs, 0.0));
  AMG_CHECK(HYPRE_ParVectorSetConstantValues(s.par_solution, 0.0));
  HYPRE_Int setup_status = 0;
#ifdef ATHENA_HYPRE_REUSE_ENABLED
  if (s.use_saved_transfers && s.HasSavedTransfers()) {
    athena_hypre_reuse_transfers(s.saved_p.data(), s.saved_cf.data(),
                                static_cast<int>(s.saved_p.size())+1);
    setup_status = athena_hypre_BoomerAMGSetup(s.solver,
        reinterpret_cast<hypre_ParCSRMatrix *>(s.par_matrix),
        reinterpret_cast<hypre_ParVector *>(s.par_rhs),
        reinterpret_cast<hypre_ParVector *>(s.par_solution));
    athena_hypre_reuse_transfers(nullptr, nullptr, 0);
  } else
#endif
  {
    setup_status = HYPRE_BoomerAMGSetup(s.solver, s.par_matrix, s.par_rhs, s.par_solution);
  }
  AMG_CHECK(setup_status);
  // The patched setup transfers ownership into the new solver and nulls the
  // borrowed slots. Any unconsumed slots are released on an error path.
  s.DestroyTransfers();
  s.use_saved_transfers = false;
  if (!s.Synchronize()) return false;
  bool within_budget = true;
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t free_after = 0;
  if (!s.FreeBytes(free_after)) return false;
  const std::size_t available_before = free_before + previous_bytes;
  s.device_bytes = available_before > free_after ? available_before - free_after : 0;
  if (s.device_bytes > s.max_bytes || free_after < s.reserve_bytes + kMemoryMargin) {
    s.error = "AMG setup exceeded its memory budget or reserved solver headroom";
    within_budget = false;
  }
#endif
  if (!s.Agree(within_budget)) {
    s.DestroyObjects();
    return false;
  }
  s.ready = true;
  s.account_first_apply = true;
  return true;
#undef AMG_CHECK
#endif
}

bool ImplicitAMGHypre::Apply(const double *rhs, double *solution) {
  auto &s = *state_;
  s.error.clear();
#ifndef ATHENA_HYPRE_ENABLED
  s.error = "AMG requires Athena_ENABLE_HYPRE=ON";
  return false;
#else
  if (!s.ready || rhs == nullptr || solution == nullptr) {
    s.error = "AMG apply requires a successful numerical setup and valid vectors";
    s.AbortDistributed();
    return false;
  }
  if (auto *batch = two_temperature::RadiationGroupBatch::Active()) {
    BatchRequest request{this, rhs, solution};
    batch->Submit(two_temperature::RadiationGroupBatch::Kind::AMGApply,
                  &request, &ImplicitAMGHypre::ApplyBatch);
    return request.success;
  }
  if (s.batch) {
    BatchRequest request{this, rhs, solution};
    ApplyBatch({&request});
    return request.success;
  }
  Kokkos::fence("AMG right hand side producer");
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t free_before = 0;
  if (s.account_first_apply && !s.FreeBytes(free_before)) return false;
#endif
#define AMG_CHECK(call) do { if (!s.Check((call), #call)) return false; } while (false)
  // SetValues queues GPU contributions until Assemble. This vector is already
  // assembled and all indices are owned and unique: update its storage directly.
  AMG_CHECK(HYPRE_IJVectorUpdateValues(s.rhs, s.nrows, s.rows.data(), rhs, 1));
  AMG_CHECK(HYPRE_ParVectorSetConstantValues(s.par_solution, 0.0));
  AMG_CHECK(HYPRE_BoomerAMGSolve(s.solver, s.par_matrix, s.par_rhs, s.par_solution));
  // A null index list copies the entire contiguous local vector without the
  // GPU index-validation reduction performed by the indexed GetValues path.
  AMG_CHECK(HYPRE_IJVectorGetValues(s.solution, s.nrows, nullptr, solution));
  if (!s.Synchronize()) return false;
#ifdef KOKKOS_ENABLE_CUDA
  if (s.account_first_apply) {
    // hypre creates persistent communication/smoother buffers on first use.
    // They are freed with the hierarchy during the next Setup. If their growth
    // is omitted here, every rebuild subtracts it from the reusable-byte count
    // and eventually rejects a hierarchy even though free GPU memory is stable.
    std::size_t free_after = 0;
    if (!s.FreeBytes(free_after)) return false;
    if (free_before > free_after) {
      s.device_bytes += free_before-free_after;
    } else {
      const std::size_t released = free_after-free_before;
      s.device_bytes = s.device_bytes > released ? s.device_bytes-released : 0;
    }
    if (s.device_bytes > s.max_bytes || free_after < s.reserve_bytes+kMemoryMargin) {
      s.error = "AMG first application exceeded its memory budget or reserved headroom";
      s.ready = false;
      s.AbortDistributed();
      return false;
    }
  }
#endif
  s.account_first_apply = false;
  return true;
#undef AMG_CHECK
#endif
}

void ImplicitAMGHypre::SetupBatch(const std::vector<void *> &requests) {
#ifdef ATHENA_HYPRE_ENABLED
  auto &first = *static_cast<BatchRequest *>(requests.front());
  if (requests.size() == 1) {
    first.success = first.solver->Setup(first.input);
    return;
  }
  const auto setup_start = std::chrono::steady_clock::now();
  const int lanes = static_cast<int>(requests.size());
  auto &base = *first.solver->state_;
  std::size_t max_bytes = 0, reserve_bytes = 0, entries = 0;
  bool valid = base.global_rows <= std::numeric_limits<int>::max()/lanes;
  for (void *p : requests) {
    auto &r = *static_cast<BatchRequest *>(p);
    auto &s = *r.solver->state_;
    valid = valid && r.input != nullptr && s.nrows == base.nrows &&
        s.first_row == base.first_row && s.global_rows == base.global_rows;
    max_bytes += s.max_bytes;
    reserve_bytes += s.reserve_bytes;
    entries += s.columns.size();
  }
  valid = valid && entries <= static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (!base.Agree(valid)) {
    for (void *p : requests) {
      static_cast<BatchRequest *>(p)->solver->state_->error = "Incompatible AMG group graphs";
    }
    return;
  }
  // Each physical group keeps its own A_g. Interleaving their row numbers gives
  // a block-diagonal operator with no edges between groups and one HYPRE exchange
  // per hierarchy operation; no group's diffusion matrix is reused for another.
  // Retain the allocation owner across numerical refreshes, including the
  // HYPRE runtime and its bounded GPU allocation pool.
  std::vector<int> groups;
  bool eligible = base.reuse_enabled;
  for (void *p : requests) {
    const auto &s = *static_cast<BatchRequest *>(p)->solver->state_;
    eligible = eligible && s.reuse_enabled && s.reuse_cycle == base.reuse_cycle;
    groups.push_back(s.reuse_group);
  }
  // A disabled/retry invocation cannot accidentally pick a lagged hierarchy.
  std::shared_ptr<BatchData> data;
  auto cached = base.reuse_cache.find(groups);
  if (eligible && cached != base.reuse_cache.end()) data = cached->second;
  if (!data && base.batch) {
    bool pinned = false;
    for (const auto &item : base.reuse_cache) pinned |= item.second == base.batch;
    if (!pinned) data = base.batch;
  }
  if (!data) data = std::make_shared<BatchData>();
  // Retain only transfers for cached cohorts between applications. This lets
  // more groups remain cached than retaining every complete Galerkin hierarchy.
  for (auto &item : base.reuse_cache) item.second->solver.state_->KeepTransfersOnly();
  // Release the previous active cohort before allocating/refreshing another.
  for (void *p : requests) static_cast<BatchRequest *>(p)->solver->state_->DestroyObjects();

  bool same_graph = data->lanes == lanes;
  for (int lane = 0; same_graph && lane < lanes; ++lane) {
    const auto *s = static_cast<BatchRequest *>(requests[lane])->solver->state_.get();
    same_graph = data->owners[lane] == s && data->revisions[lane] == s->graph_revision;
  }
  auto workspace = base.batch_workspace;
  bool same_workspace = workspace && workspace->lanes == lanes;
  for (int lane = 0; same_workspace && lane < lanes; ++lane) {
    const auto *owner = static_cast<BatchRequest *>(requests[lane])->solver->state_.get();
    same_workspace = workspace->owners[lane] == owner &&
                     workspace->revisions[lane] == owner->graph_revision;
  }
  if (!same_workspace) {
    workspace = std::make_shared<BatchWorkspace>();
    workspace->lanes = lanes;
    for (void *p : requests) {
      const auto *owner = static_cast<BatchRequest *>(p)->solver->state_.get();
      workspace->owners.push_back(owner);
      workspace->revisions.push_back(owner->graph_revision);
    }
    base.batch_workspace = workspace;
  }
  int reuse = eligible && same_graph && data->groups == groups &&
      data->configured && data->solver.state_->HasSavedTransfers() &&
      data->built_cycle >= 0 && !data->refresh_requested &&
      base.reuse_cycle > data->last_cycle &&
      base.reuse_cycle-data->built_cycle < base.reuse_steps;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &reuse, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  double drift = 0.0;
  if (reuse) {
    for (int lane = 0; lane < lanes; ++lane) {
      auto &r = *static_cast<BatchRequest *>(requests[lane]);
      auto source = data->source_offsets[lane];
      auto reference = data->reference_row_norm;
      const double *current = r.input;
      double lane_drift = 0.0;
      Kokkos::parallel_reduce("cell AMG row norm drift",
          Kokkos::RangePolicy<DevExeSpace>(0, data->rows),
          KOKKOS_LAMBDA(int row, double &maximum) {
            double magnitude = 0.0;
            for (int q = source(row); q < source(row+1); ++q) {
              magnitude += fabs(current[q]);
            }
            const double previous = reference(lanes*row+lane);
            const double difference = Kokkos::isfinite(magnitude) &&
                Kokkos::isfinite(previous) ? fabs(magnitude-previous)/
                fmax(previous, 1.0e-300) : 1.0e300;
            maximum = fmax(maximum, difference);
          }, Kokkos::Max<double>(lane_drift));
      drift = std::max(drift, lane_drift);
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &drift, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    reuse = std::isfinite(drift) && drift <= base.reuse_drift;
  }
  {
    // Numeric refresh and full setup both preserve the configured headroom. Evict cached cohorts
    // before setup; the current data remains alive and earns its own credit.
    const std::size_t packing_growth = same_workspace ? 0 :
        entries*sizeof(double)+lanes*base.nrows*(2*sizeof(double)+2*sizeof(int));
    first.solver->TrimReuseCache(max_bytes+reserve_bytes+kMemoryMargin+packing_growth,
                                data->solver.device_bytes());
  }
  if (!same_graph) {
    data->built_cycle = -1;
    data->lanes = lanes;
    data->rows = base.nrows;
    data->offsets.assign(1, 0);
    data->columns.clear();
    data->columns.reserve(entries);
    data->owners.resize(lanes);
    data->revisions.resize(lanes);
    data->source_offsets.resize(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
      const auto *s = static_cast<BatchRequest *>(requests[lane])->solver->state_.get();
      data->owners[lane] = s;
      data->revisions[lane] = s->graph_revision;
    }
    for (int row = 0; row < base.nrows; ++row) {
      for (int lane = 0; lane < lanes; ++lane) {
        const auto &s = *static_cast<BatchRequest *>(requests[lane])->solver->state_;
        for (int q = s.row_offsets[row]; q < s.row_offsets[row+1]; ++q) {
          data->columns.push_back(lanes*s.columns[q]+lane);
        }
        data->offsets.push_back(static_cast<int>(data->columns.size()));
      }
    }
    if (same_workspace) {
      data->source_offsets = workspace->source_offsets;
      data->destination_offsets = workspace->destination_offsets;
      data->values = workspace->values;
      data->rhs = workspace->rhs;
      data->solution = workspace->solution;
    } else {
      for (int lane = 0; lane < lanes; ++lane) {
        const auto &source = static_cast<BatchRequest *>(requests[lane])->solver->state_->row_offsets;
        auto &map = data->source_offsets[lane];
        map = DvceArray1D<int>("batched AMG source offsets", source.size());
        auto host = Kokkos::create_mirror_view(map);
        for (std::size_t q = 0; q < source.size(); ++q) host(q) = source[q];
        Kokkos::deep_copy(map, host);
      }
      data->destination_offsets = DvceArray1D<int>("batched AMG destination offsets",
                                                 data->offsets.size());
      auto offsets_host = Kokkos::create_mirror_view(data->destination_offsets);
      for (std::size_t q = 0; q < data->offsets.size(); ++q) offsets_host(q) = data->offsets[q];
      Kokkos::deep_copy(data->destination_offsets, offsets_host);
      data->values = DvceArray1D<double>("batched AMG values", entries);
      data->rhs = DvceArray1D<double>("batched AMG rhs", lanes*data->rows);
      data->solution = DvceArray1D<double>("batched AMG solution", lanes*data->rows);
    }
    data->reference_row_norm = DvceArray1D<double>("AMG row norm history", lanes*data->rows);
  }
  if (!same_workspace) {
    workspace->source_offsets = data->source_offsets;
    workspace->destination_offsets = data->destination_offsets;
    workspace->values = data->values;
    workspace->rhs = data->rhs;
    workspace->solution = data->solution;
  } else {
    // A cached cohort can outlive a batch-size change. Bind only buffers whose
    // exact owner/revision signature matches the current batch.
    data->source_offsets = workspace->source_offsets;
    data->destination_offsets = workspace->destination_offsets;
    data->values = workspace->values;
    data->rhs = workspace->rhs;
    data->solution = workspace->solution;
  }
  // Numerical A and RAP are refreshed every solve. Only interpolation is
  // cached; shared packing buffers therefore need no old coefficient copies.
  for (int lane = 0; lane < lanes; ++lane) {
    auto &r = *static_cast<BatchRequest *>(requests[lane]);
    auto source = data->source_offsets[lane];
    auto destination = data->destination_offsets;
    auto values = data->values;
    const double *input = r.input;
    // Each CSR row is contiguous in both matrices. Two row-offset arrays replace
    // an entry-sized scatter map for every lane, saving memory on cell graphs.
    Kokkos::parallel_for("radiation batch matrix", Kokkos::RangePolicy<DevExeSpace>(0, data->rows),
        KOKKOS_LAMBDA(int row) {
          const int begin = source(row), end = source(row+1);
          const int target = destination(lanes*row+lane);
          for (int q = begin; q < end; ++q) values(target+q-begin) = input[q];
        });
  }
  const int first_row = lanes*base.first_row, global_rows = lanes*base.global_rows;
  int configure = !same_graph || !data->configured || data->max_bytes != max_bytes ||
      data->reserve_bytes != reserve_bytes;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &configure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  // Reuse validation of unchanged integer topology. Setup still rebuilds every
  // numerical hierarchy and checks current memory on every call.
  const bool graph_ok = !configure || data->solver.ConfigureGraph(data->offsets, data->columns,
      first_row, global_rows, max_bytes, reserve_bytes);
  data->configured = graph_ok;
  data->max_bytes = max_bytes;
  data->reserve_bytes = reserve_bytes;
  if (graph_ok && workspace->counts.size()) {
    auto &numeric = *data->solver.state_;
    numeric.counts = workspace->counts;
    numeric.host_counts = workspace->host_counts;
    numeric.rows = workspace->rows;
    numeric.cols = workspace->cols;
  }
  data->solver.state_->use_saved_transfers = reuse;
  const bool success = graph_ok && data->solver.Setup(data->values.data());
  if (success) {
    workspace->counts = data->solver.state_->counts;
    workspace->host_counts = data->solver.state_->host_counts;
    workspace->rows = data->solver.state_->rows;
    workspace->cols = data->solver.state_->cols;
    auto reference = data->reference_row_norm;
    auto offsets = data->destination_offsets;
    auto values = data->values;
    Kokkos::parallel_for("cell AMG store row norm",
        Kokkos::RangePolicy<DevExeSpace>(0, lanes*data->rows),
        KOKKOS_LAMBDA(int row) {
          double magnitude = 0.0;
          for (int q = offsets(row); q < offsets(row+1); ++q) magnitude += fabs(values(q));
          reference(row) = magnitude;
        });
    data->groups = groups;
    data->last_cycle = base.reuse_cycle;
    if (!reuse) {
      data->built_cycle = base.reuse_cycle;
      data->refresh_requested = false;
      data->baseline_iterations.assign(lanes, 0);
    }
    if (eligible && groups.front() >= 0 && groups.front()/lanes < base.reuse_slots) {
      base.reuse_cache[groups] = data;
      while (base.reuse_cache.size() > static_cast<std::size_t>(base.reuse_slots)) {
        base.reuse_cache.erase(std::prev(base.reuse_cache.end()));
      }
    }
  }
  if (eligible) {
    int rank = 0;
#if MPI_PARALLEL_ENABLED
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (rank == 0) {
      std::cout << "# cell-amg reuse: cycle=" << base.reuse_cycle
                << " first_group=" << groups.front() << " lanes=" << lanes
                << " reused=" << reuse << " refresh=RAP row_norm_drift=" << drift
                << " built_cycle=" << data->built_cycle
                << " cached_cohorts=" << base.reuse_cache.size()
                << " hierarchy_bytes=" << data->solver.device_bytes()
                << " setup_seconds=" << std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-setup_start).count() << std::endl;
    }
  }

  for (int lane = 0; lane < lanes; ++lane) {
    auto &r = *static_cast<BatchRequest *>(requests[lane]);
    auto &s = *r.solver->state_;
    r.success = success;
    s.error = data->solver.error();
    s.ready = success;
    s.hierarchy_reused = reuse && success;
    if (success) {
      s.batch = data;
      s.batch_lane = lane;
      s.device_bytes = data->solver.device_bytes()/lanes;
    }
  }
#endif
}

void ImplicitAMGHypre::ApplyBatch(const std::vector<void *> &requests) {
#ifdef ATHENA_HYPRE_ENABLED
  std::vector<bool> done(requests.size(), false);
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (done[i]) continue;
    auto &first = *static_cast<BatchRequest *>(requests[i]);
    const auto data = first.solver->state_->batch;
    if (!data) {
      first.success = first.solver->Apply(first.input, first.output);
      done[i] = true;
      continue;
    }
    std::vector<BatchRequest *> same;
    for (std::size_t j = i; j < requests.size(); ++j) {
      auto &r = *static_cast<BatchRequest *>(requests[j]);
      if (!done[j] && r.solver->state_->batch == data) { same.push_back(&r); done[j] = true; }
    }
    const int lanes = data->lanes;
    auto rhs = data->rhs, solution = data->solution;
    // Completed or temporarily waiting lanes contribute zero RHSs. Their
    // operators remain fixed until both independent Krylov solves finish.
    Kokkos::deep_copy(DevExeSpace(), rhs, 0.0);
    for (auto *r : same) {
      const int lane = r->solver->state_->batch_lane;
      const double *input = r->input;
      Kokkos::parallel_for("radiation batch rhs", Kokkos::RangePolicy<DevExeSpace>(0, data->rows),
          KOKKOS_LAMBDA(int row) { rhs(lanes*row+lane) = input[row]; });
    }
    const bool success = data->solver.Apply(rhs.data(), solution.data());
    for (auto *r : same) {
      r->success = success;
      r->solver->state_->error = data->solver.error();
      if (!success) continue;
      const int lane = r->solver->state_->batch_lane;
      double *output = r->output;
      Kokkos::parallel_for("radiation batch correction",
          Kokkos::RangePolicy<DevExeSpace>(0, data->rows),
          KOKKOS_LAMBDA(int row) { output[row] = solution(lanes*row+lane); });
    }
    Kokkos::fence("radiation batch correction consumer");
  }
#endif
}
