#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_HYPRE_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_HYPRE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Optional hypre wrapper. Public arguments deliberately contain no hypre types.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class ImplicitAMGHypre {
 public:
  ImplicitAMGHypre();
  ~ImplicitAMGHypre();
  ImplicitAMGHypre(const ImplicitAMGHypre &) = delete;
  ImplicitAMGHypre &operator=(const ImplicitAMGHypre &) = delete;

  static bool Available();
  static bool ReuseAvailable();
  // CSR contains this rank's rows of a symmetric global matrix, with sorted
  // global columns and host metadata. ConfigureGraph and Setup are collective
  // in MPI builds; nonempty row intervals must cover the matrix in rank order.
  // Identical topology is retained; memory limits may be refreshed independently.
  bool ConfigureGraph(const std::vector<int> &row_offsets,
                      const std::vector<int> &global_columns,
                      int first_row, int global_rows, std::size_t max_bytes,
                      std::size_t reserve_bytes = 0);
  // Numerical arrays reside in Athena's default memory space and are always double.
  // Setup rebuilds the numerical Galerkin hierarchy for the current matrix.
  bool Setup(const double *values);
  // Apply one fixed symmetric V-cycle from a zero initial guess. Vectors contain
  // only owned rows. Runtime failures abort multi-rank calls to avoid deadlocks.
  bool Apply(const double *rhs, double *solution);
  const std::string &error() const;
  // CUDA free-memory delta, including cached graph and hierarchy; zero on CPU.
  std::size_t device_bytes() const;
  // This lane's share of packing arrays outside the HYPRE hierarchy budget.
  std::size_t batch_workspace_bytes() const;

  // Release a completed shared hierarchy, retaining this lane's integer graph.
  // Call only after every Krylov solve in the batch has finished.
  void ReleaseBatch();

  // Optional transfer cache (Athena_ENABLE_HYPRE_REUSE); numerical A/RAP stay current.
  // Configure before the first setup; steps=0 disables reuse. No environment overrides.
  void ConfigureReuse(int steps, int slots, double drift);
  // Context is supplied for each physical group, including solves which skip setup.
  void SetReuseContext(int group, int cycle, bool enabled);
  void ObserveIterations(int iterations, bool converged);
  bool hierarchy_reused() const;
  // Eviction is collective and happens between completed batches only.
  void TrimReuseCache(std::size_t required, std::size_t credit);

  // Public for nvcc extended device lambdas; internal batch callbacks only.
  static void SetupBatch(const std::vector<void *> &requests);
  static void ApplyBatch(const std::vector<void *> &requests);

 private:
  struct BatchWorkspace;
  struct BatchData;
  struct BatchRequest;
  struct State;
  std::unique_ptr<State> state_;
};

#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_HYPRE_HPP_
