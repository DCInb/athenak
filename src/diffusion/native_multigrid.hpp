#ifndef DIFFUSION_NATIVE_MULTIGRID_HPP_
#define DIFFUSION_NATIVE_MULTIGRID_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <cstddef>
#include <memory>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace diffusion {

// Native aggregation multigrid for M + graph Laplacian. Owned rows are contiguous
// in MPI rank order. CSR contains off-diagonal global columns only; edge weights
// are nonnegative and symmetric. Mass is strictly positive (boundary penalties
// may be included). All numerical arrays live in Athena's default memory space.
// Configure/Setup/Apply are collective, including on ranks with no owned rows.
// The fixed symmetric V-cycle is suitable for ordinary PCG in the volume metric.
class NativeMultigrid {
 public:
  struct State;  // Public for nvcc extended device lambdas.
  NativeMultigrid();
  ~NativeMultigrid();
  NativeMultigrid(const NativeMultigrid &) = delete;
  NativeMultigrid &operator=(const NativeMultigrid &) = delete;

  void Configure(const std::vector<int> &offsets, const std::vector<int> &columns,
                 int first_row, int global_rows, bool device_mpi = false,
                 std::size_t max_bytes = 1024ULL*1024*1024);
  // Reuse integer topology while its strong connections remain unchanged.
  // Refresh numerical levels and rebuild from the first changed strength graph.
  void Setup(const DvceArray1D<double> &conductance, const DvceArray1D<double> &mass);
  void Apply(const DvceArray1D<double> &rhs, const DvceArray1D<double> &solution);
#if MPI_PARALLEL_ENABLED
  // Optional cooperative scheduler: all requests are posted before this wait.
  // The callback must complete them before returning, including an empty list.
  using HaloWait = void (*)(int, MPI_Request *);
  void SetHaloWait(HaloWait wait);
#endif
  int levels() const;
  std::size_t bytes() const;

 private:
  std::unique_ptr<State> state_;
};

}  // namespace diffusion
#endif  // DIFFUSION_NATIVE_MULTIGRID_HPP_
