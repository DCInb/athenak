#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_BATCH_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_BATCH_HPP_
// Cooperative group scheduling: all Kokkos, CUDA and MPI calls stay on the
// calling host thread. Each lane retains the ordinary scalar solver's stack.
// Only explicitly submitted independent operations may be combined.
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace two_temperature {

class RadiationGroupBatch {
 public:
  enum class Kind { AMGSetup, AMGApply, Halo, Reduction };
  using Execute = void (*)(const std::vector<void *> &);
  explicit RadiationGroupBatch(bool timing = false);
  ~RadiationGroupBatch();
  RadiationGroupBatch(const RadiationGroupBatch &) = delete;
  RadiationGroupBatch &operator=(const RadiationGroupBatch &) = delete;
  static bool Supported();
  // Non-null only inside a lane, never inside an executing batch callback.
  static RadiationGroupBatch *Active();
  void Run(const std::vector<std::function<void()>> &work);
  void Submit(Kind kind, void *request, Execute execute);
  struct Counts {
    std::uint64_t requests[4] = {}, batches[4] = {}, paired[4] = {};
    // Exclusive scheduler callback wall time, never suspended-lane time.
    double seconds[4] = {};
  };
  const Counts &counts() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

#if MPI_PARALLEL_ENABLED
// Built-in contiguous host reductions can share one collective while keeping
// every lane's entries separate. Other operations retain ordinary MPI semantics.
int RadiationBatchAllreduce(const void *send, void *recv, int count,
                            MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif

}  // namespace two_temperature
#endif
