#ifndef FAST_ELECTRONS_FAST_ELECTRON_MEMORY_HPP_
#define FAST_ELECTRONS_FAST_ELECTRON_MEMORY_HPP_

#include <cstdint>
#include <string>
#include "athena.hpp"

class ParameterInput;

namespace fast_electrons::memory {
struct Snapshot {
  std::uint64_t total = 0, peak = 0, packets = 0, sources = 0;
  std::uint64_t mesh = 0, runtime = 0, diagnostics = 0;
  std::uint64_t allocations = 0, allocated_bytes = 0, untracked_frees = 0;
};

// Install before constructing the mesh. Existing allocation-tool callbacks are chained.
// The budget covers all Kokkos allocations in the execution memory space on this rank.
void Initialize(ParameterInput *pin);
bool Reporting();
// Start a rank-local log in the run directory; flush startup allocation records.
void Report(const std::string &record);
void Context(const char *phase, std::uint64_t live, bool known = true);
void Time(Real time);
Snapshot Get();
void Preflight(std::uint64_t additional_bytes, const char *purpose);
}  // namespace fast_electrons::memory
#endif
