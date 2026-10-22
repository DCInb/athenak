// Aggregate allocation accounting, including old/new overlap and non-particle arrays.
#include "fast_electrons/fast_electron_memory.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "globals.hpp"
#include "parameter_input.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace fast_electrons::memory {
namespace {
enum class Kind { packet, source, mesh, runtime, diagnostic };
struct Allocation { std::uint64_t bytes; Kind kind; };
struct State {
  bool enabled = false, report = false, live_known = true;
  std::uint64_t budget = 0, headroom = 0, live = 0;
  Real time = 0.0;
  std::string phase = "initialization";
  Snapshot current;
  std::ofstream log;
  std::vector<std::string> pending;
  std::map<const void *, Allocation> allocations;
  std::mutex mutex;
  Kokkos::Tools::allocateDataFunction previous_allocate = nullptr;
  Kokkos::Tools::deallocateDataFunction previous_free = nullptr;
};
State &Data() { static State state; return state; }

void Log(State &state, const std::string &record) {
  if (state.log.is_open()) {
    state.log << record << '\n';
    if (!state.log) Kokkos::abort("### FATAL ERROR: writing fast-electron memory log");
  } else {
    state.pending.push_back(record);
  }
  std::cout << record << std::endl;
}

Kind Category(const char *label) {
  const std::string name(label);
  if (name.find("fe-source-templates") != std::string::npos) return Kind::source;
  if (name.find("fe-sampling-") != std::string::npos) return Kind::diagnostic;
  if (name.find("packet") != std::string::npos) return Kind::packet;
  if (name.find("Kokkos::") == 0) return Kind::runtime;
  return Kind::mesh;
}
std::uint64_t &CategoryBytes(Snapshot &s, Kind kind) {
  switch (kind) {
    case Kind::packet: return s.packets;
    case Kind::source: return s.sources;
    case Kind::runtime: return s.runtime;
    case Kind::diagnostic: return s.diagnostics;
    default: return s.mesh;
  }
}
bool ExecutionSpace(Kokkos::Tools::SpaceHandle space) {
  return std::strcmp(space.name, DevMemSpace::name()) == 0;
}

void Check(const State &state, std::uint64_t additional, const char *purpose) {
  const auto used = state.current.total;
  if (state.budget && (state.headroom > state.budget ||
      used > state.budget-state.headroom ||
      additional > state.budget-state.headroom-used)) {
    std::ostringstream error;
    error << "### FATAL ERROR: fast-electron aggregate memory budget exceeded rank="
          << global_variable::my_rank << " purpose=" << purpose
          << " tracked_bytes=" << used << " additional_bytes=" << additional
          << " headroom_bytes=" << state.headroom << " budget_bytes=" << state.budget
          << "; no particles or energy were discarded";
    Kokkos::abort(error.str().c_str());
  }
}

void Allocated(Kokkos::Tools::SpaceHandle space, const char *label, const void *ptr,
               std::uint64_t bytes) {
  auto &state = Data();
  if (state.previous_allocate) state.previous_allocate(space, label, ptr, bytes);
  if (!bytes || !ExecutionSpace(space)) return;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.allocations.count(ptr)) Kokkos::abort("Duplicate tracked device allocation");
  // Kokkos notifies after allocation. Explicit Preflight guards particle growth
  // before allocation; this check also enforces the budget on all other modules.
  Check(state, bytes, label);
  const auto kind = Category(label);
  state.allocations.emplace(ptr, Allocation{bytes, kind});
  auto &s = state.current;
  s.total += bytes;
  CategoryBytes(s, kind) += bytes;
  ++s.allocations;
  s.allocated_bytes += bytes;
  if (s.total > s.peak) {
    s.peak = s.total;
    if (state.report) {
      std::ostringstream out;
      out << std::setprecision(std::numeric_limits<Real>::max_digits10)
          << "# fast_electrons_allocation rank=" << global_variable::my_rank
          << " time=" << state.time << " phase=" << state.phase
          << " live_known=" << state.live_known << " live_packets=" << state.live
          << " total_bytes=" << s.total << " packet_bytes=" << s.packets
          << " source_bytes=" << s.sources << " mesh_bytes=" << s.mesh
          << " runtime_bytes=" << s.runtime << " diagnostic_bytes=" << s.diagnostics
          << " request_bytes=" << bytes << " label=" << std::quoted(label);
      Log(state, out.str());
    }
  }
}
void Freed(Kokkos::Tools::SpaceHandle space, const char *label, const void *ptr,
           std::uint64_t bytes) {
  auto &state = Data();
  if (state.previous_free) state.previous_free(space, label, ptr, bytes);
  if (!bytes || !ExecutionSpace(space)) return;
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto found = state.allocations.find(ptr);
  if (found == state.allocations.end()) {
    // The execution space can allocate startup scratch before tools are installed.
    state.current.untracked_frees += bytes;
    return;
  }
  state.current.total -= found->second.bytes;
  CategoryBytes(state.current, found->second.kind) -= found->second.bytes;
  state.allocations.erase(found);
}
std::uint64_t Mib(ParameterInput *pin, const char *name) {
  const Real value = pin->GetOrAddReal("fast_electrons", name, 0.0);
  if (!std::isfinite(value) || value < 0.0 || value >=
      static_cast<Real>(std::numeric_limits<std::uint64_t>::max()/1048576)) {
    Kokkos::abort((std::string("### FATAL ERROR: invalid fast_electrons/")+name).c_str());
  }
  return static_cast<std::uint64_t>(value*1048576);
}
}  // namespace

void Initialize(ParameterInput *pin) {
  if (!pin->DoesBlockExist("fast_electrons")) return;
  auto &state = Data();
  state.report = pin->GetOrAddBoolean("fast_electrons", "memory_diagnostics", false);
  state.budget = Mib(pin, "memory_budget_mib");
  state.headroom = Mib(pin, "memory_headroom_mib");
  state.enabled = state.report || state.budget || state.headroom;
  if (!state.enabled) return;
  Check(state, 0, "initialization");
  const auto callbacks = Kokkos::Tools::Experimental::get_callbacks();
  state.previous_allocate = callbacks.allocate_data;
  state.previous_free = callbacks.deallocate_data;
  Kokkos::Tools::Experimental::set_allocate_data_callback(Allocated);
  Kokkos::Tools::Experimental::set_deallocate_data_callback(Freed);
}
bool Reporting() { return Data().report; }
void Report(const std::string &record) {
  auto &s = Data();
  if (!s.report) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.log.is_open()) {
    s.log.open("fast_electron_memory.rank"+std::to_string(global_variable::my_rank)+".log");
    if (!s.log) Kokkos::abort("### FATAL ERROR: opening fast-electron memory log");
    for (const auto &line : s.pending) s.log << line << '\n';
    s.pending.clear();
  }
  Log(s, record);
  if (s.phase == "advance_end") {
    s.log.flush();
    if (!s.log) Kokkos::abort("### FATAL ERROR: flushing fast-electron memory log");
  }
}
void Context(const char *phase, std::uint64_t live, bool known) {
  auto &s = Data();
  if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  s.phase = phase; s.live = live; s.live_known = known;
}
void Time(Real time) {
  auto &s = Data();
  if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  s.time = time;
}
Snapshot Get() {
  auto &s = Data();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.current;
}
void Preflight(std::uint64_t additional_bytes, const char *purpose) {
  auto &s = Data();
  if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  Check(s, additional_bytes, purpose);
  if (!(s.budget || s.headroom)) return;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  std::size_t available = 0, total = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  const bool success = cudaMemGetInfo(&available, &total) == cudaSuccess;
#else
  const bool success = hipMemGetInfo(&available, &total) == hipSuccess;
#endif
  if (!success) Kokkos::abort("### FATAL ERROR: cannot query device memory for budget");
  // Driver/context, other processes, and allocator metadata are included in free
  // memory. Reserve a small allocation-header allowance in addition to user headroom.
  if (s.headroom > available || available-s.headroom < 256 ||
      additional_bytes > available-s.headroom-256) {
    std::ostringstream error;
    error << "### FATAL ERROR: insufficient device memory rank="
          << global_variable::my_rank << " purpose=" << purpose
          << " free_bytes=" << available << " request_bytes=" << additional_bytes
          << " headroom_bytes=" << s.headroom;
    Kokkos::abort(error.str().c_str());
  }
#endif
}
}  // namespace fast_electrons::memory
