#include "two_temperature/thermal_radiation_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <ucontext.h>
#endif

namespace two_temperature {
namespace {
thread_local RadiationGroupBatch *active_batch = nullptr;
}

struct RadiationGroupBatch::State {
  struct Lane {
#if defined(__linux__)
    ucontext_t context{};
    void *mapping = MAP_FAILED;
    std::size_t mapping_size = 0;
    ~Lane() { if (mapping != MAP_FAILED) munmap(mapping, mapping_size); }
#endif
    std::function<void()> work;
    void *request = nullptr;
    Execute execute = nullptr;
    Kind kind = Kind::Halo;
    bool done = false;
    std::exception_ptr error;
  };
#if defined(__linux__)
  ucontext_t scheduler{};
#endif
  std::vector<std::unique_ptr<Lane>> lanes;
  int current = -1;
  Counts counts;
  bool timing = false;

  static void Enter(int index) {
    auto &s = *active_batch->state_;
    auto &lane = *s.lanes[index];
    try { lane.work(); } catch (...) { lane.error = std::current_exception(); }
    lane.done = true;
    // Returning switches to uc_link, the scheduler; no exception crosses stacks.
  }
};

RadiationGroupBatch::RadiationGroupBatch(bool timing) : state_(new State) {
  state_->timing = timing;
}
RadiationGroupBatch::~RadiationGroupBatch() = default;
bool RadiationGroupBatch::Supported() {
#if defined(__linux__)
  return true;
#else
  return false;
#endif
}
RadiationGroupBatch *RadiationGroupBatch::Active() {
  return active_batch != nullptr && active_batch->state_->current >= 0 ? active_batch : nullptr;
}
const RadiationGroupBatch::Counts &RadiationGroupBatch::counts() const {
  return state_->counts;
}

void RadiationGroupBatch::Run(const std::vector<std::function<void()>> &work) {
#if defined(__linux__)
  if (active_batch != nullptr) throw std::runtime_error("Nested radiation group batch");
  auto &s = *state_;
  s.lanes.clear();
  for (std::size_t i = 0; i < work.size(); ++i) {
    auto lane = std::make_unique<State::Lane>();
    lane->work = work[i];
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    constexpr std::size_t stack_bytes = 2*1024*1024;
    lane->mapping_size = stack_bytes+2*page;
    lane->mapping = mmap(nullptr, lane->mapping_size, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (lane->mapping == MAP_FAILED ||
        mprotect(static_cast<char *>(lane->mapping)+page, stack_bytes,
                 PROT_READ | PROT_WRITE) != 0 || getcontext(&lane->context) != 0) {
      throw std::runtime_error("Cannot allocate radiation group stack");
    }
    lane->context.uc_stack.ss_sp = static_cast<char *>(lane->mapping)+page;
    lane->context.uc_stack.ss_size = stack_bytes;
    lane->context.uc_link = &s.scheduler;
    makecontext(&lane->context, reinterpret_cast<void (*)()>(&State::Enter), 1,
                static_cast<int>(i));
    s.lanes.push_back(std::move(lane));
  }
  active_batch = this;
  try {
    while (true) {
      bool unfinished = false;
      for (std::size_t i = 0; i < s.lanes.size(); ++i) {
        auto &lane = *s.lanes[i];
        if (!lane.done && lane.request == nullptr) {
          s.current = static_cast<int>(i);
          if (swapcontext(&s.scheduler, &lane.context) != 0) {
            throw std::runtime_error("Cannot resume radiation group");
          }
          s.current = -1;
        }
        if (lane.error) std::rethrow_exception(lane.error);
        unfinished = unfinished || !lane.done;
      }
      if (!unfinished) break;
      State::Lane *first = nullptr;
      std::vector<void *> requests;
      std::vector<State::Lane *> selected;
      // Advance cheap host reductions and halos before flushing an unpaired AMG
      // call. This lets the other lane reach the same expensive stage even when
      // its preceding Krylov iteration took a different branch.
      for (auto &lane : s.lanes) {
        if (lane->request == nullptr) continue;
        if (first == nullptr || static_cast<int>(lane->kind) > static_cast<int>(first->kind)) {
          first = lane.get();
        }
      }
      if (first == nullptr) throw std::runtime_error("Radiation batch made no progress");
      for (auto &lane : s.lanes) {
        if (lane->request == nullptr) continue;
        if (lane->kind == first->kind && lane->execute == first->execute) {
          requests.push_back(lane->request);
          selected.push_back(lane.get());
        }
      }
      const int k = static_cast<int>(first->kind);
      s.counts.requests[k] += requests.size();
      ++s.counts.batches[k];
      if (requests.size() > 1) ++s.counts.paired[k];
      // Setup already synchronizes its producer and the completed HYPRE work.
      // An extra entry fence would prevent preceding GPU work from overlapping
      // the callback's host graph preparation. Time each callback including its
      // existing waits, without adding synchronization to the scheduler.
      using Clock = std::chrono::steady_clock;
      Clock::time_point begin;
      if (s.timing) {
        begin = Clock::now();
      }
      first->execute(requests);
      if (s.timing) {
        s.counts.seconds[k] += std::chrono::duration<double>(Clock::now()-begin).count();
      }
      for (auto *lane : selected) lane->request = nullptr;
    }
  } catch (...) {
    s.current = -1;
    active_batch = nullptr;
    throw;
  }
  active_batch = nullptr;
  s.lanes.clear();
#else
  throw std::runtime_error("Radiation group batching requires Linux ucontext");
#endif
}

void RadiationGroupBatch::Submit(Kind kind, void *request, Execute execute) {
#if defined(__linux__)
  auto &s = *state_;
  if (Active() != this || request == nullptr) {
    throw std::runtime_error("Invalid radiation batch submission");
  }
  auto &lane = *s.lanes[s.current];
  lane.kind = kind;
  lane.request = request;
  lane.execute = execute;
  if (swapcontext(&lane.context, &s.scheduler) != 0) {
    throw std::runtime_error("Cannot suspend radiation group");
  }
#endif
}

#if MPI_PARALLEL_ENABLED
namespace {
struct Reduction {
  const void *send;
  void *recv;
  int count, width, result = MPI_SUCCESS;
  MPI_Datatype type;
  MPI_Op op;
  MPI_Comm comm;

  static void Execute(const std::vector<void *> &requests) {
    std::vector<bool> done(requests.size(), false);
    for (std::size_t i = 0; i < requests.size(); ++i) {
      if (done[i]) continue;
      auto &first = *static_cast<Reduction *>(requests[i]);
      std::vector<Reduction *> same;
      std::size_t count = 0;
      for (std::size_t j = i; j < requests.size(); ++j) {
        auto &r = *static_cast<Reduction *>(requests[j]);
        if (!done[j] && r.type == first.type && r.op == first.op && r.comm == first.comm &&
            count+r.count <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          same.push_back(&r); count += r.count; done[j] = true;
        }
      }
      if (same.size() == 1) {
        first.result = MPI_Allreduce(first.send, first.recv, first.count,
                                     first.type, first.op, first.comm);
        continue;
      }
      std::vector<unsigned char> buffer(count*first.width);
      std::size_t offset = 0;
      for (auto *r : same) {
        const std::size_t bytes = static_cast<std::size_t>(r->count)*r->width;
        std::memcpy(buffer.data()+offset, r->send == MPI_IN_PLACE ? r->recv : r->send, bytes);
        offset += bytes;
      }
      const int result = MPI_Allreduce(MPI_IN_PLACE, buffer.data(), static_cast<int>(count),
                                       first.type, first.op, first.comm);
      offset = 0;
      for (auto *r : same) {
        const std::size_t bytes = static_cast<std::size_t>(r->count)*r->width;
        std::memcpy(r->recv, buffer.data()+offset, bytes);
        r->result = result;
        offset += bytes;
      }
    }
  }
};
}

int RadiationBatchAllreduce(const void *send, void *recv, int count,
                            MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  auto *batch = RadiationGroupBatch::Active();
  if (batch == nullptr || count < 0 || comm != MPI_COMM_WORLD ||
      (datatype != MPI_DOUBLE && datatype != MPI_FLOAT && datatype != MPI_INT) ||
      (op != MPI_SUM && op != MPI_MIN && op != MPI_MAX)) {
    return MPI_Allreduce(send, recv, count, datatype, op, comm);
  }
  int width = 0;
  MPI_Type_size(datatype, &width);
  Reduction request{send, recv, count, width, MPI_SUCCESS, datatype, op, comm};
  batch->Submit(RadiationGroupBatch::Kind::Reduction, &request, &Reduction::Execute);
  return request.result;
}
#endif
}  // namespace two_temperature
