// Allocation-only Kokkos tool. It adds no kernel callbacks or device fences.
// Build: c++ -O2 -std=c++17 -shared -fPIC -I kokkos/core/src
//        tst/benchmarks/kokkos_memory_profile.cpp -o memory_profile.so
#include "impl/Kokkos_Profiling_C_Interface.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

namespace {
struct Allocation { std::string space, label; uint64_t bytes; };
struct Counter { uint64_t live=0, peak=0, allocations=0; };
std::map<std::pair<std::string, const void*>, Allocation> live;
std::map<std::string, Counter> spaces;
std::map<std::pair<std::string, std::string>, Counter> labels;
std::map<std::string, std::vector<Allocation>> peak_allocations;
std::mutex guard;
std::chrono::steady_clock::time_point started;
std::string output;
std::ofstream events;
int rank=0;
uint64_t unmatched_frees=0;
std::vector<Allocation> untracked_deallocations;
uint64_t duplicate_allocations=0;
std::map<std::tuple<std::string,std::string,uint64_t,uint64_t>,uint64_t> duplicates;

std::string quote(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c=='"' || c=='\\') out << '\\' << c;
    else if (c=='\n') out << "\\n";
    else if (c=='\r') out << "\\r";
    else if (c=='\t') out << "\\t";
    else if (c>=32) out << c;
  }
  out << '"';
  return out.str();
}

void write_packet_event(const char *event, const Allocation &a) {
  if (!events.is_open() || a.space != "Cuda" ||
      a.label.find("packet") == std::string::npos) return;
  uint64_t packet_bytes=0;
  for (const auto &[key,c] : labels) {
    if (key.first=="Cuda" && key.second.find("packet")!=std::string::npos) {
      packet_bytes+=c.live;
    }
  }
  events << std::setprecision(12) << std::chrono::duration<double>(
      std::chrono::steady_clock::now()-started).count() << ',' << event << ','
      << quote(a.label) << ',' << a.bytes << ',' << spaces["Cuda"].live << ','
      << packet_bytes << '\n';
}

void write_report(bool final) {
  if (output.empty()) return;
  const auto temporary=output+".tmp";
  std::ofstream out(temporary);
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  out << "{\"rank\":" << rank << ",\"pid\":" << getpid()
      << ",\"final\":" << (final?"true":"false")
      << ",\"elapsed_s\":" << std::chrono::duration<double>(
          std::chrono::steady_clock::now()-started).count()
      << ",\"host_max_rss_kib\":" << usage.ru_maxrss
      << ",\"unmatched_frees\":" << unmatched_frees
      << ",\"duplicate_allocations\":" << duplicate_allocations
      << ",\"untracked_deallocations\":[";
  bool first_untracked=true;
  for (const auto &a : untracked_deallocations) {
    if (!first_untracked) out << ',';
    first_untracked=false;
    out << "{\"space\":" << quote(a.space) << ",\"label\":" << quote(a.label)
        << ",\"bytes\":" << a.bytes << '}';
  }
  out << "],\"duplicates\":[";
  bool first_duplicate=true;
  for (const auto &[key,count] : duplicates) {
    if (!first_duplicate) out << ',';
    first_duplicate=false;
    out << "{\"label\":" << quote(std::get<0>(key)) << ",\"previous_label\":"
        << quote(std::get<1>(key)) << ",\"bytes\":" << std::get<2>(key)
        << ",\"previous_bytes\":" << std::get<3>(key) << ",\"count\":" << count << '}';
  }
  out << "],\"spaces\":[";
  bool first=true;
  for (const auto &[space,c] : spaces) {
    if (!first) out << ',';
    first=false;
    out << "{\"space\":" << quote(space) << ",\"live_bytes\":" << c.live
        << ",\"peak_bytes\":" << c.peak << ",\"allocations\":" << c.allocations
        << ",\"at_peak\":[";
    bool first_allocation=true;
    for (const auto &a : peak_allocations[space]) {
      if (!first_allocation) out << ',';
      first_allocation=false;
      out << "{\"label\":" << quote(a.label) << ",\"bytes\":" << a.bytes << '}';
    }
    out << "]}";
  }
  out << "],\"labels\":[";
  first=true;
  for (const auto &[key,c] : labels) {
    if (!first) out << ',';
    first=false;
    out << "{\"space\":" << quote(key.first) << ",\"label\":" << quote(key.second)
        << ",\"live_bytes\":" << c.live << ",\"peak_bytes\":" << c.peak
        << ",\"allocations\":" << c.allocations << '}';
  }
  out << "]}\n";
  out.close();
  std::rename(temporary.c_str(), output.c_str());
}
}  // namespace

extern "C" void kokkosp_init_library(const int, const uint64_t, const uint32_t,
                                     Kokkos_Profiling_KokkosPDeviceInfo*) {
  started=std::chrono::steady_clock::now();
  if (const char *value=std::getenv("OMPI_COMM_WORLD_RANK")) rank=std::atoi(value);
  const char *directory=std::getenv("KOKKOS_MEMORY_OUTPUT");
  output=std::string(directory?directory:".")+"/memory-rank"+std::to_string(rank)+".json";
  if (const char *trace=std::getenv("KOKKOS_MEMORY_TRACE"); trace && std::string(trace)=="1") {
    events.open(output+".events.csv");
    events << "elapsed_s,event,label,bytes,cuda_live_bytes,packet_live_bytes\n";
  }
}

extern "C" void kokkosp_request_tool_settings(const uint32_t,
                                              Kokkos_Tools_ToolSettings *settings) {
  settings->requires_global_fencing=false;
}

extern "C" void kokkosp_allocate_data(const Kokkos_Profiling_SpaceHandle space,
                                      const char *label, const void *ptr,
                                      const uint64_t bytes) {
  // CUDA permits repeated null, zero-byte team scratch allocations.
  if (bytes==0) return;
  std::lock_guard<std::mutex> lock(guard);
  const std::string name=space.name, tag=label?label:"";
  if (live.count({name,ptr})) {
    ++duplicate_allocations;
    const auto &old=live.at({name,ptr});
    ++duplicates[{tag,old.label,bytes,old.bytes}];
  }
  live[{name,ptr}]={name,tag,bytes};
  auto &s=spaces[name];
  auto &l=labels[{name,tag}];
  s.live+=bytes; ++s.allocations;
  l.live+=bytes; ++l.allocations; l.peak=std::max(l.peak,l.live);
  if (s.live>s.peak) {
    s.peak=s.live;
    auto &snapshot=peak_allocations[name];
    snapshot.clear();
    for (const auto &[key,a] : live) if (a.space==name) snapshot.push_back(a);
    write_report(false);
  }
  write_packet_event("allocate", {name,tag,bytes});
}

extern "C" void kokkosp_deallocate_data(const Kokkos_Profiling_SpaceHandle space,
                                        const char *label, const void *ptr,
                                        const uint64_t bytes) {
  if (bytes==0) return;
  std::lock_guard<std::mutex> lock(guard);
  auto found=live.find({space.name,ptr});
  if (found==live.end()) {
    ++unmatched_frees;
    untracked_deallocations.push_back({space.name,label?label:"",bytes});
    return;
  }
  const auto a=found->second;
  spaces[a.space].live-=a.bytes;
  labels[{a.space,a.label}].live-=a.bytes;
  live.erase(found);
  write_packet_event("deallocate", a);
}

extern "C" void kokkosp_finalize_library() {
  std::lock_guard<std::mutex> lock(guard);
  write_report(true);
  if (events.is_open()) events.flush();
}
