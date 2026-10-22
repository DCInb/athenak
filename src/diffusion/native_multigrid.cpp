//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! Native unsmoothed aggregation: A x = m x + sum_j g_ij (x_i-x_j).
//! The fine level can eliminate an independent set exactly; remaining levels
//! use constant interpolation and Galerkin aggregation. Assemble coarse masses
//! and crossing edges separately, avoiding cancellation of internal fluxes.
//! Two-stage Chebyshev-Jacobi smoothing on either side of a symmetric coarse
//! correction gives a fixed SPD preconditioner. No vendor sparse library is used.

#include "diffusion/native_multigrid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Sort.hpp>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace diffusion {
namespace {
[[noreturn]] void Fail(const std::string &message) {
  std::cerr << "### FATAL ERROR in native multigrid: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

template <typename T>
DvceArray1D<T> Upload(const std::vector<T> &source, const char *name) {
  DvceArray1D<T> result(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        std::string(name)), source.size());
  Kokkos::View<const T *, HostMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), source.size());
  Kokkos::deep_copy(result, host);
  return result;
}

// Keep low parts only in the bounded direct root (at most 64 rows). A stiff
// inverse has nearly equal columns; rounding those columns to one double
// destroys their difference before the root matrix-vector product even starts.
// TwoSum/FMA building blocks: Hida, Li, Bailey, "Library for Double-Double and
// Quad-Double Arithmetic", https://www.davidhbailey.com/dhbpapers/qd.pdf.
struct RootScalar {
  double hi, lo;
  KOKKOS_INLINE_FUNCTION RootScalar(double h = 0.0, double l = 0.0) : hi(h), lo(l) {}
  KOKKOS_INLINE_FUNCTION static RootScalar Sum(double a, double b) {
    const double s = a+b, v = s-a;
    return RootScalar(s, (a-(s-v))+(b-v));
  }
  KOKKOS_INLINE_FUNCTION RootScalar operator+(RootScalar b) const {
    const RootScalar s = Sum(hi, b.hi), t = Sum(lo, b.lo);
    const RootScalar u = Sum(s.hi, s.lo+t.hi);
    return Sum(u.hi, u.lo+t.lo);
  }
  KOKKOS_INLINE_FUNCTION RootScalar operator-() const { return RootScalar(-hi, -lo); }
  KOKKOS_INLINE_FUNCTION RootScalar operator-(RootScalar b) const { return *this+(-b); }
  KOKKOS_INLINE_FUNCTION RootScalar operator*(RootScalar b) const {
    const double p = hi*b.hi;
    const double e = Kokkos::fma(hi, b.hi, -p)+hi*b.lo+lo*b.hi;
    return Sum(p, e+lo*b.lo);
  }
  KOKKOS_INLINE_FUNCTION RootScalar operator/(RootScalar b) const {
    const double q = hi/b.hi;
    const RootScalar r = *this-b*RootScalar(q);
    const double q2 = r.hi/b.hi;
    const RootScalar r2 = r-b*RootScalar(q2);
    return RootScalar(q)+RootScalar(q2)+RootScalar(r2.hi/b.hi);
  }
  KOKKOS_INLINE_FUNCTION RootScalar &operator+=(RootScalar b) { return *this=*this+b; }
  KOKKOS_INLINE_FUNCTION RootScalar &operator-=(RootScalar b) { return *this=*this-b; }
};
static_assert(sizeof(RootScalar) == 2*sizeof(double), "Root MPI layout must be two doubles");
static_assert(std::is_trivially_copyable<RootScalar>::value, "Root scalar must be copyable");
KOKKOS_INLINE_FUNCTION RootScalar RootSqrt(RootScalar a) {
  const double r = sqrt(a.hi);
  return RootScalar(r)+(a-RootScalar(r)*RootScalar(r))/RootScalar(2.0*r);
}

using Policy = Kokkos::RangePolicy<DevExeSpace>;
using TeamPolicy = Kokkos::TeamPolicy<DevExeSpace>;
using TeamMember = TeamPolicy::member_type;
constexpr bool kDeviceTeams =
    !std::is_same<DevExeSpace, Kokkos::DefaultHostExecutionSpace>::value;
constexpr int kRootSize = 64;
constexpr int kReplicatedSize = 2048;
// Rank-local aggregation can stall just above the normal agglomeration size.
// Bound the exceptional root, then finish its coarsening across rank boundaries.
constexpr int kStalledReplicatedSize = 32768;
constexpr int kMaxLevels = 24;
constexpr int kAggregateSize = 8;
constexpr double kOmega = 2.0/3.0;
// For an M-matrix graph, spectrum(D^-1 A) lies in (0,2]. A degree-two
// Chebyshev residual polynomial on [0.3,2] has |q(lambda)|<1 on (0,2],
// including below its smoothing interval, so its combined correction is SPD.
constexpr double kChebyshevFirst = 1.0/(1.15-0.85*0.7071067811865475244);
constexpr double kChebyshevSecond = 1.0/(1.15+0.85*0.7071067811865475244);
// Piecewise-constant interpolation under-corrects smooth error. A fixed
// positive weight strengthens the coarse correction without introducing a
// residual-dependent (nonlinear) preconditioner. The symmetric cycle is
// B = Sbar + (I-S^T A) P (1.5 Bc) P^T (I-A S), hence remains SPD.
constexpr double kCoarseCorrection = 1.5;

struct ProfileRegion {
  explicit ProfileRegion(const std::string &name) { Kokkos::Profiling::pushRegion(name); }
  ~ProfileRegion() { Kokkos::Profiling::popRegion(); }
};
}  // namespace

struct NativeMultigrid::State {
  struct Level {
    int n = 0, first = 0, global = 0, ghosts = 0;
    std::vector<int> offsets, columns, counts, starts;
    std::vector<int> recv_counts, recv_offsets, send_counts, send_offsets, send_ids;
    DvceArray1D<int> row, col, send_index, parent, members, member_offsets;
    DvceArray1D<int> edge_terms, edge_offsets;
    DvceArray1D<unsigned char> strength, boundary_mass;
    bool strength_ready = false, schur = false, coarsening_stalled = false;
    DvceArray1D<int> retained, edge_right, edge_pivot;
    DvceArray1D<double> g, mass, diagonal, rhs, x, next, send;
    HostArray1D<double> host_send, host_recv;
    DvceArray2D<RootScalar> factor, inverse;
    DvceArray1D<double> root_rhs;
    HostArray1D<double> host_root_rhs;
    std::unique_ptr<State> replica;
    std::vector<int> matrix_counts, matrix_starts;
  };
  std::vector<std::unique_ptr<Level>> levels;
  int rank = 0, ranks = 1;
  bool device_mpi = false, ready = false;
  std::size_t max_bytes = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  NativeMultigrid::HaloWait halo_wait = nullptr;
#endif

  ~State() {
#if MPI_PARALLEL_ENABLED
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
  }

  std::size_t Bytes() const {
    std::size_t bytes = 0;
    for (const auto &p : levels) {
      const auto &l = *p;
      bytes += sizeof(int)*(l.row.size()+l.col.size()+l.send_index.size()+
          l.parent.size()+l.members.size()+l.member_offsets.size()+
          l.edge_terms.size()+l.edge_offsets.size()+
          l.retained.size()+l.edge_right.size()+l.edge_pivot.size());
      bytes += (l.strength.size()+l.boundary_mass.size())*sizeof(unsigned char);
      bytes += sizeof(double)*(l.g.size()+l.mass.size()+l.diagonal.size()+
          l.rhs.size()+l.x.size()+l.next.size()+l.send.size()+
          l.root_rhs.size());
      bytes += sizeof(RootScalar)*(l.factor.size()+l.inverse.size());
      if (l.replica) bytes += l.replica->Bytes();
    }
    return bytes;
  }

  int Depth() const {
    if (levels.empty()) return 0;
    return levels.size()+(levels.back()->replica ? levels.back()->replica->Depth()-1 : 0);
  }

  void Require(bool valid, const char *message) const {
    int failed = valid ? 0 : 1;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm);
#endif
    if (failed) Fail(message);
  }

  void Budget(std::size_t extra) const {
    const std::size_t used = Bytes();
    Require(used <= max_bytes && extra <= max_bytes-used,
            "native_mg_max_mib is too small for the hierarchy");
  }

  void Ownership(Level &l) {
    l.counts.assign(ranks, 0);
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&l.n, 1, MPI_INT, l.counts.data(), 1, MPI_INT, comm);
#else
    l.counts[0] = l.n;
#endif
    l.starts.assign(ranks+1, 0);
    for (int r = 0; r < ranks; ++r) {
      if (l.counts[r] > std::numeric_limits<int>::max()-l.starts[r]) {
        Fail("global graph exceeds 32-bit indexing");
      }
      l.starts[r+1] = l.starts[r]+l.counts[r];
    }
    l.first = l.starts[rank];
    l.global = l.starts.back();
  }

  void Allocate(Level &l) {
    ProfileRegion region("native_mg_allocate");
    // Remote columns are sorted by owner and then by global index. Only these
    // halo values cross ranks; fine-grid vectors/matrices are never replicated.
    std::vector<int> remote;
    if (ranks > 1) {
      for (int c : l.columns) if (c < l.first || c >= l.first+l.n) remote.push_back(c);
    }
    std::sort(remote.begin(), remote.end());
    remote.erase(std::unique(remote.begin(), remote.end()), remote.end());
    l.ghosts = remote.size();
    l.recv_counts.assign(ranks, 0);
    l.recv_offsets.assign(ranks, 0);
    l.send_counts.assign(ranks, 0);
    l.send_offsets.assign(ranks, 0);
    for (int c : remote) {
      const int owner = std::upper_bound(l.starts.begin(), l.starts.end(), c)-
                        l.starts.begin()-1;
      ++l.recv_counts[owner];
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoall(l.recv_counts.data(), 1, MPI_INT,
                  l.send_counts.data(), 1, MPI_INT, comm);
#endif
    for (int r = 1; r < ranks; ++r) {
      l.recv_offsets[r] = l.recv_offsets[r-1]+l.recv_counts[r-1];
      l.send_offsets[r] = l.send_offsets[r-1]+l.send_counts[r-1];
    }
    l.send_ids.resize(l.send_offsets.back()+l.send_counts.back());
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(remote.data(), l.recv_counts.data(), l.recv_offsets.data(), MPI_INT,
                  l.send_ids.data(), l.send_counts.data(), l.send_offsets.data(),
                  MPI_INT, comm);
#endif
    for (int &c : l.send_ids) c -= l.first;
    std::vector<int> columns;
    if (ranks > 1) {
      columns.reserve(l.columns.size());
      for (int c : l.columns) {
        columns.push_back(c >= l.first && c < l.first+l.n ? c-l.first :
            l.n+std::lower_bound(remote.begin(), remote.end(), c)-remote.begin());
      }
    }
    const auto &device_columns = ranks == 1 ? l.columns : columns;
    const std::size_t n = l.n, e = device_columns.size(), h = l.ghosts;
    const std::size_t s = l.send_ids.size();
    Budget(sizeof(int)*(n+1+e+s)+sizeof(double)*(e+5*n+2*h+s));
    l.row = Upload(l.offsets, "native mg rows");
    l.col = Upload(device_columns, "native mg columns");
    l.send_index = Upload(l.send_ids, "native mg send indices");
    l.g = DvceArray1D<double>("native mg conductance", e);
    l.mass = DvceArray1D<double>("native mg mass", n);
    l.diagonal = DvceArray1D<double>("native mg diagonal", n);
    l.rhs = DvceArray1D<double>("native mg rhs", n);
    l.x = DvceArray1D<double>("native mg solution", n+h);
    l.next = DvceArray1D<double>("native mg smooth scratch", n+h);
    l.send = DvceArray1D<double>("native mg send", s);
    if (!device_mpi) {
      l.host_send = HostArray1D<double>("native mg host send", s);
      l.host_recv = HostArray1D<double>("native mg host receive", h);
    }
  }

  void Halo(Level &l) {
#if MPI_PARALLEL_ENABLED
    if (l.ghosts == 0 && l.send_ids.empty()) {
      // Empty ranks must yield at the same distributed level as other ranks;
      // otherwise they can enter a later collective while a peer is suspended.
      if (halo_wait) halo_wait(0, nullptr);
      return;
    }
    auto x = l.x;
    auto ids = l.send_index;
    auto send = l.send;
    Kokkos::parallel_for("native_mg_pack_halo", Policy(0, send.extent_int(0)),
        KOKKOS_LAMBDA(int q) { send(q) = x(ids(q)); });
    auto recv = Kokkos::subview(x, std::make_pair(l.n, l.n+l.ghosts));
    double *send_ptr = send.data(), *recv_ptr = recv.data();
    if (!device_mpi) {
      Kokkos::deep_copy(l.host_send, send);
      send_ptr = l.host_send.data();
      recv_ptr = l.host_recv.data();
    } else {
      DevExeSpace().fence("native mg before MPI");
    }
    std::vector<MPI_Request> requests;
    requests.reserve(2*ranks);
    for (int r = 0; r < ranks; ++r) if (l.recv_counts[r]) {
      MPI_Request request;
      MPI_Irecv(recv_ptr+l.recv_offsets[r], l.recv_counts[r], MPI_DOUBLE,
                r, 0, comm, &request);
      requests.push_back(request);
    }
    for (int r = 0; r < ranks; ++r) if (l.send_counts[r]) {
      MPI_Request request;
      MPI_Isend(send_ptr+l.send_offsets[r], l.send_counts[r], MPI_DOUBLE,
                r, 0, comm, &request);
      requests.push_back(request);
    }
    if (halo_wait) halo_wait(requests.size(), requests.data());
    else MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    if (!device_mpi) Kokkos::deep_copy(recv, l.host_recv);
#endif
  }

  void Diagonal(Level &l) {
    auto row = l.row;
    auto g = l.g;
    auto mass = l.mass;
    auto diagonal = l.diagonal;
    int invalid = 0;
    Kokkos::parallel_reduce("native_mg_validate", Policy(0, l.n),
    KOKKOS_LAMBDA(int i, int &bad) {
      double d = mass(i);
      if (!(d > 0.0) || !Kokkos::isfinite(d)) ++bad;
      for (int q = row(i); q < row(i+1); ++q) {
        const double a = g(q);
        if (!(a >= 0.0) || !Kokkos::isfinite(a)) ++bad;
        d += a;
      }
      if (!Kokkos::isfinite(d)) ++bad;
      diagonal(i) = d;
    }, invalid);
    Require(invalid == 0, "nonpositive mass, negative conductance, or nonfinite matrix");
  }

  void DeviceCoarseGraph(Level &fine, Level &coarse, const std::vector<int> &incoming) {
    // Sort (aggregate row, aggregate column) keys on the execution space. This
    // replaces millions of small host sorts. Only the unique integer graph
    // returns to the host for ownership/halo construction; numerical data stays
    // on device. Equal keys give the exact segmented Galerkin edge sums.
    ProfileRegion region("native_mg_coarse_graph");
    using Key = std::uint64_t;
    const Key invalid = std::numeric_limits<Key>::max();
    const int edges = fine.g.extent_int(0), owned = fine.n, first = coarse.first;
    auto parent = fine.parent;
    auto remote = Upload(incoming, "native mg remote aggregate");
    auto row = fine.row;
    auto col = fine.col;
    DvceArray1D<Key> keys("native mg aggregate keys", edges);
    DvceArray1D<int> terms("native mg sorted terms", edges);
    Kokkos::parallel_for("native_mg_edge_keys", Policy(0, owned),
    KOKKOS_LAMBDA(int i) {
      const int a = parent(i);
      for (int q = row(i); q < row(i+1); ++q) {
        const int j = col(q);
        const int b = j < owned ? (parent(j) < 0 ? -1 : first+parent(j)) : remote(j-owned);
        keys(q) = a < 0 || b < 0 || b == first+a ? invalid :
            (static_cast<Key>(a) << 32) | b;
        terms(q) = q;
      }
    });
    Kokkos::Experimental::sort_by_key(DevExeSpace(), keys, terms);
    int unique = 0, retained = 0;
    Kokkos::parallel_reduce("native_mg_count_coarse_edges", Policy(0, edges),
    KOKKOS_LAMBDA(int q, int &count, int &kept) {
      if (keys(q) != invalid) {
        ++kept;
        if (q == 0 || keys(q) != keys(q-1)) ++count;
      }
    }, unique, retained);
    Budget(sizeof(int)*(static_cast<std::size_t>(retained)+unique+1));
    DvceArray1D<Key> unique_keys("native mg unique keys", unique);
    fine.edge_offsets = DvceArray1D<int>("native mg edge offsets", unique+1);
    fine.edge_terms = DvceArray1D<int>("native mg edge terms", retained);
    auto offsets = fine.edge_offsets;
    Kokkos::parallel_scan("native_mg_coarse_edge_offsets", Policy(0, retained+1),
    KOKKOS_LAMBDA(int q, int &index, bool final) {
      if (q == retained) {
        if (final) offsets(index) = retained;
      } else if (q == 0 || keys(q) != keys(q-1)) {
        if (final) { unique_keys(index) = keys(q); offsets(index) = q; }
        ++index;
      }
    });
    Kokkos::deep_copy(fine.edge_terms,
                      Kokkos::subview(terms, std::make_pair(0, retained)));
    auto host_keys = Kokkos::create_mirror_view_and_copy(HostMemSpace(), unique_keys);
    coarse.offsets.assign(coarse.n+1, 0);
    coarse.columns.resize(unique);
    for (int q = 0; q < unique; ++q) {
      ++coarse.offsets[(host_keys(q) >> 32)+1];
      coarse.columns[q] = static_cast<int>(host_keys(q) & 0xffffffffULL);
    }
    for (int a = 0; a < coarse.n; ++a) coarse.offsets[a+1] += coarse.offsets[a];
  }

  bool UpdateStrength(Level &l) {
    const bool initialize = !l.strength_ready;
    if (initialize) {
      Budget(l.g.size()*sizeof(unsigned char));
      l.strength = DvceArray1D<unsigned char>("native mg strength", l.g.size());
      l.strength_ready = true;
    }
    auto row = l.row;
    auto g = l.g;
    auto strength = l.strength;
    auto mass = l.mass;
    int changed = 0;
    Kokkos::parallel_reduce("native_mg_strength", Policy(0, l.n),
    KOKKOS_LAMBDA(int i, int &changes) {
      double maximum = 0.0, diffusion = 0.0;
      for (int q = row(i); q < row(i+1); ++q) {
        maximum = fmax(maximum, g(q));
        diffusion += g(q);
      }
      for (int q = row(i); q < row(i+1); ++q) {
        // Mass-dominated rows are already well conditioned under diagonal
        // smoothing. Giving them a shared coarse value can inject errors much
        // larger than their tiny physical coupling to a brighter region.
        const unsigned char next = diffusion > 0.5*mass(i) &&
            g(q) > 0.0 && g(q) >= 0.25*maximum;
        if (initialize || strength(q) != next) ++changes;
        strength(q) = next;
      }
    }, changed);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, comm);
#endif
    return changed != 0;
  }

  bool Coarsen(Level &fine, Level &coarse) {
    ProfileRegion region("native_mg_coarsen_"+std::to_string(fine.global));
    fine.coarsening_stalled = false;
    // Only integer strength decisions, never numerical matrix values, return
    // to the host for the symbolic partition.
    UpdateStrength(fine);
    auto strong = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine.strength);
    Kokkos::Profiling::pushRegion("native_mg_partition");
    std::vector<int> parent(fine.n, -1), members, member_offsets{0};
    // Zero interpolation rows are handled entirely by the symmetric smoother;
    // the remaining coarse matrix is still exactly P^T A P.
    std::vector<unsigned char> active(fine.n, 0);
    for (int i = 0; i < fine.n; ++i) {
      for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) active[i] |= strong(q);
    }
    members.reserve(fine.n);
    member_offsets.reserve(fine.n/4+1);
    // Bounded breadth-first aggregates follow strong local edges. Aggregates
    // never straddle ranks; coarse edges still include all remote couplings.
    for (int seed = 0; seed < fine.n; ++seed) {
      if (!active[seed] || parent[seed] >= 0) continue;
      const int aggregate = member_offsets.size()-1;
      const std::size_t begin = members.size();
      members.push_back(seed);
      parent[seed] = aggregate;
      for (std::size_t p = begin;
           p < members.size() && members.size()-begin < kAggregateSize; ++p) {
        const int i = members[p];
        for (int q = fine.offsets[i];
             q < fine.offsets[i+1] && members.size()-begin < kAggregateSize; ++q) {
          const int j = fine.columns[q]-fine.first;
          if (j >= 0 && j < fine.n && active[j] && parent[j] < 0 && strong(q)) {
            parent[j] = aggregate;
            members.push_back(j);
          }
        }
      }
      member_offsets.push_back(members.size());
    }
    Kokkos::Profiling::popRegion();
    // Attach orphan singletons through a strong edge to an existing aggregate.
    // Never group unrelated rows to force a coarsening ratio: that injects
    // corrections between disconnected components and into faint, weakly
    // coupled cells. Large star aggregates are supported by the gather kernels.
    const int aggregates = member_offsets.size()-1;
    for (int i = 0; i < fine.n; ++i) {
      const int a = parent[i];
      if (a < 0 || member_offsets[a+1]-member_offsets[a] != 1) continue;
      for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) {
        const int j = fine.columns[q]-fine.first;
        if (j < 0 || j >= fine.n || !strong(q)) continue;
        const int b = parent[j];
        if (b >= 0 && member_offsets[b+1]-member_offsets[b] > 1) {
          parent[i] = b;
          break;
        }
      }
    }
    std::vector<int> counts(aggregates, 0), remap(aggregates, -1);
    for (int a : parent) if (a >= 0) ++counts[a];
    member_offsets.assign(1, 0);
    for (int a = 0; a < aggregates; ++a) if (counts[a]) {
      remap[a] = member_offsets.size()-1;
      member_offsets.push_back(member_offsets.back()+counts[a]);
    }
    members.resize(member_offsets.back());
    auto cursor = member_offsets;
    for (int i = 0; i < fine.n; ++i) if (parent[i] >= 0) {
      parent[i] = remap[parent[i]];
      members[cursor[parent[i]]++] = i;
    }
    coarse.n = member_offsets.size()-1;
    Ownership(coarse);
    if (coarse.global == 0) return false;
    if (coarse.global > 0.65*fine.global) {
      fine.coarsening_stalled = true;
      return false;
    }
    std::vector<int> outgoing(fine.send_ids.size()), incoming(fine.ghosts);
    for (std::size_t q = 0; q < outgoing.size(); ++q) {
      const int a = parent[fine.send_ids[q]];
      outgoing[q] = a < 0 ? -1 : coarse.first+a;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(outgoing.data(), fine.send_counts.data(), fine.send_offsets.data(),
                  MPI_INT,
                  incoming.data(), fine.recv_counts.data(), fine.recv_offsets.data(),
                  MPI_INT, comm);
#endif
    Budget(sizeof(int)*(parent.size()+members.size()+member_offsets.size()));
    fine.parent = Upload(parent, "native mg parent");
    fine.members = Upload(members, "native mg members");
    fine.member_offsets = Upload(member_offsets, "native mg member offsets");
    // P has zero rows in mass-dominated cells. An edge from an active cell to
    // such a row contributes to the Galerkin mass, not an off-diagonal entry.
    Budget(fine.g.size()*sizeof(unsigned char));
    fine.boundary_mass = DvceArray1D<unsigned char>("native mg boundary mass", fine.g.size());
    auto device_parent = fine.parent;
    auto remote = Upload(incoming, "native mg remote active aggregate");
    auto device_row = fine.row;
    auto device_col = fine.col;
    auto boundary = fine.boundary_mass;
    const int owned = fine.n;
    Kokkos::parallel_for("native_mg_boundary_mass", Policy(0, owned),
    KOKKOS_LAMBDA(int i) {
      for (int q = device_row(i); q < device_row(i+1); ++q) {
        const int j = device_col(q);
        boundary(q) = j < owned ? device_parent(j) < 0 : remote(j-owned) < 0;
      }
    });
    if constexpr (kDeviceTeams) {
      DeviceCoarseGraph(fine, coarse, incoming);
      Allocate(coarse);
      return true;
    }
    auto local_columns = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine.col);
    Kokkos::Profiling::pushRegion("native_mg_coarse_graph");
    std::vector<int> terms, offsets{0};
    terms.reserve(fine.columns.size());
    offsets.reserve(fine.columns.size()/2+1);
    coarse.offsets.reserve(coarse.n+1);
    coarse.columns.reserve(fine.columns.size()/2);
    std::vector<std::pair<int, int>> entries;
    entries.reserve(128);
    coarse.offsets.push_back(0);
    for (int a = 0; a < coarse.n; ++a) {
      entries.clear();
      for (int p = member_offsets[a]; p < member_offsets[a+1]; ++p) {
        const int i = members[p];
        for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) {
          const int j = local_columns(q);
          const int b = j < fine.n ? (parent[j] < 0 ? -1 : coarse.first+parent[j]) :
              incoming[j-fine.n];
          if (b >= 0 && b != coarse.first+a) entries.emplace_back(b, q);
        }
      }
      // Reuse one contiguous buffer instead of allocating a tree node and a
      // vector for every coarse edge. This dominates cold setup on large grids.
      std::sort(entries.begin(), entries.end());
      for (std::size_t p = 0; p < entries.size();) {
        const int column = entries[p].first;
        coarse.columns.push_back(column);
        do {
          terms.push_back(entries[p++].second);
        } while (p < entries.size() && entries[p].first == column);
        offsets.push_back(terms.size());
      }
      coarse.offsets.push_back(coarse.columns.size());
    }
    Kokkos::Profiling::popRegion();
    Budget(sizeof(int)*(terms.size()+offsets.size()));
    fine.edge_terms = Upload(terms, "native mg edge terms");
    fine.edge_offsets = Upload(offsets, "native mg edge offsets");
    Allocate(coarse);
    return true;
  }

  // Eliminate an independent set of interior vertices exactly. The Schur
  // complement of a positive mass plus graph Laplacian is another such graph:
  // g'ij = gij + sum_f gif*gfj/df, m'i = mi + sum_f gif*mf/df.
  // This supplies energy-minimizing interpolation instead of a constant on
  // each fine aggregate. Keep remote-adjacent vertices, so interpolation and
  // its transpose need no fine-level halo or reverse communication.
  bool Eliminate(Level &fine, Level &coarse) {
    ProfileRegion region("native_mg_elimination_graph");
    std::vector<int> state(fine.n, 0), parent(fine.n, -1), retained;
    for (int i = 0; i < fine.n; ++i) {
      if (state[i] != 0 || fine.offsets[i+1] == fine.offsets[i] ||
          fine.offsets[i+1]-fine.offsets[i] > 8) continue;
      bool interior = true;
      for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) {
        const int j = fine.columns[q]-fine.first;
        interior &= j >= 0 && j < fine.n;
      }
      if (!interior) continue;
      state[i] = -1;
      for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) {
        state[fine.columns[q]-fine.first] = 1;
      }
    }
    for (int i = 0; i < fine.n; ++i) if (state[i] != -1) {
      parent[i] = retained.size();
      retained.push_back(i);
    }
    coarse.n = retained.size();
    Ownership(coarse);
    // Small MPI subdomains can have mostly interface vertices that must remain
    // coarse. In that case elimination adds fill without enough reduction;
    // retain the original smoothed aggregation cycle instead.
    if (coarse.global == 0 || coarse.global > 0.65*fine.global) return false;
    std::vector<int> outgoing(fine.send_ids.size()), incoming(fine.ghosts);
    for (std::size_t q = 0; q < outgoing.size(); ++q) {
      outgoing[q] = coarse.first+parent[fine.send_ids[q]];
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(outgoing.data(), fine.send_counts.data(), fine.send_offsets.data(),
                  MPI_INT, incoming.data(), fine.recv_counts.data(),
                  fine.recv_offsets.data(), MPI_INT, comm);
#endif
    auto local_columns = Kokkos::create_mirror_view_and_copy(HostMemSpace(), fine.col);
    struct Term { int column, left, right, pivot; };
    std::vector<Term> entries;
    std::vector<int> left, right, pivot, offsets{0};
    coarse.offsets.reserve(coarse.n+1);
    coarse.offsets.push_back(0);
    for (int a = 0; a < coarse.n; ++a) {
      const int i = retained[a];
      entries.clear();
      for (int q = fine.offsets[i]; q < fine.offsets[i+1]; ++q) {
        const int j = local_columns(q);
        if (j >= fine.n || parent[j] >= 0) {
          const int b = j < fine.n ? coarse.first+parent[j] : incoming[j-fine.n];
          entries.push_back({b, q, -1, -1});
        } else {
          for (int t = fine.offsets[j]; t < fine.offsets[j+1]; ++t) {
            const int k = local_columns(t);
            if (k != i) entries.push_back({coarse.first+parent[k], q, t, j});
          }
        }
      }
      std::sort(entries.begin(), entries.end(), [](const Term &a, const Term &b) {
        return a.column != b.column ? a.column < b.column : a.pivot < b.pivot;
      });
      for (std::size_t q = 0; q < entries.size();) {
        const int column = entries[q].column;
        coarse.columns.push_back(column);
        do {
          left.push_back(entries[q].left);
          right.push_back(entries[q].right);
          pivot.push_back(entries[q].pivot);
          ++q;
        } while (q < entries.size() && entries[q].column == column);
        offsets.push_back(left.size());
      }
      coarse.offsets.push_back(coarse.columns.size());
    }
    Budget(sizeof(int)*(parent.size()+retained.size()+3*left.size()+offsets.size()));
    fine.parent = Upload(parent, "native mg retained indices");
    fine.retained = Upload(retained, "native mg retained vertices");
    fine.edge_terms = Upload(left, "native mg Schur left terms");
    fine.edge_right = Upload(right, "native mg Schur right terms");
    fine.edge_pivot = Upload(pivot, "native mg Schur pivots");
    fine.edge_offsets = Upload(offsets, "native mg Schur offsets");
    fine.schur = true;
    Allocate(coarse);
    return true;
  }

  void SchurMatrix(Level &fine, Level &coarse) {
    auto retained = fine.retained;
    auto parent = fine.parent;
    auto row = fine.row;
    auto col = fine.col;
    auto fm = fine.mass;
    auto fg = fine.g;
    auto fd = fine.diagonal;
    auto cm = coarse.mass;
    const int owned = fine.n;
    Kokkos::parallel_for("native_mg_Schur_mass", Policy(0, coarse.n),
    KOKKOS_LAMBDA(int a) {
      const int i = retained(a);
      double value = fm(i);
      for (int q = row(i); q < row(i+1); ++q) {
        const int j = col(q);
        if (j < owned && parent(j) < 0) value += fg(q)*(fm(j)/fd(j));
      }
      cm(a) = value;
    });
    auto left = fine.edge_terms;
    auto right = fine.edge_right;
    auto pivot = fine.edge_pivot;
    auto offsets = fine.edge_offsets;
    auto cg = coarse.g;
    Kokkos::parallel_for("native_mg_Schur_edges", Policy(0, cg.extent_int(0)),
    KOKKOS_LAMBDA(int q) {
      double value = 0.0;
      for (int p = offsets(q); p < offsets(q+1); ++p) {
        value += right(p) < 0 ? fg(left(p)) : fg(left(p))*(fg(right(p))/fd(pivot(p)));
      }
      cg(q) = value;
    });
    Diagonal(coarse);
  }

  void SchurCycle(std::size_t level) {
    auto &fine = *levels[level];
    auto &coarse = *levels[level+1];
    auto retained = fine.retained;
    auto parent = fine.parent;
    auto row = fine.row;
    auto col = fine.col;
    auto g = fine.g;
    auto diagonal = fine.diagonal;
    auto mass = fine.mass;
    auto rhs = fine.rhs;
    auto crhs = coarse.rhs;
    const int owned = fine.n;
    Kokkos::parallel_for("native_mg_Schur_restrict", Policy(0, coarse.n),
    KOKKOS_LAMBDA(int a) {
      const int i = retained(a);
      double value = rhs(i);
      for (int q = row(i); q < row(i+1); ++q) {
        const int j = col(q);
        if (j < owned && parent(j) < 0) value += g(q)*(rhs(j)/diagonal(j));
      }
      crhs(a) = value;
    });
    Cycle(level+1);
    auto x = fine.x;
    auto cx = coarse.x;
    Kokkos::parallel_for("native_mg_Schur_prolong", Policy(0, fine.n),
    KOKKOS_LAMBDA(int i) {
      if (parent(i) >= 0) {
        x(i) = cx(parent(i));
      } else {
        // Evaluate the exact interpolation as differences around an anchor.
        // This preserves a constant without summing rounded partition weights.
        int anchor_edge = row(i);
        double strongest = 0.0;
        for (int q = row(i); q < row(i+1); ++q) {
          if (g(q) > strongest) { strongest = g(q); anchor_edge = q; }
        }
        // Mass-dominated rows need the direct weighted sum: subtracting a
        // remote scale there would erase their physically tiny response.
        const double anchor = strongest > mass(i) ?
            cx(parent(col(anchor_edge))) : 0.0;
        double value = rhs(i)/diagonal(i)-(mass(i)/diagonal(i))*anchor;
        for (int q = row(i); q < row(i+1); ++q) {
          value += (g(q)/diagonal(i))*(cx(parent(col(q)))-anchor);
        }
        x(i) = anchor+value;
      }
    });
  }

  void Galerkin(Level &fine, Level &coarse) {
    if (fine.schur) { SchurMatrix(fine, coarse); return; }
    auto members = fine.members;
    auto offsets = fine.member_offsets;
    auto fm = fine.mass;
    auto cm = coarse.mass;
    auto row = fine.row;
    auto fg = fine.g;
    auto boundary = fine.boundary_mass;
    Kokkos::parallel_for("native_mg_coarse_mass", Policy(0, coarse.n),
    KOKKOS_LAMBDA(int a) {
      double value = 0.0;
      for (int p = offsets(a); p < offsets(a+1); ++p) {
        const int i = members(p);
        value += fm(i);
        for (int q = row(i); q < row(i+1); ++q) if (boundary(q)) value += fg(q);
      }
      cm(a) = value;
    });
    auto terms = fine.edge_terms;
    auto edges = fine.edge_offsets;
    auto cg = coarse.g;
    Kokkos::parallel_for("native_mg_coarse_edges", Policy(0, cg.extent_int(0)),
    KOKKOS_LAMBDA(int q) {
      double value = 0.0;
      for (int p = edges(q); p < edges(q+1); ++p) value += fg(terms(p));
      cg(q) = value;
    });
    Diagonal(coarse);
  }

  void FactorRoot(Level &l) {
    if (l.global > kRootSize) return;
    const int n = l.global, first = l.first;
    if (l.factor.extent_int(0) != n) {
      Budget(sizeof(RootScalar)*2*static_cast<std::size_t>(n)*n+sizeof(double)*n);
      l.factor = DvceArray2D<RootScalar>("native mg root factor", n, n);
      l.inverse = DvceArray2D<RootScalar>("native mg root inverse", n, n);
      l.root_rhs = DvceArray1D<double>("native mg root rhs", n);
      l.host_root_rhs = HostArray1D<double>("native mg root host rhs", n);
    }
    auto factor = l.factor;
    Kokkos::deep_copy(factor, RootScalar(0.0));
    auto global_columns = Upload(l.columns, "native mg root global columns");
    auto row = l.row;
    auto g = l.g;
    auto mass = l.mass;
    Kokkos::parallel_for("native_mg_root_matrix", Policy(0, l.n),
    KOKKOS_LAMBDA(int i) {
      factor(first+i, first+i) = mass(i);
      for (int q = row(i); q < row(i+1); ++q) factor(first+i, global_columns(q)) = g(q);
    });
#if MPI_PARALLEL_ENABLED
    if (ranks > 1) {
      auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), factor);
      MPI_Allreduce(MPI_IN_PLACE, host.data(), 2*n*n, MPI_DOUBLE, MPI_SUM, comm);
      Kokkos::deep_copy(factor, host);
    }
#endif
    int invalid = 0;
    Kokkos::parallel_reduce("native_mg_root_cholesky", Policy(0, 1),
    KOKKOS_LAMBDA(int, int &bad) {
      // Factor the positive mass/conductance representation, not a diagonal
      // containing their rounded sum. Ordinary Cholesky subtracts diffusion
      // terms of size g to recover a pivot of size mass, losing the constant
      // mode when g/mass is large. Exact elimination instead has positive
      // updates m'_i=m_i+g_ik*m_k/d_k and g'_ij=g_ij+g_ik*g_jk/d_k.
      // Store those Schur complements in the unprocessed lower triangle and
      // the ordinary Cholesky columns in the processed part. Ratios precede
      // products to avoid squaring a large conductance.
      for (int k = 0; k < n; ++k) {
        const RootScalar pivot_mass = factor(k, k);
        RootScalar diagonal = pivot_mass;
        for (int i = k+1; i < n; ++i) diagonal += factor(i, k);
        if (!(diagonal.hi > 0.0) || !Kokkos::isfinite(diagonal.hi) || !Kokkos::isfinite(diagonal.lo)) { ++bad; return; }
        const RootScalar root = RootSqrt(diagonal);
        for (int i = k+1; i < n; ++i) {
          const RootScalar edge = factor(i, k), fraction = edge/diagonal;
          factor(i, i) += fraction*pivot_mass;
          for (int j = i+1; j < n; ++j) factor(j, i) += fraction*factor(j, k);
          factor(i, k) = -edge/root;
        }
        factor(k, k) = root;
      }
    }, invalid);
    Require(invalid == 0, "coarse Cholesky lost positive definiteness");
    // Invert this bounded (<=64) factor once per numerical setup. A serial
    // triangular substitution on the GPU in every cycle has high dependent
    // load latency. Parallel columns here turn every subsequent root solve
    // into a short, coalesced matrix-vector product without host transfers.
    auto inverse = l.inverse;
    Kokkos::parallel_for("native_mg_root_inverse", Policy(0, n),
    KOKKOS_LAMBDA(int column) {
      for (int i = 0; i < n; ++i) {
        RootScalar value(i == column ? 1.0 : 0.0);
        for (int j = 0; j < i; ++j) value -= factor(i, j)*inverse(j, column);
        inverse(i, column) = value/factor(i, i);
      }
      for (int i = n-1; i >= 0; --i) {
        RootScalar value = inverse(i, column);
        for (int j = i+1; j < n; ++j) value -= factor(j, i)*inverse(j, column);
        inverse(i, column) = value/factor(i, i);
      }
    });
    Kokkos::parallel_for("native_mg_root_symmetrize", Policy(0, n),
    KOKKOS_LAMBDA(int i) {
      for (int j = 0; j < i; ++j) {
        const RootScalar value = (inverse(i, j)+inverse(j, i))*RootScalar(0.5);
        inverse(i, j) = value;
        inverse(j, i) = value;
      }
    });
  }

  void SetupReplica(Level &l) {
#if MPI_PARALLEL_ENABLED
    // Agglomerate the bounded bottom of the distributed hierarchy. Repeating
    // tiny halo exchanges at every coarse level costs much more than its GPU
    // arithmetic. One gather per V-cycle lets every rank solve this small graph
    // locally, including connections across the former rank boundaries.
    if (!l.replica) {
      l.replica = std::make_unique<State>();
      auto &local = *l.replica;
      MPI_Comm_dup(MPI_COMM_SELF, &local.comm);
      local.max_bytes = max_bytes-Bytes();
      local.levels.emplace_back(new Level);
      auto &root = *local.levels.front();
      root.n = l.global;
      local.Ownership(root);
      std::vector<int> degrees(l.n), all_degrees(l.global);
      for (int i = 0; i < l.n; ++i) degrees[i] = l.offsets[i+1]-l.offsets[i];
      MPI_Allgatherv(degrees.data(), l.n, MPI_INT, all_degrees.data(),
                    l.counts.data(), l.starts.data(), MPI_INT, comm);
      root.offsets.assign(l.global+1, 0);
      for (int i = 0; i < l.global; ++i) root.offsets[i+1] = root.offsets[i]+all_degrees[i];
      const int edges = l.columns.size();
      l.matrix_counts.resize(ranks);
      l.matrix_starts.assign(ranks+1, 0);
      MPI_Allgather(&edges, 1, MPI_INT, l.matrix_counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < ranks; ++r) {
        l.matrix_starts[r+1] = l.matrix_starts[r]+l.matrix_counts[r];
      }
      root.columns.resize(l.matrix_starts.back());
      MPI_Allgatherv(l.columns.data(), edges, MPI_INT, root.columns.data(),
                    l.matrix_counts.data(), l.matrix_starts.data(), MPI_INT, comm);
      local.Allocate(root);
      l.host_root_rhs = HostArray1D<double>("native mg replicated rhs", l.global);
    }
    auto &local = *l.replica;
    local.max_bytes = max_bytes-(Bytes()-local.Bytes());
    auto &root = *local.levels.front();
    auto g = Kokkos::create_mirror_view_and_copy(HostMemSpace(), l.g);
    auto mass = Kokkos::create_mirror_view_and_copy(HostMemSpace(), l.mass);
    auto all_g = Kokkos::create_mirror_view(root.g);
    auto all_mass = Kokkos::create_mirror_view(root.mass);
    MPI_Allgatherv(g.data(), l.g.extent_int(0), MPI_DOUBLE, all_g.data(),
                  l.matrix_counts.data(), l.matrix_starts.data(), MPI_DOUBLE, comm);
    MPI_Allgatherv(mass.data(), l.n, MPI_DOUBLE, all_mass.data(),
                  l.counts.data(), l.starts.data(), MPI_DOUBLE, comm);
    Kokkos::deep_copy(root.g, all_g);
    Kokkos::deep_copy(root.mass, all_mass);
    local.Setup(root.g, root.mass);
#endif
  }

  void Smooth(Level &l, int sweeps, bool zero) {
    auto row = l.row;
    auto col = l.col;
    auto g = l.g;
    auto mass = l.mass;
    auto diagonal = l.diagonal;
    auto rhs = l.rhs;
    const bool chebyshev = sweeps == 2;
    if (zero) {
      auto x = l.x;
      const double omega = chebyshev ? kChebyshevFirst : kOmega;
      Kokkos::parallel_for("native_mg_smooth_zero", Policy(0, l.n),
      KOKKOS_LAMBDA(int i) { x(i) = omega*rhs(i)/diagonal(i); });
      --sweeps;
    }
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      const double omega = chebyshev ?
          (sweep == 0 ? kChebyshevSecond : kChebyshevFirst) : kOmega;
      Halo(l);
      auto x = l.x;
      auto next = l.next;
      if (kDeviceTeams && l.g.size() > 16*l.mass.size()) {
        const int n = l.n;
        const int lanes = l.g.size() > 16*l.mass.size() ? 32 : 8;
        Kokkos::parallel_for("native_mg_smooth", TeamPolicy((n+7)/8, 8, lanes),
        KOKKOS_LAMBDA(const TeamMember &team) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 8), [&](int slot) {
            const int i = 8*team.league_rank()+slot;
            if (i >= n) return;
            const double center = x(i);
            double diffusion = 0.0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, row(i), row(i+1)),
                [&](int q, double &sum) { sum += g(q)*(center-x(col(q))); }, diffusion);
            Kokkos::single(Kokkos::PerThread(team), [&]() {
              next(i) = center+omega*(rhs(i)-mass(i)*center-diffusion)/diagonal(i);
            });
          });
        });
      } else {
        Kokkos::parallel_for("native_mg_smooth", Policy(0, l.n),
        KOKKOS_LAMBDA(int i) {
          const double center = x(i);
          double ax = mass(i)*center;
          for (int q = row(i); q < row(i+1); ++q) ax += g(q)*(center-x(col(q)));
          next(i) = center+omega*(rhs(i)-ax)/diagonal(i);
        });
      }
      std::swap(l.x, l.next);
    }
  }

  void Cycle(std::size_t level) {
    auto &l = *levels[level];
    if (l.schur) { SchurCycle(level); return; }
    if (l.replica) {
      auto &local = *l.replica;
      auto &root = *local.levels.front();
#if MPI_PARALLEL_ENABLED
      if (device_mpi) {
        DevExeSpace().fence("native mg replicated rhs before MPI");
        MPI_Allgatherv(l.rhs.data(), l.n, MPI_DOUBLE, root.rhs.data(), l.counts.data(),
                      l.starts.data(), MPI_DOUBLE, comm);
      } else {
        auto owned = Kokkos::subview(l.host_root_rhs, std::make_pair(l.first, l.first+l.n));
        Kokkos::deep_copy(owned, l.rhs);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, l.host_root_rhs.data(),
                      l.counts.data(), l.starts.data(), MPI_DOUBLE, comm);
        Kokkos::deep_copy(root.rhs, l.host_root_rhs);
      }
#endif
      local.Cycle(0);
      Kokkos::deep_copy(DevExeSpace(), Kokkos::subview(l.x, std::make_pair(0, l.n)),
          Kokkos::subview(root.x, std::make_pair(l.first, l.first+l.n)));
      return;
    }
    if (l.factor.size()) {
      const int n = l.global, first = l.first;
      auto b = l.root_rhs;
      if (ranks == 1) {
        Kokkos::deep_copy(DevExeSpace(), b, l.rhs);
      } else {
#if MPI_PARALLEL_ENABLED
        if (device_mpi) {
          DevExeSpace().fence("native mg root before MPI");
          MPI_Allgatherv(l.rhs.data(), l.n, MPI_DOUBLE, b.data(), l.counts.data(),
                        l.starts.data(), MPI_DOUBLE, comm);
        } else {
          auto local = Kokkos::subview(l.host_root_rhs, std::make_pair(first, first+l.n));
          Kokkos::deep_copy(local, l.rhs);
          MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, l.host_root_rhs.data(),
                        l.counts.data(), l.starts.data(), MPI_DOUBLE, comm);
          Kokkos::deep_copy(b, l.host_root_rhs);
        }
#endif
      }
      auto inverse = l.inverse;
      auto x = l.x;
      Kokkos::parallel_for("native_mg_root_solve", Policy(0, l.n),
      KOKKOS_LAMBDA(int i) {
        RootScalar value;
        for (int j = 0; j < n; ++j) value += inverse(j, first+i)*RootScalar(b(j));
        x(i) = value.hi+value.lo;
      });
      return;
    }
    if (level+1 == levels.size()) {
      // For very large rank counts or disconnected graphs, keep a bounded,
      // fixed positive polynomial instead of replicating an unbounded root.
      Smooth(l, 16, true);
      return;
    }
    // Exact fine-grid elimination already removes the eliminated error modes.
    // Its first reduced graph is denser; one symmetric Jacobi pre/post pair
    // halves the expensive halo/stencil passes there. For omega=2/3 its
    // symmetric correction is >=(4/9)D^-1, retaining the metric bound above.
    const int sweeps = level == 1 && levels.front()->schur ? 1 : 2;
    Smooth(l, sweeps, true);
    Halo(l);
    auto &coarse = *levels[level+1];
    auto members = l.members;
    auto offsets = l.member_offsets;
    auto row = l.row;
    auto col = l.col;
    auto g = l.g;
    auto mass = l.mass;
    auto x = l.x;
    auto rhs = l.rhs;
    auto crhs = coarse.rhs;
    if (kDeviceTeams && l.g.size() <= 16*l.mass.size()) {
      // Short stencil rows need only one GPU thread. Assigning a vector team
      // to six edges wastes lanes and makes restriction's nested reductions
      // more expensive than the sparse products themselves. Form one residual
      // per row, then gather the small aggregate sums in a second kernel.
      auto residual = l.next;
      Kokkos::parallel_for("native_mg_residual", Policy(0, l.n),
      KOKKOS_LAMBDA(int i) {
        const double center = x(i);
        double value = rhs(i)-mass(i)*center;
        for (int q = row(i); q < row(i+1); ++q) value -= g(q)*(center-x(col(q)));
        residual(i) = value;
      });
      Kokkos::parallel_for("native_mg_restrict", Policy(0, coarse.n),
      KOKKOS_LAMBDA(int a) {
        double value = 0.0;
        for (int p = offsets(a); p < offsets(a+1); ++p) value += residual(members(p));
        crhs(a) = value;
      });
    } else if (kDeviceTeams) {
      const int lanes = l.g.size() > 16*l.mass.size() ? 32 : 8;
      Kokkos::parallel_for("native_mg_restrict_residual",
          TeamPolicy(coarse.n, kAggregateSize, lanes),
      KOKKOS_LAMBDA(const TeamMember &team) {
        const int a = team.league_rank();
        double value = 0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, offsets(a), offsets(a+1)),
        [&](int p, double &sum) {
          const int i = members(p);
          const double center = x(i);
          double diffusion = 0.0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, row(i), row(i+1)),
              [&](int q, double &v) { v += g(q)*(center-x(col(q))); }, diffusion);
          sum += rhs(i)-mass(i)*center-diffusion;
        }, value);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { crhs(a) = value; });
      });
    } else {
      Kokkos::parallel_for("native_mg_restrict_residual", Policy(0, coarse.n),
      KOKKOS_LAMBDA(int a) {
        double value = 0.0;
        for (int p = offsets(a); p < offsets(a+1); ++p) {
          const int i = members(p);
          const double center = x(i);
          double residual = rhs(i)-mass(i)*center;
          for (int q = row(i); q < row(i+1); ++q) residual -= g(q)*(center-x(col(q)));
          value += residual;
        }
        crhs(a) = value;
      });
    }
    Cycle(level+1);
    auto cx = coarse.x;
    auto parent = l.parent;
    Kokkos::parallel_for("native_mg_prolong", Policy(0, l.n),
    KOKKOS_LAMBDA(int i) {
      if (parent(i) >= 0) x(i) += kCoarseCorrection*cx(parent(i));
    });
    Smooth(l, sweeps, false);
  }

  void Setup(const DvceArray1D<double> &conductance, const DvceArray1D<double> &mass);
};

NativeMultigrid::NativeMultigrid() : state_(new State) {}
NativeMultigrid::~NativeMultigrid() = default;

void NativeMultigrid::Configure(const std::vector<int> &offsets,
    const std::vector<int> &columns, int first_row, int global_rows,
    bool device_mpi, std::size_t max_bytes) {
  ProfileRegion region("native_mg_configure");
  auto &s = *state_;
#if MPI_PARALLEL_ENABLED
  if (s.comm == MPI_COMM_NULL) MPI_Comm_dup(MPI_COMM_WORLD, &s.comm);
  MPI_Comm_rank(s.comm, &s.rank);
  MPI_Comm_size(s.comm, &s.ranks);
#endif
  bool valid = !offsets.empty() && offsets.front() == 0 && global_rows > 0 &&
      columns.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      offsets.back() == static_cast<int>(columns.size()) && max_bytes > 0;
  for (std::size_t i = 1; i < offsets.size(); ++i) valid &= offsets[i] >= offsets[i-1];
  s.Require(valid, "invalid CSR graph or memory budget");
  s.levels.clear();
  s.ready = false;
  s.device_mpi = device_mpi;
  s.max_bytes = max_bytes;
  s.levels.emplace_back(new State::Level);
  auto &l = *s.levels[0];
  l.offsets = offsets;
  l.columns = columns;
  l.n = offsets.size()-1;
  s.Ownership(l);
  s.Require(l.first == first_row && l.global == global_rows,
            "row intervals must cover the graph in MPI rank order");
  for (int i = 0; i < l.n; ++i) {
    int previous = -1;
    for (int q = offsets[i]; q < offsets[i+1]; ++q) {
      valid &= columns[q] >= 0 && columns[q] < global_rows &&
               columns[q] != first_row+i && columns[q] > previous;
      previous = columns[q];
    }
  }
  s.Require(valid, "CSR columns must be sorted, unique, and off-diagonal");
  s.Allocate(l);
}

void NativeMultigrid::Setup(const DvceArray1D<double> &conductance,
                           const DvceArray1D<double> &mass) {
  state_->Setup(conductance, mass);
}

void NativeMultigrid::State::Setup(const DvceArray1D<double> &conductance,
                                 const DvceArray1D<double> &mass) {
  auto &s = *this;
  if (s.levels.empty()) Fail("Configure must precede Setup");
  auto &fine = *s.levels.front();
  s.Require(conductance.size() == fine.g.size() && mass.size() == fine.mass.size(),
            "incorrect numerical array lengths");
  Kokkos::deep_copy(fine.g, conductance);
  Kokkos::deep_copy(fine.mass, mass);
  s.Diagonal(fine);
  if (s.ready) {
    for (std::size_t q = 0; q < s.levels.size(); ++q) {
      auto &current = *s.levels[q];
      if (q+1 == s.levels.size() && (current.global <= kRootSize || current.replica)) break;
      if (!current.schur && s.UpdateStrength(current)) {
        // The graph can be unchanged while opacity groups (or Newton updates)
        // move the strong connections. Rebuild from the first affected level,
        // collectively, before it is used as a fixed Krylov preconditioner.
        s.levels.resize(q+1);
        current.parent = {};
        current.members = {};
        current.member_offsets = {};
        current.edge_terms = {};
        current.edge_offsets = {};
        current.boundary_mass = {};
        s.ready = false;
        break;
      }
      if (q+1 < s.levels.size()) s.Galerkin(current, *s.levels[q+1]);
    }
  }
  if (!s.ready) {
    while (s.levels.back()->global > kRootSize && s.levels.size() < kMaxLevels &&
           !(s.ranks > 1 && s.levels.back()->global <= kReplicatedSize)) {
      auto next = std::make_unique<State::Level>();
      auto &current = *s.levels.back();
      const bool eliminated = s.levels.size() == 1 &&
          current.global > kReplicatedSize && s.Eliminate(current, *next);
      if (!eliminated && !s.Coarsen(current, *next)) break;
      s.Galerkin(current, *next);
      s.levels.push_back(std::move(next));
    }
    s.ready = true;
  }
  auto &root = *s.levels.back();
  if (s.ranks > 1 && root.global > kRootSize &&
      (root.global <= kReplicatedSize ||
       (root.coarsening_stalled && root.global <= kStalledReplicatedSize))) {
    s.SetupReplica(root);
  } else {
    s.FactorRoot(root);
  }
}

void NativeMultigrid::Apply(const DvceArray1D<double> &rhs,
                           const DvceArray1D<double> &solution) {
  auto &s = *state_;
  if (!s.ready) Fail("Setup must precede Apply");
  auto &fine = *s.levels.front();
  if (rhs.extent_int(0) != fine.n || solution.extent_int(0) != fine.n) {
    Fail("incorrect vector lengths");
  }
  Kokkos::deep_copy(DevExeSpace(), fine.rhs, rhs);
  s.Cycle(0);
  Kokkos::deep_copy(DevExeSpace(), solution,
                    Kokkos::subview(fine.x, std::make_pair(0, fine.n)));
}

int NativeMultigrid::levels() const { return state_->Depth(); }
#if MPI_PARALLEL_ENABLED
void NativeMultigrid::SetHaloWait(HaloWait wait) { state_->halo_wait = wait; }
#endif
std::size_t NativeMultigrid::bytes() const { return state_->Bytes(); }
}  // namespace diffusion
