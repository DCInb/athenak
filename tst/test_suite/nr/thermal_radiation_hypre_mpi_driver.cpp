//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! Distributed public-API oracle; pass a mode argument and two or more ranks.
//! Malformed inputs exist only on rank one and must fail collectively.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "athena.hpp"
#include <mpi.h>
#include "two_temperature/thermal_radiation_hypre.hpp"
#include "two_temperature/thermal_radiation_batch.hpp"
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#endif

namespace {
void Check(bool condition, const std::string &message) {
  int local = condition ? 1 : 0, total = 0, ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (total != ranks) {
    std::cerr << "FAIL " << message << " successful_ranks=" << total << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
}

// Exercise actual coarsening and the replicated dense coarse solve, including
// its nearly singular constant mode. All reference actions use a host CSR
// independent of HYPRE, and cross-rank inner products check the whole V-cycle.
void CoarseOracle(int rank, int ranks) {
  constexpr int width = 48, n = width*width;
  const int first = rank*n/ranks, last = (rank+1)*n/ranks, local = last-first;
  std::vector<int> counts(ranks), offsets(ranks), rows{0}, columns;
  for (int p = 0; p < ranks; ++p) {
    offsets[p] = p*n/ranks;
    counts[p] = (p+1)*n/ranks-offsets[p];
  }
  std::vector<double> values;
  for (int row = first; row < last; ++row) {
    const int x = row%width, y = row/width;
    std::map<int, double> entries{{row, 0.02}};
    for (int col : {y*width+(x+1)%width, y*width+(x+width-1)%width,
                    ((y+1)%width)*width+x, ((y+width-1)%width)*width+x}) {
      const double weight = 1+(row+col)%11;
      entries[row] += weight;
      entries[col] -= weight;
    }
    for (auto entry : entries) {
      columns.push_back(entry.first);
      values.push_back(entry.second);
    }
    rows.push_back(values.size());
  }
  Kokkos::View<double *, DevMemSpace> dv("values", values.size()),
      dr("rhs", local), dx("solution", local);
  auto hv = Kokkos::create_mirror_view(dv), hr = Kokkos::create_mirror_view(dr),
       hx = Kokkos::create_mirror_view(dx);
  ImplicitAMGHypre amg;
  auto setup = [&]() {
    for (std::size_t i = 0; i < values.size(); ++i) hv(i) = values[i];
    Kokkos::deep_copy(dv, hv);
    Check(amg.ConfigureGraph(rows, columns, first, n, 512UL*1024*1024), amg.error());
    Check(amg.Setup(dv.data()), amg.error());
  };
  auto apply = [&](const std::vector<double> &rhs) {
    for (int i = 0; i < local; ++i) hr(i) = rhs[i];
    Kokkos::deep_copy(dr, hr);
    Check(amg.Apply(dr.data(), dx.data()), amg.error());
    Kokkos::deep_copy(hx, dx);
    return std::vector<double>(hx.data(), hx.data()+local);
  };
  auto dot = [&](const std::vector<double> &a, const std::vector<double> &b) {
    double sum = 0;
    for (int i = 0; i < local; ++i) sum += a[i]*b[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum;
  };
  auto matvec = [&](const std::vector<double> &x) {
    std::vector<double> global(n), result(local, 0);
    MPI_Allgatherv(x.data(), local, MPI_DOUBLE, global.data(), counts.data(),
                   offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < local; ++i) {
      for (int j = rows[i]; j < rows[i+1]; ++j) result[i] += values[j]*global[columns[j]];
    }
    return result;
  };
  setup();
  double symmetry = 0, linearity = 0;
  for (int seed = 1; seed <= 6; ++seed) {
    std::vector<double> a(local), b(local), c(local), error(local);
    for (int i = 0; i < local; ++i) {
      a[i] = std::sin((first+i+0.3)*seed);
      b[i] = std::cos((first+i+0.7)*(seed+0.25));
      c[i] = 1.3*a[i]-0.4*b[i];
    }
    auto ma = apply(a), mb = apply(b), mc = apply(c);
    const double scale = std::sqrt(dot(a, a)*dot(mb, mb))+
                         std::sqrt(dot(b, b)*dot(ma, ma));
    symmetry = std::max(symmetry, std::abs(dot(a, mb)-dot(b, ma))/scale);
    for (int i = 0; i < local; ++i) error[i] = mc[i]-(1.3*ma[i]-0.4*mb[i]);
    linearity = std::max(linearity, std::sqrt(dot(error, error)/dot(mc, mc)));
    Check(dot(a, ma) > 0, "distributed V-cycle positivity");
  }
  Check(symmetry < 3e-12 && linearity < 3e-12, "distributed V-cycle symmetry/linearity");
  double contraction = 0, stiff_residual = 0;
  // A changed numerical matrix on the same topology must be solved afresh.
  for (int refresh = 0; refresh < 3; ++refresh) {
    if (refresh) {
      if (refresh == 1) {
        for (double &value : values) value *= 2;
      } else {
        for (int i = 0; i < local; ++i) {
          for (int j = rows[i]; j < rows[i+1]; ++j) {
            values[j] /= 128;
            if (columns[j] == first+i) values[j] += 1e-8-0.04/128;
          }
        }
      }
      setup();
    }
    std::vector<double> rhs(local), solution(local, 0);
    for (int i = 0; i < local; ++i) {
      rhs[i] = refresh == 2 ? 1 : 0.2+std::sin((first+i)*0.41);
    }
    auto residual = rhs, z = apply(residual), direction = z;
    if (refresh == 2) {
      auto action = matvec(z);
      for (double &value : action) value -= 1;
      contraction = std::sqrt(dot(action, action)/n);
      // A V-cycle is an approximate inverse: a small one-cycle residual is
      // not required. Check the slow-mode amplitude, then the PCG true residual.
      Check(std::abs(1e-8*dot(z, rhs)/n-1) < 0.2,
            "distributed near-Neumann constant amplitude");
    }
    const double scale = dot(rhs, rhs);
    double rz = dot(residual, z);
    for (int iteration = 0; iteration < 300; ++iteration) {
      auto action = matvec(direction);
      const double pap = dot(direction, action);
      Check(pap > 0 && rz > 0, "distributed PCG positive denominator");
      const double alpha = rz/pap;
      for (int i = 0; i < local; ++i) {
        solution[i] += alpha*direction[i];
        residual[i] -= alpha*action[i];
      }
      if (dot(residual, residual) < 1e-20*scale) break;
      z = apply(residual);
      const double next = dot(residual, z), beta = next/rz;
      for (int i = 0; i < local; ++i) direction[i] = z[i]+beta*direction[i];
      rz = next;
    }
    auto error = matvec(solution);
    for (int i = 0; i < local; ++i) error[i] -= rhs[i];
    const double true_residual = std::sqrt(dot(error, error)/scale);
    Check(true_residual < (refresh == 2 ? 1e-7 : 2e-10),
          "distributed PCG true residual");
    if (refresh == 2) stiff_residual = true_residual;
  }
#ifdef KOKKOS_ENABLE_CUDA
  // A repeated setup/solve must not lose track of persistent buffers allocated
  // on first application and released at the next setup. Check against actual
  // free device memory, not the hierarchy's estimated allocation formula.
  std::size_t memory_free = 0, memory_total = 0;
  Check(cudaMemGetInfo(&memory_free, &memory_total) == cudaSuccess,
        "query memory before repeated hierarchies");
  const std::size_t available = memory_free+amg.device_bytes();
  std::size_t maximum_accounting_drift = 0;
  const std::vector<double> memory_rhs(local, 1.0);
  for (int refresh = 0; refresh < 32; ++refresh) {
    setup();
    apply(memory_rhs);
    Check(cudaMemGetInfo(&memory_free, &memory_total) == cudaSuccess,
          "query memory after repeated hierarchies");
    const std::size_t current = memory_free+amg.device_bytes();
    const std::size_t drift = current > available ? current-available : available-current;
    maximum_accounting_drift = std::max(maximum_accounting_drift, drift);
  }
  Check(maximum_accounting_drift <= 2UL*1024*1024,
        "reusable GPU bytes must track repeated setup/application allocations");
  if (rank == 0) {
    std::cout << "PASS mode=memory-rebuilds rebuilds=32 maximum_accounting_drift="
              << maximum_accounting_drift << std::endl;
  }
#endif
  if (rank == 0) {
    std::cout << "PASS mode=coarse-oracle symmetry=" << symmetry
              << " linearity=" << linearity << " constant_contraction=" << contraction
              << " stiff_true_residual=" << stiff_residual
              << std::endl;
  }
}
}  // namespace

void BatchOracle(int rank, int ranks) {
  using two_temperature::RadiationGroupBatch;
  using two_temperature::RadiationBatchAllreduce;
  constexpr int n = 2304;
  const int local = n/ranks, first = rank*local;
  std::vector<int> rows{0}, columns;
  for (int i = 0; i < local; ++i) {
    const int row = first+i;
    if (row > 0) columns.push_back(row-1);
    columns.push_back(row);
    if (row+1 < n) columns.push_back(row+1);
    rows.push_back(static_cast<int>(columns.size()));
  }
  ImplicitAMGHypre amg[2];
  const bool expect_reuse = std::getenv("ATHENA_EXPECT_TRANSFER_REUSE") != nullptr;
  Check(!expect_reuse || ImplicitAMGHypre::ReuseAvailable(),
        "transfer-reuse support is available when requested");
  DvceArray1D<double> dv[2], dr[2], dx[2];
  std::vector<double> values[2];
  for (int lane = 0; lane < 2; ++lane) {
    dv[lane] = DvceArray1D<double>("batch oracle values", columns.size());
    dr[lane] = DvceArray1D<double>("batch oracle rhs", local);
    dx[lane] = DvceArray1D<double>("batch oracle correction", local);
    values[lane].resize(columns.size());
    amg[lane].ConfigureReuse(expect_reuse ? 4 : 0, 5, 64.0);
    Check(amg[lane].ConfigureGraph(rows, columns, first, n, 512UL*1024*1024),
          amg[lane].error());
  }
  double errors[2] = {}, symmetry[2] = {};
  RadiationGroupBatch scheduler;
  for (int refresh = 0; refresh < 2; ++refresh) {
    std::vector<std::function<void()>> work;
    for (int lane = 0; lane < 2; ++lane) {
      work.emplace_back([&, lane, refresh]() {
        const double diffusion = lane == 0 ? 2.0 : 11.0;
        const double mass = (lane == 0 ? 1.0 : 0.002)*(refresh+1);
        for (int i = 0; i < local; ++i) {
          for (int q = rows[i]; q < rows[i+1]; ++q) {
            values[lane][q] = columns[q] == first+i ? mass+2*diffusion : -diffusion;
          }
        }
        auto hv = Kokkos::create_mirror_view(dv[lane]);
        for (std::size_t i = 0; i < values[lane].size(); ++i) hv(i) = values[lane][i];
        Kokkos::deep_copy(dv[lane], hv);
        amg[lane].SetReuseContext(lane, refresh, true);
        Check(amg[lane].Setup(dv[lane].data()), amg[lane].error());
        Check(amg[lane].hierarchy_reused() == (expect_reuse && refresh > 0),
              "requested transfer-reuse path was exercised");
        auto dot = [&](const std::vector<double> &a, const std::vector<double> &b) {
          double sum = 0;
          for (int i = 0; i < local; ++i) sum += a[i]*b[i];
          RadiationBatchAllreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          return sum;
        };
        auto apply = [&](const std::vector<double> &rhs) {
          auto host = Kokkos::create_mirror_view(dr[lane]);
          for (int i = 0; i < local; ++i) host(i) = rhs[i];
          Kokkos::deep_copy(dr[lane], host);
          Check(amg[lane].Apply(dr[lane].data(), dx[lane].data()), amg[lane].error());
          auto result = Kokkos::create_mirror_view_and_copy(HostMemSpace(), dx[lane]);
          return std::vector<double>(result.data(), result.data()+local);
        };
        auto action = [&](const std::vector<double> &x) {
          std::vector<double> global(n), y(local, 0);
          MPI_Allgather(x.data(), local, MPI_DOUBLE, global.data(), local, MPI_DOUBLE,
                        MPI_COMM_WORLD);
          for (int i = 0; i < local; ++i) {
            for (int q = rows[i]; q < rows[i+1]; ++q) {
              y[i] += values[lane][q]*global[columns[q]];
            }
          }
          return y;
        };
        std::vector<double> a(local), b(local), c(local), error(local), zero(local, 0);
        for (int i = 0; i < local; ++i) {
          a[i] = std::sin((first+i+1)*0.31);
          b[i] = 0.2+std::cos((first+i+1)*0.67);
          c[i] = 1.3*a[i]-0.4*b[i];
        }
        auto ma = apply(a), mb = apply(b), mc = apply(c), mz = apply(zero);
        symmetry[lane] = std::abs(dot(a, mb)-dot(b, ma))/
            (std::sqrt(dot(a, a)*dot(mb, mb))+std::sqrt(dot(b, b)*dot(ma, ma)));
        for (int i = 0; i < local; ++i) error[i] = mc[i]-(1.3*ma[i]-0.4*mb[i]);
        Check(symmetry[lane] < 3e-12, "batched AMG symmetry");
        Check(dot(a, ma) > 0, "batched AMG positive correction");
        Check(std::sqrt(dot(error, error)/dot(mc, mc)) < 3e-12, "batched AMG linearity");
        Check(dot(mz, mz) == 0.0, "batched AMG zero/other-group isolation");
        const auto rhs = action(b);
        auto residual = rhs, z = apply(residual), direction = z, solution = zero;
        const double scale = dot(rhs, rhs);
        double rz = dot(residual, z);
        for (int iteration = 0; iteration < 300; ++iteration) {
          auto ad = action(direction);
          const double pap = dot(direction, ad);
          Check(pap > 0 && rz > 0, "batched PCG SPD denominators");
          const double alpha = rz/pap;
          for (int i = 0; i < local; ++i) {
            solution[i] += alpha*direction[i];
            residual[i] -= alpha*ad[i];
          }
          if (dot(residual, residual) < 1e-24*scale) break;
          z = apply(residual);
          const double next = dot(residual, z), beta = next/rz;
          for (int i = 0; i < local; ++i) direction[i] = z[i]+beta*direction[i];
          rz = next;
        }
        auto ax = action(solution);
        for (int i = 0; i < local; ++i) error[i] = ax[i]-rhs[i];
        errors[lane] = std::sqrt(dot(error, error)/scale);
        Check(errors[lane] < 2e-11, "batched independent true residual");
      });
    }
    scheduler.Run(work);
  }
  Check(scheduler.counts().paired[0] == 2 && scheduler.counts().paired[1] > 0,
        "AMG setup and apply were actually batched");
  if (rank == 0) std::cout << "PASS mode=batch-oracle residual0=" << errors[0]
      << " residual1=" << errors[1] << " paired=" << scheduler.counts().paired[1] << std::endl;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string mode = argc > 1 ? argv[1] : "valid";
  Check(mode == "batch-oracle" ? ranks >= 2 && 2304%ranks == 0 :
        mode == "coarse-oracle" ? ranks >= 2 : ranks == 2,
        "batch oracle needs >=2 ranks dividing 2304; coarse oracle >=2; other modes two");
  Kokkos::initialize(argc, argv);
  if (mode == "batch-oracle") {
    BatchOracle(rank, ranks);
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
  }
  if (mode == "coarse-oracle") {
    CoarseOracle(rank, ranks);
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
  }
  {
    // SPD global matrix with off-rank edges and nonzero first row on rank one.
    const std::vector<int> rows = rank == 0 ? std::vector<int>{0, 2, 5}
                                           : std::vector<int>{0, 3, 5};
    const std::vector<int> columns = rank == 0 ? std::vector<int>{0, 1, 0, 1, 2}
                                              : std::vector<int>{1, 2, 3, 2, 3};
    const std::vector<double> values = rank == 0 ? std::vector<double>{4, -1, -1, 4, -1}
                                                : std::vector<double>{-1, 4, -1, -1, 4};
    constexpr std::size_t cap = 512UL * 1024 * 1024;
    Kokkos::View<double *, DevMemSpace> dv("values", 5), dr("rhs", 2), dx("answer", 2);
    auto hv = Kokkos::create_mirror_view(dv), hr = Kokkos::create_mirror_view(dr),
         hx = Kokkos::create_mirror_view(dx);
    for (int i = 0; i < 5; ++i) hv(i) = values[i];
    // Exact solution is [1, 2, 3, 4]. Each V-cycle is only an approximate inverse.
    hr(0) = rank == 0 ? 2 : 6;
    hr(1) = rank == 0 ? 4 : 13;
    Kokkos::deep_copy(dv, hv);
    Kokkos::deep_copy(dr, hr);
    ImplicitAMGHypre amg;
    Check(amg.Available(), "backend unavailable");
    if (mode == "bad-csr") {
      auto bad_rows = rows;
      if (rank == 1) bad_rows.back() = 6;
      Check(!amg.ConfigureGraph(bad_rows, columns, 2*rank, 4, cap),
            "rank-local invalid CSR must fail collectively");
    } else if (mode == "memory-cap") {
      Check(!amg.ConfigureGraph(rows, columns, 2*rank, 4, rank == 1 ? 0 : cap),
            "rank-local memory cap must fail collectively");
    }
    Check(amg.ConfigureGraph(rows, columns, 2*rank, 4, cap), amg.error());
    if (mode == "null-setup") {
      Check(!amg.Setup(rank == 1 ? nullptr : dv.data()),
            "rank-local null matrix must fail collectively");
    }
    Check(amg.Setup(dv.data()), amg.error());
#ifdef KOKKOS_ENABLE_CUDA
    int device = -1;
    cudaDeviceProp properties;
    Check(cudaGetDevice(&device) == cudaSuccess &&
              cudaGetDeviceProperties(&properties, device) == cudaSuccess,
          "CUDA device identity after HYPRE setup");
    std::ostringstream identity;
    identity << "rank=" << rank << " cuda_device=" << device << " uuid=";
    for (unsigned char byte : properties.uuid.bytes)
      identity << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    std::cout << identity.str() << std::endl;
#endif
    if (mode == "null-apply") {
      if (rank == 0) std::cerr << "null-apply entered" << std::endl;
      amg.Apply(rank == 1 ? nullptr : dr.data(), dx.data());
      // A return would violate the MPI runtime-failure contract.
      Check(false, "rank-local null Apply unexpectedly returned");
    }
    const double rhs[2] = {hr(0), hr(1)};
    double solution[2] = {0, 0}, global[4] = {0, 0, 0, 0}, residual = 1;
    int iterations = 0;
    for (; iterations < 100 && residual > 1e-12; ++iterations) {
      Check(amg.Apply(dr.data(), dx.data()), amg.error());
      Kokkos::deep_copy(hx, dx);
      for (int i = 0; i < 2; ++i) solution[i] += hx(i);
      MPI_Allgather(solution, 2, MPI_DOUBLE, global, 2, MPI_DOUBLE, MPI_COMM_WORLD);
      double local_square = 0, square = 0;
      for (int i = 0; i < 2; ++i) {
        const int row = 2*rank+i;
        double action = 4*global[row];
        if (row > 0) action -= global[row-1];
        if (row < 3) action -= global[row+1];
        hr(i) = rhs[i]-action;
        local_square += hr(i)*hr(i);
      }
      MPI_Allreduce(&local_square, &square, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      residual = std::sqrt(square/(4+16+36+169));
      Kokkos::deep_copy(dr, hr);
    }
    Check(residual < 1e-12, "independent distributed true residual");
    for (int i = 0; i < 2; ++i) {
      Check(std::isfinite(solution[i]) && std::abs(solution[i]-(2*rank+i+1)) < 2e-11,
            "independent distributed solution");
    }
    if (rank == 0) std::cout << "PASS mode=" << mode << " iterations=" << iterations
                             << " true_residual=" << residual << std::endl;
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
