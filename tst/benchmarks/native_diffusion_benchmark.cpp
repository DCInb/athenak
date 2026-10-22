// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
// Single-device A/B benchmark: identical double-precision PCG and diffusion
// matrix for Jacobi, native multigrid, and (when enabled) HYPRE. Includes setup.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "diffusion/native_multigrid.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#if ATHENA_HYPRE_ENABLED
#include "two_temperature/thermal_radiation_hypre.hpp"
#endif

namespace {
using Vector = DvceArray1D<double>;
using Policy = Kokkos::RangePolicy<DevExeSpace>;
Kokkos::Timer profile_clock;
std::vector<std::pair<std::string, double>> profile_stack;
std::map<std::string, double> profile_times;
void PushProfile(const char *name) {
  profile_stack.emplace_back(name, profile_clock.seconds());
}
void PopProfile() {
  const auto entry = profile_stack.back();
  profile_stack.pop_back();
  if (entry.first.find("native_mg_") == 0) {
    profile_times[entry.first] += profile_clock.seconds()-entry.second;
  }
}
double Dot(const Vector &x, const Vector &y) {
  double value = 0;
  Kokkos::parallel_reduce("benchmark_dot", Policy(0, x.extent_int(0)),
      KOKKOS_LAMBDA(int i, double &sum) { sum += x(i)*y(i); }, value);
  return value;
}
template <typename T>
DvceArray1D<T> Upload(const std::vector<T> &source) {
  DvceArray1D<T> device("benchmark graph", source.size());
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < source.size(); ++i) host(i) = source[i];
  Kokkos::deep_copy(device, host);
  return device;
}

int Benchmark(const std::string &backend, int width, int updates, double stiffness) {
  if (width < 2 || width > 256 || updates < 1 || !(stiffness > 0)) return 2;
  if (backend != "native-mg" && backend != "jacobi" && backend != "hypre") return 2;
#if !ATHENA_HYPRE_ENABLED
  if (backend == "hypre") {
    std::cerr << "Rebuild with Athena_ENABLE_HYPRE=ON for the HYPRE comparison\n";
    return 2;
  }
#endif
  const int n = width*width*width;
  std::vector<int> offsets{0}, columns, csr_offsets{0}, csr_columns, slots;
  for (int i = 0; i < n; ++i) {
    std::vector<int> neighbors;
    for (int stride : {1, width, width*width}) {
      const int p = (i/stride)%width;
      if (p > 0) neighbors.push_back(i-stride);
      if (p+1 < width) neighbors.push_back(i+stride);
    }
    std::sort(neighbors.begin(), neighbors.end());
    columns.insert(columns.end(), neighbors.begin(), neighbors.end());
    offsets.push_back(columns.size());
    neighbors.push_back(i);
    std::sort(neighbors.begin(), neighbors.end());
    int edge = offsets[i];
    for (int j : neighbors) {
      csr_columns.push_back(j);
      slots.push_back(i == j ? -1 : edge++);
    }
    csr_offsets.push_back(csr_columns.size());
  }
  auto row = Upload(offsets), col = Upload(columns);
  auto hrow = Upload(csr_offsets), hslot = Upload(slots);
  Vector g("benchmark edges", columns.size()), mass("benchmark mass", n);
  Vector diagonal("benchmark diagonal", n), values("benchmark CSR", csr_columns.size());
  Vector rhs("benchmark rhs", n), x("benchmark x", n), r("benchmark r", n);
  Vector z("benchmark z", n), p("benchmark p", n), ap("benchmark Ap", n);
  Vector exact("benchmark exact", n);
  diffusion::NativeMultigrid native;
#if ATHENA_HYPRE_ENABLED
  ImplicitAMGHypre hypre;
#endif
  DevExeSpace().fence();
  Kokkos::Timer timer;
  if (backend == "native-mg") {
    native.Configure(offsets, columns, 0, n, false, 8ULL*1024*1024*1024);
  }
#if ATHENA_HYPRE_ENABLED
  if (backend == "hypre" &&
      !hypre.ConfigureGraph(csr_offsets, csr_columns, 0, n, 8ULL*1024*1024*1024)) {
    std::cerr << hypre.error() << std::endl;
    return 1;
  }
#endif
  const double configure = timer.seconds();
  const auto matvec = [&](const Vector &v, const Vector &out) {
    Kokkos::parallel_for("benchmark_matvec", Policy(0, n), KOKKOS_LAMBDA(int i) {
      double value = mass(i)*v(i);
      for (int q = row(i); q < row(i+1); ++q) value += g(q)*(v(i)-v(col(q)));
      out(i) = value;
    });
  };
  const auto apply = [&]() {
    if (backend == "native-mg") native.Apply(r, z);
    if (backend == "jacobi") {
      Kokkos::parallel_for("benchmark_jacobi", Policy(0, n),
          KOKKOS_LAMBDA(int i) { z(i) = r(i)/diagonal(i); });
    }
#if ATHENA_HYPRE_ENABLED
    if (backend == "hypre" && !hypre.Apply(r.data(), z.data())) {
      std::cerr << hypre.error() << std::endl;
      std::abort();
    }
#endif
  };
  std::cout << "backend,cells,update,configure_s,setup_s,solve_s,"
               "iterations,residual,error\n";
  for (int update = 0; update < updates; ++update) {
    const double strength = stiffness*(1+0.1*update);
    Kokkos::parallel_for("benchmark_matrix", Policy(0, n), KOKKOS_LAMBDA(int i) {
      mass(i) = 1.0+0.2*sin(0.17*i);
      double d = mass(i);
      for (int q = row(i); q < row(i+1); ++q) {
        const int j = col(q), a = i < j ? i : j;
        // A heterogeneous but symmetric graph with a strong material interface.
        const double ki = i%width < width/2 ? 1.0 : 0.001;
        const double kj = j%width < width/2 ? 1.0 : 0.001;
        const double value = strength*(1.0+0.3*sin(0.23*a))*2*ki*kj/(ki+kj);
        g(q) = value;
        d += value;
      }
      diagonal(i) = d;
      exact(i) = 1+0.1*sin(0.13*i)+0.1*cos(0.037*i);
      for (int q = hrow(i); q < hrow(i+1); ++q) {
        values(q) = hslot(q) < 0 ? d : -g(hslot(q));
      }
    });
    matvec(exact, rhs);
    DevExeSpace().fence();
    timer.reset();
    if (backend == "native-mg") native.Setup(g, mass);
#if ATHENA_HYPRE_ENABLED
    if (backend == "hypre" && !hypre.Setup(values.data())) {
      std::cerr << hypre.error() << std::endl;
      return 1;
    }
#endif
    DevExeSpace().fence();
    const double setup = timer.seconds();
    timer.reset();
    Kokkos::deep_copy(x, 0.0);
    Kokkos::deep_copy(r, rhs);
    const double norm = Dot(rhs, rhs);
    apply();
    Kokkos::deep_copy(p, z);
    double rz = Dot(r, z), rr = norm;
    int iterations = 0;
    for (; iterations < 2000 && rr > 1e-20*norm; ++iterations) {
      matvec(p, ap);
      const double alpha = rz/Dot(p, ap);
      Kokkos::parallel_for("benchmark_update", Policy(0, n), KOKKOS_LAMBDA(int i) {
        x(i) += alpha*p(i);
        r(i) -= alpha*ap(i);
      });
      rr = Dot(r, r);
      if (rr <= 1e-20*norm) { ++iterations; break; }
      apply();
      const double next = Dot(r, z), beta = next/rz;
      Kokkos::parallel_for("benchmark_direction", Policy(0, n),
          KOKKOS_LAMBDA(int i) { p(i) = z(i)+beta*p(i); });
      rz = next;
    }
    DevExeSpace().fence();
    const double solve = timer.seconds();
    matvec(x, ap);
    Kokkos::parallel_for("benchmark_true_residual", Policy(0, n), KOKKOS_LAMBDA(int i) {
      r(i) = rhs(i)-ap(i);
      z(i) = x(i)-exact(i);
    });
    const double residual = sqrt(Dot(r, r)/norm);
    const double error = sqrt(Dot(z, z)/Dot(exact, exact));
    std::cout << std::setprecision(10) << backend << ',' << n << ',' << update << ','
              << (update == 0 ? configure : 0) << ',' << setup << ',' << solve << ','
              << iterations << ',' << residual << ',' << error << std::endl;
    if (!(residual < 2e-10) || !(error < 1e-5) || iterations == 2000) return 1;
  }
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
  const std::string backend = argc > 1 ? argv[1] : "native-mg";
  const int width = argc > 2 ? std::stoi(argv[2]) : 32;
  const int updates = argc > 3 ? std::stoi(argv[3]) : 3;
  const double stiffness = argc > 4 ? std::stod(argv[4]) : 1000;
  Kokkos::initialize(argc, argv);
  const bool profile = argc > 5 && std::string(argv[5]) == "--profile";
  if (profile) {
    Kokkos::Tools::Experimental::set_push_region_callback(PushProfile);
    Kokkos::Tools::Experimental::set_pop_region_callback(PopProfile);
  }
  const int status = Benchmark(backend, width, updates, stiffness);
  if (profile) {
    Kokkos::Tools::Experimental::set_push_region_callback(nullptr);
    Kokkos::Tools::Experimental::set_pop_region_callback(nullptr);
    for (const auto &entry : profile_times) {
      std::cerr << entry.first << "," << entry.second << " seconds (inclusive)\n";
    }
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return status;
}
