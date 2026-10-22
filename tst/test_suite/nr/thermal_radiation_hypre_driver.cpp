//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2026 AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_hypre_driver.cpp
//! \brief Independent host-matvec oracle for the optional BoomerAMG adapter.
//
// Configure Athena with Athena_ENABLE_HYPRE=ON and HYPRE_ROOT pointing to a
// compatible hypre >= 2.33 installation, then build and run:
//   cmake --build BUILD --target thermal_radiation_hypre_driver
//   BUILD/thermal_radiation_hypre_driver
// Use one MPI rank and, for a CUDA build, one visible GPU. The target uses the
// configured Kokkos backend and hypre precision/MPI/CUDA settings; no Athena
// problem generator or application executable is needed. The CPU and GPU paths
// test symmetry, linearity, positivity, independent PCG residuals, numerical
// refresh, and the near-Neumann constant mode. Scalar refresh also guards the
// coarse-level SpMV beta-preservation regression seen with CUDA 12.3.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "two_temperature/thermal_radiation_hypre.hpp"

namespace {
using Vec = std::vector<double>;
double Dot(const Vec &a, const Vec &b) {
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}
void Check(bool condition, const std::string &why) {
  if (!condition) throw std::runtime_error(why);
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1) {
    std::cerr << "The standalone BoomerAMG oracle requires one MPI rank\n";
    MPI_Finalize();
    return 1;
  }
#endif
  Kokkos::initialize(argc, argv);
  int exitcode = 0;
  try {
    constexpr int width = 24, n = width * width;
    std::vector<std::map<int, double>> matrix(n);
    for (int p = 0; p < n; ++p) matrix[p][p] = 0.02;
    for (int y = 0; y < width; ++y)
      for (int x = 0; x < width; ++x) {
        int p = y * width + x;
        for (int q : {y * width + (x + 1) % width, ((y + 1) % width) * width + x}) {
          // Heterogeneous, strictly positive diffusion, periodic boundaries.
          const double c = (1 + ((p + q) % 11)) * (x < width / 2 ? 1.0 : 10.0);
          matrix[p][p] += c;
          matrix[q][q] += c;
          matrix[p][q] -= c;
          matrix[q][p] -= c;
        }
      }
    std::vector<int> rows{0}, columns;
    Vec values;
    for (const auto &row : matrix) {
      for (auto entry : row) {
        columns.push_back(entry.first);
        values.push_back(entry.second);
      }
      rows.push_back(columns.size());
    }
    Kokkos::View<double *, DevMemSpace> dv("values", values.size()), dr("rhs", n),
        dx("solution", n);
    auto hv = Kokkos::create_mirror_view(dv), hr = Kokkos::create_mirror_view(dr),
         hx = Kokkos::create_mirror_view(dx);
    ImplicitAMGHypre amg;
    Check(amg.Available(), "AMG build unavailable");
    Check(amg.ConfigureGraph(rows, columns, 0, n, 512UL * 1024 * 1024), amg.error());
    for (std::size_t i = 0; i < values.size(); ++i) hv(i) = values[i];
    Kokkos::deep_copy(dv, hv);
    Check(amg.Setup(dv.data()), amg.error());
    auto apply = [&](const Vec &rhs) {
      for (int i = 0; i < n; ++i) hr(i) = rhs[i];
      Kokkos::deep_copy(dr, hr);
      Check(amg.Apply(dr.data(), dx.data()), amg.error());
      Kokkos::deep_copy(hx, dx);
      Vec z(n);
      for (int i = 0; i < n; ++i) {
        z[i] = hx(i);
        Check(std::isfinite(z[i]), "Nonfinite AMG correction");
      }
      return z;
    };
    auto matvec = [&](const Vec &v) {
      Vec av(n, 0);
      for (int i = 0; i < n; ++i)
        for (int j = rows[i]; j < rows[i + 1]; ++j) av[i] += values[j] * v[columns[j]];
      return av;
    };
    double symmetry = 0, linearity = 0, minimum_quadratic = 1e300;
    for (int seed = 1; seed <= 12; ++seed) {
      Vec a(n), b(n), c(n);
      for (int i = 0; i < n; ++i) {
        a[i] = std::sin((i + 0.3) * seed);
        b[i] = std::cos((i + 0.7) * (seed + 0.25));
        c[i] = 1.3 * a[i] - 0.4 * b[i];
      }
      Vec ma = apply(a), mb = apply(b), mc = apply(c), difference(n);
      double scale =
          std::sqrt(Dot(a, a) * Dot(mb, mb)) + std::sqrt(Dot(b, b) * Dot(ma, ma));
      symmetry = std::max(symmetry, std::abs(Dot(a, mb) - Dot(b, ma)) / scale);
      for (int i = 0; i < n; ++i) difference[i] = mc[i] - (1.3 * ma[i] - 0.4 * mb[i]);
      linearity =
          std::max(linearity, std::sqrt(Dot(difference, difference) / Dot(mc, mc)));
      minimum_quadratic = std::min(minimum_quadratic, Dot(a, ma) / Dot(a, a));
    }
    std::cerr << "symmetry=" << symmetry << " linearity=" << linearity
              << " minimum_quadratic=" << minimum_quadratic << "\n";
    Check(symmetry < 3e-12, "AMG V-cycle symmetry defect");
    Check(linearity < 3e-12, "AMG V-cycle linearity defect");
    Check(minimum_quadratic > 0, "AMG V-cycle nonpositive quadratic form");
    Vec b(n), x(n, 0), r(n), z, p;
    for (int i = 0; i < n; ++i)
      b[i] = 0.2 + std::sin(i * 0.41) + 0.5 * std::cos(i * 0.07);
    r = b;
    z = apply(r);
    p = z;
    double rz = Dot(r, z), rhsnorm = Dot(b, b), relative = 1;
    int iterations = 0;
    for (; iterations < 300 && relative > 1e-10; ++iterations) {
      Vec ap = matvec(p);
      Check(Dot(p, ap) > 0 && rz > 0, "Nonpositive PCG denominator");
      const double alpha = rz / Dot(p, ap);
      for (int i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
      relative = std::sqrt(Dot(r, r) / rhsnorm);
      if (relative <= 1e-10) {
        ++iterations;
        break;
      }
      z = apply(r);
      const double next = Dot(r, z), beta = next / rz;
      for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
      rz = next;
    }
    Vec ax = matvec(x);
    for (int i = 0; i < n; ++i) ax[i] -= b[i];
    const double true_residual = std::sqrt(Dot(ax, ax) / rhsnorm);
    Check(true_residual < 2e-10, "AMG PCG failed independent true residual");
    // Numerical update on exactly the same topology must replace the hierarchy.
    const Vec old = apply(b);
    for (std::size_t i = 0; i < values.size(); ++i) hv(i) = 2 * values[i];
    Kokkos::deep_copy(dv, hv);
    Check(amg.ConfigureGraph(rows, columns, 0, n, 512UL * 1024 * 1024), amg.error());
    Check(amg.Setup(dv.data()), amg.error());
    Vec updated = apply(b), error(n);
    for (int i = 0; i < n; ++i) error[i] = 2 * updated[i] - old[i];
    const double update_error = std::sqrt(Dot(error, error) / Dot(old, old));
    // GPU PMIS may choose a different hierarchy on each setup. The difference
    // between approximate V-cycles is diagnostic, not an algebraic invariant.
    std::cerr << "randomized_hierarchy_scale_difference=" << update_error << "\n";
    x.assign(n, 0);
    r = b;
    z = apply(r);
    p = z;
    rz = Dot(r, z);
    relative = 1;
    int updated_iterations = 0;
    for (; updated_iterations < 300 && relative > 1e-10; ++updated_iterations) {
      Vec ap = matvec(p);
      for (double &value : ap) value *= 2;
      Check(Dot(p, ap) > 0 && rz > 0, "Nonpositive updated PCG denominator");
      const double alpha = rz / Dot(p, ap);
      for (int i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
      relative = std::sqrt(Dot(r, r) / rhsnorm);
      if (relative <= 1e-10) {
        ++updated_iterations;
        break;
      }
      z = apply(r);
      const double next = Dot(r, z), beta = next / rz;
      for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
      rz = next;
    }
    ax = matvec(x);
    for (int i = 0; i < n; ++i) ax[i] = 2 * ax[i] - b[i];
    const double updated_true_residual = std::sqrt(Dot(ax, ax) / rhsnorm);
    Check(updated_true_residual < 2e-10, "Updated matrix PCG true residual failed");

    std::cerr << "original_pcg_iterations=" << iterations
              << " original_true_residual=" << true_residual
              << " updated_pcg_iterations=" << updated_iterations
              << " updated_true_residual=" << updated_true_residual << "\n";
    // A scalar graph has no randomized coarsening: exact diagonal solutions test
    // same-topology numerical refresh without assuming a repeated hierarchy.
    ImplicitAMGHypre scalar;
    Check(scalar.ConfigureGraph({0, 1}, {0}, 0, 1, 512UL * 1024 * 1024), scalar.error());
    hr(0) = 2;
    Kokkos::deep_copy(dr, hr);
    double scalar_update_error = 0;
    for (double diagonal : {4.0, 8.0}) {
      hv(0) = diagonal;
      Kokkos::deep_copy(dv, hv);
      Check(scalar.Setup(dv.data()), scalar.error());
      Check(scalar.Apply(dr.data(), dx.data()), scalar.error());
      Kokkos::deep_copy(hx, dx);
      std::cerr << "scalar_diagonal=" << diagonal << " correction=" << hx(0)
                << " expected=" << 2 / diagonal << "\n";
      scalar_update_error = std::max(scalar_update_error, std::abs(hx(0) - 2 / diagonal));
    }
    std::cerr << "scalar_update_error=" << scalar_update_error << "\n";

    // Near-Neumann diffusion is dominated by its constant mode. Eight diagonal
    // coarse sweeps can be SPD yet almost fail to correct this slowest mode.
    constexpr double epsilon = 1e-8;
    for (int i = 0; i < n; ++i)
      for (int j = rows[i]; j < rows[i + 1]; ++j) {
        // Normalize the diffusion row scale to O(1), giving stiffness O(1e8).
        values[j] /= 1024;
        if (columns[j] == i) values[j] += epsilon - 0.02 / 1024;
        hv(j) = values[j];
      }
    Kokkos::deep_copy(dv, hv);
    Check(amg.Setup(dv.data()), amg.error());
    Vec constant(n, 1), correction = apply(constant),
                        constant_residual = matvec(correction);
    for (int i = 0; i < n; ++i) constant_residual[i] -= 1;
    const double constant_contraction =
        std::sqrt(Dot(constant_residual, constant_residual) / n);
    const double constant_mean =
        std::accumulate(correction.begin(), correction.end(), 0.0) / n;
    std::cerr << "constant_contraction=" << constant_contraction
              << " amplitude_ratio=" << epsilon * constant_mean << "\n";
    double fresh_constant_contraction = 0, fresh_constant_amplitude = 0;
    {
      ImplicitAMGHypre fresh;
      Check(fresh.ConfigureGraph(rows, columns, 0, n, 512UL * 1024 * 1024),
            fresh.error());
      Check(fresh.Setup(dv.data()), fresh.error());
      Check(fresh.Apply(dr.data(), dx.data()), fresh.error());
      Kokkos::deep_copy(hx, dx);
      Vec fresh_solution(n);
      for (int i = 0; i < n; ++i) fresh_solution[i] = hx(i);
      Vec fresh_residual = matvec(fresh_solution);
      for (double &v : fresh_residual) v -= 1;
      fresh_constant_contraction = std::sqrt(Dot(fresh_residual, fresh_residual) / n);
      fresh_constant_amplitude =
          epsilon * std::accumulate(fresh_solution.begin(), fresh_solution.end(), 0.0) /
          n;
      std::cerr << "fresh_stiff_contraction=" << fresh_constant_contraction
                << " fresh_stiff_amplitude=" << fresh_constant_amplitude << "\n";
    }
    Check(scalar_update_error < 1e-13, "Exact same-topology scalar update failed");
    Check(std::abs(epsilon * constant_mean - 1) < 0.2,
          "Near-Neumann constant amplitude incorrect");
    x.assign(n, 0);
    r = constant;
    z = apply(r);
    p = z;
    rz = Dot(r, z);
    relative = 1;
    int stiff_iterations = 0;
    for (; stiff_iterations < 300 && relative > 1e-10; ++stiff_iterations) {
      Vec ap = matvec(p);
      Check(Dot(p, ap) > 0 && rz > 0, "Nonpositive stiff PCG denominator");
      const double alpha = rz / Dot(p, ap);
      for (int i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
      relative = std::sqrt(Dot(r, r) / n);
      if (relative <= 1e-10) {
        ++stiff_iterations;
        break;
      }
      z = apply(r);
      const double next = Dot(r, z), beta = next / rz;
      for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
      rz = next;
    }
    ax = matvec(x);
    for (int i = 0; i < n; ++i) ax[i] -= 1;
    const double stiff_true_residual = std::sqrt(Dot(ax, ax) / n);
    std::cerr << "stiff_pcg_iterations=" << stiff_iterations
              << " true_residual=" << stiff_true_residual << "\n";
    Check(stiff_true_residual < 1e-7, "Near-Neumann PCG true residual failed");
    std::cout << "{\"passed\":true,\"nodes\":" << n << ",\"symmetry_defect\":" << symmetry
              << ",\"linearity_defect\":" << linearity
              << ",\"minimum_rayleigh\":" << minimum_quadratic
              << ",\"pcg_iterations\":" << iterations
              << ",\"true_residual\":" << true_residual
              << ",\"randomized_hierarchy_scale_difference\":" << update_error
              << ",\"updated_pcg_iterations\":" << updated_iterations
              << ",\"updated_true_residual\":" << updated_true_residual
              << ",\"scalar_update_error\":" << scalar_update_error
              << ",\"stiff_pcg_iterations\":" << stiff_iterations
              << ",\"stiff_true_residual\":" << stiff_true_residual
              << ",\"constant_contraction\":" << constant_contraction
              << ",\"constant_mean_relative_error\":"
              << std::abs(epsilon * constant_mean - 1)
              << ",\"fresh_constant_contraction\":" << fresh_constant_contraction
              << ",\"fresh_constant_amplitude\":" << fresh_constant_amplitude
              << ",\"device_bytes\":" << amg.device_bytes() << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    exitcode = 1;
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return exitcode;
}
