//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_dense_driver.cpp
//! \brief Independent physical-matrix oracle for the GPU inverse-factor product.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "two_temperature/thermal_radiation_dense.hpp"

#if !ATHENA_CUSPARSE_ENABLED
#error "This standalone oracle requires ATHENA_CUSPARSE_ENABLED=1"
#endif

namespace {
struct FactorCase {
  int n;
  std::vector<double> lower;  // Row-major dense L, independent of its CSC export.
  std::vector<int> permutation;
  std::vector<double> scaling;
};

// Construct the actual, unscaled matrix using dense matrix products. In particular,
// do not test against a second implementation of the helper's triangular solve.
// M = S E S and P M P^T = L L^T imply E = S^-1 P^T L L^T P S^-1.
std::vector<long double> PhysicalMatrix(const FactorCase &test) {
  const int n = test.n;
  std::vector<long double> p(n*n, 0.0L), ll(n*n, 0.0L), tmp(n*n, 0.0L);
  std::vector<long double> matrix(n*n, 0.0L);
  for (int i = 0; i < n; ++i) p[i*n+test.permutation[i]] = 1.0L;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        ll[i*n+j] += static_cast<long double>(test.lower[i*n+k])*test.lower[j*n+k];
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) tmp[i*n+j] += p[k*n+i]*ll[k*n+j];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) matrix[i*n+j] += tmp[i*n+k]*p[k*n+j];
      matrix[i*n+j] /= static_cast<long double>(test.scaling[i])*test.scaling[j];
    }
  }
  return matrix;
}

void UpdateFactor(const FactorCase &test, const bool null_permutation,
                  two_temperature::ImplicitAggregateDenseFactor &factor) {
  std::vector<int> columns(test.n+1), rows;
  std::vector<double> values;
  for (int col = 0; col < test.n; ++col) {
    columns[col] = static_cast<int>(rows.size());
    for (int row = col; row < test.n; ++row) {
      if (test.lower[row*test.n+col] != 0.0) {
        rows.push_back(row);
        values.push_back(test.lower[row*test.n+col]);
      }
    }
  }
  columns[test.n] = static_cast<int>(rows.size());
  const bool accepted = factor.Update(test.n, columns.data(), rows.data(), values.data(),
                null_permutation ? nullptr : test.permutation.data(),
                test.scaling.data());
  if (!accepted) {
    std::cerr << "Unexpected dense factor rejection\n";
    std::exit(2);
  }
  // These temporary CSC arrays are destroyed before any solve. Update must own
  // all factor data after returning, including after an in-place numeric update.
}

bool CheckCase(const std::string &name, const FactorCase &test,
               const std::size_t expected_analyses, const bool null_permutation,
               two_temperature::ImplicitAggregateDenseFactor &factor,
               double &largest_solution_error, double &largest_residual,
               const double solution_tolerance) {
  UpdateFactor(test, null_permutation, factor);
  const auto matrix = PhysicalMatrix(test);
  const std::size_t bytes = factor.DeviceBytes();
  bool passed = factor.AnalysisBuilds() == expected_analyses && bytes > 0;
  double case_solution_error = 0.0, case_residual = 0.0;
  DvceArray1D<Real> coarse("oracle-physical-rhs", test.n);
  auto host = Kokkos::create_mirror_view(coarse);
  // Signed solutions, a different signed RHS, and zero test repeated applications.
  for (int repeat = 0; repeat < 3; ++repeat) {
    std::vector<long double> exact(test.n), rhs(test.n);
    long double exact_max = 0.0L, rhs_max = 0.0L;
    for (int i = 0; i < test.n; ++i) {
      exact[i] = repeat == 2 ? 0.0L :
          ((i+repeat)%2 == 0 ? 1.0L : -1.0L)*(0.375L+i*0.625L+repeat*0.25L);
      exact_max = std::max(exact_max, std::abs(exact[i]));
    }
    for (int i = 0; i < test.n; ++i) {
      for (int j = 0; j < test.n; ++j) rhs[i] += matrix[i*test.n+j]*exact[j];
      host(i) = static_cast<Real>(rhs[i]);
      rhs[i] = host(i);  // The residual is measured against the RHS actually sent.
      rhs_max = std::max(rhs_max, std::abs(rhs[i]));
    }
    Kokkos::deep_copy(coarse, host);
    factor.SolveInPlace(coarse);
    Kokkos::fence();
    Kokkos::deep_copy(host, coarse);
    long double error = 0.0L, residual = 0.0L, energy = 0.0L;
    for (int i = 0; i < test.n; ++i) {
      energy += rhs[i]*host(i);
      passed = passed && std::isfinite(static_cast<double>(host(i)));
      error = std::max(error, std::abs(static_cast<long double>(host(i))-exact[i]));
      long double ax = 0.0L;
      for (int j = 0; j < test.n; ++j) ax += matrix[i*test.n+j]*host(j);
      residual = std::max(residual, std::abs(ax-rhs[i]));
    }
    if (repeat != 2) passed = passed && energy > 0.0L;
    case_solution_error = std::max(case_solution_error,
        static_cast<double>(error/std::max(1.0L, exact_max)));
    case_residual = std::max(case_residual,
        static_cast<double>(residual/std::max(1.0L, rhs_max)));
    passed = passed && factor.AnalysisBuilds() == expected_analyses &&
        factor.DeviceBytes() == bytes;
  }
  const double tolerance = 2000.0*std::numeric_limits<Real>::epsilon();
  passed = passed && case_solution_error <= solution_tolerance &&
      case_residual <= tolerance;
  largest_solution_error = std::max(largest_solution_error, case_solution_error);
  largest_residual = std::max(largest_residual, case_residual);
  if (global_variable::my_rank == 0) {
    std::cout << "{\"case\":\"" << name << "\",\"passed\":"
              << (passed ? "true" : "false") << ",\"n\":" << test.n
              << ",\"nnz\":" << factor.FactorNonzeros()
              << ",\"analysis_builds\":" << factor.AnalysisBuilds()
              << ",\"device_bytes\":" << bytes
              << ",\"solution_error\":" << case_solution_error
              << ",\"relative_residual\":" << case_residual << "}\n";
  }
  return passed;
}

int RunOracle() {
  two_temperature::ImplicitAggregateDenseFactor factor;
  FactorCase test{5,
      {2.0, 0.0, 0.0, 0.0, 0.0,
       0.3, 1.6, 0.0, 0.0, 0.0,
       0.0, -0.4, 2.2, 0.0, 0.0,
       0.25, 0.0, 0.5, 1.8, 0.0,
       -0.1, 0.2, 0.0, -0.35, 1.4},
      {3, 0, 4, 1, 2}, {0.5, 2.0, 1.25, 0.8, 3.0}};
  double largest_error = 0.0, largest_residual = 0.0;
  int failures = 0;
  if (!two_temperature::ImplicitAggregateDenseFactor::SupportsDense(8000, 724827) ||
      two_temperature::ImplicitAggregateDenseFactor::SupportsDense(13600, 2000000) ||
      two_temperature::ImplicitAggregateDenseFactor::SupportsDense(8192,
                                                                  8192)) ++failures;
  const auto check = [&](const char *name, const std::size_t analyses,
                         const bool null_permutation = false,
                         const double tolerance =
                             2000.0*std::numeric_limits<Real>::epsilon()) {
    if (!CheckCase(name, test, analyses, null_permutation, factor,
                   largest_error, largest_residual, tolerance)) ++failures;
  };
  check("permuted_scaled_repeated_rhs", 1);
  for (int row = 0; row < test.n; ++row) {
    for (int col = 0; col <= row; ++col) {
      test.lower[row*test.n+col] *= row == col ? 1.15 : -0.7;
    }
  }
  test.scaling = {1.2, 0.4, 2.5, 1.75, 0.65};
  check("numeric_and_scaling_update_reuses_analysis", 1);
  test.scaling = {0.75, 1.5, 0.6, 2.25, 1.1};
  check("scaling_only_update_reuses_analysis", 1);
  test.permutation = {1, 4, 0, 2, 3};
  check("permutation_change_rebuilds_analysis", 2);
  // Keep n and nnz unchanged while changing the actual sparsity pattern.
  test.lower[4*test.n] = 0.0;
  test.lower[2*test.n] = 0.2;
  check("same_nnz_pattern_change_rebuilds_analysis", 3);
  test = FactorCase{3, {1.8, 0.0, 0.0, -0.6, 1.25, 0.0, 0.35, 0.4, 2.1},
                    {2, 0, 1}, {2.5, 0.5, 1.5}};
  check("dimension_change_rebuilds_analysis", 4);
  test.permutation = {0, 1, 2};
  check("null_identity_permutation", 5, true);
  check("explicit_identity_reuses_analysis", 5);
  factor.Release();
  if (factor.DeviceBytes() != 0 || factor.FactorNonzeros() != 0) ++failures;
  check("release_then_rebuild", 6);
  // The diagonal spans six orders of magnitude; retain coupling, permutation
  // and unequal physical scaling while checking positive r^T E^-1 r as well as
  // the independently formed physical residual. Forward-error tolerances allow
  // the expected conditioning amplification; the residual tolerance stays strict.
  test = FactorCase{4, {1., 0., 0., 0., .001, .01, 0., 0.,
                       -.000002, .000003, .0001, 0.,
                       .00000001, -.00000002, .00000003, .000001},
                   {2, 0, 3, 1}, {0.75, 1.5, 0.9, 2.0}};
  check("conditioned_triangular_factor_1e12", 7, false, 5.e-7);
  test.lower[3*4+3] = .00001;
  check("conditioned_numeric_refresh_1e10", 7, false, 5.e-9);
  if (global_variable::my_rank == 0) {
    std::cout << "{\"test\":\"thermal_radiation_cusparse\",\"passed\":"
              << (failures == 0 ? "true" : "false")
              << ",\"cases\":11,\"solves\":33,\"failures\":" << failures
              << ",\"solution_error\":" << largest_error
              << ",\"relative_residual\":" << largest_residual << "}" << std::endl;
  }
  return failures == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
#endif
  Kokkos::initialize(argc, argv);
  std::cout << std::scientific << std::setprecision(17);
  int status = RunOracle();  // All device objects die before Kokkos finalization.
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  int global_status = 0;
  MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  status = global_status;
  MPI_Finalize();
#endif
  return status;
}
