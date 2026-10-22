#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_DENSE_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_DENSE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_dense.hpp
//! \brief Bounded GPU storage for a reusable AMR coarse inverse-factor product.

#ifndef ATHENA_CUSPARSE_ENABLED
#define ATHENA_CUSPARSE_ENABLED 0
#endif
#if ATHENA_CUSPARSE_ENABLED
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace two_temperature {

// For M=S E S and P M P^T=L L^T, store W=L^-1 and apply
// E^-1 = S P^T W^T W P S. Keeping the inverse as this factor product preserves
// positive-definite structure for the inner conjugate-gradient preconditioner.
// The storage limit counts owned Kokkos views; CUDA/cuBLAS context storage is
// outside this count and is covered separately by the caller's free-memory guard.
class ImplicitAggregateDenseFactor {
 public:
  static constexpr std::size_t kMaximumDeviceBytes = std::size_t{1} << 30;
  static bool SupportsDense(int n, int nnz) {
    if (n <= 0 || nnz < n || n > 8192) return false;
    const std::size_t nodes = static_cast<std::size_t>(n);
    return 16*nodes*nodes + 16*static_cast<std::size_t>(nnz) + 36*nodes <=
        kMaximumDeviceBytes;
  }
  ImplicitAggregateDenseFactor() = default;
  ImplicitAggregateDenseFactor(const ImplicitAggregateDenseFactor &) = delete;
  ImplicitAggregateDenseFactor &operator=(const ImplicitAggregateDenseFactor &) = delete;
  ~ImplicitAggregateDenseFactor() {
    Release();
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  // Drain outstanding applications before releasing every owned device view.
  void Release() {
    execution_space_.fence();
    dense_ = {}; inverse_ = {}; values_ = {}; scaling_ = {};
    rhs_ = {}; forward_ = {}; solution_ = {};
    rows_ = {}; columns_ = {}; permutation_ = {};
    host_columns_.clear(); host_rows_.clear(); host_permutation_.clear();
    n_ = nnz_ = 0;
  }
  // Input CSC L, permutation and scaling arrays are host-owned. Return false
  // after releasing storage if inversion cannot produce a finite, nonsingular W.
  // This setup path may synchronize; SolveInPlace keeps every operation on-device.
  bool Update(int n, const int *columns, const int *rows, const double *values,
              const int *permutation, const double *scaling) {
    if (n <= 0 || n == std::numeric_limits<int>::max() || columns == nullptr ||
        rows == nullptr || values == nullptr ||
        scaling == nullptr || columns[0] != 0 || columns[n] < n) {
      Fail("Invalid packed CSC factor dimensions");
    }
    const int nnz = columns[n];
    if (!SupportsDense(n, nnz)) Fail("Dense inverse exceeds the 1 GiB storage limit");
    std::vector<int> next_columns(columns, columns+n+1);
    std::vector<int> next_rows(rows, rows+nnz);
    std::vector<int> next_permutation(n), seen(n, -1);
    std::vector<double> next_values(values, values+nnz), next_scaling(scaling, scaling+n);
    for (int col = 0; col < n; ++col) {
      if (columns[col] < 0 || columns[col] >= columns[col+1] || columns[col+1] > nnz) {
        Fail("Invalid packed CSC factor column pointers");
      }
      bool diagonal = false;
      for (int q = columns[col]; q < columns[col+1]; ++q) {
        const int row = rows[q];
        if (row < col || row >= n || seen[row] == col || !std::isfinite(values[q])) {
          Fail("Invalid lower-triangular LL factor entry");
        }
        seen[row] = col;
        if (row == col) {
          diagonal = true;
          if (!(values[q] > 0.0)) Fail("Nonpositive LL factor diagonal");
        }
      }
      if (!diagonal) Fail("Missing LL factor diagonal");
    }
    std::fill(seen.begin(), seen.end(), -1);
    for (int row = 0; row < n; ++row) {
      const int original = permutation == nullptr ? row : permutation[row];
      if (original < 0 || original >= n || seen[original] != -1 ||
          !(scaling[row] > 0.0) || !std::isfinite(scaling[row])) {
        Fail("Invalid coarse permutation or scaling");
      }
      seen[original] = row;
      next_permutation[row] = original;
    }

    execution_space_.fence();
    if (handle_ == nullptr) {
      Check(cublasCreate(&handle_), "create handle");
      Check(cublasSetStream(handle_, execution_space_.cuda_stream()), "set stream");
      Check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "set scalar mode");
    }
    const bool new_pattern = n_ != n || nnz_ != nnz || next_columns != host_columns_ ||
        next_rows != host_rows_ || next_permutation != host_permutation_;
    std::vector<int> host_entry_columns;
    if (new_pattern) {
      Release();
      n_ = n; nnz_ = nnz;
      const std::size_t dense_size = static_cast<std::size_t>(n)*n;
      Kokkos::realloc(dense_, dense_size);
      Kokkos::realloc(inverse_, dense_size);
      Kokkos::realloc(values_, nnz);
      Kokkos::realloc(rows_, nnz);
      Kokkos::realloc(columns_, nnz);
      Kokkos::realloc(permutation_, n);
      Kokkos::realloc(scaling_, n);
      Kokkos::realloc(rhs_, n);
      Kokkos::realloc(forward_, n);
      Kokkos::realloc(solution_, n);
      host_entry_columns.resize(nnz);
      for (int col = 0; col < n; ++col) {
        for (int q = columns[col]; q < columns[col+1]; ++q) host_entry_columns[q] = col;
      }
      CopyHost(next_rows, rows_);
      CopyHost(host_entry_columns, columns_);
      CopyHost(next_permutation, permutation_);
      Kokkos::deep_copy(execution_space_, dense_, 0.0);
      host_columns_ = std::move(next_columns);
      host_rows_ = std::move(next_rows);
      host_permutation_ = std::move(next_permutation);
      ++analysis_builds_;
    }
    CopyHost(next_values, values_);
    CopyHost(next_scaling, scaling_);
    const auto dense = dense_;
    const auto entry_rows = rows_;
    const auto entry_columns = columns_;
    const auto entry_values = values_;
    Kokkos::parallel_for("thermal_rad_dense_factor_scatter",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, nnz),
    KOKKOS_LAMBDA(int q) {
      dense(static_cast<std::size_t>(entry_columns(q))*n + entry_rows(q)) =
          entry_values(q);
    });
    const auto inverse = inverse_;
    Kokkos::parallel_for("thermal_rad_dense_factor_identity",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n*n),
    KOKKOS_LAMBDA(int q) {
      inverse(q) = q/n == q%n ? 1.0 : 0.0;
    });
    Check(cublasDtrsm(handle_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                     CUBLAS_DIAG_NON_UNIT, n, n, &one_, dense_.data(), n,
                     inverse_.data(), n), "invert L");
    // Store a finite nonsingular triangular W=L^-1. Its product W^T W
    // preserves the SPD structure required by the inner conjugate-gradient solve.
    // The upper triangle is structurally zero, independent of BLAS roundoff.
    int invalid = 0;
    Kokkos::parallel_reduce("thermal_rad_dense_factor_validate",
                           Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n*n),
    KOKKOS_LAMBDA(int q, int &bad) {
      const int row = q % n, col = q / n;
      if (row < col) {
        inverse(q) = 0.0;
      } else {
        const double value = inverse(q);
        if (!Kokkos::isfinite(value) || (row == col && !(value > 0.0))) ++bad;
      }
    }, invalid);
    if (invalid != 0) {
      Release();
      return false;
    }
    // Temporary host arrays are used above. Complete every asynchronous copy and
    // numerical refresh before returning ownership to the caller.
    execution_space_.fence();
    return true;
  }
  // Apply the physical-order correction on the same CUDA stream as the caller.
  // This hot path allocates no storage and performs no host transfers or fences.
  void SolveInPlace(const DvceArray1D<Real> &coarse) {
    if (n_ == 0 || coarse.extent(0) != static_cast<std::size_t>(n_)) {
      Fail("Dense GPU coarse solve has no matching factor");
    }
    const auto permutation = permutation_;
    const auto scaling = scaling_;
    const auto rhs = rhs_;
    const auto solution = solution_;
    Kokkos::parallel_for("thermal_rad_dense_permute_scale",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n_),
    KOKKOS_LAMBDA(int i) {
      const int original = permutation(i);
      rhs(i) = scaling(original)*static_cast<double>(coarse(original));
    });
    Check(cublasDgemv(handle_, CUBLAS_OP_N, n_, n_, &one_, inverse_.data(), n_,
                     rhs_.data(), 1, &zero_, forward_.data(), 1), "apply inverse L");
    Check(cublasDgemv(handle_, CUBLAS_OP_T, n_, n_, &one_, inverse_.data(), n_,
                     forward_.data(), 1, &zero_, solution_.data(), 1),
          "apply inverse LT");
    Kokkos::parallel_for("thermal_rad_dense_unpermute_scale",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n_),
    KOKKOS_LAMBDA(int i) {
      const int original = permutation(i);
      coarse(original) = static_cast<Real>(scaling(original)*solution(i));
    });
  }
  std::size_t DeviceBytes() const {
    return sizeof(double)*(dense_.size()+inverse_.size()+values_.size()+scaling_.size()+
                           rhs_.size()+forward_.size()+solution_.size())+
        sizeof(int)*(rows_.size()+columns_.size()+permutation_.size());
  }
  int FactorNonzeros() const { return nnz_; }
  std::size_t AnalysisBuilds() const { return analysis_builds_; }

 private:
  [[noreturn]] static void Fail(const std::string &message) {
    if (global_variable::my_rank == 0) {
      std::cerr << "### FATAL ERROR in dense GPU AMR coarse solve: "
                << message << std::endl;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }
  static void Check(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      Fail(std::string(operation)+": cuBLAS status "+
           std::to_string(static_cast<int>(status)));
    }
  }
  template <typename T>
  void CopyHost(const std::vector<T> &host, DvceArray1D<T> &device) {
    using HostView = Kokkos::View<const T *, HostMemSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(execution_space_, device, HostView(host.data(), host.size()));
  }
  DevExeSpace execution_space_;
  cublasHandle_t handle_ = nullptr;
  int n_ = 0, nnz_ = 0;
  std::size_t analysis_builds_ = 0;
  double one_ = 1.0, zero_ = 0.0;
  std::vector<int> host_columns_, host_rows_, host_permutation_;
  DvceArray1D<int> rows_, columns_, permutation_;
  DvceArray1D<double> dense_, inverse_, values_, scaling_, rhs_, forward_, solution_;
};

}  // namespace two_temperature
#endif  // ATHENA_CUSPARSE_ENABLED
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_DENSE_HPP_
