#ifndef TWO_TEMPERATURE_THERMAL_RADIATION_CUSPARSE_HPP_
#define TWO_TEMPERATURE_THERMAL_RADIATION_CUSPARSE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_cusparse.hpp
//! \brief Reusable GPU triangular solves for a host-factored AMR coarse matrix.

#ifndef ATHENA_CUSPARSE_ENABLED
#define ATHENA_CUSPARSE_ENABLED 0
#endif

#if ATHENA_CUSPARSE_ENABLED
#include <cusparse.h>

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

namespace two_temperature {

// If M=S E S and P M P^T=L L^T, apply E^-1=S P^T L^-T L^-1 P S.
// CHOLMOD's Perm[i] is the original index of factor row i. The caller must export
// a packed, real, double-precision LL factor, converting a COPY of an LDL or
// supernodal factor first. The original host factor is never modified here.
class ImplicitAggregateDeviceFactor {
 public:
  ImplicitAggregateDeviceFactor() = default;
  ImplicitAggregateDeviceFactor(const ImplicitAggregateDeviceFactor &) = delete;
  ImplicitAggregateDeviceFactor &operator=(
      const ImplicitAggregateDeviceFactor &) = delete;

  ~ImplicitAggregateDeviceFactor() {
    if (handle_ != nullptr) {
      execution_space_.fence();
      DestroyPlans();
      cusparseDestroy(handle_);
    }
  }

  // All input pointers refer to host memory and can be released after return.
  // A null permutation denotes identity; scaling is in original matrix order.
  void Update(int n, const int *columns, const int *rows, const double *values,
              const int *permutation, const double *scaling) {
    if (n <= 0 || n == std::numeric_limits<int>::max() || columns == nullptr ||
        rows == nullptr || values == nullptr ||
        scaling == nullptr || columns[0] != 0 || columns[n] < n) {
      Fail("Invalid packed CSC factor dimensions");
    }
    const int nnz = columns[n];
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

    // Updates occur between groups. Drain any old uses before changing values,
    // destroying analysis descriptors or resizing their referenced allocations.
    execution_space_.fence();
    if (handle_ == nullptr) {
      Check(cusparseCreate(&handle_), "create handle");
      Check(cusparseSetStream(handle_, execution_space_.cuda_stream()),
            "set Kokkos stream");
      Check(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST),
            "set scalar mode");
    }
    const bool new_pattern = matrix_ == nullptr || next_columns != host_columns_ ||
        next_rows != host_rows_ || next_permutation != host_permutation_;
    if (new_pattern) {
      DestroyPlans();
      n_ = n;
      nnz_ = nnz;
      Kokkos::realloc(row_offsets_, n+1);
      Kokkos::realloc(column_indices_, nnz);
      Kokkos::realloc(factor_values_, nnz);
      Kokkos::realloc(permutation_, n);
      Kokkos::realloc(scaling_, n);
      Kokkos::realloc(rhs_, n);
      Kokkos::realloc(forward_, n);
      Kokkos::realloc(solution_, n);
      CopyHost(next_columns, row_offsets_);
      CopyHost(next_rows, column_indices_);
      CopyHost(next_permutation, permutation_);
    }
    CopyHost(next_values, factor_values_);
    CopyHost(next_scaling, scaling_);
    if (new_pattern) {
      // CSC(L) is CSR(L^T): one index/value allocation serves both triangular
      // solves. The transpose operation below solves L without duplicating nnz.
      Check(cusparseCreateCsr(&matrix_, n, n, nnz, row_offsets_.data(),
                             column_indices_.data(), factor_values_.data(),
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "create CSR factor");
      cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
      cusparseDiagType_t diagonal = CUSPARSE_DIAG_TYPE_NON_UNIT;
      Check(cusparseSpMatSetAttribute(matrix_, CUSPARSE_SPMAT_FILL_MODE,
                                     &fill, sizeof(fill)), "set upper factor");
      Check(cusparseSpMatSetAttribute(matrix_, CUSPARSE_SPMAT_DIAG_TYPE,
                                     &diagonal, sizeof(diagonal)), "set LL diagonal");
      Check(cusparseCreateDnVec(&rhs_descriptor_, n, rhs_.data(), CUDA_R_64F),
            "create RHS");
      Check(cusparseCreateDnVec(&forward_descriptor_, n, forward_.data(), CUDA_R_64F),
            "create intermediate vector");
      Check(cusparseCreateDnVec(&solution_descriptor_, n, solution_.data(), CUDA_R_64F),
            "create solution vector");
      CreatePlan(CUSPARSE_OPERATION_TRANSPOSE, rhs_descriptor_, forward_descriptor_,
                 forward_plan_, forward_workspace_);
      CreatePlan(CUSPARSE_OPERATION_NON_TRANSPOSE, forward_descriptor_,
                 solution_descriptor_, backward_plan_, backward_workspace_);
      ++analysis_builds_;
      host_columns_ = std::move(next_columns);
      host_rows_ = std::move(next_rows);
      host_permutation_ = std::move(next_permutation);
    } else {
      // SpSV caches numerical data internally. Updating the input allocation
      // alone is insufficient; refresh both operation-specific opaque plans.
      Check(cusparseSpSV_updateMatrix(handle_, forward_plan_, factor_values_.data(),
                                     CUSPARSE_SPSV_UPDATE_GENERAL), "update L values");
      Check(cusparseSpSV_updateMatrix(handle_, backward_plan_, factor_values_.data(),
                                     CUSPARSE_SPSV_UPDATE_GENERAL), "update LT values");
    }
    // Copies use temporary host arrays; complete them before returning ownership.
    execution_space_.fence();
  }

  // In-place physical-order correction. No allocation, transfer, host scalar
  // reduction or fence is introduced in this hot path. All work shares the same
  // Kokkos CUDA stream as restriction and prolongation in the calling solver.
  void SolveInPlace(const DvceArray1D<Real> &coarse) {
    if (matrix_ == nullptr || coarse.extent(0) != static_cast<std::size_t>(n_)) {
      Fail("GPU coarse solve has no matching factor");
    }
    const auto permutation = permutation_;
    const auto scaling = scaling_;
    const auto rhs = rhs_;
    const auto solution = solution_;
    Kokkos::parallel_for("thermal_rad_device_coarse_permute_scale",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n_),
    KOKKOS_LAMBDA(int i) {
      const int original = permutation(i);
      rhs(i) = scaling(original)*static_cast<double>(coarse(original));
    });
    Check(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_TRANSPOSE, &one_, matrix_,
                            rhs_descriptor_, forward_descriptor_, CUDA_R_64F,
                            CUSPARSE_SPSV_ALG_DEFAULT, forward_plan_), "solve L");
    Check(cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one_, matrix_,
                            forward_descriptor_, solution_descriptor_, CUDA_R_64F,
                            CUSPARSE_SPSV_ALG_DEFAULT, backward_plan_), "solve LT");
    Kokkos::parallel_for("thermal_rad_device_coarse_unpermute_scale",
                        Kokkos::RangePolicy<DevExeSpace>(execution_space_, 0, n_),
    KOKKOS_LAMBDA(int i) {
      const int original = permutation(i);
      coarse(original) = static_cast<Real>(scaling(original)*solution(i));
    });
  }

  std::size_t DeviceBytes() const {
    return sizeof(int)*(row_offsets_.size()+column_indices_.size()+permutation_.size())+
        sizeof(double)*(factor_values_.size()+scaling_.size()+rhs_.size()+forward_.size()+
                        solution_.size())+
        forward_workspace_.size()+backward_workspace_.size();
  }
  int FactorNonzeros() const { return nnz_; }
  std::size_t AnalysisBuilds() const { return analysis_builds_; }

 private:
  [[noreturn]] static void Fail(const std::string &message) {
    if (global_variable::my_rank == 0) {
      std::cerr << "### FATAL ERROR in GPU AMR coarse solve: " << message << std::endl;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }

  static void Check(cusparseStatus_t status, const char *operation) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
      Fail(std::string(operation)+": "+cusparseGetErrorString(status));
    }
  }

  template <typename T>
  void CopyHost(const std::vector<T> &host, DvceArray1D<T> &device) {
    using HostView = Kokkos::View<const T *, HostMemSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(execution_space_, device, HostView(host.data(), host.size()));
  }

  void CreatePlan(cusparseOperation_t operation, cusparseDnVecDescr_t rhs,
                  cusparseDnVecDescr_t solution, cusparseSpSVDescr_t &plan,
                  DvceArray1D<unsigned char> &workspace) {
    Check(cusparseSpSV_createDescr(&plan), "create SpSV plan");
    std::size_t bytes = 0;
    Check(cusparseSpSV_bufferSize(handle_, operation, &one_, matrix_, rhs, solution,
                                 CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, plan, &bytes),
          "query SpSV workspace");
    Kokkos::realloc(workspace, bytes);
    Check(cusparseSpSV_analysis(handle_, operation, &one_, matrix_, rhs, solution,
                               CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, plan,
                               workspace.data()), "analyze triangular solve");
  }

  void DestroyPlans() {
    if (forward_plan_ != nullptr) cusparseSpSV_destroyDescr(forward_plan_);
    if (backward_plan_ != nullptr) cusparseSpSV_destroyDescr(backward_plan_);
    if (rhs_descriptor_ != nullptr) cusparseDestroyDnVec(rhs_descriptor_);
    if (forward_descriptor_ != nullptr) cusparseDestroyDnVec(forward_descriptor_);
    if (solution_descriptor_ != nullptr) cusparseDestroyDnVec(solution_descriptor_);
    if (matrix_ != nullptr) cusparseDestroySpMat(matrix_);
    forward_plan_ = backward_plan_ = nullptr;
    rhs_descriptor_ = forward_descriptor_ = solution_descriptor_ = nullptr;
    matrix_ = nullptr;
    // Release old analysis storage before allocating a changed topology.
    Kokkos::realloc(forward_workspace_, 0);
    Kokkos::realloc(backward_workspace_, 0);
  }

  DevExeSpace execution_space_;
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matrix_ = nullptr;
  cusparseDnVecDescr_t rhs_descriptor_ = nullptr, forward_descriptor_ = nullptr,
                      solution_descriptor_ = nullptr;
  cusparseSpSVDescr_t forward_plan_ = nullptr, backward_plan_ = nullptr;
  int n_ = 0, nnz_ = 0;
  std::size_t analysis_builds_ = 0;
  double one_ = 1.0;
  std::vector<int> host_columns_, host_rows_, host_permutation_;
  DvceArray1D<int> row_offsets_, column_indices_, permutation_;
  DvceArray1D<double> factor_values_, scaling_, rhs_, forward_, solution_;
  DvceArray1D<unsigned char> forward_workspace_, backward_workspace_;
};

}  // namespace two_temperature
#endif  // ATHENA_CUSPARSE_ENABLED
#endif  // TWO_TEMPERATURE_THERMAL_RADIATION_CUSPARSE_HPP_
