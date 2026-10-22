//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file thermal_radiation_dense_guard.cpp
//! \brief Verify numerical rejection, storage release, and subsequent recovery.

#include <mpi.h>

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "two_temperature/thermal_radiation_dense.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    two_temperature::ImplicitAggregateDenseFactor factor;
    const int p[] = {0, 1}, row[] = {0};
    const double tiny[] = {1.e-310}, scale[] = {1.};
    const bool accepted = factor.Update(1, p, row, tiny, nullptr, scale);
    passed = !accepted && factor.DeviceBytes() == 0 && factor.FactorNonzeros() == 0;
    const double normal[] = {2.};
    passed = factor.Update(1, p, row, normal, nullptr, scale) && passed;
    DvceArray1D<Real> rhs("guard_rhs", 1);
    Kokkos::deep_copy(rhs, 8.);
    factor.SolveInPlace(rhs);
    auto host = Kokkos::create_mirror_view(rhs);
    Kokkos::deep_copy(host, rhs);
    passed = passed && std::abs(host(0)-2.) < 1.e-14;
    factor.Release();
    passed = passed && factor.DeviceBytes() == 0;
  }
  std::cout << "{\"test\":\"dense_overflow_reject_release_recover\",\"passed\":"
            << (passed ? "true" : "false") << "}" << std::endl;
  Kokkos::finalize();
  MPI_Finalize();
  return passed ? 0 : 1;
}
