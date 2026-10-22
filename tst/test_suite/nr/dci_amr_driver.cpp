//========================================================================================
//! \file dci_amr_driver.cpp
//! \brief Check physical density thresholds and material selection on the active device.

#include <cstdlib>
#include <iostream>

#include "DCI_3D/dci_amr.hpp"

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    constexpr int ntests = 8;
    const char *names[ntests] = {
        "ambient remains at root", "1.1 g/cc CH gets four total levels",
        "19.2 g/cc gold gets three total levels", "dense helium gets dense level",
        "trace CH in dense gold does not select CH level",
        "compressed CH/gold mixture selects CH level",
        "exact threshold does not refine", "subthreshold CH derefines"};
    DvceArray1D<int> result("dci_amr_checks", ntests);
    const DCIAMRPolicy policy{1.0/1.1, 2, 3, 2};
    Kokkos::parallel_for("dci_amr_policy", ntests, KOKKOS_LAMBDA(const int n) {
      int wanted = -1;
      int expected = -2;
      if (n == 0) {
        wanted = policy.TargetLevel(1.0e-5/1.1, 0.0, 0.0);
        expected = 0;
      } else if (n == 1) {
        wanted = policy.TargetLevel(1.0, 1.0, 0.0);
        expected = 3;
      } else if (n == 2) {
        wanted = policy.TargetLevel(19.2/1.1, 0.0, 19.2/1.1);
        expected = 2;
      } else if (n == 3) {
        wanted = policy.TargetLevel(2.0/1.1, 0.0, 0.0);
        expected = 2;
      } else if (n == 4) {
        wanted = policy.TargetLevel(19.2/1.1, 0.01/1.1, 19.19/1.1);
        expected = 2;
      } else if (n == 5) {
        wanted = policy.TargetLevel(20.0/1.1, 2.0/1.1, 18.0/1.1);
        expected = 3;
      } else if (n == 6) {
        wanted = policy.TargetLevel(policy.density_threshold,
                                     policy.density_threshold, 0.0);
        expected = 0;
      } else if (n == 7) {
        wanted = policy.TargetLevel(0.99/1.1, 0.99/1.1, 0.0);
        expected = 0;
      }
      result(n) = wanted == expected;
    });
    const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
    for (int n = 0; n < ntests; ++n) {
      std::cout << (host(n) ? "PASS " : "FAIL ") << names[n] << '\n';
      failures += host(n) == 0;
    }
  }
  Kokkos::finalize();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
